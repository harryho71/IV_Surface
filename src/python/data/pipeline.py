"""
Market Data Pipeline — fetch, clean, quality-check, and persist.

Provides end-to-end convenience wrappers around the individual data modules
(fetchers, cleaners, validators, quality, expiry_classifier, loaders).

Public API::

    fetch_and_save_market_data(ticker, data_dir, rate, div_yield, ...)
        → (MarketSnapshot, DataQualityReport, pd.DataFrame)

    load_or_fetch(ticker, data_dir, rate, div_yield, force_refresh)
        → (MarketSnapshot, DataQualityReport, pd.DataFrame)

    snapshot_to_grid(snapshot, n_strikes, n_mats, strike_lo_pct, strike_hi_pct)
        → (np.ndarray, np.ndarray, np.ndarray)   # (strikes, maturities, ivs)

    list_cached_tickers(data_dir)
        → List[str]

    get_snapshot_info(snapshot)
        → Dict[str, Any]
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .cleaners import clean_option_chain
from .expiry_classifier import ExpiryClassifier, ExpiryType
from .fetchers import fetch_option_chain_yfinance
from .loaders import MarketSnapshot, load_market_json
from .quality import DataQualityPipeline, DataQualityReport

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def list_cached_tickers(data_dir: "Path | str" = "data") -> List[str]:
    """Return ticker symbols that have a cached JSON in *data_dir*/raw/."""
    raw_dir = Path(data_dir) / "raw"
    if not raw_dir.exists():
        return []
    tickers = []
    for f in sorted(raw_dir.glob("*_market_data.json")):
        # filename pattern: <TICKER>_market_data.json
        name = f.stem  # e.g. "AAPL_market_data"
        ticker = name.split("_")[0].upper()
        if ticker:
            tickers.append(ticker)
    return tickers


def get_snapshot_info(snapshot: MarketSnapshot) -> Dict:
    """Return a concise metadata dict for a :class:`~loaders.MarketSnapshot`."""
    mats = snapshot.maturities
    return {
        "ticker":      snapshot.ticker,
        "spot":        snapshot.spot,
        "forward":     snapshot.forward,
        "timestamp":   snapshot.timestamp,
        "n_maturities": int(len(mats)),
        "n_options":   snapshot.n_options(),
        "mat_min":     float(mats.min()) if len(mats) > 0 else None,
        "mat_max":     float(mats.max()) if len(mats) > 0 else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────────────

def fetch_and_save_market_data(
    ticker: str,
    data_dir: "Path | str" = "data",
    rate: float = 0.05,
    div_yield: float = 0.01,
    max_expirations: Optional[int] = None,
    min_dte_days: int = 3,
    min_strikes_per_maturity: int = 4,
) -> Tuple[MarketSnapshot, DataQualityReport, pd.DataFrame]:
    """
    Fetch, clean, quality-check and save all option data for *ticker*.

    Pipeline steps:

    1. Fetch spot price via Yahoo Finance.
    2. Retrieve all available option expiration dates.
    3. Classify expirations via :class:`~expiry_classifier.ExpiryClassifier`
       and attach ``expiry_type`` labels.
    4. Fetch each expiration's option chain via
       :func:`~fetchers.fetch_option_chain_yfinance`.
    5. Concatenate and clean the full chain with
       :func:`~cleaners.clean_option_chain`.
    6. Run :class:`~quality.DataQualityPipeline` on the cleaned data.
    7. Build and save the project JSON to
       ``data/raw/<TICKER>_market_data.json``.
    8. Save a flat cleaned CSV to
       ``data/processed/<TICKER>_cleaned.csv``.

    Args:
        ticker:                  Underlying symbol (e.g. ``'AAPL'``).
        data_dir:                Root data directory (contains raw/ and
                                 processed/ subdirectories).
        rate:                    Risk-free rate for log-moneyness and forward.
        div_yield:               Continuous dividend yield.
        max_expirations:         Limit fetched expirations (useful for testing).
        min_dte_days:            Drop expirations with fewer days-to-expiry
                                 (avoids pin-risk / settlement noise).
        min_strikes_per_maturity: Minimum valid strikes required to include a
                                 maturity slice in the surface JSON.

    Returns:
        ``(MarketSnapshot, DataQualityReport, cleaned_df)``

    Raises:
        ValueError: If no option data can be fetched or cleaned.
    """
    ticker = ticker.upper()
    data_dir = Path(data_dir)
    raw_dir  = data_dir / "raw"
    proc_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("fetch_and_save_market_data ticker=%s", ticker)

    # ── 1. Spot price ──────────────────────────────────────────────────────
    yf_ticker = yf.Ticker(ticker)
    spot: float = 100.0
    try:
        spot = float(yf_ticker.fast_info.last_price)
    except Exception:
        try:
            spot = float(yf_ticker.info.get("regularMarketPrice", 100.0))
        except Exception:
            logger.warning("Could not determine spot for %s, using %.1f", ticker, spot)

    # ── 2. All expiration dates ────────────────────────────────────────────
    all_expirations: List[str] = list(yf_ticker.options or [])
    if not all_expirations:
        raise ValueError(f"No option expirations found for {ticker!r}")

    # ── 3. Classify each expiration ────────────────────────────────────────
    classifier = ExpiryClassifier()
    today_d = date.today()

    expiry_types: Dict[str, str] = {}
    filtered: List[str] = []
    for exp_str in all_expirations:
        try:
            exp_date = date.fromisoformat(exp_str)
            dte = (exp_date - today_d).days
            if dte < min_dte_days:
                continue  # skip pin-risk / already expired
            etype = classifier.classify(exp_date, today=today_d)
            expiry_types[exp_str] = etype.value
            filtered.append(exp_str)
        except Exception:
            expiry_types[exp_str] = ExpiryType.OTHER.value
            filtered.append(exp_str)

    if max_expirations:
        filtered = filtered[:max_expirations]

    logger.info("Using %d/%d expirations for %s", len(filtered), len(all_expirations), ticker)

    # ── 4. Fetch all chains ────────────────────────────────────────────────
    all_chains: List[pd.DataFrame] = []
    for exp in filtered:
        try:
            chain = fetch_option_chain_yfinance(ticker, exp)
            if not chain.empty:
                all_chains.append(chain)
        except Exception as exc:
            logger.warning("Failed to fetch %s expiry %s: %s", ticker, exp, exc)

    if not all_chains:
        raise ValueError(f"No option chain data fetched for {ticker!r}")

    raw_df = pd.concat(all_chains, ignore_index=True)
    logger.info("Fetched %d raw option quotes for %s", len(raw_df), ticker)

    # ── 5. Attach expiry_type label before cleaning ────────────────────────
    raw_df["expiry_type"] = raw_df["expiration"].map(
        lambda e: expiry_types.get(str(e), ExpiryType.OTHER.value)
    )

    # ── 6. Clean ───────────────────────────────────────────────────────────
    # clean_option_chain expects: strike, bid, ask, impliedVolatility,
    #                             type, expiration
    cleaned_df = clean_option_chain(
        raw_df,
        spot_price=spot,
        risk_free_rate=rate,
        dividend_yield=div_yield,
    )
    if cleaned_df.empty:
        raise ValueError(f"Cleaning produced empty DataFrame for {ticker!r}")

    # Carry expiry_type through (join on expiration date)
    exp_type_map = (
        raw_df[["expiration", "expiry_type"]]
        .drop_duplicates("expiration")
        .set_index("expiration")["expiry_type"]
    )
    # cleaned_df has maturity (float) not expiration; re-derive from raw
    # (we add it as extra info only — does not affect surface construction)
    logger.info("Cleaned to %d option quotes", len(cleaned_df))

    # ── 7. Data quality check ──────────────────────────────────────────────
    quality_df = cleaned_df.rename(
        columns={"implied_volatility": "iv"}, errors="ignore"
    ).copy()
    quality_report = DataQualityPipeline().run(quality_df, spot=spot)
    logger.info("Quality: %s", quality_report.summary())

    # ── 8. Build surface JSON (calls only, standard practice) ──────────────
    calls_df = cleaned_df[cleaned_df["type"] == "call"].copy()
    if calls_df.empty:
        logger.warning("No calls found; falling back to all option types")
        calls_df = cleaned_df.copy()

    data_dict: Dict[str, Dict] = {}
    for mat in sorted(calls_df["maturity"].unique()):
        if mat <= 0:
            continue
        slice_df = calls_df[calls_df["maturity"] == mat].copy()
        iv_col = "implied_volatility"
        if iv_col not in slice_df.columns:
            continue
        valid = (
            slice_df[["strike", iv_col]]
            .dropna()
            .pipe(lambda d: d[d[iv_col] > 0.0])
            .sort_values("strike")
        )
        if len(valid) < min_strikes_per_maturity:
            continue
        data_dict[str(mat)] = {
            "strikes": [float(v) for v in valid["strike"].tolist()],
            "ivs":     [float(v) for v in valid[iv_col].tolist()],
        }

    if not data_dict:
        raise ValueError(f"No valid maturity slices for {ticker!r}")

    surface_json = {
        "ticker":    ticker,
        "spot":      float(spot),
        "forward":   float(spot * np.exp((rate - div_yield) * 1.0)),
        "timestamp": pd.Timestamp.now().isoformat(),
        "data":      data_dict,
    }

    # ── 9. Persist ─────────────────────────────────────────────────────────
    json_path = raw_dir  / f"{ticker}_market_data.json"
    csv_path  = proc_dir / f"{ticker}_cleaned.csv"

    with json_path.open("w") as fh:
        json.dump(surface_json, fh, indent=2)
    logger.info("Saved JSON  → %s", json_path)

    cleaned_df.to_csv(csv_path, index=False)
    logger.info("Saved CSV   → %s", csv_path)

    snapshot = load_market_json(json_path)
    return snapshot, quality_report, cleaned_df


# ──────────────────────────────────────────────────────────────────────────────
# Load-or-fetch convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def load_or_fetch(
    ticker: str,
    data_dir: "Path | str" = "data",
    rate: float = 0.05,
    div_yield: float = 0.01,
    force_refresh: bool = False,
    **kwargs,
) -> Tuple[MarketSnapshot, Optional[DataQualityReport], Optional[pd.DataFrame]]:
    """
    Load cached market data for *ticker*, or fetch from Yahoo Finance if missing.

    Args:
        ticker:        Underlying symbol.
        data_dir:      Root data directory.
        rate:          Risk-free rate (used only when fetching).
        div_yield:     Continuous dividend yield (used only when fetching).
        force_refresh: If ``True``, always fetch from live source even if a
                       cached file exists.
        **kwargs:      Extra keyword arguments forwarded to
                       :func:`fetch_and_save_market_data`.

    Returns:
        ``(MarketSnapshot, DataQualityReport | None, cleaned_df | None)``
        Quality report and cleaned DataFrame are ``None`` when loading from
        cache (they are not persisted separately).
    """
    ticker = ticker.upper()
    data_dir = Path(data_dir)
    json_path = data_dir / "raw" / f"{ticker}_market_data.json"

    if not force_refresh and json_path.exists():
        logger.info("Loading cached data for %s from %s", ticker, json_path)
        snapshot = load_market_json(json_path)
        # Try to load cached cleaned CSV for quality display
        csv_path = data_dir / "processed" / f"{ticker}_cleaned.csv"
        cleaned_df: Optional[pd.DataFrame] = None
        quality_report: Optional[DataQualityReport] = None
        if csv_path.exists():
            try:
                cleaned_df = pd.read_csv(csv_path)
                quality_df = cleaned_df.rename(
                    columns={"implied_volatility": "iv"}, errors="ignore"
                )
                quality_report = DataQualityPipeline().run(quality_df, spot=snapshot.spot)
            except Exception as exc:
                logger.warning("Could not load cleaned CSV: %s", exc)
        return snapshot, quality_report, cleaned_df

    return fetch_and_save_market_data(
        ticker=ticker, data_dir=data_dir,
        rate=rate, div_yield=div_yield, **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Grid builder
# ──────────────────────────────────────────────────────────────────────────────

def snapshot_to_grid(
    snapshot: MarketSnapshot,
    n_strikes: int = 20,
    n_mats: Optional[int] = None,
    strike_lo_pct: float = 0.80,
    strike_hi_pct: float = 1.20,
    min_strikes_in_slice: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a :class:`~loaders.MarketSnapshot` onto a regular (K, T) grid.

    Parameters
    ----------
    snapshot:
        Loaded market snapshot.
    n_strikes:
        Number of points in the common strike grid.
    n_mats:
        Number of maturities to include; ``None`` means keep all that pass the
        minimum-strike filter.
    strike_lo_pct:
        Lower bound of common strike grid as a fraction of spot
        (e.g. ``0.80`` → 80 % of spot).
    strike_hi_pct:
        Upper bound of common strike grid as a fraction of spot.
    min_strikes_in_slice:
        Minimum number of valid strike/IV pairs required to include a maturity.

    Returns
    -------
    strikes    : (n_strikes,)    float64 array
    maturities : (n_mats_out,)   float64 array
    ivs        : (n_strikes, n_mats_out)  float64 array
        IV values in decimal (e.g. 0.20 = 20 %).  Boundary extrapolation is
        clamped (flat) so values remain in-sample.
    """
    spot = snapshot.spot
    K_lo = spot * strike_lo_pct
    K_hi = spot * strike_hi_pct
    common_strikes = np.linspace(K_lo, K_hi, n_strikes)

    # Filter and select maturities
    valid_mats = []
    for T in snapshot.maturities:
        src_k = snapshot.strikes_by_maturity.get(T, np.array([]))
        src_iv = snapshot.ivs_by_maturity.get(T, np.array([]))
        # Require at least min_strikes_in_slice valid data points
        valid_mask = np.isfinite(src_iv) & (src_iv > 0) & np.isfinite(src_k)
        if valid_mask.sum() >= min_strikes_in_slice:
            valid_mats.append(T)

    if not valid_mats:
        raise ValueError(
            f"No maturity slices with ≥{min_strikes_in_slice} valid strikes "
            f"for {snapshot.ticker}"
        )

    # Subsample maturities if requested
    if n_mats is not None and n_mats < len(valid_mats):
        idx = np.round(np.linspace(0, len(valid_mats) - 1, n_mats)).astype(int)
        valid_mats = [valid_mats[i] for i in idx]

    maturities_out = np.array(valid_mats, dtype=float)
    ivs_grid = np.full((n_strikes, len(maturities_out)), np.nan)

    for j, T in enumerate(maturities_out):
        src_k  = snapshot.strikes_by_maturity[T]
        src_iv = snapshot.ivs_by_maturity[T]

        # Keep only valid, finite data
        mask = np.isfinite(src_iv) & (src_iv > 0) & np.isfinite(src_k)
        src_k  = src_k[mask]
        src_iv = src_iv[mask]

        if len(src_k) < 2:
            ivs_grid[:, j] = src_iv[0] if len(src_iv) == 1 else np.nan
            continue

        # Sort by strike
        order  = np.argsort(src_k)
        src_k  = src_k[order]
        src_iv = src_iv[order]

        # Linear interpolation; clamp extrapolation
        ivs_grid[:, j] = np.interp(
            common_strikes, src_k, src_iv,
            left=src_iv[0], right=src_iv[-1],
        )

    # Replace any remaining NaN rows with column mean (robustness)
    col_means = np.nanmean(ivs_grid, axis=0)
    for j in range(ivs_grid.shape[1]):
        nan_mask = np.isnan(ivs_grid[:, j])
        ivs_grid[nan_mask, j] = col_means[j]

    # Clip to [0.001, 3.0]
    ivs_grid = np.clip(ivs_grid, 0.001, 3.0)

    return common_strikes, maturities_out, ivs_grid
