"""
Data Loaders — load market data snapshots from disk.

Supports three formats:
  1. Project JSON format (data/raw/<TICKER>_market_data.json)
     Structure: {ticker, spot, forward, timestamp, data: {maturity: {strikes, ivs, ...}}}

  2. Flat CSV format — one row per (strike, maturity) point

  3. Pickle format — any DataFrame previously saved with df.to_pickle()

Functions:
  load_market_json(path)            → MarketSnapshot dataclass
  load_iv_csv(path)                 → pd.DataFrame  (strike, maturity, iv columns)
  load_pickle(path)                 → pd.DataFrame
  load_raw_ticker(ticker, data_dir) → MarketSnapshot  (convenience wrapper)
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """
    A single market data snapshot as loaded from disk.

    Attributes:
        ticker:              Underlying symbol (e.g. 'SPY').
        spot:                Spot price at snapshot time.
        forward:             Forward price — may be None.
        timestamp:           ISO-8601 string of snapshot time.
        maturities:          Sorted 1-D array of maturities in years.
        strikes_by_maturity: dict mapping maturity → 1-D strike array.
        ivs_by_maturity:     dict mapping maturity → 1-D IV array (decimal).
        raw:                 The original parsed dict (full provenance).
    """

    ticker: str
    spot: float
    forward: Optional[float]
    timestamp: str
    maturities: np.ndarray
    strikes_by_maturity: Dict[float, np.ndarray]
    ivs_by_maturity: Dict[float, np.ndarray]
    raw: Dict = field(default_factory=dict, repr=False)

    def to_flat_dataframe(self) -> pd.DataFrame:
        """
        Convert to a flat DataFrame with columns
        ``(ticker, spot, maturity, strike, iv)``.
        """
        rows = []
        for mat in self.maturities:
            strikes = self.strikes_by_maturity[mat]
            ivs = self.ivs_by_maturity[mat]
            for k, iv in zip(strikes, ivs):
                rows.append({"ticker": self.ticker, "spot": self.spot,
                              "maturity": mat, "strike": k, "iv": iv})
        return pd.DataFrame(rows)

    def n_options(self) -> int:
        return sum(len(v) for v in self.ivs_by_maturity.values())

    def n_maturities(self) -> int:
        return len(self.maturities)


# ──────────────────────────────────────────────────────────────────────────────
# JSON loader (project format)
# ──────────────────────────────────────────────────────────────────────────────

def load_market_json(path: Union[str, Path]) -> MarketSnapshot:
    """
    Load a project-format market data JSON file.

    Expected JSON structure::

        {
            "ticker":    "SPY",
            "spot":      682.85,
            "forward":   683.10,           # optional
            "timestamp": "2026-02-18T16:00:00Z",
            "data": {
                "<maturity_years>": {
                    "strikes": [480, 490, ...],
                    "ivs":     [0.18, 0.17, ...]
                },
                ...
            }
        }

    Args:
        path: Path to the JSON file.

    Returns:
        :class:`MarketSnapshot`

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")

    with path.open() as fh:
        raw = json.load(fh)

    ticker = raw.get("ticker", path.stem.split("_")[0])
    spot = float(raw.get("spot", float("nan")))
    forward = raw.get("forward")
    if forward is not None:
        forward = float(forward)
    timestamp = str(raw.get("timestamp", ""))

    data_dict = raw.get("data", {})
    maturities_raw = sorted(data_dict.keys(), key=float)
    maturities = np.array([float(m) for m in maturities_raw])

    strikes_by_maturity: Dict[float, np.ndarray] = {}
    ivs_by_maturity: Dict[float, np.ndarray] = {}

    for m_str, m_float in zip(maturities_raw, maturities):
        entry = data_dict[m_str]
        strikes_arr = np.asarray(entry.get("strikes", []), dtype=float)
        ivs_arr = np.asarray(entry.get("ivs", []), dtype=float)
        n = min(len(strikes_arr), len(ivs_arr))
        strikes_arr = strikes_arr[:n]
        ivs_arr = ivs_arr[:n]
        order = np.argsort(strikes_arr)
        strikes_by_maturity[m_float] = strikes_arr[order]
        ivs_by_maturity[m_float] = ivs_arr[order]

    logger.info("load_market_json ticker=%s spot=%.2f maturities=%d total_options=%d",
                ticker, spot, len(maturities),
                sum(len(v) for v in ivs_by_maturity.values()))

    return MarketSnapshot(ticker=ticker, spot=spot, forward=forward,
                          timestamp=timestamp, maturities=maturities,
                          strikes_by_maturity=strikes_by_maturity,
                          ivs_by_maturity=ivs_by_maturity, raw=raw)


# ──────────────────────────────────────────────────────────────────────────────
# CSV loader
# ──────────────────────────────────────────────────────────────────────────────

def load_iv_csv(
    path: Union[str, Path],
    *,
    strike_col: str = "strike",
    maturity_col: str = "maturity",
    iv_col: str = "iv",
    sep: str = ",",
) -> pd.DataFrame:
    """
    Load an IV surface from a flat CSV file.

    The CSV must contain columns for strike, maturity (years), and IV (decimal).
    Column names are configurable via keyword arguments.

    Returns:
        DataFrame with columns ``strike``, ``maturity``, ``iv``.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, sep=sep)

    for col, std in [(strike_col, "strike"), (maturity_col, "maturity"), (iv_col, "iv")]:
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not found in {path}; available: {list(df.columns)}")
        df = df.rename(columns={col: std})

    for col in ("strike", "maturity", "iv"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["strike", "maturity", "iv"])
    df = df[(df["strike"] > 0) & (df["maturity"] > 0) & (df["iv"] > 0)]
    after = len(df)
    if after < before:
        logger.info("load_iv_csv dropped %d invalid rows", before - after)

    return df.sort_values(["maturity", "strike"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Pickle loader
# ──────────────────────────────────────────────────────────────────────────────

def load_pickle(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a DataFrame from a pickle file created with ``df.to_pickle()``.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the unpickled object is not a DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with path.open("rb") as fh:
        obj = pickle.load(fh)

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame from {path}, got {type(obj).__name__}")

    logger.info("load_pickle loaded %d rows from %s", len(obj), path)
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_ticker(
    ticker: str,
    data_dir: Union[str, Path] = "data/raw",
) -> MarketSnapshot:
    """
    Load the project raw JSON snapshot for a *ticker* from *data_dir*.

    Looks for ``<data_dir>/<TICKER>_market_data.json``.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{ticker.upper()}_market_data.json"
    if not path.exists():
        candidates = list(data_dir.glob(f"*{ticker}*market_data.json"))
        if candidates:
            path = candidates[0]
        else:
            raise FileNotFoundError(
                f"No market data file found for {ticker!r} in {data_dir}"
            )
    return load_market_json(path)
