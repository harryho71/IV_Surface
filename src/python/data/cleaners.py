"""Option data cleaning and normalization."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _per_expiry_spread_filter(
	df: pd.DataFrame,
	spread_ratio_col: str,
	expiry_col: str,
	sigma_multiplier: float = 4.0,
	flat_cap: float = 0.20,
) -> "tuple[pd.Series, int]":
	"""
	Per-expiry spread filter: remove quotes whose relative bid-ask spread
	exceeds mean + sigma_multiplier × std within the same expiry.

	This is the recommended approach (§2.2): a flat threshold misses the
	point that a 10% spread may be fine for a far-OTM 2-year option but
	terrible for a 1-week ATM.  Statistical filtering adapts to each expiry's
	liquidity profile.

	Also applies a hard cap of `flat_cap` (default 20%) as a safety net.

	Returns:
		(keep_mask, n_removed)
	"""
	keep = pd.Series(True, index=df.index)
	sr = df[spread_ratio_col]

	# Hard cap — always remove extreme outliers
	keep &= (sr <= flat_cap) | sr.isna()

	# Per-expiry statistical filter
	for expiry, group in df.groupby(expiry_col):
		group_sr = sr.loc[group.index].dropna()
		if len(group_sr) < 4:
			# Too few quotes for statistics — keep them all, apply cap only
			continue
		mean_sr = group_sr.mean()
		std_sr = group_sr.std(ddof=1)
		threshold = mean_sr + sigma_multiplier * std_sr
		too_wide = sr.loc[group.index] > threshold
		keep.loc[group.index] &= ~too_wide

	n_removed = int((~keep).sum())
	return keep, n_removed


def clean_option_chain(
	df: pd.DataFrame,
	spot_price: float,
	risk_free_rate: float,
	dividend_yield: float = 0.0,
	forward_price: Optional[float] = None,
	spread_sigma_multiplier: float = 4.0,
) -> pd.DataFrame:
	"""
	Prepare option data for IV surface construction.

	Steps:
	1.  Calculate mid-price: (Bid + Ask) / 2
	2.  Convert expiration dates → days to expiration (integer)
	3.  Convert to years: days / 365.25
	4.  Calculate spot moneyness: K / S  (backward-compat column)
	5a. Remove zero-bid / zero-ask quotes (hard filter, §2.2)
	5b. Remove per-expiry spread outliers (mean + N·sigma per expiry, §2.2)
	6.  Calculate log-moneyness: k = ln(K / F) where F is the forward price.
	    If `forward_price` is None, approximates F ≈ S·exp((r − q)·T).
	7.  Aggregate: one row per unique (strike, maturity) pair
	8.  Handle missing values
	9.  Sort by (maturity, strike)

	Args:
		df:                    Raw option chain DataFrame.
		spot_price:            Current spot price S.
		risk_free_rate:        Risk-free rate r (used for F ≈ S·e^{(r−q)T}).
		dividend_yield:        Continuous dividend yield q.
		forward_price:         Override F for ATM-moneyness; if None, computed
		                       from spot, rate, div_yield, maturity.
		spread_sigma_multiplier: Multiplier for per-expiry statistical spread
		                       filter (§2.2). Default 4.0 (mean + 4σ).

	Returns:
		Cleaned DataFrame with columns:
		strike, maturity, days_to_expiry, moneyness, log_moneyness,
		forward, mid_price, bid_ask_spread, type, implied_volatility
	"""
	if df.empty:
		return df.copy()

	working = df.copy()
	total_rows = len(working)

	working["bid"] = pd.to_numeric(working.get("bid"), errors="coerce")
	working["ask"] = pd.to_numeric(working.get("ask"), errors="coerce")
	working["strike"] = pd.to_numeric(working.get("strike"), errors="coerce")
	working["implied_volatility"] = pd.to_numeric(working.get("impliedVolatility"), errors="coerce")
	if "implied_volatility" not in working.columns:
		working["implied_volatility"] = pd.to_numeric(working.get("implied_volatility"), errors="coerce")

	working["mid_price"] = (working["bid"] + working["ask"]) / 2.0
	working["bid_ask_spread"] = working["ask"] - working["bid"]

	expiration = pd.to_datetime(working.get("expiration"), errors="coerce")
	# Ensure both sides are tz-naive for arithmetic (yfinance returns tz-naive dates)
	if expiration.dt.tz is not None:
		expiration = expiration.dt.tz_localize(None)
	today = pd.Timestamp.now().normalize()  # tz-naive
	working["days_to_expiry"] = (expiration - today).dt.days
	working["maturity"] = working["days_to_expiry"] / 365.25
	working["moneyness"] = working["strike"] / float(spot_price)

	# §2.2 hard filter: zero or negative bid — no price information
	zero_bid_mask = working["bid"].fillna(0.0) > 0
	removed_zero_bid = int((~zero_bid_mask).sum())
	working = working[zero_bid_mask].copy()
	logger.info("clean_option_chain removed_zero_bid=%d", removed_zero_bid)

	# §2.2 per-expiry statistical spread filter
	working["_spread_ratio"] = (
		working["bid_ask_spread"] / working["mid_price"].replace(0, np.nan)
	)
	# Use raw expiration date as grouping key for per-expiry stats
	working["_expiry_key"] = expiration.loc[working.index].dt.normalize()
	keep_mask, removed_spread = _per_expiry_spread_filter(
		working,
		spread_ratio_col="_spread_ratio",
		expiry_col="_expiry_key",
		sigma_multiplier=spread_sigma_multiplier,
	)
	working = working[keep_mask].copy()
	logger.info("clean_option_chain removed_by_spread=%d (per-expiry mean+%gσ)",
				removed_spread, spread_sigma_multiplier)

	# §1 / §5 log-moneyness: k = ln(K / F(T))
	# If a single forward_price override is provided, use it; otherwise
	# approximate F(T) ≈ S · exp((r − q) · T) per maturity row.
	if forward_price is not None:
		working["forward"] = float(forward_price)
	else:
		T_clipped = working["maturity"].clip(lower=1e-6)
		working["forward"] = float(spot_price) * np.exp(
			(float(risk_free_rate) - float(dividend_yield)) * T_clipped
		)

	working["log_moneyness"] = np.log(working["strike"] / working["forward"].replace(0, np.nan))

	grouped = working.groupby(["strike", "maturity", "type"], as_index=False).mean(numeric_only=True)

	for col in ["type"]:
		if col in working.columns:
			grouped[col] = working.groupby(["strike", "maturity", "type"], as_index=False)[col].first()[col]

	grouped = grouped.sort_values(["maturity", "strike"], ascending=True)

	grouped = grouped.ffill()
	grouped = grouped.bfill()

	remaining = len(grouped)
	coverage = 100.0 * remaining / total_rows if total_rows else 0.0
	logger.info("clean_option_chain removed_zero_bid=%d", removed_zero_bid)
	logger.info("clean_option_chain rows_remaining=%d (%.2f%%)", remaining, coverage)

	columns = [
		"strike",
		"maturity",
		"days_to_expiry",
		"moneyness",
		"log_moneyness",
		"forward",
		"mid_price",
		"bid_ask_spread",
		"type",
		"implied_volatility",
	]

	for col in columns:
		if col not in grouped.columns:
			grouped[col] = np.nan

	# Drop internal helper columns before returning
	return grouped[columns].copy()
