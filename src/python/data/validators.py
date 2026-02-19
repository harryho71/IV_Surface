"""Option chain validation and quality control."""

from __future__ import annotations

from typing import Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OptionDataValidator:
	"""Validates option market data for pricing quality."""

	ACCEPTABLE_IV_RANGE = (0.001, 3.0)  # 0.1% to 300%
	MAX_BID_ASK_SPREAD_PERCENT = 0.10   # 10% spread tolerance

	@staticmethod
	def validate_chain(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Comprehensive option chain validation.

		Checks:
		1. Strike > 0, expiration > today
		2. Bid < Ask (or handle crossed quotes)
		3. No duplicate (strike, expiration) pairs
		4. IV within reasonable range (0.1% - 300%)
		5. Put-call parity violations (flag for investigation)
		6. Spot price consistency across all options
		7. Bid-ask spread reasonableness
		8. Volume/liquidity checks

		Returns:
			(valid_data, rejected_data): DataFrames of valid and rejected rows
		"""
		if df.empty:
			return df.copy(), df.copy()

		working = df.copy()
		working["expiration"] = pd.to_datetime(working.get("expiration"), errors="coerce")
		working["strike"] = pd.to_numeric(working.get("strike"), errors="coerce")
		working["bid"] = pd.to_numeric(working.get("bid"), errors="coerce")
		working["ask"] = pd.to_numeric(working.get("ask"), errors="coerce")
		working["impliedVolatility"] = pd.to_numeric(
			working.get("impliedVolatility"), errors="coerce"
		)
		working["volume"] = pd.to_numeric(working.get("volume"), errors="coerce")

		mask_valid = pd.Series(True, index=working.index)
		reasons = {}

		# Strike > 0
		mask_strike = working["strike"] > 0
		reasons["invalid_strike"] = (~mask_strike).sum()
		mask_valid &= mask_strike

		# Expiration > today  (keep everything tz-naive; yfinance dates are tz-naive)
		today = pd.Timestamp.now().normalize()
		expiration_tz = working["expiration"]
		if expiration_tz.dt.tz is not None:
			expiration_tz = expiration_tz.dt.tz_localize(None)
		mask_exp = expiration_tz > today
		reasons["invalid_expiration"] = (~mask_exp).sum()
		mask_valid &= mask_exp

		# Bid < Ask
		mask_bid_ask = (working["bid"] < working["ask"]) | working["bid"].isna() | working["ask"].isna()
		reasons["crossed_quotes"] = (~mask_bid_ask).sum()
		mask_valid &= mask_bid_ask

		# Duplicates
		dup_mask = working.duplicated(subset=["strike", "expiration"], keep=False)
		reasons["duplicate_strike_exp"] = dup_mask.sum()
		mask_valid &= ~dup_mask

		# IV range
		iv_min, iv_max = OptionDataValidator.ACCEPTABLE_IV_RANGE
		mask_iv = working["impliedVolatility"].between(iv_min, iv_max, inclusive="both") | working["impliedVolatility"].isna()
		reasons["iv_out_of_range"] = (~mask_iv).sum()
		mask_valid &= mask_iv

		# Spread check
		mid = (working["bid"] + working["ask"]) / 2.0
		spread = (working["ask"] - working["bid"]) / mid.replace(0, np.nan)
		mask_spread = (spread <= OptionDataValidator.MAX_BID_ASK_SPREAD_PERCENT) | spread.isna()
		reasons["wide_spread"] = (~mask_spread).sum()
		mask_valid &= mask_spread

		# Volume/liquidity check
		mask_volume = (working["volume"] > 0) | working["volume"].isna()
		reasons["zero_volume"] = (~mask_volume).sum()
		mask_valid &= mask_volume

		# Spot consistency (if spot column exists)
		if "spot" in working.columns:
			spot_vals = pd.to_numeric(working["spot"], errors="coerce")
			spot_std = spot_vals.dropna().std()
			mask_spot = spot_std is np.nan or spot_std <= 1e-6
			if not mask_spot:
				reasons["spot_inconsistent"] = len(working)
				mask_valid &= False

		# Put-call parity (if call/put pairs exist)
		if {"type", "spot", "rate", "maturity"}.issubset(working.columns):
			calls = working[working["type"] == "call"]
			puts = working[working["type"] == "put"]
			merged = calls.merge(
				puts,
				on=["strike", "expiration"],
				suffixes=("_call", "_put"),
			)
			violations = 0
			for _, row in merged.iterrows():
				is_valid, _ = OptionDataValidator.check_put_call_parity(
					row["mid_price_call"] if "mid_price_call" in row else row["bid_call"],
					row["mid_price_put"] if "mid_price_put" in row else row["bid_put"],
					row["strike"],
					row["spot_call"],
					row["rate_call"],
					row["maturity_call"],
					row.get("dividend_yield_call", 0.0),
					0.02,
				)
				if not is_valid:
					violations += 1
			reasons["put_call_parity"] = violations

		valid_df = working[mask_valid].copy()
		rejected_df = working[~mask_valid].copy()

		total = len(working)
		for key, count in reasons.items():
			if total > 0:
				logger.info("validation %s: %d (%.2f%%)", key, count, 100.0 * count / total)
			else:
				logger.info("validation %s: %d", key, count)

		return valid_df, rejected_df

	@staticmethod
	def check_put_call_parity(
		call_price: float,
		put_price: float,
		strike: float,
		spot: float,
		rate: float,
		maturity: float,
		dividend_yield: float = 0.0,
		tolerance: float = 0.02,
	) -> Tuple[bool, float]:
		"""
		Verify put-call parity: C - P = S*exp(-q*T) - K*exp(-r*T).

		Returns:
			(is_valid, parity_diff)
		"""
		lhs = call_price - put_price
		rhs = spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity)
		diff = abs(lhs - rhs)
		return diff <= tolerance, diff
