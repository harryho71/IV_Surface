"""Data fetchers for option market data from multiple sources."""

from __future__ import annotations

from typing import Dict, Optional
import logging
import time

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_option_chain_yfinance(
	ticker: str,
	expiration_date: Optional[str] = None,
) -> pd.DataFrame:
	"""
	Fetch option chain from Yahoo Finance.

	Args:
		ticker: Stock symbol (e.g., 'AAPL', 'SPY').
		expiration_date: YYYY-MM-DD (default: next expiration).

	Returns:
		DataFrame with columns:
		strike, bid, ask, volume, openInterest, impliedVolatility, type, expiration
	"""
	timestamp = pd.Timestamp.utcnow()
	logger.info("fetch_option_chain_yfinance source=yfinance ticker=%s time=%s", ticker, timestamp)

	try:
		yf_ticker = yf.Ticker(ticker)
		expirations = yf_ticker.options or []
		if not expirations:
			logger.warning("No expirations found for ticker=%s", ticker)
			return pd.DataFrame(
				columns=[
					"strike",
					"bid",
					"ask",
					"volume",
					"openInterest",
					"impliedVolatility",
					"type",
					"expiration",
				]
			)

		exp = expiration_date or expirations[0]
		chain = yf_ticker.option_chain(exp)

		calls = chain.calls.copy()
		calls["type"] = "call"
		puts = chain.puts.copy()
		puts["type"] = "put"

		combined = pd.concat([calls, puts], ignore_index=True)
		combined["expiration"] = exp

		keep_cols = [
			"strike",
			"bid",
			"ask",
			"volume",
			"openInterest",
			"impliedVolatility",
			"type",
			"expiration",
		]
		for col in keep_cols:
			if col not in combined.columns:
				combined[col] = pd.NA

		return combined[keep_cols].copy()
	except Exception as exc:  # pragma: no cover - network errors
		logger.exception("fetch_option_chain_yfinance failed: %s", exc)
		return pd.DataFrame(
			columns=[
				"strike",
				"bid",
				"ask",
				"volume",
				"openInterest",
				"impliedVolatility",
				"type",
				"expiration",
			]
		)


def fetch_from_csv(filepath: str, columns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
	"""
	Load option chain from CSV file with flexible column mapping.

	Args:
		filepath: Path to CSV file.
		columns: Dict mapping standard names to file column names
				 {'strike': 'Strike', 'bid': 'Bid_Price', ...}

	Returns:
		Standardized DataFrame with columns:
		strike, bid, ask, volume, openInterest, impliedVolatility, type, expiration
	"""
	timestamp = pd.Timestamp.utcnow()
	logger.info("fetch_from_csv source=csv path=%s time=%s", filepath, timestamp)

	try:
		df = pd.read_csv(filepath)
	except Exception as exc:  # pragma: no cover - file errors
		logger.exception("fetch_from_csv failed to read: %s", exc)
		return pd.DataFrame(
			columns=[
				"strike",
				"bid",
				"ask",
				"volume",
				"openInterest",
				"impliedVolatility",
				"type",
				"expiration",
			]
		)

	mapping = columns or {}
	standardized = pd.DataFrame()
	standardized["strike"] = df.get(mapping.get("strike", "strike"))
	standardized["bid"] = df.get(mapping.get("bid", "bid"))
	standardized["ask"] = df.get(mapping.get("ask", "ask"))
	standardized["volume"] = df.get(mapping.get("volume", "volume"))
	standardized["openInterest"] = df.get(mapping.get("openInterest", "openInterest"))
	standardized["impliedVolatility"] = df.get(mapping.get("impliedVolatility", "impliedVolatility"))
	standardized["type"] = df.get(mapping.get("type", "type"))
	standardized["expiration"] = df.get(mapping.get("expiration", "expiration"))

	for col in ["strike", "bid", "ask", "volume", "openInterest", "impliedVolatility"]:
		standardized[col] = pd.to_numeric(standardized[col], errors="coerce")

	return standardized


def fetch_from_api(api_endpoint: str, params: Optional[Dict[str, str]] = None) -> pd.DataFrame:
	"""
	Generic API fetcher with pagination and rate limiting.

	Args:
		api_endpoint: Base endpoint URL.
		params: Query parameters dict.

	Returns:
		DataFrame with columns:
		strike, bid, ask, volume, openInterest, impliedVolatility, type, expiration
	"""
	timestamp = pd.Timestamp.utcnow()
	logger.info("fetch_from_api source=api url=%s time=%s", api_endpoint, timestamp)

	all_rows = []
	page = 1
	params = dict(params or {})

	while True:
		params["page"] = page
		try:
			response = requests.get(api_endpoint, params=params, timeout=30)
			response.raise_for_status()
			payload = response.json()
		except Exception as exc:  # pragma: no cover - network errors
			logger.exception("fetch_from_api failed: %s", exc)
			break

		rows = payload.get("data") if isinstance(payload, dict) else payload
		if not rows:
			break

		all_rows.extend(rows)

		has_more = False
		if isinstance(payload, dict):
			has_more = bool(payload.get("has_more", False))
		if not has_more:
			break

		page += 1
		time.sleep(0.25)

	df = pd.DataFrame(all_rows)
	if df.empty:
		return pd.DataFrame(
			columns=[
				"strike",
				"bid",
				"ask",
				"volume",
				"openInterest",
				"impliedVolatility",
				"type",
				"expiration",
			]
		)

	keep_cols = [
		"strike",
		"bid",
		"ask",
		"volume",
		"openInterest",
		"impliedVolatility",
		"type",
		"expiration",
	]
	for col in keep_cols:
		if col not in df.columns:
			df[col] = pd.NA

	for col in ["strike", "bid", "ask", "volume", "openInterest", "impliedVolatility"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	return df[keep_cols].copy()
