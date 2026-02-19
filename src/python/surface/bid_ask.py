"""
Bid/Ask Surface Generator

Generates bid and ask IV surfaces from a mid IV surface using:
  1. generate()              — flat uniform spread in basis points
  2. generate_with_skew()    — wider wings, tighter ATM
  3. generate_from_market()  — derive mid from observed bid/ask quotes
  4. validate_spreads()      — sanity-check an existing bid/mid/ask triplet

All methods guarantee:
  bid_ivs ≤ mid_ivs ≤ ask_ivs  and  bid_ivs > 0

Spread is expressed in basis points (bps) where 100 bps = 1 vol point.

Example::

    gen = BidAskSurfaceGenerator()
    bid, ask, report = gen.generate(mid_ivs, spread_bps=10)
    print(report.summary())

    bid2, ask2, report2 = gen.generate_with_skew(
        mid_ivs, strikes, atm_strike=100, base_spread_bps=10, wing_multiplier=2.0
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpreadReport:
    """Summary statistics for a generated bid/ask spread surface."""

    mean_spread_bps: float
    max_spread_bps: float
    min_spread_bps: float
    mean_mid_iv: float
    bid_ask_ratio: float        # mean_spread / mean_mid_iv  (fraction of vol)
    within_market: bool         # True if all spreads ≤ max_spread_bps

    def summary(self) -> str:
        return (
            f"SpreadReport | mean={self.mean_spread_bps:.1f} bps "
            f"min={self.min_spread_bps:.1f} max={self.max_spread_bps:.1f} | "
            f"ratio={self.bid_ask_ratio * 100:.2f}% of vol | "
            f"{'OK' if self.within_market else 'WIDE'}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

class BidAskSurfaceGenerator:
    """
    Generates bid / ask IV surfaces from a mid IV surface.

    Args:
        min_spread_bps: Floor on the spread (prevents zero spread).
                        Default 5 bps.
        max_spread_bps: Cap on the spread (prevents unreasonable widening).
                        Default 500 bps (= 5 vol points).
    """

    def __init__(
        self,
        min_spread_bps: float = 5.0,
        max_spread_bps: float = 500.0,
    ) -> None:
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_vol(bps: np.ndarray) -> np.ndarray:
        """Convert basis-point array to decimal vol units (100 bps = 0.01)."""
        return bps / 10_000.0

    def _clip(self, spread_bps: np.ndarray) -> np.ndarray:
        return np.clip(spread_bps, self.min_spread_bps, self.max_spread_bps)

    def _build_report(
        self,
        spread_bps: np.ndarray,
        mid_ivs: np.ndarray,
    ) -> SpreadReport:
        ratio = float(
            np.mean(spread_bps / 10_000.0) / (np.mean(mid_ivs) + 1e-12)
        )
        return SpreadReport(
            mean_spread_bps=float(np.mean(spread_bps)),
            max_spread_bps=float(np.max(spread_bps)),
            min_spread_bps=float(np.min(spread_bps)),
            mean_mid_iv=float(np.mean(mid_ivs)),
            bid_ask_ratio=ratio,
            within_market=bool(np.all(spread_bps <= self.max_spread_bps)),
        )

    # ── public constructors ────────────────────────────────────────────────

    def generate(
        self,
        mid_ivs: np.ndarray,
        spread_bps: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray, SpreadReport]:
        """
        Flat spread: bid = mid − half_spread, ask = mid + half_spread.

        Args:
            mid_ivs:    (n_strikes, n_maturities) mid IV surface.
            spread_bps: Total bid/ask spread in basis points.

        Returns:
            ``(bid_ivs, ask_ivs, SpreadReport)``
        """
        mid_ivs = np.asarray(mid_ivs, dtype=float)
        clipped = self._clip(np.full_like(mid_ivs, spread_bps))
        half = self._to_vol(clipped) / 2.0
        bid_ivs = np.maximum(mid_ivs - half, 1e-4)
        ask_ivs = mid_ivs + half
        return bid_ivs, ask_ivs, self._build_report(clipped, mid_ivs)

    def generate_with_skew(
        self,
        mid_ivs: np.ndarray,
        strikes: np.ndarray,
        atm_strike: float,
        base_spread_bps: float = 10.0,
        wing_multiplier: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, SpreadReport]:
        """
        Strike-dependent spread: wings are wider than ATM.

        ``spread(K) = base × [1 + (wing_multiplier − 1) × moneyness²]``

        where ``moneyness = |(K − ATM) / ATM|``.

        Args:
            mid_ivs:          (n_strikes, n_maturities) mid IV surface.
            strikes:          1-D array of strikes, length n_strikes.
            atm_strike:       ATM reference strike.
            base_spread_bps:  Spread at ATM in basis points.
            wing_multiplier:  Multiplier applied at the extreme wings (≥ 1).

        Returns:
            ``(bid_ivs, ask_ivs, SpreadReport)``
        """
        mid_ivs = np.asarray(mid_ivs, dtype=float)
        moneyness = np.abs((strikes - atm_strike) / atm_strike)         # (n_K,)
        spread_vec = base_spread_bps * (
            1.0 + (wing_multiplier - 1.0) * moneyness ** 2
        )                                                                # (n_K,)
        spread_bps_arr = self._clip(
            spread_vec[:, np.newaxis] * np.ones_like(mid_ivs)
        )                                                                # (n_K, n_T)
        half = self._to_vol(spread_bps_arr) / 2.0
        bid_ivs = np.maximum(mid_ivs - half, 1e-4)
        ask_ivs = mid_ivs + half
        return bid_ivs, ask_ivs, self._build_report(spread_bps_arr, mid_ivs)

    def generate_from_market(
        self,
        bid_ivs: np.ndarray,
        ask_ivs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SpreadReport]:
        """
        Derive mid from observed market bid/ask IVs.

        Args:
            bid_ivs: Observed bid IVs.
            ask_ivs: Observed ask IVs (must be ≥ bid_ivs everywhere).

        Returns:
            ``(bid_ivs, mid_ivs, ask_ivs, SpreadReport)``

        Raises:
            ValueError: If ``ask_ivs < bid_ivs`` at any point.
        """
        bid_ivs = np.asarray(bid_ivs, dtype=float)
        ask_ivs = np.asarray(ask_ivs, dtype=float)
        if np.any(ask_ivs < bid_ivs):
            raise ValueError("ask_ivs must be ≥ bid_ivs everywhere")
        mid_ivs = (bid_ivs + ask_ivs) / 2.0
        spread_bps = (ask_ivs - bid_ivs) * 10_000.0
        clipped = self._clip(spread_bps)
        return bid_ivs, mid_ivs, ask_ivs, self._build_report(clipped, mid_ivs)

    def validate_spreads(
        self,
        bid_ivs: np.ndarray,
        ask_ivs: np.ndarray,
        mid_ivs: np.ndarray,
    ) -> dict:
        """
        Validate that bid ≤ mid ≤ ask everywhere and report statistics.

        Returns a dict with keys:
          - ``valid``            — True if all bid ≤ mid ≤ ask
          - ``n_bid_violations`` — cells where bid > mid
          - ``n_ask_violations`` — cells where ask < mid
          - ``mean_spread_bps``  — average bid/ask spread in bps
          - ``max_spread_bps``   — maximum bid/ask spread in bps
        """
        bid_ivs = np.asarray(bid_ivs, dtype=float)
        ask_ivs = np.asarray(ask_ivs, dtype=float)
        mid_ivs = np.asarray(mid_ivs, dtype=float)
        n_bid = int(np.sum(bid_ivs > mid_ivs))
        n_ask = int(np.sum(ask_ivs < mid_ivs))
        spread_bps = (ask_ivs - bid_ivs) * 10_000.0
        return {
            "valid": n_bid == 0 and n_ask == 0,
            "n_bid_violations": n_bid,
            "n_ask_violations": n_ask,
            "mean_spread_bps": float(np.mean(spread_bps)),
            "max_spread_bps": float(np.max(spread_bps)),
        }
