"""
Forward Curve Builder.

The fundamental requirement is to work in *forward space*:

    k(K, T) = ln(K / F(T))    (log-moneyness)

where  F(T) = S · exp((r(T) − q(T)) · T)   for a continuous dividend model,
or     F(T) = (S − PV(divs)) · exp(r(T) · T)  when discrete cash dividends
       are present.

This module provides:

  ForwardCurve           — dataclass holding F(T) for a set of maturities.
  ForwardCurveBuilder    — builds F(T) from a rate/dividend schedule.
  infer_forward_from_pcp — infer F from put-call parity on market quotes.
  log_moneyness          — vectorised k = ln(K / F(T)) helper.

Put–Call Parity sanity check (§2.3):
  validate_forward_consistency — compare parity-inferred F with model F;
  flag options whose implied forward deviates by more than a threshold.

Usage::

    builder = ForwardCurveBuilder(spot=100.0, rate=0.05, div_yield=0.01)
    curve   = builder.build(maturities=np.array([0.25, 0.5, 1.0]))
    k       = log_moneyness(strikes=np.array([95, 100, 105]),
                            forward=curve.forward(0.25))
    print(k)   # [-0.051, 0.0, 0.049]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DiscreteDividend:
    """Single cash or yield dividend."""

    ex_date_years: float        # time to ex-date in years from valuation date
    amount: float               # cash amount (set yield=0) OR
    yield_rate: float = 0.0     # continuous dividend yield applicable before ex-date
    is_cash: bool = True        # True = cash, False = yield override


@dataclass
class ForwardCurve:
    """
    Resolved forward prices at a set of maturities.

    Attributes:
        spot:           Spot price at valuation time.
        maturities:     Sorted array of maturities in years.
        forwards:       F(T) for each maturity.
        rates:          r(T) used for each maturity.
        div_yields:     Continuous equivalent dividend yield at each maturity.
        method:         'continuous_yield' | 'discrete_dividends' | 'pcp_inferred'
    """

    spot: float
    maturities: np.ndarray
    forwards: np.ndarray
    rates: np.ndarray
    div_yields: np.ndarray
    method: str = "continuous_yield"

    def forward(self, maturity: float) -> float:
        """
        Interpolate (or extrapolate flat) to get F(T) at an arbitrary maturity.

        Uses linear interpolation in forward space; flat extrapolation at
        boundaries (conservative, avoids spurious behaviour).
        """
        if len(self.maturities) == 1:
            return float(self.forwards[0])
        f = interp1d(
            self.maturities,
            self.forwards,
            kind="linear",
            bounds_error=False,
            fill_value=(self.forwards[0], self.forwards[-1]),
        )
        return float(f(maturity))

    def log_moneyness_grid(
        self,
        strikes: np.ndarray,
        maturity: float,
    ) -> np.ndarray:
        """Return k = ln(K / F(T)) for a strike array at a single maturity."""
        F = self.forward(maturity)
        return np.log(np.asarray(strikes, dtype=float) / F)

    def atm_vol_moneyness(self, maturity: float) -> float:
        """At-the-money log-moneyness (= 0 by construction)."""
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────

class ForwardCurveBuilder:
    """
    Build a :class:`ForwardCurve` from market inputs.

    Two modes:
    1. **Continuous yield** (default) — constant or maturity-dependent rate
       and dividend yield:
           F(T) = S · exp((r(T) − q(T)) · T)

    2. **Discrete dividends** — subtract PV of future dividends first:
           PV(divs, T) = Σ  d_i · exp(−r · t_i)  for t_i < T
           F(T) = (S − PV(divs, T)) · exp(r · T)

    Rate and dividend yield curves may be passed as scalars (flat) or as
    ``(maturity, value)`` pairs (bootstrapped).

    Args:
        spot:            Current spot price.
        rate:            Risk-free rate — scalar or array of shape (n, 2)
                         where column 0 = maturity and column 1 = rate.
        div_yield:       Continuous dividend yield — same format as rate.
        discrete_divs:   Optional list of :class:`DiscreteDividend`.

    Example::

        # Flat rate + yield
        builder = ForwardCurveBuilder(spot=100.0, rate=0.05, div_yield=0.02)
        curve   = builder.build(np.array([0.25, 0.5, 1.0]))
        assert abs(curve.forward(1.0) - 100 * np.exp(0.03)) < 1e-8

        # Rate curve
        rate_curve = np.array([[0.25, 0.045], [0.5, 0.050], [1.0, 0.052]])
        builder2   = ForwardCurveBuilder(spot=100.0, rate=rate_curve, div_yield=0.0)
    """

    def __init__(
        self,
        spot: float,
        rate: "float | np.ndarray" = 0.0,
        div_yield: "float | np.ndarray" = 0.0,
        discrete_divs: Optional[List[DiscreteDividend]] = None,
    ) -> None:
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        self.spot = float(spot)
        self._rate_curve = self._parse_curve(rate, "rate")
        self._div_curve = self._parse_curve(div_yield, "div_yield")
        self.discrete_divs = discrete_divs or []

    # ── curve helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_curve(
        value: "float | np.ndarray",
        name: str,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Return (maturities, values) arrays.  Scalar → single-point 'curve'."""
        if np.isscalar(value):
            t = np.array([0.0, 100.0])
            v = np.array([float(value), float(value)])
            return t, v
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1 and len(arr) == 2:
            # Already (t, v) pair
            arr = arr.reshape(1, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} must be scalar or Nx2 array, got shape {arr.shape}")
        return arr[:, 0], arr[:, 1]

    def _interp_rate(self, T: np.ndarray) -> np.ndarray:
        """Interpolate risk-free rate at maturities T."""
        t_arr, v_arr = self._rate_curve
        if len(t_arr) == 1:
            return np.full_like(T, v_arr[0], dtype=float)
        f = interp1d(t_arr, v_arr, kind="linear", bounds_error=False,
                     fill_value=(v_arr[0], v_arr[-1]))
        return f(T)

    def _interp_div(self, T: np.ndarray) -> np.ndarray:
        """Interpolate continuous dividend yield at maturities T."""
        t_arr, v_arr = self._div_curve
        if len(t_arr) == 1:
            return np.full_like(T, v_arr[0], dtype=float)
        f = interp1d(t_arr, v_arr, kind="linear", bounds_error=False,
                     fill_value=(v_arr[0], v_arr[-1]))
        return f(T)

    # ── discrete dividend PV ───────────────────────────────────────────────

    def _pv_discrete_divs(self, T: float, rate_T: float) -> float:
        """Sum PV of cash dividends with ex-date before T."""
        pv = 0.0
        for div in self.discrete_divs:
            if div.is_cash and div.ex_date_years < T:
                pv += div.amount * np.exp(-rate_T * div.ex_date_years)
        return pv

    # ── public ────────────────────────────────────────────────────────────

    def build(self, maturities: np.ndarray) -> ForwardCurve:
        """
        Build a :class:`ForwardCurve` at the requested maturities.

        Args:
            maturities: 1-D array of maturities in years (need not be sorted).

        Returns:
            ForwardCurve with F(T) at each requested maturity.
        """
        mats = np.asarray(maturities, dtype=float)
        if np.any(mats <= 0):
            raise ValueError("All maturities must be positive.")

        rates = self._interp_rate(mats)
        div_yields = self._interp_div(mats)

        if self.discrete_divs:
            forwards = np.array([
                (self.spot - self._pv_discrete_divs(float(T), float(r)))
                * np.exp(float(r) * float(T))
                for T, r in zip(mats, rates)
            ])
            method = "discrete_dividends"
        else:
            forwards = self.spot * np.exp((rates - div_yields) * mats)
            method = "continuous_yield"

        # Guard against negative forwards (can happen if cash divs > spot)
        if np.any(forwards <= 0):
            logger.warning(
                "Negative forward prices detected — check dividend inputs. "
                "Clamping to 1e-8."
            )
            forwards = np.maximum(forwards, 1e-8)

        return ForwardCurve(
            spot=self.spot,
            maturities=np.sort(mats),
            forwards=forwards[np.argsort(mats)],
            rates=rates[np.argsort(mats)],
            div_yields=div_yields[np.argsort(mats)],
            method=method,
        )

    def forward_at(self, T: float) -> float:
        """Convenience: compute F at a single maturity."""
        return self.build(np.array([T])).forwards[0]


# ──────────────────────────────────────────────────────────────────────────────
# Put–Call Parity forward inference  (§2.3)
# ──────────────────────────────────────────────────────────────────────────────

def infer_forward_from_pcp(
    calls: np.ndarray,
    puts: np.ndarray,
    strikes: np.ndarray,
    maturity: float,
    rate: float,
    min_pairs: int = 3,
) -> Tuple[float, float]:
    """
    Infer the implied forward price from put-call parity.

        C − P = e^{-rT}(F − K)    ⟹    F = K + e^{rT}(C − P)

    Robust estimate: median over all available (call, put, strike) pairs.

    Args:
        calls:     Call mid-prices, shape (n,).
        puts:      Put mid-prices, shape (n,).
        strikes:   Strike prices, shape (n,).
        maturity:  Time to maturity in years.
        rate:      Risk-free rate (flat).
        min_pairs: Minimum number of valid pairs required.

    Returns:
        (forward_estimate, std_dev) — median implied F and std across pairs.

    Raises:
        ValueError: If fewer than ``min_pairs`` valid pairs are available.
    """
    calls = np.asarray(calls, dtype=float)
    puts = np.asarray(puts, dtype=float)
    strikes = np.asarray(strikes, dtype=float)

    valid = (calls > 0) & (puts > 0) & (strikes > 0)
    if valid.sum() < min_pairs:
        raise ValueError(
            f"Need ≥ {min_pairs} valid (call, put, strike) pairs for PCP inference; "
            f"got {valid.sum()}"
        )

    disc = np.exp(rate * maturity)
    implied_forwards = strikes[valid] + disc * (calls[valid] - puts[valid])
    return float(np.median(implied_forwards)), float(np.std(implied_forwards))


# ──────────────────────────────────────────────────────────────────────────────
# Forward-consistency validation  (§2.3)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ForwardConsistencyResult:
    """Result of comparing model-forward vs PCP-implied forward."""

    maturity: float
    model_forward: float
    pcp_forward: float
    deviation_pct: float          # |pcp - model| / model × 100
    pcp_std: float
    passed: bool
    flagged_strike_indices: List[int]  # quotes that imply crazy forwards


def validate_forward_consistency(
    df: pd.DataFrame,
    model_curve: ForwardCurve,
    maturity_col: str = "maturity",
    strike_col: str = "strike",
    call_mid_col: str = "call_mid",
    put_mid_col: str = "put_mid",
    rate: float = 0.0,
    tolerance_pct: float = 2.0,
) -> List[ForwardConsistencyResult]:
    """
    Compare model-implied forwards with put–call-parity-implied forwards.

    For each maturity slice that has both call and put quotes at the same
    strike, compute the PCP-implied forward and flag slices where the
    deviation from the model forward exceeds *tolerance_pct* percent.

    Args:
        df:              DataFrame with option quotes.
        model_curve:     ForwardCurve from :class:`ForwardCurveBuilder`.
        maturity_col:    Column name for maturity in years.
        strike_col:      Column name for strike.
        call_mid_col:    Column name for call mid-price.
        put_mid_col:     Column name for put mid-price.
        rate:            Risk-free rate (flat; used for PCP discount).
        tolerance_pct:   Flag if |PCP F − model F| / model F > tolerance (%).

    Returns:
        List of :class:`ForwardConsistencyResult`, one per maturity.
    """
    results = []

    for mat, group in df.groupby(maturity_col):
        calls = group.get(call_mid_col)
        puts = group.get(put_mid_col)
        if calls is None or puts is None:
            continue
        strikes = group[strike_col].values
        call_vals = calls.values
        put_vals = puts.values

        try:
            pcp_fwd, pcp_std = infer_forward_from_pcp(
                call_vals, put_vals, strikes, float(mat), rate, min_pairs=3
            )
        except ValueError:
            continue

        model_fwd = model_curve.forward(float(mat))
        dev_pct = abs(pcp_fwd - model_fwd) / model_fwd * 100.0
        passed = dev_pct <= tolerance_pct

        # Flag individual strikes where implied forward is very far out
        disc = np.exp(rate * float(mat))
        valid = (call_vals > 0) & (put_vals > 0) & (strikes > 0)
        per_strike_fwd = np.where(valid, strikes + disc * (call_vals - put_vals), np.nan)
        flag_idx = np.where(
            np.abs(per_strike_fwd - model_fwd) / model_fwd * 100 > tolerance_pct * 3
        )[0].tolist()

        results.append(ForwardConsistencyResult(
            maturity=float(mat),
            model_forward=model_fwd,
            pcp_forward=pcp_fwd,
            deviation_pct=dev_pct,
            pcp_std=pcp_std,
            passed=passed,
            flagged_strike_indices=flag_idx,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Log-moneyness helper
# ──────────────────────────────────────────────────────────────────────────────

def log_moneyness(
    strikes: np.ndarray,
    forward: float,
) -> np.ndarray:
    """
    Compute log-moneyness k = ln(K / F).

    k = 0   ⟹  ATM
    k < 0   ⟹  OTM puts / ITM calls
    k > 0   ⟹  OTM calls / ITM puts

    Args:
        strikes: Array of strike prices.
        forward: Forward price F(T).

    Returns:
        Array of log-moneyness values.
    """
    K = np.asarray(strikes, dtype=float)
    if forward <= 0:
        raise ValueError(f"forward must be positive, got {forward}")
    return np.log(K / forward)
