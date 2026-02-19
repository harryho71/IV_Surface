"""
Benchmark Structure Pricer

Prices standard benchmark structures used in FX/equity vol markets:
  - Risk reversals  (25Δ, 10Δ)  → IV(call_Δ) - IV(put_Δ)
  - Butterflies     (25Δ, 10Δ)  → 0.5*(IV(call_Δ)+IV(put_Δ)) - IV(ATM)
  - Calendar spreads             → C(far_T) - C(near_T) at fixed strike

These are compared against market observables to validate model quality.
Tolerance: |model - market| < 0.5 vol points (bank standard).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass
class BenchmarkResult:
    """Result for a single benchmark structure."""
    structure: str          # "25d_rr", "25d_bf", "10d_rr", "10d_bf", "calendar"
    maturity: float
    model_value: float      # Model-implied value in vol points
    market_value: Optional[float] = None
    diff: Optional[float] = None       # |model - market|
    within_tolerance: Optional[bool] = None


@dataclass
class BenchmarkReport:
    """Full benchmark validation report."""
    results: List[BenchmarkResult]
    max_diff: float
    n_within_tolerance: int
    n_total: int
    tolerance: float
    passed: bool

    def summary(self) -> str:
        lines = [
            "Benchmark Structure Report",
            f"  Tolerance : {self.tolerance * 100:.1f} vol pts",
            f"  Structures: {self.n_total}",
            f"  Within tol: {self.n_within_tolerance}/{self.n_total}",
            f"  Max diff  : {self.max_diff * 100:.3f} vol pts",
            f"  Status    : {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


class BenchmarkStructurePricer:
    """
    Prices benchmark volatility structures for model validation.

    Delta convention: Black-Scholes spot delta (call delta ∈ (0,1)).
      25Δ call  → Δ_call = 0.25
      25Δ put   → Δ_put  = −0.25, i.e. Δ_call = e^{−qT} − 0.25
    """

    def __init__(self, tolerance_vol_pts: float = 0.005):
        """
        Args:
            tolerance_vol_pts: Acceptable |model - market| in vol units (0.005 = 0.5 vol pts).
        """
        self.tolerance = tolerance_vol_pts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bs_call_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or K <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return float(np.exp(-q * T) * norm.cdf(d1))

    @staticmethod
    def _bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    def _build_iv_interpolator(
        self, strikes: np.ndarray, ivs: np.ndarray
    ):
        return interp1d(strikes, ivs, kind="cubic", fill_value="extrapolate", bounds_error=False)

    def _find_delta_strike(
        self,
        target_call_delta: float,
        strikes: np.ndarray,
        ivs: np.ndarray,
        maturity: float,
        spot: float,
        rate: float = 0.0,
        div_yield: float = 0.0,
    ) -> float:
        """
        Find the strike K such that BS call delta == target_call_delta.
        Falls back to approximate log-moneyness if root-finding fails.
        """
        iv_func = self._build_iv_interpolator(strikes, ivs)

        def objective(K):
            sigma = max(float(iv_func(K)), 1e-4)
            return self._bs_call_delta(spot, K, maturity, rate, div_yield, sigma) - target_call_delta

        # bracket: delta is monotone decreasing in K
        try:
            K_star = brentq(objective, strikes[0] * 0.5, strikes[-1] * 1.5, xtol=1e-4)
            return float(K_star)
        except Exception:
            # Approximate: use ATM vol and invert analytically
            iv_atm = float(iv_func(spot))
            sigma_safe = max(iv_atm, 0.01)
            d1_target = norm.ppf(target_call_delta * np.exp(div_yield * maturity))
            K_approx = spot * np.exp(
                (rate - div_yield + 0.5 * sigma_safe ** 2) * maturity
                - d1_target * sigma_safe * np.sqrt(maturity)
            )
            return float(np.clip(K_approx, strikes[0], strikes[-1]))

    def _get_iv_at(self, K: float, strikes: np.ndarray, ivs: np.ndarray) -> float:
        iv_func = self._build_iv_interpolator(strikes, ivs)
        return max(float(iv_func(K)), 1e-4)

    # ------------------------------------------------------------------
    # Public pricing methods
    # ------------------------------------------------------------------

    def price_risk_reversal(
        self,
        delta: float,
        strikes: np.ndarray,
        ivs: np.ndarray,
        maturity: float,
        spot: float,
        rate: float = 0.0,
        div_yield: float = 0.0,
    ) -> float:
        """
        Risk reversal = IV(Δ_call) − IV(Δ_put).

        Args:
            delta: Absolute delta magnitude (e.g. 0.25 for 25Δ).

        Returns:
            RR in vol units (positive = upside skew).
        """
        # 25Δ call: Δ_call = 0.25
        K_call = self._find_delta_strike(delta, strikes, ivs, maturity, spot, rate, div_yield)
        # 25Δ put:  Δ_call = e^{−qT} − 0.25
        target_put_side = np.exp(-div_yield * maturity) - delta
        K_put = self._find_delta_strike(target_put_side, strikes, ivs, maturity, spot, rate, div_yield)

        iv_call = self._get_iv_at(K_call, strikes, ivs)
        iv_put = self._get_iv_at(K_put, strikes, ivs)
        return iv_call - iv_put

    def price_butterfly(
        self,
        delta: float,
        strikes: np.ndarray,
        ivs: np.ndarray,
        maturity: float,
        spot: float,
        rate: float = 0.0,
        div_yield: float = 0.0,
    ) -> float:
        """
        Butterfly = 0.5*(IV(Δ_call) + IV(Δ_put)) − IV(ATM).

        Returns:
            BF in vol units (positive = wings more expensive than ATM).
        """
        K_call = self._find_delta_strike(delta, strikes, ivs, maturity, spot, rate, div_yield)
        target_put_side = np.exp(-div_yield * maturity) - delta
        K_put = self._find_delta_strike(target_put_side, strikes, ivs, maturity, spot, rate, div_yield)

        iv_call = self._get_iv_at(K_call, strikes, ivs)
        iv_put = self._get_iv_at(K_put, strikes, ivs)
        # ATM: Δ_call = 0.5  (ATMF approximation)
        target_atm = 0.5 * np.exp(-div_yield * maturity)
        K_atm = self._find_delta_strike(target_atm, strikes, ivs, maturity, spot, rate, div_yield)
        iv_atm = self._get_iv_at(K_atm, strikes, ivs)

        return 0.5 * (iv_call + iv_put) - iv_atm

    def price_calendar_spread(
        self,
        strike: float,
        near_maturity: float,
        far_maturity: float,
        iv_near: float,
        iv_far: float,
        spot: float,
        rate: float = 0.0,
        div_yield: float = 0.0,
    ) -> float:
        """
        Calendar spread = C(far_T) − C(near_T) at fixed strike.

        Should be ≥ 0 for no calendar arbitrage.
        """
        c_near = self._bs_call_price(spot, strike, near_maturity, rate, div_yield, iv_near)
        c_far = self._bs_call_price(spot, strike, far_maturity, rate, div_yield, iv_far)
        return c_far - c_near

    # ------------------------------------------------------------------
    # Full benchmark sweep
    # ------------------------------------------------------------------

    def run_full_benchmark(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray,
        spot: float,
        rate: float = 0.0,
        div_yield: float = 0.0,
        market_structures: Optional[Dict[str, float]] = None,
    ) -> BenchmarkReport:
        """
        Price all benchmark structures across all maturities.

        Args:
            strikes: (n_strikes,)
            maturities: (n_maturities,)
            ivs: (n_strikes, n_maturities)
            spot: spot price
            rate: risk-free rate
            div_yield: dividend yield
            market_structures: Optional dict mapping key → market value.
                Keys: "{25d|10d}_{rr|bf}_{T:.4f}"  e.g. "25d_rr_0.2500"

        Returns:
            BenchmarkReport
        """
        results: List[BenchmarkResult] = []

        for t_idx, maturity in enumerate(maturities):
            iv_slice = ivs[:, t_idx]

            for delta_label, delta_val in [("25d", 0.25), ("10d", 0.10)]:
                # Risk reversal
                rr = self.price_risk_reversal(delta_val, strikes, iv_slice, maturity, spot, rate, div_yield)
                rr_key = f"{delta_label}_rr_{maturity:.4f}"
                mkt_rr = market_structures.get(rr_key) if market_structures else None
                diff_rr = abs(rr - mkt_rr) if mkt_rr is not None else None
                results.append(BenchmarkResult(
                    structure=f"{delta_label}_rr",
                    maturity=maturity,
                    model_value=rr,
                    market_value=mkt_rr,
                    diff=diff_rr,
                    within_tolerance=(diff_rr <= self.tolerance) if diff_rr is not None else None,
                ))

                # Butterfly
                bf = self.price_butterfly(delta_val, strikes, iv_slice, maturity, spot, rate, div_yield)
                bf_key = f"{delta_label}_bf_{maturity:.4f}"
                mkt_bf = market_structures.get(bf_key) if market_structures else None
                diff_bf = abs(bf - mkt_bf) if mkt_bf is not None else None
                results.append(BenchmarkResult(
                    structure=f"{delta_label}_bf",
                    maturity=maturity,
                    model_value=bf,
                    market_value=mkt_bf,
                    diff=diff_bf,
                    within_tolerance=(diff_bf <= self.tolerance) if diff_bf is not None else None,
                ))

        # Calendar spreads between consecutive maturities at ATM strike
        if len(maturities) >= 2:
            atm_k_idx = int(np.argmin(np.abs(strikes - spot)))
            atm_strike = strikes[atm_k_idx]
            for t_idx in range(len(maturities) - 1):
                cs = self.price_calendar_spread(
                    strike=atm_strike,
                    near_maturity=maturities[t_idx],
                    far_maturity=maturities[t_idx + 1],
                    iv_near=ivs[atm_k_idx, t_idx],
                    iv_far=ivs[atm_k_idx, t_idx + 1],
                    spot=spot,
                    rate=rate,
                    div_yield=div_yield,
                )
                results.append(BenchmarkResult(
                    structure="calendar",
                    maturity=maturities[t_idx + 1],
                    model_value=cs,
                    within_tolerance=(cs >= -1e-8),  # non-negative = no calendar arb
                ))

        # Aggregate
        diffs = [r.diff for r in results if r.diff is not None]
        max_diff = float(max(diffs)) if diffs else 0.0
        n_total = sum(1 for r in results if r.within_tolerance is not None)
        n_within = sum(1 for r in results if r.within_tolerance is True)

        return BenchmarkReport(
            results=results,
            max_diff=max_diff,
            n_within_tolerance=n_within,
            n_total=n_total,
            tolerance=self.tolerance,
            passed=(max_diff <= self.tolerance) if diffs else True,
        )
