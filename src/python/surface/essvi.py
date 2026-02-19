"""
eSSVI — Extended SSVI Surface.

An (e)SSVI-style surface is the most pragmatic high-performance choice
for equity/FX vol surface construction today.

This module extends the standard SSVI parametrisation
(Gatheral–Jacquier 2014) with a maturity-dependent correlation parameter
ρ(θ) and an extended φ function that gives more flexibility to match
observed skew dynamics across maturities while preserving the analytic
no-arbitrage guarantees.

Standard SSVI:
    w(k; θ) = θ/2 · [1 + ρ·φ(θ)·k + √((φ(θ)·k + ρ)² + (1−ρ²))]
    φ(θ)    = η / θ^γ
    ρ        = const

Extended SSVI (eSSVI) — this module:
    ρ(θ)    = ρ₀ · exp(−λ_ρ · θ)      (decaying correlation with ATM var)
    φ(θ)    = η / θ^γ                 (unchanged; power-law)
    Arbitrage-free constraints: same Gatheral–Jacquier butterfly + calendar
    conditions, now enforced *per θ* rather than globally.

Arbitrage-free parameter constraints (from G-J, Theorem 4.2):
    Butterfly:   θ · φ(θ)² · (1 + |ρ(θ)|) ≤ 4          ← one inequality per θ
    Calendar:    dθ/dT ≥ 0                               ← θ non-decreasing

Why extend ρ?
  The implied correlation between forward and vol is empirically
  non-constant across maturities: equity index smiles are steep in the
  short end and flatten at the long end.  A single ρ forces a trade-off.
  ρ(θ) with a decay rate λ_ρ resolves this without adding more parameters
  than necessary.

Usage::

    from src.python.surface.essvi import ESSVIParameters, ESSVISurface

    surf = ESSVISurface(
        forward=100.0,
        maturities=np.array([0.25, 0.5, 1.0, 2.0]),
    )
    result = surf.calibrate(
        strikes=np.linspace(80, 120, 9),
        market_ivs=np.random.uniform(0.15, 0.30, (9, 4)),
    )
    w = surf.total_variance(log_moneyness=0.0, maturity=0.5)
    sigma = surf.implied_vol(log_moneyness=0.0, maturity=0.5)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import Bounds, minimize

try:
    pass
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Parameter containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ESSVIParameters:
    """
    eSSVI model parameters.

    Attributes:
        theta_curve: ATM total-variance values θ(T) at each maturity (n_T,).
        maturities:  Maturities corresponding to theta_curve (n_T,), years.
        eta:         φ(θ) scale parameter; φ(θ) = η / θ^γ.
        gamma:       φ(θ) power; γ ∈ (0, 1) for realistic smiles.
        rho_0:       Long-run correlation intercept; ρ(θ) = ρ₀·exp(−λ_ρ·θ).
        lambda_rho:  Correlation decay rate; λ_ρ ≥ 0.  At λ_ρ=0, recovers
                     standard SSVI with constant ρ = ρ₀.
    """

    theta_curve: np.ndarray
    maturities: np.ndarray
    eta: float
    gamma: float
    rho_0: float
    lambda_rho: float

    # ── arbitrage-free constraint check ───────────────────────────────────

    def butterfly_bound(self) -> np.ndarray:
        """
        Return butterfly constraint value per maturity.

        Constraint:  θ · φ(θ)² · (1 + |ρ(θ)|) ≤ 4
        Returns array of (lhs - 4); negative = satisfied.
        """
        phi = self.eta / (self.theta_curve ** self.gamma)
        rho = self.rho_0 * np.exp(-self.lambda_rho * self.theta_curve)
        lhs = self.theta_curve * phi ** 2 * (1.0 + np.abs(rho))
        return lhs - 4.0  # ≤ 0 required

    def is_arbitrage_free(self) -> bool:
        """True if all no-arb constraints are satisfied."""
        # Butterfly per slice
        if np.any(self.butterfly_bound() > 0):
            return False
        # Calendar: θ non-decreasing
        if np.any(np.diff(self.theta_curve) < 0):
            return False
        # ρ₀ ∈ (−1, 1)
        if abs(self.rho_0) >= 1.0:
            return False
        return True

    def rho_at_theta(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate ρ(θ) = ρ₀ · exp(−λ_ρ · θ)."""
        return self.rho_0 * np.exp(-self.lambda_rho * np.asarray(theta))

    def phi_at_theta(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate φ(θ) = η / θ^γ."""
        return self.eta / (np.asarray(theta) ** self.gamma)

    # ── serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theta_curve": self.theta_curve.tolist(),
            "maturities": self.maturities.tolist(),
            "eta": float(self.eta),
            "gamma": float(self.gamma),
            "rho_0": float(self.rho_0),
            "lambda_rho": float(self.lambda_rho),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ESSVIParameters:
        return cls(
            theta_curve=np.array(data["theta_curve"]),
            maturities=np.array(data["maturities"]),
            eta=float(data["eta"]),
            gamma=float(data["gamma"]),
            rho_0=float(data["rho_0"]),
            lambda_rho=float(data.get("lambda_rho", 0.0)),
        )


@dataclass
class ESSVICalibrationResult:
    """Result of an eSSVI calibration."""

    parameters: ESSVIParameters
    rmse: float
    max_error: float
    converged: bool
    iterations: int
    arbitrage_free: bool
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": self.parameters.to_dict(),
            "fit_quality": {
                "rmse": float(self.rmse),
                "max_error": float(self.max_error),
                "converged": self.converged,
                "iterations": self.iterations,
            },
            "arbitrage_free": self.arbitrage_free,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core surface class
# ──────────────────────────────────────────────────────────────────────────────

class ESSVISurface:
    """
    Extended SSVI volatility surface.

    Parametrisation (extends Gatheral–Jacquier 2014):

        w(k, T) = θ(T)/2 · [1 + ρ(θ)·φ(θ)·k + √((φ(θ)·k + ρ(θ))² + (1−ρ(θ)²))]

    where
        θ(T)   = ATM total variance at maturity T  (interpolated)
        φ(θ)   = η / θ^γ
        ρ(θ)   = ρ₀ · exp(−λ_ρ · θ)

    λ_ρ = 0 recovers standard SSVI with constant ρ = ρ₀.

    Args:
        forward:    Forward price F(T) (used only if market_ivs are in
                    strike space; log-moneyness inputs bypass this).
        maturities: Sorted maturities array (years).
    """

    def __init__(
        self,
        forward: float = 100.0,
        maturities: Optional[np.ndarray] = None,
    ) -> None:
        self.forward = forward
        self.maturities = np.asarray(maturities) if maturities is not None else np.array([])
        self.parameters: Optional[ESSVIParameters] = None
        self._theta_interp = None
        self.logger = logging.getLogger(__name__)

    # ── core formula ───────────────────────────────────────────────────────

    @staticmethod
    def _w_slice(
        k: np.ndarray,
        theta: float,
        phi: float,
        rho: float,
    ) -> np.ndarray:
        """
        eSSVI total variance for a single maturity slice.

        w(k) = θ/2 · [1 + ρφk + √((φk + ρ)² + (1−ρ²))]
        """
        phi_k = phi * k
        disc = np.sqrt((phi_k + rho) ** 2 + (1.0 - rho ** 2))
        return theta / 2.0 * (1.0 + rho * phi_k + disc)

    def total_variance(
        self,
        log_moneyness: "float | np.ndarray",
        maturity: float,
    ) -> "float | np.ndarray":
        """
        Evaluate eSSVI total variance w(k, T).

        Args:
            log_moneyness: k = ln(K/F), scalar or array.
            maturity:      T in years.

        Returns:
            w(k, T) ≥ 0.
        """
        if self.parameters is None:
            raise RuntimeError("Model not calibrated — call calibrate() first.")

        theta = self._interpolate_theta(maturity)
        phi = self.parameters.phi_at_theta(np.array([theta]))[0]
        rho = self.parameters.rho_at_theta(np.array([theta]))[0]
        k = np.asarray(log_moneyness, dtype=float)
        w = self._w_slice(k, theta, phi, rho)
        return float(w) if w.ndim == 0 else w

    def implied_vol(
        self,
        log_moneyness: "float | np.ndarray",
        maturity: float,
    ) -> "float | np.ndarray":
        """
        Implied volatility σ(k, T) = √(w(k, T) / T).
        """
        w = self.total_variance(log_moneyness, maturity)
        return np.sqrt(np.maximum(w, 0.0) / max(maturity, 1e-10))

    def implied_vol_grid(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
    ) -> np.ndarray:
        """
        Compute IV on a (n_K, n_T) grid.

        Args:
            strikes:   (n_K,) array — raw strike prices.
            maturities:(n_T,) array — maturities in years.

        Returns:
            (n_K, n_T) array of implied vols.
        """
        n_K, n_T = len(strikes), len(maturities)
        ivs = np.zeros((n_K, n_T))
        for t_idx, T in enumerate(maturities):
            k = np.log(strikes / self.forward)
            ivs[:, t_idx] = self.implied_vol(k, T)
        return ivs

    # ── theta interpolation ────────────────────────────────────────────────

    def _interpolate_theta(self, T: float) -> float:
        """
        Interpolate ATM total variance θ(T).

        Uses linear interpolation in (T, θ) space.  Flat extrapolation at
        boundaries ensures no calendar-arb violations outside the grid.
        """
        if self._theta_interp is None:
            raise RuntimeError("call calibrate() first")
        return float(np.clip(
            float(self._theta_interp(T)),
            1e-6,
            None,
        ))

    def _build_theta_interp(self) -> None:
        from scipy.interpolate import interp1d
        p = self.parameters
        self._theta_interp = interp1d(
            p.maturities,
            p.theta_curve,
            kind="linear",
            bounds_error=False,
            fill_value=(p.theta_curve[0], p.theta_curve[-1]),
        )

    # ── calibration ────────────────────────────────────────────────────────

    def calibrate(
        self,
        strikes: np.ndarray,
        market_ivs: np.ndarray,
        maturities: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        prev_params: Optional[ESSVIParameters] = None,
        tikhonov_lambda: float = 0.01,
        max_iter: int = 500,
    ) -> ESSVICalibrationResult:
        """
        Calibrate eSSVI to market implied vols.

        Args:
            strikes:       (n_K,) or (n_K, n_T) array of strikes.
            market_ivs:    (n_K, n_T) array of market implied vols.
            maturities:    (n_T,) maturities; uses self.maturities if None.
            weights:       Optional (n_K, n_T) weight grid (e.g. vega weights).
                           Default: uniform.
            prev_params:   Previous day's parameters for Tikhonov
                           regularisation (warm-start + stability).
            tikhonov_lambda: Regularisation strength vs prev_params.
            max_iter:      Maximum optimiser iterations.

        Returns:
            :class:`ESSVICalibrationResult`.
        """
        mats = maturities if maturities is not None else self.maturities
        mats = np.asarray(mats, dtype=float)
        self.maturities = mats

        n_K = market_ivs.shape[0]
        n_T = len(mats)

        if strikes.ndim == 1:
            strikes_2d = np.tile(strikes[:, None], (1, n_T))
        else:
            strikes_2d = strikes

        # Log-moneyness grid
        k_grid = np.log(strikes_2d / self.forward)

        # Weights
        W = weights if weights is not None else np.ones_like(market_ivs)

        # Initial guess
        x0 = self._make_initial_guess(market_ivs, mats)

        # Bounds
        bnds = self._make_bounds(n_T)

        # Linear constraints: θ non-decreasing
        if n_T > 1:
            A = np.zeros((n_T - 1, n_T + 4))
            for i in range(n_T - 1):
                A[i, i] = -1.0
                A[i, i + 1] = 1.0
            lc = {"type": "ineq", "fun": lambda x, A=A: A @ x}
            constraints = [lc]
        else:
            constraints = []

        result = minimize(
            self._objective,
            x0,
            args=(k_grid, market_ivs, mats, W, prev_params, tikhonov_lambda),
            method="SLSQP",
            bounds=bnds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-9},
        )

        p = self._x_to_params(result.x, n_T, mats)
        self.parameters = p
        self._build_theta_interp()

        # Residuals
        iv_hat = self.implied_vol_grid(strikes_2d[:, 0], mats)
        residuals = iv_hat - market_ivs
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        max_err = float(np.max(np.abs(residuals)))

        return ESSVICalibrationResult(
            parameters=p,
            rmse=rmse,
            max_error=max_err,
            converged=result.success,
            iterations=result.nit,
            arbitrage_free=p.is_arbitrage_free(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ── optimisation internals ─────────────────────────────────────────────

    def _make_initial_guess(
        self, market_ivs: np.ndarray, mats: np.ndarray
    ) -> np.ndarray:
        n_T = len(mats)
        theta0 = np.array([
            float((market_ivs[:, j].mean() ** 2) * mats[j])
            for j in range(n_T)
        ])
        # Enforce monotonicity in initial guess
        for i in range(1, n_T):
            theta0[i] = max(theta0[i], theta0[i - 1] + 1e-6)
        return np.concatenate([theta0, [1.0, 0.5, -0.3, 0.5]])

    def _make_bounds(self, n_T: int) -> Bounds:
        lower = np.concatenate([
            np.full(n_T, 1e-4),   # theta ≥ 1e-4
            [0.01, 0.01, -0.99, 0.0],  # eta, gamma, rho_0, lambda_rho
        ])
        upper = np.concatenate([
            np.full(n_T, 2.0),    # theta ≤ 2.0
            [20.0, 0.99, 0.99, 5.0],
        ])
        return Bounds(lower, upper)

    def _x_to_params(
        self, x: np.ndarray, n_T: int, mats: np.ndarray
    ) -> ESSVIParameters:
        return ESSVIParameters(
            theta_curve=x[:n_T].copy(),
            maturities=mats.copy(),
            eta=float(x[n_T]),
            gamma=float(x[n_T + 1]),
            rho_0=float(x[n_T + 2]),
            lambda_rho=float(x[n_T + 3]),
        )

    def _objective(
        self,
        x: np.ndarray,
        k_grid: np.ndarray,
        market_ivs: np.ndarray,
        mats: np.ndarray,
        W: np.ndarray,
        prev_params: Optional[ESSVIParameters],
        lam: float,
    ) -> float:
        n_T = len(mats)
        p = self._x_to_params(x, n_T, mats)

        # Butterfly penalty: max(0, θφ²(1+|ρ|) - 4)²
        bf_viol = np.maximum(0.0, p.butterfly_bound())
        arb_penalty = 1e4 * float(np.sum(bf_viol ** 2))

        # Fit residuals
        fit_loss = 0.0
        for j, T in enumerate(mats):
            theta = float(p.theta_curve[j])
            phi = float(p.phi_at_theta(np.array([theta]))[0])
            rho = float(p.rho_at_theta(np.array([theta]))[0])
            w_model = self._w_slice(k_grid[:, j], theta, phi, rho)
            sigma_sq_T = np.maximum(w_model, 1e-8)
            sigma_model = np.sqrt(sigma_sq_T / max(T, 1e-10))
            fit_loss += float(np.sum(W[:, j] * (sigma_model - market_ivs[:, j]) ** 2))

        # Tikhonov regularisation vs previous day
        reg_loss = 0.0
        if prev_params is not None and lam > 0:
            x_prev = np.concatenate([
                prev_params.theta_curve,
                [prev_params.eta, prev_params.gamma, prev_params.rho_0, prev_params.lambda_rho],
            ])
            reg_loss = lam * float(np.sum((x - x_prev) ** 2))

        return fit_loss + arb_penalty + reg_loss

    # ── persistence ────────────────────────────────────────────────────────

    def save_parameters(self, filepath: "str | Path") -> None:
        """Save calibrated parameters to JSON."""
        if self.parameters is None:
            raise ValueError("No parameters — calibrate first.")
        with open(filepath, "w") as f:
            json.dump(self.parameters.to_dict(), f, indent=2)

    def load_parameters(self, filepath: "str | Path") -> None:
        """Load parameters from JSON."""
        with open(filepath) as f:
            data = json.load(f)
        self.parameters = ESSVIParameters.from_dict(data)
        self._build_theta_interp()


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: reduce eSSVI → standard SSVI
# ──────────────────────────────────────────────────────────────────────────────

def essvi_to_ssvi_params(p: ESSVIParameters) -> Dict[str, Any]:
    """
    Convert eSSVI parameters to the closest standard-SSVI representation.

    Uses the median ρ(θ) across the theta_curve as the constant ρ.
    Useful for downstream systems that only understand standard SSVI.
    """
    rho_values = p.rho_at_theta(p.theta_curve)
    return {
        "theta_curve": p.theta_curve.tolist(),
        "maturities": p.maturities.tolist(),
        "eta": p.eta,
        "gamma": p.gamma,
        "rho": float(np.median(rho_values)),
    }
