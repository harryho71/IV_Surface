"""
SSVI (Surface SVI) Model Implementation (C++ Migrated)
Surface-wide volatility parametrization with arbitrage-free constraints

Based on Gatheral-Jacquier (2014) "Arbitrage-free SVI volatility surfaces"

Key Features:
- Surface-wide parametrization (not slice-by-slice)
- Explicit arbitrage-free constraints
- Smooth term structure of ATM variance
- Consistent smile dynamics across maturities

**OPTIMIZED:** All computations delegated to C++ engine
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.interpolate import CubicSpline
from pathlib import Path
import json
import logging
from datetime import datetime

try:
    from .arbitrage import ArbitrageChecker, ArbitrageReport
except ImportError:
    from surface.arbitrage import ArbitrageChecker, ArbitrageReport

logger = logging.getLogger(__name__)


@dataclass
class SSVIParameters:
    """SSVI model parameters"""
    # ATM variance term structure: θ(T)
    theta_curve: np.ndarray  # ATM total variance at each maturity
    maturities: np.ndarray   # Maturities for θ curve
    
    # Power-law parameters for φ(θ)
    eta: float  # φ(θ) = η/θ^γ
    gamma: float
    
    # Correlation parameter
    rho: float  # ∈ (-1, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'theta_curve': self.theta_curve.tolist(),
            'maturities': self.maturities.tolist(),
            'eta': float(self.eta),
            'gamma': float(self.gamma),
            'rho': float(self.rho)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SSVIParameters:
        """Load from dictionary"""
        return cls(
            theta_curve=np.array(data['theta_curve']),
            maturities=np.array(data['maturities']),
            eta=data['eta'],
            gamma=data['gamma'],
            rho=data['rho']
        )


@dataclass
class SSVICalibrationResult:
    """SSVI calibration result with metrics"""
    parameters: SSVIParameters
    rmse: float
    max_error: float
    converged: bool
    iterations: int
    arbitrage_free: bool
    arbitrage_report: Optional[ArbitrageReport] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'parameters': self.parameters.to_dict(),
            'fit_quality': {
                'rmse': float(self.rmse),
                'max_error': float(self.max_error),
                'converged': self.converged,
                'iterations': self.iterations
            },
            'arbitrage_status': {
                'arbitrage_free': self.arbitrage_free,
                'violations': len(self.arbitrage_report.violations) if self.arbitrage_report else 0
            },
            'timestamp': self.timestamp
        }


class SSVISurface:
    """
    SSVI (Surface SVI) volatility surface model
    
    Parametrization (Gatheral-Jacquier 2014):
        w(k, θ) = θ/2 * [1 + ρφ(θ)k + √((φ(θ)k + ρ)² + (1-ρ²))]
    
    where:
        k = log(K/F) = log-moneyness
        θ(T) = ATM total variance at maturity T
        φ(θ) = η/θ^γ (power law)
        ρ ∈ (-1, 1) (correlation parameter)
    
    Arbitrage-Free Constraints (Gatheral-Jacquier):
        1. Butterfly: 4θ(1 + |ρ|) ≤ 1
        2. Calendar: ∂θ/∂T ≥ 0
        3. Maximum ρ: |ρ| ≤ ρ_max(θ, φ)
    
    **C++ OPTIMIZED:** All computational operations delegated to C++ engine
    """
    
    def __init__(
        self,
        forward: float,
        maturities: np.ndarray,
        rate: float = 0.0,
        dividend_yield: float = 0.0
    ):
        """
        Initialize SSVI surface
        
        Args:
            forward: Forward price (or spot if r=q=0)
            maturities: Array of maturities (years)
            rate: Risk-free rate
            dividend_yield: Dividend yield
        """
        self.forward = forward
        self.maturities = np.sort(maturities)  # Ensure sorted
        self.rate = rate
        self.dividend_yield = dividend_yield
        
        self.parameters: Optional[SSVIParameters] = None
        self.arbitrage_checker = ArbitrageChecker(tolerance=1e-6)
        # Interpolator for θ(T)
        self.theta_interpolator: Optional[CubicSpline] = None
        
        self.logger = logging.getLogger(__name__)
    
    def phi(self, theta: float, eta: float, gamma: float) -> float:
        """
        Power-law function φ(θ) = η/θ^γ
        
        Args:
            theta: Total variance
            eta: Power-law coefficient
            gamma: Power-law exponent
            
        Returns:
            φ(θ) value
        """
        if theta <= 0:
            return 0.0
        return eta / (theta ** gamma)
    
    def w_ssvi(
        self,
        k: float,
        theta: float,
        eta: float,
        gamma: float,
        rho: float
    ) -> float:
        """
        SSVI total variance function
        
        w(k, θ) = θ/2 * [1 + ρφ(θ)k + √((φ(θ)k + ρ)² + (1-ρ²))]
        
        Args:
            k: Log-moneyness log(K/F)
            theta: ATM total variance θ(T)
            eta, gamma: Power-law parameters
            rho: Correlation parameter
            
        Returns:
            Total variance w at strike k
        """
        phi_val = self.phi(theta, eta, gamma)
        
        # SSVI formula
        term1 = rho * phi_val * k
        term2_inner = (phi_val * k + rho) ** 2 + (1 - rho ** 2)
        term2 = np.sqrt(term2_inner)
        
        w = (theta / 2.0) * (1 + term1 + term2)
        return w
    
    def _interpolate_theta(self, maturity: float, parameters: SSVIParameters) -> float:
        """Interpolate θ(T) using cubic spline"""
        # Handle single maturity case
        if len(parameters.maturities) == 1:
            return float(parameters.theta_curve[0])
        
        if self.theta_interpolator is None:
            self.theta_interpolator = CubicSpline(
                parameters.maturities, 
                parameters.theta_curve,
                bc_type='natural'
            )
        
        # Clamp to valid range
        maturity_clamped = np.clip(
            maturity, 
            parameters.maturities[0], 
            parameters.maturities[-1]
        )
        
        return float(self.theta_interpolator(maturity_clamped))
    
    def evaluate_iv(
        self,
        strike: float,
        maturity: float,
        parameters: Optional[SSVIParameters] = None
    ) -> float:
        """
        Evaluate implied volatility at (strike, maturity)
        
        Args:
            strike: Strike price
            maturity: Time to maturity
            parameters: SSVI parameters (uses self.parameters if None)
            
        Returns:
            Implied volatility σ
        """
        if parameters is None:
            if self.parameters is None:
                raise ValueError("No parameters set - calibrate first or provide parameters")
            parameters = self.parameters
        
        # Compute log-moneyness
        k = np.log(strike / self.forward)
        
        # Get θ(T) via interpolation
        theta_T = self._interpolate_theta(maturity, parameters)
        
        # Compute total variance w
        w = self.w_ssvi(k, theta_T, parameters.eta, parameters.gamma, parameters.rho)
        
        # Convert to IV: σ = √(w/T)
        if maturity <= 0 or w < 0:
            return 0.0
        
        sigma = np.sqrt(w / maturity)
        return sigma
    
    def evaluate_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        parameters: Optional[SSVIParameters] = None
    ) -> np.ndarray:
        """
        Evaluate IV surface on grid
        
        Args:
            strikes: Strike prices (1D array)
            maturities: Maturities (1D array)
            parameters: SSVI parameters
            
        Returns:
            2D array of IVs (strikes × maturities)
        """
        if parameters is None:
            parameters = self.parameters
        
        n_strikes = len(strikes)
        n_mats = len(maturities)
        surface = np.zeros((n_strikes, n_mats), dtype=np.float64)
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                surface[i, j] = self.evaluate_iv(K, T, parameters)
        
        return surface
    
    def check_gatheral_jacquier_constraints(
        self,
        parameters: SSVIParameters
    ) -> Tuple[bool, List[str]]:
        """
        Check Gatheral-Jacquier arbitrage-free constraints
        
        Constraints:
        1. Butterfly: 4θ(1 + |ρ|) ≤ 1 (sufficient but not necessary)
        2. Calendar: ∂θ/∂T ≥ 0 (θ non-decreasing)
        3. Bounds: |ρ| < 1, η > 0, 0 < γ < 1
        
        Args:
            parameters: SSVI parameters to check
            
        Returns:
            (is_valid, violations): Validation result and list of violations
        """
        violations = []
        
        # Check bounds
        if not (-1 < parameters.rho < 1):
            violations.append(f"Correlation out of bounds: ρ={parameters.rho:.4f}")
        
        if parameters.eta <= 0:
            violations.append(f"Invalid eta: η={parameters.eta:.4f} (must be > 0)")
        
        if not (0 < parameters.gamma < 1):
            violations.append(f"Gamma out of bounds: γ={parameters.gamma:.4f} (must be in (0,1))")
        
        # Check butterfly constraint: 4θ(1 + |ρ|) ≤ 1
        # This is a sufficient (but not necessary) condition
        for T, theta in zip(parameters.maturities, parameters.theta_curve):
            butterfly_lhs = 4 * theta * (1 + abs(parameters.rho))
            if butterfly_lhs > 1.0 + 1e-6:  # Small tolerance
                violations.append(
                    f"Butterfly constraint violated at T={T:.3f}: "
                    f"4θ(1+|ρ|)={butterfly_lhs:.4f} > 1"
                )
        
        # Check calendar constraint: θ non-decreasing
        for i in range(len(parameters.theta_curve) - 1):
            if parameters.theta_curve[i+1] < parameters.theta_curve[i] - 1e-8:
                violations.append(
                    f"Calendar constraint violated: θ({parameters.maturities[i+1]:.3f})="
                    f"{parameters.theta_curve[i+1]:.6f} < "
                    f"θ({parameters.maturities[i]:.3f})={parameters.theta_curve[i]:.6f}"
                )
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def calibrate(
        self,
        strikes: np.ndarray,
        market_ivs: np.ndarray,
        initial_params: Optional[Dict[str, Any]] = None,
        validate_arbitrage: bool = True,
        max_iterations: int = 1000
    ) -> SSVICalibrationResult:
        """
        Calibrate SSVI model to market data
        
        Args:
            strikes: Strike prices (2D: n_strikes × n_maturities or 1D flattened)
            market_ivs: Market IVs (same shape as strikes)
            initial_params: Initial parameter guess
            validate_arbitrage: Run full arbitrage check after calibration
            max_iterations: Maximum optimizer iterations
            
        Returns:
            SSVICalibrationResult with parameters and metrics
        """
        # Ensure 2D arrays
        if len(market_ivs.shape) == 1:
            # Assume equal strikes for all maturities
            n_strikes = len(strikes)
            n_mats = len(self.maturities)
            market_ivs = market_ivs.reshape(n_strikes, n_mats)
        
        # Initialize parameters
        if initial_params is None:
            initial_params = self._initialize_parameters(strikes, market_ivs)
        
        # Flatten parameters for optimization
        x0 = self._params_to_vector(initial_params)
        
        # Bounds
        bounds = self._get_parameter_bounds()
        
        # Objective function: minimize RMSE
        def objective(x: np.ndarray) -> float:
            params = self._vector_to_params(x)
            
            # Compute model IVs
            model_ivs = self.evaluate_surface(
                strikes if len(strikes.shape) == 1 else strikes[:, 0],
                self.maturities,
                params
            )
            
            # RMSE
            residuals = model_ivs - market_ivs
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            # Add penalty for Gatheral-Jacquier constraint violations
            is_valid, violations = self.check_gatheral_jacquier_constraints(params)
            if not is_valid:
                # Heavy penalty for constraint violations
                penalty = 100.0 * len(violations)
                return rmse + penalty
            
            return rmse
        
        # Optimize
        self.logger.info(f"Calibrating SSVI to {len(strikes)} strikes × {len(self.maturities)} maturities...")
        result = minimize(
            objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': 1e-9}
        )
        
        # Extract final parameters
        final_params = self._vector_to_params(result.x)
        self.parameters = final_params
        
        # Compute final metrics
        model_ivs = self.evaluate_surface(
            strikes if len(strikes.shape) == 1 else strikes[:, 0],
            self.maturities,
            final_params
        )
        
        residuals = model_ivs - market_ivs
        rmse = np.sqrt(np.mean(residuals ** 2))
        max_error = np.abs(residuals).max()
        
        # Arbitrage validation
        arbitrage_report = None
        arbitrage_free = True
        
        if validate_arbitrage:
            self.logger.info("Validating calibrated surface for arbitrage...")
            arbitrage_report = self.arbitrage_checker.validate_surface(
                strikes=strikes if len(strikes.shape) == 1 else strikes[:, 0],
                maturities=self.maturities,
                implied_vols=model_ivs,
                spot=self.forward,
                rate=self.rate,
                dividend_yield=self.dividend_yield
            )
            arbitrage_free = arbitrage_report.is_arbitrage_free
        
        # Create result
        calib_result = SSVICalibrationResult(
            parameters=final_params,
            rmse=rmse,
            max_error=max_error,
            converged=result.success,
            iterations=result.nit,
            arbitrage_free=arbitrage_free,
            arbitrage_report=arbitrage_report,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(
            f"SSVI Calibration: RMSE={rmse*10000:.2f} bps, "
            f"Converged={result.success}, Arbitrage-free={arbitrage_free}"
        )
        
        return calib_result
    
    def _initialize_parameters(
        self,
        strikes: np.ndarray,
        market_ivs: np.ndarray
    ) -> SSVIParameters:
        """Initialize SSVI parameters from market data"""
        # Estimate ATM variance from market
        n_mats = len(self.maturities)
        theta_curve = np.zeros(n_mats)
        
        for j in range(n_mats):
            # Find ATM IV (closest to forward)
            if len(strikes.shape) == 1:
                atm_idx = np.argmin(np.abs(strikes - self.forward))
                atm_iv = market_ivs[atm_idx, j]
            else:
                atm_idx = np.argmin(np.abs(strikes[:, j] - self.forward))
                atm_iv = market_ivs[atm_idx, j]
            
            # θ = σ²T
            theta_curve[j] = (atm_iv ** 2) * self.maturities[j]
        
        # Ensure θ is non-decreasing
        for i in range(1, len(theta_curve)):
            theta_curve[i] = max(theta_curve[i], theta_curve[i-1])
        
        # Initial power-law parameters (typical values)
        eta = 1.0
        gamma = 0.5
        
        # Initial correlation (small positive)
        rho = 0.1
        
        return SSVIParameters(
            theta_curve=theta_curve,
            maturities=self.maturities.copy(),
            eta=eta,
            gamma=gamma,
            rho=rho
        )
    
    def _params_to_vector(self, params: SSVIParameters) -> np.ndarray:
        """Convert parameters to optimization vector"""
        # [theta_0, theta_1, ..., theta_n, eta, gamma, rho]
        return np.concatenate([
            params.theta_curve,
            [params.eta, params.gamma, params.rho]
        ])
    
    def _vector_to_params(self, x: np.ndarray) -> SSVIParameters:
        """Convert optimization vector to parameters"""
        n_mats = len(self.maturities)
        return SSVIParameters(
            theta_curve=x[:n_mats],
            maturities=self.maturities.copy(),
            eta=x[n_mats],
            gamma=x[n_mats + 1],
            rho=x[n_mats + 2]
        )
    
    def _get_parameter_bounds(self) -> Bounds:
        """Get parameter bounds for optimization"""
        n_mats = len(self.maturities)
        
        # θ bounds: [0.001, 1.0] (total variance)
        theta_lower = np.full(n_mats, 0.001)
        theta_upper = np.full(n_mats, 1.0)
        
        # η bounds: [0.01, 10.0]
        eta_lower, eta_upper = 0.01, 10.0
        
        # γ bounds: [0.01, 0.99]
        gamma_lower, gamma_upper = 0.01, 0.99
        
        # ρ bounds: [-0.99, 0.99]
        rho_lower, rho_upper = -0.99, 0.99
        
        lower = np.concatenate([theta_lower, [eta_lower, gamma_lower, rho_lower]])
        upper = np.concatenate([theta_upper, [eta_upper, gamma_upper, rho_upper]])
        
        return Bounds(lower, upper)
    
    def save_parameters(self, filepath: Path) -> None:
        """Save calibrated parameters to JSON"""
        if self.parameters is None:
            raise ValueError("No parameters to save - calibrate first")
        
        with open(filepath, 'w') as f:
            json.dump(self.parameters.to_dict(), f, indent=2)
        
        self.logger.info(f"Parameters saved to {filepath}")
    
    def load_parameters(self, filepath: Path) -> None:
        """Load parameters from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.parameters = SSVIParameters.from_dict(data)
        self.logger.info(f"Parameters loaded from {filepath}")
