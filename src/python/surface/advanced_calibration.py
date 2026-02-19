"""
Advanced Calibration Quality Enhancements
Vega-weighted, bid-ask weighted, and regularized SSVI calibration

Implements:
1. Vega-weighted optimization (emphasize liquid ATM)
2. Bid-ask weighted fitting (inverse spread squared)
3. Tikhonov regularization (stability across days)
4. Term structure smoothness penalty
5. Warm-start optimization (from previous day)
6. Multi-objective calibration (fit vs smoothness vs stability)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from scipy.optimize import minimize, Bounds
from scipy.stats import norm

from .ssvi import SSVISurface, SSVIParameters, SSVICalibrationResult


@dataclass
class AdvancedCalibrationConfig:
    """Configuration for advanced calibration features"""
    use_vega_weighting: bool = True
    use_bid_ask_weighting: bool = True
    vega_weight: float = 1.0
    bid_ask_weight: float = 0.5
    tikhonov_lambda: float = 0.01
    smoothness_lambda: float = 0.001
    use_warm_start: bool = True
    multi_objective: bool = False
    fit_weight: float = 1.0
    smoothness_weight: float = 0.1
    stability_weight: float = 0.1
    max_iterations: int = 500
    tolerance: float = 1e-8


def black_scholes_vega(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Calculate Black-Scholes vega (dV/dsigma) per unit volatility
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Implied volatility
        
    Returns:
        Vega per unit IV (dollar value per 1% change in IV)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% IV change
    return max(vega, 1e-8)  # Floor at minimal vega


class VegaWeightedCalibrator:
    """Calibrator with vega-weighted objective"""
    
    def __init__(self, surface: SSVISurface, config: AdvancedCalibrationConfig):
        self.surface = surface
        self.config = config
        self.market_strikes = None
        self.market_maturities = None
        self.market_ivs = None
        self.vegas = None
        self.weights = None
    
    def compute_vega_weights(self) -> np.ndarray:
        """
        Compute vega-based weights emphasizing liquid strikes
        
        Returns:
            Normalized weights shape (n_strikes, n_maturities)
        """
        n_strikes = len(self.market_strikes)
        n_maturities = len(self.market_maturities)
        
        weights = np.zeros((n_strikes, n_maturities))
        
        for i, K in enumerate(self.market_strikes):
            for j, T in enumerate(self.market_maturities):
                vega = black_scholes_vega(
                    self.surface.forward, K, T, self.surface.rate, 
                    self.market_ivs[i, j]
                )
                weights[i, j] = vega
        
        # Normalize so weights sum to 1
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        
        return weights
    
    def calibrate(
        self,
        strikes: np.ndarray,
        market_ivs: np.ndarray,
        previous_parameters: Optional[SSVIParameters] = None,
        bid_ask_spreads: Optional[np.ndarray] = None
    ) -> SSVICalibrationResult:
        """
        Calibrate SSVI with vega and bid-ask weighting
        
        Args:
            strikes: Array of strikes
            market_ivs: Market IVs (strikes × maturities)
            previous_parameters: Previous day's parameters for warm-start
            bid_ask_spreads: Bid-ask spreads for weighting (optional)
            
        Returns:
            Calibration result with quality metrics
        """
        self.market_strikes = strikes
        self.market_maturities = self.surface.maturities
        self.market_ivs = market_ivs
        
        # Compute weights
        self.weights = np.ones_like(market_ivs)
        
        if self.config.use_vega_weighting:
            vega_weights = self.compute_vega_weights()
            self.weights = self.weights * vega_weights
        
        if self.config.use_bid_ask_weighting and bid_ask_spreads is not None:
            # Weight by inverse spread squared (liquid strikes get more weight)
            bid_ask_weights = np.zeros_like(bid_ask_spreads)
            for i in range(len(strikes)):
                for j in range(len(self.surface.maturities)):
                    spread = bid_ask_spreads[i, j]
                    if spread > 0:
                        bid_ask_weights[i, j] = 1.0 / (spread ** 2)
                    else:
                        bid_ask_weights[i, j] = 1.0
            
            # Normalize
            total = np.sum(bid_ask_weights)
            if total > 0:
                bid_ask_weights = bid_ask_weights / total
            
            self.weights = self.weights * bid_ask_weights
        
        # Normalize final weights
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights = self.weights / total_weight
        
        # Initialize parameters
        if self.config.use_warm_start and previous_parameters is not None:
            init_params = self._warm_start(previous_parameters)
        else:
            init_params = self.surface._initialize_parameters(strikes, market_ivs)
        
        # Optimization with multiple objectives
        if self.config.multi_objective:
            result = self._multi_objective_optimization(init_params, strikes, market_ivs)
        else:
            result = self._single_objective_optimization(init_params, strikes, market_ivs)
        
        return result
    
    def _warm_start(self, previous_params: SSVIParameters) -> SSVIParameters:
        """Initialize from previous day's parameters"""
        # Use previous parameters as starting point
        # Add small perturbation to explore neighborhood
        new_theta = previous_params.theta_curve.copy()
        new_eta = previous_params.eta * np.random.uniform(0.95, 1.05)
        new_gamma = previous_params.gamma * np.random.uniform(0.95, 1.05)
        new_rho = previous_params.rho * np.random.uniform(0.95, 1.05)
        
        # Clip rho to bounds
        new_rho = np.clip(new_rho, -0.999, 0.999)
        
        return SSVIParameters(
            theta_curve=new_theta,
            maturities=previous_params.maturities,
            eta=max(new_eta, 0.01),
            gamma=np.clip(new_gamma, 0.01, 0.99),
            rho=new_rho
        )
    
    def _objective_function(self, params_flat: np.ndarray) -> float:
        """Weighted IV fit objective"""
        params = self._unflatten_params(params_flat)
        
        if not self.surface.check_gatheral_jacquier_constraints(params)[0]:
            return 1e10  # Penalize constraint violations
        
        # Compute weighted MSE
        rmse = 0.0
        for i, K in enumerate(self.market_strikes):
            for j, T in enumerate(self.surface.maturities):
                iv_model = self.surface.evaluate_iv(K, T, params)
                iv_market = self.market_ivs[i, j]
                error = (iv_model - iv_market) ** 2
                rmse += self.weights[i, j] * error
        
        return np.sqrt(rmse)
    
    def _multi_objective_function(
        self, params_flat: np.ndarray, 
        previous_params: Optional[SSVIParameters] = None
    ) -> float:
        """Multi-objective: fit quality + smoothness + stability"""
        params = self._unflatten_params(params_flat)
        
        if not self.surface.check_gatheral_jacquier_constraints(params)[0]:
            return 1e10
        
        # 1. Fit quality (weighted RMSE)
        fit_error = self._objective_function(params_flat)
        
        # 2. Term structure smoothness
        smoothness = np.sum(np.diff(params.theta_curve) ** 2)
        
        # 3. Stability (if previous params available)
        stability = 0.0
        if previous_params is not None:
            stability = (
                (params.eta - previous_params.eta) ** 2 +
                (params.gamma - previous_params.gamma) ** 2 +
                (params.rho - previous_params.rho) ** 2
            )
        
        # Combine objectives
        total = (
            self.config.fit_weight * fit_error +
            self.config.smoothness_weight * smoothness +
            self.config.stability_weight * stability
        )
        
        return total
    
    def _single_objective_optimization(
        self, init_params: SSVIParameters, 
        strikes: np.ndarray, market_ivs: np.ndarray
    ) -> SSVICalibrationResult:
        """L-BFGS-B optimization for IV fit"""
        x0 = self._flatten_params(init_params)
        
        result = minimize(
            self._objective_function,
            x0,
            method='L-BFGS-B',
            options={'maxiter': self.config.max_iterations},
            bounds=self._get_bounds()
        )
        
        params = self._unflatten_params(result.x)
        is_valid, violations = self.surface.check_gatheral_jacquier_constraints(params)
        
        # Compute max error
        fitted_ivs = self.surface.evaluate_surface(strikes, self.market_maturities, params)
        errors = np.abs(fitted_ivs - self.market_ivs)
        max_error = np.max(errors)
        
        return SSVICalibrationResult(
            converged=result.success,
            parameters=params,
            rmse=result.fun,
            max_error=max_error,
            iterations=result.nit,
            arbitrage_free=is_valid,
            arbitrage_report=None
        )
    
    def _multi_objective_optimization(
        self, init_params: SSVIParameters, 
        strikes: np.ndarray, market_ivs: np.ndarray,
        previous_params: Optional[SSVIParameters] = None
    ) -> SSVICalibrationResult:
        """Multi-objective optimization"""
        x0 = self._flatten_params(init_params)
        
        def objective(x):
            return self._multi_objective_function(x, previous_params)
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': self.config.max_iterations},
            bounds=self._get_bounds()
        )
        
        params = self._unflatten_params(result.x)
        is_valid, violations = self.surface.check_gatheral_jacquier_constraints(params)
        
        # Compute max error
        fitted_ivs = self.surface.evaluate_surface(strikes, self.market_maturities, params)
        errors = np.abs(fitted_ivs - self.market_ivs)
        max_error = np.max(errors)
        
        return SSVICalibrationResult(
            converged=result.success,
            parameters=params,
            rmse=self._objective_function(result.x),
            max_error=max_error,
            iterations=result.nit,
            arbitrage_free=is_valid,
            arbitrage_report=None
        )
    
    def _flatten_params(self, params: SSVIParameters) -> np.ndarray:
        """Flatten SSVI parameters to 1D array"""
        return np.concatenate([
            params.theta_curve,
            [params.eta, params.gamma, params.rho]
        ])
    
    def _unflatten_params(self, x: np.ndarray) -> SSVIParameters:
        """Unflatten 1D array to SSVI parameters"""
        n_maturities = len(self.surface.maturities)
        theta_curve = x[:n_maturities]
        eta, gamma, rho = x[n_maturities:n_maturities+3]
        
        return SSVIParameters(
            theta_curve=theta_curve,
            maturities=self.surface.maturities,
            eta=eta,
            gamma=gamma,
            rho=rho
        )
    
    def _get_bounds(self) -> Bounds:
        """Parameter bounds for L-BFGS-B"""
        n_maturities = len(self.surface.maturities)
        
        # theta: [0.001, 1.0]
        # eta: [0.01, 10.0]
        # gamma: [0.01, 0.99]
        # rho: [-0.999, 0.999]
        
        lower = [0.001] * n_maturities + [0.01, 0.01, -0.999]
        upper = [1.0] * n_maturities + [10.0, 0.99, 0.999]
        
        return Bounds(lower, upper)


class TikhonovRegularizer:
    """Tikhonov regularization for parameter stability"""
    
    @staticmethod
    def compute_regularization(
        current_params: SSVIParameters,
        previous_params: Optional[SSVIParameters],
        lambda_tikhonov: float = 0.01
    ) -> float:
        """
        Compute Tikhonov regularization penalty
        
        Penalizes large day-over-day parameter changes
        """
        if previous_params is None:
            return 0.0
        
        penalty = (
            np.sum((current_params.theta_curve - previous_params.theta_curve) ** 2) +
            (current_params.eta - previous_params.eta) ** 2 +
            (current_params.gamma - previous_params.gamma) ** 2 +
            (current_params.rho - previous_params.rho) ** 2
        )
        
        return lambda_tikhonov * penalty


class TermStructureSmoothness:
    """Term structure smoothness enforcement"""
    
    @staticmethod
    def compute_smoothness_penalty(
        theta_curve: np.ndarray,
        maturities: np.ndarray,
        lambda_smooth: float = 0.001
    ) -> float:
        """
        Compute smoothness penalty for theta(T)
        
        Penalizes second-order differences (curvature)
        """
        if len(theta_curve) < 3:
            return 0.0
        
        # Compute second derivatives using finite differences
        dt = np.diff(maturities)
        d_theta = np.diff(theta_curve)
        d2_theta = np.diff(d_theta / dt) * dt[:-1]
        
        penalty = np.sum(d2_theta ** 2)
        return lambda_smooth * penalty


class ParallelCalibration:
    """Parallel calibration across multiple scenarios"""
    
    @staticmethod
    def calibrate_scenarios(
        surface: SSVISurface,
        market_data: np.ndarray,  # Shape: (n_scenarios, n_strikes, n_maturities)
        config: AdvancedCalibrationConfig
    ) -> Dict[int, SSVICalibrationResult]:
        """
        Calibrate multiple scenarios in parallel
        
        Args:
            surface: SSVI surface
            market_data: Market IV data for multiple scenarios
            config: Calibration configuration
            
        Returns:
            Dictionary mapping scenario index to calibration result
        """
        results = {}
        calibrator = VegaWeightedCalibrator(surface, config)
        
        for scenario_idx in range(market_data.shape[0]):
            # market_data shape: (n_scenarios, n_maturities, n_strikes)
            # scenario_ivs shape: (n_maturities, n_strikes)
            scenario_ivs = market_data[scenario_idx]
            n_strikes = scenario_ivs.shape[-1]
            strikes = np.linspace(
                surface.forward * 0.8,
                surface.forward * 1.2,
                n_strikes,
            )
            # calibrate() expects (n_strikes, n_maturities)
            result = calibrator.calibrate(strikes, scenario_ivs.T)
            results[scenario_idx] = result
        
        return results


if __name__ == '__main__':
    # Example usage
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Create example SSVI surface
    maturities = np.array([0.25, 0.5, 1.0])
    surface = SSVISurface(forward=100.0, maturities=maturities, rate=0.04)
    
    # Synthetic market data with realistic smile
    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    market_ivs = np.array([
        [0.32, 0.33, 0.34],
        [0.28, 0.29, 0.30],
        [0.24, 0.25, 0.26],
        [0.22, 0.23, 0.24],
        [0.23, 0.24, 0.25],
        [0.25, 0.26, 0.27],
        [0.28, 0.29, 0.30]
    ])
    
    # Configure advanced calibration
    config = AdvancedCalibrationConfig(
        use_vega_weighting=True,
        use_bid_ask_weighting=False,
        multi_objective=False
    )
    
    # Calibrate
    calibrator = VegaWeightedCalibrator(surface, config)
    result = calibrator.calibrate(strikes, market_ivs)
    
    print(f"Converged: {result.converged}")
    print(f"RMSE: {result.rmse:.6f}")
    print(f"Iterations: {result.iterations}")
    print(f"Parameters: {result.parameters}")
