"""
Total Variance Framework
Implements total variance interpolation with arbitrage enforcement
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

try:
    from ..cpp_unified_engine import get_unified_cpp_engine
except ImportError:
    from cpp_unified_engine import get_unified_cpp_engine

from .ssvi import SSVISurface, SSVIParameters


@dataclass
class TotalVariancePoint:
    """Total variance at a specific (K, T) point"""
    strike: float
    maturity: float
    total_variance: float
    call_price: Optional[float] = None
    butterfly_check: Optional[bool] = None


@dataclass
class TotalVarianceConfig:
    """Configuration for total variance interpolation"""
    # Interpolation settings
    use_monotonic: bool = True
    use_convex: bool = True
    use_wing_bounds: bool = True
    enforce_lee_bounds: bool = True
    
    # Regularization
    smoothness_lambda: float = 0.001
    monotonicity_penalty: float = 1.0
    convexity_penalty: float = 1.0
    
    # Bounds and constraints
    min_variance_per_year: float = 0.001
    max_variance_per_year: float = 10.0
    min_maturity: float = 1e-4
    max_maturity: float = 30.0
    
    # Performance
    interpolation_method: str = "cubic"  # cubic, linear, piecewise
    cache_interpolators: bool = True


class TotalVarianceInterpolator:
    """Interpolates IV surfaces in total variance space"""
    
    def __init__(self, config: Optional[TotalVarianceConfig] = None):
        """Initialize total variance interpolator"""
        self.config = config or TotalVarianceConfig()
        self.interpolators_cache: Dict[float, interp1d] = {}

        self.cpp_engine = get_unified_cpp_engine()
        if self.cpp_engine is None:
            raise RuntimeError("C++ engine unavailable — TotalVarianceInterpolator requires C++ engine")
        
    @staticmethod
    def sigma_to_total_variance(
        sigma: np.ndarray,
        maturities: np.ndarray
    ) -> np.ndarray:
        """Convert implied volatility to total variance

        w(K,T) = sigma^2(K,T) * T
        Maturities broadcast as columns: sigma shape is (n_strikes, n_maturities).
        """
        T = maturities.reshape(1, -1) if maturities.ndim == 1 else maturities
        return sigma ** 2 * T
    
    @staticmethod
    def total_variance_to_sigma(
        w: np.ndarray,
        maturities: np.ndarray
    ) -> np.ndarray:
        """Convert total variance back to implied volatility
        
        σ(K,T) = √(w(K,T) / T)
        """
        T = maturities.reshape(1, -1) if maturities.ndim == 1 else maturities
        return np.sqrt(np.maximum(w / T, 1e-8))
    
    def compute_total_variance_grid(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray
    ) -> np.ndarray:
        """Convert IV grid to total variance grid
        
        Args:
            strikes: Strike prices (n_strikes,)
            maturities: Maturities (n_maturities,)
            ivs: Implied volatilities (n_strikes, n_maturities)
        
        Returns:
            Total variance grid (n_strikes, n_maturities)
        """
        return self.cpp_engine.tv_sigma_to_total_variance(ivs, maturities)
    
    def enforce_monotonicity(
        self,
        strikes: np.ndarray,
        w_grid: np.ndarray
    ) -> np.ndarray:
        """Enforce ∂w/∂T ≥ 0 (total variance increases with maturity)

        Args:
            strikes: Strike prices
            w_grid: Total variance grid (n_strikes, n_maturities)

        Returns:
            Corrected total variance grid
        """
        return self.cpp_engine.tv_enforce_monotonicity(w_grid)
    
    def enforce_convexity(
        self,
        strikes: np.ndarray,
        w: np.ndarray
    ) -> np.ndarray:
        """Enforce convexity of call prices: ∂²C/∂K² ≥ 0
        
        Related to total variance via:
        C = S * N(d1) - K*e^{-rT} * N(d2)
        ∂²C/∂K² = e^{-rT} / (σ√T) * φ(d2) * (1 + ...) ≥ 0
        
        In practice, ensure smooth and non-decreasing slope
        
        Args:
            strikes: Strike prices
            w: Total variance array for single maturity
        
        Returns:
            Convexity-enforced total variance
        """
        w_corrected = w.copy()
        
        # Compute first differences (slope)
        dw = np.diff(w_corrected)
        
        # Slope should be non-decreasing (convexity check)
        d2w = np.diff(dw)
        
        # If any second derivative is too negative, correct
        violations = d2w < -1e-6
        if np.any(violations):
            # Linear interpolation between endpoints
            idx_violations = np.where(violations)[0]
            for idx in idx_violations:
                # Smooth over 3-point window
                if idx + 2 < len(w_corrected):
                    w_corrected[idx + 1] = 0.5 * (w_corrected[idx] + w_corrected[idx + 2])
        
        return w_corrected
    
    def compute_lee_bounds(
        self,
        strikes: np.ndarray,
        maturity: float,
        w_atm: float
    ) -> Tuple[float, float]:
        """Compute Lee moment bounds for wing slopes
        
        Lee (2004) formula:
        lim_{k→∞} log(C(K,T)/C(F,T)) / (k/√w) → function of higher moments
        
        For practical purposes:
        Left wing slope ≤ √(2π/w_T) * (forward call IV at 0.9F)
        Right wing slope ≤ √(2π/w_T) * (forward call IV at 1.1F)
        
        Args:
            strikes: Strike prices
            maturity: Time to maturity
            w_atm: ATM total variance
        
        Returns:
            (left_bound, right_bound) for wing slopes
        """
        left_bound, right_bound = self.cpp_engine.tv_compute_lee_bounds(w_atm)
        return abs(left_bound), abs(right_bound)
    
    def interpolate_strikes_cubic(
        self,
        strikes_input: np.ndarray,
        w_input: np.ndarray,
        strikes_output: np.ndarray
    ) -> np.ndarray:
        """Cubic spline interpolation of total variance in strike dimension
        
        Args:
            strikes_input: Input strikes
            w_input: Total variance at input strikes
            strikes_output: Output strikes for interpolation
        
        Returns:
            Interpolated total variance
        """
        try:
            # Use cubic spline with boundary conditions
            cs = CubicSpline(strikes_input, w_input, bc_type="natural")
            w_output = cs(strikes_output)
            
            # Ensure non-negative
            w_output = np.maximum(w_output, self.config.min_variance_per_year)
            
            return w_output
        except Exception as e:
            # Fallback to linear interpolation
            return np.interp(strikes_output, strikes_input, w_input)
    
    def interpolate_maturities_linear(
        self,
        maturities_input: np.ndarray,
        w_input_list: List[np.ndarray],
        maturities_output: np.ndarray
    ) -> List[np.ndarray]:
        """Linear interpolation of total variance in maturity dimension
        
        Preserves monotonicity: w(K, T) non-decreasing in T
        
        Args:
            maturities_input: Input maturities
            w_input_list: List of total variance arrays for each maturity
            maturities_output: Output maturities
        
        Returns:
            List of interpolated total variance arrays
        """
        n_strikes = len(w_input_list[0])
        w_output_list = []
        
        for strike_idx in range(n_strikes):
            # Extract total variance for this strike across maturities
            w_strike = np.array([w[strike_idx] for w in w_input_list])
            
            # Linear interpolation (preserves monotonicity)
            f = interp1d(
                maturities_input, w_strike,
                kind='linear', fill_value='extrapolate'
            )
            
            w_strike_output = f(maturities_output)
            w_strike_output = np.maximum(w_strike_output, self.config.min_variance_per_year)
            
            w_output_list.append(w_strike_output)
        
        return w_output_list
    
    def interpolate_surface(
        self,
        strikes_grid: np.ndarray,
        maturities_grid: np.ndarray,
        w_grid: np.ndarray,
        strikes_new: np.ndarray,
        maturities_new: np.ndarray
    ) -> np.ndarray:
        """Interpolate total variance surface
        
        Args:
            strikes_grid: Original strikes (n_strikes,)
            maturities_grid: Original maturities (n_maturities,)
            w_grid: Total variance grid (n_strikes, n_maturities)
            strikes_new: New strikes for interpolation (n_new_strikes,)
            maturities_new: New maturities for interpolation (n_new_maturities,)
        
        Returns:
            Interpolated total variance grid (n_new_strikes, n_new_maturities)
        """
        # Step 1: Interpolate in strike dimension for each maturity
        w_intermediate = []
        for t_idx in range(w_grid.shape[1]):
            w_t = self.interpolate_strikes_cubic(
                strikes_grid, w_grid[:, t_idx], strikes_new
            )
            w_intermediate.append(w_t)
        
        # Step 2: Interpolate in maturity dimension for each strike
        w_output_list = self.interpolate_maturities_linear(
            maturities_grid, w_intermediate, maturities_new
        )
        
        # Convert list to grid: w_output_list is already (n_strikes, n_maturities)
        w_output = np.array(w_output_list)
        
        # Apply constraints
        if self.config.use_monotonic:
            # Enforce monotonicity in maturity
            for t_idx in range(1, w_output.shape[1]):
                w_output[:, t_idx] = np.maximum(
                    w_output[:, t_idx],
                    w_output[:, t_idx - 1] * 1.001
                )
        
        if self.config.use_convex:
            # Enforce convexity in strikes for each maturity
            for t_idx in range(w_output.shape[1]):
                w_output[:, t_idx] = self.enforce_convexity(
                    strikes_new, w_output[:, t_idx]
                )
        
        return w_output
    
    def extrapolate_wings(
        self,
        strikes_market: np.ndarray,
        w_market: np.ndarray,
        maturity: float,
        strikes_extrapolate: np.ndarray
    ) -> np.ndarray:
        """Extrapolate wings using Lee moment bounds
        
        Args:
            strikes_market: Liquid strike prices
            w_market: Total variance at liquid strikes
            maturity: Time to maturity
            strikes_extrapolate: Strikes to extrapolate to
        
        Returns:
            Extrapolated total variance
        """
        w_extrap = self.cpp_engine.tv_wing_extrapolation(
            strikes_market, w_market, maturity, strikes_extrapolate
        )
        if np.any(np.isnan(w_extrap)):
            raise RuntimeError(
                "C++ wing extrapolation failed (tv_wing_extrap not available in this build)"
            )
        return np.clip(
            w_extrap,
            self.config.min_variance_per_year,
            self.config.max_variance_per_year
        )
    
    def validate_arbitrage_free(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        w_grid: np.ndarray,
        spot: float = 1.0,
        rate: float = 0.0
    ) -> Tuple[bool, Dict]:
        """Validate arbitrage-free conditions
        
        Args:
            strikes: Strike prices
            maturities: Maturities
            w_grid: Total variance grid
            spot: Spot price
            rate: Risk-free rate
        
        Returns:
            (is_valid, violations_dict)
        """
        is_valid, counts = self.cpp_engine.tv_validate_arbitrage_free(w_grid)
        violations = {
            'calendar_arb': ([{'count': counts['calendar_violations']}] * counts['calendar_violations'])
            if counts['calendar_violations'] > 0 else [],
            'butterfly_arb': ([{'count': counts['butterfly_violations']}] * counts['butterfly_violations'])
            if counts['butterfly_violations'] > 0 else [],
            'wing_violations': []
        }
        return is_valid, violations


class TotalVarianceCalibrator:
    """Calibrates total variance surface from SSVI"""
    
    def __init__(self, config: Optional[TotalVarianceConfig] = None):
        """Initialize total variance calibrator"""
        self.config = config or TotalVarianceConfig()
        self.interpolator = TotalVarianceInterpolator(config)
    
    def calibrate_from_ssvi(
        self,
        ssvi_surface: SSVISurface,
        ssvi_params: SSVIParameters,
        strikes: np.ndarray,
        maturities: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Calibrate total variance from SSVI parameters
        
        Args:
            ssvi_surface: SSVI surface object
            ssvi_params: Calibrated SSVI parameters
            strikes: Strike grid
            maturities: Maturity grid
        
        Returns:
            (w_grid, diagnostics)
        """
        # Evaluate SSVI IV surface
        ivs = ssvi_surface.evaluate_surface(strikes, maturities, ssvi_params)
        
        # Convert to total variance
        w_grid = self.interpolator.compute_total_variance_grid(
            strikes, maturities, ivs
        )
        
        # Diagnostics
        diagnostics = {
            'mean_variance': np.mean(w_grid),
            'min_variance': np.min(w_grid),
            'max_variance': np.max(w_grid),
            'calendar_arb_violations': 0,
            'butterfly_arb_violations': 0
        }
        
        # Validate arbitrage
        is_valid, violations = self.interpolator.validate_arbitrage_free(
            strikes, maturities, w_grid
        )
        diagnostics['is_arbitrage_free'] = is_valid
        diagnostics['violations'] = violations
        
        return w_grid, diagnostics


if __name__ == '__main__':
    # Example usage
    config = TotalVarianceConfig(
        use_monotonic=True,
        use_convex=True,
        use_wing_bounds=True
    )
    
    interpolator = TotalVarianceInterpolator(config)
    
    # Sample data
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])
    ivs = np.array([
        [0.25, 0.26, 0.27],
        [0.23, 0.24, 0.25],
        [0.21, 0.22, 0.23],
        [0.23, 0.24, 0.25],
        [0.25, 0.26, 0.27]
    ])
    
    # Convert to total variance
    w_grid = interpolator.compute_total_variance_grid(strikes, maturities, ivs)
    
    print("Total Variance Grid:")
    print(w_grid)
    
    # Validate arbitrage
    is_valid, violations = interpolator.validate_arbitrage_free(
        strikes, maturities, w_grid
    )
    
    print(f"\nArbitrage-Free: {is_valid}")
    print(f"Violations: {violations}")
