"""
Unified C++ Engine Bridge

All computational-intensive operations delegated to C++ engine:
- Arbitrage checking (butterfly, calendar, total variance)
- SSVI evaluation and calibration
- Advanced calibration with regularization
- Total variance framework

This module provides unified access to all C++ computational engines via subprocess interface.
"""

import subprocess
import os
from typing import List, Tuple, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class UnifiedCppEngine:
    """
    Unified interface to all C++ computational engines
    Consolidates all subprocess communication
    """
    
    def __init__(self, exe_path: Optional[str] = None):
        """
        Initialize unified C++ engine
        
        Args:
            exe_path: Path to sabr_cli executable (auto-detected if None)
        """
        if exe_path is None:
            # Auto-detect from project structure
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            exe_path = os.path.join(project_root, 'build', 'sabr_cli.exe')
        
        if not os.path.exists(exe_path):
            raise FileNotFoundError(f"C++ engine not found at {exe_path}. Build with: cmake --build build --target sabr_cli")
        
        self.exe_path = exe_path
        self._verify_engine()
    
    def _verify_engine(self):
        """Verify C++ engine is working"""
        try:
            # Test a simple operation
            result = subprocess.run(
                [self.exe_path, 'tv_lee_bounds', '0.01'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"C++ engine failed verification: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("C++ engine timeout - subprocess not responding")
        except Exception as e:
            raise RuntimeError(f"C++ engine verification failed: {e}")
    
    # ==================== ARBITRAGE CHECKING ====================
    
    def black_scholes_prices(
        self,
        spot: float,
        strikes: np.ndarray,
        maturity: float,
        rate: float,
        ivs: np.ndarray,
        dividend_yield: float = 0.0
    ) -> np.ndarray:
        """
        Compute Black-Scholes call prices
        
        Args:
            spot: Spot price
            strikes: Array of strike prices
            maturity: Time to maturity (years)
            rate: Risk-free rate
            ivs: Array of implied volatilities
            dividend_yield: Dividend yield
            
        Returns:
            Array of call prices
        """
        if len(strikes) != len(ivs):
            raise ValueError("strikes and ivs must have same length")
        
        strikes_csv = ','.join(f"{k:.10e}" for k in strikes)
        ivs_csv = ','.join(f"{v:.10e}" for v in ivs)
        
        cmd = [
            self.exe_path, 'bs_prices',
            str(spot), strikes_csv, str(maturity), str(rate), ivs_csv, str(dividend_yield)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return np.array([float(x) for x in result.stdout.strip().split(',')])
    
    def check_butterfly_arbitrage(
        self,
        strikes: np.ndarray,
        call_prices: np.ndarray,
        tolerance: float = 1e-6
    ) -> int:
        """
        Check butterfly arbitrage violations
        
        Args:
            strikes: Strike prices (sorted)
            call_prices: Call prices
            tolerance: Violation tolerance
            
        Returns:
            Number of violations
        """
        if len(strikes) < 3:
            return 0
        
        strikes_csv = ','.join(f"{k:.10e}" for k in strikes)
        prices_csv = ','.join(f"{p:.10e}" for p in call_prices)
        
        cmd = [
            self.exe_path, 'check_butterfly',
            strikes_csv, prices_csv, str(tolerance)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    
    def check_calendar_arbitrage(
        self,
        maturities: np.ndarray,
        call_prices: np.ndarray,
        tolerance: float = 1e-6
    ) -> int:
        """
        Check calendar arbitrage violations
        
        Args:
            maturities: Time to maturity (sorted)
            call_prices: Call prices at fixed strike
            tolerance: Violation tolerance
            
        Returns:
            Number of violations
        """
        if len(maturities) < 2:
            return 0
        
        maturities_csv = ','.join(f"{t:.10e}" for t in maturities)
        prices_csv = ','.join(f"{p:.10e}" for p in call_prices)
        
        cmd = [
            self.exe_path, 'check_calendar',
            maturities_csv, prices_csv, str(tolerance)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    
    # ==================== SSVI EVALUATION ====================
    
    def sabr_evaluate(
        self,
        forward: float,
        strike: float,
        maturity: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ) -> float:
        """
        Evaluate SABR volatility
        
        Args:
            forward, strike, maturity: SABR inputs
            alpha, beta, rho, nu: SABR parameters
            
        Returns:
            Implied volatility
        """
        cmd = [
            self.exe_path, 'eval',
            str(forward), str(strike), str(maturity),
            str(alpha), str(beta), str(rho), str(nu)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    
    def sabr_calibrate(
        self,
        maturity: float,
        strikes: List[float],
        market_ivs: List[float],
        initial_params: Optional[List[float]] = None,
        forward: Optional[float] = None
    ) -> Tuple[List[float], float, int, bool]:
        """
        Calibrate SABR to market data
        
        Args:
            maturity: Time to maturity
            strikes: Strike prices
            market_ivs: Observed IVs
            initial_params: Initial guess [alpha, beta, rho, nu]
            forward: Forward price
            
        Returns:
            (parameters, rmse, iterations, converged)
        """
        if initial_params is None:
            initial_params = [0.2, 0.5, -0.3, 0.5]
        if forward is None:
            forward = np.mean(strikes)
        
        strikes_csv = ','.join(f"{k:.10e}" for k in strikes)
        ivs_csv = ','.join(f"{v:.10e}" for v in market_ivs)
        params_csv = ','.join(f"{p:.10e}" for p in initial_params)
        
        cmd = [
            self.exe_path, 'calibrate',
            str(maturity), strikes_csv, ivs_csv, params_csv, str(forward)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        values = result.stdout.strip().split(',')
        
        params = [float(values[i]) for i in range(4)]
        rmse = float(values[4])
        iterations = int(values[5])
        converged = bool(int(values[6]))
        
        return params, rmse, iterations, converged
    
    # ==================== ADVANCED CALIBRATION ====================
    
    # Advanced calibration intensive operations are pure Python optimization loops using scipy
    
    # ==================== TOTAL VARIANCE ====================
    
    def tv_sigma_to_total_variance(
        self,
        sigma_grid: np.ndarray,
        maturities: np.ndarray
    ) -> np.ndarray:
        """
        Convert IV grid to total variance
        
        Args:
            sigma_grid: IV grid (n_strikes, n_maturities)
            maturities: Maturities (n_maturities,)
            
        Returns:
            Total variance grid (n_strikes, n_maturities)
        """
        n_strikes, n_maturities = sigma_grid.shape
        
        sigma_csv = ','.join(f"{s:.10e}" for s in sigma_grid.flatten())
        maturities_csv = ','.join(f"{t:.10e}" for t in maturities)
        
        cmd = [
            self.exe_path, 'tv_sigma_to_w',
            sigma_csv, maturities_csv
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        w_flat = np.array([float(x) for x in result.stdout.strip().split(',')])
        return w_flat.reshape((n_strikes, n_maturities))
    
    def tv_total_variance_to_sigma(
        self,
        w_grid: np.ndarray,
        maturities: np.ndarray
    ) -> np.ndarray:
        """
        Convert total variance to IV grid
        
        Args:
            w_grid: Total variance grid (n_strikes, n_maturities)
            maturities: Maturities (n_maturities,)
            
        Returns:
            IV grid (n_strikes, n_maturities)
        """
        n_strikes, n_maturities = w_grid.shape
        
        w_csv = ','.join(f"{w:.10e}" for w in w_grid.flatten())
        maturities_csv = ','.join(f"{t:.10e}" for t in maturities)
        
        cmd = [
            self.exe_path, 'tv_w_to_sigma',
            w_csv, maturities_csv
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        sigma_flat = np.array([float(x) for x in result.stdout.strip().split(',')])
        return sigma_flat.reshape((n_strikes, n_maturities))
    
    def tv_enforce_monotonicity(
        self,
        w_grid: np.ndarray
    ) -> np.ndarray:
        """
        Enforce total variance monotonicity
        
        Args:
            w_grid: Total variance grid (n_strikes, n_maturities)
            
        Returns:
            Corrected grid with ∂w/∂T ≥ 0
        """
        n_strikes, n_maturities = w_grid.shape
        
        w_csv = ','.join(f"{w:.10e}" for w in w_grid.flatten())
        
        cmd = [
            self.exe_path, 'tv_enforce_monotonic',
            w_csv, str(n_strikes), str(n_maturities)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        w_flat = np.array([float(x) for x in result.stdout.strip().split(',')])
        return w_flat.reshape((n_strikes, n_maturities))
    
    def tv_validate_arbitrage_free(
        self,
        w_grid: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Validate total variance arbitrage-free
        
        Args:
            w_grid: Total variance grid (n_strikes, n_maturities)
            tolerance: Violation tolerance
            
        Returns:
            (is_valid, violations_dict)
        """
        n_strikes, n_maturities = w_grid.shape
        
        w_csv = ','.join(f"{w:.10e}" for w in w_grid.flatten())
        
        cmd = [
            self.exe_path, 'tv_validate',
            w_csv, str(n_strikes), str(n_maturities), str(tolerance)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(',')
        
        calendar_viols = int(parts[0]) if len(parts) > 0 else 0
        butterfly_viols = int(parts[1]) if len(parts) > 1 else 0
        
        is_valid = (calendar_viols == 0 and butterfly_viols == 0)
        
        return is_valid, {
            'calendar_violations': calendar_viols,
            'butterfly_violations': butterfly_viols
        }
    
    def tv_compute_lee_bounds(
        self,
        w_atm: float
    ) -> Tuple[float, float]:
        """
        Compute Lee wing bounds
        
        Args:
            w_atm: ATM total variance
            
        Returns:
            (left_bound, right_bound)
        """
        cmd = [self.exe_path, 'tv_lee_bounds', str(w_atm)]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(',')
        
        return float(parts[0]), float(parts[1])
    
    # ==================== IV SOLVER & SURFACE INTERPOLATION ====================
    
    def iv_solve(
        self,
        option_type: str,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        market_price: float,
        dividend_yield: float = 0.0
    ) -> Optional[float]:
        """
        Solve for implied volatility given option price
        
        Args:
            option_type: 'call' or 'put'
            spot: Spot price
            strike: Strike price
            maturity: Time to maturity (years)
            rate: Risk-free rate
            market_price: Observed market price
            dividend_yield: Dividend yield
            
        Returns:
            Implied volatility or None if failed
        """
        cmd = [
            self.exe_path, 'iv', option_type, str(spot), str(strike),
            str(maturity), str(rate), str(market_price), str(dividend_yield)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
            value = result.stdout.strip()
            if not value or value.lower() == 'nan':
                return None
            return float(value)
        except Exception as e:
            logger.warning(f"IV solve failed: {e}")
            return None
    
    def iv_solve_batch(
        self,
        option_types: List[str],
        spots: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        rate: float,
        market_prices: np.ndarray,
        dividend_yield: float = 0.0
    ) -> np.ndarray:
        """
        Batch solve for implied volatilities
        
        Args:
            option_types: Array of 'call' or 'put'
            spots: Array of spot prices
            strikes: Array of strikes
            maturities: Array of maturities
            rate: Risk-free rate
            market_prices: Array of market prices
            dividend_yield: Dividend yield
            
        Returns:
            Array of implied volatilities (NaN where failed)
        """
        n = len(strikes)
        out = np.full(n, np.nan, dtype=float)
        
        for i in range(n):
            iv = self.iv_solve(
                option_types[i],
                float(spots[i]),
                float(strikes[i]),
                float(maturities[i]),
                rate,
                float(market_prices[i]),
                dividend_yield
            )
            if iv is not None:
                out[i] = iv
        
        return out
    
    def interpolate_iv_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray,
        query_strikes: np.ndarray,
        query_maturities: np.ndarray,
        method: str = 'cubic_spline'
    ) -> np.ndarray:
        """
        Interpolate IV surface on query grid
        
        Args:
            strikes: Market strike prices
            maturities: Market maturities
            ivs: Market IVs (must match strikes/maturities lengths)
            query_strikes: Query strike points
            query_maturities: Query maturity points
            method: Interpolation method ('cubic_spline', 'linear')
            
        Returns:
            2D array of interpolated IVs (len(query_strikes) × len(query_maturities))
        """
        strikes_csv = ','.join(f"{k:.10e}" for k in strikes)
        maturities_csv = ','.join(f"{t:.10e}" for t in maturities)
        ivs_csv = ','.join(f"{v:.10e}" for v in ivs)
        query_strikes_csv = ','.join(f"{k:.10e}" for k in query_strikes)
        query_maturities_csv = ','.join(f"{t:.10e}" for t in query_maturities)
        
        cmd = [
            self.exe_path, 'interp_surface', method,
            strikes_csv, maturities_csv, ivs_csv,
            query_strikes_csv, query_maturities_csv
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            values = result.stdout.strip().split(',')
            if not values or values == ['']:
                return np.array([])
            
            # Reshape to (len(query_strikes), len(query_maturities))
            data = np.array([float(v) for v in values])
            return data.reshape(len(query_strikes), len(query_maturities))
        except Exception as e:
            logger.warning(f"Surface interpolation failed: {e}")
            return np.array([])
    
    def extrapolate_iv_surface(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        extrapolate_strikes: np.ndarray,
        maturity: float,
        method: str = 'lee_bounds'
    ) -> np.ndarray:
        """
        Extrapolate IV surface wings with arbitrage-free constraints
        
        Args:
            strikes: Market strikes
            ivs: Market IVs at those strikes
            extrapolate_strikes: Strikes to extrapolate to
            maturity: Maturity for which to extrapolate
            method: Extrapolation method ('lee_bounds', 'sabr', 'flat')
            
        Returns:
            Array of extrapolated IVs
        """
        strikes_csv = ','.join(f"{k:.10e}" for k in strikes)
        ivs_csv = ','.join(f"{v:.10e}" for v in ivs)
        extrap_csv = ','.join(f"{k:.10e}" for k in extrapolate_strikes)
        
        cmd = [
            self.exe_path, 'extrap_surface', method,
            strikes_csv, ivs_csv, extrap_csv, str(maturity)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
            values = result.stdout.strip().split(',')
            return np.array([float(v) for v in values])
        except Exception as e:
            logger.warning(f"Surface extrapolation failed: {e}")
            return np.full(len(extrapolate_strikes), np.nan)
    
    # ==================== TOTAL VARIANCE ADVANCED ====================
    
    def tv_wing_extrapolation(
        self,
        strikes: np.ndarray,
        w_market: np.ndarray,
        maturity: float,
        extrapolate_strikes: np.ndarray,
        method: str = 'lee_bounds'
    ) -> np.ndarray:
        """
        Extrapolate total variance wings with arbitrage-free Lee bounds
        
        Args:
            strikes: Market strikes
            w_market: Market total variance at those strikes
            maturity: Maturity (years)
            extrapolate_strikes: Strikes to extrapolate to
            method: Extrapolation method ('lee_bounds', 'sabr', 'flat')
            
        Returns:
            Array of extrapolated total variances
        """
        strikes_csv = ','.join(f"{k:.10e}" for k in strikes)
        w_csv = ','.join(f"{w:.10e}" for w in w_market)
        extrap_csv = ','.join(f"{k:.10e}" for k in extrapolate_strikes)
        
        cmd = [
            self.exe_path, 'tv_wing_extrap', method,
            strikes_csv, w_csv, extrap_csv, str(maturity)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
            values = result.stdout.strip().split(',')
            return np.array([float(v) for v in values])
        except Exception as e:
            logger.warning(f"TV wing extrapolation failed: {e}")
            return np.full(len(extrapolate_strikes), np.nan)
    
    def tv_quadratic_variation(
        self,
        strikes: np.ndarray,
        total_vars: np.ndarray,
        maturity: float
    ) -> float:
        """
        Compute realized quadratic variation from total variance
        
        Args:
            strikes: Strike prices
            total_vars: Total variance at each strike
            maturity: Maturity (years)
            
        Returns:
            Quadratic variation value
        """
        strikes_csv = ','.join(f"{k:.10e}" for k in strikes)
        w_csv = ','.join(f"{w:.10e}" for w in total_vars)
        
        cmd = [
            self.exe_path, 'tv_quad_var',
            strikes_csv, w_csv, str(maturity)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Quadratic variation computation failed: {e}")
            return np.nan
    
    def tv_enforce_no_arbitrage(
        self,
        strikes: np.ndarray,
        total_vars: np.ndarray,
        maturity: float,
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        Enforce no-arbitrage constraints on total variance
        
        Args:
            strikes: Strike prices
            total_vars: Total variance at each strike
            maturity: Maturity (years)
            tolerance: Arbitrage tolerance
            
        Returns:
            Adjusted total variance enforcing no-arbitrage
        """
        strikes_csv = ','.join(f"{k:.10e}" for k in strikes)
        w_csv = ','.join(f"{w:.10e}" for w in total_vars)
        
        cmd = [
            self.exe_path, 'tv_enforce_no_arb',
            strikes_csv, w_csv, str(maturity), str(tolerance)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
            values = result.stdout.strip().split(',')
            return np.array([float(v) for v in values])
        except Exception as e:
            logger.warning(f"No-arbitrage enforcement failed: {e}")
            return total_vars.copy()
    
    # ==================== UTILITY METHODS ====================
    
    def is_available(self) -> bool:
        """Check if C++ engine is available"""
        try:
            self._verify_engine()
            return True
        except Exception:
            return False


# Global singleton instance
_unified_engine = None


def get_unified_cpp_engine() -> Optional[UnifiedCppEngine]:
    """Get or create global unified C++ engine instance"""
    global _unified_engine
    
    if _unified_engine is None:
        try:
            _unified_engine = UnifiedCppEngine()
        except FileNotFoundError as e:
            logger.warning(f"C++ engine not available: {e}")
            return None
    
    return _unified_engine


def is_unified_engine_available() -> bool:
    """Check if C++ engine is available"""
    try:
        engine = get_unified_cpp_engine()
        return engine is not None and engine.is_available()
    except Exception:
        return False

