"""
Quote Adjustment Framework
Adjusts market quotes to remove arbitrage violations while minimizing distortion

Key Features:
- Butterfly spread adjustment (strike dimension)
- Calendar spread adjustment (time dimension)
- Quadratic programming: minimize ||adjusted - original||²
- Constraint preservation: maintains arbitrage-free conditions
- Audit trail: logs all adjustments with rationale
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.optimize import minimize, Bounds
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
class AdjustmentReport:
    """Complete quote adjustment report"""
    # Input data
    original_ivs: np.ndarray
    adjusted_ivs: np.ndarray
    strikes: np.ndarray
    maturity: float
    
    # Adjustment metrics
    max_adjustment: float  # Maximum absolute change
    rmse_adjustment: float  # RMS adjustment magnitude
    num_adjusted: int  # Number of quotes changed
    
    # Arbitrage status
    original_arbitrage_report: ArbitrageReport
    adjusted_arbitrage_report: ArbitrageReport
    
    # Metadata
    adjustment_type: str  # 'butterfly', 'calendar', 'combined'
    success: bool
    iterations: int
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'adjustment_summary': {
                'type': self.adjustment_type,
                'success': self.success,
                'num_quotes_adjusted': self.num_adjusted,
                'max_adjustment': float(self.max_adjustment),
                'rmse_adjustment': float(self.rmse_adjustment),
                'iterations': self.iterations
            },
            'original_status': {
                'arbitrage_free': self.original_arbitrage_report.is_arbitrage_free,
                'butterfly_violations': self.original_arbitrage_report.butterfly_violations,
                'calendar_violations': self.original_arbitrage_report.calendar_violations
            },
            'adjusted_status': {
                'arbitrage_free': self.adjusted_arbitrage_report.is_arbitrage_free,
                'butterfly_violations': self.adjusted_arbitrage_report.butterfly_violations,
                'calendar_violations': self.adjusted_arbitrage_report.calendar_violations
            },
            'market_data': {
                'maturity': float(self.maturity),
                'num_strikes': len(self.strikes),
                'strike_range': [float(self.strikes.min()), float(self.strikes.max())]
            },
            'timestamp': self.timestamp
        }


class QuoteAdjuster:
    """
    Adjust market quotes to remove arbitrage violations
    
    Optimization Framework:
    - Objective: minimize ||adjusted_ivs - original_ivs||²
    - Constraints: Arbitrage-free conditions (butterfly, calendar)
    - Bounds: IV ∈ [iv_min, iv_max] (typically [0.01, 5.0])
    
    Production Requirements:
    - Minimal quote distortion (preserve market fit)
    - Hard arbitrage constraints (zero tolerance for violations)
    - Complete audit trail (log all adjustments)
    - Validation: verify adjusted surface is arbitrage-free
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        max_adjustment: float = 0.10,  # 10 vol points max adjustment
        iv_bounds: Tuple[float, float] = (0.01, 5.0),
        log_adjustments: bool = True,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize quote adjuster
        
        Args:
            tolerance: Arbitrage violation tolerance
            max_adjustment: Maximum allowed IV adjustment (absolute)
            iv_bounds: Min/max bounds for IVs
            log_adjustments: Enable JSON logging
            log_dir: Directory for adjustment logs
        """
        self.tolerance = tolerance
        self.max_adjustment = max_adjustment
        self.iv_bounds = iv_bounds
        self.log_adjustments = log_adjustments
        self.log_dir = log_dir or Path('output/adjustments')
        
        self.arbitrage_checker = ArbitrageChecker(tolerance=tolerance)
        self.logger = logging.getLogger(__name__)
        
        # Create log directory
        if self.log_adjustments:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def adjust_butterfly_arbitrage(
        self,
        strikes: np.ndarray,
        market_ivs: np.ndarray,
        maturity: float,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        max_iterations: int = 1000
    ) -> AdjustmentReport:
        """
        Adjust IVs to remove butterfly arbitrage violations
        
        Optimization Problem:
            minimize: Σ(σ_adjusted - σ_original)²
            subject to: C(K₁) - 2C(K₂) + C(K₃) ≥ -tolerance for all triplets
                       σ_min ≤ σ ≤ σ_max
        
        Args:
            strikes: Strike prices (sorted ascending)
            market_ivs: Original market implied volatilities
            maturity: Time to maturity
            spot: Spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield
            max_iterations: Maximum optimizer iterations
            
        Returns:
            AdjustmentReport with original and adjusted IVs
        """
        # Validate input
        if len(strikes) != len(market_ivs):
            raise ValueError("Strikes and IVs must have same length")
        if len(strikes) < 3:
            # Not enough strikes for butterfly check
            self.logger.warning("Less than 3 strikes - no butterfly adjustment needed")
            return self._create_no_adjustment_report(
                strikes, market_ivs, maturity, spot, rate, dividend_yield, 'butterfly'
            )
        
        # Check original surface
        original_report = self.arbitrage_checker.validate_surface(
            strikes=strikes,
            maturities=np.array([maturity]),
            implied_vols=market_ivs.reshape(-1, 1),
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield
        )
        
        if original_report.is_arbitrage_free:
            self.logger.info("Surface already arbitrage-free - no adjustment needed")
            return self._create_no_adjustment_report(
                strikes, market_ivs, maturity, spot, rate, dividend_yield, 'butterfly'
            )
        
        # Define optimization objective: minimize squared deviations
        def objective(ivs: np.ndarray) -> float:
            return np.sum((ivs - market_ivs) ** 2)
        
        # Define butterfly constraints as callable functions
        # For each triplet (i-1, i, i+1), compute butterfly spread
        def butterfly_constraint(ivs: np.ndarray, idx: int) -> float:
            """
            Butterfly constraint for triplet centered at idx
            Returns: butterfly_spread + tolerance (must be ≥ 0)
            """
            # Get strikes
            K1, K2, K3 = strikes[idx-1], strikes[idx], strikes[idx+1]
            sigma1, sigma2, sigma3 = ivs[idx-1], ivs[idx], ivs[idx+1]
            
            # Compute call prices
            c1 = self._black_scholes_call(spot, K1, maturity, rate, sigma1, dividend_yield)
            c2 = self._black_scholes_call(spot, K2, maturity, rate, sigma2, dividend_yield)
            c3 = self._black_scholes_call(spot, K3, maturity, rate, sigma3, dividend_yield)
            
            # Butterfly spread (must be non-negative)
            butterfly = c1 - 2*c2 + c3
            return butterfly + self.tolerance
        
        # Create constraint list
        constraints = []
        for i in range(1, len(strikes) - 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda ivs, idx=i: butterfly_constraint(ivs, idx)
            })
        
        # Bounds: IVs within [iv_min, iv_max] and max_adjustment from original
        lower_bounds = np.maximum(
            self.iv_bounds[0],
            market_ivs - self.max_adjustment
        )
        upper_bounds = np.minimum(
            self.iv_bounds[1],
            market_ivs + self.max_adjustment
        )
        bounds = Bounds(lower_bounds, upper_bounds)
        
        # Run optimization
        self.logger.info(f"Adjusting {len(strikes)} IVs to remove butterfly arbitrage...")
        result = minimize(
            objective,
            x0=market_ivs,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations, 'ftol': 1e-9}
        )
        
        if not result.success:
            self.logger.warning(f"Optimization did not fully converge: {result.message}")
        
        adjusted_ivs = result.x
        
        # Validate adjusted surface
        adjusted_report = self.arbitrage_checker.validate_surface(
            strikes=strikes,
            maturities=np.array([maturity]),
            implied_vols=adjusted_ivs.reshape(-1, 1),
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield
        )
        
        # Compute adjustment metrics
        adjustments = adjusted_ivs - market_ivs
        max_adj = np.abs(adjustments).max()
        rmse_adj = np.sqrt(np.mean(adjustments ** 2))
        num_adj = np.sum(np.abs(adjustments) > 1e-6)
        
        # Create report
        report = AdjustmentReport(
            original_ivs=market_ivs,
            adjusted_ivs=adjusted_ivs,
            strikes=strikes,
            maturity=maturity,
            max_adjustment=max_adj,
            rmse_adjustment=rmse_adj,
            num_adjusted=int(num_adj),
            original_arbitrage_report=original_report,
            adjusted_arbitrage_report=adjusted_report,
            adjustment_type='butterfly',
            success=result.success and adjusted_report.is_arbitrage_free,
            iterations=result.nit,
            timestamp=datetime.now().isoformat()
        )
        
        # Log adjustment
        if self.log_adjustments:
            self._log_adjustment(report)
        
        return report
    
    def adjust_calendar_arbitrage(
        self,
        strike: float,
        maturities: np.ndarray,
        market_ivs: np.ndarray,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        max_iterations: int = 1000
    ) -> AdjustmentReport:
        """
        Adjust IVs to remove calendar arbitrage violations
        
        Optimization Problem:
            minimize: Σ(σ_adjusted - σ_original)²
            subject to: C(T₁) ≤ C(T₂) for all T₁ < T₂
                       σ_min ≤ σ ≤ σ_max
        
        Args:
            strike: Strike price (fixed)
            maturities: Time to maturities (sorted ascending)
            market_ivs: Original market implied volatilities
            spot: Spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield
            max_iterations: Maximum optimizer iterations
            
        Returns:
            AdjustmentReport with original and adjusted IVs
        """
        if len(maturities) != len(market_ivs):
            raise ValueError("Maturities and IVs must have same length")
        if len(maturities) < 2:
            self.logger.warning("Less than 2 maturities - no calendar adjustment needed")
            # Create minimal report
            strikes_dummy = np.array([strike])
            return self._create_no_adjustment_report(
                strikes_dummy, market_ivs[:1], maturities[0], spot, rate, dividend_yield, 'calendar'
            )
        
        # Check original surface (need to convert to 2D format)
        # For calendar check, we have single strike, multiple maturities
        # Format: strikes × maturities (1 x n_maturities)
        strikes_array = np.array([strike])
        ivs_2d = market_ivs.reshape(1, -1)  # [1 strike x n_maturities]
        
        original_report = self.arbitrage_checker.validate_surface(
            strikes=strikes_array,
            maturities=maturities,
            implied_vols=ivs_2d,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield
        )
        
        if original_report.calendar_violations == 0:
            self.logger.info("No calendar arbitrage - no adjustment needed")
            return AdjustmentReport(
                original_ivs=market_ivs,
                adjusted_ivs=market_ivs.copy(),
                strikes=strikes_array,
                maturity=maturities[0],
                max_adjustment=0.0,
                rmse_adjustment=0.0,
                num_adjusted=0,
                original_arbitrage_report=original_report,
                adjusted_arbitrage_report=original_report,
                adjustment_type='calendar',
                success=True,
                iterations=0,
                timestamp=datetime.now().isoformat()
            )
        
        # Define optimization objective
        def objective(ivs: np.ndarray) -> float:
            return np.sum((ivs - market_ivs) ** 2)
        
        # Define calendar constraints
        def calendar_constraint(ivs: np.ndarray, idx: int) -> float:
            """
            Calendar constraint for pair (idx, idx+1)
            Returns: C(T_{idx+1}) - C(T_idx) (must be ≥ 0)
            """
            T1, T2 = maturities[idx], maturities[idx+1]
            sigma1, sigma2 = ivs[idx], ivs[idx+1]
            
            c1 = self._black_scholes_call(spot, strike, T1, rate, sigma1, dividend_yield)
            c2 = self._black_scholes_call(spot, strike, T2, rate, sigma2, dividend_yield)
            
            return c2 - c1 + self.tolerance
        
        # Create constraint list
        constraints = []
        for i in range(len(maturities) - 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda ivs, idx=i: calendar_constraint(ivs, idx)
            })
        
        # Bounds
        lower_bounds = np.maximum(self.iv_bounds[0], market_ivs - self.max_adjustment)
        upper_bounds = np.minimum(self.iv_bounds[1], market_ivs + self.max_adjustment)
        bounds = Bounds(lower_bounds, upper_bounds)
        
        # Run optimization
        self.logger.info(f"Adjusting {len(maturities)} IVs to remove calendar arbitrage...")
        result = minimize(
            objective,
            x0=market_ivs,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations, 'ftol': 1e-9}
        )
        
        if not result.success:
            self.logger.warning(f"Optimization did not converge: {result.message}")
        
        adjusted_ivs = result.x
        
        # Validate adjusted surface
        ivs_2d_adjusted = adjusted_ivs.reshape(1, -1)  # [1 x n_maturities]
        adjusted_report = self.arbitrage_checker.validate_surface(
            strikes=strikes_array,
            maturities=maturities,
            implied_vols=ivs_2d_adjusted,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield
        )
        
        # Compute metrics
        adjustments = adjusted_ivs - market_ivs
        max_adj = np.abs(adjustments).max()
        rmse_adj = np.sqrt(np.mean(adjustments ** 2))
        num_adj = np.sum(np.abs(adjustments) > 1e-6)
        
        # Create report
        report = AdjustmentReport(
            original_ivs=market_ivs,
            adjusted_ivs=adjusted_ivs,
            strikes=strikes_array,
            maturity=maturities[0],  # Use first maturity as reference
            max_adjustment=max_adj,
            rmse_adjustment=rmse_adj,
            num_adjusted=int(num_adj),
            original_arbitrage_report=original_report,
            adjusted_arbitrage_report=adjusted_report,
            adjustment_type='calendar',
            success=result.success and adjusted_report.calendar_violations == 0,
            iterations=result.nit,
            timestamp=datetime.now().isoformat()
        )
        
        # Log
        if self.log_adjustments:
            self._log_adjustment(report)
        
        return report
    
    def _black_scholes_call(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """Black-Scholes call price for constraint computation"""
        from scipy.stats import norm
        
        if T <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        return call
    
    def _create_no_adjustment_report(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        maturity: float,
        spot: float,
        rate: float,
        dividend_yield: float,
        adjustment_type: str
    ) -> AdjustmentReport:
        """Create report when no adjustment is needed"""
        # Validate surface
        ivs_2d = ivs.reshape(-1, 1) if len(ivs.shape) == 1 else ivs
        report = self.arbitrage_checker.validate_surface(
            strikes=strikes,
            maturities=np.array([maturity]),
            implied_vols=ivs_2d,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield
        )
        
        return AdjustmentReport(
            original_ivs=ivs,
            adjusted_ivs=ivs.copy(),
            strikes=strikes,
            maturity=maturity,
            max_adjustment=0.0,
            rmse_adjustment=0.0,
            num_adjusted=0,
            original_arbitrage_report=report,
            adjusted_arbitrage_report=report,
            adjustment_type=adjustment_type,
            success=True,
            iterations=0,
            timestamp=datetime.now().isoformat()
        )
    
    def _log_adjustment(self, report: AdjustmentReport) -> None:
        """Log adjustment to JSON file"""
        status = 'SUCCESS' if report.success else 'PARTIAL'
        filename = f"adjustment_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        self.logger.info(f"Adjustment logged to {filepath}")
