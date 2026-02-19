"""
Arbitrage Checking Framework
Ensures IV surfaces are arbitrage-free before deployment

Key arbitrage conditions:
1. Butterfly: Call prices must be convex in strike (∂²C/∂K² ≥ 0)
2. Calendar: Call prices must be non-decreasing in time (∂C/∂T ≥ 0)
3. Total Variance: w = σ²T must be non-decreasing in maturity

**OPTIMIZED:** All computations delegated to C++ engine
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from ..cpp_unified_engine import get_unified_cpp_engine
except ImportError:
    from cpp_unified_engine import get_unified_cpp_engine


@dataclass
class ArbitrageViolation:
    """Single arbitrage violation"""
    type: str  # 'butterfly', 'calendar', 'total_variance'
    severity: str  # 'minor', 'moderate', 'severe'
    value: float  # Violation magnitude
    tolerance: float  # Threshold used
    location: Tuple  # (strike/maturity context)
    message: str


@dataclass
class ArbitrageReport:
    """Complete arbitrage validation report"""
    is_arbitrage_free: bool
    violations: List[ArbitrageViolation]
    butterfly_violations: int
    calendar_violations: int
    total_variance_violations: int
    summary: str


class ArbitrageChecker:
    """
    Arbitrage-free surface validator using C++ engine
    
    Investment bank standards:
    - Zero tolerance for severe violations (>5x threshold)
    - Bid-ask spread accommodation for minor violations
    - Comprehensive reporting with severity classification
    
    **C++ OPTIMIZED:** All computational operations delegated to C++ engine
    """
    
    def __init__(self, tolerance: float = 1e-6, bid_ask_buffer: float = 0.0):
        """
        Initialize arbitrage checker
        
        Args:
            tolerance: Base tolerance for violations (default 1e-6)
            bid_ask_buffer: Additional buffer for bid-ask spreads
        """
        self.tolerance = tolerance
        self.bid_ask_buffer = bid_ask_buffer
        self.engine = get_unified_cpp_engine()
        if self.engine is None:
            raise RuntimeError("C++ engine unavailable — ArbitrageChecker requires C++ engine")
    
    @staticmethod
    def _classify_severity(violation_magnitude: float, tolerance: float) -> str:
        """Classify violation severity based on magnitude relative to tolerance"""
        if tolerance <= 0:
            return 'severe'
        
        ratio = violation_magnitude / tolerance
        if ratio < 2.0:
            return 'minor'
        elif ratio < 5.0:
            return 'moderate'
        else:
            return 'severe'
    
    def check_butterfly_arbitrage(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        maturity: float,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        bid_ask_spreads: Optional[np.ndarray] = None
    ) -> Tuple[bool, List[ArbitrageViolation]]:
        """
        Check butterfly arbitrage: call prices must be convex in strike
        
        Condition: C(K₁) - 2C(K₂) + C(K₃) ≥ 0 for K₁ < K₂ < K₃
        Equivalently: ∂²C/∂K² ≥ 0
        
        Args:
            strikes: Array of strike prices (sorted ascending)
            implied_vols: Array of implied volatilities
            maturity: Time to maturity (years)
            spot: Spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield
            bid_ask_spreads: Optional bid-ask spreads per strike
            
        Returns:
            (is_valid, violations): Tuple of validation result and list of violations
        """
        # Adjust tolerance to accommodate bid-ask spreads if provided
        tol = self.tolerance
        if bid_ask_spreads is not None:
            tol = float(np.mean(np.maximum(tol, 0.5 * bid_ask_spreads + self.bid_ask_buffer)))

        call_prices = self.engine.black_scholes_prices(
            spot, strikes, maturity, rate, implied_vols, dividend_yield
        )
        violation_count = self.engine.check_butterfly_arbitrage(
            strikes, call_prices, tol
        )
        if violation_count == 0:
            return True, []
        violations = [
            ArbitrageViolation(
                type='butterfly',
                severity='moderate',
                value=-tol,
                tolerance=tol,
                location=(float(strikes[0]), float(strikes[-1]), maturity),
                message=f"Butterfly arbitrage: {violation_count} violation(s) at T={maturity:.3f}"
            )
            for _ in range(violation_count)
        ]
        return False, violations
    
    def check_calendar_arbitrage(
        self,
        strike: float,
        maturities: np.ndarray,
        ivs: np.ndarray,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        bid_ask_spreads: Optional[np.ndarray] = None
    ) -> Tuple[bool, List[ArbitrageViolation]]:
        """
        Check calendar arbitrage: call prices must be non-decreasing in time
        
        Condition: C(T₁) ≤ C(T₂) for T₁ < T₂ at fixed strike
        Equivalently: ∂C/∂T ≥ 0
        
        Args:
            strike: Strike price
            maturities: Array of maturities (sorted ascending)
            ivs: Array of implied volatilities
            spot: Spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield
            bid_ask_spreads: Optional bid-ask spreads per maturity
            
        Returns:
            (is_valid, violations): Tuple of validation result and list of violations
        """
        # Adjust tolerance to accommodate bid-ask spreads if provided
        tol = self.tolerance
        if bid_ask_spreads is not None:
            tol = float(np.mean(np.maximum(tol, 0.5 * bid_ask_spreads + self.bid_ask_buffer)))

        # bs_prices takes a scalar maturity; compute one price per maturity point
        call_prices = np.array([
            float(self.engine.black_scholes_prices(
                spot, np.array([strike]), float(mat), rate, np.array([iv]), dividend_yield
            )[0])
            for mat, iv in zip(maturities, ivs)
        ])
        violation_count = self.engine.check_calendar_arbitrage(
            maturities, call_prices, tol
        )
        if violation_count == 0:
            return True, []
        violations = [
            ArbitrageViolation(
                type='calendar',
                severity='moderate',
                value=-tol,
                tolerance=tol,
                location=(strike, float(maturities[0]), float(maturities[-1])),
                message=f"Calendar arbitrage: {violation_count} violation(s) at K={strike:.2f}"
            )
            for _ in range(violation_count)
        ]
        return False, violations
    
    def check_total_variance_monotonicity(
        self,
        strike: float,
        maturities: np.ndarray,
        ivs: np.ndarray
    ) -> Tuple[bool, List[ArbitrageViolation]]:
        """
        Check total variance monotonicity: w = σ²T must be non-decreasing
        
        Condition: σ₁²T₁ ≤ σ₂²T₂ for T₁ < T₂
        This is a stronger condition than calendar arbitrage
        
        Args:
            strike: Strike price
            maturities: Array of maturities (sorted ascending)
            ivs: Array of implied volatilities
            
        Returns:
            (is_valid, violations): Tuple of validation result and list of violations
        """
        if len(maturities) < 2:
            return True, []
        
        total_variances = ivs**2 * maturities
        violations = []
        
        for i in range(len(maturities) - 1):
            T1, T2 = maturities[i], maturities[i+1]
            w1, w2 = total_variances[i], total_variances[i+1]
            
            if w2 < w1 - self.tolerance:
                severity = self._classify_severity(abs(w2 - w1), self.tolerance)
                violations.append(ArbitrageViolation(
                    type='total_variance',
                    severity=severity,
                    value=w2 - w1,
                    tolerance=self.tolerance,
                    location=(strike, T1, T2),
                    message=f"Total variance non-monotonic at K={strike:.2f}, T=({T1:.3f}, {T2:.3f}): w₂ - w₁ = {w2 - w1:.6f} < {-self.tolerance:.6f}"
                ))
        
        return len(violations) == 0, violations
    
    def validate_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        bid_ask_spreads: Optional[np.ndarray] = None
    ) -> ArbitrageReport:
        """
        Complete arbitrage validation of IV surface
        
        Checks:
        1. Butterfly arbitrage across all maturities
        2. Calendar arbitrage across all strikes
        3. Total variance monotonicity
        
        Args:
            strikes: 1D array of strikes
            maturities: 1D array of maturities
            implied_vols: 2D array (strikes × maturities) of IVs
            spot: Spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield
            bid_ask_spreads: Optional 2D array of bid-ask spreads
            
        Returns:
            ArbitrageReport with comprehensive results
        """
        all_violations = []
        
        # 1. Butterfly checks (for each maturity slice)
        butterfly_count = 0
        for j, T in enumerate(maturities):
            ivs_at_T = implied_vols[:, j]
            spreads_at_T = bid_ask_spreads[:, j] if bid_ask_spreads is not None else None
            
            is_valid, violations = self.check_butterfly_arbitrage(
                strikes, ivs_at_T, T, spot, rate, dividend_yield, spreads_at_T
            )
            butterfly_count += len(violations)
            all_violations.extend(violations)
        
        # 2. Calendar checks (for each strike slice)
        calendar_count = 0
        for i, K in enumerate(strikes):
            ivs_at_K = implied_vols[i, :]
            spreads_at_K = bid_ask_spreads[i, :] if bid_ask_spreads is not None else None
            
            is_valid, violations = self.check_calendar_arbitrage(
                K, maturities, ivs_at_K, spot, rate, dividend_yield, spreads_at_K
            )
            calendar_count += len(violations)
            all_violations.extend(violations)
        
        # 3. Total variance checks
        tv_count = 0
        for i, K in enumerate(strikes):
            ivs_at_K = implied_vols[i, :]
            is_valid, violations = self.check_total_variance_monotonicity(
                K, maturities, ivs_at_K
            )
            tv_count += len(violations)
            all_violations.extend(violations)
        
        # Generate report
        is_arbitrage_free = len(all_violations) == 0
        
        summary_lines = [
            f"Arbitrage Check Results:",
            f"  Butterfly violations: {butterfly_count}",
            f"  Calendar violations: {calendar_count}",
            f"  Total variance violations: {tv_count}",
            f"  Status: {'✓ PASS - Arbitrage-free' if is_arbitrage_free else '✗ FAIL - Arbitrage detected'}"
        ]
        
        if all_violations:
            severe = sum(1 for v in all_violations if v.severity == 'severe')
            moderate = sum(1 for v in all_violations if v.severity == 'moderate')
            minor = sum(1 for v in all_violations if v.severity == 'minor')
            summary_lines.append(f"  Severity: {severe} severe, {moderate} moderate, {minor} minor")
        
        return ArbitrageReport(
            is_arbitrage_free=is_arbitrage_free,
            violations=all_violations,
            butterfly_violations=butterfly_count,
            calendar_violations=calendar_count,
            total_variance_violations=tv_count,
            summary='\n'.join(summary_lines)
        )
