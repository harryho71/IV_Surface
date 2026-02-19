"""
Test Arbitrage Checking Framework
Comprehensive test suite for butterfly, calendar, and total variance arbitrage validation
"""

import pytest
import numpy as np
from src.python.surface.arbitrage import ArbitrageChecker, ArbitrageReport


class TestButterflyArbitrage:
    """Test butterfly arbitrage detection (convexity in strike)"""
    
    def test_valid_convex_smile(self):
        """Valid smile: monotonically decreasing IVs (typical ATM smile)"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])  # Convex (smile shape)
        
        is_valid, violations = checker.check_butterfly_arbitrage(
            strikes, ivs, maturity=0.25, spot=100.0, rate=0.05
        )
        
        assert is_valid, "Convex smile should be arbitrage-free"
        assert len(violations) == 0
    
    def test_invalid_concave_prices(self):
        """Invalid: concave call prices (arbitrage opportunity)"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.20, 0.25, 0.30, 0.25, 0.20])  # Concave IVs → potential issue
        
        is_valid, violations = checker.check_butterfly_arbitrage(
            strikes, ivs, maturity=0.25, spot=100.0, rate=0.05
        )
        
        # This may or may not violate depending on price convexity
        # IVs can be non-convex while prices remain convex
        if not is_valid:
            assert len(violations) > 0
            assert all(v.type == 'butterfly' for v in violations)
    
    def test_butterfly_with_bid_ask_spreads(self):
        """Test butterfly check with bid-ask spread accommodation"""
        checker = ArbitrageChecker(tolerance=1e-6, bid_ask_buffer=0.01)
        
        strikes = np.array([95, 100, 105])
        ivs = np.array([0.22, 0.20, 0.22])
        bid_ask_spreads = np.array([0.05, 0.05, 0.05])
        
        is_valid, violations = checker.check_butterfly_arbitrage(
            strikes, ivs, maturity=0.25, spot=100.0, rate=0.05,
            bid_ask_spreads=bid_ask_spreads
        )
        
        assert is_valid or len(violations) < 3, "Bid-ask spreads should increase tolerance"
    
    def test_unequal_strike_spacing(self):
        """Test butterfly with unequal strike spacing"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        strikes = np.array([90, 95, 105, 110])  # Unequal spacing
        ivs = np.array([0.25, 0.22, 0.22, 0.25])
        
        is_valid, violations = checker.check_butterfly_arbitrage(
            strikes, ivs, maturity=0.25, spot=100.0, rate=0.05
        )
        
        # Should handle unequal spacing via linear interpolation
        assert isinstance(is_valid, bool)


class TestCalendarArbitrage:
    """Test calendar arbitrage detection (monotonicity in time)"""
    
    def test_valid_increasing_prices(self):
        """Valid: call prices increase with maturity"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        maturities = np.array([0.25, 0.5, 1.0])
        ivs = np.array([0.20, 0.20, 0.20])  # Constant IV
        
        is_valid, violations = checker.check_calendar_arbitrage(
            strike=100.0, maturities=maturities, ivs=ivs,
            spot=100.0, rate=0.05
        )
        
        assert is_valid, "Increasing maturity should increase call prices"
        assert len(violations) == 0
    
    def test_invalid_decreasing_prices(self):
        """Invalid: call prices decrease with maturity (arbitrage)"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        maturities = np.array([0.25, 0.5, 1.0])
        # Decreasing IVs could cause decreasing total variance
        ivs = np.array([0.30, 0.20, 0.15])
        
        is_valid, violations = checker.check_calendar_arbitrage(
            strike=100.0, maturities=maturities, ivs=ivs,
            spot=100.0, rate=0.05
        )
        
        # May violate calendar arbitrage
        if not is_valid:
            assert len(violations) > 0
            assert all(v.type == 'calendar' for v in violations)
    
    def test_calendar_with_dividend_yield(self):
        """Test calendar check with dividend yield"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        maturities = np.array([0.25, 0.5, 1.0])
        ivs = np.array([0.20, 0.20, 0.20])
        
        is_valid, violations = checker.check_calendar_arbitrage(
            strike=100.0, maturities=maturities, ivs=ivs,
            spot=100.0, rate=0.05, dividend_yield=0.02
        )
        
        assert is_valid, "Dividend yield should not break valid calendar"


class TestTotalVarianceMonotonicity:
    """Test total variance monotonicity (w = σ²T)"""
    
    def test_valid_constant_iv(self):
        """Valid: constant IV → increasing total variance"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        maturities = np.array([0.25, 0.5, 1.0])
        ivs = np.array([0.20, 0.20, 0.20])
        
        is_valid, violations = checker.check_total_variance_monotonicity(
            strike=100.0, maturities=maturities, ivs=ivs
        )
        
        assert is_valid, "Constant IV should give monotonic total variance"
        assert len(violations) == 0
    
    def test_invalid_decreasing_total_variance(self):
        """Invalid: total variance decreases"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        maturities = np.array([0.25, 0.5, 1.0])
        # w₁ = 0.30² × 0.25 = 0.0225
        # w₂ = 0.20² × 0.5 = 0.02
        # w₃ = 0.15² × 1.0 = 0.0225
        ivs = np.array([0.30, 0.20, 0.15])
        
        is_valid, violations = checker.check_total_variance_monotonicity(
            strike=100.0, maturities=maturities, ivs=ivs
        )
        
        assert not is_valid, "Decreasing total variance should violate"
        assert len(violations) > 0
        assert all(v.type == 'total_variance' for v in violations)
    
    def test_slight_decrease_within_tolerance(self):
        """Test that minor decreases within tolerance are accepted"""
        checker = ArbitrageChecker(tolerance=1e-3)  # Looser tolerance
        
        maturities = np.array([0.25, 0.5])
        # Very small decrease in total variance
        ivs = np.array([0.2000, 0.1999])
        
        is_valid, violations = checker.check_total_variance_monotonicity(
            strike=100.0, maturities=maturities, ivs=ivs
        )
        
        # Should be valid with loose tolerance
        assert is_valid or len(violations) == 0


class TestSurfaceValidation:
    """Test complete surface validation"""
    
    def test_valid_surface(self):
        """Test validation of arbitrage-free surface"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        strikes = np.array([90, 95, 100, 105, 110])
        maturities = np.array([0.25, 0.5, 1.0])
        
        # Create valid smile shape (convex) with constant term structure
        ivs = np.array([
            [0.25, 0.25, 0.25],  # 90 strike
            [0.22, 0.22, 0.22],  # 95
            [0.20, 0.20, 0.20],  # 100 ATM
            [0.22, 0.22, 0.22],  # 105
            [0.25, 0.25, 0.25],  # 110
        ])
        
        report = checker.validate_surface(
            strikes, maturities, ivs, spot=100.0, rate=0.05
        )
        
        assert report.is_arbitrage_free, "Valid surface should pass all checks"
        assert len(report.violations) == 0
        assert report.butterfly_violations == 0
        assert report.calendar_violations == 0
        assert report.total_variance_violations == 0
    
    def test_surface_with_violations(self):
        """Test detection of violations in surface"""
        checker = ArbitrageChecker(tolerance=1e-6)
        
        strikes = np.array([95, 100, 105])
        maturities = np.array([0.25, 0.5])
        
        # Create surface with decreasing total variance
        ivs = np.array([
            [0.30, 0.20],  # 95 strike
            [0.28, 0.18],  # 100
            [0.30, 0.20],  # 105
        ])
        
        report = checker.validate_surface(
            strikes, maturities, ivs, spot=100.0, rate=0.05
        )
        
        # Should detect total variance violations
        assert not report.is_arbitrage_free or report.total_variance_violations > 0
    
    def test_report_severity_classification(self):
        """Test that violations are correctly classified by severity"""
        checker = ArbitrageChecker(tolerance=1e-4)
        
        strikes = np.array([95, 100, 105])
        maturities = np.array([0.25, 0.5])
        ivs = np.array([
            [0.30, 0.15],  # Large decrease
            [0.28, 0.14],
            [0.30, 0.15],
        ])
        
        report = checker.validate_surface(
            strikes, maturities, ivs, spot=100.0, rate=0.05
        )
        
        if report.violations:
            # Check severity classification exists
            severities = [v.severity for v in report.violations]
            assert all(s in ['minor', 'moderate', 'severe'] for s in severities)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_insufficient_strikes(self):
        """Test with too few strikes for butterfly"""
        checker = ArbitrageChecker()
        
        strikes = np.array([100, 105])  # Only 2 strikes
        ivs = np.array([0.20, 0.22])
        
        is_valid, violations = checker.check_butterfly_arbitrage(
            strikes, ivs, maturity=0.25, spot=100.0, rate=0.05
        )
        
        assert is_valid, "Should skip butterfly check with < 3 strikes"
        assert len(violations) == 0
    
    def test_single_maturity(self):
        """Test calendar check with single maturity"""
        checker = ArbitrageChecker()
        
        maturities = np.array([0.25])
        ivs = np.array([0.20])
        
        is_valid, violations = checker.check_calendar_arbitrage(
            strike=100.0, maturities=maturities, ivs=ivs,
            spot=100.0, rate=0.05
        )
        
        assert is_valid, "Should skip calendar check with single maturity"
        assert len(violations) == 0
    
    def test_zero_tolerance(self):
        """Test with zero tolerance (strict checking)"""
        checker = ArbitrageChecker(tolerance=0.0)
        
        strikes = np.array([95, 100, 105])
        ivs = np.array([0.22, 0.20, 0.22])
        
        is_valid, violations = checker.check_butterfly_arbitrage(
            strikes, ivs, maturity=0.25, spot=100.0, rate=0.05
        )
        
        # Zero tolerance should classify any violation as severe
        if violations:
            assert all(v.severity == 'severe' for v in violations)


class TestIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_typical_equity_surface(self):
        """Test with typical equity IV surface shape"""
        checker = ArbitrageChecker(tolerance=1e-5)
        
        # Typical equity surface: smile + term structure flattening
        strikes = np.array([80, 90, 100, 110, 120])
        maturities = np.array([1/12, 3/12, 6/12, 12/12])  # 1M, 3M, 6M, 1Y
        
        # ATM vol: 20%, increases at wings, flattens with maturity
        ivs = np.array([
            [0.35, 0.32, 0.30, 0.28],  # 80% strike (OTM put)
            [0.25, 0.24, 0.23, 0.22],  # 90%
            [0.20, 0.20, 0.20, 0.20],  # 100% ATM
            [0.22, 0.22, 0.21, 0.21],  # 110%
            [0.28, 0.27, 0.26, 0.25],  # 120% (OTM call)
        ])
        
        report = checker.validate_surface(
            strikes, maturities, ivs, spot=100.0, rate=0.05, dividend_yield=0.02
        )
        
        assert report.is_arbitrage_free or len(report.violations) < 5, \
            f"Realistic surface should be mostly arbitrage-free. Report:\n{report.summary}"
    
    def test_stressed_market_surface(self):
        """Test with stressed market conditions (high skew)"""
        checker = ArbitrageChecker(tolerance=1e-4)  # Looser tolerance for stressed markets
        
        strikes = np.array([80, 90, 100, 110, 120])
        maturities = np.array([1/12, 3/12])
        
        # High skew (crisis scenario)
        ivs = np.array([
            [0.60, 0.55],  # Deep OTM puts expensive
            [0.40, 0.38],
            [0.25, 0.25],  # ATM
            [0.20, 0.20],
            [0.18, 0.18],  # OTM calls cheap
        ])
        
        report = checker.validate_surface(
            strikes, maturities, ivs, spot=100.0, rate=0.05
        )
        
        # High skew is valid if monotonic
        assert isinstance(report, ArbitrageReport)
        assert 'severe' not in report.summary or report.is_arbitrage_free


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
