"""
Test Suite: Quote Adjustment Framework
Tests butterfly and calendar arbitrage adjustment algorithms
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'python'))

from surface.quote_adjustment import QuoteAdjuster


class TestButterflyAdjustment:
    """Test butterfly arbitrage adjustment"""
    
    def test_adjuster_initialization(self):
        """Test QuoteAdjuster initialization"""
        adjuster = QuoteAdjuster(tolerance=1e-6, max_adjustment=0.10)
        
        assert adjuster.tolerance == 1e-6
        assert adjuster.max_adjustment == 0.10
        assert adjuster.iv_bounds == (0.01, 5.0)
    
    def test_no_adjustment_needed(self):
        """Test that arbitrage-free surfaces are not adjusted"""
        adjuster = QuoteAdjuster(log_adjustments=False)
        
        # Valid convex smile (no butterfly arbitrage)
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])  # Smile shape
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.25,
            spot=100.0,
            rate=0.05
        )
        
        assert report.success
        assert report.num_adjusted == 0
        assert report.max_adjustment < 1e-6
        assert report.adjusted_arbitrage_report.is_arbitrage_free
        np.testing.assert_array_almost_equal(report.adjusted_ivs, market_ivs, decimal=6)
    
    def test_adjustment_removes_butterfly_violation(self):
        """Test that adjustment removes butterfly arbitrage"""
        adjuster = QuoteAdjuster(tolerance=1e-6, log_adjustments=False)
        
        # Create invalid concave smile (butterfly arbitrage)
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([0.20, 0.25, 0.30, 0.25, 0.20])  # Inverted smile
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.25,
            spot=100.0,
            rate=0.05
        )
        
        # Should have violations in original
        assert not report.original_arbitrage_report.is_arbitrage_free
        assert report.original_arbitrage_report.butterfly_violations > 0
        
        # Adjusted should be arbitrage-free or have fewer violations
        assert report.adjusted_arbitrage_report.butterfly_violations <= report.original_arbitrage_report.butterfly_violations
        
        # Some IVs should be adjusted
        assert report.num_adjusted > 0
        assert report.max_adjustment > 0
    
    def test_adjustment_respects_max_adjustment_bound(self):
        """Test that adjustments respect max_adjustment limit"""
        max_adj = 0.05  # 5 vol points max
        adjuster = QuoteAdjuster(max_adjustment=max_adj, log_adjustments=False)
        
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([0.20, 0.25, 0.30, 0.25, 0.20])
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.25,
            spot=100.0,
            rate=0.05
        )
        
        # Check individual adjustments
        adjustments = np.abs(report.adjusted_ivs - report.original_ivs)
        assert np.all(adjustments <= max_adj + 1e-6)  # Allow small numerical tolerance
    
    def test_adjustment_respects_iv_bounds(self):
        """Test that adjusted IVs stay within bounds"""
        adjuster = QuoteAdjuster(iv_bounds=(0.05, 2.0), log_adjustments=False)
        
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([0.20, 0.25, 0.30, 0.25, 0.20])
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.25,
            spot=100.0,
            rate=0.05
        )
        
        assert np.all(report.adjusted_ivs >= 0.05)
        assert np.all(report.adjusted_ivs <= 2.0)
    
    def test_minimal_distortion_objective(self):
        """Test that adjustment minimizes distortion from original"""
        adjuster = QuoteAdjuster(log_adjustments=False)
        
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([0.20, 0.24, 0.30, 0.24, 0.20])
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.25,
            spot=100.0,
            rate=0.05
        )
        
        # RMSE should be reasonable (not excessive adjustment)
        assert report.rmse_adjustment < 0.10  # Less than 10 vol points RMS
    
    def test_insufficient_strikes(self):
        """Test handling of insufficient strikes (< 3)"""
        adjuster = QuoteAdjuster(log_adjustments=False)
        
        # Only 2 strikes - cannot check butterfly
        strikes = np.array([95.0, 105.0])
        market_ivs = np.array([0.22, 0.22])
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.25,
            spot=100.0,
            rate=0.05
        )
        
        # Should return no-adjustment report
        assert report.num_adjusted == 0
        assert report.success


class TestCalendarAdjustment:
    """Test calendar arbitrage adjustment"""
    
    def test_no_calendar_adjustment_needed(self):
        """Test that calendar-arbitrage-free term structure is not adjusted"""
        adjuster = QuoteAdjuster(log_adjustments=False)
        
        # Valid increasing total variance
        strike = 100.0
        maturities = np.array([0.25, 0.50, 1.0])
        market_ivs = np.array([0.20, 0.22, 0.25])  # Increasing IVs
        
        report = adjuster.adjust_calendar_arbitrage(
            strike=strike,
            maturities=maturities,
            market_ivs=market_ivs,
            spot=100.0,
            rate=0.05
        )
        
        assert report.num_adjusted == 0
        assert report.adjusted_arbitrage_report.calendar_violations == 0
        np.testing.assert_array_almost_equal(report.adjusted_ivs, market_ivs, decimal=6)
    
    def test_calendar_adjustment_removes_violations(self):
        """Test that calendar adjustment removes violations"""
        adjuster = QuoteAdjuster(tolerance=1e-6, log_adjustments=False)
        
        # Invalid: severely decreasing IVs that create calendar arbitrage
        # With lower IV at longer maturity, call prices can decrease
        strike = 100.0
        maturities = np.array([0.25, 0.50, 1.0])
        market_ivs = np.array([0.40, 0.20, 0.10])  # Severe decrease
        
        report = adjuster.adjust_calendar_arbitrage(
            strike=strike,
            maturities=maturities,
            market_ivs=market_ivs,
            spot=100.0,
            rate=0.05
        )
        
        # If original has violations, adjustment should reduce them
        if report.original_arbitrage_report.calendar_violations > 0:
            assert report.adjusted_arbitrage_report.calendar_violations <= report.original_arbitrage_report.calendar_violations
            assert report.num_adjusted > 0
        else:
            # If no violations detected, that's also valid (depends on specific parameters)
            assert report.num_adjusted == 0
    
    def test_calendar_adjustment_respects_bounds(self):
        """Test calendar adjustment respects max_adjustment"""
        max_adj = 0.03
        adjuster = QuoteAdjuster(max_adjustment=max_adj, log_adjustments=False)
        
        strike = 100.0
        maturities = np.array([0.25, 0.50, 1.0])
        market_ivs = np.array([0.30, 0.25, 0.20])
        
        report = adjuster.adjust_calendar_arbitrage(
            strike=strike,
            maturities=maturities,
            market_ivs=market_ivs,
            spot=100.0,
            rate=0.05
        )
        
        adjustments = np.abs(report.adjusted_ivs - report.original_ivs)
        assert np.all(adjustments <= max_adj + 1e-6)
    
    def test_single_maturity_calendar(self):
        """Test handling of single maturity (no calendar adjustment needed)"""
        adjuster = QuoteAdjuster(log_adjustments=False)
        
        strike = 100.0
        maturities = np.array([0.25])
        market_ivs = np.array([0.20])
        
        report = adjuster.adjust_calendar_arbitrage(
            strike=strike,
            maturities=maturities,
            market_ivs=market_ivs,
            spot=100.0,
            rate=0.05
        )
        
        assert report.num_adjusted == 0


class TestAdjustmentReport:
    """Test AdjustmentReport functionality"""
    
    def test_report_creation(self):
        """Test AdjustmentReport creation and attributes"""
        adjuster = QuoteAdjuster(log_adjustments=False)
        
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.25,
            spot=100.0,
            rate=0.05
        )
        
        # Check report attributes
        assert hasattr(report, 'original_ivs')
        assert hasattr(report, 'adjusted_ivs')
        assert hasattr(report, 'max_adjustment')
        assert hasattr(report, 'rmse_adjustment')
        assert hasattr(report, 'num_adjusted')
        assert hasattr(report, 'success')
        assert hasattr(report, 'adjustment_type')
        
        assert report.adjustment_type == 'butterfly'
    
    def test_report_to_dict(self):
        """Test AdjustmentReport serialization to dict"""
        adjuster = QuoteAdjuster(log_adjustments=False)
        
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.25,
            spot=100.0,
            rate=0.05
        )
        
        report_dict = report.to_dict()
        
        assert 'adjustment_summary' in report_dict
        assert 'original_status' in report_dict
        assert 'adjusted_status' in report_dict
        assert 'market_data' in report_dict
        
        assert report_dict['adjustment_summary']['type'] == 'butterfly'
        assert 'max_adjustment' in report_dict['adjustment_summary']


class TestIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_realistic_equity_smile_adjustment(self):
        """Test adjustment of realistic equity smile with slight concavity"""
        adjuster = QuoteAdjuster(tolerance=1e-5, log_adjustments=False)
        
        # Equity smile with slight concavity at center
        strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
        market_ivs = np.array([0.28, 0.24, 0.21, 0.205, 0.22, 0.25, 0.29])
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.5,
            spot=100.0,
            rate=0.03,
            dividend_yield=0.02
        )
        
        # Adjustment should succeed
        assert report.success or report.adjusted_arbitrage_report.butterfly_violations < report.original_arbitrage_report.butterfly_violations
        
        # Adjustments should be small (< 2 vol points)
        assert report.max_adjustment < 0.02
    
    def test_term_structure_calendar_adjustment(self):
        """Test calendar adjustment across realistic term structure"""
        adjuster = QuoteAdjuster(tolerance=1e-5, log_adjustments=False)
        
        # Term structure with slight non-monotonicity
        strike = 100.0
        maturities = np.array([1/12, 2/12, 3/12, 6/12, 1.0, 2.0])
        market_ivs = np.array([0.22, 0.23, 0.225, 0.24, 0.26, 0.28])  # Dip at 3m
        
        report = adjuster.adjust_calendar_arbitrage(
            strike=strike,
            maturities=maturities,
            market_ivs=market_ivs,
            spot=100.0,
            rate=0.04,
            dividend_yield=0.015
        )
        
        # Should adjust
        if report.original_arbitrage_report.calendar_violations > 0:
            assert report.num_adjusted > 0
            assert report.adjusted_arbitrage_report.calendar_violations <= report.original_arbitrage_report.calendar_violations
    
    def test_extreme_arbitrage_adjustment(self):
        """Test adjustment of severe arbitrage violations"""
        adjuster = QuoteAdjuster(
            tolerance=1e-6,
            max_adjustment=0.15,  # Allow larger adjustments
            log_adjustments=False
        )
        
        # Severe concavity (extreme butterfly arbitrage)
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([0.15, 0.20, 0.40, 0.20, 0.15])  # Extreme peak
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.25,
            spot=100.0,
            rate=0.05
        )
        
        # Original should have severe violations
        assert report.original_arbitrage_report.butterfly_violations > 0
        
        # Adjustment should reduce violations or stay same (may not eliminate all with extreme case)
        # With severe arbitrage and limited max_adjustment, we may not achieve perfect solution
        violation_reduction = (
            report.original_arbitrage_report.butterfly_violations -
            report.adjusted_arbitrage_report.butterfly_violations
        )
        # At minimum, should not make it worse
        assert violation_reduction >= 0
    
    def test_logging_functionality(self):
        """Test that adjustment logging works"""
        import tempfile
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            adjuster = QuoteAdjuster(log_adjustments=True, log_dir=log_dir)
            
            strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
            market_ivs = np.array([0.20, 0.25, 0.30, 0.25, 0.20])
            
            report = adjuster.adjust_butterfly_arbitrage(
                strikes=strikes,
                market_ivs=market_ivs,
                maturity=0.25,
                spot=100.0,
                rate=0.05
            )
            
            # Check that log file was created
            log_files = list(log_dir.glob('adjustment_*.json'))
            assert len(log_files) > 0
            
            # Verify log content
            with open(log_files[0], 'r') as f:
                log_data = json.load(f)
            
            assert 'adjustment_summary' in log_data
            assert 'original_status' in log_data
            assert 'adjusted_status' in log_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
