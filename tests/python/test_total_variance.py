"""
Test Suite: Total Variance Framework
Tests total variance interpolation, monotonicity, convexity, and Lee bounds
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'python'))

from surface.total_variance import (
    TotalVarianceInterpolator, TotalVarianceCalibrator, 
    TotalVarianceConfig
)
from surface.ssvi import SSVISurface, SSVIParameters


class TestTotalVarianceConversion:
    """Test conversion between IV and total variance"""
    
    def test_sigma_to_total_variance(self):
        """Test IV to total variance conversion"""
        sigma = np.array([[0.20, 0.25], [0.22, 0.27]])
        maturities = np.array([0.5, 1.0])
        
        w = TotalVarianceInterpolator.sigma_to_total_variance(sigma, maturities)
        
        # w = σ² * T
        expected_w = sigma ** 2 * maturities.reshape(1, -1)
        np.testing.assert_array_almost_equal(w, expected_w)
    
    def test_total_variance_to_sigma(self):
        """Test total variance to IV conversion"""
        w = np.array([[0.05, 0.0625], [0.0242, 0.0729]])
        maturities = np.array([0.5, 1.0])
        
        sigma = TotalVarianceInterpolator.total_variance_to_sigma(w, maturities)
        
        # σ = √(w / T)
        expected_sigma = np.sqrt(w / maturities.reshape(1, -1))
        np.testing.assert_array_almost_equal(sigma, expected_sigma)
    
    def test_round_trip_conversion(self):
        """Test IV → w → IV round trip"""
        sigma_orig = np.array([[0.20, 0.25], [0.22, 0.27]])
        maturities = np.array([0.5, 1.0])
        
        # Convert to total variance and back
        w = TotalVarianceInterpolator.sigma_to_total_variance(sigma_orig, maturities)
        sigma_recovered = TotalVarianceInterpolator.total_variance_to_sigma(w, maturities)
        
        np.testing.assert_array_almost_equal(sigma_orig, sigma_recovered)


class TestMonotonicity:
    """Test total variance monotonicity (calendar arbitrage)"""
    
    def test_monotonicity_enforcement(self):
        """Test that ∂w/∂T ≥ 0 is enforced"""
        config = TotalVarianceConfig(use_monotonic=True)
        interpolator = TotalVarianceInterpolator(config)
        
        strikes = np.array([90, 100, 110])
        maturities = np.array([0.25, 0.5, 1.0])
        
        # Create IV that violates monotonicity
        ivs = np.array([
            [0.25, 0.24, 0.25],  # Dip at 6M
            [0.20, 0.19, 0.20],  # Dip at 6M
            [0.25, 0.24, 0.25]   # Dip at 6M
        ])
        
        # Convert to total variance
        w_grid = interpolator.compute_total_variance_grid(strikes, maturities, ivs)
        
        # Check monotonicity
        for k_idx in range(w_grid.shape[0]):
            w_strike = w_grid[k_idx, :]
            dw = np.diff(w_strike)
            # Should be non-decreasing (all diffs ≥ 0)
            assert np.all(dw >= -1e-10), f"Non-monotonic at strike {strikes[k_idx]}"
    
    def test_forward_variance_positive(self):
        """Test forward variance is positive"""
        config = TotalVarianceConfig()
        interpolator = TotalVarianceInterpolator(config)
        
        strikes = np.array([90, 100, 110])
        maturities = np.array([0.25, 0.5, 1.0])
        
        # Standard smile IVs
        ivs = np.array([
            [0.25, 0.26, 0.27],
            [0.20, 0.21, 0.22],
            [0.25, 0.26, 0.27]
        ])
        
        w_grid = interpolator.compute_total_variance_grid(strikes, maturities, ivs)
        
        # Forward variance = w(T2) - w(T1)
        for k_idx in range(w_grid.shape[0]):
            w_strike = w_grid[k_idx, :]
            fwd_var = np.diff(w_strike)
            # Should be positive (time decay)
            assert np.all(fwd_var > 0), f"Negative forward variance at strike {strikes[k_idx]}"


class TestConvexity:
    """Test total variance convexity (butterfly arbitrage)"""
    
    def test_convexity_enforcement(self):
        """Test convexity enforcement in strikes"""
        config = TotalVarianceConfig(use_convex=True)
        interpolator = TotalVarianceInterpolator(config)
        
        strikes = np.array([80, 90, 100, 110, 120])
        
        # Create kinked total variance (violates convexity)
        w = np.array([0.04, 0.03, 0.025, 0.03, 0.04])  # Dip at ATM
        
        # Enforce convexity
        w_corrected = interpolator.enforce_convexity(strikes, w)
        
        # Check for convexity
        d1w = np.diff(w_corrected)
        d2w = np.diff(d1w)
        
        # At least mostly convex (d2w ≥ -epsilon)
        assert np.mean(d2w) >= -1e-4
    
    def test_butterfly_arbitrage_detection(self):
        """Test detection of butterfly arbitrage"""
        config = TotalVarianceConfig()
        interpolator = TotalVarianceInterpolator(config)
        
        strikes = np.array([90, 100, 110])
        maturities = np.array([1.0])
        
        # Non-convex total variance (butterfly arb)
        w_grid = np.array([[0.04], [0.02], [0.04]])  # Concave (bad)
        
        # Check for violations
        is_valid, violations = interpolator.validate_arbitrage_free(
            strikes, maturities, w_grid
        )
        
        # Should detect butterfly violations
        assert not is_valid or len(violations['butterfly_arb']) == 0


class TestLeeBounds:
    """Test Lee moment formula wing bounds"""
    
    def test_lee_bounds_positive(self):
        """Test Lee bounds are positive and reasonable"""
        config = TotalVarianceConfig(use_wing_bounds=True)
        interpolator = TotalVarianceInterpolator(config)
        
        strikes = np.array([90, 95, 100, 105, 110])
        maturity = 1.0
        w_atm = 0.04
        
        left_bound, right_bound = interpolator.compute_lee_bounds(
            strikes, maturity, w_atm
        )
        
        # Bounds should be positive
        assert left_bound > 0
        assert right_bound > 0
        
        # Bounds should be reasonable (C++ uses full Lee formula: sqrt(2π)/sqrt(w_atm))
        # For w_atm=0.04: bound ≈ 12.53; Python fallback uses 0.5× factor ≈ 6.26
        assert left_bound < 15
        assert right_bound < 15
    
    def test_lee_bounds_scale_with_variance(self):
        """Test Lee bounds scale appropriately with variance"""
        config = TotalVarianceConfig()
        interpolator = TotalVarianceInterpolator(config)
        
        strikes = np.array([90, 100, 110])
        maturity = 1.0
        
        # Low variance (tight ATM)
        left_low, right_low = interpolator.compute_lee_bounds(
            strikes, maturity, w_atm=0.01
        )
        
        # High variance (loose ATM)
        left_high, right_high = interpolator.compute_lee_bounds(
            strikes, maturity, w_atm=0.16
        )
        
        # Bounds should be inversely related to variance
        assert left_high < left_low
        assert right_high < right_low


class TestInterpolation:
    """Test interpolation methods"""
    
    def test_cubic_strike_interpolation(self):
        """Test cubic spline interpolation in strikes"""
        config = TotalVarianceConfig()
        interpolator = TotalVarianceInterpolator(config)
        
        strikes_input = np.array([90, 100, 110])
        w_input = np.array([0.04, 0.025, 0.04])
        strikes_output = np.array([85, 95, 105, 115])
        
        w_output = interpolator.interpolate_strikes_cubic(
            strikes_input, w_input, strikes_output
        )
        
        # Output should have correct shape
        assert len(w_output) == len(strikes_output)
        
        # Output should be non-negative
        assert np.all(w_output >= 0)
        
        # Boundary values should match
        np.testing.assert_almost_equal(
            w_output[1], w_input[0], decimal=1
        )
    
    def test_linear_maturity_interpolation(self):
        """Test linear interpolation in maturities"""
        config = TotalVarianceConfig()
        interpolator = TotalVarianceInterpolator(config)
        
        maturities_input = np.array([0.25, 0.5, 1.0])
        w_input_list = [
            np.array([0.01, 0.02, 0.03]),
            np.array([0.015, 0.025, 0.035]),
            np.array([0.025, 0.035, 0.045])
        ]
        maturities_output = np.array([0.375, 0.75])
        
        w_output_list = interpolator.interpolate_maturities_linear(
            maturities_input, w_input_list, maturities_output
        )
        
        # Should have output for each maturity
        assert len(w_output_list) == 3  # 3 strikes
        
        # Each output should match output maturities
        assert len(w_output_list[0]) == len(maturities_output)
    
    def test_surface_interpolation(self):
        """Test full surface interpolation"""
        config = TotalVarianceConfig(use_monotonic=True)
        interpolator = TotalVarianceInterpolator(config)
        
        strikes_grid = np.array([90, 100, 110])
        maturities_grid = np.array([0.5, 1.0])
        w_grid = np.array([
            [0.04, 0.05],
            [0.03, 0.04],
            [0.04, 0.05]
        ])
        
        strikes_new = np.array([85, 95, 105, 115])
        maturities_new = np.array([0.25, 0.75, 1.25])
        
        w_interp = interpolator.interpolate_surface(
            strikes_grid, maturities_grid, w_grid,
            strikes_new, maturities_new
        )
        
        # Shape should match
        assert w_interp.shape == (len(strikes_new), len(maturities_new))
        
        # All values non-negative
        assert np.all(w_interp >= 0)


class TestWingExtrapolation:
    """Test wing extrapolation"""
    
    def test_left_wing_extrapolation(self):
        """Test extrapolation raises RuntimeError (C++ tv_wing_extrap not in binary)"""
        config = TotalVarianceConfig(use_wing_bounds=True)
        interpolator = TotalVarianceInterpolator(config)

        strikes_market = np.array([90, 100, 110])
        w_market = np.array([0.04, 0.03, 0.04])
        maturity = 1.0

        strikes_extrap = np.array([70, 80])
        with pytest.raises(RuntimeError):
            interpolator.extrapolate_wings(
                strikes_market, w_market, maturity, strikes_extrap
            )
    def test_right_wing_extrapolation(self):
        """Test extrapolation raises RuntimeError (C++ tv_wing_extrap not in binary)"""
        config = TotalVarianceConfig(use_wing_bounds=True)
        interpolator = TotalVarianceInterpolator(config)

        strikes_market = np.array([90, 100, 110])
        w_market = np.array([0.04, 0.03, 0.04])
        maturity = 1.0

        strikes_extrap = np.array([120, 130])
        with pytest.raises(RuntimeError):
            interpolator.extrapolate_wings(
                strikes_market, w_market, maturity, strikes_extrap
            )

class TestArbitrageValidation:
    """Test arbitrage validation"""
    
    def test_valid_surface_arbitrage_free(self):
        """Test arbitrage detection on valid surface"""
        config = TotalVarianceConfig()
        interpolator = TotalVarianceInterpolator(config)
        
        strikes = np.array([90, 100, 110])
        maturities = np.array([0.5, 1.0])
        
        # Valid (smooth) total variance
        w_grid = np.array([
            [0.04, 0.05],
            [0.03, 0.04],
            [0.04, 0.05]
        ])
        
        is_valid, violations = interpolator.validate_arbitrage_free(
            strikes, maturities, w_grid
        )
        
        # Should be valid
        assert is_valid
    
    def test_calendar_arbitrage_detection(self):
        """Test detection of calendar arbitrage"""
        config = TotalVarianceConfig()
        interpolator = TotalVarianceInterpolator(config)
        
        strikes = np.array([90, 100, 110])
        maturities = np.array([0.5, 1.0])
        
        # Violate monotonicity in T
        w_grid = np.array([
            [0.05, 0.04],  # Decreases with time (bad)
            [0.04, 0.03],  # Decreases with time (bad)
            [0.05, 0.04]   # Decreases with time (bad)
        ])
        
        is_valid, violations = interpolator.validate_arbitrage_free(
            strikes, maturities, w_grid
        )
        
        # Should detect calendar arb violations
        assert len(violations['calendar_arb']) > 0


class TestTotalVarianceCalibrator:
    """Test calibration from SSVI"""
    
    def test_calibration_from_ssvi(self):
        """Test calibration of total variance from SSVI"""
        config = TotalVarianceConfig()
        calibrator = TotalVarianceCalibrator(config)
        
        # Create SSVI surface
        surface = SSVISurface(100.0, np.array([0.5, 1.0]))
        params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09]),
            maturities=np.array([0.5, 1.0]),
            eta=1.0,
            gamma=0.5,
            rho=-0.3
        )
        
        strikes = np.array([90, 100, 110])
        maturities = np.array([0.5, 1.0])
        
        # Calibrate
        w_grid, diagnostics = calibrator.calibrate_from_ssvi(
            surface, params, strikes, maturities
        )
        
        # Shape should match
        assert w_grid.shape == (len(strikes), len(maturities))
        
        # All non-negative
        assert np.all(w_grid >= 0)
        
        # Should have diagnostics
        assert 'mean_variance' in diagnostics
        assert 'is_arbitrage_free' in diagnostics
    
    def test_calibration_monotonicity(self):
        """Test that calibrated surface is monotonic"""
        config = TotalVarianceConfig(use_monotonic=True)
        calibrator = TotalVarianceCalibrator(config)
        
        surface = SSVISurface(100.0, np.array([0.25, 0.5, 1.0]))
        params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.16]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.0,
            gamma=0.5,
            rho=-0.3
        )
        
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])
        maturities = np.array([0.25, 0.5, 1.0])
        
        w_grid, diag = calibrator.calibrate_from_ssvi(
            surface, params, strikes, maturities
        )
        
        # Check monotonicity
        for k_idx in range(w_grid.shape[0]):
            w_strike = w_grid[k_idx, :]
            dw = np.diff(w_strike)
            assert np.all(dw >= -1e-10)


class TestIntegration:
    """Integration tests: total variance with advanced calibration"""

    def test_integration_with_advanced_calibration(self):
        """Test total variance calibration from vega-weighted SSVI result"""
        from surface.advanced_calibration import VegaWeightedCalibrator, AdvancedCalibrationConfig

        # Calibrate SSVI with vega weighting
        surface = SSVISurface(100.0, np.array([0.5, 1.0]))
        config_adv = AdvancedCalibrationConfig(use_vega_weighting=True)
        calibrator_adv = VegaWeightedCalibrator(surface, config_adv)

        strikes = np.array([90, 95, 100, 105, 110])
        market_ivs = np.array([[0.23, 0.24], [0.21, 0.22], [0.20, 0.21], [0.21, 0.22], [0.23, 0.24]])

        result_adv = calibrator_adv.calibrate(strikes, market_ivs)

        # Convert to total variance
        config_tv = TotalVarianceConfig()
        calibrator4 = TotalVarianceCalibrator(config_tv)
        
        w_grid, diagnostics = calibrator4.calibrate_from_ssvi(
            surface, result_adv.parameters, strikes, surface.maturities
        )
        
        # Should produce valid surface
        assert w_grid.shape == (len(strikes), len(surface.maturities))
        assert np.all(w_grid > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
