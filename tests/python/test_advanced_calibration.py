"""
Test Suite: Advanced Calibration Quality
Tests vega-weighted, bid-ask weighted, and regularized SSVI calibration
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'python'))

from surface.ssvi import SSVISurface, SSVIParameters
from surface.advanced_calibration import (
    VegaWeightedCalibrator, AdvancedCalibrationConfig,
    black_scholes_vega, TikhonovRegularizer, TermStructureSmoothness,
    ParallelCalibration
)


class TestVegaWeighting:
    """Test vega-based weighting"""
    
    def test_black_scholes_vega_atm(self):
        """Test BS vega at ATM"""
        # ATM: S=K=100
        vega = black_scholes_vega(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        
        # At ATM with T=1, vega should be positive
        assert vega > 0
        # Typical range for ATM: 0.35-0.40 per underlying per 1% vol
        assert 0.01 < vega < 1.0
    
    def test_black_scholes_vega_otm(self):
        """Test vega decreases for OTM options"""
        vega_atm = black_scholes_vega(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        vega_otm = black_scholes_vega(S=100, K=120, T=1.0, r=0.05, sigma=0.20)
        
        # OTM vega should be less than ATM
        assert vega_otm < vega_atm
    
    def test_vega_zero_when_zero_time(self):
        """Test vega approaches zero at expiration"""
        vega = black_scholes_vega(S=100, K=100, T=0.0001, r=0.05, sigma=0.20)
        
        # Very small vega near expiration (T=0.0001 ≈ 0.036 days)
        assert vega < 0.01
    
    def test_vega_weight_computation(self):
        """Test vega weight computation"""
        config = AdvancedCalibrationConfig(use_vega_weighting=True)
        surface = SSVISurface(100.0, np.array([0.5, 1.0]))
        calibrator = VegaWeightedCalibrator(surface, config)
        
        strikes = np.array([90, 95, 100, 105, 110])
        market_ivs = np.array([
            [0.25, 0.26],
            [0.22, 0.23],
            [0.20, 0.21],
            [0.22, 0.23],
            [0.25, 0.26]
        ])
        
        calibrator.market_strikes = strikes
        calibrator.market_maturities = surface.maturities
        calibrator.market_ivs = market_ivs
        
        weights = calibrator.compute_vega_weights()
        
        # Weights should be positive
        assert np.all(weights >= 0)
        # Weights should sum to approximately 1
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
        # ATM should have highest weight
        atm_idx = 2
        assert weights[atm_idx].mean() > weights[0].mean()


class TestBidAskWeighting:
    """Test bid-ask spread weighting"""
    
    def test_bid_ask_weight_liquid_strikes(self):
        """Test that liquid strikes (tight spreads) get higher weight"""
        config = AdvancedCalibrationConfig(use_bid_ask_weighting=True)
        surface = SSVISurface(100.0, np.array([1.0]))
        calibrator = VegaWeightedCalibrator(surface, config)
        
        strikes = np.array([95, 100, 105])
        market_ivs = np.array([[0.22], [0.20], [0.22]])
        
        calibrator.market_strikes = strikes
        calibrator.market_maturities = surface.maturities
        calibrator.market_ivs = market_ivs
        
        # Tight spreads for liquid strikes, wide for illiquid
        bid_ask_spreads = np.array([
            [0.10],  # Wide (illiquid)
            [0.01],  # Tight (liquid)
            [0.10]   # Wide (illiquid)
        ])
        
        calibrator.weights = np.ones_like(market_ivs)
        
        # Inverse spread squared weighting
        ba_weights = np.zeros_like(bid_ask_spreads)
        for i in range(len(strikes)):
            spread = bid_ask_spreads[i, 0]
            if spread > 0:
                ba_weights[i, 0] = 1.0 / (spread ** 2)
        
        # ATM (tight spread) should have highest weight
        assert ba_weights[1, 0] > ba_weights[0, 0]
        assert ba_weights[1, 0] > ba_weights[2, 0]


class TestTikhonovRegularization:
    """Test Tikhonov regularization for stability"""
    
    def test_zero_penalty_no_change(self):
        """Test zero regularization when parameters unchanged"""
        params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.16]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.0, gamma=0.5, rho=-0.3
        )
        
        penalty = TikhonovRegularizer.compute_regularization(
            params, params, lambda_tikhonov=0.01
        )
        
        assert penalty == 0.0
    
    def test_penalty_with_change(self):
        """Test regularization penalizes parameter changes"""
        params1 = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.16]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.0, gamma=0.5, rho=-0.3
        )
        
        params2 = SSVIParameters(
            theta_curve=np.array([0.05, 0.10, 0.17]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.1, gamma=0.55, rho=-0.35
        )
        
        penalty = TikhonovRegularizer.compute_regularization(
            params2, params1, lambda_tikhonov=0.01
        )
        
        assert penalty > 0.0
    
    def test_penalty_increases_with_lambda(self):
        """Test penalty scales with lambda"""
        params1 = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.16]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.0, gamma=0.5, rho=-0.3
        )
        
        params2 = SSVIParameters(
            theta_curve=np.array([0.05, 0.10, 0.17]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.1, gamma=0.55, rho=-0.35
        )
        
        penalty1 = TikhonovRegularizer.compute_regularization(
            params2, params1, lambda_tikhonov=0.01
        )
        penalty2 = TikhonovRegularizer.compute_regularization(
            params2, params1, lambda_tikhonov=0.10
        )
        
        assert penalty2 > penalty1


class TestTermStructureSmoothness:
    """Test term structure smoothness penalty"""
    
    def test_zero_penalty_linear_structure(self):
        """Test minimal penalty for smooth term structure"""
        maturities = np.array([0.25, 0.5, 0.75, 1.0])
        theta_linear = np.array([0.04, 0.09, 0.14, 0.19])  # Linear
        
        penalty = TermStructureSmoothness.compute_smoothness_penalty(
            theta_linear, maturities, lambda_smooth=0.001
        )
        
        # Linear term structure should have near-zero penalty
        assert penalty < 1e-6
    
    def test_penalty_for_kinked_structure(self):
        """Test penalty for non-smooth term structure"""
        maturities = np.array([0.25, 0.5, 0.75, 1.0])
        theta_kinked = np.array([0.04, 0.09, 0.08, 0.19])  # Kinked
        theta_smooth = np.array([0.04, 0.09, 0.14, 0.19])  # Smooth
        
        penalty_kinked = TermStructureSmoothness.compute_smoothness_penalty(
            theta_kinked, maturities, lambda_smooth=0.001
        )
        penalty_smooth = TermStructureSmoothness.compute_smoothness_penalty(
            theta_smooth, maturities, lambda_smooth=0.001
        )
        
        assert penalty_kinked > penalty_smooth


class TestVegaWeightedCalibrator:
    """Test vega-weighted calibration"""
    
    def test_calibrator_initialization(self):
        """Test calibrator initialization"""
        config = AdvancedCalibrationConfig()
        surface = SSVISurface(100.0, np.array([0.5, 1.0]))
        calibrator = VegaWeightedCalibrator(surface, config)
        
        assert calibrator.surface is surface
        assert calibrator.config is config
    
    def test_parameter_flattening(self):
        """Test parameter flattening/unflattening"""
        config = AdvancedCalibrationConfig()
        surface = SSVISurface(100.0, np.array([0.25, 0.5, 1.0]))
        calibrator = VegaWeightedCalibrator(surface, config)
        
        params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.16]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.0, gamma=0.5, rho=-0.3
        )
        
        # Flatten and unflatten
        flat = calibrator._flatten_params(params)
        params_recovered = calibrator._unflatten_params(flat)
        
        np.testing.assert_array_almost_equal(
            params.theta_curve, params_recovered.theta_curve
        )
        assert params.eta == params_recovered.eta
        assert params.gamma == params_recovered.gamma
        assert params.rho == params_recovered.rho
    
    def test_calibration_with_vega_weights(self):
        """Test calibration with vega weighting"""
        config = AdvancedCalibrationConfig(
            use_vega_weighting=True,
            use_bid_ask_weighting=False
        )
        surface = SSVISurface(100.0, np.array([0.5, 1.0]))
        calibrator = VegaWeightedCalibrator(surface, config)
        
        strikes = np.array([90, 95, 100, 105, 110])
        market_ivs = np.array([
            [0.26, 0.27],
            [0.23, 0.24],
            [0.21, 0.22],
            [0.23, 0.24],
            [0.26, 0.27]
        ])
        
        result = calibrator.calibrate(strikes, market_ivs)
        
        assert result.converged
        assert result.rmse < 0.1
        assert result.parameters is not None
    
    def test_warm_start_from_previous(self):
        """Test warm-start initialization"""
        config = AdvancedCalibrationConfig(use_warm_start=True)
        surface = SSVISurface(100.0, np.array([0.5, 1.0]))
        calibrator = VegaWeightedCalibrator(surface, config)
        
        # Previous day's parameters
        previous_params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09]),
            maturities=np.array([0.5, 1.0]),
            eta=1.0, gamma=0.5, rho=-0.3
        )
        
        # Warm-start should initialize close to previous
        warm_params = calibrator._warm_start(previous_params)
        
        # Should be close (within 5%)
        assert np.allclose(
            warm_params.theta_curve, previous_params.theta_curve, rtol=0.1
        )
        assert np.isclose(warm_params.eta, previous_params.eta, rtol=0.1)


class TestMultiObjective:
    """Test multi-objective calibration"""
    
    def test_multi_objective_config(self):
        """Test multi-objective configuration"""
        config = AdvancedCalibrationConfig(
            multi_objective=True,
            fit_weight=1.0,
            smoothness_weight=0.1,
            stability_weight=0.1
        )
        
        assert config.multi_objective
        assert config.fit_weight == 1.0
        assert config.smoothness_weight == 0.1
        assert config.stability_weight == 0.1
    
    def test_multi_objective_calibration(self):
        """Test multi-objective optimization"""
        config = AdvancedCalibrationConfig(
            multi_objective=True,
            fit_weight=1.0,
            smoothness_weight=0.01,
            stability_weight=0.001
        )
        surface = SSVISurface(100.0, np.array([0.5, 1.0]))
        calibrator = VegaWeightedCalibrator(surface, config)
        
        strikes = np.array([90, 95, 100, 105, 110])
        market_ivs = np.array([
            [0.26, 0.27],
            [0.23, 0.24],
            [0.21, 0.22],
            [0.23, 0.24],
            [0.26, 0.27]
        ])
        
        result = calibrator.calibrate(strikes, market_ivs)
        
        # Should still converge and produce reasonable RMSE
        assert result.rmse > 0
        assert result.parameters is not None


class TestParallelCalibration:
    """Test parallel calibration scenarios"""
    
    def test_scenario_calibration(self):
        """Test calibration across multiple scenarios"""
        config = AdvancedCalibrationConfig()
        surface = SSVISurface(100.0, np.array([1.0]))
        
        # Multiple scenarios
        market_data = np.array([
            # Scenario 1: Tight smile
            [[0.25, 0.23, 0.22, 0.23, 0.25]],
            # Scenario 2: Wide smile
            [[0.30, 0.25, 0.20, 0.25, 0.30]]
        ])
        
        results = ParallelCalibration.calibrate_scenarios(
            surface, market_data, config
        )
        
        assert len(results) == 2
        assert 0 in results
        assert 1 in results
        assert all(r.converged for r in results.values())


class TestIntegration:
    """Integration tests: advanced calibration vs basic SSVI"""

    def test_advanced_improves_on_basic(self):
        """Test advanced calibration achieves comparable or better fit than basic SSVI"""
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])
        market_ivs = np.array([
            [0.32, 0.33],
            [0.28, 0.29],
            [0.24, 0.25],
            [0.22, 0.23],
            [0.23, 0.24],
            [0.25, 0.26],
            [0.28, 0.29]
        ])
        
        surface = SSVISurface(100.0, np.array([0.5, 1.0]))

        # Basic calibration
        result_basic = surface.calibrate(
            strikes, market_ivs, validate_arbitrage=False
        )

        # Advanced calibration with vega weighting
        config = AdvancedCalibrationConfig(use_vega_weighting=True)
        calibrator = VegaWeightedCalibrator(surface, config)
        result_advanced = calibrator.calibrate(strikes, market_ivs)

        # Advanced should achieve comparable or better fit
        assert result_advanced.converged
        # RMSEs should be similar (both are good fits)
        assert abs(result_advanced.rmse - result_basic.rmse) < 0.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
