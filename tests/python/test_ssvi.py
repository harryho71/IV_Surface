"""
Test Suite: SSVI Surface Model
Tests surface-wide volatility parametrization with arbitrage-free constraints
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'python'))

from surface.ssvi import SSVISurface, SSVIParameters


class TestSSVIBasics:
    """Test basic SSVI functionality"""
    
    def test_ssvi_initialization(self):
        """Test SSVISurface initialization"""
        forward = 100.0
        maturities = np.array([0.25, 0.5, 1.0])
        
        surface = SSVISurface(forward, maturities, rate=0.04)
        
        assert surface.forward == 100.0
        assert len(surface.maturities) == 3
        assert surface.rate == 0.04
        assert surface.parameters is None
    
    def test_phi_function(self):
        """Test power-law φ(θ) function"""
        surface = SSVISurface(100.0, np.array([1.0]))
        
        # φ(θ) = η/θ^γ
        phi_val = surface.phi(theta=0.04, eta=1.0, gamma=0.5)
        expected = 1.0 / (0.04 ** 0.5)  # 1 / 0.2 = 5.0
        
        assert abs(phi_val - expected) < 1e-6
    
    def test_w_ssvi_formula(self):
        """Test SSVI total variance formula"""
        surface = SSVISurface(100.0, np.array([1.0]))
        
        # Test at k=0 (ATM)
        w_atm = surface.w_ssvi(k=0, theta=0.04, eta=1.0, gamma=0.5, rho=0.0)
        
        # At k=0: w = θ/2 * [1 + √(1-ρ²)] = θ/2 * [1 + 1] = θ
        expected_atm = 0.04
        
        assert abs(w_atm - expected_atm) < 1e-6
    
    def test_ssvi_parameters_serialization(self):
        """Test SSVIParameters to/from dict"""
        params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.16]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.0,
            gamma=0.5,
            rho=-0.3
        )
        
        # To dict and back
        params_dict = params.to_dict()
        params_loaded = SSVIParameters.from_dict(params_dict)
        
        np.testing.assert_array_almost_equal(params.theta_curve, params_loaded.theta_curve)
        np.testing.assert_array_almost_equal(params.maturities, params_loaded.maturities)
        assert params.eta == params_loaded.eta
        assert params.gamma == params_loaded.gamma
        assert params.rho == params_loaded.rho


class TestGatheralJacquierConstraints:
    """Test arbitrage-free constraints"""
    
    def test_valid_parameters(self):
        """Test that valid parameters pass constraints"""
        surface = SSVISurface(100.0, np.array([0.25, 0.5, 1.0]))
        
        # Valid parameters
        params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.16]),  # σ² T (increasing)
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=0.5,
            gamma=0.4,
            rho=0.2
        )
        
        is_valid, violations = surface.check_gatheral_jacquier_constraints(params)
        
        assert is_valid
        assert len(violations) == 0
    
    def test_butterfly_constraint_violation(self):
        """Test butterfly constraint detection"""
        surface = SSVISurface(100.0, np.array([1.0]))
        
        # Violate butterfly: 4θ(1 + |ρ|) > 1
        # With θ=0.3, ρ=0.8: 4*0.3*(1+0.8) = 2.16 > 1
        params = SSVIParameters(
            theta_curve=np.array([0.3]),
            maturities=np.array([1.0]),
            eta=1.0,
            gamma=0.5,
            rho=0.8
        )
        
        is_valid, violations = surface.check_gatheral_jacquier_constraints(params)
        
        assert not is_valid
        assert any('Butterfly' in v for v in violations)
    
    def test_calendar_constraint_violation(self):
        """Test calendar constraint (θ non-decreasing)"""
        surface = SSVISurface(100.0, np.array([0.25, 0.5, 1.0]))
        
        # Decreasing θ violates calendar arbitrage
        params = SSVIParameters(
            theta_curve=np.array([0.16, 0.09, 0.04]),  # Decreasing!
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=0.5,
            gamma=0.4,
            rho=0.1
        )
        
        is_valid, violations = surface.check_gatheral_jacquier_constraints(params)
        
        assert not is_valid
        assert any('Calendar' in v for v in violations)
    
    def test_parameter_bound_violations(self):
        """Test parameter bound checking"""
        surface = SSVISurface(100.0, np.array([1.0]))
        
        # Test rho out of bounds
        params_rho = SSVIParameters(
            theta_curve=np.array([0.04]),
            maturities=np.array([1.0]),
            eta=1.0,
            gamma=0.5,
            rho=1.5  # Invalid!
        )
        
        is_valid, violations = surface.check_gatheral_jacquier_constraints(params_rho)
        assert not is_valid
        assert any('Correlation' in v for v in violations)
        
        # Test gamma out of bounds
        params_gamma = SSVIParameters(
            theta_curve=np.array([0.04]),
            maturities=np.array([1.0]),
            eta=1.0,
            gamma=1.5,  # Invalid! (must be in (0,1))
            rho=0.0
        )
        
        is_valid, violations = surface.check_gatheral_jacquier_constraints(params_gamma)
        assert not is_valid
        assert any('Gamma' in v for v in violations)


class TestSSVIEvaluation:
    """Test SSVI surface evaluation"""
    
    def test_evaluate_iv_at_atm(self):
        """Test IV evaluation at ATM"""
        surface = SSVISurface(100.0, np.array([1.0]))
        
        params = SSVIParameters(
            theta_curve=np.array([0.04]),  # θ = 0.04
            maturities=np.array([1.0]),
            eta=1.0,
            gamma=0.5,
            rho=0.0
        )
        
        # At ATM (K=F=100), k=0, w=θ=0.04
        # σ = √(w/T) = √(0.04/1.0) = 0.2
        iv_atm = surface.evaluate_iv(strike=100.0, maturity=1.0, parameters=params)
        
        assert abs(iv_atm - 0.2) < 1e-6
    
    def test_evaluate_surface_grid(self):
        """Test surface evaluation on grid"""
        maturities = np.array([0.25, 0.5, 1.0])
        surface = SSVISurface(100.0, maturities)
        
        params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.16]),
            maturities=maturities,
            eta=1.0,
            gamma=0.5,
            rho=-0.2
        )
        
        strikes = np.array([90.0, 100.0, 110.0])
        ivs = surface.evaluate_surface(strikes, maturities, params)
        
        assert ivs.shape == (3, 3)
        assert np.all(ivs > 0)  # All IVs positive
        assert np.all(ivs < 2.0)  # Reasonable IV range
    
    def test_smile_shape(self):
        """Test that SSVI produces correct smile shape"""
        surface = SSVISurface(100.0, np.array([1.0]))
        
        # Negative ρ should produce equity-like smile (higher left wing)
        params = SSVIParameters(
            theta_curve=np.array([0.04]),
            maturities=np.array([1.0]),
            eta=1.0,
            gamma=0.5,
            rho=-0.5  # Negative correlation
        )
        
        strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
        ivs = surface.evaluate_surface(strikes, np.array([1.0]), params)[:, 0]
        
        # Check smile shape: IVs should be higher on left wing
        iv_left = ivs[0]   # 85 strike
        iv_atm = ivs[3]    # 100 strike
        iv_right = ivs[6]  # 115 strike
        
        assert iv_left > iv_atm  # Left wing higher
        assert iv_left > iv_right  # Asymmetric smile


class TestSSVICalibration:
    """Test SSVI calibration"""
    
    def test_calibration_simple(self):
        """Test calibration on simple synthetic data"""
        maturities = np.array([0.25, 0.5, 1.0])
        surface = SSVISurface(forward=100.0, maturities=maturities, rate=0.04)
        
        # Synthetic market data (simple smile)
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([
            [0.25, 0.26, 0.27],  # 90 strike
            [0.22, 0.23, 0.24],  # 95 strike
            [0.20, 0.21, 0.22],  # 100 strike (ATM)
            [0.22, 0.23, 0.24],  # 105 strike
            [0.25, 0.26, 0.27]   # 110 strike
        ])
        
        result = surface.calibrate(
            strikes=strikes,
            market_ivs=market_ivs,
            validate_arbitrage=False,  # Skip for speed
            max_iterations=100
        )
        
        assert result.converged
        assert result.rmse < 0.05  # Reasonable fit
        assert surface.parameters is not None
    
    def test_calibration_with_arbitrage_validation(self):
        """Test calibration with arbitrage validation"""
        maturities = np.array([0.5, 1.0])
        surface = SSVISurface(forward=100.0, maturities=maturities)
        
        strikes = np.array([95.0, 100.0, 105.0])
        market_ivs = np.array([
            [0.22, 0.23],
            [0.20, 0.21],
            [0.22, 0.23]
        ])
        
        result = surface.calibrate(
            strikes=strikes,
            market_ivs=market_ivs,
            validate_arbitrage=True
        )
        
        assert result.arbitrage_report is not None
        # SSVI should produce arbitrage-free surfaces by construction
        # (though not guaranteed with real market data)
    
    def test_parameter_initialization(self):
        """Test automatic parameter initialization"""
        maturities = np.array([0.25, 0.5, 1.0])
        surface = SSVISurface(100.0, maturities)
        
        strikes = np.array([95.0, 100.0, 105.0])
        market_ivs = np.array([
            [0.22, 0.24, 0.26],
            [0.20, 0.22, 0.24],
            [0.22, 0.24, 0.26]
        ])
        
        # Test initialization
        params = surface._initialize_parameters(strikes, market_ivs)
        
        assert len(params.theta_curve) == 3
        assert params.eta > 0
        assert 0 < params.gamma < 1
        assert -1 < params.rho < 1
        
        # θ should be non-decreasing
        assert np.all(np.diff(params.theta_curve) >= -1e-8)


class TestSSVIPersistence:
    """Test parameter saving/loading"""
    
    def test_save_load_parameters(self):
        """Test saving and loading parameters"""
        surface = SSVISurface(100.0, np.array([0.25, 0.5, 1.0]))
        
        params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.16]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.0,
            gamma=0.5,
            rho=-0.3
        )
        
        surface.parameters = params
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'ssvi_params.json'
            
            # Save
            surface.save_parameters(filepath)
            assert filepath.exists()
            
            # Load into new surface
            surface2 = SSVISurface(100.0, np.array([0.25, 0.5, 1.0]))
            surface2.load_parameters(filepath)
            
            # Verify
            np.testing.assert_array_almost_equal(
                surface.parameters.theta_curve,
                surface2.parameters.theta_curve
            )
            assert surface.parameters.eta == surface2.parameters.eta
            assert surface.parameters.gamma == surface2.parameters.gamma
            assert surface.parameters.rho == surface2.parameters.rho


class TestIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_equity_smile_calibration(self):
        """Test calibration to realistic equity smile"""
        maturities = np.array([1/12, 3/12, 6/12, 1.0])
        surface = SSVISurface(forward=100.0, maturities=maturities, rate=0.03, dividend_yield=0.02)
        
        # Realistic equity smile (skew to downside)
        strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0])
        market_ivs = np.array([
            [0.32, 0.33, 0.34, 0.35],  # 85
            [0.28, 0.29, 0.30, 0.31],  # 90
            [0.24, 0.25, 0.26, 0.27],  # 95
            [0.22, 0.23, 0.24, 0.25],  # 100 ATM
            [0.23, 0.24, 0.25, 0.26],  # 105
            [0.25, 0.26, 0.27, 0.28]   # 110
        ])
        
        result = surface.calibrate(
            strikes=strikes,
            market_ivs=market_ivs,
            validate_arbitrage=True,
            max_iterations=500
        )
        
        # Should converge
        assert result.converged
        
        # Should have reasonable fit (< 200 bps RMSE)
        assert result.rmse < 0.02
        
        # θ should be non-decreasing
        assert np.all(np.diff(result.parameters.theta_curve) >= -1e-8)
    
    def test_term_structure_consistency(self):
        """Test that SSVI maintains term structure consistency"""
        maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        surface = SSVISurface(100.0, maturities)
        
        params = SSVIParameters(
            theta_curve=np.array([0.04, 0.09, 0.14, 0.16, 0.20, 0.24]),
            maturities=maturities,
            eta=1.0,
            gamma=0.5,
            rho=-0.4
        )
        
        surface.parameters = params
        
        # Evaluate at ATM across all maturities
        strike = 100.0
        ivs = np.array([surface.evaluate_iv(strike, T, params) for T in maturities])
        
        # ATM IV should show realistic term structure
        # (can increase or decrease, but should be smooth)
        diffs = np.diff(ivs)
        
        # No sudden jumps (< 5 vol points)
        assert np.all(np.abs(diffs) < 0.05)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
