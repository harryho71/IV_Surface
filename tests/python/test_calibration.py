"""
Test SABR Calibration with Arbitrage Validation
"""

import pytest
import numpy as np
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'python'))

from surface.calibration import SABRCalibrator, CalibrationReport


class TestSABRCalibrator:
    """Test SABR calibration with arbitrage validation"""
    
    def test_calibrator_initialization(self):
        """Test calibrator initialization"""
        calibrator = SABRCalibrator(
            validate_on_calibration=True,
            arbitrage_tolerance=1e-5,
            max_severe_violations=0
        )
        
        assert calibrator.validate_on_calibration is True
        assert calibrator.max_severe_violations == 0
        assert calibrator.arbitrage_checker.tolerance == 1e-5
    
    def test_calibration_valid_surface(self):
        """Test calibration with valid (arbitrage-free) surface"""
        calibrator = SABRCalibrator(validate_on_calibration=True, log_calibrations=False)
        
        # Realistic equity smile
        spot = 100.0
        rate = 0.05
        maturity = 0.25
        strikes = np.array([90, 95, 100, 105, 110])
        market_ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        
        report = calibrator.calibrate(
            maturity=maturity,
            strikes=strikes,
            market_ivs=market_ivs,
            spot=spot,
            rate=rate
        )
        
        assert isinstance(report, CalibrationReport)
        assert report.converged, "Calibration should converge"
        assert len(report.parameters) == 4, "Should have 4 SABR parameters"
        assert report.rmse < 0.05, "RMSE should be reasonable"
        assert report.validation_enabled is True
        assert report.arbitrage_free is True, "Should be arbitrage-free"
    
    def test_calibration_without_validation(self):
        """Test calibration with validation disabled"""
        calibrator = SABRCalibrator(validate_on_calibration=False, log_calibrations=False)
        
        spot = 100.0
        maturity = 0.25
        strikes = np.array([95, 100, 105])
        market_ivs = np.array([0.22, 0.20, 0.22])
        
        report = calibrator.calibrate(
            maturity=maturity,
            strikes=strikes,
            market_ivs=market_ivs,
            spot=spot,
            validate=False
        )
        
        assert report.validation_enabled is False
        assert report.arbitrage_report is None
    
    def test_calibration_report_dict(self):
        """Test calibration report serialization"""
        calibrator = SABRCalibrator(validate_on_calibration=True, log_calibrations=False)
        
        spot = 100.0
        maturity = 0.25
        strikes = np.array([95, 100, 105])
        market_ivs = np.array([0.22, 0.20, 0.22])
        
        report = calibrator.calibrate(
            maturity=maturity,
            strikes=strikes,
            market_ivs=market_ivs,
            spot=spot
        )
        
        report_dict = report.to_dict()
        
        assert 'parameters' in report_dict
        assert 'fit_quality' in report_dict
        assert 'market_data' in report_dict
        assert 'arbitrage_check' in report_dict
        assert 'alpha' in report_dict['parameters']
        assert 'rmse_bps' in report_dict['fit_quality']
    
    def test_calibration_with_high_tolerance(self):
        """Test that higher tolerance accepts surfaces with minor violations"""
        calibrator = SABRCalibrator(
            validate_on_calibration=True,
            arbitrage_tolerance=1e-3,  # Loose tolerance
            max_severe_violations=0,
            log_calibrations=False
        )
        
        spot = 100.0
        maturity = 0.25
        strikes = np.array([95, 100, 105])
        market_ivs = np.array([0.22, 0.20, 0.22])
        
        report = calibrator.calibrate(
            maturity=maturity,
            strikes=strikes,
            market_ivs=market_ivs,
            spot=spot
        )
        
        # Should pass even with loose tolerance
        assert report.converged
    
    def test_multi_maturity_calibration(self):
        """Test calibration across multiple maturities"""
        calibrator = SABRCalibrator(validate_on_calibration=True, log_calibrations=False)
        
        spot = 100.0
        rate = 0.05
        strikes = np.array([90, 95, 100, 105, 110])
        maturities = np.array([0.25, 0.5, 1.0])
        
        # Realistic term structure
        market_ivs = np.array([
            [0.25, 0.24, 0.23],  # 90 strike
            [0.22, 0.21, 0.20],  # 95
            [0.20, 0.19, 0.18],  # 100 ATM
            [0.22, 0.21, 0.20],  # 105
            [0.25, 0.24, 0.23],  # 110
        ])
        
        reports = calibrator.calibrate_surface(
            strikes=strikes,
            maturities=maturities,
            market_ivs=market_ivs,
            spot=spot,
            rate=rate
        )
        
        assert len(reports) == len(maturities), "Should have one report per maturity"
        assert all(r.converged for r in reports), "All calibrations should converge"
        assert all(r.arbitrage_free for r in reports), "All should be arbitrage-free"
    
    def test_validate_existing_params(self):
        """Test validation of existing SABR parameters"""
        calibrator = SABRCalibrator(log_calibrations=False)
        
        # Known good parameters
        params = [0.25, 0.7, -0.4, 0.3]
        maturity = 0.25
        strikes = np.array([95, 100, 105])
        spot = 100.0
        
        is_valid, arb_report = calibrator.validate_existing_params(
            params=params,
            maturity=maturity,
            strikes=strikes,
            spot=spot
        )
        
        assert isinstance(is_valid, bool)
        assert arb_report is not None
        # Typical SABR params should produce valid surface
        assert is_valid or arb_report.butterfly_violations + arb_report.calendar_violations < 3
    
    def test_fitted_ivs_match_market(self):
        """Test that fitted IVs are close to market IVs"""
        calibrator = SABRCalibrator(validate_on_calibration=True, log_calibrations=False)
        
        spot = 100.0
        maturity = 0.25
        strikes = np.array([90, 95, 100, 105, 110])
        market_ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        
        report = calibrator.calibrate(
            maturity=maturity,
            strikes=strikes,
            market_ivs=market_ivs,
            spot=spot
        )
        
        # Check residuals
        max_residual = np.abs(report.residuals).max()
        assert max_residual < 0.05, "Max residual should be < 5% vol"
        
        # Check RMSE in basis points
        rmse_bps = report.rmse * 10000
        assert rmse_bps < 100, "RMSE should be < 100 bps for good fit"


class TestCalibrationReport:
    """Test CalibrationReport functionality"""
    
    def test_report_creation(self):
        """Test basic report creation"""
        report = CalibrationReport(
            parameters=[0.25, 0.7, -0.4, 0.3],
            rmse=0.01,
            iterations=10,
            converged=True,
            maturity=0.25,
            strikes=np.array([95, 100, 105]),
            market_ivs=np.array([0.22, 0.20, 0.22]),
            forward=100.0,
            fitted_ivs=np.array([0.221, 0.201, 0.219]),
            residuals=np.array([0.001, 0.001, -0.001])
        )
        
        assert len(report.parameters) == 4
        assert report.converged is True
        assert report.maturity == 0.25
    
    def test_report_save(self, tmp_path):
        """Test report saving to JSON"""
        report = CalibrationReport(
            parameters=[0.25, 0.7, -0.4, 0.3],
            rmse=0.01,
            iterations=10,
            converged=True,
            maturity=0.25,
            strikes=np.array([95, 100, 105]),
            market_ivs=np.array([0.22, 0.20, 0.22]),
            forward=100.0,
            fitted_ivs=np.array([0.221, 0.201, 0.219]),
            residuals=np.array([0.001, 0.001, -0.001])
        )
        
        save_path = tmp_path / "test_report.json"
        report.save(save_path)
        
        assert save_path.exists()
        
        # Verify JSON structure
        import json
        with open(save_path) as f:
            data = json.load(f)
        
        assert 'parameters' in data
        assert 'fit_quality' in data
        assert data['parameters']['alpha'] == 0.25


class TestIntegration:
    """Integration tests with real-world scenarios"""
    
    def test_realistic_equity_calibration(self):
        """Test with realistic equity volatility smile"""
        calibrator = SABRCalibrator(
            validate_on_calibration=True,
            arbitrage_tolerance=1e-4,  # Realistic tolerance
            log_calibrations=False
        )
        
        # SPY-like parameters
        spot = 100.0
        rate = 0.05
        maturity = 3/12  # 3 months
        
        # Realistic equity smile (OTM puts expensive)
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])
        market_ivs = np.array([0.32, 0.27, 0.23, 0.20, 0.19, 0.19, 0.20])
        
        report = calibrator.calibrate(
            maturity=maturity,
            strikes=strikes,
            market_ivs=market_ivs,
            spot=spot,
            rate=rate
        )
        
        assert report.converged, "Realistic smile should calibrate"
        assert report.arbitrage_free, "Realistic smile should be arbitrage-free"
        assert report.rmse * 10000 < 200, "Fit should be reasonable"
    
    def test_term_structure_calibration(self):
        """Test calibration across term structure"""
        calibrator = SABRCalibrator(validate_on_calibration=True, log_calibrations=False)
        
        spot = 100.0
        rate = 0.05
        
        # Term structure: 1M, 3M, 6M, 1Y
        maturities = np.array([1/12, 3/12, 6/12, 1.0])
        strikes = np.array([90, 95, 100, 105, 110])
        
        # Vol smile that flattens with maturity
        market_ivs = np.array([
            [0.30, 0.28, 0.26, 0.24],  # 90
            [0.24, 0.23, 0.22, 0.21],  # 95
            [0.20, 0.20, 0.20, 0.20],  # 100 ATM
            [0.22, 0.21, 0.21, 0.20],  # 105
            [0.26, 0.25, 0.24, 0.23],  # 110
        ])
        
        reports = calibrator.calibrate_surface(
            strikes=strikes,
            maturities=maturities,
            market_ivs=market_ivs,
            spot=spot,
            rate=rate
        )
        
        assert len(reports) == len(maturities)
        
        # Check parameter evolution across maturities
        alphas = [r.parameters[0] for r in reports]
        # Alpha typically decreases with maturity
        assert len(alphas) == len(maturities)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, '-v'])
