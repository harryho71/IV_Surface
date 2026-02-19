"""
SABR Calibration with Arbitrage Validation
Calibration framework ensuring arbitrage-free surfaces
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import json
import logging

try:
    from ..cpp_unified_engine import get_unified_cpp_engine
    from .arbitrage import ArbitrageChecker, ArbitrageReport
except ImportError:
    from cpp_unified_engine import get_unified_cpp_engine
    from surface.arbitrage import ArbitrageChecker, ArbitrageReport

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    """Complete calibration report with arbitrage validation"""
    # Calibration results
    parameters: List[float]  # [alpha, beta, rho, nu]
    rmse: float
    iterations: int
    converged: bool
    
    # Market data
    maturity: float
    strikes: np.ndarray
    market_ivs: np.ndarray
    forward: float
    
    # Fitted IVs
    fitted_ivs: np.ndarray
    residuals: np.ndarray
    
    # Arbitrage validation
    arbitrage_report: Optional[ArbitrageReport] = None
    arbitrage_free: bool = True
    
    # Metadata
    timestamp: str = ""
    validation_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'parameters': {
                'alpha': self.parameters[0],
                'beta': self.parameters[1],
                'rho': self.parameters[2],
                'nu': self.parameters[3]
            },
            'fit_quality': {
                'rmse': float(self.rmse),
                'rmse_bps': float(self.rmse * 10000),
                'iterations': self.iterations,
                'converged': self.converged
            },
            'market_data': {
                'maturity': float(self.maturity),
                'forward': float(self.forward),
                'num_strikes': len(self.strikes),
                'strike_range': [float(self.strikes.min()), float(self.strikes.max())]
            },
            'arbitrage_check': {
                'enabled': self.validation_enabled,
                'arbitrage_free': self.arbitrage_free,
                'violations': self.arbitrage_report.butterfly_violations + 
                             self.arbitrage_report.calendar_violations + 
                             self.arbitrage_report.total_variance_violations 
                             if self.arbitrage_report else 0
            },
            'timestamp': self.timestamp
        }
    
    def save(self, filepath: Path):
        """Save calibration report to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class SABRCalibrator:
    """
    Enhanced SABR calibration with arbitrage validation
    
    Post-calibration arbitrage enforcement
    - Validates fitted surfaces for butterfly/calendar arbitrage
    - Rejects calibrations producing arbitrage violations
    - Comprehensive calibration reporting
    """
    
    def __init__(
        self,
        validate_on_calibration: bool = True,
        arbitrage_tolerance: float = 1e-5,
        max_severe_violations: int = 0,
        log_calibrations: bool = True,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize SABR calibrator with arbitrage validation
        
        Args:
            validate_on_calibration: Enable arbitrage checks after calibration
            arbitrage_tolerance: Tolerance for arbitrage violations
            max_severe_violations: Maximum allowed severe violations (default 0)
            log_calibrations: Save calibration reports to disk
            log_dir: Directory for calibration logs (default: output/calibrations/)
        """
        self.engine = get_unified_cpp_engine()
        if self.engine is None:
            raise RuntimeError("C++ engine unavailable — SABRCalibrator requires C++ engine")

        self.arbitrage_checker = ArbitrageChecker(tolerance=arbitrage_tolerance)
        self.validate_on_calibration = validate_on_calibration
        self.max_severe_violations = max_severe_violations
        self.log_calibrations = log_calibrations
        
        # Setup logging directory
        if log_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            log_dir = project_root / 'output' / 'calibrations'
        self.log_dir = log_dir
        if log_calibrations:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logging
        self.logger = logging.getLogger(__name__)
    
    def _evaluate_smile(
        self,
        forward: float,
        strikes: np.ndarray,
        maturity: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ) -> np.ndarray:
        """
        Evaluate SABR smile across multiple strikes (helper method)
        
        Uses C++ engine for efficient computation
        """
        ivs = np.zeros(len(strikes), dtype=np.float64)
        for i, K in enumerate(strikes):
            ivs[i] = self.engine.sabr_evaluate(
                forward=forward,
                strike=K,
                maturity=maturity,
                alpha=alpha,
                beta=beta,
                rho=rho,
                nu=nu
            )
        return ivs
    
    def calibrate(
        self,
        maturity: float,
        strikes: np.ndarray,
        market_ivs: np.ndarray,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        initial_params: Optional[List[float]] = None,
        forward: Optional[float] = None,
        validate: Optional[bool] = None
    ) -> CalibrationReport:
        """
        Calibrate SABR model with optional arbitrage validation
        
        Args:
            maturity: Time to maturity (years)
            strikes: Strike prices
            market_ivs: Market implied volatilities
            spot: Spot price (for arbitrage checking)
            rate: Risk-free rate (for arbitrage checking)
            dividend_yield: Dividend yield
            initial_params: Initial guess [alpha, beta, rho, nu]
            forward: Forward price (defaults to spot * exp((r-q)*T))
            validate: Override validate_on_calibration setting
            
        Returns:
            CalibrationReport with fit results and arbitrage validation
            
        Raises:
            RuntimeError: If calibration produces severe arbitrage violations
        """
        # Determine validation setting
        do_validate = validate if validate is not None else self.validate_on_calibration
        
        # Compute forward price if not provided
        if forward is None:
            forward = spot * np.exp((rate - dividend_yield) * maturity)
        
        # Run C++ SABR calibration
        params, rmse, iterations, converged = self.engine.sabr_calibrate(
            maturity=maturity,
            strikes=strikes.tolist(),
            market_ivs=market_ivs.tolist(),
            initial_params=initial_params,
            forward=forward
        )
        
        # Evaluate fitted IVs
        fitted_ivs = self._evaluate_smile(
            forward=forward,
            strikes=strikes,
            maturity=maturity,
            alpha=params[0],
            beta=params[1],
            rho=params[2],
            nu=params[3]
        )
        
        residuals = fitted_ivs - market_ivs
        
        # Create base report
        from datetime import datetime
        report = CalibrationReport(
            parameters=params,
            rmse=rmse,
            iterations=iterations,
            converged=converged,
            maturity=maturity,
            strikes=strikes,
            market_ivs=market_ivs,
            forward=forward,
            fitted_ivs=fitted_ivs,
            residuals=residuals,
            timestamp=datetime.now().isoformat(),
            validation_enabled=do_validate
        )
        
        # Arbitrage validation
        if do_validate:
            arb_report = self.arbitrage_checker.validate_surface(
                strikes=strikes,
                maturities=np.array([maturity]),
                implied_vols=fitted_ivs.reshape(-1, 1),
                spot=spot,
                rate=rate,
                dividend_yield=dividend_yield
            )
            
            report.arbitrage_report = arb_report
            report.arbitrage_free = arb_report.is_arbitrage_free
            
            # Check for severe violations
            if arb_report.violations:
                severe_count = sum(1 for v in arb_report.violations if v.severity == 'severe')
                
                if severe_count > self.max_severe_violations:
                    error_msg = (
                        f"Calibration rejected: {severe_count} severe arbitrage violations detected.\n"
                        f"Parameters: α={params[0]:.4f}, β={params[1]:.4f}, ρ={params[2]:.4f}, ν={params[3]:.4f}\n"
                        f"RMSE: {rmse*10000:.2f} bps\n"
                        f"{arb_report.summary}"
                    )
                    self.logger.error(error_msg)
                    
                    # Log failed calibration
                    if self.log_calibrations:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        log_file = self.log_dir / f"calibration_FAILED_{timestamp}.json"
                        report.save(log_file)
                        self.logger.info(f"Failed calibration logged to {log_file}")
                    
                    raise RuntimeError(error_msg)
                
                elif severe_count > 0:
                    self.logger.warning(
                        f"Calibration has {severe_count} severe violations but within tolerance "
                        f"(max={self.max_severe_violations})"
                    )
        
        # Log successful calibration
        if self.log_calibrations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "PASS" if report.arbitrage_free else "WARNING"
            log_file = self.log_dir / f"calibration_{status}_{timestamp}.json"
            report.save(log_file)
            self.logger.info(f"Calibration logged to {log_file}")
        
        return report
    
    def calibrate_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_ivs: np.ndarray,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        initial_params: Optional[List[float]] = None
    ) -> List[CalibrationReport]:
        """
        Calibrate SABR parameters across multiple maturities
        
        Args:
            strikes: 1D array of strike prices
            maturities: 1D array of maturities
            market_ivs: 2D array (strikes × maturities) of market IVs
            spot: Spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield
            initial_params: Initial guess for first maturity
            
        Returns:
            List of CalibrationReport objects (one per maturity)
        """
        reports = []
        current_params = initial_params
        
        for j, T in enumerate(maturities):
            ivs_at_T = market_ivs[:, j]
            
            self.logger.info(f"\nCalibrating T={T:.4f}y ({len(strikes)} strikes)...")
            
            try:
                report = self.calibrate(
                    maturity=T,
                    strikes=strikes,
                    market_ivs=ivs_at_T,
                    spot=spot,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    initial_params=current_params
                )
                
                # Use converged params as initial guess for next maturity
                if report.converged:
                    current_params = report.parameters
                
                reports.append(report)
                
                self.logger.info(
                    f"  ✓ Converged: {report.converged}, "
                    f"RMSE: {report.rmse*10000:.2f} bps, "
                    f"Arbitrage-free: {report.arbitrage_free}"
                )
                
            except RuntimeError as e:
                self.logger.error(f"  ✗ Calibration failed: {e}")
                # Continue with next maturity despite failure
                continue
        
        return reports
    
    def validate_existing_params(
        self,
        params: List[float],
        maturity: float,
        strikes: np.ndarray,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        forward: Optional[float] = None
    ) -> Tuple[bool, Optional[ArbitrageReport]]:
        """
        Validate existing SABR parameters for arbitrage
        
        Args:
            params: SABR parameters [alpha, beta, rho, nu]
            maturity: Time to maturity
            strikes: Strike prices
            spot: Spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield
            forward: Forward price (computed if None)
            
        Returns:
            Tuple of (is_valid, arbitrage_report)
        """
        if forward is None:
            forward = spot * np.exp((rate - dividend_yield) * maturity)
        
        # Evaluate IVs with given parameters
        fitted_ivs = self._evaluate_smile(
            forward=forward,
            strikes=strikes,
            maturity=maturity,
            alpha=params[0],
            beta=params[1],
            rho=params[2],
            nu=params[3]
        )
        
        # Validate
        arb_report = self.arbitrage_checker.validate_surface(
            strikes=strikes,
            maturities=np.array([maturity]),
            implied_vols=fitted_ivs.reshape(-1, 1),
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield
        )
        
        return arb_report.is_arbitrage_free, arb_report
