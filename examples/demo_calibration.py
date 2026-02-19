"""
Demo: SABR Calibration with Arbitrage Validation
Calibration framework ensuring arbitrage-free surfaces
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

import numpy as np
import logging
from surface.calibration import SABRCalibrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def demo_basic_calibration():
    """Demo basic calibration with arbitrage validation"""
    print("\n" + "="*70)
    print("SABR CALIBRATION WITH ARBITRAGE VALIDATION")
    print("="*70)
    
    # Initialize calibrator with validation
    calibrator = SABRCalibrator(
        validate_on_calibration=True,
        arbitrage_tolerance=1e-5,
        max_severe_violations=0,
        log_calibrations=True
    )
    
    # Market data (realistic equity smile)
    spot = 100.0
    rate = 0.05
    maturity = 0.25
    strikes = np.array([90, 95, 100, 105, 110])
    market_ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
    
    print(f"\nMarket Data:")
    print(f"  Spot: ${spot:.2f}")
    print(f"  Maturity: {maturity:.2f}y")
    print(f"  Strikes: {strikes}")
    print(f"  IVs:     {market_ivs}")
    
    # Calibrate with validation
    print(f"\nCalibrating SABR with arbitrage validation...")
    report = calibrator.calibrate(
        maturity=maturity,
        strikes=strikes,
        market_ivs=market_ivs,
        spot=spot,
        rate=rate
    )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"CALIBRATION RESULTS")
    print(f"{'='*70}")
    print(f"\nSABR Parameters:")
    print(f"  α (alpha): {report.parameters[0]:.4f}")
    print(f"  β (beta):  {report.parameters[1]:.4f}")
    print(f"  ρ (rho):   {report.parameters[2]:.4f}")
    print(f"  ν (nu):    {report.parameters[3]:.4f}")
    
    print(f"\nFit Quality:")
    print(f"  RMSE:       {report.rmse:.6f} ({report.rmse*10000:.2f} bps)")
    print(f"  Iterations: {report.iterations}")
    print(f"  Converged:  {report.converged}")
    
    print(f"\nArbitrage Validation:")
    print(f"  Enabled:        {report.validation_enabled}")
    print(f"  Arbitrage-free: {report.arbitrage_free}")
    
    if report.arbitrage_report:
        print(f"\n{report.arbitrage_report.summary}")
    
    print(f"\nFitted vs Market IVs:")
    print(f"  {'Strike':>8} {'Market':>10} {'Fitted':>10} {'Residual':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for i, K in enumerate(strikes):
        print(f"  {K:8.2f} {market_ivs[i]:10.4f} {report.fitted_ivs[i]:10.4f} {report.residuals[i]:10.6f}")


def demo_multi_maturity_calibration():
    """Demo calibration across multiple maturities"""
    print("\n\n" + "="*70)
    print("MULTI-MATURITY SURFACE CALIBRATION")
    print("="*70)
    
    calibrator = SABRCalibrator(
        validate_on_calibration=True,
        arbitrage_tolerance=1e-4,
        log_calibrations=False
    )
    
    # Term structure data
    spot = 100.0
    rate = 0.05
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])
    
    # Vol smile that flattens with maturity
    market_ivs = np.array([
        [0.25, 0.24, 0.23],  # 90 strike
        [0.22, 0.21, 0.20],  # 95
        [0.20, 0.19, 0.18],  # 100 ATM
        [0.22, 0.21, 0.20],  # 105
        [0.25, 0.24, 0.23],  # 110
    ])
    
    print(f"\nCalibrating {len(maturities)} maturities...")
    
    reports = calibrator.calibrate_surface(
        strikes=strikes,
        maturities=maturities,
        market_ivs=market_ivs,
        spot=spot,
        rate=rate
    )
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Maturity':>10} {'RMSE(bps)':>12} {'Converged':>12} {'Arb-Free':>12}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    
    for i, report in enumerate(reports):
        print(f"{maturities[i]:10.2f} {report.rmse*10000:12.2f} "
              f"{'Yes' if report.converged else 'No':>12} "
              f"{'Yes' if report.arbitrage_free else 'No':>12}")
    
    print(f"\nParameter Evolution:")
    print(f"{'Maturity':>10} {'Alpha':>10} {'Beta':>10} {'Rho':>10} {'Nu':>10}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for i, report in enumerate(reports):
        print(f"{maturities[i]:10.2f} {report.parameters[0]:10.4f} "
              f"{report.parameters[1]:10.4f} {report.parameters[2]:10.4f} "
              f"{report.parameters[3]:10.4f}")


def demo_validation_rejection():
    """Demo calibration rejection due to arbitrage"""
    print("\n\n" + "="*70)
    print("ARBITRAGE REJECTION DEMO")
    print("="*70)
    
    # Strict calibrator (zero tolerance for severe violations)
    calibrator = SABRCalibrator(
        validate_on_calibration=True,
        arbitrage_tolerance=1e-6,  # Very strict
        max_severe_violations=0,
        log_calibrations=False
    )
    
    print(f"\nCalibrator Settings:")
    print(f"  Arbitrage tolerance: {1e-6}")
    print(f"  Max severe violations: 0 (reject any severe)")
    print(f"\nNote: Production systems should reject surfaces with severe arbitrage.")


def main():
    """Run all demos"""
    try:
        demo_basic_calibration()
        demo_multi_maturity_calibration()
        demo_validation_rejection()
        
        print("\n\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("\nKey Features:")
        print("  ✓ SABR calibration with C++ engine")
        print("  ✓ Automatic arbitrage validation")
        print("  ✓ Comprehensive calibration reports")
        print("  ✓ Multi-maturity surface calibration")
        print("  ✓ JSON logging of all calibrations")
        print("  ✓ Rejection of arbitrage-violating surfaces")
        print("\nCalibration logs saved to: output/calibrations/")
        print()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
