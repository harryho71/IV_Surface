"""
Demo: Quote Adjustment Framework
Demonstrates butterfly and calendar arbitrage adjustment algorithms
"""

import numpy as np
import sys
from pathlib import Path

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from surface.quote_adjustment import QuoteAdjuster
from surface.arbitrage import ArbitrageChecker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def demo_butterfly_adjustment():
    """
    Demo 1: Butterfly Arbitrage Adjustment
    Adjust IVs to remove concavity in strike dimension
    """
    print("\n" + "="*80)
    print("DEMO 1: BUTTERFLY ARBITRAGE ADJUSTMENT")
    print("="*80)
    
    # Market data with butterfly arbitrage (inverted smile - concave)
    strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    market_ivs = np.array([0.22, 0.26, 0.32, 0.35, 0.32, 0.26, 0.22])  # Inverted
    
    print("\nMarket Data:")
    print(f"  Spot: $100.00")
    print(f"  Maturity: 0.5y")
    print(f"  Strikes: {strikes}")
    print(f"  Market IVs: {market_ivs}")
    
    # Check for arbitrage
    checker = ArbitrageChecker(tolerance=1e-6)
    report_before = checker.validate_surface(
        strikes=strikes,
        maturities=np.array([0.5]),
        implied_vols=market_ivs.reshape(-1, 1),
        spot=100.0,
        rate=0.04
    )
    
    print(f"\nOriginal Surface Status:")
    print(f"  Arbitrage-free: {report_before.is_arbitrage_free}")
    print(f"  Butterfly violations: {report_before.butterfly_violations}")
    
    # Adjust quotes
    print("\nAdjusting quotes to remove arbitrage...")
    adjuster = QuoteAdjuster(
        tolerance=1e-6,
        max_adjustment=0.05,  # Allow up to 5 vol points adjustment
        log_adjustments=True
    )
    
    adjustment_report = adjuster.adjust_butterfly_arbitrage(
        strikes=strikes,
        market_ivs=market_ivs,
        maturity=0.5,
        spot=100.0,
        rate=0.04
    )
    
    print(f"\n{'='*80}")
    print("ADJUSTMENT RESULTS")
    print("="*80)
    
    print(f"\nAdjustment Summary:")
    print(f"  Success: {adjustment_report.success}")
    print(f"  Quotes adjusted: {adjustment_report.num_adjusted}/{len(strikes)}")
    print(f"  Max adjustment: {adjustment_report.max_adjustment:.6f} ({adjustment_report.max_adjustment*100:.2f} vol points)")
    print(f"  RMSE adjustment: {adjustment_report.rmse_adjustment:.6f}")
    print(f"  Optimizer iterations: {adjustment_report.iterations}")
    
    print(f"\nAdjusted Surface Status:")
    print(f"  Arbitrage-free: {adjustment_report.adjusted_arbitrage_report.is_arbitrage_free}")
    print(f"  Butterfly violations: {adjustment_report.adjusted_arbitrage_report.butterfly_violations}")
    
    print(f"\nMarket vs Adjusted IVs:")
    print(f"    Strike     Market   Adjusted   Adjustment")
    print(f"  --------   --------   --------   ----------")
    for K, orig, adj in zip(strikes, market_ivs, adjustment_report.adjusted_ivs):
        diff = adj - orig
        sign = '+' if diff >= 0 else ''
        print(f"  {K:8.2f}   {orig:8.4f}   {adj:8.4f}   {sign}{diff:9.6f}")


def demo_calendar_adjustment():
    """
    Demo 2: Calendar Arbitrage Adjustment
    Adjust IVs to remove violations in time dimension
    """
    print("\n" + "="*80)
    print("DEMO 2: CALENDAR ARBITRAGE ADJUSTMENT")
    print("="*80)
    
    # Term structure with calendar arbitrage (decreasing total variance)
    strike = 100.0
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0])
    market_ivs = np.array([0.35, 0.32, 0.28, 0.26, 0.24, 0.22])  # Decreasing
    
    print("\nMarket Data:")
    print(f"  Spot: $100.00")
    print(f"  Strike: ${strike:.2f}")
    print(f"  Maturities (years): {maturities}")
    print(f"  Market IVs: {market_ivs}")
    
    # Check for arbitrage
    checker = ArbitrageChecker(tolerance=1e-6)
    ivs_2d = market_ivs.reshape(1, -1)  # 1 strike x n_maturities
    report_before = checker.validate_surface(
        strikes=np.array([strike]),
        maturities=maturities,
        implied_vols=ivs_2d,
        spot=100.0,
        rate=0.04
    )
    
    print(f"\nOriginal Term Structure Status:")
    print(f"  Arbitrage-free: {report_before.is_arbitrage_free}")
    print(f"  Calendar violations: {report_before.calendar_violations}")
    print(f"  Total variance violations: {report_before.total_variance_violations}")
    
    # Adjust quotes
    print("\nAdjusting quotes to remove calendar arbitrage...")
    adjuster = QuoteAdjuster(
        tolerance=1e-6,
        max_adjustment=0.08,
        log_adjustments=True
    )
    
    adjustment_report = adjuster.adjust_calendar_arbitrage(
        strike=strike,
        maturities=maturities,
        market_ivs=market_ivs,
        spot=100.0,
        rate=0.04
    )
    
    print(f"\n{'='*80}")
    print("ADJUSTMENT RESULTS")
    print("="*80)
    
    print(f"\nAdjustment Summary:")
    print(f"  Success: {adjustment_report.success}")
    print(f"  Quotes adjusted: {adjustment_report.num_adjusted}/{len(maturities)}")
    print(f"  Max adjustment: {adjustment_report.max_adjustment:.6f}")
    print(f"  RMSE adjustment: {adjustment_report.rmse_adjustment:.6f}")
    
    print(f"\nAdjusted Term Structure Status:")
    print(f"  Arbitrage-free: {adjustment_report.adjusted_arbitrage_report.is_arbitrage_free}")
    print(f"  Calendar violations: {adjustment_report.adjusted_arbitrage_report.calendar_violations}")
    
    print(f"\nMarket vs Adjusted IVs:")
    print(f"  Maturity     Market   Adjusted   Adjustment   Total Var")
    print(f"  --------   --------   --------   ----------   ---------")
    for T, orig, adj in zip(maturities, market_ivs, adjustment_report.adjusted_ivs):
        diff = adj - orig
        total_var = adj ** 2 * T
        sign = '+' if diff >= 0 else ''
        print(f"  {T:8.4f}   {orig:8.4f}   {adj:8.4f}   {sign}{diff:9.6f}   {total_var:9.6f}")


def demo_realistic_workflow():
    """
    Demo 3: Realistic Production Workflow
    Complete workflow: Data → Check → Adjust → Validate
    """
    print("\n" + "="*80)
    print("DEMO 3: REALISTIC PRODUCTION WORKFLOW")
    print("="*80)
    
    print("\nScenario: Equity index smile with slight arbitrage from market noise")
    
    # Realistic equity smile with very slight concavity
    strikes = np.array([90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0])
    market_ivs = np.array([0.28, 0.25, 0.23, 0.21, 0.205, 0.21, 0.23, 0.25, 0.28])
    
    maturity = 0.25  # 3 months
    spot = 100.0
    rate = 0.045
    dividend_yield = 0.02
    
    print("\nMarket Data:")
    print(f"  Spot: ${spot:.2f}")
    print(f"  Maturity: {maturity}y ({maturity*12:.1f} months)")
    print(f"  Rate: {rate*100:.2f}%")
    print(f"  Dividend yield: {dividend_yield*100:.2f}%")
    print(f"  Number of strikes: {len(strikes)}")
    
    # Step 1: Initial validation
    print("\nStep 1: Validate original market data...")
    checker = ArbitrageChecker(tolerance=1e-5)  # Bank tolerance
    report_initial = checker.validate_surface(
        strikes=strikes,
        maturities=np.array([maturity]),
        implied_vols=market_ivs.reshape(-1, 1),
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield
    )
    
    print(f"  Arbitrage-free: {report_initial.is_arbitrage_free}")
    print(f"  Violations: {len(report_initial.violations)}")
    
    if report_initial.violations:
        print(f"  Violation details:")
        for v in report_initial.violations[:3]:  # Show first 3
            print(f"    - {v.type}: {v.severity} (magnitude: {v.value:.2e})")
    
    # Step 2: Adjust if needed
    if not report_initial.is_arbitrage_free:
        print("\nStep 2: Adjust quotes to remove arbitrage...")
        adjuster = QuoteAdjuster(
            tolerance=1e-5,
            max_adjustment=0.03,  # Max 3 vol points
            log_adjustments=True
        )
        
        adj_report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=maturity,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield
        )
        
        print(f"  Adjustment status: {'SUCCESS' if adj_report.success else 'PARTIAL'}")
        print(f"  Quotes modified: {adj_report.num_adjusted}/{len(strikes)}")
        print(f"  Max adjustment: {adj_report.max_adjustment*100:.2f} vol points")
        print(f"  RMSE: {adj_report.rmse_adjustment*100:.2f} vol points")
        
        # Step 3: Final validation
        print("\nStep 3: Final validation of adjusted surface...")
        print(f"  Arbitrage-free: {adj_report.adjusted_arbitrage_report.is_arbitrage_free}")
        print(f"  Remaining violations: {len(adj_report.adjusted_arbitrage_report.violations)}")
        
        # Step 4: Decision
        print("\nStep 4: Production decision...")
        if adj_report.success and adj_report.rmse_adjustment < 0.01:
            print("  ✓ ACCEPT: Surface is arbitrage-free with minimal distortion")
            print(f"    - Use adjusted IVs for calibration/pricing")
            print(f"    - Log adjustment for audit trail")
        elif adj_report.adjusted_arbitrage_report.is_arbitrage_free:
            print("  ⚠ REVIEW: Arbitrage removed but with larger adjustments")
            print(f"    - Consider data quality issues")
            print(f"    - May need manual review")
        else:
            print("  ✗ REJECT: Unable to remove all arbitrage within constraints")
            print(f"    - Data quality problem - investigate source")
            print(f"    - Do NOT use for production pricing")
    else:
        print("\nStep 2-4: Surface already arbitrage-free - proceed to calibration")
        print("  ✓ ACCEPT: Use original market data")


def demo_adjustment_bounds():
    """
    Demo 4: Adjustment Bounds and Constraints
    Show how max_adjustment and IV bounds affect results
    """
    print("\n" + "="*80)
    print("DEMO 4: ADJUSTMENT BOUNDS AND CONSTRAINTS")
    print("="*80)
    
    # Market data with arbitrage
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    market_ivs = np.array([0.20, 0.28, 0.35, 0.28, 0.20])
    
    print("\nTesting different max_adjustment limits...")
    print(f"Market IVs: {market_ivs}")
    
    for max_adj in [0.02, 0.05, 0.10]:
        print(f"\nmax_adjustment = {max_adj:.2f} ({max_adj*100:.0f} vol points)")
        
        adjuster = QuoteAdjuster(
            max_adjustment=max_adj,
            log_adjustments=False
        )
        
        report = adjuster.adjust_butterfly_arbitrage(
            strikes=strikes,
            market_ivs=market_ivs,
            maturity=0.5,
            spot=100.0,
            rate=0.04
        )
        
        print(f"  Success: {report.success}")
        print(f"  Violations removed: {report.original_arbitrage_report.butterfly_violations - report.adjusted_arbitrage_report.butterfly_violations}")
        print(f"  Max adjustment used: {report.max_adjustment:.4f}")
        print(f"  RMSE: {report.rmse_adjustment:.4f}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("QUOTE ADJUSTMENT FRAMEWORK - COMPREHENSIVE DEMO")
    print("="*80)
    print("\nKey Features:")
    print("  ✓ Butterfly arbitrage adjustment (strike dimension)")
    print("  ✓ Calendar arbitrage adjustment (time dimension)")
    print("  ✓ Constrained optimization: minimize distortion")
    print("  ✓ Configurable bounds and tolerances")
    print("  ✓ Complete audit trail with JSON logging")
    print("  ✓ Production-ready workflow integration")
    
    # Run all demos
    demo_butterfly_adjustment()
    demo_calendar_adjustment()
    demo_realistic_workflow()
    demo_adjustment_bounds()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. QuoteAdjuster minimally distorts quotes to remove arbitrage")
    print("  2. Configurable max_adjustment prevents excessive changes")
    print("  3. Optimization respects IV bounds and convergence criteria")
    print("  4. Complete logging for audit compliance")
    print("  5. Production workflow: Validate → Adjust → Re-validate → Decision")
    print("\nAdjustment logs saved to: output/adjustments/")
    print("="*80 + "\n")
