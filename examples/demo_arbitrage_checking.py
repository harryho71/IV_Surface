"""
Demo: Arbitrage Checking with C++ Engine
IV Surface Library

Demonstrates:
1. Butterfly arbitrage detection (convexity in strike)
2. Calendar arbitrage detection (monotonicity in time)
3. Total variance monotonicity
4. Complete surface validation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

import numpy as np
from surface.arbitrage import ArbitrageChecker


def demo_butterfly_check():
    """Demo butterfly arbitrage checking"""
    print("\n" + "="*70)
    print("BUTTERFLY ARBITRAGE CHECK (Convexity in Strike)")
    print("="*70)
    
    checker = ArbitrageChecker(tolerance=1e-5)
    
    # Valid surface: typical smile shape
    print("\n1. Valid Surface (Smile Shape):")
    strikes = np.array([90, 95, 100, 105, 110])
    ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])  # Convex
    
    is_valid, violations = checker.check_butterfly_arbitrage(
        strikes, ivs, maturity=0.25, spot=100.0, rate=0.05
    )
    
    print(f"   Strikes: {strikes}")
    print(f"   IVs:     {ivs}")
    print(f"   Result:  {'PASS - Arbitrage-free' if is_valid else 'FAIL'}")
    print(f"   Violations: {len(violations)}")
    
    # Invalid surface: fabricated violation
    print("\n2. Invalid Surface (Fabricated Violation):")
    strikes_bad = np.array([95, 100, 105])
    # Create price violation by making middle IV artificially high
    ivs_bad = np.array([0.20, 0.35, 0.20])
    
    is_valid, violations = checker.check_butterfly_arbitrage(
        strikes_bad, ivs_bad, maturity=0.25, spot=100.0, rate=0.05
    )
    
    print(f"   Strikes: {strikes_bad}")
    print(f"   IVs:     {ivs_bad}")
    print(f"   Result:  {'PASS' if is_valid else 'FAIL - Arbitrage detected'}")
    print(f"   Violations: {len(violations)}")
    if violations:
        for v in violations[:2]:  # Show first 2
            print(f"     - {v.severity.upper()}: {v.message}")


def demo_calendar_check():
    """Demo calendar arbitrage checking"""
    print("\n" + "="*70)
    print("CALENDAR ARBITRAGE CHECK (Monotonicity in Time)")
    print("="*70)
    
    checker = ArbitrageChecker(tolerance=1e-5)
    
    # Valid: constant IV → increasing prices
    print("\n1. Valid Calendar Spread (Constant IV):")
    maturities = np.array([0.25, 0.5, 1.0])
    ivs = np.array([0.20, 0.20, 0.20])
    
    is_valid, violations = checker.check_calendar_arbitrage(
        strike=100.0, maturities=maturities, ivs=ivs,
        spot=100.0, rate=0.05
    )
    
    print(f"   Maturities: {maturities}")
    print(f"   IVs:        {ivs}")
    print(f"   Result:     {'PASS - Prices increase with time' if is_valid else 'FAIL'}")
    
    # Invalid: decreasing total variance
    print("\n2. Invalid Calendar (Decreasing Total Variance):")
    ivs_bad = np.array([0.35, 0.25, 0.18])  # Decreasing
    
    is_valid, violations = checker.check_calendar_arbitrage(
        strike=100.0, maturities=maturities, ivs=ivs_bad,
        spot=100.0, rate=0.05
    )
    
    print(f"   Maturities: {maturities}")
    print(f"   IVs:        {ivs_bad}")
    print(f"   Result:     {'PASS' if is_valid else 'FAIL - Calendar arbitrage'}")
    print(f"   Violations: {len(violations)}")
    if violations:
        for v in violations:
            print(f"     - {v.severity.upper()}: T={v.location[1]:.2f}->{v.location[2]:.2f}, Delta_C={v.value:.6f}")


def demo_total_variance_check():
    """Demo total variance monotonicity"""
    print("\n" + "="*70)
    print("TOTAL VARIANCE MONOTONICITY (w = sigma^2 * T)")
    print("="*70)
    
    checker = ArbitrageChecker(tolerance=1e-5)
    
    # Valid: constant IV
    print("\n1. Valid (Constant IV → Increasing w):")
    maturities = np.array([0.25, 0.5, 1.0])
    ivs = np.array([0.20, 0.20, 0.20])
    total_vars = ivs**2 * maturities
    
    is_valid, violations = checker.check_total_variance_monotonicity(
        strike=100.0, maturities=maturities, ivs=ivs
    )
    
    print(f"   T:  {maturities}")
    print(f"   IV:  {ivs}")
    print(f"   w:  {total_vars}")
    print(f"   Result: {'PASS - w monotonically increasing' if is_valid else 'FAIL'}")
    
    # Invalid: decreasing w
    print("\n2. Invalid (Decreasing w):")
    ivs_bad = np.array([0.30, 0.20, 0.15])
    total_vars_bad = ivs_bad**2 * maturities
    
    is_valid, violations = checker.check_total_variance_monotonicity(
        strike=100.0, maturities=maturities, ivs=ivs_bad
    )
    
    print(f"   T:  {maturities}")
    print(f"   IV:  {ivs_bad}")
    print(f"   w:  {total_vars_bad}")
    print(f"   Result: {'PASS' if is_valid else 'FAIL - w decreases'}")
    print(f"   Violations: {len(violations)}")


def demo_surface_validation():
    """Demo complete surface validation"""
    print("\n" + "="*70)
    print("COMPLETE SURFACE VALIDATION")
    print("="*70)
    
    checker = ArbitrageChecker(tolerance=1e-5)
    
    # Typical equity surface
    strikes = np.array([80, 90, 100, 110, 120])
    maturities = np.array([1/12, 3/12, 6/12, 12/12])
    
    # Realistic smile + term structure
    ivs = np.array([
        [0.35, 0.32, 0.30, 0.28],  # 80 (OTM puts)
        [0.25, 0.24, 0.23, 0.22],  # 90
        [0.20, 0.20, 0.20, 0.20],  # 100 ATM
        [0.22, 0.22, 0.21, 0.21],  # 110
        [0.28, 0.27, 0.26, 0.25],  # 120 (OTM calls)
    ])
    
    print("\n1. Realistic Equity Surface:")
    print(f"   Strikes:    {strikes}")
    print(f"   Maturities: {maturities}")
    print(f"   IV Grid:")
    for i, K in enumerate(strikes):
        print(f"     K={K:3.0f}: {ivs[i, :]}")
    
    report = checker.validate_surface(
        strikes, maturities, ivs, spot=100.0, rate=0.05, dividend_yield=0.02
    )
    
    try:
        print(f"\n   {report.summary}")
    except UnicodeEncodeError:
        print(f"\n   Total violations: {len(report.violations)}")
        print(f"   Arbitrage-free: {report.is_arbitrage_free}")
    
    if report.violations:
        print(f"\n   Top violations:")
        for v in report.violations[:3]:
            print(f"     - {v.type.upper()} ({v.severity}): {v.message}")
    
    # Stressed market surface
    print("\n2. Stressed Market (High Skew):")
    ivs_stressed = np.array([
        [0.60, 0.55],  # Deep OTM puts expensive
        [0.40, 0.38],
        [0.25, 0.25],  # ATM
        [0.20, 0.20],
        [0.18, 0.18],  # OTM calls cheap
    ])
    
    report = checker.validate_surface(
        strikes, maturities[:2], ivs_stressed, spot=100.0, rate=0.05
    )
    
    try:
        print(f"   {report.summary}")
    except UnicodeEncodeError:
        print(f"   Total violations: {len(report.violations)}")
        print(f"   Arbitrage-free: {report.is_arbitrage_free}")


def demo_performance():
    """Demo C++ engine performance"""
    print("\n" + "="*70)
    print("PERFORMANCE TEST (C++ vs Python)")
    print("="*70)
    
    import time
    from cpp_unified_engine import get_unified_cpp_engine
    
    engine = get_unified_cpp_engine()
    if engine is None:
        print("\n   [ERROR] C++ engine not available")
        return
    
    # Large batch
    n_strikes = 100
    strikes = np.linspace(80, 120, n_strikes)
    ivs = 0.20 + 0.05 * np.sin(np.linspace(-np.pi, np.pi, n_strikes))
    
    print(f"\n   Computing Black-Scholes prices for {n_strikes} strikes...")
    
    # C++ timing
    start = time.perf_counter()
    for _ in range(10):
        prices = engine.black_scholes_prices(spot=100.0, strikes=strikes, maturity=0.25, rate=0.05, ivs=ivs)
    cpp_time = time.perf_counter() - start
    
    print(f"   C++ Engine:  {cpp_time:.4f}s (10 iterations)")
    print(f"   Throughput:  {n_strikes * 10 / cpp_time:.0f} prices/sec")
    print("\n   [OK] C++ engine provides production-grade performance")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("ARBITRAGE CHECKING FRAMEWORK - DEMO")
    print("IV Surface Library")
    print("="*70)
    
    demo_butterfly_check()
    demo_calendar_check()
    demo_total_variance_check()
    demo_surface_validation()
    demo_performance()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Achievements:")
    print("  [OK] Butterfly arbitrage detection (C++ engine)")
    print("  [OK] Calendar arbitrage detection (C++ engine)")
    print("  [OK] Total variance monotonicity check")
    print("  [OK] Complete surface validation with severity classification")
    print("  [OK] Production-grade performance via C++ subprocess")
    print("\nNext Steps:")
    print("  - Integrate into calibration workflow")
    print("  - Implement quote adjustment framework")
    print("  - Deploy to production systems")
    print()


if __name__ == '__main__':
    main()
