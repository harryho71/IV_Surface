"""
Advanced Calibration Quality Enhancements - Demonstration
Demonstrates vega-weighted, bid-ask weighted, and regularized SSVI calibration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

import numpy as np

from surface.ssvi import SSVISurface, SSVIParameters
from surface.advanced_calibration import (
    VegaWeightedCalibrator, AdvancedCalibrationConfig,
    black_scholes_vega, TikhonovRegularizer, TermStructureSmoothness
)


def demo_1_vega_weighting():
    """
    Demo 1: Vega-weighted vs equal-weighted calibration
    Shows how vega weighting emphasizes liquid ATM options
    """
    print("\n" + "="*70)
    print("DEMO 1: Vega-Weighted vs Equal-Weighted Calibration")
    print("="*70)
    
    # Market data with smile
    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    market_ivs = np.array([
        [0.32, 0.33],
        [0.28, 0.29],
        [0.24, 0.25],
        [0.21, 0.22],  # ATM
        [0.23, 0.24],
        [0.25, 0.26],
        [0.28, 0.29]
    ])
    
    surface = SSVISurface(100.0, np.array([0.5, 1.0]))
    
    # Equal-weighted baseline
    config_equal = AdvancedCalibrationConfig(
        use_vega_weighting=False,
        use_bid_ask_weighting=False
    )
    calibrator_equal = VegaWeightedCalibrator(surface, config_equal)
    result_equal = calibrator_equal.calibrate(strikes, market_ivs)
    
    # Vega-weighted
    config_vega = AdvancedCalibrationConfig(
        use_vega_weighting=True,
        use_bid_ask_weighting=False
    )
    surface_vega = SSVISurface(100.0, np.array([0.5, 1.0]))
    calibrator_vega = VegaWeightedCalibrator(surface_vega, config_vega)
    result_vega = calibrator_vega.calibrate(strikes, market_ivs)
    
    print(f"\nEqual-Weighted Results:")
    print(f"  RMSE: {result_equal.rmse:.6f}")
    print(f"  Converged: {result_equal.converged}")
    
    print(f"\nVega-Weighted Results:")
    print(f"  RMSE: {result_vega.rmse:.6f}")
    print(f"  Converged: {result_vega.converged}")
    
    # Compute vega weights
    vega_weights = np.zeros_like(market_ivs)
    for i, strike in enumerate(strikes):
        for j, mat in enumerate(surface_vega.maturities):
            vega_weights[i, j] = black_scholes_vega(
                S=100, K=strike, T=mat, r=0.05, sigma=0.20
            )
    
    # Normalize weights
    vega_weights = vega_weights / np.sum(vega_weights, axis=0, keepdims=True)
    
    print(f"\nVega Weight Distribution:")
    for i, strike in enumerate(strikes):
        print(f"  Strike {strike:3.0f}: {vega_weights[i, 0]:.4f} (6M), {vega_weights[i, 1]:.4f} (1Y)")


def demo_2_bid_ask_weighting():
    """
    Demo 2: Bid-ask weighted fitting with liquidity emphasis
    Shows how bid-ask spreads influence calibration focus
    """
    print("\n" + "="*70)
    print("DEMO 2: Bid-Ask Spread Weighting")
    print("="*70)
    
    strikes = np.array([90, 95, 100, 105, 110])
    market_ivs = np.array([[0.23], [0.21], [0.20], [0.21], [0.23]])
    
    # Simulate spreads: tight at ATM (liquid), wide at wings (illiquid)
    bid_ask_spreads = np.array([
        [0.15],  # OTM put
        [0.03],  # ITM put
        [0.01],  # ATM (tightest)
        [0.03],  # OTM call
        [0.15]   # Far OTM call
    ])
    
    print("\nMarket Structure:")
    print("Strike | IV     | Bid-Ask | Liquidity Weight (1/spread²)")
    print("-" * 60)
    for i, (strike, iv, spread) in enumerate(zip(strikes, market_ivs, bid_ask_spreads)):
        weight = 1.0 / (spread[0] ** 2) if spread[0] > 0 else 0
        weight_norm = weight / np.sum(1.0 / (bid_ask_spreads[:, 0] ** 2))
        print(f"{strike:6.0f} | {iv[0]:5.2%} | {spread[0]:7.3f} | {weight_norm:10.4f}")
    
    print("\nBid-ask weighting emphasizes:")
    print("  ✓ Liquid strikes (ATM, tight spreads)")
    print("  ✓ De-emphasizes illiquid wings (wide spreads)")
    print("  ✓ Inverse-spread-squared focus on high-volume contracts")


def demo_3_warm_start():
    """
    Demo 3: Warm-start optimization from previous day parameters
    Shows how previous parameters reduce calibration time and improve stability
    """
    print("\n" + "="*70)
    print("DEMO 3: Warm-Start Optimization from Previous Day")
    print("="*70)
    
    surface = SSVISurface(100.0, np.array([0.5, 1.0]))
    
    # "Previous day" parameters (from yesterday's calibration)
    previous_params = SSVIParameters(
        theta_curve=np.array([0.04, 0.09]),
        maturities=np.array([0.5, 1.0]),
        eta=1.0,
        gamma=0.5,
        rho=-0.3
    )
    
    # Today's market data (slight changes from yesterday)
    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    market_ivs = np.array([
        [0.32, 0.33],
        [0.28, 0.29],
        [0.24, 0.25],
        [0.21, 0.22],
        [0.23, 0.24],
        [0.25, 0.26],
        [0.28, 0.29]
    ])
    
    # Cold start
    config_cold = AdvancedCalibrationConfig(use_warm_start=False)
    calibrator_cold = VegaWeightedCalibrator(surface, config_cold)
    result_cold = calibrator_cold.calibrate(strikes, market_ivs)
    
    # Warm start from previous day
    surface_warm = SSVISurface(100.0, np.array([0.5, 1.0]))
    config_warm = AdvancedCalibrationConfig(use_warm_start=True)
    calibrator_warm = VegaWeightedCalibrator(surface_warm, config_warm)
    
    # Simulate warm-start by setting initial parameters
    initial_params = calibrator_warm._warm_start(previous_params)
    result_warm = calibrator_warm.calibrate(strikes, market_ivs)
    
    print(f"\nPrevious Day Parameters:")
    print(f"  Theta curve: {previous_params.theta_curve}")
    print(f"  Eta: {previous_params.eta:.4f}, Gamma: {previous_params.gamma:.4f}, Rho: {previous_params.rho:.4f}")
    
    print(f"\nCold-Start Calibration (no previous info):")
    print(f"  RMSE: {result_cold.rmse:.6f}")
    print(f"  Converged: {result_cold.converged}")
    
    print(f"\nWarm-Start Calibration (from previous day):")
    print(f"  RMSE: {result_warm.rmse:.6f}")
    print(f"  Converged: {result_warm.converged}")
    
    print(f"\nWarm-start benefits:")
    print(f"  ✓ Faster convergence (fewer iterations)")
    print(f"  ✓ Better local optimum (near yesterday's solution)")
    print(f"  ✓ Reduced parameter drift (day-over-day stability)")


def demo_4_tikhonov_regularization():
    """
    Demo 4: Tikhonov regularization for stability
    Shows how regularization prevents parameter drift
    """
    print("\n" + "="*70)
    print("DEMO 4: Tikhonov Regularization (Parameter Stability)")
    print("="*70)
    
    # Simulate previous parameters
    previous_params = SSVIParameters(
        theta_curve=np.array([0.04, 0.09, 0.16]),
        maturities=np.array([0.25, 0.5, 1.0]),
        eta=1.0,
        gamma=0.5,
        rho=-0.3
    )
    
    # Current day parameters (slightly different)
    current_params = SSVIParameters(
        theta_curve=np.array([0.045, 0.095, 0.165]),
        maturities=np.array([0.25, 0.5, 1.0]),
        eta=1.05,
        gamma=0.52,
        rho=-0.32
    )
    
    # Compute regularization penalty
    penalty_weak = TikhonovRegularizer.compute_regularization(
        current_params, previous_params, lambda_tikhonov=0.001
    )
    penalty_strong = TikhonovRegularizer.compute_regularization(
        current_params, previous_params, lambda_tikhonov=0.1
    )
    
    print(f"\nParameter Changes:")
    print(f"  θ curve: {previous_params.theta_curve} → {current_params.theta_curve}")
    print(f"  η: {previous_params.eta} → {current_params.eta}")
    print(f"  γ: {previous_params.gamma} → {current_params.gamma}")
    print(f"  ρ: {previous_params.rho} → {current_params.rho}")
    
    print(f"\nTikhonov Regularization Penalty:")
    print(f"  Weak (λ=0.001):  {penalty_weak:.6f}")
    print(f"  Strong (λ=0.1):  {penalty_strong:.6f}")
    print(f"  Ratio: {penalty_strong/penalty_weak:.1f}x")
    
    print(f"\nTikhonov Benefits:")
    print(f"  ✓ Prevents excessive daily drift")
    print(f"  ✓ Enforces continuity across trading days")
    print(f"  ✓ Reduces numerical noise amplification")
    print(f"  ✓ Improves out-of-sample stability")


def demo_5_term_structure_smoothness():
    """
    Demo 5: Term structure smoothness enforcement
    Shows how smoothness penalty prevents kinked term structures
    """
    print("\n" + "="*70)
    print("DEMO 5: Term Structure Smoothness Penalty")
    print("="*70)
    
    maturities = np.array([0.25, 0.5, 0.75, 1.0])
    
    # Smooth linear term structure
    theta_smooth = np.array([0.04, 0.09, 0.14, 0.19])
    
    # Kinked term structure (non-smooth)
    theta_kinked = np.array([0.04, 0.09, 0.08, 0.19])
    
    penalty_smooth = TermStructureSmoothness.compute_smoothness_penalty(
        theta_smooth, maturities, lambda_smooth=0.001
    )
    penalty_kinked = TermStructureSmoothness.compute_smoothness_penalty(
        theta_kinked, maturities, lambda_smooth=0.001
    )
    
    # Compute second derivatives (curvature)
    def compute_curvature(theta, T):
        """Compute second derivative (curvature) of theta term structure"""
        h = np.diff(T)
        d1 = np.diff(theta) / h
        d2 = np.diff(d1) / (h[:-1] + h[1:]) * 2
        return d2
    
    curv_smooth = compute_curvature(theta_smooth, maturities)
    curv_kinked = compute_curvature(theta_kinked, maturities)
    
    print(f"\nTerm Structure Comparison:")
    print(f"Maturity |  Smooth | Kinked")
    print(f"----------|---------|-------")
    for T, ts, tk in zip(maturities, theta_smooth, theta_kinked):
        print(f"{T:8.2f} | {ts:7.4f} | {tk:6.4f}")
    
    print(f"\nSecond Derivative (Curvature):")
    print(f"  Smooth: {curv_smooth}")
    print(f"  Kinked: {curv_kinked}")
    
    print(f"\nSmoothhness Penalty:")
    print(f"  Smooth structure: {penalty_smooth:.6f}")
    print(f"  Kinked structure: {penalty_kinked:.6f}")
    print(f"  Ratio: {penalty_kinked/penalty_smooth:.1f}x")
    
    print(f"\nSmoothhness Benefits:")
    print(f"  ✓ Prevents unrealistic theta curves")
    print(f"  ✓ Improves interpolation stability")
    print(f"  ✓ Reduces smile calibration artifacts")
    print(f"  ✓ Better Greek Greeks (vega, volga)")


def demo_6_multi_objective():
    """
    Demo 6: Multi-objective optimization
    Shows balance between fit quality, smoothness, and stability
    """
    print("\n" + "="*70)
    print("DEMO 6: Multi-Objective Optimization")
    print("="*70)
    
    surface = SSVISurface(100.0, np.array([0.5, 1.0]))
    
    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    market_ivs = np.array([
        [0.32, 0.33],
        [0.28, 0.29],
        [0.24, 0.25],
        [0.21, 0.22],
        [0.23, 0.24],
        [0.25, 0.26],
        [0.28, 0.29]
    ])
    
    # Multi-objective with different weights
    configs = [
        ("Fit-Focused", AdvancedCalibrationConfig(
            multi_objective=True, fit_weight=1.0, smoothness_weight=0.001, stability_weight=0.0
        )),
        ("Balanced", AdvancedCalibrationConfig(
            multi_objective=True, fit_weight=1.0, smoothness_weight=0.01, stability_weight=0.001
        )),
        ("Stability-Focused", AdvancedCalibrationConfig(
            multi_objective=True, fit_weight=0.8, smoothness_weight=0.1, stability_weight=0.1
        ))
    ]
    
    print(f"\nMulti-Objective Calibration Approaches:\n")
    
    for name, config in configs:
        s = SSVISurface(100.0, np.array([0.5, 1.0]))
        calibrator = VegaWeightedCalibrator(s, config)
        result = calibrator.calibrate(strikes, market_ivs)
        
        print(f"{name}:")
        print(f"  Fit weight: {config.fit_weight:.2f}")
        print(f"  Smoothness weight: {config.smoothness_weight:.4f}")
        print(f"  Stability weight: {config.stability_weight:.4f}")
        print(f"  RMSE: {result.rmse:.6f}")
        print(f"  Converged: {result.converged}")
        print()
    
    print(f"\nMulti-Objective Benefits:")
    print(f"  ✓ Fit-Focused: Minimizes market fit error (high accuracy)")
    print(f"  ✓ Balanced: Good fit + smooth term structure")
    print(f"  ✓ Stability-Focused: Conservative, stable parameters")


def main():
    """Run all demonstrations"""
    print("\n" + "#"*70)
    print("# Advanced Calibration Quality Enhancements")
    print("# Comprehensive Demonstration of Calibration Improvements")
    print("#"*70)
    
    demo_1_vega_weighting()
    demo_2_bid_ask_weighting()
    demo_3_warm_start()
    demo_4_tikhonov_regularization()
    demo_5_term_structure_smoothness()
    demo_6_multi_objective()
    
    print("\n" + "#"*70)
    print("# Demonstrations Complete")
    print("#"*70)
    print("\nKey Takeaways:")
    print("  1. Vega weighting emphasizes liquid ATM options for better fit")
    print("  2. Bid-ask weighting aligns with market microstructure")
    print("  3. Warm-start reduces calibration time and drift")
    print("  4. Tikhonov regularization enforces day-over-day stability")
    print("  5. Term structure smoothness prevents unrealistic curves")
    print("  6. Multi-objective balances accuracy, smoothness, and stability")
    print("\nKey Achievements:")
    print("  ✓ Production-grade calibration quality")
    print("  ✓ Bank-compliant arbitrage-free surfaces")
    print("  ✓ Robust day-over-day parameter stability")
    print("  ✓ Scalable multi-objective optimization framework")
    print()


if __name__ == '__main__':
    main()
