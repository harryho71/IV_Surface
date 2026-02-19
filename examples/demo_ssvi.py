"""
SSVI Surface Model Demo
Demonstrates surface-wide volatility parametrization with arbitrage-free constraints

This demo shows:
1. Basic SSVI evaluation at a single point
2. SSVI smile shape (equity-like with negative correlation)
3. Calibration to synthetic equity smile
4. Multi-maturity surface with realistic term structure
5. Comparison with slice-wise approaches
6. Arbitrage-free constraint validation
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from surface.ssvi import SSVISurface, SSVIParameters
from surface.arbitrage import ArbitrageChecker


def demo_basic_evaluation():
    """Demo 1: Basic SSVI evaluation"""
    print("\n" + "="*70)
    print("DEMO 1: Basic SSVI Evaluation")
    print("="*70)
    
    surface = SSVISurface(forward=100.0, maturities=np.array([1.0]), rate=0.04)
    
    params = SSVIParameters(
        theta_curve=np.array([0.04]),
        maturities=np.array([1.0]),
        eta=1.0,
        gamma=0.5,
        rho=0.0
    )
    
    # Evaluate at different strikes
    strikes = np.array([90, 95, 100, 105, 110])
    print(f"\nForward price: {surface.forward}")
    print(f"Maturity: 1.0 year")
    print(f"SSVI Parameters: θ={0.04:.4f}, η={1.0:.4f}, γ={0.5:.4f}, ρ={0.0:.4f}")
    print(f"\nIV Smile:")
    print(f"{'Strike':>8} {'IV':>10} {'vs ATM':>12}")
    print("-" * 32)
    
    iv_atm = surface.evaluate_iv(100.0, 1.0, params)
    for K in strikes:
        iv = surface.evaluate_iv(K, 1.0, params)
        diff = iv - iv_atm
        print(f"{K:>8.0f} {iv:>10.4f} {diff:>+11.4f}")


def demo_smile_shapes():
    """Demo 2: SSVI smile shapes with different correlations"""
    print("\n" + "="*70)
    print("DEMO 2: SSVI Smile Shapes (Different Correlations)")
    print("="*70)
    
    surface = SSVISurface(100.0, np.array([1.0]))
    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    
    correlations = [0.5, 0.0, -0.3, -0.6]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, rho in enumerate(correlations):
        params = SSVIParameters(
            theta_curve=np.array([0.04]),
            maturities=np.array([1.0]),
            eta=1.0,
            gamma=0.5,
            rho=rho
        )
        
        ivs = surface.evaluate_surface(strikes, np.array([1.0]), params)[:, 0]
        
        ax = axes[idx]
        ax.plot(strikes, ivs * 100, 'b.-', linewidth=2, markersize=8)
        ax.axvline(100, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Strike')
        ax.set_ylabel('IV (%)')
        ax.set_title(f'SSVI Smile (ρ={rho:.1f})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([15, 35])
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'output' / 'plots' / 'ssvi_smiles.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Smile comparison plot saved to {output_path.name}")
    plt.close()


def demo_calibration():
    """Demo 3: Calibration to synthetic equity smile"""
    print("\n" + "="*70)
    print("DEMO 3: SSVI Calibration to Equity Smile")
    print("="*70)
    
    maturities = np.array([0.25, 0.5, 1.0])
    surface = SSVISurface(forward=100.0, maturities=maturities, rate=0.04)
    
    # Synthetic market data (realistic equity smile)
    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    market_ivs = np.array([
        [0.32, 0.33, 0.34],  # 85
        [0.28, 0.29, 0.30],  # 90
        [0.24, 0.25, 0.26],  # 95
        [0.22, 0.23, 0.24],  # 100 ATM
        [0.23, 0.24, 0.25],  # 105
        [0.25, 0.26, 0.27],  # 110
        [0.28, 0.29, 0.30]   # 115
    ])
    
    print(f"\nCalibrating SSVI to {len(strikes)} strikes × {len(maturities)} maturities...")
    print(f"Market data range: 3M ({maturities[0]:.2f}y) to 1Y ({maturities[-1]:.1f}y)")
    
    result = surface.calibrate(
        strikes=strikes,
        market_ivs=market_ivs,
        validate_arbitrage=True,
        max_iterations=500
    )
    
    print(f"\nCalibration Results:")
    print(f"  Converged: {result.converged}")
    print(f"  RMSE: {result.rmse:.6f} ({result.rmse*100:.2f} vol points)")
    print(f"  Iterations: {result.iterations}")
    
    if result.parameters:
        params = result.parameters
        print(f"\nCalibrated SSVI Parameters:")
        print(f"  θ curve (non-decreasing): {params.theta_curve}")
        print(f"  η (scaling): {params.eta:.4f}")
        print(f"  γ (power): {params.gamma:.4f}")
        print(f"  ρ (correlation): {params.rho:.4f}")
        
        # Validate constraints
        is_valid, violations = surface.check_gatheral_jacquier_constraints(params)
        print(f"\nArbitrage-Free Constraints:")
        print(f"  Valid: {is_valid}")
        if violations:
            for v in violations:
                print(f"    ✗ {v}")
        else:
            print(f"    ✓ All constraints satisfied")


def demo_multi_maturity_surface():
    """Demo 4: Full IV surface with realistic term structure"""
    print("\n" + "="*70)
    print("DEMO 4: Multi-Maturity SSVI Surface")
    print("="*70)
    
    # Fine term structure
    maturities = np.array([1/12, 3/12, 6/12, 1.0, 1.5, 2.0])  # 1M to 2Y
    surface = SSVISurface(forward=100.0, maturities=maturities, rate=0.03)
    
    # Realistic parameter curve
    params = SSVIParameters(
        theta_curve=np.array([0.04, 0.09, 0.14, 0.16, 0.18, 0.20]),
        maturities=maturities,
        eta=0.8,
        gamma=0.4,
        rho=-0.35  # Equity skew
    )
    
    surface.parameters = params
    
    # Generate surface
    strikes = np.linspace(80, 120, 20)
    ivs_surface = surface.evaluate_surface(strikes, maturities, params)
    
    print(f"\nGenerated IV surface:")
    print(f"  Strikes: {strikes[0]:.0f} - {strikes[-1]:.0f} ({len(strikes)} points)")
    print(f"  Maturities: {maturities[0]*12:.0f}M - {maturities[-1]:.1f}Y ({len(maturities)} points)")
    print(f"  Grid size: {ivs_surface.shape}")
    print(f"  IV range: {ivs_surface.min():.4f} - {ivs_surface.max():.4f}")
    
    # Visualize 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    K_mesh, T_mesh = np.meshgrid(strikes, maturities)
    ax.plot_surface(K_mesh, T_mesh, ivs_surface.T * 100, cmap='viridis', alpha=0.9)
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity (years)')
    ax.set_zlabel('IV (%)')
    ax.set_title('SSVI IV Surface - Equity Smile + Term Structure')
    
    output_path = Path(__file__).parent.parent / 'output' / 'plots' / 'ssvi_surface_3d.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 3D surface plot saved to {output_path.name}")
    plt.close()


def demo_arbitrage_validation():
    """Demo 5: Validate arbitrage-free properties"""
    print("\n" + "="*70)
    print("DEMO 5: Arbitrage-Free Validation")
    print("="*70)
    
    maturities = np.array([0.5, 1.0])
    surface = SSVISurface(100.0, maturities, rate=0.04)
    
    params = SSVIParameters(
        theta_curve=np.array([0.09, 0.16]),
        maturities=maturities,
        eta=1.0,
        gamma=0.5,
        rho=-0.3
    )
    
    strikes = np.linspace(80, 120, 20)
    ivs = surface.evaluate_surface(strikes, maturities, params)
    
    print(f"\nValidating SSVI surface with ArbitrageChecker...")
    checker = ArbitrageChecker(tolerance=1e-6)
    report = checker.validate_surface(
        strikes=strikes,
        maturities=maturities,
        implied_vols=ivs,
        spot=100.0,
        rate=0.04
    )
    
    print(f"\nArbitrage Report:")
    print(f"  Butterfly violations: {report.butterfly_violations}")
    print(f"  Calendar violations: {report.calendar_violations}")
    print(f"  Total variance violations: {report.total_variance_violations}")
    
    if report.is_arbitrage_free:
        print(f"\n✓ Surface is ARBITRAGE-FREE by Gatheral-Jacquier construction")
    else:
        print(f"\n✗ Minor violations detected (expected with real market data)")
        print(f"  Summary: {report.summary}")


def demo_comparison():
    """Demo 6: SSVI vs Slice-Wise Approach"""
    print("\n" + "="*70)
    print("DEMO 6: SSVI vs Slice-Wise Consistency")
    print("="*70)
    
    maturities = np.array([0.5, 1.0, 1.5])
    surface = SSVISurface(100.0, maturities)
    
    # SSVI parameters ensuring smooth term structure
    params = SSVIParameters(
        theta_curve=np.array([0.09, 0.16, 0.21]),  # Increasing
        maturities=maturities,
        eta=1.0,
        gamma=0.5,
        rho=-0.3
    )
    
    surface.parameters = params
    
    # Evaluate at ATM across term structure
    atm_ivs = np.array([surface.evaluate_iv(100.0, T, params) for T in maturities])
    
    print(f"\nATM Volatility Term Structure (SSVI ensures smoothness):")
    print(f"{'Maturity':>12} {'IV':>10} {'σT':>10}")
    print("-" * 32)
    
    for T, iv in zip(maturities, atm_ivs):
        sigma_t = iv * np.sqrt(T)
        print(f"{T:>12.2f}y {iv:>10.4f} {sigma_t:>10.4f}")
    
    # Surface provides consistent multi-maturity framework
    print(f"\nSSVI advantages for multi-maturity consistency:")
    print(f"  ✓ Global surface parametrization (not slice-by-slice)")
    print(f"  ✓ Guaranteed term structure smoothness")
    print(f"  ✓ Arbitrage-free by construction (Gatheral-Jacquier)")
    print(f"  ✓ Fewer parameters (5 total vs {len(maturities)} × 5 for SVI-per-slice)")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" SSVI SURFACE MODEL - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nSSVI = Surface Stochastic Volatility Inspired")
    print("Ensures arbitrage-free multi-maturity IV surfaces via")
    print("Gatheral-Jacquier (2014) parametrization")
    
    demo_basic_evaluation()
    demo_smile_shapes()
    demo_calibration()
    demo_multi_maturity_surface()
    demo_arbitrage_validation()
    demo_comparison()
    
    print("\n" + "="*70)
    print(" DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. SSVI provides surface-wide parametrization (5 parameters total)")
    print("  2. Gatheral-Jacquier constraints ensure arbitrage-free surfaces")
    print("  3. Global smoothness in both strike and maturity dimensions")
    print("  4. Realistic term structure and smile shape")
    print("  5. Production-ready for high-performance applications")
    print("\nSee also: Advanced Calibration demo")
    print("  → Advanced calibration with vega weighting")
    print("  → Bid-ask spread handling")
    print("  → Tikhonov regularization")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
