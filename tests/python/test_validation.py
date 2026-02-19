"""
Test Suite: Validation Suite — Comprehensive Tests

Covers:
  SurfaceValidator          — dense_grid_check, heatmap
  BenchmarkStructurePricer  — RR, BF, calendar spread
  GreeksCalculator + GreeksValidator — analytical values, smoothness
  ParameterDynamicsMonitor  — PCA, jump detection, outlier flagging
  Backtester                — run, predict, measure_accuracy, stress_test
  plot_greeks_dashboard     — file saved
"""

import numpy as np
import pytest

from src.python.surface.validation import SurfaceValidator
from src.python.surface.benchmark import BenchmarkStructurePricer
from src.python.surface.greeks import GreeksCalculator, GreeksValidator, plot_greeks_dashboard
from src.python.surface.parameter_monitoring import ParameterDynamicsMonitor
from src.python.surface.backtesting import Backtester, StressScenario


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_surface():
    """Flat 20 % IV surface — arbitrage-free by construction."""
    strikes    = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    ivs        = np.full((len(strikes), len(maturities)), 0.20)
    return strikes, maturities, ivs


@pytest.fixture
def skewed_surface():
    """Realistic negatively-skewed surface (equity-like)."""
    strikes    = np.linspace(80, 120, 9)
    maturities = np.array([0.25, 0.5, 1.0])
    ivs = np.zeros((len(strikes), len(maturities)))
    for t_idx, T in enumerate(maturities):
        for k_idx, K in enumerate(strikes):
            moneyness = np.log(K / 100.0)
            skew  = -0.10
            smile = 0.03
            iv = 0.20 + skew * moneyness + smile * moneyness ** 2
            ivs[k_idx, t_idx] = max(iv, 0.05 + 0.01 * t_idx)
    return strikes, maturities, ivs


# ===========================================================================
# 5.1 — SurfaceValidator (original + extended)
# ===========================================================================

class TestSurfaceValidator:

    def test_flat_surface_no_violations(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        validator = SurfaceValidator(tolerance=1e-6)
        summary = validator.dense_grid_check(
            strikes=strikes, maturities=maturities, ivs=ivs,
            spot=100.0, rate=0.0,
        )
        assert summary.is_valid
        assert summary.total_violations == 0
        assert summary.severe_violations == 0

    def test_total_variance_violation_detected(self):
        strikes    = np.array([90.0, 100.0, 110.0])
        maturities = np.array([0.5, 1.0])
        ivs = np.array([[0.50, 0.05],
                        [0.50, 0.05],
                        [0.50, 0.05]])
        validator = SurfaceValidator(tolerance=1e-6)
        summary = validator.dense_grid_check(
            strikes=strikes, maturities=maturities, ivs=ivs, spot=100.0,
        )
        assert not summary.is_valid
        assert summary.total_violations > 0

    def test_report_violations_keys(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        validator = SurfaceValidator()
        summary = validator.dense_grid_check(
            strikes=strikes, maturities=maturities, ivs=ivs, spot=100.0,
        )
        counts = validator.report_violations(summary.report)
        assert set(counts.keys()) == {"butterfly", "calendar", "total_variance"}

    def test_heatmap_saved(self, tmp_path, flat_surface):
        strikes, maturities, ivs = flat_surface
        validator = SurfaceValidator()
        summary = validator.dense_grid_check(
            strikes=strikes, maturities=maturities, ivs=ivs, spot=100.0,
        )
        out = tmp_path / "heatmap.png"
        saved = validator.visualize_violations_heatmap(
            strikes=strikes, maturities=maturities,
            violations=summary.report.violations,
            save_path=str(out),
        )
        assert saved is not None
        assert out.exists()

    def test_bid_ask_spreads_accepted(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        spreads = np.full_like(ivs, 0.01)
        validator = SurfaceValidator()
        summary = validator.dense_grid_check(
            strikes=strikes, maturities=maturities, ivs=ivs,
            spot=100.0, bid_ask_spreads=spreads,
        )
        assert summary.total_violations == 0


# ===========================================================================
# 5.2 — BenchmarkStructurePricer
# ===========================================================================

class TestBenchmarkStructurePricer:

    def test_risk_reversal_returns_float(self, skewed_surface):
        strikes, maturities, ivs = skewed_surface
        pricer = BenchmarkStructurePricer()
        rr = pricer.price_risk_reversal(
            delta=0.25, strikes=strikes, ivs=ivs[:, 0],
            maturity=maturities[0], spot=100.0,
        )
        assert isinstance(rr, float)

    def test_butterfly_returns_float(self, skewed_surface):
        strikes, maturities, ivs = skewed_surface
        pricer = BenchmarkStructurePricer()
        bf = pricer.price_butterfly(
            delta=0.25, strikes=strikes, ivs=ivs[:, 0],
            maturity=maturities[0], spot=100.0,
        )
        assert isinstance(bf, float)

    def test_calendar_spread_non_negative_for_arb_free(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        pricer = BenchmarkStructurePricer()
        atm_k = int(np.argmin(np.abs(strikes - 100.0)))
        cs = pricer.price_calendar_spread(
            strike=strikes[atm_k],
            near_maturity=maturities[0],
            far_maturity=maturities[1],
            iv_near=ivs[atm_k, 0],
            iv_far=ivs[atm_k, 1],
            spot=100.0,
        )
        assert cs >= -1e-8

    def test_full_benchmark_result_count(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        pricer = BenchmarkStructurePricer()
        report = pricer.run_full_benchmark(
            strikes=strikes, maturities=maturities, ivs=ivs, spot=100.0,
        )
        # 4 structures per maturity + (n_mat-1) calendar spreads
        expected_min = len(maturities) * 4 + len(maturities) - 1
        assert len(report.results) >= expected_min

    def test_no_market_data_max_diff_zero(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        pricer = BenchmarkStructurePricer()
        report = pricer.run_full_benchmark(
            strikes=strikes, maturities=maturities, ivs=ivs, spot=100.0,
        )
        assert report.max_diff == 0.0

    def test_10d_rr_vs_25d_rr_steep_skew(self):
        strikes = np.linspace(70, 130, 13)
        ivs = np.array([0.35 - 0.015 * i for i in range(13)])
        pricer = BenchmarkStructurePricer()
        rr_25 = pricer.price_risk_reversal(
            delta=0.25, strikes=strikes, ivs=ivs, maturity=0.5, spot=100.0,
        )
        rr_10 = pricer.price_risk_reversal(
            delta=0.10, strikes=strikes, ivs=ivs, maturity=0.5, spot=100.0,
        )
        assert rr_10 < rr_25


# ===========================================================================
# 5.3 — GreeksCalculator + GreeksValidator
# ===========================================================================

class TestGreeksCalculator:

    def test_delta_in_unit_interval(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        calc = GreeksCalculator()
        g = calc.compute(strikes, maturities, ivs, spot=100.0, rate=0.05)
        assert np.all(g["delta"] >= 0.0)
        assert np.all(g["delta"] <= 1.0)

    def test_gamma_non_negative(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        calc = GreeksCalculator()
        g = calc.compute(strikes, maturities, ivs, spot=100.0, rate=0.05)
        assert np.all(g["gamma"] >= 0.0)

    def test_vega_non_negative(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        calc = GreeksCalculator()
        g = calc.compute(strikes, maturities, ivs, spot=100.0, rate=0.05)
        assert np.all(g["vega"] >= 0.0)

    def test_atm_delta_near_half(self):
        calc = GreeksCalculator()
        g = calc.compute(
            np.array([100.0]), np.array([1.0]),
            np.array([[0.20]]), spot=100.0, rate=0.0,
        )
        assert abs(g["delta"][0, 0] - 0.5) < 0.05

    def test_deep_otm_delta_near_zero(self):
        calc = GreeksCalculator()
        g = calc.compute(
            np.array([200.0]), np.array([0.25]),
            np.array([[0.20]]), spot=100.0, rate=0.0,
        )
        assert g["delta"][0, 0] < 0.05

    def test_all_greeks_present_and_correct_shape(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        calc = GreeksCalculator()
        g = calc.compute(strikes, maturities, ivs, spot=100.0)
        for key in ("delta", "gamma", "vega", "theta", "vanna", "volga"):
            assert key in g
            assert g[key].shape == (len(strikes), len(maturities))


class TestGreeksValidator:

    def test_smooth_surface_passes(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        calc = GreeksCalculator()
        g = calc.compute(strikes, maturities, ivs, spot=100.0)
        validator = GreeksValidator(oscillation_threshold=0.05)
        report = validator.validate(g, strikes, maturities)
        assert report.is_smooth

    def test_injected_oscillation_detected(self):
        strikes    = np.linspace(80, 120, 10)
        maturities = np.array([0.5, 1.0, 2.0])
        ivs        = np.full((len(strikes), len(maturities)), 0.20)
        calc = GreeksCalculator()
        g = calc.compute(strikes, maturities, ivs, spot=100.0)
        # Inject a clear alternating zigzag so delta wiggles: +0.3, -0.3, +0.3, ...
        zigzag = np.where(np.arange(len(strikes)) % 2 == 0, 0.3, -0.3)
        g["delta"][:, 0] += zigzag
        validator = GreeksValidator(oscillation_threshold=0.01)
        report = validator.validate(g, strikes, maturities)
        assert not report.is_smooth
        assert report.delta_oscillations > 0

    def test_summary_string_contains_pass_or_fail(self, flat_surface):
        strikes, maturities, ivs = flat_surface
        calc = GreeksCalculator()
        g = calc.compute(strikes, maturities, ivs, spot=100.0)
        validator = GreeksValidator()
        report = validator.validate(g, strikes, maturities)
        assert "PASS" in report.summary() or "FAIL" in report.summary()


class TestGreeksDashboard:

    def test_dashboard_png_saved(self, tmp_path, flat_surface):
        strikes, maturities, ivs = flat_surface
        calc = GreeksCalculator()
        g = calc.compute(strikes, maturities, ivs, spot=100.0, rate=0.02)
        out = tmp_path / "greeks_dashboard.png"
        saved = plot_greeks_dashboard(
            greeks=g, strikes=strikes, maturities=maturities,
            save_path=str(out),
        )
        assert saved is not None
        assert out.exists()


# ===========================================================================
# 5.4 — ParameterDynamicsMonitor
# ===========================================================================

class TestParameterDynamicsMonitor:

    @pytest.fixture
    def smooth_history(self):
        rng = np.random.default_rng(42)
        steps = rng.normal(0, 0.01, (60, 5))
        return np.cumsum(steps, axis=0) + 0.20

    def test_pca_output_shapes(self, smooth_history):
        monitor = ParameterDynamicsMonitor(n_components=3)
        report = monitor.analyze(smooth_history)
        assert report.n_components == 3
        assert report.loadings.shape == (3, smooth_history.shape[1])
        assert report.scores.shape == (smooth_history.shape[0], 3)

    def test_explained_variance_le_one(self, smooth_history):
        monitor = ParameterDynamicsMonitor(n_components=3)
        report = monitor.analyze(smooth_history)
        assert report.cumulative_explained <= 1.0 + 1e-9

    def test_target_met_full_components(self):
        rng = np.random.default_rng(0)
        history = rng.normal(0, 1, (50, 3))
        monitor = ParameterDynamicsMonitor(n_components=3, explained_variance_target=0.90)
        report = monitor.analyze(history)
        assert report.target_met

    def test_jump_detected_at_correct_index(self, smooth_history):
        history = smooth_history.copy()
        history[20, :] *= 3.0   # 200 % jump
        monitor = ParameterDynamicsMonitor(jump_threshold=0.10)
        report = monitor.analyze(history)
        assert 20 in report.jump_flags

    def test_outlier_detected(self, smooth_history):
        history = smooth_history.copy()
        history[45, :] = 500.0
        monitor = ParameterDynamicsMonitor(outlier_sigma=2.0)
        report = monitor.analyze(history)
        assert 45 in report.outlier_flags

    def test_summary_contains_pca_monitor(self, smooth_history):
        monitor = ParameterDynamicsMonitor()
        report = monitor.analyze(smooth_history)
        assert "PCA Monitor" in report.summary

    def test_plot_scores_saved(self, tmp_path, smooth_history):
        monitor = ParameterDynamicsMonitor(n_components=3)
        report = monitor.analyze(smooth_history)
        out = tmp_path / "pca_scores.png"
        saved = monitor.plot_scores(report, save_path=str(out))
        assert saved is not None
        assert out.exists()


# ===========================================================================
# 5.5 — Backtester
# ===========================================================================

class TestBacktester:

    @pytest.fixture
    def history(self):
        strikes    = np.linspace(90, 110, 5)
        maturities = np.array([0.25, 0.5, 1.0])
        snapshots  = []
        for i in range(10):
            ivs = np.full((5, 3), 0.20 + 0.005 * np.sin(i))
            snapshots.append({
                "date":       f"2026-01-{i+1:02d}",
                "strikes":    strikes,
                "maturities": maturities,
                "ivs":        ivs,
            })
        return snapshots

    def test_run_produces_correct_n_dates(self, history):
        bt = Backtester()
        report = bt.run(history)
        assert report.n_dates == len(history) - 1

    def test_carry_forward_small_rmse(self, history):
        bt = Backtester(rmse_threshold=0.01)
        report = bt.run(history)
        assert report.mean_rmse < 0.01

    def test_passed_flag_truthy(self, history):
        bt = Backtester(rmse_threshold=0.05)
        report = bt.run(history)
        assert report.passed

    def test_predict_shape(self, history):
        curr = history[0]
        bt = Backtester()
        pred = bt.predict(
            np.asarray(curr["ivs"]),
            np.asarray(curr["strikes"]),
            np.asarray(curr["maturities"]),
        )
        assert pred.shape == np.asarray(curr["ivs"]).shape

    def test_measure_accuracy_values(self):
        predicted = np.full((5, 3), 0.20)
        realised  = np.full((5, 3), 0.21)
        bt = Backtester()
        m = bt.measure_accuracy(predicted, realised)
        assert abs(m["rmse"] - 0.01) < 1e-9
        assert abs(m["mae"]  - 0.01) < 1e-9
        assert abs(m["bias"] + 0.01) < 1e-9   # predicted < realised → negative bias

    def test_stress_test_default_scenarios(self):
        base_ivs   = np.full((5, 3), 0.20)
        strikes    = np.linspace(90, 110, 5)
        maturities = np.array([0.25, 0.5, 1.0])
        bt = Backtester()
        results = bt.stress_test(base_ivs, strikes, maturities, spot=100.0)
        assert len(results) == len(Backtester.DEFAULT_SCENARIOS)
        for sc in Backtester.DEFAULT_SCENARIOS:
            assert sc.name in results
            assert results[sc.name]["shocked_ivs"].shape == base_ivs.shape

    def test_stress_test_vol_spike_raises_ivs(self):
        base_ivs   = np.full((5, 3), 0.20)
        bt = Backtester()
        results = bt.stress_test(
            base_ivs, np.linspace(90, 110, 5), np.array([0.25, 0.5, 1.0]),
            spot=100.0,
            scenarios=[StressScenario("spike", spot_shock=0.0, vol_shock=0.10)],
        )
        assert np.all(results["spike"]["shocked_ivs"] > base_ivs - 1e-9)

    def test_rmse_plot_saved(self, tmp_path, history):
        bt = Backtester()
        report = bt.run(history)
        out = tmp_path / "backtest_rmse.png"
        saved = bt.plot_backtest_rmse(report, save_path=str(out))
        assert saved is not None
        assert out.exists()

    def test_raises_for_single_snapshot(self):
        history = [{"date": "d1", "strikes": np.array([100.0]),
                    "maturities": np.array([0.5]), "ivs": np.array([[0.2]])}]
        bt = Backtester()
        with pytest.raises(ValueError):
            bt.run(history)

    def test_summary_string(self, history):
        bt = Backtester()
        report = bt.run(history)
        assert "Backtest Report" in report.summary()
