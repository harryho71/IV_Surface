"""
Test Suite: Operational Controls — comprehensive test suite

Covers:
  CalibrationLog            (TestCalibrationLog        — 6 tests)
  ParameterVersionCtrl      (TestParameterVersionCtrl  — 7 tests)
  AlertSystem               (TestAlertSystem            — 9 tests)
  OverrideManager           (TestOverrideManager        — 7 tests)
  BidAskSurfaceGenerator    (TestBidAskSurfaceGenerator — 7 tests)
  DailyCalibrationPipeline  (TestDailyCalibrationPipeline — 9 tests)
  TraderAdjustments         (TestTraderAdjustments      — 7 tests)

Total: 52 tests
"""

from __future__ import annotations


import numpy as np
import pytest

# ── Module imports ─────────────────────────────────────────────────────────────
from src.python.surface.audit import CalibrationLog, CalibrationRecord, ParameterVersionControl
from src.python.surface.alerting import AlertSystem
from src.python.surface.overrides import Override, OverrideManager, TraderAdjustments
from src.python.surface.bid_ask import BidAskSurfaceGenerator
from src.python.surface.pipeline import (
    DailyCalibrationPipeline,
    PipelineConfig,
    PipelineReport,
    STEP_ORDER,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a fresh temp directory for each test."""
    return tmp_path


@pytest.fixture()
def flat_ivs():
    """5×4 flat 20 % IV surface."""
    strikes = np.linspace(80, 120, 5)
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    ivs = np.full((len(strikes), len(maturities)), 0.20)
    return strikes, maturities, ivs


@pytest.fixture()
def sample_params():
    return {"rho": -0.30, "eta": 0.50, "gamma": 0.40, "beta": 0.70}


@pytest.fixture()
def sample_market_snapshot():
    return {"spot": 100.0, "rate": 0.05, "n_options": 25}


@pytest.fixture()
def sample_convergence():
    return {"rmse": 0.0021, "iterations": 47, "converged": True}


@pytest.fixture()
def sample_arb_result():
    return {"butterfly": True, "calendar": True, "total_variance": True}


# ═══════════════════════════════════════════════════════════════════════════════
# 6.1  CalibrationLog
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrationLog:

    def _make_log(self, tmp_dir) -> CalibrationLog:
        return CalibrationLog(log_dir=tmp_dir / "cal_log")

    def test_record_returns_string_id(
        self, tmp_dir, sample_params, sample_market_snapshot,
        sample_convergence, sample_arb_result
    ):
        log = self._make_log(tmp_dir)
        rid = log.record(
            model="ssvi",
            operator="quant",
            market_data_snapshot=sample_market_snapshot,
            calibration_parameters=sample_params,
            convergence_metrics=sample_convergence,
            arbitrage_check_results=sample_arb_result,
        )
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_get_retrieves_record(
        self, tmp_dir, sample_params, sample_market_snapshot,
        sample_convergence, sample_arb_result
    ):
        log = self._make_log(tmp_dir)
        rid = log.record(
            model="ssvi",
            operator="quant",
            market_data_snapshot=sample_market_snapshot,
            calibration_parameters=sample_params,
            convergence_metrics=sample_convergence,
            arbitrage_check_results=sample_arb_result,
        )
        rec = log.get(rid)
        assert isinstance(rec, CalibrationRecord)
        assert rec.record_id == rid
        assert rec.model == "ssvi"
        assert rec.calibration_parameters["rho"] == pytest.approx(-0.30)

    def test_get_raises_for_unknown_id(self, tmp_dir):
        log = self._make_log(tmp_dir)
        with pytest.raises(KeyError):
            log.get("does-not-exist")

    def test_count_increments(
        self, tmp_dir, sample_params, sample_market_snapshot,
        sample_convergence, sample_arb_result
    ):
        log = self._make_log(tmp_dir)
        for _ in range(3):
            log.record(
                model="ssvi",
                operator="quant",
                market_data_snapshot=sample_market_snapshot,
                calibration_parameters=sample_params,
                convergence_metrics=sample_convergence,
                arbitrage_check_results=sample_arb_result,
            )
        assert log.count() == 3

    def test_list_records_all(
        self, tmp_dir, sample_params, sample_market_snapshot,
        sample_convergence, sample_arb_result
    ):
        log = self._make_log(tmp_dir)
        for model in ("ssvi", "sabr", "svi"):
            log.record(
                model=model,
                operator="quant",
                market_data_snapshot=sample_market_snapshot,
                calibration_parameters=sample_params,
                convergence_metrics=sample_convergence,
                arbitrage_check_results=sample_arb_result,
            )
        records = log.list_records()
        assert len(records) == 3

    def test_list_records_filtered_by_model(
        self, tmp_dir, sample_params, sample_market_snapshot,
        sample_convergence, sample_arb_result
    ):
        log = self._make_log(tmp_dir)
        for model in ("ssvi", "sabr", "ssvi"):
            log.record(
                model=model,
                operator="quant",
                market_data_snapshot=sample_market_snapshot,
                calibration_parameters=sample_params,
                convergence_metrics=sample_convergence,
                arbitrage_check_results=sample_arb_result,
            )
        ssvi_records = log.list_records(model="ssvi")
        assert len(ssvi_records) == 2
        assert all(r.model == "ssvi" for r in ssvi_records)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.2  ParameterVersionControl
# ═══════════════════════════════════════════════════════════════════════════════

class TestParameterVersionControl:

    def _make_vc(self, tmp_dir) -> ParameterVersionControl:
        return ParameterVersionControl(vc_dir=tmp_dir / "vc")

    def test_commit_returns_version_id(self, tmp_dir, sample_params):
        vc = self._make_vc(tmp_dir)
        vid = vc.commit(sample_params, "initial fit")
        assert isinstance(vid, str)
        assert len(vid) > 0

    def test_head_points_to_latest_commit(self, tmp_dir, sample_params):
        vc = self._make_vc(tmp_dir)
        v1 = vc.commit(sample_params, "v1")
        v2 = vc.commit({**sample_params, "rho": -0.40}, "v2")
        assert vc.head == v2

    def test_checkout_returns_correct_params(self, tmp_dir, sample_params):
        vc = self._make_vc(tmp_dir)
        v1 = vc.commit(sample_params, "initial")
        _ = vc.commit({**sample_params, "rho": -0.50}, "updated")
        params = vc.checkout(v1)
        assert params["rho"] == pytest.approx(-0.30)

    def test_diff_detects_changed_params(self, tmp_dir, sample_params):
        vc = self._make_vc(tmp_dir)
        v1 = vc.commit(sample_params, "v1")
        new_params = dict(sample_params)
        new_params["rho"] = -0.50   # changed
        v2 = vc.commit(new_params, "v2")
        diff = vc.diff(v1, v2)
        assert "rho" in diff
        assert diff["rho"]["from"] == pytest.approx(-0.30)
        assert diff["rho"]["to"] == pytest.approx(-0.50)
        # eta, gamma, beta unchanged → not in diff
        assert "eta" not in diff

    def test_diff_change_pct_correct(self, tmp_dir):
        vc = self._make_vc(tmp_dir)
        v1 = vc.commit({"rho": -0.40}, "v1")
        v2 = vc.commit({"rho": -0.48}, "v2")        # 20 % increase in magnitude
        diff = vc.diff(v1, v2)
        # change_pct = (−0.48 − (−0.40)) / |−0.40| × 100 = −20 %
        assert abs(diff["rho"]["change_pct"]) == pytest.approx(20.0, rel=0.01)

    def test_rollback_restores_params(self, tmp_dir, sample_params):
        vc = self._make_vc(tmp_dir)
        v1 = vc.commit(sample_params, "v1")
        _  = vc.commit({**sample_params, "rho": -0.99}, "corrupted")
        params = vc.rollback(v1)
        assert params["rho"] == pytest.approx(-0.30)
        assert vc.head == v1

    def test_history_sorted_ascending(self, tmp_dir, sample_params):
        vc = self._make_vc(tmp_dir)
        for i in range(4):
            vc.commit({**sample_params, "rho": -0.30 - i * 0.05}, f"v{i}")
        hist = vc.history()
        assert len(hist) == 4
        timestamps = [v.timestamp for v in hist]
        assert timestamps == sorted(timestamps)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.3  AlertSystem
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlertSystem:

    def _sys(self) -> AlertSystem:
        return AlertSystem(persist=False)

    def test_no_alerts_for_small_change(self):
        s = self._sys()
        alerts = s.detect_parameter_jumps(
            {"rho": -0.31},
            {"rho": -0.30},
            threshold=0.10,
        )
        assert alerts == []

    def test_warning_for_10pct_change(self):
        s = self._sys()
        alerts = s.detect_parameter_jumps(
            {"rho": -0.34},       # 13.3% change — clearly above 10% threshold
            {"rho": -0.30},
            threshold=0.10,
        )
        assert len(alerts) == 1
        assert alerts[0].level == "WARNING"
        assert alerts[0].source == "parameter_jump"

    def test_critical_for_20pct_change(self):
        s = self._sys()
        alerts = s.detect_parameter_jumps(
            {"rho": -0.42},       # 40% change — clearly above 2× threshold = CRITICAL
            {"rho": -0.30},
            threshold=0.10,
        )
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"

    def test_arbitrage_alert_on_severe_violation(self):
        s = self._sys()
        report = {
            "is_arbitrage_free": False,
            "violations": [
                {"severity": "severe", "type": "butterfly", "value": -0.1},
            ],
        }
        alerts = s.check_arbitrage_violations(report)
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"

    def test_arbitrage_no_alert_when_clean(self):
        s = self._sys()
        report = {"is_arbitrage_free": True, "violations": []}
        alerts = s.check_arbitrage_violations(report)
        assert alerts == []

    def test_convergence_critical_when_not_converged(self):
        s = self._sys()
        alerts = s.check_convergence(rmse=0.02, converged=False)
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"

    def test_convergence_warning_when_high_rmse(self):
        s = self._sys()
        alerts = s.check_convergence(rmse=0.007, converged=True, rmse_threshold=0.005)
        assert len(alerts) == 1
        assert alerts[0].level == "WARNING"

    def test_convergence_no_alert_when_passing(self):
        s = self._sys()
        alerts = s.check_convergence(rmse=0.002, converged=True, rmse_threshold=0.005)
        assert alerts == []

    def test_get_alerts_filtered_by_level(self):
        s = self._sys()
        s.detect_parameter_jumps({"rho": -0.36}, {"rho": -0.30})
        s.check_convergence(rmse=0.007, converged=True, rmse_threshold=0.005)
        crits = s.get_alerts(level="CRITICAL")
        warns = s.get_alerts(level="WARNING")
        assert all(a.level == "CRITICAL" for a in crits)
        assert all(a.level == "WARNING" for a in warns)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.4  OverrideManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverrideManager:

    def _mgr(self, tmp_dir, require_approval=True) -> OverrideManager:
        return OverrideManager(log_dir=tmp_dir / "logs", require_approval=require_approval)

    def test_apply_creates_pending_override(self, tmp_dir):
        mgr = self._mgr(tmp_dir, require_approval=True)
        ov = mgr.apply_override("rho", -0.30, -0.35, "Desk adj", user="trader1")
        assert isinstance(ov, Override)
        assert ov.approved is False
        assert ov.active is False

    def test_no_approval_required_makes_active(self, tmp_dir):
        mgr = self._mgr(tmp_dir, require_approval=False)
        ov = mgr.apply_override("rho", -0.30, -0.35, "Auto", user="system")
        assert ov.active is True

    def test_approve_makes_override_active(self, tmp_dir):
        mgr = self._mgr(tmp_dir)
        ov = mgr.apply_override("rho", -0.30, -0.35, "reason", user="trader1")
        approved = mgr.approve(ov.override_id, approver="risk_mgr")
        assert approved.active is True
        assert approved.approved_by == "risk_mgr"

    def test_reject_makes_override_inactive(self, tmp_dir):
        mgr = self._mgr(tmp_dir)
        ov = mgr.apply_override("rho", -0.30, -0.35, "reason", user="trader1")
        rejected = mgr.reject(ov.override_id, approver="risk_mgr", reason="Out of band")
        assert rejected.active is False
        assert rejected.rejected is True

    def test_get_active_overrides_only_returns_active(self, tmp_dir):
        mgr = self._mgr(tmp_dir, require_approval=False)
        ov1 = mgr.apply_override("rho", -0.30, -0.35, "r1", user="u1")
        ov2 = mgr.apply_override("eta", 0.50, 0.55, "r2", user="u2")
        mgr.deactivate(ov1.override_id)
        active = mgr.get_active_overrides()
        assert len(active) == 1
        assert active[0].override_id == ov2.override_id

    def test_flag_in_reports_annotates_dict(self, tmp_dir):
        mgr = self._mgr(tmp_dir, require_approval=False)
        mgr.apply_override("rho", -0.30, -0.35, "reason", user="trader1")
        report = mgr.flag_in_reports({"surface_id": "SPX_20260218"})
        assert "_overrides_applied" in report
        assert report["_overrides_applied"][0]["param"] == "rho"

    def test_count_breakdown(self, tmp_dir):
        mgr = self._mgr(tmp_dir, require_approval=True)
        ov = mgr.apply_override("rho", -0.30, -0.35, "r", user="u")
        mgr.apply_override("eta", 0.50, 0.55, "r", user="u")
        mgr.approve(ov.override_id, "risk")
        counts = mgr.count()
        assert counts["total"] == 2
        assert counts["pending"] == 1
        assert counts["active"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 6.5  BidAskSurfaceGenerator
# ═══════════════════════════════════════════════════════════════════════════════

class TestBidAskSurfaceGenerator:

    def test_generate_bid_less_than_ask(self, flat_ivs):
        _, _, ivs = flat_ivs
        gen = BidAskSurfaceGenerator()
        bid, ask, _ = gen.generate(ivs, spread_bps=10.0)
        assert np.all(bid <= ask)

    def test_generate_bid_ask_straddle_mid(self, flat_ivs):
        _, _, ivs = flat_ivs
        gen = BidAskSurfaceGenerator()
        bid, ask, _ = gen.generate(ivs, spread_bps=10.0)
        mid = (bid + ask) / 2.0
        np.testing.assert_allclose(mid, ivs, atol=1e-10)

    def test_generate_spread_matches_bps(self, flat_ivs):
        _, _, ivs = flat_ivs
        gen = BidAskSurfaceGenerator()
        bid, ask, _ = gen.generate(ivs, spread_bps=20.0)
        spread = (ask - bid) * 10_000.0
        np.testing.assert_allclose(spread, 20.0, atol=1e-6)

    def test_generate_with_skew_wings_wider(self, flat_ivs):
        strikes, _, ivs = flat_ivs
        atm = 100.0
        gen = BidAskSurfaceGenerator()
        _, _, report = gen.generate_with_skew(
            ivs, strikes, atm_strike=atm, base_spread_bps=10.0, wing_multiplier=3.0
        )
        assert report.max_spread_bps > report.min_spread_bps

    def test_generate_from_market_computes_mid(self, flat_ivs):
        _, _, mid = flat_ivs
        half = 0.005
        bid = mid - half
        ask = mid + half
        gen = BidAskSurfaceGenerator()
        _, computed_mid, _, _ = gen.generate_from_market(bid, ask)
        np.testing.assert_allclose(computed_mid, mid, atol=1e-10)

    def test_generate_from_market_raises_if_ask_lt_bid(self, flat_ivs):
        _, _, mid = flat_ivs
        gen = BidAskSurfaceGenerator()
        with pytest.raises(ValueError, match="ask_ivs must be"):
            gen.generate_from_market(mid + 0.01, mid - 0.01)

    def test_validate_spreads_detects_violation(self, flat_ivs):
        _, _, ivs = flat_ivs
        gen = BidAskSurfaceGenerator()
        bad_bid = ivs + 0.01       # bid > mid
        result = gen.validate_spreads(bad_bid, ivs + 0.02, ivs)
        assert not result["valid"]
        assert result["n_bid_violations"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6.6  DailyCalibrationPipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestDailyCalibrationPipeline:

    def test_run_returns_pipeline_report(self, tmp_dir):
        pipeline = DailyCalibrationPipeline(
            PipelineConfig(output_dir=str(tmp_dir), dry_run=True)
        )
        report = pipeline.run("2026-02-18")
        assert isinstance(report, PipelineReport)

    def test_default_run_passes(self, tmp_dir):
        pipeline = DailyCalibrationPipeline(
            PipelineConfig(output_dir=str(tmp_dir), dry_run=True)
        )
        report = pipeline.run("2026-02-18")
        assert report.passed is True

    def test_report_has_all_steps(self, tmp_dir):
        pipeline = DailyCalibrationPipeline(
            PipelineConfig(output_dir=str(tmp_dir), dry_run=True)
        )
        report = pipeline.run("2026-02-18")
        step_names = [s.name for s in report.steps]
        assert step_names == STEP_ORDER

    def test_register_step_replaces_default(self, tmp_dir):
        called = {"n": 0}

        def my_calibrate(context):
            called["n"] += 1
            return {"model": "custom", "converged": True, "rmse": 0.001}

        pipeline = DailyCalibrationPipeline(
            PipelineConfig(output_dir=str(tmp_dir), dry_run=True)
        )
        pipeline.register_step("calibrate", my_calibrate)
        pipeline.run("2026-02-18")
        assert called["n"] == 1

    def test_register_step_raises_for_unknown_name(self, tmp_dir):
        pipeline = DailyCalibrationPipeline(
            PipelineConfig(output_dir=str(tmp_dir))
        )
        with pytest.raises(ValueError, match="Unknown step"):
            pipeline.register_step("nonexistent", lambda ctx: {})

    def test_dry_run_skips_publish_and_archive(self, tmp_dir):
        pipeline = DailyCalibrationPipeline(
            PipelineConfig(output_dir=str(tmp_dir), dry_run=True)
        )
        report = pipeline.run("2026-02-18")
        skipped = {s.name for s in report.steps if s.status == "skipped"}
        assert "publish" in skipped
        assert "archive" in skipped

    def test_failing_step_marks_pipeline_failed(self, tmp_dir):
        def bad_calibrate(context):
            raise RuntimeError("NLopt failed to converge")

        pipeline = DailyCalibrationPipeline(
            PipelineConfig(output_dir=str(tmp_dir), dry_run=True)
        )
        pipeline.register_step("calibrate", bad_calibrate)
        report = pipeline.run("2026-02-18")
        assert report.passed is False
        assert "calibrate" in report.failure_reason

    def test_steps_after_failure_are_skipped(self, tmp_dir):
        def bad_fetch(date, context):
            raise RuntimeError("Market data unavailable")

        pipeline = DailyCalibrationPipeline(
            PipelineConfig(output_dir=str(tmp_dir), dry_run=True)
        )
        pipeline.register_step("fetch_data", bad_fetch)
        report = pipeline.run("2026-02-18")
        # Every step after fetch_data should be skipped
        skipped = {s.name for s in report.steps if s.status == "skipped"}
        assert "validate_data" in skipped
        assert "calibrate" in skipped

    def test_report_persisted_to_disk(self, tmp_dir):
        pipeline = DailyCalibrationPipeline(
            PipelineConfig(output_dir=str(tmp_dir), dry_run=True)
        )
        report = pipeline.run("2026-02-18")
        reports_dir = tmp_dir / "reports" / "2026-02-18"
        json_files = list(reports_dir.glob("*.json"))
        assert len(json_files) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 6.7  TraderAdjustments
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraderAdjustments:

    def _ta(self, tmp_dir) -> TraderAdjustments:
        return TraderAdjustments(user="trader1", log_dir=tmp_dir / "tweaks")

    def test_atm_shift_up_increases_all_ivs(self, tmp_dir, flat_ivs):
        _, _, ivs = flat_ivs
        ta = self._ta(tmp_dir)
        adjusted = ta.allow_atm_shift(ivs, amount=0.01, reason="Test shift up")
        assert np.all(adjusted >= ivs)

    def test_atm_shift_raises_without_reason(self, tmp_dir, flat_ivs):
        _, _, ivs = flat_ivs
        ta = self._ta(tmp_dir)
        with pytest.raises(ValueError, match="non-empty reason"):
            ta.allow_atm_shift(ivs, amount=0.01, reason="")

    def test_skew_adjustment_linear_in_strike(self, tmp_dir, flat_ivs):
        strikes, _, ivs = flat_ivs
        ta = self._ta(tmp_dir)
        adjusted = ta.allow_skew_adjustment(ivs, strikes, atm_strike=100.0,
                                             amount=0.05, reason="Steepen")
        # Below ATM: Δσ < 0 (amount > 0, K < ATM → negative tilt)
        below_atm = strikes < 100.0
        above_atm = strikes > 100.0
        diff = adjusted - ivs
        assert np.all(diff[below_atm, :] < 0)
        assert np.all(diff[above_atm, :] > 0)

    def test_wing_adjustment_symmetric(self, tmp_dir, flat_ivs):
        strikes, _, ivs = flat_ivs
        ta = self._ta(tmp_dir)
        adjusted = ta.allow_wing_adjustment(ivs, strikes, atm_strike=100.0,
                                             amount=0.10, reason="Add curvature")
        diff = adjusted - ivs
        # Both wings should increase; ATM (closest to 100) changes least
        atm_idx = np.argmin(np.abs(strikes - 100.0))
        wing_idx = 0  # most OTM put
        assert diff[wing_idx, 0] > diff[atm_idx, 0]

    def test_preserve_arbitrage_repairs_calendar(self, tmp_dir, flat_ivs):
        strikes, maturities, ivs = flat_ivs
        ta = self._ta(tmp_dir)
        # Deliberately create a calendar violation: invert maturities 0 and 1
        bad_ivs = ivs.copy()
        bad_ivs[:, 1] = bad_ivs[:, 0] * 0.5   # lower IV at longer maturity
        repaired, is_clean = ta.preserve_arbitrage_free(bad_ivs, strikes, maturities)
        assert is_clean is False
        # After repair, total variance should be non-decreasing
        for k in range(len(strikes)):
            tv = repaired[k, :] ** 2 * maturities
            assert np.all(np.diff(tv) >= -1e-8)

    def test_adjustment_logged_to_history(self, tmp_dir, flat_ivs):
        _, _, ivs = flat_ivs
        ta = self._ta(tmp_dir)
        ta.allow_atm_shift(ivs, amount=0.005, reason="Morning fix")
        ta.allow_atm_shift(ivs, amount=-0.003, reason="Afternoon correction")
        assert len(ta.get_adjustment_history()) == 2

    def test_summary_contains_user_and_count(self, tmp_dir, flat_ivs):
        _, _, ivs = flat_ivs
        ta = self._ta(tmp_dir)
        ta.allow_atm_shift(ivs, amount=0.005, reason="Reason")
        s = ta.summary()
        assert "trader1" in s
        assert "1 adjustment" in s
