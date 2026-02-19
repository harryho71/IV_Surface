"""
Tests for the Data Quality pipeline (data/loaders.py + data/quality.py).

Covers:
  TestMarketSnapshot    — loaders.MarketSnapshot helper methods
  TestLoadMarketJson    — load_market_json / load_raw_ticker
  TestLoadCsv           — load_iv_csv
  TestLoadPickle        — load_pickle
  TestStaleQuoteDetector
  TestOutlierDetector
  TestTermStructureChecker
  TestCoverageChecker
  TestDataQualityPipeline
  TestDataQualityReport
  TestDataInit          — public __init__.py exports
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.python.data.loaders import (
    MarketSnapshot,
    load_iv_csv,
    load_market_json,
    load_pickle,
    load_raw_ticker,
)
from src.python.data.quality import (
    CoverageChecker,
    DataQualityPipeline,
    DataQualityReport,
    OutlierDetector,
    StaleQuoteDetector,
    TermStructureChecker,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_df(
    maturities=(0.25, 0.5, 1.0),
    n_strikes=7,
    spot=100.0,
    base_iv=0.20,
) -> pd.DataFrame:
    """Build a clean synthetic IV DataFrame."""
    rows = []
    for mat in maturities:
        strikes = np.linspace(spot * 0.85, spot * 1.15, n_strikes)
        # slight skew so smile is not perfectly flat
        ivs = base_iv + 0.02 * (strikes - spot) / spot * (-1)
        ivs = np.clip(ivs, 0.05, 0.80)
        for k, v in zip(strikes, ivs):
            rows.append({"maturity": mat, "strike": float(k), "iv": float(v)})
    return pd.DataFrame(rows)


def _make_snapshot(ticker="TEST", spot=100.0) -> MarketSnapshot:
    mats = np.array([0.25, 0.5, 1.0])
    strikes_by_mat = {}
    ivs_by_mat = {}
    for m in mats:
        ks = np.linspace(spot * 0.85, spot * 1.15, 7)
        vs = 0.20 + 0.01 * np.arange(7)
        strikes_by_mat[float(m)] = ks
        ivs_by_mat[float(m)] = vs
    return MarketSnapshot(
        ticker=ticker,
        spot=spot,
        forward=None,
        timestamp="2025-01-01T00:00:00",
        maturities=mats,
        strikes_by_maturity=strikes_by_mat,
        ivs_by_maturity=ivs_by_mat,
        raw={},
    )


# ──────────────────────────────────────────────────────────────────────────────
# MarketSnapshot helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestMarketSnapshot:
    def test_n_maturities(self):
        snap = _make_snapshot()
        assert snap.n_maturities() == 3

    def test_n_options(self):
        snap = _make_snapshot()
        # 3 maturities × 7 strikes = 21
        assert snap.n_options() == 21

    def test_to_flat_dataframe_columns(self):
        snap = _make_snapshot()
        df = snap.to_flat_dataframe()
        assert {"maturity", "strike", "iv"}.issubset(df.columns)

    def test_to_flat_dataframe_shape(self):
        snap = _make_snapshot()
        df = snap.to_flat_dataframe()
        assert len(df) == 21

    def test_to_flat_dataframe_spot_column(self):
        snap = _make_snapshot(spot=123.45)
        df = snap.to_flat_dataframe()
        assert "spot" in df.columns
        assert (df["spot"] == 123.45).all()

    def test_maturities_sorted(self):
        snap = _make_snapshot()
        assert list(snap.maturities) == sorted(snap.maturities)


# ──────────────────────────────────────────────────────────────────────────────
# load_market_json
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadMarketJson:
    def _write_json(self, tmp_path: Path) -> Path:
        data = {
            "ticker": "TST",
            "spot": 200.0,
            "forward": None,
            "timestamp": "2025-06-01T09:30:00",
            "data": {
                "0.25": {
                    "strikes": [190.0, 195.0, 200.0, 205.0, 210.0],
                    "ivs": [0.22, 0.21, 0.20, 0.21, 0.22],
                    "types": ["call"] * 5,
                    "expiry": "2025-09-01",
                    "days": 92,
                },
                "0.5": {
                    "strikes": [185.0, 195.0, 200.0, 205.0, 215.0],
                    "ivs": [0.23, 0.22, 0.21, 0.22, 0.23],
                    "types": ["call"] * 5,
                    "expiry": "2025-12-01",
                    "days": 183,
                },
            },
        }
        p = tmp_path / "TST_market_data.json"
        p.write_text(json.dumps(data))
        return p

    def test_returns_snapshot(self, tmp_path):
        p = self._write_json(tmp_path)
        snap = load_market_json(p)
        assert isinstance(snap, MarketSnapshot)

    def test_ticker(self, tmp_path):
        p = self._write_json(tmp_path)
        snap = load_market_json(p)
        assert snap.ticker == "TST"

    def test_spot(self, tmp_path):
        p = self._write_json(tmp_path)
        snap = load_market_json(p)
        assert snap.spot == 200.0

    def test_maturities_count(self, tmp_path):
        p = self._write_json(tmp_path)
        snap = load_market_json(p)
        assert snap.n_maturities() == 2

    def test_strikes_shape(self, tmp_path):
        p = self._write_json(tmp_path)
        snap = load_market_json(p)
        assert snap.n_options() == 10  # 5 + 5

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_market_json(tmp_path / "nonexistent.json")

    def test_load_raw_ticker_convenience(self, tmp_path):
        self._write_json(tmp_path)
        snap = load_raw_ticker("TST", data_dir=tmp_path)
        assert snap.ticker == "TST"

    def test_load_raw_ticker_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_ticker("ZZZ", data_dir=tmp_path)

    def test_real_spy_file_if_exists(self):
        """Optional smoke-test against the bundled SPY data file."""
        p = Path("data/raw/SPY_market_data.json")
        if not p.exists():
            pytest.skip("SPY data file not present")
        snap = load_market_json(p)
        assert snap.ticker == "SPY"
        assert snap.spot > 0
        assert snap.n_maturities() >= 1


# ──────────────────────────────────────────────────────────────────────────────
# load_iv_csv
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadCsv:
    def test_basic_load(self, tmp_path):
        df_orig = _make_df()
        p = tmp_path / "test.csv"
        df_orig.to_csv(p, index=False)
        df = load_iv_csv(p)
        assert set(df.columns) >= {"maturity", "strike", "iv"}
        assert len(df) == len(df_orig)

    def test_custom_column_names(self, tmp_path):
        df_orig = pd.DataFrame({
            "T": [0.25, 0.25, 0.5],
            "K": [95.0, 100.0, 100.0],
            "sigma": [0.20, 0.18, 0.22],
        })
        p = tmp_path / "custom.csv"
        df_orig.to_csv(p, index=False)
        df = load_iv_csv(p, maturity_col="T", strike_col="K", iv_col="sigma")
        assert set(df.columns) >= {"maturity", "strike", "iv"}

    def test_semicolon_separator(self, tmp_path):
        p = tmp_path / "semi.csv"
        p.write_text("maturity;strike;iv\n0.25;100.0;0.20\n0.5;100.0;0.22\n")
        df = load_iv_csv(p, sep=";")
        assert len(df) == 2

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_iv_csv(tmp_path / "missing.csv")


# ──────────────────────────────────────────────────────────────────────────────
# load_pickle
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadPickle:
    def test_basic_load(self, tmp_path):
        df = _make_df()
        p = tmp_path / "test.pkl"
        df.to_pickle(p)
        df2 = load_pickle(p)
        pd.testing.assert_frame_equal(df, df2)

    def test_not_dataframe_raises(self, tmp_path):
        p = tmp_path / "bad.pkl"
        with open(p, "wb") as f:
            pickle.dump({"not": "a dataframe"}, f)
        with pytest.raises(TypeError):
            load_pickle(p)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pickle(tmp_path / "nope.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# StaleQuoteDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestStaleQuoteDetector:
    def test_clean_data_passes(self):
        df = _make_df()
        result = StaleQuoteDetector().check(df)
        assert result["passed"] is True
        assert result["n_stale_runs"] == 0
        assert result["n_flat_slices"] == 0

    def test_flat_slice_detected(self):
        df = _make_df()
        # force one slice to be completely flat
        mask = df["maturity"] == 0.25
        df.loc[mask, "iv"] = 0.20
        result = StaleQuoteDetector(sd_threshold=1e-4).check(df)
        assert result["n_flat_slices"] >= 1
        assert 0.25 in result["flat_maturities"]

    def test_constant_run_detected(self):
        df = _make_df()
        mask = df["maturity"] == 0.5
        idx = df[mask].index[:4]
        df.loc[idx, "iv"] = 0.21  # 4-in-a-row identical
        result = StaleQuoteDetector(min_run=3).check(df)
        # Either the run or flat-slice test should fire
        assert result["n_stale_runs"] > 0 or result["n_flat_slices"] > 0

    def test_stale_fraction_range(self):
        df = _make_df()
        result = StaleQuoteDetector().check(df)
        assert 0.0 <= result["stale_fraction"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# OutlierDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestOutlierDetector:
    def test_clean_data_passes(self):
        df = _make_df()
        result = OutlierDetector().check(df)
        assert result["passed"] is True
        assert result["n_outliers"] == 0

    def test_single_outlier_detected(self):
        df = _make_df(n_strikes=15)
        # inject a massive outlier in one slice
        idx = df[df["maturity"] == 0.25].index[0]
        df.loc[idx, "iv"] = 5.0   # impossible vol
        result = OutlierDetector(z_threshold=2.5, iqr_multiplier=2.0).check(df)
        assert result["n_outliers"] >= 1
        assert idx in result["outlier_indices"]

    def test_by_maturity_dict(self):
        df = _make_df()
        result = OutlierDetector().check(df)
        # Keys should be maturities (or empty dict)
        assert isinstance(result["by_maturity"], dict)

    def test_outlier_fraction_range(self):
        df = _make_df()
        result = OutlierDetector().check(df)
        assert 0.0 <= result["outlier_fraction"] <= 1.0

    def test_too_few_points_skipped(self):
        df = pd.DataFrame({
            "maturity": [0.25, 0.25, 0.25],
            "strike": [95.0, 100.0, 105.0],
            "iv": [0.20, 0.19, 5.0],  # obvious outlier but only 3 rows
        })
        result = OutlierDetector().check(df)
        # slice has < 4 rows, detector skips it
        assert result["n_outliers"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# TermStructureChecker
# ──────────────────────────────────────────────────────────────────────────────

class TestTermStructureChecker:
    def test_normal_term_structure_passes(self):
        df = _make_df()
        # make ATM vol increase with maturity
        for i, mat in enumerate([0.25, 0.5, 1.0]):
            df.loc[df["maturity"] == mat, "iv"] += i * 0.02
        result = TermStructureChecker().check(df, spot=100.0)
        assert result["n_inversions"] == 0

    def test_inversion_detected(self):
        df = _make_df()
        # force ATM vol to drop sharply from T=0.5 to T=1.0
        df.loc[df["maturity"] == 1.0, "iv"] = 0.05
        df.loc[df["maturity"] == 0.5, "iv"] = 0.30
        result = TermStructureChecker(inversion_tolerance=0.02).check(df, spot=100.0)
        assert result["n_inversions"] >= 1

    def test_atm_ivs_list_length(self):
        df = _make_df()
        result = TermStructureChecker().check(df, spot=100.0)
        assert len(result["atm_ivs"]) == 3  # 3 maturities

    def test_jump_detected(self):
        df = _make_df()
        df.loc[df["maturity"] == 1.0, "iv"] = 0.60  # huge jump from 0.5
        result = TermStructureChecker(jump_threshold=0.10).check(df, spot=100.0)
        assert result["n_jumps"] >= 1

    def test_passed_flag_type(self):
        df = _make_df()
        result = TermStructureChecker().check(df, spot=100.0)
        assert isinstance(result["passed"], bool)


# ──────────────────────────────────────────────────────────────────────────────
# CoverageChecker
# ──────────────────────────────────────────────────────────────────────────────

class TestCoverageChecker:
    def test_full_coverage_passes(self):
        df = _make_df(n_strikes=8)
        result = CoverageChecker(min_strikes_per_slice=5).check(df)
        assert result["passed"] is True
        assert result["coverage_score"] == pytest.approx(1.0)

    def test_nan_iv_reduces_coverage(self):
        df = _make_df()
        df.iloc[:5, df.columns.get_loc("iv")] = np.nan
        result = CoverageChecker().check(df)
        assert result["coverage_score"] < 1.0

    def test_sparse_slice_flagged(self):
        df = _make_df()
        # keep only 2 strikes in one maturity slice
        mask = df["maturity"] == 0.25
        idx_to_drop = df[mask].index[2:]
        df = df.drop(idx_to_drop)
        result = CoverageChecker(min_strikes_per_slice=5).check(df)
        assert result["n_sparse_slices"] >= 1
        assert 0.25 in result["sparse_slices"]

    def test_too_few_maturities_fails(self):
        df = _make_df(maturities=(0.25,))
        result = CoverageChecker(min_maturities=3).check(df)
        assert result["passed"] is False

    def test_strikes_by_maturity_dict(self):
        df = _make_df()
        result = CoverageChecker().check(df)
        assert isinstance(result["strikes_by_maturity"], dict)
        assert len(result["strikes_by_maturity"]) == 3


# ──────────────────────────────────────────────────────────────────────────────
# DataQualityPipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestDataQualityPipeline:
    def test_returns_report(self):
        df = _make_df()
        report = DataQualityPipeline().run(df, spot=100.0)
        assert isinstance(report, DataQualityReport)

    def test_clean_data_score_high(self):
        df = _make_df(n_strikes=10, maturities=(0.25, 0.5, 1.0, 2.0))
        report = DataQualityPipeline().run(df, spot=100.0)
        assert report.score >= 75.0

    def test_clean_data_passes(self):
        df = _make_df(n_strikes=10)
        report = DataQualityPipeline(pass_threshold=70.0).run(df, spot=100.0)
        assert report.passed is True

    def test_score_range(self):
        df = _make_df()
        report = DataQualityPipeline().run(df, spot=100.0)
        assert 0.0 <= report.score <= 100.0

    def test_grade_a_for_clean_data(self):
        df = _make_df(n_strikes=15, maturities=(0.25, 0.5, 1.0, 2.0))
        report = DataQualityPipeline().run(df, spot=100.0)
        assert report.grade in ("A", "B")

    def test_bad_data_fails(self):
        # Two maturities (below min), many NaN IVs, giant outliers
        df = pd.DataFrame({
            "maturity": [0.25] * 5 + [0.5] * 3,
            "strike": list(np.linspace(90, 110, 5)) + list(np.linspace(90, 110, 3)),
            "iv": [np.nan] * 5 + [5.0, 0.001, 0.20],
        })
        report = DataQualityPipeline(pass_threshold=75.0).run(df, spot=100.0)
        assert report.passed is False

    def test_implied_volatility_column_alias(self):
        df = _make_df()
        df = df.rename(columns={"iv": "implied_volatility"})
        report = DataQualityPipeline().run(df, spot=100.0)
        assert isinstance(report, DataQualityReport)

    def test_spot_inferred_from_column(self):
        df = _make_df()
        df["spot"] = 100.0
        report = DataQualityPipeline().run(df)  # no spot kwarg
        assert isinstance(report, DataQualityReport)

    def test_recommendations_not_empty(self):
        df = _make_df()
        report = DataQualityPipeline().run(df, spot=100.0)
        assert len(report.recommendations) >= 1

    def test_n_total(self):
        df = _make_df()
        report = DataQualityPipeline().run(df, spot=100.0)
        assert report.n_total == len(df)

    def test_n_valid_leq_n_total(self):
        df = _make_df()
        df.iloc[0, df.columns.get_loc("iv")] = np.nan
        report = DataQualityPipeline().run(df, spot=100.0)
        assert report.n_valid <= report.n_total


# ──────────────────────────────────────────────────────────────────────────────
# DataQualityReport
# ──────────────────────────────────────────────────────────────────────────────

class TestDataQualityReport:
    def _make_report(self) -> DataQualityReport:
        df = _make_df()
        return DataQualityPipeline().run(df, spot=100.0)

    def test_summary_is_string(self):
        report = self._make_report()
        assert isinstance(report.summary(), str)

    def test_summary_contains_score(self):
        report = self._make_report()
        assert str(int(report.score)) in report.summary() or "Score=" in report.summary()

    def test_summary_contains_pass_fail(self):
        report = self._make_report()
        assert "PASS" in report.summary() or "FAIL" in report.summary()

    def test_raw_checks_keys(self):
        report = self._make_report()
        assert set(report.raw_checks.keys()) == {
            "coverage", "stale", "outliers", "term_structure"
        }

    def test_passed_consistent_with_score(self):
        report = self._make_report()
        expected_pass = report.score >= report.pass_threshold
        assert report.passed == expected_pass


# ──────────────────────────────────────────────────────────────────────────────
# Public __init__.py exports
# ──────────────────────────────────────────────────────────────────────────────

class TestDataInit:
    def test_imports_from_package(self):
        from src.python.data import (  # noqa: F401
            DataQualityPipeline,
            DataQualityReport,
            MarketSnapshot,
            OptionDataValidator,
            clean_option_chain,
            fetch_from_csv,
            load_market_json,
            load_raw_ticker,
        )

    def test_quality_classes_in_all(self):
        import src.python.data as d

        for name in (
            "DataQualityPipeline",
            "DataQualityReport",
            "StaleQuoteDetector",
            "OutlierDetector",
            "TermStructureChecker",
            "CoverageChecker",
        ):
            assert name in d.__all__

    def test_loader_symbols_in_all(self):
        import src.python.data as d

        for name in (
            "MarketSnapshot",
            "load_market_json",
            "load_iv_csv",
            "load_pickle",
            "load_raw_ticker",
        ):
            assert name in d.__all__
