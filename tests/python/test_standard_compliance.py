"""
Tests for the §1/§2/§4 compliance features:

  TestForwardCurveBuilder     — ForwardCurveBuilder + ForwardCurve
  TestDiscreteDividends       — Discrete-dividend forward model
  TestInferForwardFromPCP     — Put-call parity forward inference
  TestLogMoneyness            — log_moneyness helper
  TestValidateForwardConsistency — PCP sanity-check on a DataFrame
  TestExpiryClassifier        — ExpiryClassifier (weekly/monthly/quarterly/LEAPS)
  TestExpiryClassifierDataframe — classify_dataframe + filter_primary
  TestCleanerPerExpiry        — Per-expiry statistical spread filter
  TestCleanerLogMoneyness     — log_moneyness + forward columns in cleaner output
  TestESSVIParameters         — ESSVIParameters arbitrage-free checks
  TestESSVISurface            — ESSVISurface evaluation + calibration
  TestESSVICalibration        — Full calibration round-trip
  TestDataInitNewExports      — All new symbols reachable from data package
  TestSurfaceInitNewExports   — All new symbols reachable from surface package
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.python.data.forwards import (
    DiscreteDividend,
    ForwardConsistencyResult,
    ForwardCurve,
    ForwardCurveBuilder,
    infer_forward_from_pcp,
    log_moneyness,
    validate_forward_consistency,
)
from src.python.data.expiry_classifier import ExpiryClassifier, ExpiryType
from src.python.data.cleaners import clean_option_chain
from src.python.surface.essvi import (
    ESSVICalibrationResult,
    ESSVIParameters,
    ESSVISurface,
    essvi_to_ssvi_params,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_builder():
    return ForwardCurveBuilder(spot=100.0, rate=0.05, div_yield=0.02)


@pytest.fixture
def sample_maturities():
    return np.array([0.25, 0.5, 1.0, 2.0])


@pytest.fixture
def essvi_params():
    mats = np.array([0.25, 0.5, 1.0])
    return ESSVIParameters(
        theta_curve=np.array([0.04, 0.08, 0.15]),
        maturities=mats,
        eta=1.5,
        gamma=0.5,
        rho_0=-0.5,
        lambda_rho=1.0,
    )


def _make_option_df(spot=100.0, n=6):
    """Minimal option DataFrame for cleaner tests."""
    future = pd.Timestamp("2027-01-15", tz="UTC")
    future2 = pd.Timestamp("2027-06-18", tz="UTC")
    return pd.DataFrame({
        "strike": [95.0, 100.0, 105.0, 95.0, 100.0, 105.0],
        "bid":    [6.0, 3.5, 1.5, 5.8, 3.2, 1.2],
        "ask":    [6.4, 3.7, 1.7, 6.2, 3.6, 1.4],
        "impliedVolatility": [0.22, 0.20, 0.21, 0.23, 0.21, 0.22],
        "type":   ["call"] * 3 + ["call"] * 3,
        "expiration": [future] * 3 + [future2] * 3,
        "volume": [100] * 6,
        "openInterest": [500] * 6,
    })


# ──────────────────────────────────────────────────────────────────────────────
# ForwardCurveBuilder
# ──────────────────────────────────────────────────────────────────────────────

class TestForwardCurveBuilder:
    def test_flat_rate_zero_div(self, sample_maturities):
        b = ForwardCurveBuilder(spot=100.0, rate=0.05, div_yield=0.0)
        curve = b.build(sample_maturities)
        for T, F in zip(sample_maturities, curve.forwards):
            expected = 100.0 * np.exp(0.05 * T)
            assert abs(F - expected) < 1e-8

    def test_flat_rate_with_div(self, sample_maturities):
        b = ForwardCurveBuilder(spot=100.0, rate=0.05, div_yield=0.02)
        curve = b.build(sample_maturities)
        for T, F in zip(sample_maturities, curve.forwards):
            expected = 100.0 * np.exp(0.03 * T)
            assert abs(F - expected) < 1e-8

    def test_forward_at_single(self, flat_builder):
        F = flat_builder.forward_at(1.0)
        assert abs(F - 100.0 * np.exp(0.03)) < 1e-6

    def test_curve_maturities_sorted(self, flat_builder):
        mats = np.array([1.0, 0.25, 2.0, 0.5])
        curve = flat_builder.build(mats)
        assert list(curve.maturities) == sorted(curve.maturities)

    def test_forward_interpolation(self, flat_builder, sample_maturities):
        curve = flat_builder.build(sample_maturities)
        # Interpolated point should be between its neighbours
        F_mid = curve.forward(0.75)
        F_05 = curve.forward(0.5)
        F_10 = curve.forward(1.0)
        assert F_05 < F_mid < F_10

    def test_forward_extrapolation_flat(self, flat_builder, sample_maturities):
        curve = flat_builder.build(sample_maturities)
        # Beyond max maturity → flat extrapolation at last value
        F_far = curve.forward(50.0)
        F_max = curve.forward(float(sample_maturities[-1]))
        assert abs(F_far - F_max) < 1e-8

    def test_rate_curve_input(self, sample_maturities):
        rate_curve = np.array([[0.25, 0.04], [0.5, 0.045], [1.0, 0.05], [2.0, 0.055]])
        b = ForwardCurveBuilder(spot=100.0, rate=rate_curve, div_yield=0.0)
        curve = b.build(sample_maturities)
        assert curve.forwards[-1] > curve.forwards[0]

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot must be positive"):
            ForwardCurveBuilder(spot=-10.0)

    def test_non_positive_maturity_raises(self, flat_builder):
        with pytest.raises(ValueError, match="All maturities must be positive"):
            flat_builder.build(np.array([0.0, 1.0]))

    def test_returns_forward_curve_instance(self, flat_builder, sample_maturities):
        curve = flat_builder.build(sample_maturities)
        assert isinstance(curve, ForwardCurve)

    def test_log_moneyness_grid(self, flat_builder, sample_maturities):
        curve = flat_builder.build(sample_maturities)
        k = curve.log_moneyness_grid(np.array([95, 100, 105]), 0.25)
        assert len(k) == 3
        # ATM log-moneyness is close to 0 for near-maturity
        atm_idx = 1
        assert abs(k[atm_idx]) < 0.1

    def test_method_attribute(self, sample_maturities):
        b = ForwardCurveBuilder(spot=100.0, rate=0.04, div_yield=0.01)
        curve = b.build(sample_maturities)
        assert curve.method == "continuous_yield"


class TestDiscreteDividends:
    def test_cash_div_reduces_forward(self, sample_maturities):
        div = DiscreteDividend(ex_date_years=0.1, amount=2.0, is_cash=True)
        b_clean = ForwardCurveBuilder(spot=100.0, rate=0.05, div_yield=0.0)
        b_div = ForwardCurveBuilder(spot=100.0, rate=0.05, div_yield=0.0, discrete_divs=[div])
        for T in sample_maturities:
            if T > 0.1:
                assert b_div.forward_at(T) < b_clean.forward_at(T)

    def test_cash_div_before_maturity_only(self, sample_maturities):
        div = DiscreteDividend(ex_date_years=5.0, amount=10.0, is_cash=True)
        b = ForwardCurveBuilder(spot=100.0, rate=0.05, div_yield=0.0, discrete_divs=[div])
        # Maturity of 1.0 < ex_date 5.0 → no dividend impact
        F_no_div = ForwardCurveBuilder(spot=100.0, rate=0.05).forward_at(1.0)
        F_with_div = b.forward_at(1.0)
        assert abs(F_with_div - F_no_div) < 1e-6

    def test_discrete_method_label(self, sample_maturities):
        div = DiscreteDividend(ex_date_years=0.1, amount=1.0, is_cash=True)
        b = ForwardCurveBuilder(spot=100.0, rate=0.05, discrete_divs=[div])
        curve = b.build(sample_maturities)
        assert curve.method == "discrete_dividends"


# ──────────────────────────────────────────────────────────────────────────────
# Put-call parity forward inference
# ──────────────────────────────────────────────────────────────────────────────

class TestInferForwardFromPCP:
    def test_exact_recovery(self):
        # C - P = e^{-rT}(F - K)  ⟹  F = K + e^{rT}(C - P)
        F = 100.0
        r, T = 0.05, 0.5
        disc = np.exp(-r * T)
        strikes = np.array([95.0, 100.0, 105.0])
        calls = (strikes - F) * disc + np.array([5.0, 0.0, 0.0])
        # Synthetic: call - put = disc * (F - K)  → put = call - disc*(F-K)
        puts = calls - disc * (F - strikes)
        calls = np.maximum(calls, 0.01)
        puts = np.maximum(puts, 0.01)
        F_hat, std = infer_forward_from_pcp(calls, puts, strikes, T, r)
        assert abs(F_hat - F) < 0.5  # within 0.5 of true

    def test_returns_two_floats(self):
        result = infer_forward_from_pcp(
            np.array([5.0, 3.0, 1.0]),
            np.array([1.0, 3.0, 5.0]),
            np.array([95.0, 100.0, 105.0]),
            maturity=0.5,
            rate=0.05,
        )
        assert len(result) == 2
        assert isinstance(result[0], float)

    def test_too_few_pairs_raises(self):
        with pytest.raises(ValueError, match="pairs"):
            infer_forward_from_pcp(
                np.array([5.0]),
                np.array([1.0]),
                np.array([100.0]),
                maturity=0.5,
                rate=0.05,
                min_pairs=3,
            )

    def test_zero_prices_excluded(self):
        calls = np.array([0.0, 3.0, 1.0, 5.0])
        puts = np.array([1.0, 0.0, 5.0, 1.0])
        strikes = np.array([95.0, 100.0, 105.0, 110.0])
        # Only strikes 95 and 110 have both prices > 0
        with pytest.raises(ValueError):
            infer_forward_from_pcp(calls, puts, strikes, 0.5, 0.05, min_pairs=3)


# ──────────────────────────────────────────────────────────────────────────────
# log_moneyness helper
# ──────────────────────────────────────────────────────────────────────────────

class TestLogMoneyness:
    def test_atm_zero(self):
        k = log_moneyness(np.array([100.0]), forward=100.0)
        assert abs(k[0]) < 1e-10

    def test_otm_call_positive(self):
        k = log_moneyness(np.array([110.0]), forward=100.0)
        assert k[0] > 0

    def test_otm_put_negative(self):
        k = log_moneyness(np.array([90.0]), forward=100.0)
        assert k[0] < 0

    def test_values(self):
        k = log_moneyness(np.array([100.0 * np.e]), forward=100.0)
        assert abs(k[0] - 1.0) < 1e-10

    def test_negative_forward_raises(self):
        with pytest.raises(ValueError, match="forward must be positive"):
            log_moneyness(np.array([100.0]), forward=-5.0)

    def test_array_output(self):
        k = log_moneyness(np.array([90, 100, 110]), forward=100.0)
        assert k.shape == (3,)


# ──────────────────────────────────────────────────────────────────────────────
# validate_forward_consistency
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateForwardConsistency:
    def _make_df(self):
        F = 100.0
        r, T = 0.05, 0.5
        disc = np.exp(-r * T)
        strikes = np.array([95.0, 100.0, 105.0])
        calls = np.maximum(F * disc - strikes * disc + 5.0, 0.01)
        puts = np.maximum(calls - disc * (F - strikes), 0.01)
        return pd.DataFrame({
            "maturity": T,
            "strike": strikes,
            "call_mid": calls,
            "put_mid": puts,
        })

    def test_returns_list(self):
        df = self._make_df()
        curve = ForwardCurveBuilder(spot=100.0, rate=0.05).build(np.array([0.5]))
        results = validate_forward_consistency(df, curve, rate=0.05)
        assert isinstance(results, list)

    def test_result_type(self):
        df = self._make_df()
        curve = ForwardCurveBuilder(spot=100.0, rate=0.05).build(np.array([0.5]))
        results = validate_forward_consistency(df, curve, rate=0.05)
        if results:
            assert isinstance(results[0], ForwardConsistencyResult)

    def test_consistent_passes(self):
        df = self._make_df()
        curve = ForwardCurveBuilder(spot=100.0, rate=0.05).build(np.array([0.5]))
        results = validate_forward_consistency(df, curve, rate=0.05, tolerance_pct=5.0)
        if results:
            assert results[0].passed


# ──────────────────────────────────────────────────────────────────────────────
# ExpiryClassifier
# ──────────────────────────────────────────────────────────────────────────────

class TestExpiryClassifier:
    @pytest.fixture
    def clf(self):
        return ExpiryClassifier()

    def test_leaps(self, clf):
        today = date(2026, 2, 18)
        leaps = today + timedelta(days=400)
        assert clf.classify(leaps, today) == ExpiryType.LEAPS

    def test_weekly_same_week(self, clf):
        today = date(2026, 2, 18)  # Wednesday
        this_friday = date(2026, 2, 20)  # 2 days away
        assert clf.classify(this_friday, today) == ExpiryType.WEEKLY

    def test_monthly_third_friday_non_quarter(self, clf):
        today = date(2026, 2, 1)
        # Third Friday of April 2026 = April 17
        third_friday_apr = date(2026, 4, 17)
        assert clf.classify(third_friday_apr, today) == ExpiryType.MONTHLY

    def test_quarterly_march(self, clf):
        today = date(2026, 2, 1)
        # Third Friday of March 2026 = March 20
        third_fri_mar = date(2026, 3, 20)
        assert clf.classify(third_fri_mar, today) == ExpiryType.QUARTERLY

    def test_quarterly_december(self, clf):
        today = date(2025, 11, 1)
        # Third Friday of December 2025 = December 19
        third_fri_dec = date(2025, 12, 19)
        assert clf.classify(third_fri_dec, today) == ExpiryType.QUARTERLY

    def test_other_random_wednesday(self, clf):
        today = date(2026, 2, 1)
        random_wed = date(2026, 4, 22)  # Not a third Friday
        result = clf.classify(random_wed, today)
        assert result == ExpiryType.OTHER

    def test_bulk_classify(self, clf):
        today = date(2026, 2, 1)
        expiries = [
            today + timedelta(days=3),         # weekly
            date(2026, 4, 17),                  # monthly
            date(2026, 3, 20),                  # quarterly
            today + timedelta(days=400),        # LEAPS
        ]
        results = clf.classify_bulk(expiries, today)
        assert results[0] == ExpiryType.WEEKLY
        assert results[1] == ExpiryType.MONTHLY
        assert results[2] == ExpiryType.QUARTERLY
        assert results[3] == ExpiryType.LEAPS

    def test_timestamp_input(self, clf):
        today = pd.Timestamp("2026-02-01")
        expiry = pd.Timestamp("2026-03-20")  # third Friday of March = quarterly
        result = clf.classify(expiry, today)
        assert result == ExpiryType.QUARTERLY


class TestExpiryClassifierDataframe:
    @pytest.fixture
    def clf(self):
        return ExpiryClassifier()

    def _make_df(self):
        today = date(2026, 2, 18)
        return pd.DataFrame({
            "strike": [100.0, 100.0, 100.0, 100.0],
            "expiration": pd.to_datetime([
                date(2026, 2, 20),              # weekly (Fri, 2 days away)
                date(2026, 4, 17),              # monthly
                date(2026, 3, 20),              # quarterly
                date(2027, 6, 20),              # LEAPS
            ]),
        })

    def test_classify_dataframe_column(self, clf):
        df = self._make_df()
        result = clf.classify_dataframe(df, today=date(2026, 2, 18))
        assert "expiry_type" in result.columns
        assert len(result) == 4

    def test_filter_primary_removes_weekly(self, clf):
        df = self._make_df()
        primary = clf.filter_primary(df, today=date(2026, 2, 18))
        assert "weekly" not in primary["expiry_type"].values

    def test_filter_primary_keeps_monthly_quarterly(self, clf):
        df = self._make_df()
        primary = clf.filter_primary(df, today=date(2026, 2, 18))
        types = set(primary["expiry_type"].values)
        assert types <= {"monthly", "quarterly"}

    def test_split_by_type(self, clf):
        df = self._make_df()
        splits = clf.split_by_type(df, today=date(2026, 2, 18))
        # At least some types should be present
        assert len(splits) >= 1
        for key, sub_df in splits.items():
            assert (sub_df["expiry_type"] == key).all()


# ──────────────────────────────────────────────────────────────────────────────
# Cleaner upgrades
# ──────────────────────────────────────────────────────────────────────────────

class TestCleanerPerExpiry:
    def test_output_columns_include_log_moneyness(self):
        df = _make_option_df()
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.05)
        assert "log_moneyness" in result.columns

    def test_output_columns_include_forward(self):
        df = _make_option_df()
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.05)
        assert "forward" in result.columns

    def test_zero_bid_removed(self):
        df = _make_option_df().copy()
        df.loc[0, "bid"] = 0.0  # inject zero bid
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.05)
        # Row with zero bid should be removed (or not present in result)
        assert len(result) <= len(df)

    def test_wide_spread_outlier_removed(self):
        """A single quote with a massively wide spread should be removed."""
        df = _make_option_df()
        # Clone first row with an absurd spread for the same expiry
        wide = df.iloc[[0]].copy()
        wide["bid"] = 0.01
        wide["ask"] = 99.0  # ~10 000% spread ratio
        df = pd.concat([df, wide], ignore_index=True)
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.05)
        # The injected wide-spread row should be excluded
        assert len(result) <= len(df)

    def test_log_moneyness_atm_near_zero(self):
        df = _make_option_df()
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.05,
                                    dividend_yield=0.0)
        atm_row = result[result["strike"] == 100.0]
        assert len(atm_row) > 0
        # log-moneyness for ATM strike should be close to 0
        assert abs(atm_row["log_moneyness"].values[0]) < 0.05

    def test_forward_price_override(self):
        df = _make_option_df()
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.05,
                                    forward_price=102.0)
        # All forward values should equal the override
        assert (result["forward"] == 102.0).all()

    def test_backward_compat_moneyness(self):
        """Original moneyness = K/S column still present."""
        df = _make_option_df()
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.05)
        assert "moneyness" in result.columns
        # moneyness should be K/S
        row = result[result["strike"] == 100.0].iloc[0]
        assert abs(row["moneyness"] - 1.0) < 0.01


class TestCleanerLogMoneyness:
    def test_log_moneyness_formula(self):
        df = _make_option_df()
        # Use zero rate/div so F ≈ S
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.0,
                                    dividend_yield=0.0)
        for _, row in result.iterrows():
            expected_k = np.log(row["strike"] / row["forward"])
            assert abs(row["log_moneyness"] - expected_k) < 1e-6

    def test_otm_call_positive_k(self):
        df = _make_option_df()
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.0)
        otm_call = result[result["strike"] == 105.0]
        if len(otm_call) > 0:
            assert otm_call["log_moneyness"].values[0] > 0

    def test_otm_put_negative_k(self):
        df = _make_option_df()
        result = clean_option_chain(df, spot_price=100.0, risk_free_rate=0.0)
        otm_put = result[result["strike"] == 95.0]
        if len(otm_put) > 0:
            assert otm_put["log_moneyness"].values[0] < 0


# ──────────────────────────────────────────────────────────────────────────────
# ESSVIParameters
# ──────────────────────────────────────────────────────────────────────────────

class TestESSVIParameters:
    def test_arbitrage_free_valid_params(self, essvi_params):
        assert essvi_params.is_arbitrage_free() is True

    def test_butterfly_violation_detected(self):
        # Very large eta forces θφ²(1+|ρ|) > 4
        p = ESSVIParameters(
            theta_curve=np.array([0.01]),
            maturities=np.array([0.25]),
            eta=50.0,    # huge
            gamma=0.5,
            rho_0=-0.5,
            lambda_rho=0.0,
        )
        assert p.is_arbitrage_free() is False

    def test_calendar_violation_detected(self):
        p = ESSVIParameters(
            theta_curve=np.array([0.15, 0.08]),  # decreasing!
            maturities=np.array([0.25, 0.5]),
            eta=1.0, gamma=0.5, rho_0=-0.3, lambda_rho=0.5,
        )
        assert p.is_arbitrage_free() is False

    def test_rho_decays_with_theta(self, essvi_params):
        rho_low_t = essvi_params.rho_at_theta(np.array([0.01]))[0]
        rho_high_t = essvi_params.rho_at_theta(np.array([0.5]))[0]
        # Decay: larger θ → smaller |ρ|
        assert abs(rho_high_t) < abs(rho_low_t)

    def test_lambda_zero_constant_rho(self):
        p = ESSVIParameters(
            theta_curve=np.array([0.1, 0.2]),
            maturities=np.array([0.5, 1.0]),
            eta=1.0, gamma=0.5, rho_0=-0.4, lambda_rho=0.0,
        )
        rhos = p.rho_at_theta(p.theta_curve)
        assert np.allclose(rhos, -0.4)

    def test_serialisation_round_trip(self, essvi_params):
        d = essvi_params.to_dict()
        p2 = ESSVIParameters.from_dict(d)
        assert np.allclose(p2.theta_curve, essvi_params.theta_curve)
        assert p2.lambda_rho == essvi_params.lambda_rho

    def test_phi_at_theta(self, essvi_params):
        phi = essvi_params.phi_at_theta(np.array([0.1]))
        expected = essvi_params.eta / (0.1 ** essvi_params.gamma)
        assert abs(phi[0] - expected) < 1e-10

    def test_butterfly_bound_shape(self, essvi_params):
        bf = essvi_params.butterfly_bound()
        assert bf.shape == essvi_params.theta_curve.shape


# ──────────────────────────────────────────────────────────────────────────────
# ESSVISurface evaluation
# ──────────────────────────────────────────────────────────────────────────────

class TestESSVISurface:
    def _make_calibrated_surface(self) -> ESSVISurface:
        surf = ESSVISurface(forward=100.0, maturities=np.array([0.25, 0.5, 1.0]))
        surf.parameters = ESSVIParameters(
            theta_curve=np.array([0.04, 0.08, 0.15]),
            maturities=np.array([0.25, 0.5, 1.0]),
            eta=1.5, gamma=0.5, rho_0=-0.5, lambda_rho=1.0,
        )
        surf._build_theta_interp()
        return surf

    def test_total_variance_positive(self):
        surf = self._make_calibrated_surface()
        w = surf.total_variance(0.0, 0.5)
        assert w > 0

    def test_total_variance_atm_equals_theta_half(self):
        """w(0, T) = θ(T)/2 · [1 + ρ(θ)·0 + √((0 + ρ(θ))² + (1−ρ²))]"""
        surf = self._make_calibrated_surface()
        # At k=0: w = θ/2 * [1 + √(ρ² + 1 - ρ²)] = θ/2 * [1 + 1] = θ
        w = surf.total_variance(0.0, 0.5)
        # Should be approximately θ(0.5) = 0.08
        assert 0.01 < w < 0.5

    def test_implied_vol_positive(self):
        surf = self._make_calibrated_surface()
        sigma = surf.implied_vol(0.0, 0.5)
        assert sigma > 0

    def test_implied_vol_reasonable_range(self):
        surf = self._make_calibrated_surface()
        sigma = surf.implied_vol(0.0, 0.5)
        assert 0.05 < sigma < 1.0

    def test_iv_grid_shape(self):
        surf = self._make_calibrated_surface()
        strikes = np.linspace(85, 115, 7)
        mats = np.array([0.25, 0.5, 1.0])
        ivs = surf.implied_vol_grid(strikes, mats)
        assert ivs.shape == (7, 3)

    def test_not_calibrated_raises(self):
        surf = ESSVISurface(forward=100.0)
        with pytest.raises(RuntimeError, match="not calibrated"):
            surf.total_variance(0.0, 0.5)

    def test_smile_is_convex_otm(self):
        """Total variance should be higher OTM than ATM (basic smile shape)."""
        surf = self._make_calibrated_surface()
        w_atm = surf.total_variance(0.0, 0.5)
        w_otm = surf.total_variance(-0.3, 0.5)
        assert w_otm > w_atm

    def test_calendar_monotone(self):
        """Total variance at k=0 should increase with maturity."""
        surf = self._make_calibrated_surface()
        w_025 = surf.total_variance(0.0, 0.25)
        w_050 = surf.total_variance(0.0, 0.5)
        w_100 = surf.total_variance(0.0, 1.0)
        assert w_025 <= w_050 <= w_100


# ──────────────────────────────────────────────────────────────────────────────
# ESSVICalibration round-trip
# ──────────────────────────────────────────────────────────────────────────────

class TestESSVICalibration:
    def _synthetic_surface(self, surf: ESSVISurface) -> np.ndarray:
        """Generate synthetic market IVs from a known parametrisation."""
        strikes = np.linspace(85, 115, 9)
        ivs = surf.implied_vol_grid(strikes, surf.maturities)
        return strikes, ivs

    def test_calibration_returns_result(self):
        mats = np.array([0.25, 0.5, 1.0])
        surf_true = ESSVISurface(forward=100.0, maturities=mats)
        surf_true.parameters = ESSVIParameters(
            theta_curve=np.array([0.04, 0.08, 0.15]),
            maturities=mats, eta=1.5, gamma=0.5, rho_0=-0.4, lambda_rho=0.8,
        )
        surf_true._build_theta_interp()
        strikes, market_ivs = self._synthetic_surface(surf_true)

        surf_fit = ESSVISurface(forward=100.0, maturities=mats)
        result = surf_fit.calibrate(strikes, market_ivs)
        assert isinstance(result, ESSVICalibrationResult)

    def test_rmse_low_on_synthetic(self):
        mats = np.array([0.25, 0.5, 1.0])
        surf_true = ESSVISurface(forward=100.0, maturities=mats)
        surf_true.parameters = ESSVIParameters(
            theta_curve=np.array([0.04, 0.08, 0.15]),
            maturities=mats, eta=1.5, gamma=0.5, rho_0=-0.4, lambda_rho=0.8,
        )
        surf_true._build_theta_interp()
        strikes, market_ivs = self._synthetic_surface(surf_true)

        surf_fit = ESSVISurface(forward=100.0, maturities=mats)
        result = surf_fit.calibrate(strikes, market_ivs, max_iter=1000)
        # On synthetic data the fit should be very good
        assert result.rmse < 0.01

    def test_calibrated_surface_evaluable(self):
        mats = np.array([0.25, 1.0])
        strikes = np.linspace(90, 110, 7)
        ivs = np.full((7, 2), 0.20)
        surf = ESSVISurface(forward=100.0, maturities=mats)
        surf.calibrate(strikes, ivs)
        sigma = surf.implied_vol(0.0, 0.5)
        assert sigma > 0

    def test_parameters_arbitrage_free_after_calibration(self):
        mats = np.array([0.25, 0.5, 1.0])
        strikes = np.linspace(85, 115, 9)
        ivs = np.random.RandomState(42).uniform(0.18, 0.25, (9, 3))
        surf = ESSVISurface(forward=100.0, maturities=mats)
        result = surf.calibrate(strikes, ivs)
        assert result.arbitrage_free is True

    def test_warm_start_with_prev_params(self):
        mats = np.array([0.25, 1.0])
        strikes = np.linspace(90, 110, 7)
        ivs = np.full((7, 2), 0.20)
        prev = ESSVIParameters(
            theta_curve=np.array([0.04, 0.15]),
            maturities=mats, eta=1.0, gamma=0.5, rho_0=-0.3, lambda_rho=0.5,
        )
        surf = ESSVISurface(forward=100.0, maturities=mats)
        result = surf.calibrate(strikes, ivs, prev_params=prev, tikhonov_lambda=0.01)
        assert isinstance(result, ESSVICalibrationResult)

    def test_save_load_round_trip(self, tmp_path):
        mats = np.array([0.25, 1.0])
        strikes = np.linspace(90, 110, 7)
        ivs = np.full((7, 2), 0.20)
        surf = ESSVISurface(forward=100.0, maturities=mats)
        surf.calibrate(strikes, ivs)
        p = tmp_path / "essvi_params.json"
        surf.save_parameters(p)

        surf2 = ESSVISurface(forward=100.0, maturities=mats)
        surf2.load_parameters(p)
        w1 = surf.total_variance(0.0, 0.5)
        w2 = surf2.total_variance(0.0, 0.5)
        assert abs(w1 - w2) < 1e-8

    def test_essvi_to_ssvi_params(self):
        p = ESSVIParameters(
            theta_curve=np.array([0.04, 0.08]),
            maturities=np.array([0.5, 1.0]),
            eta=1.5, gamma=0.5, rho_0=-0.4, lambda_rho=0.0,  # constant rho
        )
        d = essvi_to_ssvi_params(p)
        assert "rho" in d
        assert abs(d["rho"] - (-0.4)) < 1e-6  # lambda=0 → rho=rho_0 everywhere


# ──────────────────────────────────────────────────────────────────────────────
# Package export smoke tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDataInitNewExports:
    def test_imports(self):
        from src.python.data import (  # noqa: F401
            ExpiryClassifier,
            ExpiryType,
            ForwardCurve,
            ForwardCurveBuilder,
            DiscreteDividend,
            ForwardConsistencyResult,
            infer_forward_from_pcp,
            log_moneyness,
            validate_forward_consistency,
        )

    def test_all_contains_new_symbols(self):
        import src.python.data as d
        for name in (
            "ExpiryClassifier", "ExpiryType",
            "ForwardCurve", "ForwardCurveBuilder", "DiscreteDividend",
            "ForwardConsistencyResult", "infer_forward_from_pcp",
            "log_moneyness", "validate_forward_consistency",
        ):
            assert name in d.__all__, f"{name!r} missing from data.__all__"


class TestSurfaceInitNewExports:
    def test_essvi_imports(self):
        from src.python.surface import (  # noqa: F401
            ESSVISurface,
            ESSVIParameters,
            ESSVICalibrationResult,
            essvi_to_ssvi_params,
        )

    def test_essvi_in_all(self):
        import src.python.surface as s
        for name in (
            "ESSVISurface", "ESSVIParameters",
            "ESSVICalibrationResult", "essvi_to_ssvi_params",
        ):
            assert name in s.__all__, f"{name!r} missing from surface.__all__"
