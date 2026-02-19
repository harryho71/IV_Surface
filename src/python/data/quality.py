"""
Data Quality Pipeline — brings Data Quality from 75% → 95%.

Adds the four missing components on top of the existing validators.py / cleaners.py:

  StaleQuoteDetector      — flags IVs that are suspiciously constant across
                            strikes or that appear as copy-paste duplicates.
  OutlierDetector         — z-score IQR-based IV outlier detection per
                            maturity slice.
  TermStructureChecker    — validates ATM IV is broadly non-decreasing with
                            maturity and detects extreme inversions.
  CoverageChecker         — measures completeness of the (strike × maturity)
                            grid and flags sparse slices.

  DataQualityPipeline     — orchestrates all four checks plus the existing
                            OptionDataValidator, producing a DataQualityReport.

  DataQualityReport       — overall score (0–100), per-check breakdowns,
                            severity flags, and actionable recommendations.

Usage::

    from src.python.data.quality import DataQualityPipeline
    from src.python.data.loaders import load_raw_ticker

    snap    = load_raw_ticker("SPY")
    df      = snap.to_flat_dataframe()
    report  = DataQualityPipeline().run(df, spot=snap.spot)
    print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Individual checkers
# ──────────────────────────────────────────────────────────────────────────────

class StaleQuoteDetector:
    """
    Detects suspiciously constant (stale) implied-volatility values within a
    single snapshot — a sign of copy-paste fill, market-maker withdrawal, or
    data-vendor staleness.

    Two tests are applied:
    1. **Constant-run test** — if ≥ *min_run* consecutive strikes in the same
       maturity slice share the exact same IV, the entire run is flagged.
    2. **Flat-slice test** — if the standard deviation of all IVs in a
       maturity slice is below *sd_threshold* the slice is flagged as flat.

    Args:
        min_run:      Minimum length of a same-value run to flag. Default 3.
        sd_threshold: Minimum acceptable IV std-dev within a slice. Default 1e-4.
    """

    def __init__(self, min_run: int = 3, sd_threshold: float = 1e-4) -> None:
        self.min_run = min_run
        self.sd_threshold = sd_threshold

    def check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Args:
            df: DataFrame with columns ``maturity``, ``strike``, ``iv``.

        Returns:
            dict with keys:
              - ``n_stale_runs``        — total stale (strike, maturity) pairs from run test
              - ``n_flat_slices``       — maturities with suspiciously flat smile
              - ``flat_maturities``     — list of affected maturities
              - ``stale_fraction``      — fraction of data points flagged
              - ``passed``              — True if no staleness detected
        """
        stale_mask = pd.Series(False, index=df.index)
        flat_maturities: List[float] = []

        for mat, slice_df in df.groupby("maturity"):
            ivs = slice_df["iv"].values

            # Flat-slice test
            if ivs.std() < self.sd_threshold:
                flat_maturities.append(float(mat))
                stale_mask.loc[slice_df.index] = True
                continue

            # Constant-run test (sorted by strike)
            sorted_ivs = slice_df.sort_values("strike")["iv"].values
            run = 1
            for i in range(1, len(sorted_ivs)):
                if sorted_ivs[i] == sorted_ivs[i - 1]:
                    run += 1
                    if run >= self.min_run:
                        stale_mask.loc[slice_df.index] = True
                else:
                    run = 1

        n_stale = int(stale_mask.sum())
        return {
            "n_stale_runs": n_stale,
            "n_flat_slices": len(flat_maturities),
            "flat_maturities": flat_maturities,
            "stale_fraction": n_stale / max(len(df), 1),
            "passed": n_stale == 0 and len(flat_maturities) == 0,
        }


class OutlierDetector:
    """
    Flags IV outliers within each maturity slice using a hybrid
    IQR + z-score approach.

    A point is an outlier if **both** conditions hold:
    - ``|z-score| > z_threshold`` (default 3.5)
    - ``deviation > iqr_multiplier × IQR`` (default 3.0)

    The dual condition reduces false positives from naturally skewed smiles.

    Args:
        z_threshold:     Standard-deviation multiplier for z-score. Default 3.5.
        iqr_multiplier:  IQR multiplier for Tukey fence. Default 3.0.
    """

    def __init__(self, z_threshold: float = 3.5, iqr_multiplier: float = 3.0) -> None:
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Args:
            df: DataFrame with columns ``maturity``, ``strike``, ``iv``.

        Returns:
            dict with keys:
              - ``n_outliers``      — total outlier data points
              - ``outlier_fraction``— fraction of data flagged
              - ``outlier_indices`` — list of df index labels for outliers
              - ``by_maturity``     — dict mapping maturity → n_outliers
              - ``passed``          — True if no outliers detected
        """
        outlier_indices: List[Any] = []
        by_maturity: Dict[float, int] = {}

        for mat, slice_df in df.groupby("maturity"):
            ivs = slice_df["iv"].values
            if len(ivs) < 4:
                continue

            q1, q3 = np.percentile(ivs, [25, 75])
            iqr = q3 - q1
            fence_lo = q1 - self.iqr_multiplier * iqr
            fence_hi = q3 + self.iqr_multiplier * iqr

            mean, std = ivs.mean(), ivs.std()
            if std < 1e-10:
                continue

            z = np.abs((ivs - mean) / std)
            iqr_flag = (ivs < fence_lo) | (ivs > fence_hi)
            z_flag = z > self.z_threshold
            both_flag = iqr_flag & z_flag

            flagged_indices = slice_df.index[both_flag].tolist()
            outlier_indices.extend(flagged_indices)
            by_maturity[float(mat)] = int(both_flag.sum())

        return {
            "n_outliers": len(outlier_indices),
            "outlier_fraction": len(outlier_indices) / max(len(df), 1),
            "outlier_indices": outlier_indices,
            "by_maturity": by_maturity,
            "passed": len(outlier_indices) == 0,
        }


class TermStructureChecker:
    """
    Validates ATM IV term structure for consistency.

    Checks:
    1. **Monotonicity** — ATM IV should be broadly non-decreasing with
       maturity (calendar-spread arbitrage check at the ATM level).
       Allows a configurable inversion tolerance (short end can invert).
    2. **Jump magnitude** — flags suspiciously large ATM IV changes between
       adjacent maturities (> *jump_threshold* in absolute vol).

    Args:
        inversion_tolerance: Maximum allowed downward ATM IV step before
                             flagging as a violation. Default 0.02 (2 vol pts).
        jump_threshold:      Maximum allowed upward step before flagging as a
                             suspicious jump. Default 0.10 (10 vol pts).
        atm_band:            Relative moneyness band to define "ATM".
                             Default 0.02 (±2% around spot).
    """

    def __init__(
        self,
        inversion_tolerance: float = 0.02,
        jump_threshold: float = 0.10,
        atm_band: float = 0.02,
    ) -> None:
        self.inversion_tolerance = inversion_tolerance
        self.jump_threshold = jump_threshold
        self.atm_band = atm_band

    def check(self, df: pd.DataFrame, spot: float) -> Dict[str, Any]:
        """
        Args:
            df:   DataFrame with columns ``maturity``, ``strike``, ``iv``.
            spot: Current spot price.

        Returns:
            dict with keys:
              - ``atm_ivs``           — list of (maturity, atm_iv) pairs
              - ``n_inversions``      — number of downward violations
              - ``inversion_maturities`` — list of (mat_from, mat_to) pairs
              - ``n_jumps``           — number of large upward jumps
              - ``jump_maturities``   — list of (mat_from, mat_to) pairs
              - ``passed``            — True if no violations
        """
        # Compute ATM IV per maturity
        atm_ivs: List[Tuple[float, float]] = []
        for mat in sorted(df["maturity"].unique()):
            slice_df = df[df["maturity"] == mat]
            atm_mask = (
                (slice_df["strike"] >= spot * (1 - self.atm_band)) &
                (slice_df["strike"] <= spot * (1 + self.atm_band))
            )
            atm_slice = slice_df[atm_mask]
            if atm_slice.empty:
                # Fall back to strike closest to spot
                idx = (slice_df["strike"] - spot).abs().idxmin()
                atm_iv = float(slice_df.loc[idx, "iv"])
            else:
                atm_iv = float(atm_slice["iv"].mean())
            atm_ivs.append((float(mat), atm_iv))

        inversions: List[Tuple[float, float]] = []
        jumps: List[Tuple[float, float]] = []
        for i in range(1, len(atm_ivs)):
            m0, iv0 = atm_ivs[i - 1]
            m1, iv1 = atm_ivs[i]
            delta = iv1 - iv0
            if delta < -self.inversion_tolerance:
                inversions.append((m0, m1))
            if delta > self.jump_threshold:
                jumps.append((m0, m1))

        return {
            "atm_ivs": atm_ivs,
            "n_inversions": len(inversions),
            "inversion_maturities": inversions,
            "n_jumps": len(jumps),
            "jump_maturities": jumps,
            "passed": len(inversions) == 0 and len(jumps) == 0,
        }


class CoverageChecker:
    """
    Measures completeness of the (strike × maturity) grid.

    Computes:
    - Overall coverage score (fraction of non-NaN IV points vs expected grid).
    - Per-maturity coverage (fraction of strikes with valid IVs).
    - Flags sparse slices (< *min_strikes_per_slice* valid strikes).

    Args:
        min_strikes_per_slice: Minimum strikes required per maturity.
                               Default 5.
        min_maturities:        Minimum number of maturity slices.
                               Default 3.
    """

    def __init__(
        self,
        min_strikes_per_slice: int = 5,
        min_maturities: int = 3,
    ) -> None:
        self.min_strikes = min_strikes_per_slice
        self.min_maturities = min_maturities

    def check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Args:
            df: DataFrame with columns ``maturity``, ``strike``, ``iv``.

        Returns:
            dict with keys:
              - ``n_maturities``          — number of maturity slices
              - ``n_total_points``        — total (strike, maturity) pairs
              - ``n_valid_points``        — non-NaN IV points
              - ``coverage_score``        — fraction of valid points (0–1)
              - ``sparse_slices``         — maturities with < min_strikes
              - ``n_sparse_slices``       — count of sparse slices
              - ``passed``                — True if coverage ≥ 0.90 and no sparse slices
        """
        n_maturities = df["maturity"].nunique()
        n_total = len(df)
        n_valid = int(df["iv"].notna().sum())
        coverage = n_valid / max(n_total, 1)

        sparse: List[float] = []
        by_maturity: Dict[float, int] = {}
        for mat, slice_df in df.groupby("maturity"):
            n_valid_slice = int(slice_df["iv"].notna().sum())
            by_maturity[float(mat)] = n_valid_slice
            if n_valid_slice < self.min_strikes:
                sparse.append(float(mat))

        passed = (
            coverage >= 0.90
            and len(sparse) == 0
            and n_maturities >= self.min_maturities
        )
        return {
            "n_maturities": n_maturities,
            "n_total_points": n_total,
            "n_valid_points": n_valid,
            "coverage_score": coverage,
            "strikes_by_maturity": by_maturity,
            "sparse_slices": sparse,
            "n_sparse_slices": len(sparse),
            "passed": passed,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataQualityReport:
    """
    Aggregated data quality report produced by :class:`DataQualityPipeline`.

    Attributes:
        score:           Overall quality score 0–100.
        grade:           'A' (≥90) | 'B' (≥75) | 'C' (≥60) | 'D' (< 60).
        passed:          True when score ≥ pass_threshold.
        pass_threshold:  Minimum score for "PASS" (default 75).
        n_total:         Total input data points.
        n_valid:         Points passing basic validation.
        coverage:        Coverage check results dict.
        stale:           Stale-quote check results dict.
        outliers:        Outlier check results dict.
        term_structure:  Term-structure check results dict.
        recommendations: Actionable strings for remediation.
        raw_checks:      Full check output dicts keyed by check name.
    """

    score: float
    grade: str
    passed: bool
    pass_threshold: float
    n_total: int
    n_valid: int
    coverage: Dict[str, Any]
    stale: Dict[str, Any]
    outliers: Dict[str, Any]
    term_structure: Dict[str, Any]
    recommendations: List[str]
    raw_checks: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"DataQualityReport | Score={self.score:.1f}/100 Grade={self.grade} "
            f"{status} | "
            f"coverage={self.coverage['coverage_score'] * 100:.0f}% "
            f"stale={self.stale['stale_fraction'] * 100:.1f}% "
            f"outliers={self.outliers['n_outliers']} "
            f"ts_inversions={self.term_structure['n_inversions']} "
            f"| n={self.n_total} valid={self.n_valid}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class DataQualityPipeline:
    """
    End-to-end data quality pipeline.

    Runs four checks in sequence and aggregates results into a
    :class:`DataQualityReport` with an overall 0–100 score.

    Scoring weights (sum to 100):
      - Coverage:       30 pts — completeness of the grid
      - Freshness:      25 pts — absence of stale quotes
      - Outliers:       25 pts — absence of IV outliers
      - Term structure: 20 pts — ATM IV consistency across maturities

    Args:
        pass_threshold:         Minimum score for "PASS". Default 75.
        stale_min_run:          Passed to :class:`StaleQuoteDetector`.
        stale_sd_threshold:     Passed to :class:`StaleQuoteDetector`.
        outlier_z:              Passed to :class:`OutlierDetector`.
        outlier_iqr:            Passed to :class:`OutlierDetector`.
        ts_inversion_tolerance: Passed to :class:`TermStructureChecker`.
        ts_jump_threshold:      Passed to :class:`TermStructureChecker`.
        coverage_min_strikes:   Passed to :class:`CoverageChecker`.
        coverage_min_maturities: Passed to :class:`CoverageChecker`.

    Example::

        report = DataQualityPipeline().run(df, spot=100.0)
        print(report.summary())
        if not report.passed:
            for rec in report.recommendations:
                print(" •", rec)
    """

    # Scoring weights
    _W_COVERAGE = 30.0
    _W_FRESHNESS = 25.0
    _W_OUTLIERS = 25.0
    _W_TERM_STRUCT = 20.0

    def __init__(
        self,
        *,
        pass_threshold: float = 75.0,
        stale_min_run: int = 3,
        stale_sd_threshold: float = 1e-4,
        outlier_z: float = 3.5,
        outlier_iqr: float = 3.0,
        ts_inversion_tolerance: float = 0.02,
        ts_jump_threshold: float = 0.10,
        coverage_min_strikes: int = 5,
        coverage_min_maturities: int = 3,
    ) -> None:
        self.pass_threshold = pass_threshold
        self._stale = StaleQuoteDetector(stale_min_run, stale_sd_threshold)
        self._outlier = OutlierDetector(outlier_z, outlier_iqr)
        self._ts = TermStructureChecker(ts_inversion_tolerance, ts_jump_threshold)
        self._coverage = CoverageChecker(coverage_min_strikes, coverage_min_maturities)

    # ── main entry point ───────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        spot: Optional[float] = None,
    ) -> DataQualityReport:
        """
        Run the full data quality pipeline on *df*.

        Args:
            df:   DataFrame with at minimum columns ``maturity``, ``strike``,
                  ``iv``.  May also have ``spot``, ``bid``, ``ask``, etc.
            spot: Spot price used for ATM detection.  If not provided, uses
                  ``df['spot'].iloc[0]`` if available, else median strike.

        Returns:
            :class:`DataQualityReport`
        """
        df = df.copy()
        # Normalise iv column name
        if "iv" not in df.columns and "implied_volatility" in df.columns:
            df = df.rename(columns={"implied_volatility": "iv"})

        n_total = len(df)

        # Basic validity filter (non-NaN, positive values)
        valid_mask = (
            df["iv"].notna() &
            df["strike"].notna() &
            df["maturity"].notna() &
            (df["iv"] > 0) &
            (df["strike"] > 0) &
            (df["maturity"] > 0)
        )
        df_valid = df[valid_mask].copy()
        n_valid = len(df_valid)

        if spot is None:
            if "spot" in df.columns and df["spot"].notna().any():
                spot = float(df["spot"].dropna().iloc[0])
            else:
                spot = float(df_valid["strike"].median()) if not df_valid.empty else 1.0

        # Run checks
        cov_result = self._coverage.check(df_valid)
        stale_result = self._stale.check(df_valid)
        outlier_result = self._outlier.check(df_valid)
        ts_result = self._ts.check(df_valid, spot=spot)

        # Score each check 0–1 then weight
        cov_score = self._score_coverage(cov_result)
        fresh_score = self._score_freshness(stale_result)
        out_score = self._score_outliers(outlier_result)
        ts_score = self._score_term_structure(ts_result)

        raw_total = (
            cov_score * self._W_COVERAGE
            + fresh_score * self._W_FRESHNESS
            + out_score * self._W_OUTLIERS
            + ts_score * self._W_TERM_STRUCT
        )

        # Apply validity-ratio multiplier: data with ≥ 50% invalid rows gets
        # progressively lower scores (50% valid → 1.0×, 0% valid → 0.0×).
        validity_fraction = n_valid / max(n_total, 1)
        validity_factor = min(1.0, validity_fraction / 0.50)
        total = raw_total * validity_factor

        grade = "A" if total >= 90 else "B" if total >= 75 else "C" if total >= 60 else "D"
        passed = total >= self.pass_threshold

        recommendations = self._build_recommendations(
            cov_result, stale_result, outlier_result, ts_result,
            n_total=n_total, n_valid=n_valid,
        )

        return DataQualityReport(
            score=round(total, 2),
            grade=grade,
            passed=passed,
            pass_threshold=self.pass_threshold,
            n_total=n_total,
            n_valid=n_valid,
            coverage=cov_result,
            stale=stale_result,
            outliers=outlier_result,
            term_structure=ts_result,
            recommendations=recommendations,
            raw_checks={
                "coverage": cov_result,
                "stale": stale_result,
                "outliers": outlier_result,
                "term_structure": ts_result,
            },
        )

    # ── per-check scoring helpers ──────────────────────────────────────────

    @staticmethod
    def _score_coverage(r: Dict[str, Any]) -> float:
        """0–1: full coverage = 1.0, zero coverage = 0.0."""
        base = r["coverage_score"]
        penalty = 0.1 * r["n_sparse_slices"]
        if r["n_maturities"] < 3:
            penalty += 0.3
        return max(0.0, min(1.0, base - penalty))

    @staticmethod
    def _score_freshness(r: Dict[str, Any]) -> float:
        """0–1: no stale = 1.0; linear penalty on stale fraction."""
        stale_pen = 2.0 * r["stale_fraction"]
        flat_pen = 0.15 * r["n_flat_slices"]
        return max(0.0, 1.0 - stale_pen - flat_pen)

    @staticmethod
    def _score_outliers(r: Dict[str, Any]) -> float:
        """0–1: no outliers = 1.0; linear penalty on outlier fraction."""
        return max(0.0, 1.0 - 5.0 * r["outlier_fraction"])

    @staticmethod
    def _score_term_structure(r: Dict[str, Any]) -> float:
        """0–1: no inversions/jumps = 1.0."""
        inv_pen = 0.20 * r["n_inversions"]
        jmp_pen = 0.10 * r["n_jumps"]
        return max(0.0, 1.0 - inv_pen - jmp_pen)

    # ── recommendation builder ─────────────────────────────────────────────

    @staticmethod
    def _build_recommendations(
        cov: Dict, stale: Dict, outliers: Dict, ts: Dict,
        n_total: int, n_valid: int,
    ) -> List[str]:
        recs: List[str] = []

        invalid = n_total - n_valid
        if invalid > 0:
            pct = invalid / max(n_total, 1) * 100
            recs.append(
                f"Remove or impute {invalid} invalid rows ({pct:.1f}% of data)."
            )

        if cov["n_sparse_slices"] > 0:
            mats = ", ".join(f"{m:.3f}" for m in cov["sparse_slices"])
            recs.append(
                f"Sparse maturity slices (< {cov['n_sparse_slices']} strikes): T={mats}."
                " Consider fetching additional strikes or dropping the slice."
            )
        if cov["n_maturities"] < 3:
            recs.append(
                f"Only {cov['n_maturities']} maturity slice(s) found; "
                "surface construction requires ≥ 3."
            )

        if stale["n_flat_slices"] > 0:
            mats = ", ".join(f"{m:.3f}" for m in stale["flat_maturities"])
            recs.append(
                f"Flat smile detected at T={mats} — check for stale market-maker quotes."
            )
        if stale["n_stale_runs"] > 0:
            recs.append(
                f"{stale['n_stale_runs']} data points form constant-IV runs."
                " Verify data vendor freshness for those strikes."
            )

        if outliers["n_outliers"] > 0:
            recs.append(
                f"{outliers['n_outliers']} IV outlier(s) detected "
                f"({outliers['outlier_fraction'] * 100:.2f}% of data)."
                " Review and remove or clip before calibration."
            )

        if ts["n_inversions"] > 0:
            pairs = "; ".join(f"{a:.2f}→{b:.2f}" for a, b in ts["inversion_maturities"])
            recs.append(
                f"ATM IV term-structure inversions at ({pairs}) yrs."
                " Check for data errors or apply calendar-arb repair."
            )
        if ts["n_jumps"] > 0:
            pairs = "; ".join(f"{a:.2f}→{b:.2f}" for a, b in ts["jump_maturities"])
            recs.append(
                f"Large ATM IV jumps at ({pairs}) yrs — verify expiry dates."
            )

        if not recs:
            recs.append("Data quality looks good — no critical issues detected.")

        return recs
