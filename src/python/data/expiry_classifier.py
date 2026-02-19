"""
Expiry Classifier.

"Separate weekly vs monthly vs quarterly expiries; do not mix them blindly."

This module classifies option expiration dates by type so that:
  - Weekly options are kept on their own term-structure curve.
  - Monthly / quarterly options form the primary surface.
  - LEAPS are treated as a separate long-dated segment.

Classification rules (equity index convention):

  weekly     — expiry falls on a Friday AND days_to_expiry ≤ 8,
               OR the option expires in < 7 calendar days regardless of day.
  monthly    — third Friday of the expiry month (standard monthly expiry);
               the dominant contracts in equity index options.
  quarterly  — monthly contracts in March / June / September / December
               (the four IMM/exchange quarterly months).
  leaps      — days_to_expiry ≥ 360.
  other      — anything that does not fit the above (mid-month weeklies,
               non-standard quarterly variants, etc.).

Usage::

    from src.python.data.expiry_classifier import ExpiryClassifier, ExpiryType

    classifier = ExpiryClassifier()
    etype = classifier.classify(date.fromisoformat("2026-03-20"), today=date.today())
    # → ExpiryType.MONTHLY  (third Friday of March 2026)

    df = classifier.classify_dataframe(df, expiry_col="expiration")
    # Adds an "expiry_type" column.

    primaries = classifier.filter_primary(df, expiry_col="expiration")
    # Returns monthly + quarterly (the "main" surface).
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from enum import Enum
from typing import List, Optional

import pandas as pd


class ExpiryType(str, Enum):
    """Option expiry classification."""

    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    LEAPS = "leaps"
    OTHER = "other"


class ExpiryClassifier:
    """
    Classifies option expiry dates by type.

    Args:
        weekly_max_days:    Maximum days-to-expiry for a weekly expiry.
                            Default 8 (catches same-week Friday + up to Mon
                            morning for that week).
        leaps_min_days:     Minimum days-to-expiry for a LEAPS expiry.
                            Default 360.
        quarterly_months:   The four months that define quarterly contracts.
                            Default (3, 6, 9, 12) = Mar/Jun/Sep/Dec.

    Examples::

        c = ExpiryClassifier()
        assert c.classify(date(2026, 3, 20), date(2026, 3, 9)) == ExpiryType.QUARTERLY
        assert c.classify(date(2026, 3, 6),  date(2026, 3, 3)) == ExpiryType.WEEKLY
    """

    def __init__(
        self,
        weekly_max_days: int = 8,
        leaps_min_days: int = 360,
        quarterly_months: tuple = (3, 6, 9, 12),
    ) -> None:
        self.weekly_max_days = weekly_max_days
        self.leaps_min_days = leaps_min_days
        self.quarterly_months = set(quarterly_months)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _third_friday(year: int, month: int) -> date:
        """Return the third Friday of the given month."""
        # Find first Friday
        first_day = date(year, month, 1)
        first_friday_offset = (calendar.FRIDAY - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=first_friday_offset)
        # Third = first + 14 days
        return first_friday + timedelta(days=14)

    def _is_third_friday(self, d: date) -> bool:
        """True if date is the third Friday of its month."""
        if d.weekday() != calendar.FRIDAY:
            return False
        return d == self._third_friday(d.year, d.month)

    # ── public API ──────────────────────────────────────────────────────────

    def classify(self, expiry: date, today: date) -> ExpiryType:
        """
        Classify a single expiry date.

        Args:
            expiry: The option expiration date.
            today:  The valuation / reference date.

        Returns:
            :class:`ExpiryType` enum value.
        """
        if isinstance(expiry, pd.Timestamp):
            expiry = expiry.date()
        if isinstance(today, pd.Timestamp):
            today = today.date()

        days = (expiry - today).days

        # LEAPS first (long-dated overrides other classifications)
        if days >= self.leaps_min_days:
            return ExpiryType.LEAPS

        # Weekly (short-dated, or any expiry within weekly_max_days)
        if days <= self.weekly_max_days:
            return ExpiryType.WEEKLY

        # Standard monthly or quarterly — third-Friday check
        if self._is_third_friday(expiry):
            if expiry.month in self.quarterly_months:
                return ExpiryType.QUARTERLY
            return ExpiryType.MONTHLY

        return ExpiryType.OTHER

    def classify_bulk(
        self,
        expiries: List[date],
        today: date,
    ) -> List[ExpiryType]:
        """Classify a list of expiry dates."""
        return [self.classify(e, today) for e in expiries]

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        expiry_col: str = "expiration",
        today: Optional[date] = None,
        output_col: str = "expiry_type",
    ) -> pd.DataFrame:
        """
        Add an ``expiry_type`` column to an options DataFrame.

        Args:
            df:         Options DataFrame containing an expiry column.
            expiry_col: Column with expiry dates (date, str, or Timestamp).
            today:      Reference date; defaults to today if None.
            output_col: Name of the new classification column.

        Returns:
            A copy of df with a new string column ``output_col``.
        """
        if today is None:
            today = date.today()

        result = df.copy()
        exps = pd.to_datetime(result[expiry_col], errors="coerce")
        result[output_col] = exps.apply(
            lambda ts: self.classify(ts, today).value
            if not pd.isna(ts) else ExpiryType.OTHER.value
        )
        return result

    def filter_primary(
        self,
        df: pd.DataFrame,
        expiry_col: str = "expiration",
        today: Optional[date] = None,
        include_types: Optional[List[ExpiryType]] = None,
    ) -> pd.DataFrame:
        """
        Keep only "primary" expiries — monthly + quarterly by default.

        This is the standard surface construction set: do not mix weekly
        short-dated noise into the main calibration grid.

        Args:
            df:            Options DataFrame.
            expiry_col:    Column with expiry dates.
            today:         Reference date.
            include_types: Types to keep. Default: [MONTHLY, QUARTERLY].

        Returns:
            Filtered DataFrame (new copy).
        """
        if include_types is None:
            include_types = [ExpiryType.MONTHLY, ExpiryType.QUARTERLY]

        classified = self.classify_dataframe(df, expiry_col=expiry_col, today=today)
        include_vals = {t.value for t in include_types}
        return classified[classified["expiry_type"].isin(include_vals)].copy()

    def split_by_type(
        self,
        df: pd.DataFrame,
        expiry_col: str = "expiration",
        today: Optional[date] = None,
    ) -> dict:
        """
        Return a dict ``{ExpiryType.value → sub-DataFrame}`` for all types present.

        Useful for building separate surface curves per expiry type.
        """
        classified = self.classify_dataframe(df, expiry_col=expiry_col, today=today)
        result = {}
        for etype in ExpiryType:
            sub = classified[classified["expiry_type"] == etype.value]
            if not sub.empty:
                result[etype.value] = sub.copy()
        return result
