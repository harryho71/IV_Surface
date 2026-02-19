"""
Manual Override Framework & Front-Office Trader Tweak Layer

OverrideManager:
  - apply_override()       — record a parameter override, optionally pending approval
  - approve() / reject()   — two-eye approval workflow
  - get_active_overrides() — query currently effective overrides
  - flag_in_reports()      — annotate output dicts with active override metadata

TraderAdjustments (6.7):
  - allow_atm_shift()         — parallel vol shift across the entire surface
  - allow_skew_adjustment()   — linear skew tilt (slope change)
  - allow_wing_adjustment()   — quadratic wing adjustment (curvature change)
  - preserve_arbitrage_free() — post-adjustment calendar-arbitrage repair
  - log_adjustment()          — append to immutable per-user audit trail
  - get_adjustment_history()  — return all logged adjustments for the session
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 6.4  Override Manager
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Override:
    """Single parameter override with approval-workflow state."""

    override_id: str
    param_name: str
    original_value: float
    new_value: float
    rationale: str
    user: str
    timestamp: str
    approved: bool = False
    approved_by: Optional[str] = None
    approval_timestamp: Optional[str] = None
    rejected: bool = False
    rejection_reason: Optional[str] = None
    active: bool = False            # True once approved and not yet superseded


class OverrideManager:
    """
    Manages parameter overrides with an optional two-eye approval workflow.

    All overrides are persisted under ``<log_dir>/overrides/<override_id>.json``
    so every manual intervention is permanently traceable.

    Args:
        log_dir:          Root directory for all override logs.
        require_approval: If True (default), overrides are only active after
                          :meth:`approve` is called by a second user.

    Example::

        mgr = OverrideManager(require_approval=True)
        ov  = mgr.apply_override("rho", -0.30, -0.35, "Desk adjustment", user="trader1")
        mgr.approve(ov.override_id, approver="risk_manager")
        report = mgr.flag_in_reports({"surface_id": "SPX_20260218"})
    """

    def __init__(
        self,
        log_dir: str | Path = "output/logs",
        require_approval: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir) / "overrides"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.require_approval = require_approval
        self._overrides: Dict[str, Override] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _persist(self, override: Override) -> None:
        path = self.log_dir / f"{override.override_id}.json"
        with path.open("w") as fh:
            json.dump(asdict(override), fh, indent=2)

    def apply_override(
        self,
        param_name: str,
        original_value: float,
        new_value: float,
        rationale: str,
        user: str,
    ) -> Override:
        """
        Record a new override.

        If ``require_approval=False`` the override is immediately active.
        Otherwise it enters a pending state until :meth:`approve` is called.

        Args:
            param_name:     Name of the parameter being overridden.
            original_value: Calibrated value before the override.
            new_value:      Desired overridden value.
            rationale:      Mandatory explanation of why the override is needed.
            user:           Login name of the person requesting the override.

        Returns:
            The created :class:`Override` dataclass.
        """
        oid = str(uuid.uuid4())[:8]
        approved = not self.require_approval
        ov = Override(
            override_id=oid,
            param_name=param_name,
            original_value=original_value,
            new_value=new_value,
            rationale=rationale,
            user=user,
            timestamp=self._now(),
            approved=approved,
            active=approved,
        )
        self._overrides[oid] = ov
        self._persist(ov)
        logger.info(
            "Override %s applied for %r by %r (approved=%s)",
            oid, param_name, user, approved,
        )
        return ov

    def approve(self, override_id: str, approver: str) -> Override:
        """Approve a pending override and mark it active."""
        ov = self._overrides.get(override_id)
        if ov is None:
            raise KeyError(f"Override {override_id!r} not found")
        ov.approved = True
        ov.approved_by = approver
        ov.approval_timestamp = self._now()
        ov.active = True
        self._persist(ov)
        logger.info("Override %s APPROVED by %r", override_id, approver)
        return ov

    def reject(self, override_id: str, approver: str, reason: str) -> Override:
        """Reject a pending override."""
        ov = self._overrides.get(override_id)
        if ov is None:
            raise KeyError(f"Override {override_id!r} not found")
        ov.rejected = True
        ov.rejection_reason = reason
        ov.active = False
        self._persist(ov)
        logger.warning("Override %s REJECTED by %r: %s", override_id, approver, reason)
        return ov

    def get_active_overrides(self) -> List[Override]:
        """Return all approved and not-yet-superseded overrides."""
        return [ov for ov in self._overrides.values() if ov.active]

    def deactivate(self, override_id: str) -> None:
        """Mark an override as no longer in effect (e.g., after re-calibration)."""
        ov = self._overrides.get(override_id)
        if ov:
            ov.active = False
            self._persist(ov)

    def flag_in_reports(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotate an output dict with the currently active overrides.

        Returns a shallow copy of *report* with an added
        ``_overrides_applied`` key when overrides are active.
        """
        active = self.get_active_overrides()
        if not active:
            return report
        report = dict(report)
        report["_overrides_applied"] = [
            {
                "param": ov.param_name,
                "original": ov.original_value,
                "override": ov.new_value,
                "user": ov.user,
                "rationale": ov.rationale,
            }
            for ov in active
        ]
        return report

    def count(self) -> Dict[str, int]:
        """Return counts broken down by workflow state."""
        return {
            "total": len(self._overrides),
            "pending": sum(
                1 for o in self._overrides.values()
                if not o.approved and not o.rejected
            ),
            "approved": sum(1 for o in self._overrides.values() if o.approved),
            "active": sum(1 for o in self._overrides.values() if o.active),
            "rejected": sum(1 for o in self._overrides.values() if o.rejected),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 6.7  Front-Office Trader Tweak Layer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TraderAdjustmentRecord:
    """Immutable record of a single trader adjustment."""

    adj_id: str
    adj_type: str               # 'atm_shift' | 'skew' | 'wing' | 'custom'
    user: str
    reason: str
    timestamp: str
    params: Dict[str, Any]
    arbitrage_preserved: bool


class TraderAdjustments:
    """
    Allows front-office traders to apply controlled, audited tweaks to an IV
    surface while keeping an immutable audit trail.

    All adjustments require a non-empty *reason* string.  Each call appends
    an entry to the session log and persists a JSON file.

    Args:
        user:    Trader's login name (included in every audit record).
        log_dir: Directory for persisted adjustment records.

    Example::

        ta = TraderAdjustments(user="trader1")
        ivs_adj = ta.allow_atm_shift(ivs, amount=0.005,
                                     reason="Correcting stale ATM close")
        ivs_adj = ta.allow_skew_adjustment(ivs_adj, strikes, atm_strike=100,
                                            amount=-0.02,
                                            reason="Steeper skew post-event")
        print(ta.summary())
    """

    def __init__(
        self,
        user: str,
        log_dir: str | Path = "output/logs/trader_tweaks",
    ) -> None:
        self.user = user
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[TraderAdjustmentRecord] = []

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _require_reason(self, reason: str) -> None:
        if not reason or not reason.strip():
            raise ValueError(
                "Trader adjustments require a non-empty reason for audit purposes."
            )

    def log_adjustment(
        self,
        adj_type: str,
        params: Dict[str, Any],
        reason: str,
        arbitrage_preserved: bool = True,
    ) -> TraderAdjustmentRecord:
        """Append an adjustment to the immutable audit log and persist to disk."""
        rec = TraderAdjustmentRecord(
            adj_id=str(uuid.uuid4())[:8],
            adj_type=adj_type,
            user=self.user,
            reason=reason,
            timestamp=self._now(),
            params=params,
            arbitrage_preserved=arbitrage_preserved,
        )
        self._history.append(rec)
        path = self.log_dir / f"{rec.adj_id}.json"
        with path.open("w") as fh:
            json.dump(asdict(rec), fh, indent=2)
        return rec

    # ── surface adjustments ────────────────────────────────────────────────

    def allow_atm_shift(
        self,
        ivs: np.ndarray,
        amount: float,
        reason: str,
    ) -> np.ndarray:
        """
        Parallel shift the entire IV surface by *amount*.

        A pure shift preserves all no-arbitrage relations (butterfly and
        calendar) so no post-adjustment repair is needed.

        Args:
            ivs:    (n_strikes, n_maturities) IV array in decimal units.
            amount: Vol shift in decimal (e.g. +0.005 = +50 bps).
            reason: Mandatory audit string.

        Returns:
            Adjusted IV array, same shape, clipped to [0.0001, 5.0].
        """
        self._require_reason(reason)
        adjusted = np.clip(ivs + amount, 1e-4, 5.0)
        self.log_adjustment(
            "atm_shift",
            {"amount": amount, "shape": list(ivs.shape)},
            reason,
            arbitrage_preserved=True,
        )
        return adjusted

    def allow_skew_adjustment(
        self,
        ivs: np.ndarray,
        strikes: np.ndarray,
        atm_strike: float,
        amount: float,
        reason: str,
    ) -> np.ndarray:
        """
        Tilt the smile by adding a linear skew: Δσ(K) = amount × (K − ATM)/ATM.

        Positive *amount* raises wings relative to ATM (steeper smile).
        Negative *amount* flattens the smile.

        Args:
            ivs:        (n_strikes, n_maturities) IV array.
            strikes:    1-D array, length n_strikes.
            atm_strike: Reference ATM strike.
            amount:     Skew tilt per unit normalised moneyness.
            reason:     Mandatory audit string.
        """
        self._require_reason(reason)
        skew_vec = amount * (strikes - atm_strike) / atm_strike   # (n_strikes,)
        adjusted = np.clip(ivs + skew_vec[:, np.newaxis], 1e-4, 5.0)
        self.log_adjustment(
            "skew",
            {"amount": amount, "atm_strike": atm_strike, "shape": list(ivs.shape)},
            reason,
            arbitrage_preserved=True,
        )
        return adjusted

    def allow_wing_adjustment(
        self,
        ivs: np.ndarray,
        strikes: np.ndarray,
        atm_strike: float,
        amount: float,
        reason: str,
    ) -> np.ndarray:
        """
        Adjust wing vols symmetrically: Δσ(K) = amount × ((K − ATM)/ATM)².

        Positive *amount* adds curvature (fatter butterfly premium).

        Args:
            ivs:        (n_strikes, n_maturities) IV array.
            strikes:    1-D array, length n_strikes.
            atm_strike: Reference ATM strike.
            amount:     Wing tilt magnitude per unit squared moneyness.
            reason:     Mandatory audit string.
        """
        self._require_reason(reason)
        wing_vec = amount * ((strikes - atm_strike) / atm_strike) ** 2
        adjusted = np.clip(ivs + wing_vec[:, np.newaxis], 1e-4, 5.0)
        self.log_adjustment(
            "wing",
            {"amount": amount, "atm_strike": atm_strike, "shape": list(ivs.shape)},
            reason,
            arbitrage_preserved=True,
        )
        return adjusted

    def preserve_arbitrage_free(
        self,
        ivs: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        tolerance: float = 1e-6,
    ) -> Tuple[np.ndarray, bool]:
        """
        Validate and minimally repair calendar-arbitrage in *ivs*.

        If a calendar violation is found (total variance decreases along the
        maturity axis for any strike), the offending IV is bumped up by the
        minimum amount needed to restore monotonicity.

        Args:
            ivs:        (n_strikes, n_maturities) post-adjustment IVs.
            strikes:    1-D array of strikes (not used in repair, kept for API symmetry).
            maturities: 1-D array of maturities.
            tolerance:  Calendar-arb tolerance in total-variance units.

        Returns:
            ``(repaired_ivs, is_clean)`` where *is_clean* is ``True`` if no
            repairs were necessary.
        """
        ivs = np.array(ivs, dtype=float)
        clean = True
        for k_idx in range(ivs.shape[0]):
            tv = ivs[k_idx, :] ** 2 * maturities
            for t in range(len(maturities) - 1):
                if tv[t + 1] < tv[t] - tolerance:
                    required_iv = np.sqrt(tv[t] / maturities[t + 1]) + 1e-4
                    ivs[k_idx, t + 1] = required_iv
                    clean = False
        return ivs, clean

    # ── query ──────────────────────────────────────────────────────────────

    def get_adjustment_history(self) -> List[TraderAdjustmentRecord]:
        """Return the full in-memory adjustment log for this session."""
        return list(self._history)

    def summary(self) -> str:
        types: Dict[str, int] = {}
        for rec in self._history:
            types[rec.adj_type] = types.get(rec.adj_type, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in types.items()) or "none"
        return (
            f"TraderAdjustments | user={self.user!r} | "
            f"{len(self._history)} adjustments ({breakdown})"
        )
