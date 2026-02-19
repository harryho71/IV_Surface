"""
Calibration Failure Alerting System

AlertSystem monitors calibration runs for anomalies and dispatches structured
alerts to registered handlers (email, Slack, Teams, etc.).

Detects:
  - Parameter jumps  (> threshold % day-over-day change)
  - Arbitrage violations discovered post-calibration
  - Optimiser convergence failures / high RMSE
  - Any custom condition via check_custom()

Notifications are written to a JSON log by default; arbitrary handlers can
be registered via add_handler().

Example::

    system = AlertSystem()
    alerts = system.detect_parameter_jumps(params_now, params_prev, threshold=0.10)
    alerts += system.check_convergence(rmse=0.008, converged=True)
    print(system.summary())
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

LEVELS = ("INFO", "WARNING", "CRITICAL")


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    """Single alert event."""

    alert_id: str
    level: str                      # INFO | WARNING | CRITICAL
    source: str                     # 'parameter_jump' | 'arbitrage' | 'convergence' | …
    message: str
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.level}] {self.timestamp[:19]}  {self.source}: {self.message}"


# ──────────────────────────────────────────────────────────────────────────────
# Alert system
# ──────────────────────────────────────────────────────────────────────────────

class AlertSystem:
    """
    Monitors calibration outputs and raises structured alerts.

    Handlers can be registered with :meth:`add_handler` to forward alerts to
    external channels (email, Slack, database, …).

    Args:
        log_dir: Directory where alert JSON files are persisted.
        persist: If False, alerts are kept in memory only (useful in tests).
    """

    def __init__(
        self,
        *,
        log_dir: str | Path = "output/logs/alerts",
        persist: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        if persist:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.persist = persist
        self._handlers: List[Callable[[Alert], None]] = []
        self._alerts: List[Alert] = []

    # ── handler registration ───────────────────────────────────────────────

    def add_handler(self, fn: Callable[[Alert], None]) -> None:
        """Register a callable that receives every new Alert."""
        self._handlers.append(fn)

    def _dispatch(self, alert: Alert) -> None:
        self._alerts.append(alert)
        if self.persist:
            self._save(alert)
        _LEVEL_MAP = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "CRITICAL": logging.CRITICAL,
        }
        logger.log(_LEVEL_MAP.get(alert.level, logging.WARNING), str(alert))
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as exc:
                logger.error("Alert handler raised: %s", exc)

    def _save(self, alert: Alert) -> None:
        date_str = alert.timestamp[:10]
        day_dir = self.log_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{alert.alert_id}.json"
        with path.open("w") as fh:
            json.dump(asdict(alert), fh, indent=2)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _make_id() -> str:
        return str(uuid.uuid4())[:8]

    # ── monitors ───────────────────────────────────────────────────────────

    def detect_parameter_jumps(
        self,
        params_now: Dict[str, float],
        params_prev: Dict[str, float],
        threshold: float = 0.10,
    ) -> List[Alert]:
        """
        Compare current and previous calibration parameters.

        Args:
            params_now:  Current calibration output.
            params_prev: Previous calibration output (e.g., yesterday).
            threshold:   Relative change that triggers a WARNING (default 10 %).
                         Changes ≥ 2× threshold trigger CRITICAL.

        Returns:
            List of generated alerts (also dispatched internally).
        """
        alerts: List[Alert] = []
        for k in sorted(set(params_now) & set(params_prev)):
            old, new = params_prev[k], params_now[k]
            rel = abs(new - old) / (abs(old) + 1e-12)
            if rel >= threshold:
                level = "CRITICAL" if rel >= 2 * threshold else "WARNING"
                msg = (
                    f"Parameter '{k}' changed {rel * 100:.1f}% "
                    f"(prev={old:.4f}  now={new:.4f})"
                )
                a = Alert(
                    alert_id=self._make_id(),
                    level=level,
                    source="parameter_jump",
                    message=msg,
                    timestamp=self._now(),
                    details={"param": k, "prev": old, "now": new, "rel_change": rel},
                )
                self._dispatch(a)
                alerts.append(a)
        return alerts

    def check_arbitrage_violations(
        self,
        report: Any,
    ) -> List[Alert]:
        """
        Raise alerts for any arbitrage violations in a validation report.

        Accepts either an ``ArbitrageReport`` object (with ``.violations`` and
        ``.is_arbitrage_free``) or a plain ``dict`` with equivalent keys.

        Returns:
            List of generated alerts.
        """
        alerts: List[Alert] = []

        # Normalise both dataclass and dict inputs
        if hasattr(report, "violations"):
            violations = report.violations
            is_arb_free = report.is_arbitrage_free
        else:
            is_arb_free = report.get("is_arbitrage_free", True)
            violations = report.get("violations", [])

        if is_arb_free:
            return alerts

        def _sev(v: Any) -> str:
            return getattr(v, "severity", None) or (v.get("severity") if isinstance(v, dict) else "")

        severe = [v for v in violations if _sev(v) == "severe"]
        moderate = [v for v in violations if _sev(v) == "moderate"]

        if severe:
            a = Alert(
                alert_id=self._make_id(),
                level="CRITICAL",
                source="arbitrage",
                message=f"{len(severe)} SEVERE arbitrage violation(s) detected",
                timestamp=self._now(),
                details={"severe_count": len(severe), "moderate_count": len(moderate)},
            )
            self._dispatch(a)
            alerts.append(a)
        elif moderate:
            a = Alert(
                alert_id=self._make_id(),
                level="WARNING",
                source="arbitrage",
                message=f"{len(moderate)} moderate arbitrage violation(s) detected",
                timestamp=self._now(),
                details={"moderate_count": len(moderate)},
            )
            self._dispatch(a)
            alerts.append(a)

        return alerts

    def check_convergence(
        self,
        rmse: float,
        converged: bool,
        rmse_threshold: float = 0.005,
    ) -> List[Alert]:
        """
        Raise alerts for calibration convergence failures or high RMSE.

        Args:
            rmse:           Final calibration RMSE in decimal vol units.
            converged:      Whether the optimiser reported convergence.
            rmse_threshold: RMSE (vol units) above which a WARNING is issued.
                            CRITICAL if rmse > 2 × threshold.

        Returns:
            List of generated alerts.
        """
        alerts: List[Alert] = []
        if not converged:
            a = Alert(
                alert_id=self._make_id(),
                level="CRITICAL",
                source="convergence",
                message="Calibration did NOT converge",
                timestamp=self._now(),
                details={"rmse": rmse, "converged": False},
            )
            self._dispatch(a)
            alerts.append(a)
        elif rmse > rmse_threshold:
            level = "CRITICAL" if rmse > 2 * rmse_threshold else "WARNING"
            a = Alert(
                alert_id=self._make_id(),
                level=level,
                source="convergence",
                message=(
                    f"Calibration RMSE={rmse * 100:.3f} vol pts "
                    f"exceeds threshold={rmse_threshold * 100:.3f}"
                ),
                timestamp=self._now(),
                details={"rmse": rmse, "threshold": rmse_threshold},
            )
            self._dispatch(a)
            alerts.append(a)
        return alerts

    def check_custom(
        self,
        condition: bool,
        level: str,
        source: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Alert]:
        """
        Raise a custom alert when *condition* is True.

        Args:
            condition: If True, the alert is dispatched.
            level:     'INFO' | 'WARNING' | 'CRITICAL'.
            source:    Short tag identifying the check (e.g. 'data_quality').
            message:   Human-readable description.
            details:   Optional additional data.

        Returns:
            The Alert if dispatched, else None.
        """
        if not condition:
            return None
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        a = Alert(
            alert_id=self._make_id(),
            level=level,
            source=source,
            message=message,
            timestamp=self._now(),
            details=details or {},
        )
        self._dispatch(a)
        return a

    # ── query ──────────────────────────────────────────────────────────────

    def get_alerts(
        self,
        since: Optional[str] = None,
        level: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Alert]:
        """Return in-memory alerts, optionally filtered."""
        out = list(self._alerts)
        if since:
            out = [a for a in out if a.timestamp[:10] >= since]
        if level:
            out = [a for a in out if a.level == level]
        if source:
            out = [a for a in out if a.source == source]
        return out

    def clear(self) -> None:
        """Reset in-memory alert buffer (useful in tests)."""
        self._alerts.clear()

    def summary(self) -> str:
        crit = sum(1 for a in self._alerts if a.level == "CRITICAL")
        warn = sum(1 for a in self._alerts if a.level == "WARNING")
        info = sum(1 for a in self._alerts if a.level == "INFO")
        return (
            f"AlertSystem | CRITICAL={crit} WARNING={warn} INFO={info}"
            f" | Total={len(self._alerts)}"
        )
