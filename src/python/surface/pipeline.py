"""
Daily Calibration Pipeline

DailyCalibrationPipeline orchestrates the full end-of-day IV surface workflow:

  Step 1 — fetch_data        — Fetch market data (16:00 close or override)
  Step 2 — validate_data     — Data quality checks (stale quotes, missing strikes)
  Step 3 — calibrate         — Fit SSVI / SABR to market data
  Step 4 — arbitrage_check   — Butterfly, calendar, total-variance checks
  Step 5 — validate_greeks   — Greeks smoothness (delta/gamma oscillations)
  Step 6 — generate_reports  — PDF / HTML summary charts
  Step 7 — publish           — Push to downstream risk systems (optional)
  Step 8 — archive           — Persist parameter versions & audit logs

Each step is a *pluggable callable*.  Default no-op stubs let the pipeline
run end-to-end out of the box; replace individual steps via register_step().

Step-function signatures:
  fetch_data(date: str, context: dict) → dict
  all others:  fn(context: dict) → dict

A step failure (unhandled exception) marks the pipeline as FAILED and causes
all remaining steps to be skipped.

Example::

    pipeline = DailyCalibrationPipeline(PipelineConfig(dry_run=True))
    pipeline.register_step("calibrate", my_calibrate_fn)
    report = pipeline.run("2026-02-18")
    print(report.summary())
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration & data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuration for one pipeline run."""

    output_dir: str = "output"
    rmse_threshold: float = 0.005       # vol pts — trigger CRITICAL alert
    arb_tolerance: float = 1e-6
    log_level: str = "INFO"
    dry_run: bool = False               # if True, skip publish and archive steps
    timeout_seconds: float = 300.0      # per-step advisory timeout


@dataclass
class PipelineStep:
    """Execution metadata for a single pipeline step."""

    name: str
    status: str = "pending"             # pending | running | success | failed | skipped
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    output: Any = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def failed(self) -> bool:
        return self.status == "failed"


@dataclass
class PipelineReport:
    """Complete report for one pipeline run."""

    date: str
    run_id: str
    start_time: str
    end_time: Optional[str]
    total_duration_seconds: Optional[float]
    steps: List[PipelineStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)   # shared state
    passed: bool = False
    failure_reason: Optional[str] = None

    def summary(self) -> str:
        n_ok = sum(1 for s in self.steps if s.succeeded)
        n_fail = sum(1 for s in self.steps if s.failed)
        n_skip = sum(1 for s in self.steps if s.status == "skipped")
        dur = (
            f"{self.total_duration_seconds:.2f}s"
            if self.total_duration_seconds is not None
            else "?"
        )
        status = "PASS" if self.passed else f"FAIL ({self.failure_reason or 'unknown'})"
        return (
            f"Pipeline {self.date} [{self.run_id}] | {status} | "
            f"Steps: {n_ok} OK / {n_fail} FAILED / {n_skip} SKIPPED | {dur}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Default no-op step stubs (replace via register_step)
# ──────────────────────────────────────────────────────────────────────────────

def _noop_fetch(date: str, context: dict) -> dict:
    """Stub fetch_data — returns an empty market-data placeholder."""
    return {"date": date, "source": "noop", "options": [], "n_options": 0}


def _noop_validate(context: dict) -> dict:
    """Stub validate_data — always reports valid."""
    return {"valid": True, "n_options": 0, "warnings": [], "errors": []}


def _noop_calibrate(context: dict) -> dict:
    """Stub calibrate — returns dummy SSVI parameters."""
    return {
        "model": "ssvi",
        "converged": True,
        "rmse": 0.0,
        "iterations": 0,
        "parameters": {"rho": -0.3, "eta": 0.5, "gamma": 0.4},
    }


def _noop_arbitrage(context: dict) -> dict:
    """Stub arbitrage_check — reports no violations."""
    return {
        "is_arbitrage_free": True,
        "violations": [],
        "butterfly_violations": 0,
        "calendar_violations": 0,
    }


def _noop_greeks(context: dict) -> dict:
    """Stub validate_greeks — reports smooth Greeks."""
    return {
        "is_smooth": True,
        "delta_oscillations": 0,
        "gamma_oscillations": 0,
    }


def _noop_report(context: dict) -> dict:
    """Stub generate_reports — returns empty chart list."""
    return {"report": "noop", "charts": [], "summary_file": None}


def _noop_publish(context: dict) -> dict:
    """Stub publish — dry-run placeholder."""
    return {"published": False, "reason": "dry_run or noop"}


def _noop_archive(context: dict) -> dict:
    """Stub archive — dry-run placeholder."""
    return {"archived": False, "reason": "dry_run or noop"}


_DEFAULT_STEPS: Dict[str, Callable] = {
    "fetch_data":       _noop_fetch,
    "validate_data":    _noop_validate,
    "calibrate":        _noop_calibrate,
    "arbitrage_check":  _noop_arbitrage,
    "validate_greeks":  _noop_greeks,
    "generate_reports": _noop_report,
    "publish":          _noop_publish,
    "archive":          _noop_archive,
}

STEP_ORDER: List[str] = list(_DEFAULT_STEPS.keys())

# Steps that are automatically skipped in dry_run mode
_DRY_RUN_SKIP = {"publish", "archive"}


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline engine
# ──────────────────────────────────────────────────────────────────────────────

class DailyCalibrationPipeline:
    """
    Orchestrates the end-of-day IV surface calibration workflow.

    The 8 steps are executed in order.  Each step receives the shared
    *context* dict (accumulates outputs from previous steps) and returns a
    result dict merged back into context under the step's name.

    Step function signatures::

        fetch_data(date: str, context: dict) -> dict
        all others:  fn(context: dict) -> dict

    A step can signal failure either by raising an exception or by returning
    a dict containing ``{"failed": True}``.

    Args:
        config: :class:`PipelineConfig` controlling thresholds and output paths.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._steps: Dict[str, Callable] = dict(_DEFAULT_STEPS)
        self.output_dir = Path(self.config.output_dir)

    def register_step(self, name: str, fn: Callable) -> None:
        """
        Replace the implementation of a named pipeline step.

        Args:
            name: One of the 8 canonical step names (see ``STEP_ORDER``).
            fn:   Callable — ``fn(date, context) → dict`` for *fetch_data*,
                  ``fn(context) → dict`` for all others.

        Raises:
            ValueError: If *name* is not a recognised step name.
        """
        if name not in _DEFAULT_STEPS:
            raise ValueError(
                f"Unknown step {name!r}. Valid names: {STEP_ORDER}"
            )
        self._steps[name] = fn

    def run(self, date: str) -> PipelineReport:
        """
        Execute all 8 pipeline steps for the given *date*.

        Args:
            date: ISO date string, e.g. ``'2026-02-18'``.

        Returns:
            :class:`PipelineReport` with per-step status and shared context.
        """
        run_id = str(uuid.uuid4())[:8]
        start = datetime.now(timezone.utc)
        context: Dict[str, Any] = {
            "date": date,
            "config": asdict(self.config),
        }
        steps_done: List[PipelineStep] = []
        passed = True
        failure_reason: Optional[str] = None

        logger.info("Pipeline run_id=%s  date=%s  start", run_id, date)

        for step_name in STEP_ORDER:
            fn = self._steps[step_name]
            step = PipelineStep(name=step_name)
            step.start_time = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()

            # --- dry-run skip ---
            if self.config.dry_run and step_name in _DRY_RUN_SKIP:
                step.status = "skipped"
                step.end_time = datetime.now(timezone.utc).isoformat()
                step.duration_seconds = 0.0
                steps_done.append(step)
                continue

            # --- skip after failure ---
            if not passed:
                step.status = "skipped"
                step.end_time = datetime.now(timezone.utc).isoformat()
                step.duration_seconds = 0.0
                steps_done.append(step)
                continue

            step.status = "running"
            try:
                if step_name == "fetch_data":
                    result = fn(date, context)
                else:
                    result = fn(context)

                # Allow a step to signal failure via return value
                if isinstance(result, dict) and result.get("failed"):
                    raise RuntimeError(
                        result.get("error", f"Step '{step_name}' returned failed=True")
                    )

                context[step_name] = result
                step.output = result
                step.status = "success"
                logger.debug("Step %r  SUCCESS  %.3fs", step_name, time.perf_counter() - t0)

            except Exception as exc:
                step.status = "failed"
                step.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                if passed:
                    passed = False
                    failure_reason = f"Step '{step_name}' raised {type(exc).__name__}: {exc}"
                logger.error("Step %r FAILED: %s", step_name, exc)

            finally:
                step.end_time = datetime.now(timezone.utc).isoformat()
                step.duration_seconds = time.perf_counter() - t0

            steps_done.append(step)

        end = datetime.now(timezone.utc)
        report = PipelineReport(
            date=date,
            run_id=run_id,
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            total_duration_seconds=(end - start).total_seconds(),
            steps=steps_done,
            context=context,
            passed=passed,
            failure_reason=failure_reason,
        )
        self._persist(report)
        logger.info("Pipeline run_id=%s  %s", run_id, report.summary())
        return report

    # ── persistence ────────────────────────────────────────────────────────

    def _persist(self, report: PipelineReport) -> None:
        """Save the report as JSON under ``output/reports/<date>/<run_id>.json``."""
        reports_dir = self.output_dir / "reports" / report.date
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{report.run_id}.json"
        try:
            with path.open("w") as fh:
                json.dump(report.to_dict(), fh, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not persist pipeline report: %s", exc)
