"""
Calibration Audit Trail & Parameter Version Control

CalibrationLog:
  - Records every calibration run with full provenance.
  - Persisted as JSON under <log_dir>/<YYYY-MM-DD>/<record_id>.json
  - Query by date range, model, operator.

ParameterVersionControl (6.2):
  - Git-like commit / diff / rollback / checkout for SSVI/SABR parameters.
  - Persisted under <vc_dir>/<version_id>.json with a HEAD.txt pointer.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# 6.1  Calibration Audit Trail
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationRecord:
    """Complete provenance record for one calibration run."""

    record_id: str
    timestamp: str                           # ISO-8601 UTC
    model: str                               # 'sabr' | 'ssvi' | 'svi' | …
    operator: str
    market_data_snapshot: Dict[str, Any]     # spot, rates, bid/ask summary
    calibration_parameters: Dict[str, float] # fitted parameters
    convergence_metrics: Dict[str, Any]      # rmse, iterations, converged
    arbitrage_check_results: Dict[str, Any]  # butterfly / calendar / tv pass/fail
    manual_overrides: List[Dict[str, Any]] = field(default_factory=list)
    model_version: str = "1.0.0"
    notes: str = ""


class CalibrationLog:
    """
    Append-only audit trail for calibration runs.

    Each record is persisted as ``<log_dir>/<date>/<record_id>.json`` so the
    directory itself is the database — no external dependencies required.

    Example::

        log = CalibrationLog("output/logs/calibration")
        rid = log.record(
            model="ssvi",
            operator="quant_desk",
            market_data_snapshot={"spot": 100},
            calibration_parameters={"rho": -0.3, "eta": 0.5},
            convergence_metrics={"rmse": 0.002, "converged": True},
            arbitrage_check_results={"butterfly": True, "calendar": True},
        )
        record = log.get(rid)
    """

    def __init__(self, log_dir: str | Path = "output/logs/calibration") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ── write ──────────────────────────────────────────────────────────────

    def record(
        self,
        *,
        model: str,
        operator: str,
        market_data_snapshot: Dict[str, Any],
        calibration_parameters: Dict[str, float],
        convergence_metrics: Dict[str, Any],
        arbitrage_check_results: Dict[str, Any],
        manual_overrides: Optional[List[Dict[str, Any]]] = None,
        model_version: str = "1.0.0",
        notes: str = "",
    ) -> str:
        """Persist a calibration record and return its record_id."""
        record_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        date_str = ts[:10]
        rec = CalibrationRecord(
            record_id=record_id,
            timestamp=ts,
            model=model,
            operator=operator,
            market_data_snapshot=market_data_snapshot,
            calibration_parameters=calibration_parameters,
            convergence_metrics=convergence_metrics,
            arbitrage_check_results=arbitrage_check_results,
            manual_overrides=manual_overrides or [],
            model_version=model_version,
            notes=notes,
        )
        self._save(rec, date_str)
        return record_id

    def _save(self, rec: CalibrationRecord, date_str: str) -> None:
        day_dir = self.log_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{rec.record_id}.json"
        with path.open("w") as fh:
            json.dump(asdict(rec), fh, indent=2)

    # ── read ───────────────────────────────────────────────────────────────

    def get(self, record_id: str) -> CalibrationRecord:
        """Load a single record by ID (searches all date sub-dirs)."""
        for path in self.log_dir.rglob(f"{record_id}.json"):
            with path.open() as fh:
                data = json.load(fh)
            return CalibrationRecord(**data)
        raise KeyError(f"Record {record_id!r} not found in {self.log_dir}")

    def list_records(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        model: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> List[CalibrationRecord]:
        """Return records filtered by date range / model / operator."""
        records: List[CalibrationRecord] = []
        for path in sorted(self.log_dir.rglob("*.json")):
            with path.open() as fh:
                data = json.load(fh)
            rec = CalibrationRecord(**data)
            date = rec.timestamp[:10]
            if since and date < since:
                continue
            if until and date > until:
                continue
            if model and rec.model != model:
                continue
            if operator and rec.operator != operator:
                continue
            records.append(rec)
        return records

    def count(self) -> int:
        """Return total number of stored records."""
        return sum(1 for _ in self.log_dir.rglob("*.json"))


# ──────────────────────────────────────────────────────────────────────────────
# 6.2  Parameter Version Control
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParameterVersion:
    """Immutable snapshot of a parameter set (one 'commit')."""

    version_id: str
    parent_id: Optional[str]    # None for the initial commit
    timestamp: str
    parameters: Dict[str, float]
    commit_message: str
    author: str
    checksum: str               # SHA-256 (first 12 chars) of sorted params JSON


class ParameterVersionControl:
    """
    Lightweight Git-like versioning for SSVI / SABR parameter sets.

    Versions are stored as ``<vc_dir>/<version_id>.json``.
    HEAD pointer is stored as ``<vc_dir>/HEAD.txt``.

    Example::

        vc = ParameterVersionControl("output/models/versions")
        v1 = vc.commit({"rho": -0.3, "eta": 0.5}, "initial fit", author="quant")
        v2 = vc.commit({"rho": -0.4, "eta": 0.48}, "recalibrated after data fix")
        changes = vc.diff(v1, v2)
        params  = vc.rollback(v1)
    """

    def __init__(self, vc_dir: str | Path = "output/models/versions") -> None:
        self.vc_dir = Path(vc_dir)
        self.vc_dir.mkdir(parents=True, exist_ok=True)
        self._head_file = self.vc_dir / "HEAD.txt"

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _checksum(params: Dict[str, float]) -> str:
        blob = json.dumps(dict(sorted(params.items())), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def head(self) -> Optional[str]:
        if self._head_file.exists():
            return self._head_file.read_text().strip() or None
        return None

    def _set_head(self, version_id: str) -> None:
        self._head_file.write_text(version_id)

    def _load(self, version_id: str) -> ParameterVersion:
        path = self.vc_dir / f"{version_id}.json"
        if not path.exists():
            raise KeyError(f"Version {version_id!r} not found in {self.vc_dir}")
        with path.open() as fh:
            data = json.load(fh)
        return ParameterVersion(**data)

    # ── public API ─────────────────────────────────────────────────────────

    def commit(
        self,
        parameters: Dict[str, float],
        message: str,
        author: str = "system",
    ) -> str:
        """Record a new parameter snapshot; returns version_id."""
        version_id = str(uuid.uuid4())[:8]
        ver = ParameterVersion(
            version_id=version_id,
            parent_id=self.head,
            timestamp=datetime.now(timezone.utc).isoformat(),
            parameters=dict(parameters),
            commit_message=message,
            author=author,
            checksum=self._checksum(parameters),
        )
        path = self.vc_dir / f"{version_id}.json"
        with path.open("w") as fh:
            json.dump(asdict(ver), fh, indent=2)
        self._set_head(version_id)
        return version_id

    def checkout(self, version_id: str) -> Dict[str, float]:
        """Return the parameter dict at a given version (read-only)."""
        return dict(self._load(version_id).parameters)

    def rollback(self, version_id: str) -> Dict[str, float]:
        """Set HEAD to *version_id* and return its parameters."""
        params = self.checkout(version_id)
        self._set_head(version_id)
        return params

    def diff(
        self,
        version_id_a: str,
        version_id_b: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute parameter-level differences between two versions.

        Returns a dict keyed by parameter name, with sub-keys
        ``from``, ``to``, and ``change_pct`` (percentage change).
        Only parameters that differ are included.
        """
        a = self._load(version_id_a).parameters
        b = self._load(version_id_b).parameters
        all_keys = set(a) | set(b)
        result: Dict[str, Dict[str, float]] = {}
        for k in sorted(all_keys):
            va = a.get(k, float("nan"))
            vb = b.get(k, float("nan"))
            if va != vb:
                pct = (vb - va) / (abs(va) + 1e-12) * 100.0
                result[k] = {"from": va, "to": vb, "change_pct": round(pct, 4)}
        return result

    def history(self) -> List[ParameterVersion]:
        """All versions, sorted by timestamp ascending."""
        versions: List[ParameterVersion] = []
        for path in self.vc_dir.glob("*.json"):
            if path.stem == "HEAD":
                continue
            with path.open() as fh:
                data = json.load(fh)
            versions.append(ParameterVersion(**data))
        return sorted(versions, key=lambda v: v.timestamp)

    def log(self) -> List[str]:
        """One-line summary per version (newest first)."""
        return [
            f"[{v.version_id}] {v.timestamp[:19]}  {v.author:<20}  {v.commit_message}"
            for v in reversed(self.history())
        ]
