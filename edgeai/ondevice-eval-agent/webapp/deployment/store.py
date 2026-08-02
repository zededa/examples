"""
SQLite-backed deployment store.

One DB per pod lifetime (or longer when mounted on a PVC) that records:
  - `baseline`  — the one golden reference captured on first boot
  - `run`       — every eval run (baseline, sanity, manual)
  - `drift`     — events emitted when a sanity-eval p95 diverges from baseline

The DB lives at `{SESSION_STORAGE_ROOT}/deployment/deployment.db`. When
`SESSION_STORAGE_ROOT` points at an emptyDir (the Helm default), the DB
is ephemeral and reinits from Triton on restart — that is fine for the
"auto-baseline on first boot" flow. When a PVC is mounted, it survives.

Schema notes:
  - `baseline` is intentionally a table (not a single row) so we can
    rebaseline on a model swap (new `mlflow_run_id`) without losing
    history. The `active` flag marks the current reference row.
  - Numeric columns are nullable — thermal/power readings only exist on
    real Jetson hardware, not inside generic containers.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Schema
# =============================================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS baseline (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          REAL    NOT NULL,
    model_name          TEXT    NOT NULL,
    mlflow_run_id       TEXT,
    model_type          TEXT,
    iterations          INTEGER,
    inference_mean_ms   REAL,
    inference_p50_ms    REAL,
    inference_p95_ms    REAL,
    inference_p99_ms    REAL,
    gpu_util_mean       REAL,
    junction_temp_mean  REAL,
    total_power_mean_w  REAL,
    accuracy            REAL,
    active              INTEGER NOT NULL DEFAULT 1,
    metadata_json       TEXT
);

CREATE INDEX IF NOT EXISTS idx_baseline_active ON baseline(active);

CREATE TABLE IF NOT EXISTS run (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          REAL    NOT NULL,
    kind                TEXT    NOT NULL,
    model_name          TEXT    NOT NULL,
    iterations          INTEGER,
    inference_mean_ms   REAL,
    inference_p95_ms    REAL,
    gpu_util_mean       REAL,
    junction_temp_mean  REAL,
    total_power_mean_w  REAL,
    accuracy            REAL,
    success             INTEGER NOT NULL DEFAULT 1,
    error               TEXT,
    details_json        TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_kind_created ON run(kind, created_at DESC);

CREATE TABLE IF NOT EXISTS drift (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          REAL    NOT NULL,
    drift_score         REAL    NOT NULL,
    baseline_p95_ms     REAL,
    current_p95_ms      REAL,
    run_id              INTEGER REFERENCES run(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_drift_created ON drift(created_at DESC);
"""


# =============================================================================
# DTOs
# =============================================================================

@dataclass
class Baseline:
    id: int
    created_at: float
    model_name: str
    mlflow_run_id: Optional[str]
    model_type: Optional[str]
    iterations: Optional[int]
    inference_mean_ms: Optional[float]
    inference_p50_ms: Optional[float]
    inference_p95_ms: Optional[float]
    inference_p99_ms: Optional[float]
    gpu_util_mean: Optional[float]
    junction_temp_mean: Optional[float]
    total_power_mean_w: Optional[float]
    accuracy: Optional[float]
    metadata: Dict[str, Any]


@dataclass
class Run:
    id: int
    created_at: float
    kind: str  # 'baseline' | 'sanity' | 'manual'
    model_name: str
    iterations: Optional[int]
    inference_mean_ms: Optional[float]
    inference_p95_ms: Optional[float]
    gpu_util_mean: Optional[float]
    junction_temp_mean: Optional[float]
    total_power_mean_w: Optional[float]
    accuracy: Optional[float]
    success: bool
    error: Optional[str]
    details: Dict[str, Any]


@dataclass
class DriftEvent:
    id: int
    created_at: float
    drift_score: float
    baseline_p95_ms: Optional[float]
    current_p95_ms: Optional[float]
    run_id: Optional[int]


# =============================================================================
# Store
# =============================================================================

class DeploymentStore:
    """
    Thin SQLite wrapper. Single connection guarded by a lock — writes
    are rare (baseline + every N minutes) and reads cheap, so contention
    is not a real concern.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.RLock()
        # isolation_level=None → autocommit per statement; we still wrap
        # multi-statement sequences in explicit transactions below.
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        logger.info("Deployment store ready at %s", db_path)

    # --- transactions --------------------------------------------------------

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # --- baseline ------------------------------------------------------------

    def get_active_baseline(self) -> Optional[Baseline]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM baseline WHERE active=1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return _row_to_baseline(row) if row else None

    def has_baseline_for(self, model_name: str, mlflow_run_id: Optional[str]) -> bool:
        """
        Does an active baseline already exist for this deployment identity?

        The deployment is identified by (model_name, mlflow_run_id). When
        `mlflow_run_id` is None we fall back to model_name alone — still
        prevents re-baselining every restart.
        """
        with self._lock:
            if mlflow_run_id:
                row = self._conn.execute(
                    "SELECT 1 FROM baseline WHERE active=1 AND model_name=? AND mlflow_run_id=?",
                    (model_name, mlflow_run_id),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT 1 FROM baseline WHERE active=1 AND model_name=?",
                    (model_name,),
                ).fetchone()
        return row is not None

    def save_baseline(
        self,
        *,
        model_name: str,
        mlflow_run_id: Optional[str],
        model_type: Optional[str],
        iterations: int,
        inference_mean_ms: Optional[float],
        inference_p50_ms: Optional[float],
        inference_p95_ms: Optional[float],
        inference_p99_ms: Optional[float],
        gpu_util_mean: Optional[float],
        junction_temp_mean: Optional[float],
        total_power_mean_w: Optional[float],
        accuracy: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Write a new baseline row as the active reference; deactivate any
        prior active row. Returns the new baseline id.
        """
        with self._txn() as conn:
            conn.execute("UPDATE baseline SET active=0 WHERE active=1")
            cur = conn.execute(
                """
                INSERT INTO baseline (
                    created_at, model_name, mlflow_run_id, model_type,
                    iterations,
                    inference_mean_ms, inference_p50_ms, inference_p95_ms, inference_p99_ms,
                    gpu_util_mean, junction_temp_mean, total_power_mean_w,
                    accuracy, active, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    time.time(), model_name, mlflow_run_id, model_type,
                    iterations,
                    inference_mean_ms, inference_p50_ms, inference_p95_ms, inference_p99_ms,
                    gpu_util_mean, junction_temp_mean, total_power_mean_w,
                    accuracy, json.dumps(metadata or {}),
                ),
            )
            return int(cur.lastrowid)

    # --- runs ---------------------------------------------------------------

    def save_run(
        self,
        *,
        kind: str,
        model_name: str,
        iterations: Optional[int],
        inference_mean_ms: Optional[float],
        inference_p95_ms: Optional[float],
        gpu_util_mean: Optional[float],
        junction_temp_mean: Optional[float],
        total_power_mean_w: Optional[float],
        accuracy: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO run (
                    created_at, kind, model_name, iterations,
                    inference_mean_ms, inference_p95_ms,
                    gpu_util_mean, junction_temp_mean, total_power_mean_w,
                    accuracy, success, error, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(), kind, model_name, iterations,
                    inference_mean_ms, inference_p95_ms,
                    gpu_util_mean, junction_temp_mean, total_power_mean_w,
                    accuracy, 1 if success else 0, error,
                    json.dumps(details or {}),
                ),
            )
            return int(cur.lastrowid)

    def get_latest_run(self, kind: Optional[str] = None) -> Optional[Run]:
        with self._lock:
            if kind:
                row = self._conn.execute(
                    "SELECT * FROM run WHERE kind=? ORDER BY created_at DESC LIMIT 1",
                    (kind,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM run ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
        return _row_to_run(row) if row else None

    def list_runs(self, kind: Optional[str] = None, limit: int = 20) -> List[Run]:
        with self._lock:
            if kind:
                rows = self._conn.execute(
                    "SELECT * FROM run WHERE kind=? ORDER BY created_at DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM run ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [_row_to_run(r) for r in rows]

    # --- drift --------------------------------------------------------------

    def save_drift(
        self,
        *,
        drift_score: float,
        baseline_p95_ms: Optional[float],
        current_p95_ms: Optional[float],
        run_id: Optional[int],
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO drift (created_at, drift_score, baseline_p95_ms, current_p95_ms, run_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (time.time(), drift_score, baseline_p95_ms, current_p95_ms, run_id),
            )
            return int(cur.lastrowid)

    def list_drift_events(self, limit: int = 20) -> List[DriftEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM drift ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            DriftEvent(
                id=int(r["id"]),
                created_at=float(r["created_at"]),
                drift_score=float(r["drift_score"]),
                baseline_p95_ms=_opt_float(r["baseline_p95_ms"]),
                current_p95_ms=_opt_float(r["current_p95_ms"]),
                run_id=int(r["run_id"]) if r["run_id"] is not None else None,
            )
            for r in rows
        ]

    # --- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# =============================================================================
# Row adapters
# =============================================================================

def _opt_float(v: Any) -> Optional[float]:
    return float(v) if v is not None else None


def _row_to_baseline(row: sqlite3.Row) -> Baseline:
    try:
        md = json.loads(row["metadata_json"] or "{}")
    except Exception:
        md = {}
    return Baseline(
        id=int(row["id"]),
        created_at=float(row["created_at"]),
        model_name=str(row["model_name"]),
        mlflow_run_id=row["mlflow_run_id"],
        model_type=row["model_type"],
        iterations=int(row["iterations"]) if row["iterations"] is not None else None,
        inference_mean_ms=_opt_float(row["inference_mean_ms"]),
        inference_p50_ms=_opt_float(row["inference_p50_ms"]),
        inference_p95_ms=_opt_float(row["inference_p95_ms"]),
        inference_p99_ms=_opt_float(row["inference_p99_ms"]),
        gpu_util_mean=_opt_float(row["gpu_util_mean"]),
        junction_temp_mean=_opt_float(row["junction_temp_mean"]),
        total_power_mean_w=_opt_float(row["total_power_mean_w"]),
        accuracy=_opt_float(row["accuracy"]),
        metadata=md,
    )


def _row_to_run(row: sqlite3.Row) -> Run:
    try:
        details = json.loads(row["details_json"] or "{}")
    except Exception:
        details = {}
    return Run(
        id=int(row["id"]),
        created_at=float(row["created_at"]),
        kind=str(row["kind"]),
        model_name=str(row["model_name"]),
        iterations=int(row["iterations"]) if row["iterations"] is not None else None,
        inference_mean_ms=_opt_float(row["inference_mean_ms"]),
        inference_p95_ms=_opt_float(row["inference_p95_ms"]),
        gpu_util_mean=_opt_float(row["gpu_util_mean"]),
        junction_temp_mean=_opt_float(row["junction_temp_mean"]),
        total_power_mean_w=_opt_float(row["total_power_mean_w"]),
        accuracy=_opt_float(row["accuracy"]),
        success=bool(row["success"]),
        error=row["error"],
        details=details,
    )


# =============================================================================
# Singleton access
# =============================================================================

_store: Optional[DeploymentStore] = None
_store_lock = threading.Lock()


def get_store() -> Optional[DeploymentStore]:
    """
    Return the process-wide store, initializing on first call.

    Returns None when `DEPLOYMENT_ENABLED=false` or when the DB path
    is not writable — callers should treat absence as "feature off."
    """
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        try:
            from config import get_settings
            from sessions.registry import SESSION_STORAGE_ROOT
            if not get_settings().deployment.enabled:
                return None
            db_path = os.path.join(SESSION_STORAGE_ROOT, "deployment", "deployment.db")
            _store = DeploymentStore(db_path)
        except Exception as e:
            logger.warning("Could not initialize deployment store: %s", e)
            return None
    return _store
