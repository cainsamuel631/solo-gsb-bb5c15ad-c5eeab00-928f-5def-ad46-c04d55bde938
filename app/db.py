from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

# One process-wide lock serializes every read/write transaction. The API is
# I/O-latency friendly enough that a coarse lock is preferable to subtle
# scheduling races (a batch decision must be atomic w.r.t. submit/cancel).
_lock = threading.RLock()

_conn: sqlite3.Connection | None = None


def get_lock() -> threading.RLock:
    return _lock


def connect(db_path: str) -> sqlite3.Connection:
    """Open (or reopen) the module-level SQLite connection and create tables."""
    global _conn
    if _conn is not None:
        return _conn

    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        timeout=30,
        check_same_thread=False,
        isolation_level=None,  # explicit BEGIN/COMMIT
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    _init_schema(conn)
    _conn = conn
    return _conn


def get_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("database not initialized; call connect() first")
    return _conn


def reset_for_tests() -> None:
    """Drop the module-level connection (test isolation helper)."""
    global _conn
    if _conn is not None:
        _conn.close()
    _conn = None


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tenants (
            id               TEXT PRIMARY KEY,
            weight           REAL NOT NULL,
            deficit          REAL NOT NULL DEFAULT 0,
            -- Optional per-tenant cap on concurrently allocated GPUs;
            -- NULL means unlimited (only cluster capacity constrains it).
            max_gpus         INTEGER,
            -- Optional rolling 24-hour GPU-minute budget: at most
            -- gpus * duration_min of jobs may be dispatched in the window
            -- (now-24h, now]; NULL means unlimited.
            budget_gpu_min   INTEGER,
            -- Wall-clock time up to which aging has already been compensated.
            aging_settled_at TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id              TEXT PRIMARY KEY,
            tenant_id       TEXT NOT NULL REFERENCES tenants(id),
            gpus            INTEGER NOT NULL,
            duration_min    INTEGER NOT NULL,
            cost_gpu_min    REAL NOT NULL,
            status          TEXT NOT NULL CHECK (status IN
                                ('QUEUED','RUNNING','COMPLETED','CANCELLED')),
            queued_at       TEXT NOT NULL,
            scheduled_at    TEXT,
            completed_at    TEXT,
            batch_id        INTEGER REFERENCES batches(id),
            run_index       INTEGER,
            -- Moment the job last became dependency-ready when that moment
            -- was a dependency revision (set by PATCH .../dependencies when
            -- it unblocks the job, cleared when a revision blocks it).
            -- NULL otherwise: readiness is then derived from queued_at and
            -- the prerequisites' completed_at at aging-settlement time.
            ready_since     TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_tenant_status
            ON jobs(tenant_id, status);

        -- DAG edges: job_id may only run after every depends_on job is
        -- COMPLETED. CANCELLED prerequisites stay unmet forever, leaving the
        -- dependent queued (and explicitly blocked_by the cancelled job).
        CREATE TABLE IF NOT EXISTS job_dependencies (
            job_id     TEXT NOT NULL REFERENCES jobs(id),
            depends_on TEXT NOT NULL REFERENCES jobs(id),
            PRIMARY KEY (job_id, depends_on)
        );
        CREATE INDEX IF NOT EXISTS idx_job_deps_target
            ON job_dependencies(depends_on);

        -- Idempotency records for POST /jobs requests carrying an
        -- Idempotency-Key header. The (tenant_id, non-empty key) pair owns at
        -- most one job; the normalised request fingerprint (gpus,
        -- duration_min, weight, deduplicated/sorted depends_on) is stored so a
        -- retry with different parameters can be rejected (409). The row is
        -- written in the SAME transaction as the tenant upsert, the job and
        -- its dependency edges, so a rolled-back (failed) submission never
        -- consumes the key; persistence here also makes retries across
        -- container restarts resolve to the one job already created.
        CREATE TABLE IF NOT EXISTS idempotent_jobs (
            tenant_id       TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            job_id          TEXT NOT NULL REFERENCES jobs(id),
            gpus            INTEGER NOT NULL,
            duration_min    INTEGER NOT NULL,
            weight          REAL NOT NULL,
            -- JSON array of prerequisite job ids, deduplicated and sorted.
            depends_on      TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            PRIMARY KEY (tenant_id, idempotency_key)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS batches (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at       TEXT NOT NULL,
            now_ref          TEXT NOT NULL,
            capacity_gpus    INTEGER NOT NULL,
            used_gpus_before INTEGER NOT NULL,
            available_gpus   INTEGER NOT NULL,
            rounds           INTEGER NOT NULL,
            jobs_selected    INTEGER NOT NULL
        );

        -- One row per credit/debit event: aging grants, DRR quantum grants,
        -- job dispatch debits. Makes every per-tenant balance reproducible.
        CREATE TABLE IF NOT EXISTS ledger (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id   TEXT NOT NULL REFERENCES tenants(id),
            batch_id    INTEGER REFERENCES batches(id),
            job_id      TEXT REFERENCES jobs(id),
            kind        TEXT NOT NULL CHECK (kind IN
                          ('AGING','QUANTUM','DISPATCH','RESET')),
            amount      REAL NOT NULL,
            balance_after REAL NOT NULL,
            detail      TEXT,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ledger_tenant ON ledger(tenant_id, id);

        -- Persistent GPU maintenance windows: a half-open UTC interval
        -- [start_at, end_at) during which ``reserved_gpus`` GPUs are held
        -- back from scheduling. Overlapping windows' reservations add;
        -- windows merely touching at an endpoint do not overlap. Windows are
        -- advisory to dispatch: RUNNING jobs are never preempted by them.
        CREATE TABLE IF NOT EXISTS maintenance_windows (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            start_at       TEXT NOT NULL,
            end_at         TEXT NOT NULL,
            reserved_gpus  INTEGER NOT NULL,
            created_at     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_maintenance_start
            ON maintenance_windows(start_at, end_at);
        """
    )
    # Backwards-compatible migration: databases created before per-tenant GPU
    # caps existed have no ``tenants.max_gpus`` column. Add it nullable; every
    # pre-existing tenant defaults to NULL (unlimited).
    tenant_cols = {
        r["name"]
        for r in conn.execute("PRAGMA table_info(tenants)").fetchall()
    }
    if "max_gpus" not in tenant_cols:
        conn.execute("ALTER TABLE tenants ADD COLUMN max_gpus INTEGER")
    # Same for the rolling 24h GPU-minute budget column: databases created
    # before budgets existed get a nullable column; every pre-existing tenant
    # defaults to NULL (unlimited).
    if "budget_gpu_min" not in tenant_cols:
        conn.execute("ALTER TABLE tenants ADD COLUMN budget_gpu_min INTEGER")
    # Same for the dependency-readiness clock on jobs: databases created
    # before readiness-aware aging existed get a nullable column. NULL is the
    # correct default for every pre-existing job: readiness is then derived
    # from queued_at and the prerequisites' completed_at, which is exact for
    # any job a revision never unblocked.
    job_cols = {
        r["name"]
        for r in conn.execute("PRAGMA table_info(jobs)").fetchall()
    }
    if "ready_since" not in job_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN ready_since TEXT")
