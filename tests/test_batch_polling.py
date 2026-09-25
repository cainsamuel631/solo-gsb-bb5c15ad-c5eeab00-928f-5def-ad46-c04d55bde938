"""Batch polling shape: one bulk queue/dependency read, head cursors that
advance only after dispatch.

These tests pin the refactored ``_batch_tx_body`` polling mechanics without
re-implementing its decisions (those are covered exhaustively elsewhere):

* the batch issues exactly ONE FIFO queue scan WITH joined dependency state;
* the loop never issues per-tenant head/dependency/running-gpu/has-queued
  queries round after round;
* a tenant with several affordable jobs dispatches them in FIFO order (the
  cursor advances only as each job is dispatched), while a dependency-blocked
  head stays pinned for the whole batch and earns nothing;
* /scheduler/preview and /scheduler/batch stay one code path.
"""
import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def submit(tenant, gpus, minutes, weight=1.0, when=None, depends_on=None):
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
    return scheduler._get_job_or_404(db.get_conn(), job["id"])


class _Tracer:
    """Collect normalized SQL strings executed on the shared connection."""

    def __init__(self, conn):
        self.conn = conn
        self.statements: list[str] = []

    def __enter__(self):
        self.conn.set_trace_callback(
            lambda s: self.statements.append(" ".join(s.split()))
        )
        return self

    def __exit__(self, *exc):
        self.conn.set_trace_callback(None)

    def count(self, pred):
        return sum(1 for s in self.statements if pred(s))


def test_batch_reads_fifo_queues_and_dependencies_exactly_once(conn):
    # Two costly heads (cost 600 => ten accrued rounds), a cheap two-job
    # tenant (cursor advances across rounds) and a dependency-blocked head
    # together force the old code's round-by-round requeries.
    submit("a", 6, 100, when=iso(0))                       # J000001
    submit("b", 6, 100, when=iso(1))                       # J000002
    submit("c", 1, 10, when=iso(0), depends_on=["J000001"])  # blocked J000003
    submit("d", 1, 10, when=iso(0))                        # J000004
    submit("d", 1, 10, when=iso(1))                        # J000005

    tracer = _Tracer(db.get_conn())
    with tracer:
        r = scheduler.run_batch(now=at(10))

    # d drains in rounds 1-2 (cursor advancing); a finally affords its
    # 6-GPU head exactly in round 10; c's head stays pinned.
    assert r["rounds"] == 10
    assert r["selection_order"] == ["J000004", "J000005", "J000001"]

    # Exactly one FIFO queue scan, and it carries the dependency join.
    bulk = [s for s in tracer.statements if s.startswith("SELECT j.*")]
    assert len(bulk) == 1
    assert "job_dependencies" in bulk[0]
    assert "ORDER BY j.tenant_id, j.queued_at, j.id" in bulk[0]

    # No per-round head-of-line SELECTs...
    assert tracer.count(
        lambda s: s.startswith("SELECT * FROM jobs")
        and "ORDER BY queued_at, id LIMIT 1" in s
    ) == 0
    # ...no per-head unmet-prerequisite queries...
    assert tracer.count(
        lambda s: "FROM job_dependencies d JOIN jobs j" in s
        and "j.status != ?" in s
    ) == 0
    # ...no per-tenant RUNNING-gpus lookups...
    assert tracer.count(
        lambda s: s.startswith(
            "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs "
            "WHERE tenant_id = ? AND status = ?")
    ) == 0
    # ...no round-by-round weight fetches or trailing has-queued COUNTs.
    assert tracer.count(
        lambda s: s.startswith("SELECT weight FROM tenants")
    ) == 0
    assert tracer.count(
        lambda s: "COUNT(*) AS n FROM jobs WHERE tenant_id = ?" in s
    ) == 0


def test_cursor_advances_only_after_dispatch_and_pins_blocked_head(conn):
    # Tenant a has three ready jobs; the cursor advances through all three
    # WITHIN the first round, each new head re-judged against remaining
    # capacity. Tenant b's 5-GPU head exactly fits afterwards and dispatches.
    submit("a", 1, 10, when=iso(0))
    submit("a", 1, 10, when=iso(1))
    submit("a", 1, 10, when=iso(2))
    submit("b", 5, 10, when=iso(0))               # J000004, fits (3+5=8)
    submit("c", 1, 10, when=iso(0), depends_on=["J000004"])  # blocked head
    submit("c", 1, 10, when=iso(1))               # ready behind blocked head

    r = scheduler.run_batch(now=at(5))
    # Round 1's fitting set is fixed before visiting (WDRR): a appears once,
    # so it dispatches J000001 then b dispatches J000004 (free=4). a's new
    # head is re-judged only on the next round (the fitting set is rebuilt),
    # so J000002 and J000003 each dispatch on their own following round.
    assert r["rounds"] == 3
    assert r["selection_order"] == [
        "J000001", "J000004", "J000002", "J000003"
    ]
    # c never moved: blocked head pins the cursor, strict FIFO, no credit.
    queued = {j["id"] for j in scheduler.list_jobs(status="QUEUED")}
    assert queued == {"J000005", "J000006"}
    assert scheduler.get_tenant("c")["balance_gpu_min"] == 0.0
    assert all(e["tenant_id"] != "c" for e in r["events"])

    # Completing the prerequisite releases the head, then the one behind it
    # (the cursor advances twice again within one round).
    scheduler.complete_job("J000004")
    r2 = scheduler.run_batch(now=at(6))
    assert r2["selection_order"] == ["J000005", "J000006"]


def test_preview_uses_the_same_single_read_and_writes_nothing(conn):
    submit("a", 6, 100, when=iso(0))
    submit("b", 1, 10, when=iso(1))

    tracer = _Tracer(db.get_conn())
    with tracer:
        p = scheduler.preview_batch(now=at(10))

    bulk = [s for s in tracer.statements if s.startswith("SELECT j.*")]
    assert len(bulk) == 1
    assert p["preview"] is True
    # Zero writes: the head is still QUEUED and no ledger/batch rows exist.
    assert scheduler.get_job("J000001")["status"] == "QUEUED"
    c = db.get_conn()
    assert c.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()["n"] == 0
    assert c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 0

    # The committed batch at the same instant selects the same FIFO order.
    committed = scheduler.run_batch(now=at(10))
    assert committed["selection_order"] == p["selection_order"]
