"""Readiness-aware wait aging: credit accrues from true head readiness.

Covers the fix for dependency-unlock aging settlement:

* a head job queued long before its prerequisites completed earns aging
  only from the release instant, not from arrival — even when no batch ran
  (and could advance the aging checkpoint) during the blocked wait;
* with several prerequisites the LAST completion governs;
* a dependency revision that unblocks a blocked job starts the clock at
  the revision instant, and a re-blocking revision clears it again;
* the readiness instant is combined with the tenant's existing aging
  settlement point by taking the LATER of the two (no double counting);
* /scheduler/preview and /scheduler/batch share the computation: the
  preview writes nothing and yields identical ledger amounts at the same
  state and reference instant.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def submit(tenant, gpus=1, minutes=10, weight=1.0, when=None, depends_on=None):
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
    return scheduler.get_job(job["id"])


def pin_completed(job_id, when):
    """Mark a job COMPLETED at a deterministic instant (test time control)."""
    db.get_conn().execute(
        "UPDATE jobs SET status = 'COMPLETED', completed_at = ? WHERE id = ?",
        (when, job_id),
    )


def batch(now_minutes):
    return scheduler.run_batch(now=at(now_minutes))


def aging_for(result, tenant_id):
    return [
        e for e in result["events"]
        if e["kind"] == "AGING" and e["tenant_id"] == tenant_id
    ]


# ---------------------------------------------------------------------------
# Completion-released readiness
# ---------------------------------------------------------------------------

def test_aging_starts_at_prerequisite_completion_not_arrival(conn):
    # The task's scenario: head queued 60 min, prerequisite completed only
    # 1 min before the batch, and no batch ran in between. The old ledger
    # would grant 60 min of AGING; only the truly-ready minute counts.
    pre = submit("a", when=iso(0))                            # J000001
    dep = submit("b", when=iso(0), depends_on=[pre["id"]])    # J000002
    pin_completed(pre["id"], iso(59))

    r = batch(60)
    assert r["selection_order"] == ["J000002"]
    aging = aging_for(r, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(1.0)  # not 60.0
    assert aging[0]["detail"].startswith("waited 1.000 min")


def test_aging_uses_last_completion_with_multiple_prerequisites(conn):
    p1 = submit("a", minutes=5, when=iso(0))
    p2 = submit("a", minutes=5, when=iso(0))
    dep = submit("b", minutes=5, when=iso(0), depends_on=[p1["id"], p2["id"]])
    # Ready only when the LAST prerequisite completes (t=50).
    pin_completed(p1["id"], iso(10))
    pin_completed(p2["id"], iso(50))

    r = batch(60)
    assert r["selection_order"] == ["J000003"]
    aging = aging_for(r, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(10.0)  # 60 - 50, not 60 - 10


def test_aging_release_after_intervening_blocked_batches(conn):
    # Batches while the head is blocked advance the checkpoint without
    # granting credit; the release completion still bounds the final wait.
    pre = submit("a", gpus=8, minutes=5, weight=4.0, when=iso(0))
    dep = submit("b", minutes=5, when=iso(0), depends_on=[pre["id"]])

    r1 = batch(10)  # pre saturates the cluster; b blocked, no credit
    assert r1["selection_order"] == [pre["id"]]
    assert aging_for(r1, "b") == []
    r2 = batch(20)  # still blocked: checkpoint advances, still no credit
    assert aging_for(r2, "b") == []
    assert scheduler.get_tenant("b")["balance_gpu_min"] == 0.0

    pin_completed(pre["id"], iso(59))
    r3 = batch(60)
    aging = aging_for(r3, "b")
    # Later of (settlement t=20, readiness t=59) -> 1 minute.
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(1.0)


def test_settlement_point_later_than_readiness_governs(conn):
    # Once aging has been settled past the readiness instant, the settlement
    # point governs: the wait is never double-counted from readiness again.
    pre = submit("a", gpus=8, minutes=5, weight=4.0, when=iso(0))
    # b's head is far too expensive to dispatch within one batch, so it
    # stays queued (and ready) across both settlement batches below.
    dep = submit("b", minutes=5000, when=iso(0), depends_on=[pre["id"]])
    batch(10)  # pre dispatches; b blocked -> checkpoint t=10
    pin_completed(pre["id"], iso(20))  # b ready since t=20
    # Tenant a queues another cluster-filling job before b can dispatch.
    submit("a", gpus=8, minutes=1000, weight=4.0, when=iso(21))

    r = batch(30)
    # a's new 8-GPU job takes the freed capacity; b stays queued but ready.
    assert r["selection_order"] == ["J000003"]
    aging = aging_for(r, "b")
    # Later of (settlement t=10, readiness t=20) -> 30 - 20 = 10 minutes.
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(10.0)

    r2 = batch(40)  # cluster still full; b ready and waiting
    aging2 = aging_for(r2, "b")
    # Later of (settlement t=30, readiness t=20) -> 40 - 30 = 10, not 20.
    assert len(aging2) == 1
    assert aging2[0]["amount"] == pytest.approx(10.0)


def test_prerequisite_completed_before_arrival_starts_at_arrival(conn):
    # A prerequisite finished before the job even arrived never moves the
    # clock backwards: readiness is the job's own arrival.
    pre = submit("a", minutes=5, when=iso(0))
    pin_completed(pre["id"], iso(5))
    dep = submit("b", minutes=5, when=iso(30), depends_on=[pre["id"]])
    assert dep["blocked_by"] == []  # already satisfied at submission

    r = batch(60)
    aging = aging_for(r, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(30.0)  # from arrival t=30


# ---------------------------------------------------------------------------
# Revision-released readiness
# ---------------------------------------------------------------------------

def test_unblocking_revision_starts_the_clock_at_revision_time(conn):
    pre = submit("a", minutes=5, when=iso(0))
    dep = submit("b", minutes=5, when=iso(0), depends_on=[pre["id"]])
    # Detaching the dependency releases the job at the revision instant.
    t_rev = datetime.now(timezone.utc)
    scheduler.revise_dependencies(dep["id"], [])

    r = scheduler.run_batch(now=t_rev + timedelta(minutes=1))
    aging = aging_for(r, "b")
    assert len(aging) == 1
    # ~1 minute since the revision, not the full wait since iso(0).
    assert aging[0]["amount"] == pytest.approx(1.0, abs=0.01)


def test_unblocking_revision_ignores_older_prerequisite_completions(conn):
    # The revision attaches an already-completed prerequisite: the job was
    # blocked until the revision, so the revision instant governs.
    done = submit("a", minutes=5, when=iso(0))
    pin_completed(done["id"], iso(5))
    pre = submit("a", minutes=5, when=iso(0))
    dep = submit("b", minutes=5, when=iso(0), depends_on=[pre["id"]])

    t_rev = datetime.now(timezone.utc)
    scheduler.revise_dependencies(dep["id"], [done["id"]])

    r = scheduler.run_batch(now=t_rev + timedelta(minutes=2))
    aging = aging_for(r, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(2.0, abs=0.01)


def test_reblocking_revision_clears_the_readiness_clock(conn):
    p1 = submit("a", minutes=5, when=iso(0))
    p2 = submit("a", minutes=5, when=iso(0))
    dep = submit("b", minutes=5, when=iso(0), depends_on=[p1["id"]])

    scheduler.revise_dependencies(dep["id"], [])            # releases the job
    scheduler.revise_dependencies(dep["id"], [p2["id"]])    # blocks it again
    pin_completed(p2["id"], iso(50))

    r = batch(60)
    aging = aging_for(r, "b")
    assert len(aging) == 1
    # The cleared clock falls back to the last completion (t=50); a stale
    # revision timestamp would have clamped the wait to zero.
    assert aging[0]["amount"] == pytest.approx(10.0)


def test_ready_to_ready_revision_keeps_the_clock(conn):
    # A revision that never blocks the job must not restart its wait.
    p1 = submit("a", minutes=5, when=iso(0))
    p2 = submit("a", minutes=5, when=iso(0))
    dep = submit("b", minutes=5, when=iso(0))  # ready since arrival

    scheduler.revise_dependencies(dep["id"], [p1["id"]])    # now blocked
    pin_completed(p1["id"], iso(10))                        # ready since t=10
    # Ready -> ready revision (p2 completed even earlier): clock preserved.
    pin_completed(p2["id"], iso(5))
    scheduler.revise_dependencies(dep["id"], [p2["id"]])

    r = batch(60)
    aging = aging_for(r, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(50.0)  # 60 - 10


# ---------------------------------------------------------------------------
# Preview parity and zero writes
# ---------------------------------------------------------------------------

def test_preview_matches_batch_and_writes_nothing(conn):
    pre = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[pre["id"]])
    pin_completed(pre["id"], iso(59))

    c = db.get_conn()
    state_before = {
        "ledger": c.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()["n"],
        "batches": c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"],
        "balance": scheduler.get_tenant("b")["balance_gpu_min"],
        "aging": scheduler.get_tenant("b")["aging_settled_at"],
        "status": scheduler.get_job(dep["id"])["status"],
    }
    preview = scheduler.preview_batch(now=at(60))
    # Zero writes: ledger, batches, balance, aging clock and status intact.
    assert c.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()["n"] \
        == state_before["ledger"]
    assert c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] \
        == state_before["batches"]
    assert scheduler.get_tenant("b")["balance_gpu_min"] \
        == state_before["balance"]
    assert scheduler.get_tenant("b")["aging_settled_at"] \
        == state_before["aging"]
    assert scheduler.get_job(dep["id"])["status"] == state_before["status"]

    committed = scheduler.run_batch(now=at(60))
    # Same state and reference instant -> identical ledger amounts.
    assert [
        (e["kind"], e["tenant_id"], e["amount"], e["balance_after"])
        for e in preview["events"]
    ] == [
        (e["kind"], e["tenant_id"], e["amount"], e["balance_after"])
        for e in committed["events"]
    ]
    aging = aging_for(committed, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def test_http_preview_and_batch_share_the_readiness_path(client):
    pre = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 10}).json()["id"]
    dep = client.post("/jobs", json={
        "tenant_id": "b", "gpus": 1, "duration_min": 10,
        "depends_on": [pre]}).json()["id"]
    # Head queued since t=0; prerequisite completed at t=59.
    from app import db as db_mod
    db_mod.get_conn().execute(
        "UPDATE jobs SET queued_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00+00:00", dep),
    )
    db_mod.get_conn().execute(
        "UPDATE jobs SET status = 'COMPLETED', completed_at = ? WHERE id = ?",
        ("2026-01-01T00:59:00+00:00", pre),
    )

    now = {"now": "2026-01-01T01:00:00Z"}
    p = client.post("/scheduler/preview", json=now)
    assert p.status_code == 200, p.text
    p_aging = [e for e in p.json()["events"]
               if e["kind"] == "AGING" and e["tenant_id"] == "b"]
    assert len(p_aging) == 1
    assert p_aging[0]["amount"] == pytest.approx(1.0)
    # The preview persisted nothing.
    assert client.get("/ledger").json() == []
    assert client.get(f"/jobs/{dep}").json()["status"] == "QUEUED"

    b = client.post("/scheduler/batch", json=now)
    assert b.status_code == 200, b.text
    b_aging = [e for e in b.json()["events"]
               if e["kind"] == "AGING" and e["tenant_id"] == "b"]
    # Same state and instant -> the committed ledger matches the preview.
    assert [e["amount"] for e in b_aging] == [e["amount"] for e in p_aging]
    assert b.json()["selection_order"] == p.json()["selection_order"]
