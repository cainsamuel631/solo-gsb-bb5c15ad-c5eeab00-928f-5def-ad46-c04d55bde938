#!/usr/bin/env python3
"""Dependency-free smoke test for the scheduling core (stdlib sqlite3 only).

Run:  python3 scripts/verify_core.py

It spins an in-memory database, exercises weighted DRR, aging, capacity
limits, cancellation and reproducibility, and asserts the exact outcomes.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app import scheduler  # noqa: E402

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def at(m: float) -> datetime:
    return EPOCH + timedelta(minutes=m)


def iso(m: float) -> str:
    return at(m).isoformat()


def main() -> int:
    os.environ.update(
        DB_PATH=":memory:",
        CLUSTER_GPUS="8",
        BASE_QUANTUM_GPU_MIN="60",
        AGING_RATE_GPU_MIN_PER_MIN="1",
    )
    get_settings.cache_clear()

    db.reset_for_tests()
    db.connect(":memory:")

    # --- Case 1: WDRR selection order -------------------------------------
    def submit(tid, gpus, mins, weight, when):
        job = scheduler.submit_job(tid, gpus, mins, weight)
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"]))
        return job

    submit("a", 1, 30, 1.0, iso(0))
    for i in range(3):
        submit("b", 1, 30, 3.0, iso(1 + i))

    r = scheduler.run_batch(now=at(10))
    assert r["selection_order"] == ["J000001", "J000002", "J000003", "J000004"], r["selection_order"]
    bal = {t["id"]: t["balance_gpu_min"] for t in r["tenants"]}
    # a: aging 10 + quantum 60 - cost 30 = 40
    # b: aging ~9 + 3 quanta x 180 - cost 90 = 459
    assert abs(bal["a"] - 40.0) < 1e-9, bal
    assert abs(bal["b"] - 459.0) < 1e-9, bal
    aging = {e["tenant_id"]: e["amount"]
             for e in r["events"] if e["kind"] == "AGING"}
    assert abs(aging["a"] - 10.0) < 1e-9
    assert abs(aging["b"] - 9.0) < 1e-9
    print("case 1 weighted DRR order/balance OK")

    # fresh DB for capacity case
    db.reset_for_tests()
    db.connect(":memory:")

    # --- Case 2: capacity never exceeded; blocked keeps credit ------------
    submit("a", 6, 10, 1.0, iso(0))
    submit("b", 6, 10, 1.0, iso(1))
    r1 = scheduler.run_batch(now=at(5))
    assert r1["selection_order"] == ["J000001"]
    r2 = scheduler.run_batch(now=at(6))
    assert r2["selection_order"] == []  # b blocked on capacity
    b_bal_before = scheduler.get_tenant("b")["balance_gpu_min"]
    # no quantum farming on no-op repeated batch
    r2b = scheduler.run_batch(now=at(6))
    assert r2b["selection_order"] == []
    assert scheduler.get_tenant("b")["balance_gpu_min"] == b_bal_before
    scheduler.complete_job("J000001")
    r3 = scheduler.run_batch(now=at(7))
    assert r3["selection_order"] == ["J000002"]
    assert scheduler.scheduler_state()["used_gpus"] == 6
    print("case 2 capacity constraint + retained credit OK")

    # --- Case 3: multi-round accrual within one batch + cross-batch carry --
    db.reset_for_tests()
    db.connect(":memory:")
    # Two large 6-GPU jobs, cost 600 each (6 x 100), quantum 60/round ->
    # each needs 10 rounds before it can dispatch.
    submit("a", 6, 100, 1.0, iso(0))
    submit("b", 6, 100, 1.0, iso(1))
    r = scheduler.run_batch(now=at(10))
    # Aging applies from the jobs' arrivals too: a waited 10, b 9 min.
    assert r["selection_order"] == ["J000001"], r["selection_order"]
    assert r["rounds"] == 10, r["rounds"]
    assert abs(scheduler.get_tenant("a")["balance_gpu_min"] - 10.0) < 1e-9
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 549.0) < 1e-9

    # No farming while purely capacity-blocked (same instant -> no aging):
    blocked = scheduler.run_batch(now=at(10))
    assert blocked["selection_order"] == []
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 549.0) < 1e-9
    assert all(e["kind"] != "QUANTUM" for e in blocked["events"])

    # Capacity frees up: 549 + 2 more aging + 60 quantum - 600 cost = 11.
    scheduler.complete_job("J000001")
    r2 = scheduler.run_batch(now=at(12))
    assert r2["selection_order"] == ["J000002"]
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 11.0) < 1e-9
    aging = [e for e in r2["events"] if e["kind"] == "AGING"]
    assert abs(aging[0]["amount"] - 2.0) < 1e-9
    print("case 3 multi-round accrual + cross-batch carry OK")

    # --- Case 4: aging compensates a waiting small job ---------------------
    db.reset_for_tests()
    db.connect(":memory:")
    # Heavy tenant 'a' (sorts first) saturates the cluster with an 8-GPU job;
    # small tenant 'b' has a tiny job that cannot fit afterwards.
    submit("a", 8, 10, 4.0, iso(0))   # cost 80, quantum 240 -> runs now
    submit("b", 1, 60, 1.0, iso(0))  # cost 60, blocked by capacity
    r1 = scheduler.run_batch(now=at(10))
    assert r1["selection_order"] == ["J000001"], r1["selection_order"]
    assert scheduler.scheduler_state()["used_gpus"] == 8
    # a: aging 10 + quantum 240 - cost 80 = 170.
    # b got no quantum (capacity-blocked) but keeps aging credit 10.
    assert abs(scheduler.get_tenant("a")["balance_gpu_min"] - 170.0) < 1e-9
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 10.0) < 1e-9
    scheduler.complete_job("J000001")
    r2 = scheduler.run_batch(now=at(120))
    assert r2["selection_order"] == ["J000002"], r2["selection_order"]
    # b: 10 carried aging + 110 more + quantum 60 - cost 60 = 120.
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 120.0) < 1e-9
    aging = [e for e in r2["events"] if e["kind"] == "AGING"]
    assert aging and abs(aging[0]["amount"] - 110.0) < 1e-6, aging
    # aging window resets after drain
    assert scheduler.get_tenant("b")["aging_settled_at"] is None
    print("case 4 aging compensation + window reset OK")

    # --- Case 5: oversized jobs rejected ----------------------------------
    db.reset_for_tests()
    db.connect(":memory:")
    try:
        scheduler.submit_job("a", 99, 1, 1.0)
    except scheduler.SchedulerError:
        pass
    else:
        raise AssertionError("oversized job should be rejected")
    print("case 5 oversized rejection OK")

    # --- Case 6: queue + cancel -------------------------------------------
    submit("a", 2, 10, 1.0, iso(0))
    submit("a", 2, 10, 1.0, iso(1))
    r = scheduler.run_batch(now=at(2))
    assert r["selection_order"] == ["J000001", "J000002"]
    scheduler.cancel_job("J000002")
    assert scheduler.scheduler_state()["used_gpus"] == 2
    q = scheduler.queue_overview()
    assert q["running_jobs"] == 1 and q["queued_jobs"] == 0
    print("case 6 cancel/queue state OK")

    # --- Case 7: job dependencies (cross-tenant, blocking, release) --------
    db.reset_for_tests()
    db.connect(":memory:")
    pre = submit("a", 1, 10, 1.0, iso(0))
    dep = scheduler.submit_job("b", 1, 10, 1.0, depends_on=[pre["id"]])
    db.get_conn().execute(
        "UPDATE jobs SET queued_at = ? WHERE id = ?", (iso(1), dep["id"]))
    dep = scheduler._get_job_or_404(db.get_conn(), dep["id"])
    assert dep["depends_on"] == ["J000001"]
    assert dep["blocked_by"] == [{"id": "J000001", "status": "QUEUED"}]

    # Unknown prerequisite rejected.
    try:
        scheduler.submit_job("c", 1, 10, depends_on=["J000999"])
    except scheduler.SchedulerError as e:
        assert "unknown depends_on" in str(e)
    else:
        raise AssertionError("unknown dependency should be rejected")

    r = scheduler.run_batch(now=at(10))
    assert r["selection_order"] == ["J000001"], r["selection_order"]
    # Blocked tenant earns nothing (no aging, no quantum).
    assert scheduler.get_tenant("b")["balance_gpu_min"] == 0.0
    assert all(e["tenant_id"] != "b" for e in r["events"])
    assert scheduler.run_batch(now=at(20))["selection_order"] == []

    scheduler.complete_job("J000001")
    r2 = scheduler.run_batch(now=at(30))
    assert r2["selection_order"] == ["J000002"], r2["selection_order"]

    # Cancelled prerequisite blocks forever, with a queryable reason.
    p3 = scheduler.submit_job("a", 1, 10)
    d3 = scheduler.submit_job("b", 1, 10, depends_on=[p3["id"]])
    scheduler.cancel_job(p3["id"])
    assert scheduler.run_batch(now=at(40))["selection_order"] == []
    assert scheduler.get_job(d3["id"])["blocked_by"] == [
        {"id": p3["id"], "status": "CANCELLED"}]
    print("case 7 dependencies + blocked_by OK")

    # --- Case 8: per-tenant concurrent GPU caps ---------------------------
    db.reset_for_tests()
    db.connect(":memory:")
    t = scheduler.upsert_tenant("a", 1.0, max_gpus=2)
    assert t["max_gpus"] == 2 and t["running_gpus"] == 0
    submit("a", 2, 10, 1.0, iso(0))  # J1 exactly fills the cap
    submit("a", 1, 10, 1.0, iso(1))  # J2 cap-blocked after J1
    j3 = submit("b", 4, 10, 1.0, iso(2))  # unlimited tenant still runs
    r = scheduler.run_batch(now=at(3))
    assert r["selection_order"] == ["J000001", "J000003"], r["selection_order"]
    assert scheduler.get_tenant("a")["running_gpus"] == 2
    # Cap-blocked head earns no quantum this round.
    blocked = scheduler.run_batch(now=at(3))
    assert blocked["selection_order"] == []
    assert all(
        e["kind"] != "QUANTUM" for e in blocked["events"]
        if e["tenant_id"] == "a"
    )
    # Completing releases cap headroom; J2 dispatches.
    scheduler.complete_job("J000001")
    scheduler.complete_job(j3["id"])
    r2 = scheduler.run_batch(now=at(4))
    assert r2["selection_order"] == ["J000002"], r2["selection_order"]
    # Omit keeps the cap; explicit None clears it.
    assert scheduler.upsert_tenant("a", 2.0)["max_gpus"] == 2
    assert scheduler.upsert_tenant("a", 2.0, max_gpus=None)["max_gpus"] is None
    print("case 8 per-tenant GPU caps OK")

    # --- Case 9: legacy database without tenants.max_gpus migrates --------
    import sqlite3
    legacy = os.path.join(tempfile.gettempdir(), "legacy_scheduler.db")
    if os.path.exists(legacy):
        os.remove(legacy)
    lc = sqlite3.connect(legacy)
    lc.executescript(
        """
        CREATE TABLE tenants (
            id TEXT PRIMARY KEY, weight REAL NOT NULL, deficit REAL NOT NULL
                DEFAULT 0, aging_settled_at TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            now_ref TEXT NOT NULL, capacity_gpus INTEGER NOT NULL,
            used_gpus_before INTEGER NOT NULL, available_gpus INTEGER NOT NULL,
            rounds INTEGER NOT NULL, jobs_selected INTEGER NOT NULL);
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, gpus INTEGER NOT NULL,
            duration_min INTEGER NOT NULL, cost_gpu_min REAL NOT NULL,
            status TEXT NOT NULL, queued_at TEXT NOT NULL, scheduled_at TEXT,
            completed_at TEXT, batch_id INTEGER, run_index INTEGER);
        CREATE TABLE job_dependencies (
            job_id TEXT NOT NULL, depends_on TEXT NOT NULL,
            PRIMARY KEY (job_id, depends_on));
        CREATE TABLE ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL,
            batch_id INTEGER, job_id TEXT, kind TEXT NOT NULL,
            amount REAL NOT NULL, balance_after REAL NOT NULL,
            detail TEXT, created_at TEXT NOT NULL);
        """
    )
    lc.execute(
        "INSERT INTO tenants (id, weight, deficit, created_at, updated_at) "
        "VALUES ('old', 1.0, 0, ?, ?)",
        (iso(0), iso(0)),
    )
    lc.execute(
        "INSERT INTO jobs (id, tenant_id, gpus, duration_min, cost_gpu_min, "
        "status, queued_at) VALUES ('J000001', 'old', 8, 1, 8, 'QUEUED', ?)",
        (iso(0),),
    )
    lc.commit()
    lc.close()
    db.reset_for_tests()
    db.connect(legacy)
    assert scheduler.get_tenant("old")["max_gpus"] is None
    r = scheduler.run_batch(now=at(5))
    assert r["selection_order"] == ["J000001"]
    assert scheduler.upsert_tenant("new", 1.0, max_gpus=4)["max_gpus"] == 4
    db.reset_for_tests()
    os.remove(legacy)
    print("case 9 legacy schema migration OK")

    # --- Case 10: GPU maintenance windows shape dispatch -------------------
    db.reset_for_tests()
    db.connect(":memory:")
    w = scheduler.create_maintenance_window(at(30), at(90), 2)
    assert w["reserved_gpus"] == 2
    assert [x["id"] for x in scheduler.list_maintenance_windows(at(30))] == [1]
    assert scheduler.list_maintenance_windows(at(90)) == []  # half-open end

    # Validation: end <= start, reservation beyond capacity.
    try:
        scheduler.create_maintenance_window(at(10), at(10), 1)
    except scheduler.SchedulerError:
        pass
    else:
        raise AssertionError("zero-length window should be rejected")
    try:
        scheduler.create_maintenance_window(at(0), at(10), 9)
    except scheduler.SchedulerError:
        pass
    else:
        raise AssertionError("over-capacity reservation should be rejected")

    # Cumulative admission: the peak combined reservation over a candidate's
    # coverage (persisted windows + candidate) may not exceed capacity.
    scheduler.create_maintenance_window(at(600), at(720), 5)
    try:
        scheduler.create_maintenance_window(at(660), at(780), 4)
    except scheduler.SchedulerError:
        pass
    else:
        raise AssertionError("overlapping peak 5+4 > 8 should be rejected")
    assert len(scheduler.list_maintenance_windows()) == 2  # nothing persisted
    # Peak exactly at capacity is allowed; touching endpoints never add.
    scheduler.create_maintenance_window(at(660), at(780), 3)
    scheduler.create_maintenance_window(at(780), at(900), 8)
    assert len(scheduler.list_maintenance_windows()) == 4

    # 8-GPU job predicts [0,60); the 2-GPU reservation over [30,90) blocks it.
    submit("a", 8, 60, 1.0, iso(0))
    r = scheduler.run_batch(now=at(0))
    assert r["selection_order"] == [], r["selection_order"]

    # A short 3-GPU job [0,20) ends before the window: it dispatches.
    submit("b", 3, 20, 1.0, iso(0))
    r = scheduler.run_batch(now=at(0))
    assert r["selection_order"] == ["J000002"], r["selection_order"]
    j = r["selected_jobs"][0]
    assert j["estimated_end_at"] == iso(20)
    assert j["max_maintenance_reserved_gpus"] == 0

    # Endpoint-touching windows never stack reservations.
    db.reset_for_tests()
    db.connect(":memory:")
    submit("c", 5, 60, 1.0, iso(0))
    scheduler.create_maintenance_window(at(0), at(30), 3)
    scheduler.create_maintenance_window(at(30), at(60), 3)
    r = scheduler.run_batch(now=at(0))
    assert r["selection_order"] == ["J000001"], r["selection_order"]
    assert r["selected_jobs"][0]["max_maintenance_reserved_gpus"] == 3

    # RUNNING jobs are never preempted by a newly created window. The new
    # window's 5 plus the persisted touching 3s peak exactly at capacity
    # (admissible), so the cluster is fully reserved over [0,60).
    scheduler.create_maintenance_window(at(0), at(60), 5)
    assert scheduler.get_job("J000001")["status"] == "RUNNING"
    submit("d", 1, 10, 1.0, iso(0))
    blocked = scheduler.run_batch(now=at(0))
    assert blocked["selection_order"] == []
    assert scheduler.get_job("J000001")["status"] == "RUNNING"

    # Preview shares the judgement and writes nothing (window survives, no
    # extra batch rows appear).
    batches_before = db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM batches").fetchone()["n"]
    p = scheduler.preview_batch(now=at(60))
    assert p["preview"] is True
    assert p["selection_order"] == ["J000002"]
    assert db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == batches_before
    assert len(scheduler.list_maintenance_windows()) == 3
    print("case 10 maintenance windows admission/preview OK")

    # --- Case 11: rolling 24h GPU-minute budgets ---------------------------
    db.reset_for_tests()
    db.connect(":memory:")
    t = scheduler.upsert_tenant("a", 1.0, budget_gpu_min=100)
    assert t["budget_gpu_min"] == 100 and t["budget_used_gpu_min"] == 0
    assert t["budget_remaining_gpu_min"] == 100

    # 30 x 3 = 90 admitted; a fourth 30 (120 > 100) waits.
    for _ in range(4):
        submit("a", 1, 30, 1.0, iso(0))
    jb = submit("b", 2, 10, 1.0, iso(0))  # unlimited tenant, unaffected
    r = scheduler.run_batch(now=at(5))
    # a's J1 (round 1), b's J5 (round 1), then a's J2/J3 (rounds 2/3);
    # a's J4 would take the rolling spend to 120 > 100 and stays queued.
    assert r["selection_order"] == [
        "J000001", "J000005", "J000002", "J000003"], r["selection_order"]
    a = next(x for x in r["tenants"] if x["id"] == "a")
    assert a["budget_used_gpu_min"] == 90
    assert a["budget_remaining_gpu_min"] == 10
    # Budget-blocked head earned no quantum; the other tenant dispatched.
    blocked = scheduler.run_batch(now=at(6))
    assert blocked["selection_order"] == []
    assert all(
        e["kind"] != "QUANTUM" for e in blocked["events"]
        if e["tenant_id"] == "a"
    )

    # Completion does not refund the charge.
    scheduler.complete_job("J000001")
    rr = scheduler.run_batch(now=at(7))
    assert rr["selection_order"] == []

    # Clear the budget: the blocked head dispatches on the next batch.
    assert scheduler.upsert_tenant(
        "a", 1.0, budget_gpu_min=None)["budget_gpu_min"] is None
    r2 = scheduler.run_batch(now=at(8))
    assert r2["selection_order"] == ["J000004"], r2["selection_order"]

    # Rolling window: a charge older than 24h drops out (left-open window).
    db.reset_for_tests()
    db.connect(":memory:")
    scheduler.upsert_tenant("c", 1.0, budget_gpu_min=60)
    old = submit("c", 1, 60, 1.0, iso(0))
    scheduler.run_batch(now=at(10))
    db.get_conn().execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?", (iso(10), old["id"]))
    # One minute before the edge the old charge still blocks a fresh job.
    fresh = submit("c", 1, 60, 1.0, iso(10 + 24 * 60 - 1))
    rb = scheduler.run_batch(now=at(10 + 24 * 60 - 1))
    assert rb["selection_order"] == [], rb["selection_order"]
    # Exactly 24h later: (now-24h, now] no longer contains the old charge.
    scheduler.cancel_job(fresh["id"])
    edge = submit("c", 1, 60, 1.0, iso(10 + 24 * 60))
    re = scheduler.run_batch(now=at(10 + 24 * 60))
    assert re["selection_order"] == [edge["id"]], re["selection_order"]

    # Preview shares the path and persists nothing.
    db.reset_for_tests()
    db.connect(":memory:")
    scheduler.upsert_tenant("d", 1.0, budget_gpu_min=10)
    submit("d", 1, 60, 1.0, iso(0))
    p = scheduler.preview_batch(now=at(5))
    assert p["preview"] is True and p["selection_order"] == []
    assert db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 0
    print("case 11 rolling 24h GPU-minute budgets OK")

    # --- Case 12: dependency revision on queued jobs -----------------------
    db.reset_for_tests()
    db.connect(":memory:")
    p1 = submit("a", 1, 10, 1.0, iso(0))                     # J000001
    p2 = submit("a", 1, 10, 1.0, iso(1))                     # J000002
    dep = scheduler.submit_job("b", 1, 10, 1.0, depends_on=[p1["id"]])
    db.get_conn().execute(
        "UPDATE jobs SET queued_at = ? WHERE id = ?", (iso(2), dep["id"]))
    jx = submit("c", 1, 10, 1.0, iso(3))                     # J000004

    # Wholesale replacement: deduplicated, sorted, cross-tenant.
    rev = scheduler.revise_dependencies(dep["id"], [jx["id"], p2["id"], jx["id"]])
    assert rev["depends_on"] == [p2["id"], jx["id"]]
    assert rev["blocked_by"] == [
        {"id": p2["id"], "status": "QUEUED"},
        {"id": jx["id"], "status": "QUEUED"},
    ]

    # Rejections leave the stored edges untouched.
    for bad, needle in (
        (["J000999"], "unknown depends_on"),
        ([dep["id"]], "dependency cycle"),
    ):
        try:
            scheduler.revise_dependencies(dep["id"], bad)
        except scheduler.SchedulerError as e:
            assert needle in str(e)
        else:
            raise AssertionError(f"revision {bad} should be rejected")
    # A cycle through the replacement is rejected too (p2 is a prereq of dep).
    try:
        scheduler.revise_dependencies(p2["id"], [dep["id"]])
    except scheduler.SchedulerError as e:
        assert "dependency cycle" in str(e)
    else:
        raise AssertionError("cycle-closing revision should be rejected")
    assert scheduler.get_job(dep["id"])["depends_on"] == [p2["id"], jx["id"]]

    # Empty list detaches; the next preview/batch judge the new edges.
    rev = scheduler.revise_dependencies(dep["id"], [])
    assert rev["depends_on"] == [] and rev["blocked_by"] == []
    p = scheduler.preview_batch(now=at(10))
    assert p["selection_order"] == [p1["id"], dep["id"], jx["id"], p2["id"]], \
        p["selection_order"]
    r = scheduler.run_batch(now=at(10))
    assert r["selection_order"] == [p1["id"], dep["id"], jx["id"], p2["id"]]

    # Only QUEUED jobs can be revised.
    try:
        scheduler.revise_dependencies(dep["id"], [])
    except scheduler.Conflict:
        pass
    else:
        raise AssertionError("RUNNING job revision should be rejected")

    # The submission-time idempotency fingerprint is preserved: replaying the
    # original request returns the revised job without restoring old edges.
    job, created = scheduler.submit_job_idempotent(
        "d", 1, 10, 1.0, [p1["id"]], idempotency_key="rev-key")
    assert created and job["depends_on"] == [p1["id"]]
    scheduler.revise_dependencies(job["id"], [])
    replay, created2 = scheduler.submit_job_idempotent(
        "d", 1, 10, 1.0, [p1["id"]], idempotency_key="rev-key")
    assert not created2
    assert replay["id"] == job["id"] and replay["depends_on"] == []
    print("case 12 dependency revision OK")

    # --- Case 13: maintenance-slot queries ---------------------------------
    db.reset_for_tests()
    db.connect(":memory:")
    j = submit("a", 2, 60, 1.0, iso(0))
    scheduler.create_maintenance_window(at(100), at(200), 7)
    r = scheduler.job_maintenance_slots(j["id"], at(0), at(300))
    # 7 of 8 GPUs reserved over [100,200): starts in (40, 200) are blocked;
    # the run [40,100) merely touches the window and stays feasible.
    assert r["slots"] == [
        {"start_at": iso(0), "end_at": iso(40),
         "start_inclusive": True, "end_inclusive": True},
        {"start_at": iso(200), "end_at": iso(300),
         "start_inclusive": True, "end_inclusive": False},
    ], r["slots"]
    assert r["earliest_start_at"] == iso(0)
    assert r["running_baseline_gpus"] == 0

    # A second 7-GPU window exactly one duration later leaves a single
    # isolated feasible instant (t=200) between the two blocked regions.
    scheduler.create_maintenance_window(at(260), at(320), 7)
    r = scheduler.job_maintenance_slots(j["id"], at(0), at(300))
    assert r["slots"] == [
        {"start_at": iso(0), "end_at": iso(40),
         "start_inclusive": True, "end_inclusive": True},
        {"start_at": iso(200), "end_at": iso(200),
         "start_inclusive": True, "end_inclusive": True},
    ], r["slots"]

    # Only QUEUED jobs have slots; the RUNNING baseline is fixed (no future
    # release is speculated) and shrinks the headroom for everyone else.
    assert scheduler.run_batch(now=at(0))["selection_order"] == ["J000001"]
    try:
        scheduler.job_maintenance_slots(j["id"], at(0), at(300))
    except scheduler.Conflict:
        pass
    else:
        raise AssertionError("RUNNING job should not have maintenance slots")
    j2 = submit("b", 7, 60, 1.0, iso(0))  # baseline 2 + 7 > 8 everywhere
    r = scheduler.job_maintenance_slots(j2["id"], at(0), at(300))
    assert r["running_baseline_gpus"] == 2
    assert r["slots"] == [] and r["earliest_start_at"] is None
    print("case 13 maintenance-slot queries OK")

    # --- Case 14: dependency cancel cascade -------------------------------
    db.reset_for_tests()
    db.connect(":memory:")
    t = submit("a", 8, 10, 1.0, iso(0))                    # J000001
    x = scheduler.submit_job("b", 1, 10, 1.0, depends_on=[t["id"]])
    y = scheduler.submit_job("c", 1, 10, 1.0, depends_on=[x["id"]])
    z = scheduler.submit_job(
        "b", 1, 10, 1.0, depends_on=[t["id"], y["id"]])

    # Preview: transitive closure, id-sorted, one shortest chain each.
    impact = scheduler.cancel_impact(t["id"])
    assert impact["affected_count"] == 3
    assert [j["id"] for j in impact["affected_jobs"]] == [
        x["id"], y["id"], z["id"]]
    chains = {j["id"]: j["chain"] for j in impact["affected_jobs"]}
    assert chains[x["id"]] == [t["id"], x["id"]]
    assert chains[y["id"]] == [t["id"], x["id"], y["id"]]
    assert chains[z["id"]] == [t["id"], z["id"]]  # direct edge is shortest
    # The preview writes nothing.
    assert all(
        scheduler.get_job(j)["status"] == "QUEUED"
        for j in [t["id"], x["id"], y["id"], z["id"]]
    )

    # Dispatch the target, then cascade: the RUNNING occupancy is released
    # and every still-QUEUED descendant is cancelled atomically.
    r = scheduler.run_batch(now=at(1))
    assert r["selection_order"] == [t["id"]]
    assert scheduler.scheduler_state()["used_gpus"] == 8
    res = scheduler.cancel_cascade(t["id"])
    assert res["cancelled"] == [t["id"], x["id"], y["id"], z["id"]]
    assert scheduler.scheduler_state()["used_gpus"] == 0
    assert scheduler.queue_overview()["queued_jobs"] == 0

    # Already-cancelled intermediate nodes are traversal-only: the closure
    # passes through them but the cascade never re-modifies them.
    db.reset_for_tests()
    db.connect(":memory:")
    a = submit("a", 1, 10, 1.0, iso(0))
    b = scheduler.submit_job("b", 1, 10, 1.0, depends_on=[a["id"]])
    c = scheduler.submit_job("c", 1, 10, 1.0, depends_on=[b["id"]])
    scheduler.cancel_job(b["id"])
    b_cancelled_at = scheduler.get_job(b["id"])["completed_at"]
    impact = scheduler.cancel_impact(a["id"])
    assert [j["id"] for j in impact["affected_jobs"]] == [c["id"]]
    assert impact["affected_jobs"][0]["chain"] == [a["id"], b["id"], c["id"]]
    res = scheduler.cancel_cascade(a["id"])
    assert res["cancelled"] == [a["id"], c["id"]]
    assert scheduler.get_job(b["id"])["completed_at"] == b_cancelled_at

    # 404/409 admissibility on both endpoints.
    for fn in (scheduler.cancel_impact, scheduler.cancel_cascade):
        try:
            fn("J000999")
        except scheduler.NotFound:
            pass
        else:
            raise AssertionError("unknown target should 404")
        try:
            fn(a["id"])  # already CANCELLED by the cascade above
        except scheduler.Conflict:
            pass
        else:
            raise AssertionError("cancelled target should 409")
    print("case 14 dependency cancel cascade OK")

    # --- Case 15: aging accrues only from true head readiness -------------
    db.reset_for_tests()
    db.connect(":memory:")
    pre = submit("a", 1, 10, 1.0, iso(0))
    dep = scheduler.submit_job("b", 1, 10, 1.0, depends_on=[pre["id"]])
    db.get_conn().execute(
        "UPDATE jobs SET queued_at = ? WHERE id = ?", (iso(0), dep["id"]))
    # The prerequisite completes one minute before the batch and no batch
    # ran in between: the dependent waited 60 min since arrival, but only
    # the single truly-ready minute earns aging credit.
    db.get_conn().execute(
        "UPDATE jobs SET status = 'COMPLETED', completed_at = ? WHERE id = ?",
        (iso(59), pre["id"]))
    r = scheduler.run_batch(now=at(60))
    assert r["selection_order"] == [dep["id"]]
    aging = [e for e in r["events"]
             if e["kind"] == "AGING" and e["tenant_id"] == "b"]
    assert len(aging) == 1 and abs(aging[0]["amount"] - 1.0) < 1e-9, aging

    # Several prerequisites: the LAST completion releases the job.
    db.reset_for_tests()
    db.connect(":memory:")
    p1 = submit("a", 1, 5, 1.0, iso(0))
    p2 = submit("a", 1, 5, 1.0, iso(0))
    dep = scheduler.submit_job("b", 1, 5, 1.0, depends_on=[p1["id"], p2["id"]])
    db.get_conn().execute(
        "UPDATE jobs SET queued_at = ? WHERE id = ?", (iso(0), dep["id"]))
    for jid, ts in ((p1["id"], iso(10)), (p2["id"], iso(50))):
        db.get_conn().execute(
            "UPDATE jobs SET status = 'COMPLETED', completed_at = ? "
            "WHERE id = ?", (ts, jid))
    r = scheduler.run_batch(now=at(60))
    aging = [e for e in r["events"]
             if e["kind"] == "AGING" and e["tenant_id"] == "b"]
    assert len(aging) == 1 and abs(aging[0]["amount"] - 10.0) < 1e-9, aging

    # A dependency revision that unblocks the head starts the clock at the
    # revision instant; the preview agrees with the committed batch and
    # persists nothing.
    db.reset_for_tests()
    db.connect(":memory:")
    pre = submit("a", 1, 5, 1.0, iso(0))
    dep = scheduler.submit_job("b", 1, 5, 1.0, depends_on=[pre["id"]])
    db.get_conn().execute(
        "UPDATE jobs SET queued_at = ? WHERE id = ?", (iso(0), dep["id"]))
    t_rev = datetime.now(timezone.utc)
    scheduler.revise_dependencies(dep["id"], [])
    ref = t_rev + timedelta(minutes=1)
    p = scheduler.preview_batch(now=ref)
    assert db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM ledger").fetchone()["n"] == 0
    r = scheduler.run_batch(now=ref)
    aging = [e for e in r["events"]
             if e["kind"] == "AGING" and e["tenant_id"] == "b"]
    assert len(aging) == 1 and abs(aging[0]["amount"] - 1.0) < 0.01, aging
    assert [(e["kind"], e["amount"]) for e in p["events"]] == [
        (e["kind"], e["amount"]) for e in r["events"]]
    print("case 15 readiness-aware aging OK")

    print("\nALL CORE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
