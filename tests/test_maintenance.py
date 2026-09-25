"""GPU maintenance windows: persistence, validation and dispatch admission.

A maintenance window is a half-open UTC interval ``[start_at, end_at)`` with
a number of GPUs reserved. During a batch every selected job is predicted to
run ``[now, now + duration_min)``; before each dispatch the peak combined
reservation overlapping that interval is subtracted from cluster capacity.
These tests cover the service and HTTP levels, including preview parity.
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


def window(start_min, end_min, reserved):
    return scheduler.create_maintenance_window(
        at(start_min), at(end_min), reserved
    )


# ---------------------------------------------------------------------------
# Persistence / querying / validation
# ---------------------------------------------------------------------------

def test_create_and_list_windows_persisted(conn):
    w = window(100, 120, 4)
    assert w["id"] == 1
    assert w["reserved_gpus"] == 4
    assert w["start_at"] == iso(100)
    assert w["end_at"] == iso(120)
    assert "created_at" in w

    w2 = window(200, 220, 2)
    listed = scheduler.list_maintenance_windows()
    assert [x["id"] for x in listed] == [w["id"], w2["id"]]

    # Half-open active filter: [start, end) -> active at start, not at end.
    assert [x["id"] for x in scheduler.list_maintenance_windows(at(100))] == [1]
    assert [x["id"] for x in scheduler.list_maintenance_windows(at(119))] == [1]
    assert scheduler.list_maintenance_windows(at(120)) == []
    assert scheduler.list_maintenance_windows(at(99)) == []
    assert [x["id"] for x in scheduler.list_maintenance_windows(at(200))] == [2]


def test_window_rejects_end_not_later_than_start(conn):
    with pytest.raises(scheduler.SchedulerError, match="end_at"):
        window(10, 10, 1)
    with pytest.raises(scheduler.SchedulerError, match="end_at"):
        window(10, 5, 1)


def test_window_rejects_reservation_outside_capacity(conn):
    with pytest.raises(scheduler.SchedulerError, match="cluster capacity"):
        window(0, 10, 9)           # cluster has 8 GPUs
    with pytest.raises(scheduler.SchedulerError):
        window(0, 10, -1)
    with pytest.raises(scheduler.SchedulerError):
        scheduler.create_maintenance_window(at(0), at(10), True)
    with pytest.raises(scheduler.SchedulerError):
        scheduler.create_maintenance_window(at(0), at(10), 2.5)
    # Rejected windows leave no rows behind.
    assert scheduler.list_maintenance_windows() == []


def test_zero_reservation_window_allowed(conn):
    w = window(0, 10, 0)
    assert w["reserved_gpus"] == 0
    assert len(scheduler.list_maintenance_windows()) == 1


def test_naive_datetimes_treated_as_utc(conn):
    from datetime import datetime

    w = scheduler.create_maintenance_window(
        datetime(2026, 1, 1, 1, 0), datetime(2026, 1, 1, 2, 0), 3
    )
    assert w["start_at"] == "2026-01-01T01:00:00+00:00"
    assert w["end_at"] == "2026-01-01T02:00:00+00:00"


# ---------------------------------------------------------------------------
# Cumulative reservation admission
# ---------------------------------------------------------------------------

def test_overlapping_windows_peak_over_capacity_rejected(conn):
    # 8-GPU cluster: [10:00,12:00) reserves 5, then [11:00,13:00) reserves
    # 4 -> the overlap [11:00,12:00) would peak at 9 > 8: rejected (400 at
    # the HTTP layer) and nothing is persisted.
    w = window(10 * 60, 12 * 60, 5)
    assert w["reserved_gpus"] == 5
    with pytest.raises(scheduler.SchedulerError, match="peak"):
        window(11 * 60, 13 * 60, 4)

    # The rejected window left no row behind (and no id hole either).
    listed = scheduler.list_maintenance_windows()
    assert [x["id"] for x in listed] == [w["id"]]
    w2 = window(11 * 60, 13 * 60, 3)  # peak 5+3 = 8 == capacity: allowed
    assert w2["id"] == w["id"] + 1
    assert [x["id"] for x in scheduler.list_maintenance_windows()] == [1, 2]


def test_cumulative_peak_exactly_at_capacity_allowed(conn):
    window(10 * 60, 12 * 60, 5)
    w = window(11 * 60, 13 * 60, 3)   # overlap peak 5+3 = 8 == CLUSTER_GPUS
    assert w["reserved_gpus"] == 3
    # One more GPU anywhere inside the overlap tips the peak over capacity.
    with pytest.raises(scheduler.SchedulerError, match="cluster capacity"):
        window(11 * 60, 12 * 60, 1)
    assert len(scheduler.list_maintenance_windows()) == 2


def test_touching_or_disjoint_windows_do_not_accumulate(conn):
    # Endpoint-touching windows never overlap (half-open), so their
    # reservations are admitted independently even at full capacity each.
    window(10 * 60, 12 * 60, 8)
    w = window(12 * 60, 14 * 60, 8)   # starts exactly when the first ends
    assert w["reserved_gpus"] == 8
    w2 = window(8 * 60, 10 * 60, 8)   # ends exactly when the first starts
    assert w2["reserved_gpus"] == 8
    # Fully disjoint windows never add either.
    w3 = window(20 * 60, 22 * 60, 8)
    assert w3["reserved_gpus"] == 8
    assert len(scheduler.list_maintenance_windows()) == 4


def test_cumulative_check_uses_persisted_windows_only(conn):
    # A rejected candidate must not influence later admissions or dispatch.
    window(10 * 60, 12 * 60, 5)
    with pytest.raises(scheduler.SchedulerError):
        window(11 * 60, 13 * 60, 4)
    # The persisted state alone admits a peak of exactly 8.
    window(11 * 60, 13 * 60, 3)
    # Dispatch reads the same persisted windows: at 11:30 the active
    # reservation is 5+3 = 8, so nothing fits; at 10:30 only 5 is active.
    submit("a", 1, 10, when=iso(10 * 60))
    r = scheduler.run_batch(now=at(11 * 60 + 30))
    assert r["selection_order"] == []
    assert r["maintenance_reserved_gpus_at_start"] == 8
    r2 = scheduler.run_batch(now=at(10 * 60 + 30))
    assert r2["selection_order"] == ["J000001"]
    assert r2["maintenance_reserved_gpus_at_start"] == 5


def test_cumulative_peak_rejected_http(client):
    r = client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T10:00:00Z",
        "end_at": "2026-01-01T12:00:00Z",
        "reserved_gpus": 5})
    assert r.status_code == 201, r.text

    # Overlapping [11:00,13:00) with 4 more GPUs: peak 9 > 8 -> 400.
    r2 = client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T11:00:00Z",
        "end_at": "2026-01-01T13:00:00Z",
        "reserved_gpus": 4})
    assert r2.status_code == 400
    assert "cluster capacity" in r2.json()["detail"]
    assert [x["id"] for x in client.get("/maintenance-windows").json()] == [1]

    # Peak exactly at capacity (5+3 = 8) is admitted; touching is free.
    r3 = client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T11:00:00Z",
        "end_at": "2026-01-01T13:00:00Z",
        "reserved_gpus": 3})
    assert r3.status_code == 201, r3.text
    r4 = client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T13:00:00Z",
        "end_at": "2026-01-01T14:00:00Z",
        "reserved_gpus": 8})
    assert r4.status_code == 201, r4.text
    assert [x["id"] for x in client.get("/maintenance-windows").json()] \
        == [1, 2, 3]


# ---------------------------------------------------------------------------
# Dispatch admission
# ---------------------------------------------------------------------------

def test_window_overlapping_predicted_run_blocks_dispatch(conn):
    # 8-GPU job predicted to run [0, 60); window reserves 2 during [30, 90).
    submit("a", 8, 60, when=iso(0))
    window(30, 90, 2)

    r = scheduler.run_batch(now=at(0))
    assert r["selection_order"] == []
    # Blocked calls earn no quantum (aging still settles, as elsewhere).
    assert all(e["kind"] != "QUANTUM" for e in r["events"])
    assert scheduler.scheduler_state()["used_gpus"] == 0
    assert r["maintenance_reserved_gpus_at_start"] == 0  # window starts later
    # Effective physical free GPUs are still 8 at the reference instant...
    assert r["available_gpus_at_start"] == 8


def test_job_ending_exactly_when_window_starts_dispatches(conn):
    # Run [0, 60) and window [60, 90): endpoints merely touch, no overlap.
    submit("a", 8, 60, when=iso(0))
    window(60, 90, 8)
    r = scheduler.run_batch(now=at(0))
    assert r["selection_order"] == ["J000001"]

    j = r["selected_jobs"][0]
    assert j["estimated_start_at"] == iso(0)
    assert j["estimated_end_at"] == iso(60)
    assert j["max_maintenance_reserved_gpus"] == 0


def test_window_active_at_start_reserves_immediately(conn):
    submit("a", 8, 60, when=iso(0))
    window(0, 30, 3)
    r = scheduler.run_batch(now=at(0))
    assert r["selection_order"] == []
    assert r["maintenance_reserved_gpus_at_start"] == 3
    assert r["effective_free_gpus_at_start"] == 5

    # A job that fits the reduced capacity (5) still dispatches.
    submit("b", 5, 10, when=iso(0))
    r2 = scheduler.run_batch(now=at(0))
    assert r2["selection_order"] == ["J000002"]


def test_short_job_runs_while_long_job_blocked_by_window(conn):
    # Long tenant-a job (6 GPUs, [0,60)) hits the window; short tenant-b
    # job (3 GPUs, [0,20)) ends before it. Visit order: a blocked, b runs.
    submit("a", 6, 60, when=iso(0))
    submit("b", 3, 20, when=iso(0))
    window(30, 90, 4)

    r = scheduler.run_batch(now=at(0))
    assert r["selection_order"] == ["J000002"]
    assert [j["max_maintenance_reserved_gpus"]
            for j in r["selected_jobs"]] == [0]
    assert scheduler.scheduler_state()["used_gpus"] == 3


def test_touching_windows_do_not_add_but_overlapping_windows_do(conn):
    # Two 3-GPU windows meeting at an endpoint: peak 3, not 6; a 5-GPU job
    # over [0, 60) fits (8 - 3 >= 5).
    submit("a", 5, 60, when=iso(0))
    window(0, 30, 3)
    window(30, 60, 3)
    r = scheduler.run_batch(now=at(0))
    assert r["selection_order"] == ["J000001"]
    assert r["selected_jobs"][0]["max_maintenance_reserved_gpus"] == 3

    # Fresh state: genuinely overlapping windows stack to peak 6 at [20,40),
    # so the 5-GPU job is blocked; a 2-GPU job fits the remaining 2.
    db.reset_for_tests()
    db.connect(":memory:")
    submit("a", 5, 60, when=iso(0))
    submit("b", 2, 60, when=iso(0))
    window(0, 40, 3)
    window(20, 60, 3)
    r2 = scheduler.run_batch(now=at(0))
    # a needs 5 <= 8-6 = 2: blocked; b needs 2 <= 2: dispatched.
    assert r2["selection_order"] == ["J000002"]
    assert r2["maintenance_reserved_gpus_at_start"] == 3


def test_same_batch_selected_gpus_count_against_window(conn):
    # Two 4-GPU jobs over [0,60); window reserves 2 over [30,60).
    submit("a", 4, 60, when=iso(0))
    submit("a", 4, 60, when=iso(0))
    window(30, 60, 2)

    r = scheduler.run_batch(now=at(0))
    # First dispatch: 4 <= 6 OK. Second: running 0 + selected 4 + 4 + 2 > 8.
    assert r["selection_order"] == ["J000001"]
    assert scheduler.scheduler_state()["used_gpus"] == 4
    # Repeating the batch changes nothing (J000002 still maintenance-blocked).
    assert scheduler.run_batch(now=at(0))["selection_order"] == []
    queued = [j["id"] for j in scheduler.list_jobs(status="QUEUED")]
    assert queued == ["J000002"]


def test_running_jobs_are_never_preempted_by_new_window(conn):
    # Dispatch an 8-GPU job first, then reserve the whole cluster for
    # maintenance: it keeps running.
    submit("a", 8, 60, when=iso(0))
    r0 = scheduler.run_batch(now=at(0))
    assert r0["selection_order"] == ["J000001"]

    window(10, 20, 8)
    assert scheduler.get_job("J000001")["status"] == "RUNNING"
    state = scheduler.scheduler_state()
    assert state["used_gpus"] == 8 and state["free_gpus"] == 0

    # A new candidate cannot start: its predicted run [5,35) overlaps the
    # full-cluster reservation window [10,20), so peak reserved is 8.
    submit("b", 1, 30, when=iso(5))
    r = scheduler.run_batch(now=at(5))
    assert r["selection_order"] == []
    # The window is not active at the batch instant itself (starts at t=10)...
    assert r["maintenance_reserved_gpus_at_start"] == 0
    # ...but it still blocks via the candidate interval overlap.
    assert scheduler.get_job("J000002")["status"] == "QUEUED"
    # At t=10 the reservation is active and blocks a fresh candidate too.
    assert scheduler.run_batch(now=at(10))["maintenance_reserved_gpus_at_start"] == 8
    # The earlier RUNNING job is still untouched.
    assert scheduler.get_job("J000001")["status"] == "RUNNING"


def test_past_or_zero_reservation_windows_do_not_block(conn):
    submit("a", 8, 60, when=iso(100))
    window(0, 50, 8)      # entirely in the past relative to the batch
    window(100, 160, 0)   # zero reservation
    r = scheduler.run_batch(now=at(100))
    assert r["selection_order"] == ["J000001"]
    assert r["selected_jobs"][0]["max_maintenance_reserved_gpus"] == 0


def test_window_frees_after_it_ends(conn):
    # [0,60) run blocked by window [0,30) reserving 4 at batch now=0...
    submit("a", 8, 60, when=iso(0))
    window(0, 30, 4)
    assert scheduler.run_batch(now=at(0))["selection_order"] == []
    # But a later batch with now=30 predicts [30,90), which no longer
    # overlaps the window -> dispatch.
    r = scheduler.run_batch(now=at(30))
    assert r["selection_order"] == ["J000001"]
    assert r["selected_jobs"][0]["estimated_start_at"] == iso(30)
    assert r["selected_jobs"][0]["estimated_end_at"] == iso(90)


def test_window_respected_together_with_tenant_cap(conn):
    scheduler.upsert_tenant("a", 1.0, max_gpus=3)
    submit("a", 3, 60, when=iso(0))   # cap allows 3
    window(0, 60, 6)                 # cluster allows only 2 during the run
    r = scheduler.run_batch(now=at(0))
    assert r["selection_order"] == []

    # After the window ends a later reference instant admits it (cap still 3).
    r2 = scheduler.run_batch(now=at(60))
    assert r2["selection_order"] == ["J000001"]


# ---------------------------------------------------------------------------
# Preview parity
# ---------------------------------------------------------------------------

def test_preview_shares_maintenance_judgement_and_writes_nothing(conn):
    submit("a", 6, 60, when=iso(0))
    submit("b", 3, 20, when=iso(0))
    window(30, 90, 4)

    before = {
        "windows": scheduler.list_maintenance_windows(),
        "queued": {j["id"] for j in scheduler.list_jobs(status="QUEUED")},
    }
    p = scheduler.preview_batch(now=at(0))
    assert p["preview"] is True
    assert p["selection_order"] == ["J000002"]
    pj = p["selected_jobs"][0]
    assert pj["estimated_start_at"] == iso(0)
    assert pj["estimated_end_at"] == iso(20)
    assert pj["max_maintenance_reserved_gpus"] == 0

    # Preview persists nothing, but the already-committed window survives.
    assert scheduler.list_maintenance_windows() == before["windows"]
    assert {j["id"] for j in scheduler.list_jobs(status="QUEUED")} \
        == before["queued"]
    assert db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 0

    committed = scheduler.run_batch(now=at(0))
    assert committed["selection_order"] == p["selection_order"]
    assert committed["selected_jobs"][0]["estimated_end_at"] == \
        pj["estimated_end_at"]
    assert committed["selected_jobs"][0][
        "max_maintenance_reserved_gpus"] == 0


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def test_maintenance_window_endpoints_http(client):
    r = client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T01:00:00Z",
        "end_at": "2026-01-01T03:00:00Z",
        "reserved_gpus": 4})
    assert r.status_code == 201, r.text
    w = r.json()
    assert w["id"] == 1 and w["reserved_gpus"] == 4
    assert w["start_at"] == "2026-01-01T01:00:00+00:00"
    assert w["end_at"] == "2026-01-01T03:00:00+00:00"

    r2 = client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T05:00:00Z",
        "end_at": "2026-01-01T06:00:00Z",
        "reserved_gpus": 2})
    assert r2.status_code == 201

    listed = client.get("/maintenance-windows").json()
    assert [x["id"] for x in listed] == [1, 2]

    active = client.get("/maintenance-windows",
                        params={"active_at": "2026-01-01T03:00:00Z"}).json()
    # Half-open: window 1 ends at 03:00 sharp and is no longer active.
    assert active == []
    active = client.get("/maintenance-windows",
                        params={"active_at": "2026-01-01T02:59:00Z"}).json()
    assert [x["id"] for x in active] == [1]

    # Naive timestamp is interpreted as UTC.
    r3 = client.post("/maintenance-windows", json={
        "start_at": "2026-01-02T00:00:00",
        "end_at": "2026-01-02T01:00:00",
        "reserved_gpus": 1})
    assert r3.status_code == 201
    assert r3.json()["start_at"] == "2026-01-02T00:00:00+00:00"


def test_maintenance_window_validation_http(client):
    r = client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T03:00:00Z",
        "end_at": "2026-01-01T03:00:00Z",
        "reserved_gpus": 1})
    assert r.status_code == 400
    assert "end_at" in r.json()["detail"]

    r = client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T03:00:00Z",
        "end_at": "2026-01-01T02:00:00Z",
        "reserved_gpus": 1})
    assert r.status_code == 400

    r = client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T00:00:00Z",
        "end_at": "2026-01-01T01:00:00Z",
        "reserved_gpus": 9})
    assert r.status_code == 400
    assert "cluster capacity" in r.json()["detail"]

    # Schema-level problems are 422.
    assert client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T00:00:00Z",
        "end_at": "2026-01-01T01:00:00Z",
        "reserved_gpus": -1}).status_code == 422
    assert client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T00:00:00Z",
        "end_at": "2026-01-01T01:00:00Z",
        "reserved_gpus": True}).status_code == 422
    assert client.post("/maintenance-windows", json={
        "start_at": "nope",
        "end_at": "2026-01-01T01:00:00Z",
        "reserved_gpus": 1}).status_code == 422
    assert client.get("/maintenance-windows").json() == []


def test_maintenance_blocks_dispatch_end_to_end_http(client):
    # Job predicts [00:00, 01:00); window [00:30, 01:30) reserves 2 of 8.
    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 8, "duration_min": 60})
    client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T00:30:00Z",
        "end_at": "2026-01-01T01:30:00Z",
        "reserved_gpus": 2})

    # Preview and real batch take the same maintenance decision.
    p = client.post("/scheduler/preview",
                    json={"now": "2026-01-01T00:00:00Z"}).json()
    assert p["selection_order"] == []
    b = client.post("/scheduler/batch",
                    json={"now": "2026-01-01T00:00:00Z"}).json()
    assert b["selection_order"] == []
    assert client.get("/jobs/J000001").json()["status"] == "QUEUED"

    # Batch at 00:30 predicts [00:30, 01:30): window overlaps fully -> blocked.
    b2 = client.post("/scheduler/batch",
                     json={"now": "2026-01-01T00:30:00Z"}).json()
    assert b2["selection_order"] == []

    # A window that merely touches the predicted run does not block.
    client.post("/maintenance-windows", json={
        "start_at": "2026-01-02T01:00:00Z",
        "end_at": "2026-01-02T02:00:00Z",
        "reserved_gpus": 8})
    b3 = client.post("/scheduler/batch",
                     json={"now": "2026-01-02T00:00:00Z"}).json()
    assert b3["selection_order"] == ["J000001"]
    job = b3["selected_jobs"][0]
    assert job["estimated_start_at"] == "2026-01-02T00:00:00+00:00"
    assert job["estimated_end_at"] == "2026-01-02T01:00:00+00:00"
    assert job["max_maintenance_reserved_gpus"] == 0
