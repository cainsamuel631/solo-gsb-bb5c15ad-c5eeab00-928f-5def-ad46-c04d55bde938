"""Maintenance-slot queries: feasible estimated starts for QUEUED jobs.

``GET /jobs/{id}/maintenance-slots`` (service: ``job_maintenance_slots``)
takes the current RUNNING GPU usage as a fixed baseline and overlays the
persisted maintenance windows on the job's whole predicted run
``[t, t + duration_min)``: a start instant is feasible iff baseline + job
GPUs + combined reservation never exceeds cluster capacity. These tests
cover the sweep/merge logic (overlapping windows, endpoint-touching
windows, windows crossing the query end, isolated feasible instants), the
fixed-baseline semantics and the HTTP surface.
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


def slots(job_id, start_min, end_min):
    return scheduler.job_maintenance_slots(job_id, at(start_min), at(end_min))


def slot(start_min, end_min, start_incl=True, end_incl=False):
    return {
        "start_at": iso(start_min),
        "end_at": iso(end_min),
        "start_inclusive": start_incl,
        "end_inclusive": end_incl,
    }


# ---------------------------------------------------------------------------
# Basic shape / validation
# ---------------------------------------------------------------------------

def test_no_windows_whole_range_is_one_slot(conn):
    j = submit("a", 2, 60, when=iso(0))
    r = slots(j["id"], 0, 300)
    assert r["job_id"] == j["id"]
    assert r["status"] == "QUEUED"
    assert r["gpus"] == 2 and r["duration_min"] == 60
    assert r["cluster_gpus"] == 8
    assert r["running_baseline_gpus"] == 0
    assert r["query_start_at"] == iso(0)
    assert r["query_end_at"] == iso(300)
    assert r["slots"] == [slot(0, 300)]
    assert r["earliest_start_at"] == iso(0)


def test_window_splits_range_and_earliest_is_reported(conn):
    # Job 2 GPU x 60 min; window [100,200) reserves 7 of 8 -> headroom 6.
    # Blocked starts: (100-60, 200) = (40, 200).
    j = submit("a", 2, 60, when=iso(0))
    window(100, 200, 7)
    r = slots(j["id"], 0, 300)
    assert r["slots"] == [
        slot(0, 40, end_incl=True),      # run [40,100) merely touches window
        slot(200, 300),                  # final slot ends exclusive
    ]
    assert r["earliest_start_at"] == iso(0)


def test_validation_and_job_state(conn):
    j = submit("a", 2, 60, when=iso(0))
    with pytest.raises(scheduler.SchedulerError, match="later than start"):
        slots(j["id"], 100, 100)
    with pytest.raises(scheduler.SchedulerError, match="later than start"):
        slots(j["id"], 100, 50)
    with pytest.raises(scheduler.NotFound):
        slots("J000999", 0, 100)

    scheduler.run_batch(now=at(0))  # dispatch -> RUNNING
    with pytest.raises(scheduler.Conflict, match="QUEUED"):
        slots(j["id"], 0, 100)
    scheduler.complete_job(j["id"])
    with pytest.raises(scheduler.Conflict, match="QUEUED"):
        slots(j["id"], 0, 100)

    j2 = submit("b", 1, 10, when=iso(0))
    scheduler.cancel_job(j2["id"])
    with pytest.raises(scheduler.Conflict, match="QUEUED"):
        slots(j2["id"], 0, 100)


def test_naive_datetimes_treated_as_utc(conn):
    from datetime import datetime

    j = submit("a", 2, 60, when=iso(0))
    r = scheduler.job_maintenance_slots(
        j["id"], datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 1, 2, 0)
    )
    assert r["query_start_at"] == "2026-01-01T00:00:00+00:00"
    assert r["query_end_at"] == "2026-01-01T02:00:00+00:00"
    assert r["slots"] == [slot(0, 120)]


# ---------------------------------------------------------------------------
# Merge / boundary semantics
# ---------------------------------------------------------------------------

def test_overlapping_windows_merge_and_sum(conn):
    # Reservations stack only where windows genuinely overlap: [120,150)
    # peaks at 8 > headroom 6; [100,120) and [150,180) stay at 4.
    j = submit("a", 2, 60, when=iso(0))
    window(100, 150, 4)
    window(120, 180, 4)
    r = slots(j["id"], 0, 300)
    # Blocked starts: (120-60, 150) = (60, 150).
    assert r["slots"] == [
        slot(0, 60, end_incl=True),
        slot(150, 300),
    ]
    assert r["earliest_start_at"] == iso(0)


def test_touching_windows_do_not_add(conn):
    # Two 5-GPU windows meeting at t=150: peak stays 5 <= headroom 6, so
    # nothing is blocked anywhere in the range.
    j = submit("a", 2, 60, when=iso(0))
    window(100, 150, 5)
    window(150, 200, 5)
    r = slots(j["id"], 0, 300)
    assert r["slots"] == [slot(0, 300)]
    assert r["earliest_start_at"] == iso(0)


def test_isolated_feasible_instant_between_blocked_regions(conn):
    # Two 7-GPU windows exactly one duration (60 min) apart: the single
    # instant t=160 has run [160,220) touching both windows only at their
    # endpoints -> feasible, but isolated.
    j = submit("a", 2, 60, when=iso(0))
    window(100, 160, 7)
    window(220, 280, 7)
    r = slots(j["id"], 0, 400)
    assert r["slots"] == [
        slot(0, 40, end_incl=True),
        slot(160, 160, end_incl=True),   # degenerate: isolated instant
        slot(280, 400),
    ]
    assert r["earliest_start_at"] == iso(0)


def test_isolated_feasible_instant_at_query_start(conn):
    # Window [60,120) reserves 7: every start in (0, 120) is blocked, but
    # t=0 has run [0,60) merely touching the window -> feasible, isolated.
    j = submit("a", 2, 60, when=iso(0))
    window(60, 120, 7)
    r = slots(j["id"], 0, 120)
    assert r["slots"] == [slot(0, 0, end_incl=True)]
    assert r["earliest_start_at"] == iso(0)


def test_window_crossing_query_end_still_blocks(conn):
    # Window [250,500) reaches past the query end 300; starts near the end
    # still have their whole run interval checked against it.
    j = submit("a", 2, 60, when=iso(0))
    window(250, 500, 7)
    r = slots(j["id"], 0, 300)
    # Blocked starts: (250-60, 500) = (190, 500) -> clipped to the range.
    assert r["slots"] == [slot(0, 190, end_incl=True)]
    assert r["earliest_start_at"] == iso(0)


def test_blocked_region_covering_query_start(conn):
    # Window [0,120) reserves 7: starts ( -60, 120 ) are blocked, so the
    # query start itself is infeasible and the first slot begins at 120.
    j = submit("a", 2, 60, when=iso(0))
    window(0, 120, 7)
    r = slots(j["id"], 0, 300)
    assert r["slots"] == [slot(120, 300)]
    assert r["earliest_start_at"] == iso(120)


def test_no_slots_returns_empty_list_and_null_earliest(conn):
    j = submit("a", 2, 60, when=iso(0))
    window(0, 1000, 7)  # 2 + 7 > 8 for every start in the range
    r = slots(j["id"], 0, 300)
    assert r["slots"] == []
    assert r["earliest_start_at"] is None


# ---------------------------------------------------------------------------
# Fixed RUNNING baseline
# ---------------------------------------------------------------------------

def test_running_baseline_reduces_headroom(conn):
    # 5 GPUs RUNNING -> headroom for a 2-GPU job is 8-5-2 = 1; a 2-GPU
    # reservation over [100,200) blocks starts in (40, 200).
    submit("x", 5, 600, when=iso(0))
    assert scheduler.run_batch(now=at(0))["selection_order"] == ["J000001"]
    j = submit("a", 2, 60, when=iso(0))
    window(100, 200, 2)

    r = slots(j["id"], 0, 300)
    assert r["running_baseline_gpus"] == 5
    assert r["slots"] == [
        slot(0, 40, end_incl=True),
        slot(200, 300),
    ]


def test_baseline_alone_can_eliminate_all_slots(conn):
    # 7 GPUs RUNNING + a 2-GPU job exceed the cluster even with no windows.
    submit("x", 7, 600, when=iso(0))
    scheduler.run_batch(now=at(0))
    j = submit("a", 2, 60, when=iso(0))
    r = slots(j["id"], 0, 300)
    assert r["running_baseline_gpus"] == 7
    assert r["slots"] == []
    assert r["earliest_start_at"] is None


def test_baseline_is_fixed_no_future_release_speculated(conn):
    # The RUNNING job's own duration ends at t=60, but the baseline is held
    # fixed: starts after 60 still count the 6 RUNNING GPUs.
    submit("x", 6, 60, when=iso(0))
    scheduler.run_batch(now=at(0))
    j = submit("a", 2, 60, when=iso(0))  # 6 + 2 = 8 exactly fits
    r = slots(j["id"], 0, 300)
    assert r["slots"] == [slot(0, 300)]

    j2 = submit("b", 3, 60, when=iso(0))  # 6 + 3 > 8 everywhere
    r2 = slots(j2["id"], 0, 300)
    assert r2["slots"] == []
    assert r2["earliest_start_at"] is None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def test_maintenance_slots_http(client):
    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 2, "duration_min": 60})
    client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T01:40:00Z",
        "end_at": "2026-01-01T03:20:00Z",
        "reserved_gpus": 7})

    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T05:00:00Z"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "J000001"
    assert body["status"] == "QUEUED"
    assert body["running_baseline_gpus"] == 0
    assert body["query_start_at"] == "2026-01-01T00:00:00+00:00"
    assert body["query_end_at"] == "2026-01-01T05:00:00+00:00"
    # Blocked starts: (01:40 - 60min, 03:20) = (00:40, 03:20).
    assert body["slots"] == [
        {"start_at": "2026-01-01T00:00:00+00:00",
         "end_at": "2026-01-01T00:40:00+00:00",
         "start_inclusive": True, "end_inclusive": True},
        {"start_at": "2026-01-01T03:20:00+00:00",
         "end_at": "2026-01-01T05:00:00+00:00",
         "start_inclusive": True, "end_inclusive": False},
    ]
    assert body["earliest_start_at"] == "2026-01-01T00:00:00+00:00"

    # Naive timestamps are interpreted as UTC.
    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-01-01T00:00:00", "end": "2026-01-01T01:00:00"})
    assert r.status_code == 200
    assert r.json()["query_start_at"] == "2026-01-01T00:00:00+00:00"


def test_maintenance_slots_http_errors(client):
    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 2, "duration_min": 60})

    # Unknown job -> 404.
    assert client.get("/jobs/J000999/maintenance-slots", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T01:00:00Z"}).status_code == 404

    # end not later than start -> 400.
    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-01-01T01:00:00Z",
        "end": "2026-01-01T01:00:00Z"})
    assert r.status_code == 400
    assert "later than start" in r.json()["detail"]

    # Missing/invalid query parameters -> 422.
    assert client.get(
        "/jobs/J000001/maintenance-slots").status_code == 422
    assert client.get("/jobs/J000001/maintenance-slots", params={
        "start": "nope", "end": "2026-01-01T01:00:00Z"}).status_code == 422

    # Non-QUEUED job -> 409.
    client.post("/scheduler/batch", json={"now": "2026-01-01T00:00:00Z"})
    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T01:00:00Z"})
    assert r.status_code == 409
    assert "QUEUED" in r.json()["detail"]
