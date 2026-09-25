"""Rolling 24-hour GPU-minute budget tests (service level + HTTP).

The budget caps the ``gpus * duration_min`` cost of jobs *dispatched* inside
the half-open-on-the-left window ``(now - 24h, now]`` ending at each batch's
reference instant. These tests cover:

* set / update / clear / omit-preserve through ``POST /tenants``;
* blocking dispatch that would overspend, without QUANTUM and without
  affecting other tenants; multiple same-batch selections are charged once;
* the rolling window: dispatches older than 24h fall out of the window;
* no refund on completion or cancellation;
* tenant queries evaluate the window at the server's current UTC instant;
* previews run the same path but persist nothing.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401

WINDOW = timedelta(hours=24)


def submit(tenant, gpus, minutes, weight=1.0, when=None, depends_on=None):
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
    return scheduler._get_job_or_404(db.get_conn(), job["id"])


def batch(now_minutes):
    return scheduler.run_batch(now=at(now_minutes))


def pin_dispatch(job_id, at_minutes):
    """Force a dispatched job's scheduled_at (its budget charge instant)."""
    db.get_conn().execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?",
        (iso(at_minutes), job_id),
    )


# ---------------------------------------------------------------------------
# Tenant configuration
# ---------------------------------------------------------------------------

def test_budget_defaults_to_unlimited(conn):
    submit("a", 8, 10, when=iso(0))
    t = scheduler.get_tenant("a")
    assert t["budget_gpu_min"] is None
    assert t["budget_used_gpu_min"] == 0
    assert t["budget_remaining_gpu_min"] is None
    assert batch(1)["selection_order"] == ["J000001"]


def test_budget_set_update_clear_and_omit_preserves(conn):
    t = scheduler.upsert_tenant("a", 1.0, budget_gpu_min=120)
    assert t["budget_gpu_min"] == 120 and t["budget_remaining_gpu_min"] == 120

    # Omit preserves the budget while updating the weight.
    t2 = scheduler.upsert_tenant("a", 2.0)
    assert t2["weight"] == 2.0 and t2["budget_gpu_min"] == 120

    # An explicit positive integer replaces it.
    t3 = scheduler.upsert_tenant("a", 2.0, budget_gpu_min=300)
    assert t3["budget_gpu_min"] == 300

    # Explicit None clears it.
    t4 = scheduler.upsert_tenant("a", 2.0, budget_gpu_min=None)
    assert t4["budget_gpu_min"] is None
    assert t4["budget_remaining_gpu_min"] is None

    # New tenants with/without the argument are unlimited.
    assert scheduler.upsert_tenant("b", 1.0)["budget_gpu_min"] is None
    assert scheduler.upsert_tenant("c", 1.0, budget_gpu_min=None)[
        "budget_gpu_min"] is None


def test_invalid_budget_rejected(conn):
    for bad in (0, -1, 2.5, True):
        with pytest.raises(scheduler.SchedulerError):
            scheduler.upsert_tenant("a", 1.0, budget_gpu_min=bad)


def test_budget_and_cap_are_independent(conn):
    t = scheduler.upsert_tenant(
        "a", 1.0, max_gpus=4, budget_gpu_min=120)
    assert t["max_gpus"] == 4 and t["budget_gpu_min"] == 120
    # Omitting budget leaves it; passing cap leaves budget untouched.
    t2 = scheduler.upsert_tenant("a", 1.0, max_gpus=2)
    assert t2["max_gpus"] == 2 and t2["budget_gpu_min"] == 120
    t3 = scheduler.upsert_tenant("a", 1.0, budget_gpu_min=None)
    assert t3["max_gpus"] == 2 and t3["budget_gpu_min"] is None


# ---------------------------------------------------------------------------
# Dispatch enforcement
# ---------------------------------------------------------------------------

def test_budget_blocks_overspend_and_passes_other_tenants(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=100)
    # 2 GPU x 60 = 120 > 100: a's head can never fit the budget, even though
    # the cluster (8 GPUs) has room and quantum accrues instantly (cost 120
    # is reached after 2 rounds).
    ja = submit("a", 2, 60, when=iso(0))
    # Another unlimited tenant dispatches normally in the same batch.
    jb = submit("b", 1, 30, when=iso(0))

    r = batch(10)
    assert r["selection_order"] == [jb["id"]]
    # a earned no quantum (budget-blocked head), though aging still lands.
    assert all(
        e["tenant_id"] != "a" or e["kind"] != "QUANTUM" for e in r["events"])
    assert scheduler.get_job(ja["id"])["status"] == "QUEUED"
    tenants = {t["id"]: t for t in r["tenants"]}
    # a's blocked job was not charged.
    assert tenants["a"]["budget_used_gpu_min"] == 0
    assert tenants["a"]["budget_remaining_gpu_min"] == 100
    # b's 30 charge is reflected at the batch reference instant.
    assert tenants["b"]["budget_gpu_min"] is None
    assert tenants["b"]["budget_used_gpu_min"] == 30

    # Repeating the batch still grants a no quantum.
    r2 = batch(10)
    assert r2["selection_order"] == []
    assert all(e["kind"] != "QUANTUM" for e in r2["events"])


def test_partial_budget_dispatches_until_exhausted(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=100)
    j1 = submit("a", 1, 60, when=iso(0))   # cost 60
    j2 = submit("a", 1, 60, when=iso(1))   # cost 60: 60 + 60 > 100

    r = batch(10)
    assert r["selection_order"] == [j1["id"]]
    t = next(x for x in r["tenants"] if x["id"] == "a")
    assert t["budget_used_gpu_min"] == 60
    assert t["budget_remaining_gpu_min"] == 40

    # A later batch cannot fit J2 (60 > 40 remaining): blocked, no quantum.
    r2 = batch(11)
    assert r2["selection_order"] == []
    assert all(
        e["tenant_id"] != "a" or e["kind"] != "QUANTUM" for e in r2["events"])

    # Raising the budget releases J2 on the next batch.
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=200)
    r3 = batch(12)
    assert r3["selection_order"] == [j2["id"]]
    t = next(x for x in r3["tenants"] if x["id"] == "a")
    assert t["budget_used_gpu_min"] == 120
    assert t["budget_remaining_gpu_min"] == 80


def test_budget_charges_each_same_batch_selection_once(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=90)
    j1 = submit("a", 1, 30, when=iso(0))   # 30
    j2 = submit("a", 1, 30, when=iso(1))   # 30 -> 60
    j3 = submit("a", 1, 30, when=iso(2))   # 30 -> 90 exactly: admitted
    j4 = submit("a", 1, 30, when=iso(3))   # 120 > 90: waits

    r = batch(10)
    assert r["selection_order"] == [j1["id"], j2["id"], j3["id"]]
    t = next(x for x in r["tenants"] if x["id"] == "a")
    assert t["budget_used_gpu_min"] == 90
    assert t["budget_remaining_gpu_min"] == 0
    assert scheduler.get_job(j4["id"])["status"] == "QUEUED"


def tenant_at(tenant_id, instant):
    """Tenant budget snapshot evaluated at an explicit reference instant."""
    c = db.get_conn()
    row = c.execute(
        scheduler._TENANT_SELECT + " HAVING t.id = ?", (tenant_id,)
    ).fetchone()
    return scheduler._tenant_dict(c, row, window_end=instant)


def test_spend_accumulates_across_batches_in_window(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=100)
    j1 = submit("a", 1, 60, when=iso(0))
    batch(10)
    assert tenant_at("a", at(10))["budget_used_gpu_min"] == 60

    j2 = submit("a", 1, 60, when=iso(20))
    # 60 already spent inside the window; 60 more overspends.
    r = scheduler.run_batch(now=at(30))
    assert r["selection_order"] == []
    assert scheduler.get_job(j2["id"])["status"] == "QUEUED"

    # Remove the over-budget head; a 40-cost job exactly fills the remainder.
    scheduler.cancel_job(j2["id"])
    j3 = submit("a", 1, 40, when=iso(30))
    r2 = scheduler.run_batch(now=at(40))
    assert r2["selection_order"] == [j3["id"]]
    t = next(x for x in r2["tenants"] if x["id"] == "a")
    assert t["budget_used_gpu_min"] == 100
    assert t["budget_remaining_gpu_min"] == 0


def test_fifo_head_blocks_smaller_job_behind_it(conn):
    # Strict FIFO is unchanged: a budget-blocked head is never skipped to
    # reach an affordable job behind it.
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=100)
    j1 = submit("a", 1, 60, when=iso(0))
    batch(10)
    big = submit("a", 1, 60, when=iso(20))   # 60+60 > 100
    small = submit("a", 1, 10, when=iso(21))  # would fit, but behind big
    r = scheduler.run_batch(now=at(30))
    assert r["selection_order"] == []
    assert scheduler.get_job(big["id"])["status"] == "QUEUED"
    assert scheduler.get_job(small["id"])["status"] == "QUEUED"


# ---------------------------------------------------------------------------
# Rolling window mechanics
# ---------------------------------------------------------------------------

def test_rolling_window_drops_charges_older_than_24h(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=100)
    old = submit("a", 1, 60, when=iso(0))
    batch(10)
    pin_dispatch(old["id"], 10)          # charged at t=10
    assert tenant_at("a", at(10 + 24 * 60 - 1))[
        "budget_used_gpu_min"] == 60

    # A new 60-cost job at t = 10 + 24h: the old charge is exactly 24h in
    # the past, and the window is (now-24h, now], so it has dropped out.
    new = submit("a", 1, 60, when=iso(10 + 24 * 60))
    r = scheduler.run_batch(now=at(10 + 24 * 60))
    assert r["selection_order"] == [new["id"]]
    t = next(x for x in r["tenants"] if x["id"] == "a")
    assert t["budget_used_gpu_min"] == 60

    # Boundary re-check one minute before the 24h edge: the old charge
    # (t=10) is still inside (now-24h, now] = (t=9, now], so a further 60
    # overspends and waits.
    scheduler.cancel_job(new["id"])
    edge = submit("a", 1, 60, when=iso(10 + 24 * 60 - 1))
    r_edge = scheduler.run_batch(now=at(10 + 24 * 60 - 1))
    assert r_edge["selection_order"] == []


def test_window_endpoint_exclusivity(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=1000)
    j = submit("a", 1, 60, when=iso(1440))
    batch(1440)
    pin_dispatch(j["id"], 1440)             # charged exactly at t=24h
    # At now = 2880 the charge is exactly 24h old: left-open excludes it.
    snap = tenant_at("a", at(2880))
    assert snap["budget_used_gpu_min"] == 0
    # One minute before, it still counts.
    assert tenant_at("a", at(2879))["budget_used_gpu_min"] == 60


# ---------------------------------------------------------------------------
# No refunds
# ---------------------------------------------------------------------------

def test_completion_and_cancellation_do_not_refund(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=100)
    j1 = submit("a", 1, 60, when=iso(0))
    batch(10)
    pin_dispatch(j1["id"], 10)
    assert tenant_at("a", at(10))["budget_used_gpu_min"] == 60

    # Completing does not refund the charge.
    scheduler.complete_job(j1["id"])
    t = tenant_at("a", at(15))
    assert t["budget_used_gpu_min"] == 60 and t["budget_remaining_gpu_min"] == 40

    # A 60-cost job still cannot dispatch (only 40 remains).
    j2 = submit("a", 1, 60, when=iso(20))
    assert scheduler.run_batch(now=at(20))["selection_order"] == []
    scheduler.cancel_job(j2["id"])  # never dispatched -> never charged

    # A 40-cost job dispatches; cancelling it while RUNNING keeps the charge.
    j3 = submit("a", 1, 40, when=iso(30))
    r = scheduler.run_batch(now=at(30))
    assert r["selection_order"] == [j3["id"]]
    scheduler.cancel_job(j3["id"])
    t = tenant_at("a", at(30))
    assert t["budget_used_gpu_min"] == 100
    assert t["budget_remaining_gpu_min"] == 0


def test_queued_cancellation_spends_nothing(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=10)
    j = submit("a", 1, 60, when=iso(0))   # 60 > 10, can never dispatch
    assert batch(5)["selection_order"] == []
    scheduler.cancel_job(j["id"])
    t = scheduler.get_tenant("a")
    assert t["budget_used_gpu_min"] == 0
    assert t["budget_remaining_gpu_min"] == 10


# ---------------------------------------------------------------------------
# Queries use the server's current UTC instant
# ---------------------------------------------------------------------------

def test_tenant_query_evaluates_window_at_server_now(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=1000)
    j = submit("a", 1, 60, when=iso(0))
    batch(10)
    # Pin the dispatch to ~now so it is inside the live rolling window.
    now = datetime.now(timezone.utc)
    db.get_conn().execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?",
        (now.isoformat(), j["id"]),
    )
    t = scheduler.get_tenant("a")
    assert t["budget_used_gpu_min"] == 60
    assert t["budget_remaining_gpu_min"] == 940
    # Reported window ends at (essentially) the server's current UTC time.
    end = datetime.fromisoformat(t["budget_window_end_at"])
    assert abs((end - now).total_seconds()) < 5
    assert end - datetime.fromisoformat(
        t["budget_window_start_at"]) == WINDOW

    # Pinned 25h in the past: outside the live window, used drops to zero.
    old = (now - timedelta(hours=25)).isoformat()
    db.get_conn().execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?", (old, j["id"]))
    assert scheduler.get_tenant("a")["budget_used_gpu_min"] == 0


# ---------------------------------------------------------------------------
# Preview: same decision, no persistence
# ---------------------------------------------------------------------------

def test_preview_enforces_budget_but_writes_nothing(conn):
    scheduler.upsert_tenant("a", 1.0, budget_gpu_min=100)
    j1 = submit("a", 1, 60, when=iso(0))
    j2 = submit("a", 1, 60, when=iso(1))

    preview = scheduler.preview_batch(now=at(10))
    assert preview["preview"] is True
    assert preview["selection_order"] == [j1["id"]]
    t = next(x for x in preview["tenants"] if x["id"] == "a")
    assert t["budget_used_gpu_min"] == 60
    # Nothing persisted: job still QUEUED, no charges, repeatable.
    assert scheduler.get_job(j1["id"])["status"] == "QUEUED"
    assert scheduler.get_tenant("a")["budget_used_gpu_min"] == 0
    preview2 = scheduler.preview_batch(now=at(10))
    assert preview2["selection_order"] == preview["selection_order"]

    committed = scheduler.run_batch(now=at(10))
    assert committed["selection_order"] == [j1["id"]]
    # Same prospective state at the same instant.
    pt = {x["id"]: x for x in preview["tenants"]}["a"]
    ct = {x["id"]: x for x in committed["tenants"]}["a"]
    assert (pt["budget_used_gpu_min"], pt["budget_remaining_gpu_min"]) == \
           (ct["budget_used_gpu_min"], ct["budget_remaining_gpu_min"])
    # After the commit J2 is budget-blocked.
    assert scheduler.run_batch(now=at(11))["selection_order"] == []


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def test_budget_http_set_omit_clear(client):
    r = client.post("/tenants", json={
        "tenant_id": "a", "weight": 1, "budget_gpu_min": 1440})
    assert r.status_code == 201, r.text
    assert r.json()["budget_gpu_min"] == 1440
    assert r.json()["budget_used_gpu_min"] == 0
    assert r.json()["budget_remaining_gpu_min"] == 1440

    # Omitted -> preserved.
    r = client.post("/tenants", json={"tenant_id": "a", "weight": 2})
    assert r.json()["weight"] == 2.0 and r.json()["budget_gpu_min"] == 1440

    # null -> cleared.
    r = client.post("/tenants", json={
        "tenant_id": "a", "weight": 2, "budget_gpu_min": None})
    assert r.json()["budget_gpu_min"] is None
    assert r.json()["budget_remaining_gpu_min"] is None

    # Bad values -> 422.
    for bad in (0, -5, 1.5, True):
        assert client.post("/tenants", json={
            "tenant_id": "x", "weight": 1, "budget_gpu_min": bad
        }).status_code == 422


def test_budget_http_blocks_in_shared_dispatch_path(client):
    client.post("/tenants", json={
        "tenant_id": "team-a", "weight": 1, "budget_gpu_min": 100})
    ja = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 60}).json()["id"]
    jb = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 1, "duration_min": 30}).json()["id"]

    # Preview and batch take the same decision: a blocked, b dispatched.
    p = client.post(
        "/scheduler/preview", json={"now": "2026-01-01T00:10:00Z"}).json()
    assert p["selection_order"] == [jb]
    assert client.get(f"/jobs/{ja}").json()["status"] == "QUEUED"

    b = client.post(
        "/scheduler/batch", json={"now": "2026-01-01T00:10:00Z"}).json()
    assert b["selection_order"] == [jb]
    tenants = {t["id"]: t for t in b["tenants"]}
    assert tenants["team-a"]["budget_used_gpu_min"] == 0
    assert tenants["team-a"]["budget_remaining_gpu_min"] == 100
    assert tenants["team-b"]["budget_used_gpu_min"] == 30

    # Budget-blocked a gets no QUANTUM in either call.
    assert all(
        not (e["tenant_id"] == "team-a" and e["kind"] == "QUANTUM")
        for e in b["events"])

    # Completing b does not refund anyone; a is still budget-blocked.
    client.post(f"/jobs/{jb}/complete")
    b2 = client.post(
        "/scheduler/batch", json={"now": "2026-01-01T00:20:00Z"}).json()
    assert b2["selection_order"] == []

    # Clear the budget; a dispatches.
    client.post("/tenants", json={
        "tenant_id": "team-a", "weight": 1, "budget_gpu_min": None})
    b3 = client.post(
        "/scheduler/batch", json={"now": "2026-01-01T00:30:00Z"}).json()
    assert b3["selection_order"] == [ja]


def test_budget_http_rolling_window_with_pinned_charges(client):
    from app import db as db_mod

    client.post("/tenants", json={
        "tenant_id": "a", "weight": 1, "budget_gpu_min": 100})
    # Charge 60 at t=0; it ages out at t=24h (left-open window).
    j = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 60}).json()["id"]
    client.post("/scheduler/batch", json={"now": "2026-01-01T00:00:00Z"})
    conn = db_mod.get_conn()
    conn.execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00+00:00", j))

    # At 23:59 the charge is still in the window: a second 60-cost job waits.
    j2 = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 60}).json()["id"]
    body = client.post(
        "/scheduler/batch", json={"now": "2026-01-01T23:59:00Z"}).json()
    assert body["selection_order"] == []

    # At exactly 24h the old charge dropped out: j2 dispatches.
    body = client.post(
        "/scheduler/batch", json={"now": "2026-01-02T00:00:00Z"}).json()
    assert body["selection_order"] == [j2]
    tenants = {t["id"]: t for t in body["tenants"]}
    assert tenants["a"]["budget_window_end_at"] == \
        "2026-01-02T00:00:00+00:00"
    assert tenants["a"]["budget_window_start_at"] == \
        "2026-01-01T00:00:00+00:00"
    assert tenants["a"]["budget_used_gpu_min"] == 60
