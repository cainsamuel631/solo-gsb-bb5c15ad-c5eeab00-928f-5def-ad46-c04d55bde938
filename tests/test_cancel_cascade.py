"""Dependency cancel cascade (GET /jobs/{id}/cancel-impact,
POST /jobs/{id}/cancel-cascade).

Covers: transitive closure over the cross-tenant dependency graph,
traversal through CANCELLED nodes but not through COMPLETED ones, one
shortest dependency chain per affected job, id-sorted output, the
preview's zero-write guarantee, the cascade's atomic cancel of the target
plus still-QUEUED descendants (GPU release, aging-clock reset, cancelled
intermediates left untouched), 404/409 handling, and the HTTP surface.
"""
from __future__ import annotations

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


def build_chain():
    """J1 <- J2 <- J3, J4 depends on J1 and J3 (cross-tenant)."""
    j1 = submit("team-a", when=iso(0))                        # J000001
    j2 = submit("team-b", when=iso(1), depends_on=[j1["id"]])  # J000002
    j3 = submit("team-c", when=iso(2), depends_on=[j2["id"]])  # J000003
    j4 = submit("team-b", when=iso(3),
                depends_on=[j1["id"], j3["id"]])               # J000004
    return j1, j2, j3, j4


# ---------------------------------------------------------------------------
# cancel_impact (service level)
# ---------------------------------------------------------------------------

def test_impact_linear_chain_and_shortest_chains(conn):
    j1, j2, j3, j4 = build_chain()

    impact = scheduler.cancel_impact(j1["id"])
    assert impact["job_id"] == j1["id"]
    assert impact["status"] == "QUEUED"
    assert impact["affected_count"] == 3
    # Sorted by job id; each carries ONE shortest chain from the target.
    assert [j["id"] for j in impact["affected_jobs"]] == [
        j2["id"], j3["id"], j4["id"]]
    chains = {j["id"]: j["chain"] for j in impact["affected_jobs"]}
    assert chains[j2["id"]] == [j1["id"], j2["id"]]
    assert chains[j3["id"]] == [j1["id"], j2["id"], j3["id"]]
    # J4 depends on J1 directly: the direct edge beats the path via J3.
    assert chains[j4["id"]] == [j1["id"], j4["id"]]
    # Affected entries are full job objects.
    j2_view = next(j for j in impact["affected_jobs"] if j["id"] == j2["id"])
    assert j2_view["tenant_id"] == "team-b"
    assert j2_view["status"] == "QUEUED"
    assert j2_view["depends_on"] == [j1["id"]]


def test_impact_diamond_deterministic_chain(conn):
    t = submit("a", when=iso(0))                          # J000001
    a = submit("b", depends_on=[t["id"]])                 # J000002
    b = submit("c", depends_on=[t["id"]])                 # J000003
    c = submit("d", depends_on=[a["id"], b["id"]])        # J000004

    impact = scheduler.cancel_impact(t["id"])
    chains = {j["id"]: j["chain"] for j in impact["affected_jobs"]}
    # Both [T,A,C] and [T,B,C] are shortest; id-sorted BFS picks via A.
    assert chains[c["id"]] == [t["id"], a["id"], c["id"]]
    assert impact["affected_count"] == 3


def test_impact_no_dependents(conn):
    j = submit("a")
    impact = scheduler.cancel_impact(j["id"])
    assert impact["affected_count"] == 0
    assert impact["affected_jobs"] == []


def test_impact_running_target(conn):
    j1 = submit("a", gpus=8, when=iso(0))
    j2 = submit("b", depends_on=[j1["id"]])
    scheduler.run_batch(now=at(1))
    assert scheduler.get_job(j1["id"])["status"] == "RUNNING"

    impact = scheduler.cancel_impact(j1["id"])
    assert impact["status"] == "RUNNING"
    assert [j["id"] for j in impact["affected_jobs"]] == [j2["id"]]
    assert impact["affected_jobs"][0]["chain"] == [j1["id"], j2["id"]]


def test_impact_traverses_cancelled_nodes(conn):
    j1 = submit("a", when=iso(0))
    j2 = submit("b", depends_on=[j1["id"]])
    j3 = submit("c", depends_on=[j2["id"]])
    scheduler.cancel_job(j2["id"])  # intermediate node already CANCELLED

    impact = scheduler.cancel_impact(j1["id"])
    # The cancelled J2 is traversal-only: not itself reported, but the
    # blockage propagates through it to the still-QUEUED J3.
    assert [j["id"] for j in impact["affected_jobs"]] == [j3["id"]]
    assert impact["affected_jobs"][0]["chain"] == [
        j1["id"], j2["id"], j3["id"]]


def test_impact_does_not_traverse_completed_nodes(conn):
    t = submit("a", when=iso(0))
    done = submit("b", when=iso(1))
    # Force a completed node onto a would-be path: done depends on the
    # target (edge inserted directly), and y depends only on done.
    conn.execute(
        "INSERT INTO job_dependencies (job_id, depends_on) VALUES (?, ?)",
        (done["id"], t["id"]),
    )
    conn.execute(
        "UPDATE jobs SET status = 'COMPLETED' WHERE id = ?", (done["id"],)
    )
    y = submit("c", depends_on=[done["id"]])

    impact = scheduler.cancel_impact(t["id"])
    # The COMPLETED node is a barrier: y's prerequisite is satisfied, so
    # cancelling the target cannot block y through it.
    assert impact["affected_jobs"] == []
    assert scheduler.get_job(y["id"])["blocked_by"] == []


def test_impact_unknown_and_finished_target(conn):
    with pytest.raises(scheduler.NotFound) as exc:
        scheduler.cancel_impact("J000999")
    assert exc.value.status_code == 404

    done = submit("a", gpus=8, when=iso(0))
    scheduler.run_batch(now=at(1))
    scheduler.complete_job(done["id"])
    with pytest.raises(scheduler.Conflict) as exc:
        scheduler.cancel_impact(done["id"])
    assert exc.value.status_code == 409

    cancelled = submit("a")
    scheduler.cancel_job(cancelled["id"])
    with pytest.raises(scheduler.Conflict):
        scheduler.cancel_impact(cancelled["id"])


def test_impact_preview_writes_nothing(conn):
    j1, j2, j3, j4 = build_chain()
    scheduler.run_batch(now=at(1))  # settle some aging/balance state

    def snapshot():
        jobs = conn.execute(
            "SELECT id, status, completed_at FROM jobs ORDER BY id"
        ).fetchall()
        tenants = conn.execute(
            "SELECT id, deficit, aging_settled_at, updated_at "
            "FROM tenants ORDER BY id"
        ).fetchall()
        batches = conn.execute("SELECT COUNT(*) AS n FROM batches").fetchone()
        ledger = conn.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()
        return (
            [tuple(j) for j in jobs],
            [tuple(t) for t in tenants],
            batches["n"],
            ledger["n"],
        )

    before = snapshot()
    impact = scheduler.cancel_impact(j1["id"])
    assert impact["affected_count"] == 3
    assert snapshot() == before


# ---------------------------------------------------------------------------
# cancel_cascade (service level)
# ---------------------------------------------------------------------------

def test_cascade_cancels_target_and_queued_descendants(conn):
    j1, j2, j3, j4 = build_chain()

    result = scheduler.cancel_cascade(j1["id"])
    assert result["job_id"] == j1["id"]
    assert result["cancelled_count"] == 4
    assert result["cancelled"] == [j1["id"], j2["id"], j3["id"], j4["id"]]
    assert [j["id"] for j in result["jobs"]] == result["cancelled"]
    assert all(j["status"] == "CANCELLED" for j in result["jobs"])

    # Existing job/queue interfaces reflect the cancellations.
    for jid in result["cancelled"]:
        assert scheduler.get_job(jid)["status"] == "CANCELLED"
    assert scheduler.list_jobs(status="QUEUED") == []
    assert scheduler.queue_overview()["queued_jobs"] == 0
    # A later batch has nothing to dispatch.
    assert scheduler.run_batch(now=at(2))["selection_order"] == []


def test_cascade_releases_running_target_gpus(conn):
    j1 = submit("a", gpus=8, when=iso(0))
    j2 = submit("b", depends_on=[j1["id"]])
    j3 = submit("c", gpus=8, when=iso(1))  # independent, capacity-blocked
    scheduler.run_batch(now=at(2))
    assert scheduler.get_job(j1["id"])["status"] == "RUNNING"
    assert scheduler.scheduler_state()["used_gpus"] == 8

    result = scheduler.cancel_cascade(j1["id"])
    assert result["cancelled"] == [j1["id"], j2["id"]]
    # The RUNNING target's occupancy is released for subsequent batches.
    assert scheduler.scheduler_state()["used_gpus"] == 0
    batch = scheduler.run_batch(now=at(3))
    assert batch["selection_order"] == [j3["id"]]


def test_cascade_leaves_cancelled_intermediates_untouched(conn):
    j1 = submit("a", when=iso(0))
    j2 = submit("b", depends_on=[j1["id"]])
    j3 = submit("c", depends_on=[j2["id"]])
    scheduler.cancel_job(j2["id"])
    j2_before = scheduler.get_job(j2["id"])

    result = scheduler.cancel_cascade(j1["id"])
    # J2 was already CANCELLED: traversal-only, not re-modified.
    assert result["cancelled"] == [j1["id"], j3["id"]]
    j2_after = scheduler.get_job(j2["id"])
    assert j2_after["status"] == "CANCELLED"
    assert j2_after["completed_at"] == j2_before["completed_at"]


def test_cascade_resets_aging_clock_of_drained_tenants(conn):
    j1 = submit("a", when=iso(0))
    j2 = submit("b", when=iso(1), depends_on=[j1["id"]])
    scheduler.run_batch(now=at(2))  # advances b's aging checkpoint
    assert scheduler.get_tenant("b")["aging_settled_at"] is not None

    scheduler.cancel_cascade(j1["id"])
    # b's queue drained: same aging-clock reset as a single cancellation.
    assert scheduler.get_tenant("b")["aging_settled_at"] is None


def test_cascade_keeps_unrelated_and_completed_jobs(conn):
    j1 = submit("a", when=iso(0))
    j2 = submit("b", depends_on=[j1["id"]])
    done = submit("c", gpus=2, when=iso(1))
    scheduler.run_batch(now=at(2))
    scheduler.complete_job(done["id"])
    free = submit("d", when=iso(3))  # independent queued job

    scheduler.cancel_cascade(j1["id"])
    assert scheduler.get_job(done["id"])["status"] == "COMPLETED"
    assert scheduler.get_job(free["id"])["status"] == "QUEUED"
    # The independent job still dispatches normally afterwards.
    assert scheduler.run_batch(now=at(4))["selection_order"] == [free["id"]]


def test_cascade_no_dependents(conn):
    j = submit("a")
    result = scheduler.cancel_cascade(j["id"])
    assert result["cancelled_count"] == 1
    assert result["cancelled"] == [j["id"]]
    assert scheduler.get_job(j["id"])["status"] == "CANCELLED"


def test_cascade_unknown_and_finished_target(conn):
    with pytest.raises(scheduler.NotFound) as exc:
        scheduler.cancel_cascade("J000999")
    assert exc.value.status_code == 404

    done = submit("a", gpus=8, when=iso(0))
    scheduler.run_batch(now=at(1))
    scheduler.complete_job(done["id"])
    with pytest.raises(scheduler.Conflict) as exc:
        scheduler.cancel_cascade(done["id"])
    assert exc.value.status_code == 409

    cancelled = submit("a")
    scheduler.cancel_job(cancelled["id"])
    with pytest.raises(scheduler.Conflict):
        scheduler.cancel_cascade(cancelled["id"])


def test_cascade_dependents_of_cancelled_stay_blocked(conn):
    j1 = submit("a", when=iso(0))
    j2 = submit("b", depends_on=[j1["id"]])
    scheduler.cancel_cascade(j1["id"])

    # A new submission depending on a cascade-cancelled job is doomed too:
    # it stays QUEUED and reports the cancelled prerequisite.
    j3 = submit("c", depends_on=[j2["id"]])
    assert scheduler.get_job(j3["id"])["blocked_by"] == [
        {"id": j2["id"], "status": "CANCELLED"}]
    assert scheduler.run_batch(now=at(1))["selection_order"] == []


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def test_http_impact_happy_path(client):
    j1 = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    j2 = client.post("/jobs", json={
        "tenant_id": "b", "gpus": 1, "duration_min": 5,
        "depends_on": [j1]}).json()["id"]
    j3 = client.post("/jobs", json={
        "tenant_id": "c", "gpus": 1, "duration_min": 5,
        "depends_on": [j2]}).json()["id"]

    r = client.get(f"/jobs/{j1}/cancel-impact")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == j1
    assert body["status"] == "QUEUED"
    assert body["affected_count"] == 2
    assert [j["id"] for j in body["affected_jobs"]] == [j2, j3]
    assert body["affected_jobs"][0]["chain"] == [j1, j2]
    assert body["affected_jobs"][1]["chain"] == [j1, j2, j3]

    # The preview wrote nothing.
    assert client.get(f"/jobs/{j1}").json()["status"] == "QUEUED"
    assert client.get("/queue").json()["queued_jobs"] == 3


def test_http_impact_unknown_404_and_finished_409(client):
    r = client.get("/jobs/J000999/cancel-impact")
    assert r.status_code == 404
    assert "not found" in r.json()["error"]

    j = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    client.post(f"/jobs/{j}/cancel")
    r = client.get(f"/jobs/{j}/cancel-impact")
    assert r.status_code == 409
    assert "already cancelled" in r.json()["error"]


def test_http_cascade_happy_path(client):
    j1 = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 8, "duration_min": 5}).json()["id"]
    j2 = client.post("/jobs", json={
        "tenant_id": "b", "gpus": 1, "duration_min": 5,
        "depends_on": [j1]}).json()["id"]
    j3 = client.post("/jobs", json={
        "tenant_id": "c", "gpus": 1, "duration_min": 5,
        "depends_on": [j2]}).json()["id"]
    client.post("/scheduler/batch")  # J1 -> RUNNING (holds all 8 GPUs)

    r = client.post(f"/jobs/{j1}/cancel-cascade")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == j1
    assert body["cancelled_count"] == 3
    assert body["cancelled"] == [j1, j2, j3]
    assert [j["status"] for j in body["jobs"]] == ["CANCELLED"] * 3

    # Job and queue interfaces reflect the result; GPUs were released.
    assert client.get(f"/jobs/{j1}").json()["status"] == "CANCELLED"
    assert client.get(f"/jobs/{j3}").json()["status"] == "CANCELLED"
    queue = client.get("/queue").json()
    assert queue["queued_jobs"] == 0
    assert queue["used_gpus"] == 0
    cancelled = client.get("/jobs?status=CANCELLED").json()
    assert [j["id"] for j in cancelled] == [j1, j2, j3]


def test_http_cascade_unknown_404_and_finished_409(client):
    r = client.post("/jobs/J000999/cancel-cascade")
    assert r.status_code == 404
    assert "not found" in r.json()["error"]

    j = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    client.post("/scheduler/batch")
    client.post(f"/jobs/{j}/complete")
    r = client.post(f"/jobs/{j}/cancel-cascade")
    assert r.status_code == 409
    assert "already completed" in r.json()["error"]

    # A cascade-cancelled target cannot be cascaded again.
    k = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    assert client.post(f"/jobs/{k}/cancel-cascade").status_code == 200
    assert client.post(f"/jobs/{k}/cancel-cascade").status_code == 409
