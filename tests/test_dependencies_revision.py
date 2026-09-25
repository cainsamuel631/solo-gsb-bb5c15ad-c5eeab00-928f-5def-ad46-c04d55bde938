"""Dependency revision for QUEUED jobs (PATCH /jobs/{id}/dependencies).

Covers: wholesale replacement semantics (dedupe + sort, empty list
detaches), cross-tenant prerequisites, QUEUED-only enforcement, unknown id
/ self-dependence / cycle rejections that leave the stored edges untouched,
replace (not extend) graph semantics, the next preview/batch judging the
revised edges, idempotency-fingerprint preservation across revisions, and
the HTTP surface (status codes, error bodies, request validation).
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


# ---------------------------------------------------------------------------
# Service level
# ---------------------------------------------------------------------------

def test_revise_replaces_dedupes_and_sorts(conn):
    p1 = submit("base", when=iso(0))                      # J000001
    p2 = submit("base", when=iso(1))                      # J000002
    job = submit("t", when=iso(2), depends_on=[p1["id"]])  # J000003
    queued_at = job["queued_at"]

    # Duplicates and out-of-order input normalise to the sorted unique list.
    rev = scheduler.revise_dependencies(
        job["id"], [p2["id"], p1["id"], p2["id"]]
    )
    assert rev["depends_on"] == [p1["id"], p2["id"]]
    assert rev["blocked_by"] == [
        {"id": p1["id"], "status": "QUEUED"},
        {"id": p2["id"], "status": "QUEUED"},
    ]
    # The revision is persisted and the queue position is untouched.
    stored = scheduler.get_job(job["id"])
    assert stored["depends_on"] == [p1["id"], p2["id"]]
    assert stored["queued_at"] == queued_at


def test_revise_empty_list_detaches(conn):
    p = submit("base")
    job = submit("t", depends_on=[p["id"]])
    assert job["blocked_by"] == [{"id": p["id"], "status": "QUEUED"}]

    rev = scheduler.revise_dependencies(job["id"], [])
    assert rev["depends_on"] == []
    assert rev["blocked_by"] == []
    assert scheduler.get_job(job["id"])["depends_on"] == []


def test_revise_cross_tenant_prerequisites(conn):
    p1 = submit("team-a")
    p2 = submit("team-b")
    job = submit("team-c")

    rev = scheduler.revise_dependencies(job["id"], [p2["id"], p1["id"]])
    assert rev["depends_on"] == [p1["id"], p2["id"]]


def test_revise_unknown_job_raises_not_found(conn):
    with pytest.raises(scheduler.NotFound) as exc:
        scheduler.revise_dependencies("J000999", [])
    assert exc.value.status_code == 404


def test_revise_non_queued_rejected(conn):
    running = submit("t", when=iso(0))
    scheduler.run_batch(now=at(1))
    assert scheduler.get_job(running["id"])["status"] == "RUNNING"
    with pytest.raises(scheduler.Conflict) as exc:
        scheduler.revise_dependencies(running["id"], [])
    assert exc.value.status_code == 409
    assert "only QUEUED" in str(exc.value)

    scheduler.complete_job(running["id"])
    with pytest.raises(scheduler.Conflict):
        scheduler.revise_dependencies(running["id"], [])

    cancelled = submit("t")
    scheduler.cancel_job(cancelled["id"])
    with pytest.raises(scheduler.Conflict):
        scheduler.revise_dependencies(cancelled["id"], [])


def test_revise_unknown_prerequisite_rejected_keeps_old_edges(conn):
    p = submit("base")
    job = submit("t", depends_on=[p["id"]])
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.revise_dependencies(job["id"], ["J000999"])
    assert "unknown depends_on" in str(exc.value)
    # The rejected revision changed nothing.
    assert scheduler.get_job(job["id"])["depends_on"] == [p["id"]]


def test_revise_self_dependency_rejected_keeps_old_edges(conn):
    p = submit("base")
    job = submit("t", depends_on=[p["id"]])
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.revise_dependencies(job["id"], [job["id"]])
    assert "dependency cycle" in str(exc.value)
    assert scheduler.get_job(job["id"])["depends_on"] == [p["id"]]


def test_revise_cycle_rejected_keeps_old_edges(conn):
    p = submit("base")                       # J000001
    job = submit("t", depends_on=[p["id"]])  # J000002 -> J000001
    # Making the prerequisite depend on its own dependent closes a cycle.
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.revise_dependencies(p["id"], [job["id"]])
    assert "dependency cycle" in str(exc.value)
    assert scheduler.get_job(p["id"])["depends_on"] == []
    assert scheduler.get_job(job["id"])["depends_on"] == [p["id"]]


def test_revise_substitutes_not_extends_edges(conn):
    p = submit("base")                       # J000001
    job = submit("t", depends_on=[p["id"]])  # edge J000002 -> J000001

    # While J000002 -> J000001 exists, J000001 -> J000002 would close a cycle.
    with pytest.raises(scheduler.SchedulerError):
        scheduler.revise_dependencies(p["id"], [job["id"]])

    # Replacing J000002's edges with the empty set really removes the edge...
    assert scheduler.revise_dependencies(job["id"], [])["depends_on"] == []
    # ...so the previously cyclic edge is now a plain acyclic one.
    rev = scheduler.revise_dependencies(p["id"], [job["id"]])
    assert rev["depends_on"] == [job["id"]]


def test_revise_then_preview_and_batch_use_new_edges(conn):
    p = submit("a", when=iso(0))                        # J000001
    job = submit("b", when=iso(1), depends_on=[p["id"]])  # J000002

    # Blocked: the preview only dispatches the prerequisite.
    assert scheduler.preview_batch(now=at(2))["selection_order"] == [p["id"]]

    # Detached: the very next preview (and batch) judge the new edges.
    scheduler.revise_dependencies(job["id"], [])
    preview = scheduler.preview_batch(now=at(2))
    assert preview["selection_order"] == [p["id"], job["id"]]

    # Re-attaching blocks it again, effective immediately.
    scheduler.revise_dependencies(job["id"], [p["id"]])
    assert scheduler.preview_batch(now=at(2))["selection_order"] == [p["id"]]

    # The committed batch follows the latest revision.
    scheduler.revise_dependencies(job["id"], [])
    batch = scheduler.run_batch(now=at(2))
    assert batch["selection_order"] == [p["id"], job["id"]]
    assert scheduler.get_job(job["id"])["status"] == "RUNNING"


def test_revise_does_not_touch_tenant_or_queue_state(conn):
    p = submit("base", when=iso(0))
    job = submit("t", when=iso(1), depends_on=[p["id"]])
    tenant_before = scheduler.get_tenant("t")
    other = submit("t", when=iso(2))

    scheduler.revise_dependencies(job["id"], [])

    tenant_after = scheduler.get_tenant("t")
    assert tenant_after["weight"] == tenant_before["weight"]
    assert tenant_after["balance_gpu_min"] == tenant_before["balance_gpu_min"]
    assert tenant_after["aging_settled_at"] == tenant_before["aging_settled_at"]
    # FIFO positions (queued_at) of the revised job and its neighbours stand.
    assert scheduler.get_job(job["id"])["queued_at"] == job["queued_at"]
    assert scheduler.get_job(other["id"])["queued_at"] == other["queued_at"]


def test_revise_preserves_idempotency_fingerprint(conn):
    p1 = submit("base")  # J000001
    p2 = submit("base")  # J000002
    job, created = scheduler.submit_job_idempotent(
        "t", 1, 10, 2.0, [p1["id"]], idempotency_key="k1"
    )
    assert created is True
    assert job["depends_on"] == [p1["id"]]

    # Revise the dependencies out of band.
    scheduler.revise_dependencies(job["id"], [p2["id"]])

    # Replaying the ORIGINAL submission returns the revised job as-is: the
    # stored fingerprint still matches the original parameters and the
    # revision is not overwritten.
    replay, created2 = scheduler.submit_job_idempotent(
        "t", 1, 10, 2.0, [p1["id"]], idempotency_key="k1"
    )
    assert created2 is False
    assert replay["id"] == job["id"]
    assert replay["depends_on"] == [p2["id"]]

    # The fingerprint was NOT rewritten to the revised dependency set.
    with pytest.raises(scheduler.Conflict):
        scheduler.submit_job_idempotent(
            "t", 1, 10, 2.0, [p2["id"]], idempotency_key="k1"
        )
    assert scheduler.get_tenant("t")["weight"] == 2.0


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def test_http_patch_dependencies_happy_path(client):
    p1 = client.post("/jobs", json={
        "tenant_id": "base", "gpus": 1, "duration_min": 5}).json()["id"]
    p2 = client.post("/jobs", json={
        "tenant_id": "base", "gpus": 1, "duration_min": 5}).json()["id"]
    job = client.post("/jobs", json={
        "tenant_id": "t", "gpus": 1, "duration_min": 5}).json()["id"]

    r = client.patch(f"/jobs/{job}/dependencies",
                     json={"depends_on": [p2, p1, p2]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == job
    assert body["status"] == "QUEUED"
    assert body["depends_on"] == [p1, p2]
    assert body["blocked_by"] == [
        {"id": p1, "status": "QUEUED"}, {"id": p2, "status": "QUEUED"}]

    # Persisted: GET reflects the revision.
    assert client.get(f"/jobs/{job}").json()["depends_on"] == [p1, p2]


def test_http_patch_empty_array_detaches(client):
    p = client.post("/jobs", json={
        "tenant_id": "base", "gpus": 1, "duration_min": 5}).json()["id"]
    job = client.post("/jobs", json={
        "tenant_id": "t", "gpus": 1, "duration_min": 5,
        "depends_on": [p]}).json()["id"]

    r = client.patch(f"/jobs/{job}/dependencies", json={"depends_on": []})
    assert r.status_code == 200, r.text
    assert r.json()["depends_on"] == []
    assert r.json()["blocked_by"] == []


def test_http_patch_unknown_job_404(client):
    r = client.patch("/jobs/J000999/dependencies", json={"depends_on": []})
    assert r.status_code == 404
    assert "not found" in r.json()["error"]


def test_http_patch_non_queued_409(client):
    j = client.post("/jobs", json={
        "tenant_id": "t", "gpus": 1, "duration_min": 5}).json()["id"]
    client.post("/scheduler/batch")  # QUEUED -> RUNNING
    r = client.patch(f"/jobs/{j}/dependencies", json={"depends_on": []})
    assert r.status_code == 409
    assert "only QUEUED" in r.json()["error"]


def test_http_patch_unknown_prerequisite_400_keeps_old(client):
    p = client.post("/jobs", json={
        "tenant_id": "base", "gpus": 1, "duration_min": 5}).json()["id"]
    job = client.post("/jobs", json={
        "tenant_id": "t", "gpus": 1, "duration_min": 5,
        "depends_on": [p]}).json()["id"]

    r = client.patch(f"/jobs/{job}/dependencies",
                     json={"depends_on": ["J000999"]})
    assert r.status_code == 400
    assert "unknown depends_on" in r.json()["error"]
    assert client.get(f"/jobs/{job}").json()["depends_on"] == [p]


def test_http_patch_self_and_cycle_400_keep_old(client):
    p = client.post("/jobs", json={
        "tenant_id": "base", "gpus": 1, "duration_min": 5}).json()["id"]
    job = client.post("/jobs", json={
        "tenant_id": "t", "gpus": 1, "duration_min": 5,
        "depends_on": [p]}).json()["id"]

    # Self-dependence.
    r = client.patch(f"/jobs/{job}/dependencies",
                     json={"depends_on": [job]})
    assert r.status_code == 400
    assert "dependency cycle" in r.json()["error"]

    # Cycle through the prerequisite.
    r = client.patch(f"/jobs/{p}/dependencies", json={"depends_on": [job]})
    assert r.status_code == 400
    assert "dependency cycle" in r.json()["error"]

    assert client.get(f"/jobs/{job}").json()["depends_on"] == [p]
    assert client.get(f"/jobs/{p}").json()["depends_on"] == []


def test_http_patch_body_validation_422(client):
    j = client.post("/jobs", json={
        "tenant_id": "t", "gpus": 1, "duration_min": 5}).json()["id"]
    url = f"/jobs/{j}/dependencies"
    assert client.patch(url, json={}).status_code == 422
    assert client.patch(url, json={"depends_on": None}).status_code == 422
    assert client.patch(url, json={"depends_on": "J000001"}).status_code == 422
    assert client.patch(url, json={"depends_on": [""]}).status_code == 422
    assert client.patch(url, json={"depends_on": ["  "]}).status_code == 422
    assert client.patch(url, json={"depends_on": [123]}).status_code == 422
    # The failed validations changed nothing.
    assert client.get(f"/jobs/{j}").json()["depends_on"] == []


def test_http_patch_then_preview_and_batch(client):
    p = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    job = client.post("/jobs", json={
        "tenant_id": "b", "gpus": 1, "duration_min": 5,
        "depends_on": [p]}).json()["id"]

    body = client.post("/scheduler/preview").json()
    assert body["selection_order"] == [p]

    r = client.patch(f"/jobs/{job}/dependencies", json={"depends_on": []})
    assert r.status_code == 200

    body = client.post("/scheduler/preview").json()
    assert body["selection_order"] == [p, job]
    # Preview wrote nothing: the job is still queued with its new edges.
    assert client.get(f"/jobs/{job}").json()["status"] == "QUEUED"

    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [p, job]


def test_http_idempotent_replay_returns_revised_job(client):
    p1 = client.post("/jobs", json={
        "tenant_id": "base", "gpus": 1, "duration_min": 5}).json()["id"]
    p2 = client.post("/jobs", json={
        "tenant_id": "base", "gpus": 1, "duration_min": 5}).json()["id"]

    headers = {"Idempotency-Key": "rev-key"}
    original = {"tenant_id": "t", "gpus": 1, "duration_min": 5,
                "depends_on": [p1]}
    r = client.post("/jobs", json=original, headers=headers)
    assert r.status_code == 201
    job = r.json()["id"]

    # Revise the dependencies, then replay the original submission: the
    # replay returns the revised job and does not restore the old edges.
    assert client.patch(
        f"/jobs/{job}/dependencies", json={"depends_on": [p2]}
    ).status_code == 200
    r = client.post("/jobs", json=original, headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == job
    assert r.json()["depends_on"] == [p2]

    # The fingerprint still holds the ORIGINAL parameters.
    r = client.post("/jobs", json={**original, "depends_on": [p2]},
                    headers=headers)
    assert r.status_code == 409
