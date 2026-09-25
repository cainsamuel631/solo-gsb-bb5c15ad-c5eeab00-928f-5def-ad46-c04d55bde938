"""End-to-end HTTP tests against the FastAPI app."""

import pytest


def test_full_lifecycle_via_http(client):
    # Submit jobs for two tenants.
    r = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 30, "weight": 1})
    assert r.status_code == 201, r.text
    j1 = r.json()["id"]

    r = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 6, "duration_min": 10, "weight": 2})
    assert r.status_code == 201, r.text
    j2 = r.json()["id"]

    # 2 + 6 = 8 fits; both selected in DRR tenant order.
    r = client.post("/scheduler/batch")
    assert r.status_code == 200
    body = r.json()
    assert body["selection_order"] == [j1, j2]
    assert body["used_gpus_after"] == 8
    assert body["free_gpus_after"] == 0
    tenants = {t["id"]: t for t in body["tenants"]}
    # a: ~quantum 60 - cost 60 (~0 aging, sub-second elapsed)
    assert tenants["team-a"]["balance_gpu_min"] == pytest.approx(0.0, abs=0.5)
    # b: ~quantum 120 - cost 60
    assert tenants["team-b"]["balance_gpu_min"] == pytest.approx(60.0, abs=0.5)
    kinds = [e["kind"] for e in body["events"]]
    assert "QUANTUM" in kinds and "DISPATCH" in kinds

    # Third job cannot fit while cluster full.
    r = client.post("/jobs", json={
        "tenant_id": "team-c", "gpus": 4, "duration_min": 10, "weight": 1})
    j3 = r.json()["id"]
    r = client.post("/scheduler/batch")
    assert r.json()["selection_order"] == []

    # Complete j1 (2 GPUs freed) -> still only 2 free, j3 needs 4 -> blocked.
    r = client.post(f"/jobs/{j1}/complete")
    assert r.status_code == 200
    r = client.post("/scheduler/batch")
    assert r.json()["selection_order"] == []

    # Complete j2 -> 8 free -> j3 runs.
    client.post(f"/jobs/{j2}/complete")
    r = client.post("/scheduler/batch")
    assert r.json()["selection_order"] == [j3]

    # Queries.
    q = client.get("/queue").json()
    assert q["queued_jobs"] == 0
    assert q["used_gpus"] == 4
    jobs = client.get("/jobs?status=RUNNING").json()
    assert [j["id"] for j in jobs] == [j3]
    tenants_list = client.get("/tenants").json()
    assert {t["id"] for t in tenants_list} == {"team-a", "team-b", "team-c"}
    ledger = client.get("/ledger?tenant_id=team-b&limit=10").json()
    assert all(e["tenant_id"] == "team-b" for e in ledger)
    assert client.get("/health").json() == {"status": "ok"}


def test_validation_errors(client):
    r = client.post("/jobs", json={
        "tenant_id": "x", "gpus": 0, "duration_min": 10})
    assert r.status_code == 422
    r = client.post("/jobs", json={
        "tenant_id": "x", "gpus": 100, "duration_min": 10})
    assert r.status_code == 400  # exceeds cluster capacity
    r = client.get("/jobs/nope/complete" if False else "/jobs/nope")
    assert r.status_code == 404


def test_cancel_endpoint(client):
    j = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    r = client.post(f"/jobs/{j}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "CANCELLED"
    assert client.post(f"/jobs/{j}/cancel").status_code == 409


def test_dependency_flow_via_http(client):
    # Cross-tenant prerequisite: team-b waits on team-a's job.
    r = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 1, "duration_min": 10})
    assert r.status_code == 201
    j1 = r.json()["id"]

    r = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 1, "duration_min": 10,
        "depends_on": [j1]})
    assert r.status_code == 201, r.text
    j2 = r.json()["id"]
    assert r.json()["depends_on"] == [j1]
    assert r.json()["blocked_by"] == [{"id": j1, "status": "QUEUED"}]

    # First batch: only the prerequisite dispatches.
    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [j1]
    # team-b receives no quantum while dependency-blocked.
    assert all(e["tenant_id"] != "team-b" for e in body["events"])

    # Queue response marks the blocked head with reason.
    q = client.get("/queue").json()
    b_group = next(g for g in q["queues"] if g["tenant_id"] == "team-b")
    queued_job = b_group["jobs"][0]
    assert queued_job["id"] == j2
    assert queued_job["blocked_by"] == [{"id": j1, "status": "RUNNING"}]

    # Another batch still cannot run j2 while j1 is RUNNING.
    assert client.post("/scheduler/batch").json()["selection_order"] == []

    # Completing the prerequisite releases j2 on the next batch.
    assert client.post(f"/jobs/{j1}/complete").status_code == 200
    assert client.get(f"/jobs/{j2}").json()["blocked_by"] == []
    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [j2]


def test_dependency_validation_errors_http(client):
    # Unknown prerequisite -> 400.
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5,
        "depends_on": ["J000099"]})
    assert r.status_code == 400
    assert "unknown depends_on" in r.json()["error"]

    # Malformed depends_on -> 422.
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5,
        "depends_on": [""]})
    assert r.status_code == 422

    # Sequential ids make a self-dependence addressable: first job is J000001,
    # the next submission will be J000002 depending on itself -> cycle 400.
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5})
    assert r.status_code == 201
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5,
        "depends_on": ["J000002"]})
    assert r.status_code == 400
    assert "dependency cycle" in r.json()["error"]
    # Rolled back: J000002 does not exist.
    assert client.get("/jobs/J000002").status_code == 404


def test_cancelled_prerequisite_blocks_dependent_http(client):
    j1 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 1, "duration_min": 5}).json()["id"]
    j2 = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 1, "duration_min": 5,
        "depends_on": [j1]}).json()["id"]

    # Cancel the prerequisite while it is still queued.
    assert client.post(f"/jobs/{j1}/cancel").status_code == 200

    # The dependent stays QUEUED and its reason is queryable everywhere.
    detail = client.get(f"/jobs/{j2}").json()
    assert detail["status"] == "QUEUED"
    assert detail["blocked_by"] == [{"id": j1, "status": "CANCELLED"}]

    assert client.post("/scheduler/batch").json()["selection_order"] == []

    q = client.get("/queue").json()
    b_group = next(g for g in q["queues"] if g["tenant_id"] == "team-b")
    assert b_group["jobs"][0]["blocked_by"] == [
        {"id": j1, "status": "CANCELLED"}]

    listed = client.get("/jobs?status=QUEUED&tenant_id=team-b").json()
    assert [j["id"] for j in listed] == [j2]
    assert listed[0]["blocked_by"] == [{"id": j1, "status": "CANCELLED"}]


def test_gpu_usage_window_via_http(client):
    from app import db

    # Dispatch two jobs (2 GPU team-a, 4 GPU team-b), then pin their real
    # RUNNING intervals so assertions are exact.
    j1 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 30}).json()["id"]
    j2 = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 4, "duration_min": 60}).json()["id"]
    client.post("/scheduler/batch")
    conn = db.get_conn()
    conn.execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?",
        ("2026-01-01T00:10:00+00:00", j1))
    conn.execute(
        "UPDATE jobs SET status = 'COMPLETED', completed_at = ? WHERE id = ?",
        ("2026-01-01T00:40:00+00:00", j1))
    conn.execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?",
        ("2026-01-01T00:30:00+00:00", j2))

    # j1 fully inside [00:00,01:00): 2 x 30 = 60; j2 still RUNNING, truncated
    # at 01:00: 4 x 30 = 120.
    r = client.get(
        "/usage/gpu",
        params={"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T01:00:00Z"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["start"] == "2026-01-01T00:00:00+00:00"
    assert body["end"] == "2026-01-01T01:00:00+00:00"
    assert body["tenant_id"] is None
    tenants = {t["tenant_id"]: t for t in body["tenants"]}
    assert tenants["team-a"] == {
        "tenant_id": "team-a", "gpu_minutes": 60.0, "overlapping_jobs": 1}
    assert tenants["team-b"]["gpu_minutes"] == 120.0
    assert tenants["team-b"]["overlapping_jobs"] == 1
    assert body["global"] == {"gpu_minutes": 180.0, "overlapping_jobs": 2}

    # Right-open boundary: window ending exactly when j2 starts excludes it.
    r = client.get("/usage/gpu", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T00:30:00Z"})
    body = r.json()
    tenants = {t["tenant_id"]: t for t in body["tenants"]}
    assert tenants["team-a"]["gpu_minutes"] == 40.0  # 2 x 20
    assert tenants["team-b"]["gpu_minutes"] == 0.0
    assert tenants["team-b"]["overlapping_jobs"] == 0
    assert body["global"]["overlapping_jobs"] == 1


def test_gpu_usage_tenant_filter_via_http(client):
    from app import db

    j = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 30}).json()["id"]
    client.post("/scheduler/batch")
    conn = db.get_conn()
    conn.execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?",
        ("2026-01-01T00:10:00+00:00", j))

    r = client.get("/usage/gpu", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T01:00:00Z",
        "tenant_id": "team-a"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [t["tenant_id"] for t in body["tenants"]] == ["team-a"]
    assert body["global"]["gpu_minutes"] == 100.0  # 2 x 50 (still RUNNING)

    # Unknown tenant -> 404.
    r = client.get("/usage/gpu", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T01:00:00Z",
        "tenant_id": "ghost"})
    assert r.status_code == 404


def test_gpu_usage_invalid_window_via_http(client):
    # end <= start -> 400.
    r = client.get("/usage/gpu", params={
        "start": "2026-01-01T01:00:00Z",
        "end": "2026-01-01T01:00:00Z"})
    assert r.status_code == 400
    assert "end must be later" in r.json()["detail"]

    r = client.get("/usage/gpu", params={
        "start": "2026-01-01T02:00:00Z",
        "end": "2026-01-01T01:00:00Z"})
    assert r.status_code == 400

    # Missing / malformed params -> 422.
    assert client.get("/usage/gpu", params={
        "start": "2026-01-01T00:00:00Z"}).status_code == 422
    assert client.get("/usage/gpu", params={
        "start": "not-a-time", "end": "2026-01-01T01:00:00Z"
        }).status_code == 422


def test_gpu_usage_empty_database_zeros(client):
    r = client.get("/usage/gpu", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T01:00:00Z"})
    assert r.status_code == 200
    body = r.json()
    assert body["tenants"] == []
    assert body["global"] == {"gpu_minutes": 0.0, "overlapping_jobs": 0}


def test_tenant_gpu_cap_flow_via_http(client):
    # Create tenant with a 2-GPU concurrent cap.
    r = client.post("/tenants", json={
        "tenant_id": "team-a", "weight": 1, "max_gpus": 2})
    assert r.status_code == 201, r.text
    assert r.json()["max_gpus"] == 2
    assert r.json()["running_gpus"] == 0

    j1 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 5}).json()["id"]
    j2 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 1, "duration_min": 5}).json()["id"]

    # J1 fills the cap; J2 waits and earns no quantum.
    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [j1]
    tenants = {t["id"]: t for t in body["tenants"]}
    assert tenants["team-a"]["running_gpus"] == 2
    assert tenants["team-a"]["max_gpus"] == 2

    # J1 fills the cap; J2 waits and earns no quantum. (Aging may still land,
    # exactly as it does for a capacity-blocked head.)
    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == []
    assert all(e["kind"] != "QUANTUM" for e in body["events"])

    # Re-running the batch while cap-blocked still hands out no quantum.
    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == []
    assert all(e["kind"] != "QUANTUM" for e in body["events"])

    # Completion releases cap headroom; J2 runs.
    assert client.post(f"/jobs/{j1}/complete").status_code == 200
    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [j2]

    # Clear the cap explicitly with null; GET reflects unlimited.
    r = client.post("/tenants", json={
        "tenant_id": "team-a", "weight": 1, "max_gpus": None})
    assert r.status_code == 201
    assert r.json()["max_gpus"] is None

    r = client.get("/tenants/team-a")
    assert r.json()["max_gpus"] is None


def test_tenant_cap_omitted_preserves_existing_cap(client):
    assert client.post("/tenants", json={
        "tenant_id": "team-a", "weight": 1, "max_gpus": 3}).status_code == 201
    # Update weight without sending max_gpus: cap is preserved.
    r = client.post("/tenants", json={
        "tenant_id": "team-a", "weight": 2})
    assert r.status_code == 201
    body = r.json()
    assert body["weight"] == 2.0 and body["max_gpus"] == 3


def test_tenant_cap_does_not_starve_other_tenants_http(client):
    client.post("/tenants", json={
        "tenant_id": "team-a", "weight": 1, "max_gpus": 2})
    ja1 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 5}).json()["id"]
    client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 5})
    jb = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 4, "duration_min": 5}).json()["id"]

    # a's cap stops at 2 GPUs, but b still gets its 4 (2 + 4 <= 8).
    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [ja1, jb]
    used = {t["id"]: t["running_gpus"] for t in body["tenants"]}
    assert used["team-a"] == 2 and used["team-b"] == 4


def test_tenant_cap_validation_http(client):
    # max_gpus must be a positive integer.
    r = client.post("/tenants", json={
        "tenant_id": "x", "weight": 1, "max_gpus": 0})
    assert r.status_code == 422
    r = client.post("/tenants", json={
        "tenant_id": "x", "weight": 1, "max_gpus": -2})
    assert r.status_code == 422
    # Booleans must not be coerced into an integer cap.
    r = client.post("/tenants", json={
        "tenant_id": "x", "weight": 1, "max_gpus": True})
    assert r.status_code == 422
    # New tenant without the field is unlimited (default behaviour).
    r = client.post("/tenants", json={"tenant_id": "y", "weight": 1})
    assert r.status_code == 201 and r.json()["max_gpus"] is None
