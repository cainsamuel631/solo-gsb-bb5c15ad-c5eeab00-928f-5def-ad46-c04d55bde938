"""Batch job-graph submission (POST /jobs/graph).

Covers the service layer and HTTP surface: sequential id allocation in array
order with ref→id mapping, forward refs, external + cross-tenant
prerequisites, dependency deduplication, immediate readiness judgements by
preview/batch, tenant auto-register/weight update (existing caps and budgets
untouched), and the whole-batch rejections — duplicate ref, per-tenant
weight disagreement, over-capacity GPUs, unknown external ids / batch refs,
self-dependence and cycles — each rolling every tenant/queue change back.
"""
from __future__ import annotations

import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def graph_item(ref, tenant, gpus=1, minutes=10, weight=1.0,
               depends_on=None, depends_on_refs=None):
    return {
        "ref": ref,
        "tenant_id": tenant,
        "gpus": gpus,
        "duration_min": minutes,
        "weight": weight,
        "depends_on": depends_on or [],
        "depends_on_refs": depends_on_refs or [],
    }


# ---------------------------------------------------------------------------
# Service level: happy paths
# ---------------------------------------------------------------------------

def test_graph_creates_jobs_with_sequential_ids_and_ref_map(conn):
    pre = scheduler.submit_job("base", 1, 5)  # J000001
    result = scheduler.submit_job_graph([
        graph_item("a", "team-a"),               # J000002
        graph_item("b", "team-b"),               # J000003
    ])
    assert result["ref_ids"] == {"a": "J000002", "b": "J000003"}
    assert [j["id"] for j in result["jobs"]] == ["J000002", "J000003"]
    assert all(j["status"] == "QUEUED" for j in result["jobs"])
    # Ids continue after pre-existing jobs, and the prereq points at one.
    assert pre["id"] == "J000001"


def test_graph_forward_ref_resolves_and_blocks_readiness(conn):
    result = scheduler.submit_job_graph([
        # Appears first but depends on a later item.
        graph_item("train", "team-a", depends_on_refs=["prep"]),
        graph_item("prep", "team-a"),
    ])
    ids = result["ref_ids"]
    assert list(ids) == ["train", "prep"]
    train = next(j for j in result["jobs"] if j["id"] == ids["train"])
    prep = next(j for j in result["jobs"] if j["id"] == ids["prep"])
    assert train["depends_on"] == [ids["prep"]]
    assert train["blocked_by"] == [{"id": ids["prep"], "status": "QUEUED"}]
    assert prep["depends_on"] == [] and prep["blocked_by"] == []

    # The very next preview already applies the dependency rules: team-a's
    # FIFO head (train, lower id) is blocked, so nothing of team-a dispatches
    # until prep completes -- even though prep itself is ready.
    assert scheduler.preview_batch(now=at(1))["selection_order"] == []


def test_graph_external_and_cross_tenant_prerequisites(conn):
    ext = scheduler.submit_job("team-x", 1, 5)  # J000001
    result = scheduler.submit_job_graph([
        graph_item("a", "team-a", depends_on=[ext["id"]]),
        graph_item("b", "team-b", depends_on_refs=["a"]),
    ])
    ids = result["ref_ids"]
    a = next(j for j in result["jobs"] if j["id"] == ids["a"])
    b = next(j for j in result["jobs"] if j["id"] == ids["b"])
    assert a["depends_on"] == [ext["id"]]
    assert a["blocked_by"] == [{"id": ext["id"], "status": "QUEUED"}]
    # Cross-tenant in-batch edge.
    assert b["depends_on"] == [ids["a"]]


def test_graph_dependencies_deduplicated_across_both_lists(conn):
    ext = scheduler.submit_job("base", 1, 5)
    result = scheduler.submit_job_graph([
        graph_item("a", "team-a"),
        graph_item("b", "team-a",
                   depends_on=[ext["id"], ext["id"]],
                   depends_on_refs=["a", "a"]),
    ])
    b = result["jobs"][1]
    assert b["depends_on"] == [ext["id"], result["ref_ids"]["a"]]


def test_graph_batch_dispatches_per_existing_dependency_rules(conn):
    ext = scheduler.submit_job("base", 1, 5)
    result = scheduler.submit_job_graph([
        graph_item("a", "team-a", depends_on=[ext["id"]]),
        graph_item("b", "team-b"),
    ])
    ids = result["ref_ids"]
    # First batch: the external prerequisite and the independent team-b job
    # run (tenant order base < team-b); team-a's job stays blocked.
    batch = scheduler.run_batch(now=at(1))
    assert batch["selection_order"] == [ext["id"], ids["b"]]
    # Completing the prerequisite releases a on the next batch.
    scheduler.complete_job(ext["id"])
    batch = scheduler.run_batch(now=at(2))
    assert batch["selection_order"] == [ids["a"]]


def test_graph_auto_registers_tenant_at_agreed_weight(conn):
    result = scheduler.submit_job_graph([
        graph_item("a", "team-a", weight=2.5),
        graph_item("b", "team-a", weight=2.5),
        graph_item("c", "team-b", weight=1.0),
    ])
    tenant = scheduler.get_tenant("team-a")
    assert tenant["weight"] == 2.5
    assert tenant["queued_jobs"] == 2
    assert scheduler.get_tenant("team-b")["weight"] == 1.0
    assert len(result["jobs"]) == 3


def test_graph_updates_existing_tenant_weight_but_keeps_cap_and_budget(conn):
    scheduler.upsert_tenant(
        "team-a", 1.0, max_gpus=3, budget_gpu_min=120
    )
    scheduler.submit_job_graph([
        graph_item("a", "team-a", weight=2.0),
        graph_item("b", "team-a", weight=2.0),
    ])
    tenant = scheduler.get_tenant("team-a")
    assert tenant["weight"] == 2.0
    assert tenant["max_gpus"] == 3
    assert tenant["budget_gpu_min"] == 120


def test_graph_fifo_order_inside_batch_is_array_order(conn):
    result = scheduler.submit_job_graph([
        graph_item("first", "t", gpus=1),
        graph_item("second", "t", gpus=1),
    ])
    queued = scheduler.list_jobs(tenant_id="t")
    assert [j["id"] for j in queued] == [
        result["ref_ids"]["first"], result["ref_ids"]["second"]
    ]


# ---------------------------------------------------------------------------
# Service level: rejections and whole-batch rollback
# ---------------------------------------------------------------------------

def test_graph_empty_or_non_list_rejected(conn):
    with pytest.raises(scheduler.SchedulerError):
        scheduler.submit_job_graph([])
    with pytest.raises(scheduler.SchedulerError):
        scheduler.submit_job_graph("not-a-list")


def test_graph_duplicate_ref_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            graph_item("dup", "t"), graph_item("dup", "t"),
        ])
    assert "duplicate ref" in str(exc.value)
    assert scheduler.list_jobs() == []
    assert scheduler.list_tenants() == []


def test_graph_inconsistent_tenant_weight_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            graph_item("a", "t", weight=1.0),
            graph_item("b", "t", weight=2.0),
        ])
    assert "inconsistent weights" in str(exc.value)
    # Nothing was created, and no weight leaked through.
    assert scheduler.list_jobs() == []
    assert scheduler.list_tenants() == []


def test_graph_weight_mismatch_does_not_change_existing_tenant(conn):
    scheduler.upsert_tenant("t", 3.0)
    with pytest.raises(scheduler.SchedulerError):
        scheduler.submit_job_graph([
            graph_item("a", "t", weight=1.0),
            graph_item("b", "t", weight=2.0),
        ])
    assert scheduler.get_tenant("t")["weight"] == 3.0
    assert scheduler.list_jobs() == []


def test_graph_over_capacity_rejected_and_rolled_back(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            graph_item("ok", "other"),
            graph_item("big", "t", gpus=99),
        ])
    assert "exceeds cluster capacity" in str(exc.value)
    assert scheduler.list_jobs() == []
    assert scheduler.list_tenants() == []


def test_graph_unknown_external_prerequisite_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            graph_item("a", "t", depends_on=["J000999"]),
        ])
    assert "unknown depends_on" in str(exc.value)
    assert scheduler.list_jobs() == []


def test_graph_unknown_batch_ref_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            graph_item("a", "t", depends_on_refs=["ghost"]),
        ])
    assert "unknown batch ref" in str(exc.value)
    assert scheduler.list_jobs() == []


def test_graph_self_dependency_via_own_ref_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            graph_item("a", "t", depends_on_refs=["a"]),
        ])
    assert "dependency cycle" in str(exc.value)
    assert scheduler.list_jobs() == []


def test_graph_cycle_inside_batch_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            graph_item("a", "t", depends_on_refs=["b"]),
            graph_item("b", "t", depends_on_refs=["c"]),
            graph_item("c", "t", depends_on_refs=["a"]),
        ])
    assert "dependency cycle" in str(exc.value)
    assert scheduler.list_jobs() == []
    assert scheduler.list_tenants() == []


def test_graph_existing_edges_participate_in_whole_graph_check(conn):
    # Existing graph: J000002 -> J000001.
    ext1 = scheduler.submit_job("base", 1, 5)             # J000001
    ext2 = scheduler.submit_job("base", 1, 5,             # J000002
                                depends_on=[ext1["id"]])
    # A valid batch may freely depend on nodes of the existing graph...
    ok = scheduler.submit_job_graph([
        graph_item("ok", "t", depends_on=[ext2["id"]]),
    ])
    assert ok["jobs"][0]["blocked_by"] == [
        {"id": ext2["id"], "status": "QUEUED"}]

    # ...but a batch whose new edges close a cycle (even one contained among
    # the new nodes, while also referencing existing prerequisites) is
    # rejected against the whole graph and leaves the prior batch's jobs
    # untouched.
    before = {j["id"] for j in scheduler.list_jobs()}
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            graph_item("n1", "t", depends_on=[ext1["id"]],
                       depends_on_refs=["n2"]),
            graph_item("n2", "t", depends_on_refs=["n1"]),
        ])
    assert "dependency cycle" in str(exc.value)
    assert {j["id"] for j in scheduler.list_jobs()} == before


def test_graph_parameter_shape_errors(conn):
    bad_payloads = [
        [{"tenant_id": "t", "gpus": 1, "duration_min": 1, "weight": 1}],
        [graph_item("a", " ", gpus=1)],
        [graph_item("a", "t", gpus=0)],
        [graph_item("a", "t", gpus=True)],
        [graph_item("a", "t", gpus=1, minutes=0)],
        [graph_item("a", "t", weight=0)],
        [graph_item("a", "t", depends_on="J000001")],
        [graph_item("a", "t", depends_on_refs=[""])],
        [graph_item("a", "t", depends_on_refs=[123])],
        [42],
    ]
    for payload in bad_payloads:
        with pytest.raises(scheduler.SchedulerError):
            scheduler.submit_job_graph(payload)
    assert scheduler.list_jobs() == []
    assert scheduler.list_tenants() == []


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def test_http_graph_happy_path(client):
    ext = client.post("/jobs", json={
        "tenant_id": "team-x", "gpus": 1, "duration_min": 5}).json()["id"]

    r = client.post("/jobs/graph", json={"jobs": [
        {"ref": "train", "tenant_id": "team-a", "gpus": 2,
         "duration_min": 10, "weight": 1,
         "depends_on": [ext], "depends_on_refs": ["prep"]},
        {"ref": "prep", "tenant_id": "team-a", "gpus": 1,
         "duration_min": 5, "weight": 1},
        {"ref": "serve", "tenant_id": "team-b", "gpus": 1,
         "duration_min": 5, "weight": 2,
         "depends_on_refs": ["train", "prep"]},
    ]})
    assert r.status_code == 201, r.text
    body = r.json()
    assert set(body["ref_ids"]) == {"train", "prep", "serve"}
    ids = body["ref_ids"]
    # Array order -> sequential ids.
    assert list(ids.values()) == sorted(ids.values())
    jobs = {j["id"]: j for j in body["jobs"]}
    assert jobs[ids["train"]]["depends_on"] == [ext, ids["prep"]]
    # Sorted by job id (train J...2 < prep J...3).
    assert jobs[ids["serve"]]["depends_on"] == \
        [ids["train"], ids["prep"]]
    # New jobs are immediately visible and judged by preview/batch.
    preview = client.post("/scheduler/preview").json()
    assert ext in preview["selection_order"]
    assert ids["train"] not in preview["selection_order"]


def test_http_graph_weight_mismatch_400_and_rollback(client):
    r = client.post("/jobs/graph", json={"jobs": [
        {"ref": "a", "tenant_id": "t", "gpus": 1,
         "duration_min": 1, "weight": 1},
        {"ref": "b", "tenant_id": "t", "gpus": 1,
         "duration_min": 1, "weight": 2},
    ]})
    assert r.status_code == 400
    assert "inconsistent weights" in r.json()["error"]
    # Whole batch rolled back: no tenant, no jobs, no queue entries.
    assert client.get("/tenants/t").status_code == 404
    assert client.get("/jobs").json() == []


def test_http_graph_cycle_and_unknown_refs_400(client):
    r = client.post("/jobs/graph", json={"jobs": [
        {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 1,
         "weight": 1, "depends_on_refs": ["b"]},
        {"ref": "b", "tenant_id": "t", "gpus": 1, "duration_min": 1,
         "weight": 1, "depends_on_refs": ["a"]},
    ]})
    assert r.status_code == 400
    assert "dependency cycle" in r.json()["error"]

    r = client.post("/jobs/graph", json={"jobs": [
        {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 1,
         "weight": 1, "depends_on_refs": ["ghost"]},
    ]})
    assert r.status_code == 400
    assert "unknown batch ref" in r.json()["error"]

    r = client.post("/jobs/graph", json={"jobs": [
        {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 1,
         "weight": 1, "depends_on": ["J000999"]},
    ]})
    assert r.status_code == 400
    assert "unknown depends_on" in r.json()["error"]

    assert client.get("/jobs").json() == []


def test_http_graph_body_validation_422(client):
    # Empty jobs array.
    assert client.post("/jobs/graph", json={"jobs": []}).status_code == 422
    # Missing jobs field.
    assert client.post("/jobs/graph", json={}).status_code == 422
    # Missing required fields (ref/tenant/gpus/duration/weight required).
    assert client.post("/jobs/graph", json={"jobs": [
        {"tenant_id": "t", "gpus": 1, "duration_min": 1}
    ]}).status_code == 422
    # Blank ref, zero gpus, non-string dependency entries.
    assert client.post("/jobs/graph", json={"jobs": [
        {"ref": "  ", "tenant_id": "t", "gpus": 1,
         "duration_min": 1, "weight": 1}
    ]}).status_code == 422
    assert client.post("/jobs/graph", json={"jobs": [
        {"ref": "a", "tenant_id": "t", "gpus": 0,
         "duration_min": 1, "weight": 1}
    ]}).status_code == 422
    assert client.post("/jobs/graph", json={"jobs": [
        {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 1,
         "weight": 1, "depends_on_refs": [""]}
    ]}).status_code == 422


def test_http_graph_failed_batch_preserves_queue_and_weights(client):
    # Pre-existing tenant with a fixed weight and a queued job. The job is
    # submitted with weight 7 so the single-submission weight update leaves
    # the tenant at exactly 7 (isolating what the failed graph batch does).
    client.post("/tenants", json={"tenant_id": "t", "weight": 7.0})
    j1 = client.post("/jobs", json={
        "tenant_id": "t", "gpus": 1, "duration_min": 5, "weight": 7.0}
    ).json()["id"]

    r = client.post("/jobs/graph", json={"jobs": [
        {"ref": "a", "tenant_id": "t", "gpus": 1,
         "duration_min": 1, "weight": 1},
        {"ref": "b", "tenant_id": "t", "gpus": 1,
         "duration_min": 1, "weight": 2},
    ]})
    assert r.status_code == 400

    # Weight unchanged and the original queue is untouched.
    assert client.get("/tenants/t").json()["weight"] == 7.0
    jobs = client.get("/jobs?tenant_id=t").json()
    assert [j["id"] for j in jobs] == [j1]


def test_http_graph_then_preview_and_batch(client):
    # External prerequisite (team-x) plus an in-batch chain across tenants.
    ext = client.post("/jobs", json={
        "tenant_id": "team-x", "gpus": 1, "duration_min": 5}).json()["id"]
    r = client.post("/jobs/graph", json={"jobs": [
        {"ref": "a", "tenant_id": "team-a", "gpus": 1,
         "duration_min": 5, "weight": 1, "depends_on": [ext]},
        {"ref": "b", "tenant_id": "team-b", "gpus": 1,
         "duration_min": 5, "weight": 1, "depends_on_refs": ["a"]},
    ]})
    ids = r.json()["ref_ids"]

    # First batch dispatches only the external prerequisite; team-a's head
    # (a) is blocked and team-b transitively blocked.
    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [ext]
    client.post(f"/jobs/{ext}/complete")

    # Next batch releases a; b still waits on a.
    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [ids["a"]]
    client.post(f"/jobs/{ids['a']}/complete")

    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [ids["b"]]
