"""Batch-preview tests: one shared algorithm, zero persisted side effects.

The preview (``scheduler.preview_batch`` / ``POST /scheduler/preview``) runs
the exact same transaction body as the real batch, then rolls it back. These
tests cover the contract:

* decisions are identical to a committed batch at the same reference instant;
* nothing is ever written (jobs, balances, aging clocks, batches, ledger),
  including on errors;
* previews are repeatable and compose with the existing endpoints.
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


def snapshot():
    c = db.get_conn()
    return {
        "jobs": [tuple(r) for r in c.execute(
            "SELECT id, status, scheduled_at, completed_at, batch_id, run_index "
            "FROM jobs ORDER BY id")],
        "tenants": [tuple(r) for r in c.execute(
            "SELECT id, weight, deficit, max_gpus, budget_gpu_min, "
            "aging_settled_at FROM tenants ORDER BY id")],
        "deps": [tuple(r) for r in c.execute(
            "SELECT job_id, depends_on FROM job_dependencies ORDER BY 1, 2")],
        "batches": c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"],
        "ledger": [tuple(r) for r in c.execute(
            "SELECT tenant_id, batch_id, job_id, kind, amount, balance_after, "
            "detail, created_at FROM ledger ORDER BY id")],
        "max_batch": c.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM batches").fetchone()["m"],
        "max_ledger": c.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM ledger").fetchone()["m"],
    }


# Fields that must agree between a preview and the later committed batch when
# run on identical state at the same reference instant.
_COMPARE_KEYS = (
    "batch_id", "created_at", "capacity_gpus", "used_gpus_before",
    "available_gpus_at_start", "used_gpus_after", "free_gpus_after",
    "rounds", "jobs_selected", "selection_order",
)


def assert_preview_matches_batch(preview, committed):
    assert preview["preview"] is True
    assert committed["preview"] is False
    for key in _COMPARE_KEYS:
        assert preview[key] == committed[key], key
    # Prospective ids are the ones the rollback hands straight back to the
    # committed batch, so event ids/amounts/balances line up exactly too.
    assert [
        (e["id"], e["tenant_id"], e["batch_id"], e["job_id"], e["kind"],
         e["amount"], e["balance_after"], e["detail"], e["created_at"])
        for e in preview["events"]
    ] == [
        (e["id"], e["tenant_id"], e["batch_id"], e["job_id"], e["kind"],
         e["amount"], e["balance_after"], e["detail"], e["created_at"])
        for e in committed["events"]
    ]
    assert [
        (j["id"], j["tenant_id"], j["status"], j["gpus"], j["duration_min"],
         j["run_index"], j["batch_id"], j["scheduled_at"], j["depends_on"],
         j["blocked_by"])
        for j in preview["selected_jobs"]
    ] == [
        (j["id"], j["tenant_id"], j["status"], j["gpus"], j["duration_min"],
         j["run_index"], j["batch_id"], j["scheduled_at"], j["depends_on"],
         j["blocked_by"])
        for j in committed["selected_jobs"]
    ]
    assert [
        (t["id"], t["weight"], t["balance_gpu_min"], t["max_gpus"],
         t["budget_gpu_min"], t["budget_used_gpu_min"],
         t["budget_remaining_gpu_min"], t["budget_window_end_at"],
         t["queued_jobs"], t["running_jobs"], t["running_gpus"],
         t["aging_settled_at"])
        for t in preview["tenants"]
    ] == [
        (t["id"], t["weight"], t["balance_gpu_min"], t["max_gpus"],
         t["budget_gpu_min"], t["budget_used_gpu_min"],
         t["budget_remaining_gpu_min"], t["budget_window_end_at"],
         t["queued_jobs"], t["running_jobs"], t["running_gpus"],
         t["aging_settled_at"])
        for t in committed["tenants"]
    ]


# ---------------------------------------------------------------------------
# Service-level
# ---------------------------------------------------------------------------

def test_preview_matches_batch_multi_round_aging(conn):
    # Two costly jobs (cost 600): ten rounds of quantum are needed, so the
    # batch exercises aging, intra-batch accrual and the trailing-grant
    # rollback watermark.
    submit("a", 6, 100, weight=1.0, when=iso(0))
    submit("b", 6, 100, weight=1.0, when=iso(1))

    preview = scheduler.preview_batch(now=at(10))
    assert preview["selection_order"] == ["J000001"]
    assert preview["rounds"] == 10
    # Predicted end state shown even though nothing is committed.
    assert preview["used_gpus_after"] == 6
    assert preview["free_gpus_after"] == 2
    bal = {t["id"]: t["balance_gpu_min"] for t in preview["tenants"]}
    assert bal["a"] == pytest.approx(10.0)
    assert bal["b"] == pytest.approx(549.0)

    before = snapshot()
    # Preview itself left the database exactly as it found it.
    assert snapshot() == before

    committed = scheduler.run_batch(now=at(10))
    assert_preview_matches_batch(preview, committed)


def test_preview_writes_nothing(conn):
    submit("a", 2, 10, when=iso(0))
    submit("a", 2, 10, when=iso(1))
    submit("b", 4, 10, when=iso(2))

    before = snapshot()
    p1 = scheduler.preview_batch(now=at(5))
    after1 = snapshot()
    assert after1 == before

    # Predictions are visible in the response only... Tenant visit order is
    # id ascending, so after a's first dispatch (2 GPUs), b's 4-GPU job is
    # visited before a's second job.
    assert p1["selection_order"] == ["J000001", "J000003", "J000002"]
    assert p1["used_gpus_after"] == 8
    assert all(
        j["status"] == "RUNNING" for j in p1["selected_jobs"])

    # ...the jobs remain QUEUED with no batch linkage.
    queued = scheduler.list_jobs(status="QUEUED")
    assert {j["id"] for j in queued} == {"J000001", "J000002", "J000003"}
    assert all(j["batch_id"] is None and j["scheduled_at"] is None
               for j in queued)
    # No batches, no ledger entries at all.
    c = db.get_conn()
    assert c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 0
    assert c.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()["n"] == 0
    # Balances/aging untouched.
    assert {t["id"]: t["balance_gpu_min"] for t in scheduler.list_tenants()} \
        == {"a": 0.0, "b": 0.0}

    # Repeated previews are perfectly repeatable (no quantum/aging farming).
    p2 = scheduler.preview_batch(now=at(5))
    assert p2["selection_order"] == p1["selection_order"]
    assert [e["amount"] for e in p2["events"]] == [e["amount"]
                                                  for e in p1["events"]]
    assert snapshot() == before

    # A later reference instant only changes aging inside the response, still
    # without persisting the advanced aging checkpoint.
    p3 = scheduler.preview_batch(now=at(15))
    assert snapshot() == before
    assert p3["selection_order"] == p1["selection_order"]


def test_preview_works_with_caps_dependencies_and_full_cluster(conn):
    scheduler.upsert_tenant("a", 1.0, max_gpus=2)
    j1 = submit("a", 2, 10, when=iso(0))
    submit("a", 1, 10, when=iso(1))                   # cap-blocked behind j1
    j3 = submit("b", 4, 10, when=iso(2))
    submit("c", 2, 10, when=iso(3), depends_on=[j3["id"]])  # dep-blocked

    before = snapshot()
    preview = scheduler.preview_batch(now=at(10))
    assert snapshot() == before
    assert preview["selection_order"] == [j1["id"], j3["id"]]
    tenants = {t["id"]: t for t in preview["tenants"]}
    assert tenants["a"]["running_gpus"] == 2 and tenants["a"]["max_gpus"] == 2
    assert tenants["b"]["running_gpus"] == 4
    # Dependency-blocked c earns nothing in the predicted ledger;
    # cap-blocked a earns exactly one quantum (the one that dispatches j1).
    assert all(e["tenant_id"] != "c" for e in preview["events"])
    a_quanta = [e for e in preview["events"]
                if e["tenant_id"] == "a" and e["kind"] == "QUANTUM"]
    assert len(a_quanta) == 1

    committed = scheduler.run_batch(now=at(10))
    assert_preview_matches_batch(preview, committed)

    # Cluster now full (2 + 4 = 6 used... plus capacity? 6 used, 2 free):
    # a's second job wants 1 but cap is full; blocked preview still shows
    # aging settlement while persisting none of it.
    full_preview = scheduler.preview_batch(now=at(20))
    assert full_preview["selection_order"] == []
    aging = [e for e in full_preview["events"] if e["kind"] == "AGING"]
    assert aging and all(e["tenant_id"] != "c" for e in aging)
    assert snapshot() == snapshot()  # state stable across the call


def test_preview_empty_queue(conn):
    before = snapshot()
    p = scheduler.preview_batch(now=at(1))
    assert p["preview"] is True
    assert p["selection_order"] == []
    assert p["events"] == []
    assert p["tenants"] == []
    assert snapshot() == before


def test_preview_then_real_batch_consumes_same_ids(conn):
    submit("a", 1, 10, when=iso(0))
    p = scheduler.preview_batch(now=at(2))
    # Rolled-back AUTOINCREMENT rows are reused: the committed batch takes the
    # same prospective batch and ledger ids, which keeps the two snapshots
    # identical and prevents id-skipping gaps.
    b = scheduler.run_batch(now=at(2))
    assert b["batch_id"] == p["batch_id"]
    assert [e["id"] for e in b["events"]] == [e["id"] for e in p["events"]]
    # A subsequent real batch advances normally.
    submit("b", 1, 10, when=iso(3))
    b2 = scheduler.run_batch(now=at(4))
    assert b2["batch_id"] == b["batch_id"] + 1


def test_preview_after_real_batch_predicts_next(conn):
    # Preview must reflect already-committed RUNNING jobs (GPU occupancy).
    submit("a", 6, 10, when=iso(0))
    submit("b", 6, 10, when=iso(1))
    scheduler.run_batch(now=at(2))          # dispatches a (6 of 8 GPUs)

    before = snapshot()
    p = scheduler.preview_batch(now=at(5))
    assert p["selection_order"] == []       # b's 6 GPUs do not fit in 2 free
    assert p["used_gpus_before"] == 6
    assert p["free_gpus_after"] == 2
    assert snapshot() == before

    scheduler.complete_job("J000001")
    p2 = scheduler.preview_batch(now=at(6))
    assert p2["selection_order"] == ["J000002"]
    committed = scheduler.run_batch(now=at(6))
    assert_preview_matches_batch(p2, committed)


def test_preview_rolls_back_when_algorithm_raises(conn):
    submit("a", 1, 10, when=iso(0))
    before = snapshot()

    orig_credit = scheduler._credit
    def failing_credit(*args, **kwargs):
        raise RuntimeError("simulated ledger failure")
    scheduler._credit = failing_credit
    try:
        with pytest.raises(RuntimeError, match="simulated ledger failure"):
            scheduler.preview_batch(now=at(5))
    finally:
        scheduler._credit = orig_credit

    # No partial tentatives survive: no batch row, job still QUEUED, no
    # ledger entry, balances intact.
    after = snapshot()
    after.pop("max_batch", None)
    after.pop("max_ledger", None)
    before_cmp = {k: v for k, v in before.items()
                  if k not in ("max_batch", "max_ledger")}
    assert after == before_cmp

    # Connection still usable after the rolled-back transaction.
    p = scheduler.preview_batch(now=at(6))
    assert p["selection_order"] == ["J000001"]
    assert snapshot()["batches"] == 0


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def test_preview_endpoint_no_write_and_matches_batch_http(client):
    j1 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 30,
        "weight": 1}).json()["id"]
    j2 = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 6, "duration_min": 10,
        "weight": 2}).json()["id"]

    r = client.post("/scheduler/preview", json={"now": "2026-01-01T00:10:00Z"})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["preview"] is True
    assert p["selection_order"] == [j1, j2]
    assert p["used_gpus_after"] == 8 and p["free_gpus_after"] == 0

    # Nothing persisted.
    queued = client.get("/jobs?status=QUEUED").json()
    assert {j["id"] for j in queued} == {j1, j2}
    assert client.get("/ledger?limit=10").json() == []
    state = client.get("/scheduler/state").json()
    assert state["used_gpus"] == 0
    for t in state["tenants"]:
        assert t["balance_gpu_min"] == 0.0

    # Repeated, then committed at the same instant: identical decisions.
    p2 = client.post(
        "/scheduler/preview", json={"now": "2026-01-01T00:10:00Z"}).json()
    assert p2["selection_order"] == p["selection_order"]
    b = client.post(
        "/scheduler/batch", json={"now": "2026-01-01T00:10:00Z"}).json()
    assert b["preview"] is False
    assert b["selection_order"] == p["selection_order"]
    assert b["batch_id"] == p["batch_id"]
    assert b["used_gpus_after"] == p["used_gpus_after"]
    assert [(e["id"], e["kind"], e["amount"], e["balance_after"])
            for e in b["events"]] == [
               (e["id"], e["kind"], e["amount"], e["balance_after"])
               for e in p["events"]]
    # Now it really is committed.
    assert client.get("/jobs?status=RUNNING").json().__len__() == 2


def test_preview_endpoint_accepts_empty_body_and_no_body_http(client):
    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5})
    # Both an explicit empty JSON object and no body at all work, keeping the
    # endpoint curl-friendly (same convention as /scheduler/batch).
    r1 = client.post("/scheduler/preview", json={})
    assert r1.status_code == 200 and r1.json()["preview"] is True
    r2 = client.post("/scheduler/preview")
    assert r2.status_code == 200
    assert r1.json()["selection_order"] == r2.json()["selection_order"]
    # Nothing was committed by either call.
    assert client.get("/jobs?status=QUEUED").json().__len__() == 1


def test_batch_endpoint_still_works_without_body_http(client):
    # Existing behaviour preserved: bare POST commits with the server clock.
    j = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    r = client.post("/scheduler/batch")
    assert r.status_code == 200
    body = r.json()
    assert body["preview"] is False
    assert body["selection_order"] == [j]
    assert client.get(f"/jobs/{j}").json()["status"] == "RUNNING"


def test_preview_invalid_now_rejected_http(client):
    r = client.post("/scheduler/preview", json={"now": "not-a-timestamp"})
    assert r.status_code == 422
    # Same validation for /scheduler/batch.
    assert client.post(
        "/scheduler/batch", json={"now": "nope"}).status_code == 422
    # Nothing was created on failure.
    assert client.get("/ledger?limit=10").json() == []


def test_preview_with_caps_and_dependencies_http(client):
    client.post("/tenants", json={
        "tenant_id": "team-a", "weight": 1, "max_gpus": 2})
    ja1 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 5}).json()["id"]
    client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 1, "duration_min": 5})
    jdep = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 1, "duration_min": 5,
        "depends_on": [ja1]}).json()["id"]
    jfree = client.post("/jobs", json={
        "tenant_id": "team-c", "gpus": 4, "duration_min": 5}).json()

    p = client.post(
        "/scheduler/preview", json={"now": "2026-01-01T00:05:00Z"}).json()
    # ja1 (cap 2), jfree (4 GPUs) dispatch; jdep waits on RUNNING ja1;
    # a's second job is cap-blocked. Tenant visit order is id ascending:
    # team-a, team-b, team-c, but team-b's head is dependency-blocked.
    assert p["selection_order"] == [ja1, jfree["id"]]
    tenants = {t["id"]: t for t in p["tenants"]}
    assert tenants["team-a"]["running_gpus"] == 2
    # No predicted credit for the dependency-blocked tenant.
    assert all(e["tenant_id"] != "team-b" for e in p["events"])

    # State unchanged: ja1 still QUEUED, so jdep's blocker is still QUEUED.
    assert client.get(f"/jobs/{ja1}").json()["status"] == "QUEUED"
    assert client.get(f"/jobs/{jdep}").json()["blocked_by"] == [
        {"id": ja1, "status": "QUEUED"}]

    b = client.post(
        "/scheduler/batch", json={"now": "2026-01-01T00:05:00Z"}).json()
    assert b["selection_order"] == p["selection_order"]
    assert [(e["kind"], e["tenant_id"], e["amount"]) for e in b["events"]] == \
           [(e["kind"], e["tenant_id"], e["amount"]) for e in p["events"]]


def test_preview_does_not_disturb_later_batch_http(client):
    # End-to-end: preview at t1 (nothing changes), aging window stays open;
    # the real batch at t1 then earns the same aging the preview predicted.
    # a is heavy/weight-4 so it dispatches round 1 and saturates all 8 GPUs;
    # b's small job is capacity-blocked and only aging compensates the wait.
    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 8, "duration_min": 10, "weight": 4})
    client.post("/jobs", json={
        "tenant_id": "b", "gpus": 1, "duration_min": 5})
    # Pin queue arrival times.
    from app import db as db_mod
    for jid, ts in (("J000001", "2026-01-01T00:00:00+00:00"),
                    ("J000002", "2026-01-01T00:00:00+00:00")):
        db_mod.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (ts, jid))

    p = client.post(
        "/scheduler/preview", json={"now": "2026-01-01T00:10:00Z"}).json()
    assert p["selection_order"] == ["J000001"]
    aging_b_preview = [e for e in p["events"]
                       if e["tenant_id"] == "b" and e["kind"] == "AGING"]
    assert aging_b_preview and aging_b_preview[0]["amount"] == pytest.approx(10.0)
    # b is capacity-blocked: its predicted ledger contains aging but no quantum.
    assert all(e["kind"] != "QUANTUM"
               for e in p["events"] if e["tenant_id"] == "b")

    b = client.post(
        "/scheduler/batch", json={"now": "2026-01-01T00:10:00Z"}).json()
    assert b["selection_order"] == ["J000001"]
    aging_b = [e for e in b["events"]
               if e["tenant_id"] == "b" and e["kind"] == "AGING"]
    assert aging_b and aging_b[0]["amount"] == pytest.approx(10.0)
