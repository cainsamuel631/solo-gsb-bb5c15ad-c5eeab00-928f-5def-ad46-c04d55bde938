from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import db, scheduler
from .config import get_settings
from .scheduler import SchedulerError


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db.connect(settings.db_path)
    yield
    conn = db.get_conn()
    conn.close()


app = FastAPI(
    title="Multi-tenant GPU Fair-Queuing Scheduler",
    version="1.0.0",
    description=(
        "Weighted Deficit Round-Robin GPU job scheduler with wait-time aging. "
        "Cost = gpus * duration_min (GPU-minutes); capacity is never exceeded."
    ),
    lifespan=lifespan,
)


@app.exception_handler(SchedulerError)
async def scheduler_error_handler(request, exc: SchedulerError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TenantIn(BaseModel):
    tenant_id: str = Field(..., min_length=1, examples=["team-a"])
    weight: float = Field(..., gt=0, examples=[2.0])
    max_gpus: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on GPUs the tenant may hold concurrently (RUNNING "
            "jobs). Provide a positive integer to set/update the cap; send "
            "null explicitly to clear it. Omit the field entirely to leave "
            "an existing tenant's cap unchanged (new tenants get unlimited). "
            "Note: the field must be omitted, not null, to preserve a cap."
        ),
        examples=[4],
    )
    budget_gpu_min: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional rolling 24-hour GPU-minute budget: the tenant may be "
            "dispatched at most this many gpus*duration_min of job cost in "
            "the window (now-24h, now] ending at each batch's reference "
            "instant. Provide a positive integer to set/update the budget; "
            "send null explicitly to clear it (unlimited). Omit the field "
            "entirely to leave an existing tenant's budget unchanged (new "
            "tenants get unlimited). The field must be omitted, not null, "
            "to preserve a budget."
        ),
        examples=[1440],
    )

    @field_validator("max_gpus", mode="before")
    @classmethod
    def _reject_bool_for_int(cls, value):
        # Run before Pydantic's bool->int coercion, since JSON booleans must
        # not be accepted as an integer GPU cap.
        if isinstance(value, bool):
            raise ValueError("max_gpus must be a positive integer or null")
        return value

    @field_validator("budget_gpu_min", mode="before")
    @classmethod
    def _reject_bool_for_budget(cls, value):
        # JSON booleans must not be accepted as an integer GPU-minute budget.
        if isinstance(value, bool):
            raise ValueError(
                "budget_gpu_min must be a positive integer or null"
            )
        return value


class JobIn(BaseModel):
    tenant_id: str = Field(..., min_length=1, examples=["team-a"])
    gpus: int = Field(..., gt=0, examples=[2])
    duration_min: int = Field(..., gt=0, examples=[60])
    weight: float = Field(
        default=1.0, gt=0,
        description="Tenant weight; auto-registers the tenant or updates it.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "Job ids that must reach COMPLETED before this job is schedulable. "
            "Dependencies are cluster-global (cross-tenant allowed); unknown "
            "ids and dependency cycles are rejected."
        ),
        examples=[["J000001"]],
    )

    @field_validator("depends_on")
    @classmethod
    def _non_empty_dependency_ids(cls, value: list[str]) -> list[str]:
        if any(not isinstance(d, str) or not d.strip() for d in value):
            raise ValueError("depends_on entries must be non-empty strings")
        return value


class GraphJobIn(BaseModel):
    ref: str = Field(
        ..., min_length=1,
        description=(
            "Client-side label unique within this batch request. Other items "
            "reference this job through depends_on_refs; the response maps "
            "each ref to the allocated job id."
        ),
        examples=["preprocess"],
    )
    tenant_id: str = Field(..., min_length=1, examples=["team-a"])
    gpus: int = Field(..., gt=0, examples=[2])
    duration_min: int = Field(..., gt=0, examples=[60])
    weight: float = Field(
        ..., gt=0,
        description=(
            "Tenant weight. All items of the same tenant in one batch must "
            "agree on it; the batch auto-registers the tenant or updates it "
            "exactly once."
        ),
        examples=[1.0],
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "Prerequisite job ids that must already exist (cluster-global; "
            "cross-tenant allowed) and reach COMPLETED before this job is "
            "schedulable. To reference another item of the same batch use "
            "depends_on_refs instead."
        ),
        examples=[["J000001"]],
    )
    depends_on_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Refs of other items in THIS batch that must COMPLETE first. "
            "Forward references are allowed (items need not be topologically "
            "sorted); an unknown ref rejects the whole batch."
        ),
        examples=[["preprocess"]],
    )

    @field_validator("ref")
    @classmethod
    def _non_empty_ref(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ref must be a non-empty string")
        return value

    @field_validator("depends_on", "depends_on_refs")
    @classmethod
    def _non_empty_dependency_ids(cls, value: list[str]) -> list[str]:
        if any(not isinstance(d, str) or not d.strip() for d in value):
            raise ValueError("dependency entries must be non-empty strings")
        return value


class JobGraphIn(BaseModel):
    jobs: list[GraphJobIn] = Field(
        ..., min_length=1,
        description=(
            "Non-empty array of jobs submitted atomically as one dependency "
            "graph. Any validation failure (duplicate ref, inconsistent "
            "weights for one tenant, over-capacity GPUs, unknown external "
            "job id or batch ref, whole-graph cycle) rejects the whole "
            "batch: no tenant weight, job or queue is changed."
        ),
    )


class JobDependenciesIn(BaseModel):
    depends_on: list[str] = Field(
        ...,
        description=(
            "Full replacement set of prerequisite job ids (cluster-global; "
            "cross-tenant allowed). Deduplicated and sorted by job id; send "
            "an empty array to detach the job from all prerequisites. "
            "Unknown ids, self-dependence and replacement-induced cycles "
            "are rejected and leave the current dependencies unchanged."
        ),
        examples=[["J000001", "J000002"]],
    )

    @field_validator("depends_on")
    @classmethod
    def _non_empty_dependency_ids(cls, value: list[str]) -> list[str]:
        if any(not isinstance(d, str) or not d.strip() for d in value):
            raise ValueError("depends_on entries must be non-empty strings")
        return value


class BatchIn(BaseModel):
    now: datetime | None = Field(
        default=None,
        description=(
            "Optional reference instant (UTC, ISO-8601) for aging settlement. "
            "Omit to use the server clock. Passing the same value to a preview "
            "and the subsequent real batch keeps their decisions identical."
        ),
        examples=["2026-09-23T12:00:00Z"],
    )

    def reference(self) -> datetime | None:
        if self.now is None:
            return None
        # Match the service layer: naive timestamps are interpreted as UTC;
        # offset-aware ones are normalised to UTC.
        if self.now.tzinfo is None:
            return self.now.replace(tzinfo=timezone.utc)
        return self.now.astimezone(timezone.utc)


class MaintenanceWindowIn(BaseModel):
    start_at: datetime = Field(
        ..., description="Window start (UTC, ISO-8601), inclusive.",
        examples=["2026-09-24T02:00:00Z"],
    )
    end_at: datetime = Field(
        ..., description="Window end (UTC, ISO-8601), exclusive; must be "
                         "strictly later than start_at.",
        examples=["2026-09-24T04:00:00Z"],
    )
    reserved_gpus: int = Field(
        ..., ge=0,
        description=(
            "GPUs withheld from scheduling throughout the half-open window. "
            "Overlapping windows add: the peak combined reservation over the "
            "new window's coverage (existing windows plus this one) must not "
            "exceed cluster capacity."
        ),
        examples=[4],
    )

    @field_validator("reserved_gpus", mode="before")
    @classmethod
    def _reject_bool_for_int(cls, value):
        # JSON booleans must not be accepted as an integer GPU reservation.
        if isinstance(value, bool):
            raise ValueError("reserved_gpus must be an integer")
        return value

    def start(self) -> datetime:
        if self.start_at.tzinfo is None:
            return self.start_at.replace(tzinfo=timezone.utc)
        return self.start_at.astimezone(timezone.utc)

    def end(self) -> datetime:
        if self.end_at.tzinfo is None:
            return self.end_at.replace(tzinfo=timezone.utc)
        return self.end_at.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Tenant / state
# ---------------------------------------------------------------------------

@app.post("/tenants", tags=["tenants"], status_code=201)
def create_or_update_tenant(body: TenantIn):
    # Distinguish "field omitted" (preserve existing value) from an explicit
    # JSON null (clear it) for both the concurrent-GPU cap and the rolling
    # 24h GPU-minute budget.
    kwargs = {}
    if "max_gpus" in body.model_fields_set:
        kwargs["max_gpus"] = body.max_gpus
    if "budget_gpu_min" in body.model_fields_set:
        kwargs["budget_gpu_min"] = body.budget_gpu_min
    return scheduler.upsert_tenant(body.tenant_id, body.weight, **kwargs)


@app.get("/tenants", tags=["tenants"])
def get_tenants():
    """All tenants with weights and current GPU-minute balances."""
    return scheduler.list_tenants()


@app.get("/tenants/{tenant_id}", tags=["tenants"])
def get_tenant(tenant_id: str):
    try:
        return scheduler.get_tenant(tenant_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/scheduler/state", tags=["scheduler"])
def scheduler_state():
    """Cluster capacity, free/used GPUs, tuning knobs and tenant balances."""
    return scheduler.scheduler_state()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.post("/jobs", tags=["jobs"], status_code=201)
def submit_job(
    body: JobIn,
    response: Response,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        description=(
            "Optional client-generated idempotency key. A non-empty value "
            "identifies, together with the request body's tenant_id, one "
            "logical submission. On a retry the previously saved request "
            "fingerprint is compared inside the transaction before any "
            "capacity check: the first request creates the job (201); retries "
            "with identical parameters return the job's current object (200) "
            "without re-applying any side effect, even if CLUSTER_GPUS was "
            "later lowered so the original request now exceeds the present "
            "capacity, while the same key reused with different parameters "
            "gets 409 — including when those new parameters themselves exceed "
            "current capacity. Only a key not seen before is validated "
            "against current capacity and creates the job; a failed "
            "submission does not consume the key. Omit the header (or send it "
            "blank) for the original non-idempotent behaviour."
        ),
    ),
):
    # A missing or blank header keeps the original interface exactly.
    key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else None
    if key is None:
        return scheduler.submit_job(
            tenant_id=body.tenant_id,
            gpus=body.gpus,
            duration_min=body.duration_min,
            weight=body.weight,
            depends_on=body.depends_on,
        )
    job, created = scheduler.submit_job_idempotent(
        tenant_id=body.tenant_id,
        gpus=body.gpus,
        duration_min=body.duration_min,
        weight=body.weight,
        depends_on=body.depends_on,
        idempotency_key=key,
    )
    if not created:
        # Replay of a previously committed submission: the job's current
        # object (it may since have been dispatched/completed/cancelled).
        response.status_code = 200
    return job


@app.post("/jobs/graph", tags=["jobs"], status_code=201)
def submit_job_graph(body: JobGraphIn):
    """Atomically submit a batch of jobs forming one dependency graph.

    Every item carries a batch-unique ``ref`` (client label), its
    ``tenant_id``, ``gpus``, ``duration_min`` and ``weight``, plus two
    prerequisite lists: ``depends_on`` names job ids that already exist
    (cross-tenant allowed) while ``depends_on_refs`` names refs of other
    items in this same batch — forward references are allowed, so the items
    need not be topologically sorted. All items for one tenant must agree on
    ``weight`` (a mismatch rejects the whole batch).

    Parameter shapes, duplicate refs, per-tenant weight agreement, cluster
    capacity, external job-id existence, in-batch ref existence and a cycle
    check over the whole would-be graph all run inside one write
    transaction; any failure rolls the entire batch back with no change to
    tenant weights or queues. On success the sequential job ids are
    allocated in array order, tenants are auto-registered/updated once at
    the agreed weight, and every dependency edge (deduplicated and sorted)
    is written. Returns the ref→id mapping plus the created job objects;
    they are ordinary QUEUED jobs, so the next ``/scheduler/preview`` or
    ``/scheduler/batch`` immediately judges them through the existing
    dependency readiness path.
    """
    return scheduler.submit_job_graph(
        [item.model_dump() for item in body.jobs]
    )


@app.get("/jobs", tags=["jobs"])
def list_jobs(
    status: str | None = Query(
        None, description="QUEUED|RUNNING|COMPLETED|CANCELLED"),
    tenant_id: str | None = None,
):
    try:
        return scheduler.list_jobs(status=status, tenant_id=tenant_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str):
    try:
        return scheduler.get_job(job_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.patch("/jobs/{job_id}/dependencies", tags=["jobs"])
def revise_job_dependencies(job_id: str, body: JobDependenciesIn):
    """Atomically replace a QUEUED job's prerequisite set.

    The body's ``depends_on`` array substitutes every existing edge (send
    ``[]`` to detach the job from all prerequisites); ids are deduplicated
    and sorted, and cross-tenant prerequisites are allowed. Unknown ids,
    self-dependence and replacement-induced cycles are rejected (400) with
    the stored dependencies left unchanged; a non-QUEUED job gets 409 and
    an unknown job 404. The revision commits under the same lock and
    write-transaction style as batch scheduling, so the next preview or
    committed batch already judges the new edges. The first submission's
    idempotency fingerprint is preserved: replaying the original
    ``POST /jobs`` request returns this job's current (revised) object
    without restoring the original dependencies.
    """
    return scheduler.revise_dependencies(job_id, body.depends_on)


@app.post("/jobs/{job_id}/cancel", tags=["jobs"])
def cancel_job(job_id: str):
    """Cancel a QUEUED (or RUNNING) job. Queued cancellation frees no GPU;
    RUNNING cancellation releases GPUs for subsequent batches."""
    try:
        return scheduler.cancel_job(job_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/jobs/{job_id}/cancel-impact", tags=["jobs"])
def cancel_impact(job_id: str):
    """Preview the queued descendants cancelling this job would block forever.

    Computes the transitive closure over the cluster-global (cross-tenant)
    dependency graph: CANCELLED nodes are traversed through (they can never
    complete, so the blockage propagates past them) while COMPLETED nodes
    stop it. Returns the affected QUEUED jobs sorted by job id, each with
    one shortest dependency chain from the target. Pure read: nothing is
    written. Unknown job -> 404; COMPLETED/CANCELLED target -> 409.
    """
    return scheduler.cancel_impact(job_id)


@app.post("/jobs/{job_id}/cancel-cascade", tags=["jobs"])
def cancel_cascade(job_id: str):
    """Cancel this job and every still-QUEUED job its cancellation dooms.

    Recomputes the impact closure inside the same process-wide write lock
    and ``BEGIN IMMEDIATE`` transaction as single-job cancellation and
    dependency revision, then atomically cancels the target and every
    QUEUED job in the closure (a RUNNING target's GPUs are released for
    subsequent batches). Already-CANCELLED intermediate nodes are used for
    traversal only and are not modified again. Unknown job -> 404;
    COMPLETED/CANCELLED target -> 409.
    """
    return scheduler.cancel_cascade(job_id)


@app.post("/jobs/{job_id}/complete", tags=["jobs"])
def complete_job(job_id: str):
    """Mark a RUNNING job COMPLETED, releasing its GPUs."""
    try:
        return scheduler.complete_job(job_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/jobs/{job_id}/maintenance-slots", tags=["jobs"])
def job_maintenance_slots(
    job_id: str,
    start: datetime = Query(
        ..., description="Range start (UTC, ISO-8601), inclusive."),
    end: datetime = Query(
        ..., description="Range end (UTC, ISO-8601), exclusive."),
):
    """Feasible estimated-start slots for a QUEUED job within ``[start, end)``.

    The current RUNNING GPU usage is held as a fixed baseline across the
    whole horizon (no future releases are speculated) and the persisted
    maintenance windows are overlaid on the job's entire predicted run
    ``[t, t + duration_min)``: a start instant ``t`` is feasible iff
    baseline + job GPUs + combined reservation never exceeds cluster
    capacity. Returns the merged feasible slots — each with inclusive
    markers at both ends (a degenerate slot is an isolated feasible
    instant) — plus the earliest feasible instant (``null`` when the list
    is empty). Only QUEUED jobs are evaluated (others get 409); the result
    is purely a resource-capacity view. Returns 400 when ``end`` is not
    later than ``start``.
    """
    try:
        return scheduler.job_maintenance_slots(job_id, start=start, end=end)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


# ---------------------------------------------------------------------------
# Queue & scheduling
# ---------------------------------------------------------------------------

@app.get("/queue", tags=["scheduler"])
def get_queue():
    """FIFO per-tenant queues, positions, weights and balances."""
    return scheduler.queue_overview()


@app.post("/scheduler/batch", tags=["scheduler"])
def schedule_next_batch(body: BatchIn | None = None):
    """Run one WDRR+aging batch; returns selection order and balances.

    The request body is optional (a bare ``POST`` keeps the original
    behaviour and uses the server clock as the reference instant).
    """
    return scheduler.run_batch(now=None if body is None else body.reference())


@app.post("/scheduler/preview", tags=["scheduler"])
def preview_next_batch(body: BatchIn | None = None):
    """Dry-run a batch without writing anything.

    Runs the exact same WDRR/aging/cap/dependency decision path as
    ``/scheduler/batch`` inside a transaction that is rolled back, so the
    response — predicted dispatch order, GPU usage after the batch, tenant
    balances and credit events — matches the next committed batch at the same
    reference instant on the same database state, while job statuses,
    balances, aging clocks, batches and the ledger are all left untouched
    (errors leave no partial changes). The response carries
    ``"preview": true``; its ``batch_id``/event ids are prospective and are
    not persisted.
    """
    return scheduler.preview_batch(
        now=None if body is None else body.reference()
    )


@app.get("/ledger", tags=["scheduler"])
def get_ledger(
    tenant_id: str | None = None,
    batch_id: int | None = None,
    limit: int = Query(default=200, ge=1, le=10_000),
):
    """Credit/debit audit trail (AGING / QUANTUM / DISPATCH)."""
    return scheduler.list_ledger(
        tenant_id=tenant_id, batch_id=batch_id, limit=limit
    )


@app.get("/usage/gpu", tags=["usage"])
def get_gpu_usage(
    start: datetime = Query(
        ..., description="Window start (UTC, ISO-8601), inclusive."),
    end: datetime = Query(
        ..., description="Window end (UTC, ISO-8601), exclusive."),
    tenant_id: str | None = Query(
        None, description="Optional tenant filter; defaults to all tenants."),
):
    """Per-tenant and global GPU-minutes over a [start, end) UTC window.

    Usage derives from actual RUNNING intervals: finished jobs use
    scheduled_at -> completed_at, still-RUNNING jobs are truncated to ``end``.
    QUEUED jobs never count. Returns 400 when end is not later than start.
    """
    try:
        return scheduler.gpu_usage(start=start, end=end, tenant_id=tenant_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GPU maintenance windows
# ---------------------------------------------------------------------------

@app.post("/maintenance-windows", tags=["maintenance"], status_code=201)
def create_maintenance_window(body: MaintenanceWindowIn):
    """Register a persistent half-open UTC maintenance window.

    During ``[start_at, end_at)`` the given number of GPUs is reserved and
    withheld from dispatch (overlapping windows add; touching endpoints do
    not overlap). Returns 400 when ``end_at`` is not later than ``start_at``,
    when the reservation alone exceeds cluster capacity, or when the peak
    combined reservation over the new window's coverage — existing windows
    plus this one, evaluated inside the write transaction — would exceed
    the current cluster capacity (a peak exactly equal to it is allowed).
    RUNNING jobs are never preempted; only subsequent batch/preview
    dispatch respects the window.
    """
    try:
        return scheduler.create_maintenance_window(
            start=body.start(),
            end=body.end(),
            reserved_gpus=body.reserved_gpus,
        )
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/maintenance-windows", tags=["maintenance"])
def list_maintenance_windows(
    active_at: datetime | None = Query(
        None,
        description="Optional UTC instant; only windows active then "
                    "([start_at, end_at)) are returned.",
    ),
):
    """List maintenance windows (chronological), optionally active at ``active_at``."""
    if active_at is None:
        return scheduler.list_maintenance_windows()
    instant = (
        active_at.replace(tzinfo=timezone.utc)
        if active_at.tzinfo is None
        else active_at.astimezone(timezone.utc)
    )
    return scheduler.list_maintenance_windows(active_at=instant)
