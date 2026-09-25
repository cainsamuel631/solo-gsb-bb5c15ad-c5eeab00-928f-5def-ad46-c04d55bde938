"""Fair-queuing core: weighted deficit round-robin with wait-time aging.

The module is deliberately framework-free; FastAPI only translates HTTP into
calls here. All functions are serialized by ``app.db.get_lock()`` and use
explicit SQLite transactions so that a scheduling batch is atomic with
respect to submissions, cancellations and completions.

Cost unit
=========
A job's cost is its resource footprint:

    cost_gpu_min = gpus * duration_min

A tenant's *balance* (deficit counter) is measured in the same unit.

WDRR per batch
==============
1. Each active tenant (one with QUEUED jobs, tenant id order) receives
   ``base_quantum * weight`` GPU-minutes of credit every DRR round.
2. Visiting tenants in fixed order, the head-of-line job is dispatched when
   ``balance >= job_cost`` AND enough free GPUs exist AND the tenant's own
   concurrent-GPU cap (optional ``max_gpus``) still has room.
3. A full round in which nothing is dispatched ends the batch: blocked jobs
   keep their credit, and a later batch (after completions or aging) retries.

Per-tenant GPU caps
===================
A tenant may carry an optional ``max_gpus`` cap on concurrently allocated
GPUs (NULL = unlimited, the default for old and auto-registered tenants).
A head job fits only when ``running_gpus + head_gpus <= max_gpus`` in
addition to fitting the cluster's free capacity. A cap-blocked tenant — its
RUNNING jobs already fill the cap, or its head job alone exceeds the cap —
receives no QUANTUM for that round (so the endpoint cannot be farmed), but
other tenants in the same round are unaffected and still dispatch. Aging
compensation for cap-blocked tenants works exactly as for capacity-blocked
ones, bounding starvation. Completing or cancelling a RUNNING job releases
the GPUs against the cap immediately. Caps are set/cleared through
``POST /tenants``; submitting a job never changes an existing cap.

Rolling 24-hour GPU-minute budgets
==================================
A tenant may additionally carry an optional ``budget_gpu_min``: a positive
integer cap on the total job cost (``gpus * duration_min``) of jobs **dispatched**
inside the rolling 24-hour window ending at the batch reference instant, i.e.
the half-open-on-the-left interval ``(now - 24h, now]`` (NULL = unlimited, the
default for old and auto-registered tenants). Every dispatched job's full
estimated cost is charged exactly once at dispatch, keyed by ``scheduled_at``;
completing or cancelling a job never refunds the charge, so the spent amount is
the sum over the tenant's jobs (COMPLETED, still RUNNING or cancelled after
dispatch) whose dispatch instant lies inside the window. Jobs that remain
QUEUED — including budget-blocked ones — spend nothing.

Before each candidate dispatch in a batch the scheduler adds the candidate's
cost to (a) the cost of earlier batches' jobs dispatched inside ``(now-24h, now]``
and (b) the cost of jobs selected earlier in *this* batch (those became RUNNING
with ``scheduled_at = now`` and are therefore already inside the window, so they
are counted once through the same running total). A candidate that would exceed
the budget stays QUEUED and — exactly like a cap- or capacity-blocked head —
receives no QUANTUM for that round (the endpoint cannot be farmed), while every
other tenant is evaluated independently and keeps dispatching. The same code
path runs in a preview, whose rollback also discards the prospective charges;
nothing about a budget is written outside a committed batch.

``GET /tenants`` / ``GET /tenants/{id}`` report the cap as ``budget_gpu_min``
together with the ``budget_used_gpu_min`` and ``budget_remaining_gpu_min``
(``null`` when uncapped) evaluated at the server's current UTC instant, so the
rolling window is always measured live. Budgets are set/cleared through
``POST /tenants`` with the same omit/preserve vs explicit ``null``/clear
convention as ``max_gpus``; submitting a job never changes an existing budget.

Aging compensation
==================
While a tenant has a queued job, elapsed wall-clock time since its last
settlement is converted at ``AGING_RATE`` into extra credit (optionally
capped per batch). When the tenant's queue drains the aging clock resets, so
a later freshly submitted small job starts a clean wait window. Aging lets
small jobs behind large, high-weight tenants accumulate credit until they
become affordable, bounding starvation.

Only *scheduler-facing* waiting is compensated: a head-of-line job blocked
on unfinished prerequisites is not starving, it is simply not ready. A
tenant whose head is dependency-blocked at batch time receives no credit
(its checkpoint is merely advanced), and the same rule is applied
retrospectively when no batch observed the blocked period: credit accrues
from the moment the head job truly became ready — the LAST completion
among its prerequisites, or the instant a dependency revision made a
blocked job ready (recorded on the job as ``ready_since``) — taking the
LATER of that readiness instant and the tenant's existing aging settlement
point. A head job queued for 60 minutes whose prerequisite completed one
minute before the batch therefore earns one minute of aging, not sixty.

Reproducibility
===============
Tenant visit order is tenant-id ascending, jobs are FIFO per tenant, job ids
are sequential, and ``run_batch`` accepts an injectable ``now``/seed-free
logic: identical inputs always produce identical selection order.

Batch preview
=============
``preview_batch`` runs the *same* transaction body as ``run_batch`` — WDRR,
aging settlement, tenant caps and dependency readiness are one code path, not
two copies — but rolls the transaction back instead of committing. A preview
therefore writes nothing: no batch row, no ledger entries, no job status,
balance or aging-clock changes, even on error. The only deliberate
difference in the returned snapshot is ``preview: true`` (and the rollback
itself). Because nothing is persisted, a preview followed by a real batch at
the same reference instant on the same database state yields the same
selection order, GPU usage, balances and credit events, and even reuses the
same prospective batch/ledger row ids.

Job dependencies
================
``POST /jobs`` accepts ``depends_on``: a list of job ids (from ANY tenant)
that must be COMPLETED before the new job becomes schedulable. Unknown ids
and dependency cycles (including self-dependence) are rejected at submit.
Submission therefore never closes a cycle, but the whole graph is rechecked
defensively. During a batch, a tenant is visited only when its head-of-line
job's prerequisites are all COMPLETED; a dependency-blocked head receives
neither quantum nor aging credit (its aging checkpoint is merely advanced),
so blocked dependencies can never be farmed into spendable balance. When
the blockage lifts — the last prerequisite completes, or a revision
detaches the job — aging credit accrues only from that readiness instant
(see "Aging compensation"), never retrospectively from submission. A
CANCELLED prerequisite can never complete: dependents stay QUEUED forever
and ``blocked_by`` (on the job object and in ``GET /queue``) reports the
prerequisite id together with its current status.

Batch graph submission
======================
``POST /jobs/graph`` atomically submits a non-empty *batch* of jobs in one
request. Every item carries a batch-unique ``ref`` (client-side label), its
``tenant_id``, ``gpus``, ``duration_min`` and ``weight``, and two
prerequisite lists: ``depends_on`` names already-existing job ids while
``depends_on_refs`` names other items' ``ref`` values inside the same batch.
Forward references (an item depending on an item appearing later in the
array) are allowed, so the items need not be topologically sorted, and
external prerequisites may belong to any tenant — dependencies are
cluster-global exactly as for single submissions. All items belonging to the
same tenant must agree on ``weight`` (a mismatch rejects the whole batch);
the agreed weight then creates/updates that tenant once.

Every validation step runs inside the same process-wide lock and
``BEGIN IMMEDIATE`` write transaction as a single submission: parameter
shapes, duplicate refs, per-tenant weight agreement, each item's GPU request
against cluster capacity, the existence of every external job id (a ref that
matches no item and no existing job is "unknown"), and a cycle check over
the *whole* would-be graph — existing edges plus every new edge — so the
batch can never close a cycle (a item naming its own preallocated id or its
own ref is a self-dependence and rejected as such). Only after every check
passes are the sequential job ids allocated (in array order), tenants
upserted, jobs inserted (QUEUED) and dependency edges written; any error
rolls the entire batch back: no tenants, weights, jobs, edges or queue
positions change. On success the response maps each ``ref`` to its created
job id and returns the created job objects (array order). The new jobs are
ordinary QUEUED jobs immediately: the very next preview or committed batch
judges their dependencies through the exact same readiness code path as
single-submitted jobs.

Dependency revision
===================
``PATCH /jobs/{id}/dependencies`` rewrites a QUEUED job's prerequisite set:
the request's ``depends_on`` list atomically REPLACES every existing edge
(an empty list detaches the job entirely). Ids are deduplicated and sorted
exactly as at submission, prerequisites may belong to any tenant, and the
same rejections apply — unknown ids, self-dependence and any cycle the
replacement would close — validated against the would-be graph in which the
job's old edges are substituted (not added to), so a rejected revision
leaves the stored edges untouched. Only QUEUED jobs can be revised: once
dispatched (or finished) a job's prerequisites are historical fact. The
revision runs under the same process-wide lock and ``BEGIN IMMEDIATE``
write-transaction style as batch scheduling, so it is atomic with respect
to any concurrent batch/preview, and the next preview or committed batch
already judges the new edges. The job keeps its ``queued_at`` queue
position and no tenant balance/aging state is touched; the job's own
readiness clock (``ready_since``) is the one thing updated — set to the
revision instant when the revision releases a previously blocked job,
cleared when a revision (re)blocks the job, preserved when a ready job
stays ready — so aging credit accrues from the true readiness moment (see
"Aging compensation").

Dependency cancel cascade
=========================
``GET /jobs/{id}/cancel-impact`` previews, for a QUEUED or RUNNING job, the
queued descendants that cancelling it would permanently block: the transitive
closure over the cluster-global dependency graph (cross-tenant), traversing
through CANCELLED nodes — a cancelled prerequisite can never complete, so the
blockage propagates past it — but never through COMPLETED ones (a satisfied
prerequisite blocks nobody, so the closure stops there). Every affected
QUEUED job is reported, sorted by job id, together with one shortest
dependency chain from the target (each job on the chain depends on the
previous one). The preview is a pure read and writes nothing.

``POST /jobs/{id}/cancel-cascade`` recomputes the same closure inside the
shared process-wide write lock and ``BEGIN IMMEDIATE`` transaction used by
single-job cancellation and dependency revision, then atomically cancels the
target and every still-QUEUED job in the closure (a RUNNING target's GPUs
are released for subsequent batches). Already-CANCELLED intermediate nodes
are traversal-only and are never re-modified. Tenants whose queue drained
get their aging clock reset, exactly as with a single cancellation. An
unknown target gets 404, a COMPLETED/CANCELLED one 409 — on both endpoints.

Idempotent submission
=====================
``POST /jobs`` accepts an optional ``Idempotency-Key`` request header. A
missing or blank header keeps the original, non-idempotent interface. A
non-empty key is scoped to the request body's ``tenant_id``: the pair
``(tenant_id, key)`` identifies exactly one logical submission.

The first successful request commits, in one existing SQLite transaction,
the tenant auto-register/weight upsert, the job, its dependency edges and an
``idempotent_jobs`` row holding the normalised fingerprint — ``gpus``,
``duration_min``, ``weight`` and the deduplicated, sorted ``depends_on`` —
plus the resulting job id. On a retry the stored record is looked up
**inside the write transaction, before any admission decision or write**, so
the ordering is always:

1. the ``(tenant_id, key)`` pair already has a saved record -> compare the
   saved fingerprint with the request first. An equal fingerprint resolves
   to that record and returns the job's **current** object (the job may since
   have been dispatched, completed or cancelled; HTTP 200) **without any
   capacity re-validation** — even when ``CLUSTER_GPUS`` was lowered after the
   original submission so the saved request now exceeds the present capacity.
   The retry never updates the tenant weight, ``updated_at``, the job's
   ``queued_at`` or its dependencies; a differing fingerprint is rejected
   with 409 **regardless of current capacity** (including the case where the
   new parameters themselves exceed it).
2. no saved record (first submission with the key) -> only then is the
   request admitted against the cluster's *current* capacity and, if it fits,
   the tenant/job/edges/fingerprint are inserted. Any failure rolls the whole
   transaction back, so a failed first attempt does not consume the key and a
   corrected retry still creates the job.

Failures detected before the transaction opens (basic parameter shape)
likewise never touch the record. A submission **without** a key keeps the
original interface and is capacity-checked by the same pre-transaction rule
as always.

A later dependency revision (``PATCH /jobs/{id}/dependencies``) never
rewrites the stored fingerprint: a replay of the original submission still
matches it and returns the job's *current* — possibly revised — object,
without restoring the original edges.

Concurrent first-submissions are serialised by the process-wide write lock
and ``BEGIN IMMEDIATE``; for several processes/containers sharing the SQLite
file, the ``PRIMARY KEY (tenant_id, idempotency_key)`` constraint plus
``busy_timeout`` make the loser resolve as a replay of the winner — exactly
one job is ever created per key. Because the record lives in the same
persistent database file, a client or container restart followed by a retry
(even one that redeploys a smaller ``CLUSTER_GPUS``) still returns the single
already-created job.

GPU maintenance windows
=======================
A maintenance window is a persistent, half-open UTC interval
``[start_at, end_at)`` with a number of GPUs reserved for maintenance
(``POST /maintenance-windows``; listed via ``GET /maintenance-windows``).
Inputs with ``end_at <= start_at`` or a reservation outside
``0..cluster_gpus`` are rejected; RUNNING jobs are never preempted by a
newly created window.

Creation is also admitted cumulatively: inside the write transaction the
candidate window is overlaid on the already-persisted ones and the peak
combined reservation over the candidate's own half-open coverage is
computed with the same sweep the scheduler uses at dispatch time
(overlapping windows add, endpoint-touching ones do not). A peak
exceeding the cluster's *current* capacity rejects the creation and
nothing is persisted; a peak exactly equal to the capacity is
admissible.

A batch treats its reference instant (``now``) as every selected job's
estimated start; with the job's existing ``duration_min`` its predicted
run is the half-open interval ``[now, now + duration_min)``. Before each
candidate dispatch the scheduler finds the largest (peak) combined
reservation of the maintenance windows overlapping that predicted
interval — reservations of windows that overlap at the same instant add,
while windows that merely meet at an endpoint do not overlap — and admits
the job only when

    RUNNING GPUs + GPUs selected in this batch + candidate GPUs
        <= cluster_gpus - peak reservation over the candidate interval.

A short job that ends before a window starts can therefore dispatch
while a longer candidate overlapping the window must wait. The exact same
check runs in a real batch and in a preview (one code path), so identical
``now`` and state give identical selection; only the committed batch
writes anything.

Maintenance-slot queries
========================
``job_maintenance_slots`` (``GET /jobs/{id}/maintenance-slots``) answers,
for a QUEUED job only, *when it could start* inside a caller-supplied
half-open UTC range ``[start, end)``. The GPUs held by RUNNING jobs at
query time are taken as a FIXED baseline for the whole horizon — no future
releases are speculated — and the saved maintenance windows are overlaid
on the job's entire predicted run ``[t, t + duration_min)``: a candidate
start ``t`` is feasible iff at every instant of that run

    RUNNING baseline + job GPUs + combined reservation <= cluster_gpus.

The feasible start instants are returned as merged slots with explicit
inclusive/exclusive markers at both ends (the query range is half-open, so
a final slot always ends exclusive; blocked regions are open intervals, so
a slot squeezed between two of them can degenerate to a single isolated
instant, reported with both ends inclusive), together with the earliest
feasible instant (``null`` when no slot exists). The result is purely a
resource-capacity view: tenant caps, budgets, balances, dependencies and
other queued jobs are not consulted.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from . import db
from .config import Settings, get_settings

QUEUED, RUNNING, COMPLETED, CANCELLED = "QUEUED", "RUNNING", "COMPLETED", "CANCELLED"

# Float comparison tolerance for the affordability test.
_EPS = 1e-9


class SchedulerError(Exception):
    """Base class for expected, HTTP-mappable service errors."""

    status_code = 400


class NotFound(SchedulerError):
    status_code = 404


class Conflict(SchedulerError):
    status_code = 409


def _now_iso(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _as_utc(value: datetime) -> datetime:
    """Normalise a datetime to timezone-aware UTC; naive inputs are treated
    as UTC (ISO-8601 timestamps without an offset are conventionally UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _job_id(conn: sqlite3.Connection) -> str:
    # Deterministic, human-readable sequential ids (J000001, ...).
    row = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()
    return f"J{row['n'] + 1:06d}"


def _credit(
    conn: sqlite3.Connection,
    tenant_id: str,
    amount: float,
    kind: str,
    created_at: str,
    batch_id: int | None = None,
    job_id: str | None = None,
    detail: str | None = None,
) -> float:
    """Apply a signed ledger entry and return the resulting balance."""
    conn.execute(
        "UPDATE tenants SET deficit = deficit + ?, updated_at = ? WHERE id = ?",
        (amount, created_at, tenant_id),
    )
    row = conn.execute(
        "SELECT deficit FROM tenants WHERE id = ?", (tenant_id,)
    ).fetchone()
    conn.execute(
        """INSERT INTO ledger
               (tenant_id, batch_id, job_id, kind, amount, balance_after,
                detail, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (tenant_id, batch_id, job_id, kind, amount, row["deficit"],
         detail, created_at),
    )
    return row["deficit"]


def _has_queued(conn: sqlite3.Connection, tenant_id: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE tenant_id = ? AND status = ?",
        (tenant_id, QUEUED),
    ).fetchone()
    return row["n"] > 0


# ---------------------------------------------------------------------------
# Job dependencies (DAG)
# ---------------------------------------------------------------------------

def _depends_on(conn: sqlite3.Connection, job_id: str) -> list[str]:
    """Prerequisite job ids in submission order (id ascending)."""
    rows = conn.execute(
        """SELECT depends_on FROM job_dependencies
            WHERE job_id = ? ORDER BY depends_on""",
        (job_id,),
    ).fetchall()
    return [r["depends_on"] for r in rows]


def _unmet_prerequisites(
    conn: sqlite3.Connection, job_id: str
) -> list[dict[str, str]]:
    """Prerequisites that have not COMPLETED, with the reason (their status).

    QUEUED/RUNNING prerequisites may still release the job; a CANCELLED one
    can never complete, so the dependent stays queued forever and the entry
    here explains why.
    """
    rows = conn.execute(
        """SELECT j.id AS id, j.status AS status
             FROM job_dependencies d
             JOIN jobs j ON j.id = d.depends_on
            WHERE d.job_id = ? AND j.status != ?
            ORDER BY j.id""",
        (job_id, COMPLETED),
    ).fetchall()
    return [{"id": r["id"], "status": r["status"]} for r in rows]


def _head_ready_since(
    conn: sqlite3.Connection, head: sqlite3.Row
) -> datetime:
    """Instant the (dependency-ready) head job's wait became scheduler-facing.

    Waiting on prerequisites is not scheduler starvation, so aging credit
    must not reach back past the moment the head job truly became
    schedulable. When the job carries a ``ready_since`` stamp — written by
    ``revise_dependencies`` at the exact transition that produced the
    current ready state — that stamp IS the readiness moment. Otherwise it
    is derived from the job's arrival (``queued_at``) and its
    prerequisites: with several prerequisites the LAST completion releases
    the job, so the newest ``completed_at`` governs. (A timestamp later
    than the batch reference instant simply clamps the computed wait to
    zero at the call site.)
    """
    queued = _as_utc(_parse_iso(head["queued_at"]))
    revised = head["ready_since"]
    if revised is not None:
        # Definitive stamp from the revision that (re)established the
        # ready state; never earlier than the job's own arrival.
        return max(_as_utc(_parse_iso(revised)), queued)
    ready = queued
    row = conn.execute(
        """SELECT MAX(j.completed_at) AS done
             FROM job_dependencies d
             JOIN jobs j ON j.id = d.depends_on
            WHERE d.job_id = ?""",
        (head["id"],),
    ).fetchone()
    if row["done"] is not None:
        done = _as_utc(_parse_iso(row["done"]))
        if done > ready:
            ready = done
    return ready


def _find_cycle(
    conn: sqlite3.Connection,
    extra_edges: dict[str, list[str]] | None = None,
    replace_edges: dict[str, list[str]] | None = None,
) -> list[str] | None:
    """Return a job-id cycle in the dependency graph, or None when it is a DAG.

    ``extra_edges`` maps a (possibly not-yet-inserted) job id to its
    prerequisites, so a candidate submission can be validated against the
    graph it would create. ``replace_edges`` maps an existing job id to the
    prerequisite set that would *replace* its current edges, so a dependency
    revision is validated against the graph the replacement would produce
    (the job's stored edges are substituted, not added to). Iterative DFS
    with WHITE/GRAY/BLACK coloring.
    """
    extra_edges = extra_edges or {}
    replace_edges = replace_edges or {}
    graph: dict[str, list[str]] = {}
    rows = conn.execute(
        "SELECT job_id, depends_on FROM job_dependencies"
    ).fetchall()
    for r in rows:
        graph.setdefault(r["job_id"], []).append(r["depends_on"])
        graph.setdefault(r["depends_on"], [])
    for node, prereqs in replace_edges.items():
        graph[node] = list(prereqs)
        for p in prereqs:
            graph.setdefault(p, [])
    for node, prereqs in extra_edges.items():
        graph.setdefault(node, []).extend(prereqs)
        for p in prereqs:
            graph.setdefault(p, [])

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {node: WHITE for node in graph}
    for start in sorted(graph):
        if color.get(start, BLACK) != WHITE:
            continue
        # Stack of (node, next-neighbour-index); parallel pos[] stack records
        # the path so the GRAY edge closing a cycle names the exact nodes.
        stack: list[str] = [start]
        pos: list[int] = [0]
        color[start] = GRAY
        while stack:
            node = stack[-1]
            neighbours = graph[node]
            i = pos[-1]
            if i < len(neighbours):
                nxt = neighbours[i]
                pos[-1] = i + 1
                if color.get(nxt, WHITE) == GRAY:
                    cycle = stack[stack.index(nxt):] + [nxt]
                    return cycle
                if color.get(nxt, WHITE) == WHITE:
                    color[nxt] = GRAY
                    stack.append(nxt)
                    pos.append(0)
            else:
                color[node] = BLACK
                stack.pop()
                pos.pop()
    return None


# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------

# Sentinel for upsert_tenant: no max_gpus argument was supplied, so an
# existing tenant keeps its current cap (and a new tenant gets unlimited).
UNSET = object()


def upsert_tenant(
    tenant_id: str,
    weight: float,
    max_gpus: int | None | object = UNSET,
    budget_gpu_min: int | None | object = UNSET,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Create a tenant or update its weight (and optionally its cap/budget).

    Existing balances are retained. ``max_gpus`` accepts a positive integer to
    set the concurrent-GPU cap, ``None`` to clear it, or may be omitted to
    leave an existing tenant's cap untouched. ``budget_gpu_min`` follows the
    exact same convention for the rolling 24-hour GPU-minute budget.
    """
    if not tenant_id or not tenant_id.strip():
        raise SchedulerError("tenant_id must be a non-empty string")
    if weight <= 0:
        raise SchedulerError("weight must be > 0")
    if max_gpus is not UNSET:
        # bool is a subclass of int and must not be accepted.
        if max_gpus is not None and (
            not isinstance(max_gpus, int)
            or isinstance(max_gpus, bool)
            or max_gpus <= 0
        ):
            raise SchedulerError("max_gpus must be a positive integer or null")
    if budget_gpu_min is not UNSET:
        if budget_gpu_min is not None and (
            not isinstance(budget_gpu_min, int)
            or isinstance(budget_gpu_min, bool)
            or budget_gpu_min <= 0
        ):
            raise SchedulerError(
                "budget_gpu_min must be a positive integer or null"
            )
    settings = settings or get_settings()
    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT id FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone()
        if existing is None:
            conn.execute(
                """INSERT INTO tenants
                       (id, weight, deficit, max_gpus, budget_gpu_min,
                        aging_settled_at, created_at, updated_at)
                   VALUES (?, ?, 0, ?, ?, NULL, ?, ?)""",
                (
                    tenant_id,
                    float(weight),
                    (None if max_gpus is UNSET else max_gpus),
                    (None if budget_gpu_min is UNSET else budget_gpu_min),
                    now,
                    now,
                ),
            )
        else:
            conn.execute(
                "UPDATE tenants SET weight = ?, updated_at = ? WHERE id = ?",
                (float(weight), now, tenant_id),
            )
            if max_gpus is not UNSET:
                conn.execute(
                    "UPDATE tenants SET max_gpus = ? WHERE id = ?",
                    (max_gpus, tenant_id),
                )
            if budget_gpu_min is not UNSET:
                conn.execute(
                    "UPDATE tenants SET budget_gpu_min = ? WHERE id = ?",
                    (budget_gpu_min, tenant_id),
                )
        return get_tenant(tenant_id)


def get_tenant(tenant_id: str) -> dict[str, Any]:
    with db.get_lock():
        conn = db.get_conn()
        row = conn.execute(
            _TENANT_SELECT + " HAVING t.id = ?", (tenant_id,)
        ).fetchone()
        if row is None:
            raise NotFound(f"tenant '{tenant_id}' not found")
        return _tenant_dict(conn, row)


_TENANT_SELECT = """
    SELECT t.id, t.weight, t.deficit, t.max_gpus, t.budget_gpu_min,
           t.aging_settled_at,
           SUM(CASE WHEN j.status = 'QUEUED'   THEN 1 ELSE 0 END) AS queued,
           SUM(CASE WHEN j.status = 'RUNNING'  THEN 1 ELSE 0 END) AS running,
           COALESCE(SUM(CASE WHEN j.status = 'RUNNING'
                             THEN j.gpus ELSE 0 END), 0) AS running_gpus
      FROM tenants t
      LEFT JOIN jobs j ON j.tenant_id = t.id
     GROUP BY t.id
"""

# Half-open-on-the-left rolling window: strictly later than window_start
# (now - 24h) and at or before window_end (now). Only jobs that were actually
# dispatched have a scheduled_at; each is charged its full estimated cost
# exactly once, and completion/cancellation never refunds it, so the charge
# survives whatever the job's later status is.
_BUDGET_SPENT_SELECT = """
    SELECT tenant_id,
           COALESCE(SUM(gpus * duration_min), 0) AS spent
      FROM jobs
     WHERE scheduled_at IS NOT NULL
       AND scheduled_at > ?
       AND scheduled_at <= ?
       AND tenant_id IS ?
     GROUP BY tenant_id
"""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _budget_window_end(window_end: datetime | None) -> datetime:
    """Default a budget query's window end to the server's current UTC instant."""
    return _as_utc(window_end) if window_end is not None else _now_utc()


def _budget_spent_map(
    conn: sqlite3.Connection,
    window_end: datetime,
) -> dict[str, int]:
    """GPU-minutes charged per tenant over ``(window_end - 24h, window_end]``.

    Charges are ``gpus * duration_min`` of dispatched jobs (any non-QUEUED
    status: RUNNING, COMPLETED, or CANCELLED after dispatch) keyed by
    ``scheduled_at``. Timestamps are normalised UTC ISO-8601, so the
    half-open window compares lexicographically.
    """
    end = _as_utc(window_end)
    start = end - timedelta(hours=24)
    rows = conn.execute(
        _BUDGET_SPENT_SELECT.replace(" AND tenant_id IS ?", ""),
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    return {r["tenant_id"]: int(r["spent"]) for r in rows}


def _budget_spent_for(
    conn: sqlite3.Connection,
    tenant_id: str,
    window_end: datetime,
) -> int:
    end = _as_utc(window_end)
    start = end - timedelta(hours=24)
    row = conn.execute(
        _BUDGET_SPENT_SELECT,
        (start.isoformat(), end.isoformat(), tenant_id),
    ).fetchone()
    return int(row["spent"]) if row is not None else 0


def _tenant_dict(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    window_end: datetime | None = None,
    spent: int | None = None,
) -> dict[str, Any]:
    # Budget queries default to the server's current UTC instant; batch
    # snapshots pass the batch reference instant so the snapshot is
    # self-consistent with its selection decision.
    end = _budget_window_end(window_end)
    if spent is None:
        spent = _budget_spent_for(conn, row["id"], end)
    budget = row["budget_gpu_min"]
    return {
        "id": row["id"],
        "weight": round(row["weight"], 6),
        "balance_gpu_min": round(row["deficit"], 6),
        # Optional concurrent-GPU cap; None means unlimited.
        "max_gpus": row["max_gpus"],
        # Optional rolling 24h GPU-minute budget cap; None means unlimited.
        "budget_gpu_min": budget,
        # Cost already charged for dispatches in (now-24h, now].
        "budget_used_gpu_min": spent,
        # Headroom left in the same window; null when the tenant is uncapped.
        "budget_remaining_gpu_min": (
            None if budget is None else max(budget - spent, 0)
        ),
        # Window the two figures above were evaluated against.
        "budget_window_start_at": (end - timedelta(hours=24)).isoformat(),
        "budget_window_end_at": end.isoformat(),
        "queued_jobs": row["queued"],
        "running_jobs": row["running"],
        # GPUs currently held by the tenant's RUNNING jobs (cap usage).
        "running_gpus": row["running_gpus"],
        "aging_settled_at": row["aging_settled_at"],
    }


def _list_tenants(
    conn: sqlite3.Connection,
    window_end: datetime | None = None,
) -> list[dict[str, Any]]:
    end = _budget_window_end(window_end)
    rows = conn.execute(_TENANT_SELECT + " ORDER BY t.id").fetchall()
    spent_map = _budget_spent_map(conn, end)
    return [_tenant_dict(conn, r, end, spent_map.get(r["id"], 0))
            for r in rows]


def list_tenants() -> list[dict[str, Any]]:
    with db.get_lock():
        return _list_tenants(db.get_conn())


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

def submit_job(
    tenant_id: str,
    gpus: int,
    duration_min: int,
    weight: float = 1.0,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    """Submit without an idempotency key (original interface)."""
    return _submit_job(
        tenant_id, gpus, duration_min, weight, depends_on,
        idempotency_key=None,
    )[0]


def submit_job_idempotent(
    tenant_id: str,
    gpus: int,
    duration_min: int,
    weight: float,
    depends_on: list[str],
    idempotency_key: str,
) -> tuple[dict[str, Any], bool]:
    """Submit a job identified by ``(tenant_id, idempotency_key)``.

    Returns ``(job, created)``: ``created=True`` for the first successful
    submission (HTTP 201) and ``created=False`` for a replay of the same
    parameters (HTTP 200).

    Semantics (see the module-level "Idempotent submission" section):

    * Inside the write transaction the saved fingerprint is compared FIRST:
      an existing (tenant, key) record with the same fingerprint returns the
      job's *current* object (``created=False``) with no capacity
      re-validation — this still holds after a restart whose ``CLUSTER_GPUS``
      was lowered below the saved request — while a record whose fingerprint
      differs raises :class:`Conflict` (409) even if the new parameters
      exceed the current capacity.
    * Only an unmatched first submission is admitted against the cluster's
      current capacity, after which the fingerprint (gpus, duration_min,
      weight, deduplicated+sorted depends_on) is stored in the same
      transaction as the job, the tenant upsert and the dependency edges; a
      replay then never re-touches tenant weight, queue time or dependencies.
    * A failed first submission (unknown prerequisite, cycle, capacity
      overflow, etc.) rolls the whole transaction back, so the key is not
      consumed and a later corrected request still creates the job.
    * Concurrency is serialised by the process lock plus a
      ``BEGIN IMMEDIATE`` write transaction; with multiple containers sharing
      the SQLite file the UNIQUE primary key plus ``busy_timeout`` provide the
      same guarantee: exactly one job is ever created per key, also across
      container restarts (the record is persisted).
    """
    return _submit_job(
        tenant_id, gpus, duration_min, weight, depends_on,
        idempotency_key=idempotency_key,
    )


def _submit_job(
    tenant_id: str,
    gpus: int,
    duration_min: int,
    weight: float = 1.0,
    depends_on: list[str] | None = None,
    idempotency_key: str | None = None,
) -> tuple[dict[str, Any], bool]:
    if not tenant_id or not tenant_id.strip():
        raise SchedulerError("tenant_id must be a non-empty string")
    if not isinstance(gpus, int) or gpus <= 0:
        raise SchedulerError("gpus must be a positive integer")
    if not isinstance(duration_min, int) or duration_min <= 0:
        raise SchedulerError("duration_min must be a positive integer")
    if weight <= 0:
        raise SchedulerError("weight must be > 0")
    if depends_on is None:
        depends_on = []
    if not isinstance(depends_on, list) or any(
        not isinstance(d, str) or not d.strip() for d in depends_on
    ):
        raise SchedulerError(
            "depends_on must be a list of non-empty job id strings"
        )
    # The stored fingerprint uses the same normalised form as the job itself:
    # prerequisite ids deduplicated and sorted.
    depends_on = sorted(set(depends_on))

    settings = get_settings()
    # Keyed submissions postpone the capacity gate until the transaction has
    # checked for a stored fingerprint: a replay (same params) must return the
    # job's current state even when CLUSTER_GPUS was lowered after the job was
    # created, and a key reuse with different params must 409 even when the new
    # params themselves exceed the current capacity. Only an unmatched first
    # submission is judged against current capacity; a keyless request keeps
    # the original pre-transaction validation.
    if idempotency_key is None and gpus > settings.cluster_gpus:
        raise SchedulerError(
            f"job requests {gpus} GPUs which exceeds cluster capacity "
            f"{settings.cluster_gpus}; it could never run"
        )

    now = _now_iso()
    cost = float(gpus * duration_min)
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")

        # Idempotent replay check: an existing record for (tenant, key) either
        # resolves to the already-created job or proves the parameters
        # diverged. This precedes every write AND every admission decision,
        # so a replay updates neither the tenant's weight/updated_at nor any
        # queue timestamp, and is never re-validated against a capacity that
        # may have shrunk since the original submission.
        if idempotency_key is not None:
            existing = conn.execute(
                "SELECT * FROM idempotent_jobs "
                "WHERE tenant_id = ? AND idempotency_key = ?",
                (tenant_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                return _idempotent_replay(conn, existing, tenant_id, gpus,
                                          duration_min, weight, depends_on)

            # First submission with this key: judge the request against the
            # cluster's CURRENT capacity, inside the transaction. Failure here
            # (and any later failure) rolls the whole transaction back, so the
            # key stays unconsumed.
            if gpus > settings.cluster_gpus:
                raise SchedulerError(
                    f"job requests {gpus} GPUs which exceeds cluster capacity "
                    f"{settings.cluster_gpus}; it could never run"
                )

        # Auto-register unknown tenants so a single POST /jobs is enough.
        if conn.execute(
            "SELECT 1 FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone() is None:
            conn.execute(
                """INSERT INTO tenants
                       (id, weight, deficit, max_gpus, aging_settled_at,
                        created_at, updated_at)
                   VALUES (?, ?, 0, NULL, NULL, ?, ?)""",
                (tenant_id, float(weight), now, now),
            )
        else:
            # Latest submission may adjust the tenant's weight.
            conn.execute(
                "UPDATE tenants SET weight = ?, updated_at = ? WHERE id = ?",
                (float(weight), now, tenant_id),
            )

        job_id = _job_id(conn)

        # Reject unknown prerequisites. Prerequisites may belong to other
        # tenants: dependencies are cluster-global, not tenant-scoped.
        if depends_on:
            placeholders = ",".join("?" for _ in depends_on)
            found = {
                r["id"] for r in conn.execute(
                    f"SELECT id FROM jobs WHERE id IN ({placeholders})",
                    depends_on,
                ).fetchall()
            }
            missing = [
                d for d in depends_on
                if d not in found and d != job_id
            ]
            if missing:
                raise SchedulerError(
                    f"unknown depends_on job id(s): {', '.join(missing)}"
                )

        # Reject anything that would close a cycle (including self
        # dependencies such as depends_on == the new job's own id).
        cycle = _find_cycle(conn, {job_id: depends_on})
        if cycle is not None:
            raise SchedulerError(
                "dependency cycle detected: " + " -> ".join(cycle)
            )

        conn.execute(
            """INSERT INTO jobs
                   (id, tenant_id, gpus, duration_min, cost_gpu_min, status,
                    queued_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (job_id, tenant_id, gpus, duration_min, cost, QUEUED, now),
        )
        for prereq in depends_on:
            conn.execute(
                "INSERT INTO job_dependencies (job_id, depends_on) VALUES (?, ?)",
                (job_id, prereq),
            )
        if idempotency_key is not None:
            # Written in the same transaction as the job and the tenant
            # upsert: on commit the key is consumed once, on any failure the
            # whole transaction rolls back and the key stays free. The
            # PRIMARY KEY (tenant_id, idempotency_key) is the cross-process
            # backstop for concurrent first-submissions that were not
            # serialised by the in-process lock.
            try:
                conn.execute(
                    """INSERT INTO idempotent_jobs
                           (tenant_id, idempotency_key, job_id, gpus,
                            duration_min, weight, depends_on, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (tenant_id, idempotency_key, job_id, gpus,
                     duration_min, float(weight),
                     json.dumps(depends_on), now),
                )
            except sqlite3.IntegrityError:
                # Another process/container committed the same key between
                # the replay SELECT above and this INSERT. Roll back our
                # tentative job/tenant writes and resolve as a replay.
                conn.rollback()
                conn.execute("BEGIN IMMEDIATE")
                winner = conn.execute(
                    "SELECT * FROM idempotent_jobs "
                    "WHERE tenant_id = ? AND idempotency_key = ?",
                    (tenant_id, idempotency_key),
                ).fetchone()
                return _idempotent_replay(
                    conn, winner, tenant_id, gpus, duration_min, weight,
                    depends_on,
                )
        # aging_settled_at is left untouched: a non-empty queue keeps accruing
        # wait time, and a drained queue already has it reset to NULL.
        return _get_job_or_404(conn, job_id), True


def _idempotent_replay(
    conn: sqlite3.Connection,
    record: sqlite3.Row,
    tenant_id: str,
    gpus: int,
    duration_min: int,
    weight: float,
    depends_on: list[str],
) -> tuple[dict[str, Any], bool]:
    """Resolve a request whose (tenant, key) was already committed.

    Same fingerprint -> the job's current object (it may since have been
    dispatched, completed or cancelled) with ``created=False``. Different
    fingerprint -> 409. No tenant weight, queue-time or dependency writes
    happen on either branch.
    """
    same = (
        record["gpus"] == gpus
        and record["duration_min"] == duration_min
        and record["weight"] == float(weight)
        and json.loads(record["depends_on"]) == depends_on
    )
    if not same:
        raise Conflict(
            f"Idempotency-Key '{record['idempotency_key']}' for tenant "
            f"'{tenant_id}' was already used with different parameters; "
            f"existing job: {record['job_id']}"
        )
    return _get_job_or_404(conn, record["job_id"]), False


def submit_job_graph(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Atomically submit a batch of jobs as one dependency graph.

    Each item is a dict with ``ref`` (unique non-empty label within the
    batch), ``tenant_id``, ``gpus``, ``duration_min``, ``weight``,
    ``depends_on`` (existing job ids, any tenant) and ``depends_on_refs``
    (refs of other items in the same batch). See the module-level
    "Batch graph submission" section for the full contract: every validation
    (shapes, unique refs, per-tenant weight agreement, capacity, external
    references, whole-graph cycles) runs in one ``BEGIN IMMEDIATE``
    transaction; any failure rolls the whole batch back without touching
    tenant weights or queues. Success allocates sequential ids in array
    order, upserts each tenant at the batch-agreed weight, inserts the jobs
    and every (deduplicated, external + in-batch) edge, and returns
    ``{"ref_ids": {ref: job_id}, "jobs": [job, ...]}`` in array order.
    """
    # ---- Basic shape validation (mirrors a single JobIn/Job submission) ----
    if not isinstance(items, list) or len(items) == 0:
        raise SchedulerError("jobs must be a non-empty array")

    norm: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise SchedulerError(f"jobs[{i}] must be an object")
        ref = item.get("ref")
        tenant_id = item.get("tenant_id")
        gpus = item.get("gpus")
        duration_min = item.get("duration_min")
        weight = item.get("weight", 1.0)
        depends_on = item.get("depends_on", []) or []
        depends_on_refs = item.get("depends_on_refs", []) or []
        label = f"jobs[{i}]" + (f" (ref '{ref}')" if isinstance(ref, str) else "")

        if not isinstance(ref, str) or not ref.strip():
            raise SchedulerError(f"{label}: ref must be a non-empty string")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise SchedulerError(f"{label}: tenant_id must be a non-empty string")
        if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus <= 0:
            raise SchedulerError(f"{label}: gpus must be a positive integer")
        if (
            not isinstance(duration_min, int)
            or isinstance(duration_min, bool)
            or duration_min <= 0
        ):
            raise SchedulerError(
                f"{label}: duration_min must be a positive integer"
            )
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) \
                or weight <= 0:
            raise SchedulerError(f"{label}: weight must be > 0")
        for field_name, values in (
            ("depends_on", depends_on),
            ("depends_on_refs", depends_on_refs),
        ):
            if not isinstance(values, list) or any(
                not isinstance(d, str) or not d.strip() for d in values
            ):
                raise SchedulerError(
                    f"{label}: {field_name} entries must be non-empty strings"
                )
        norm.append({
            "ref": ref,
            "tenant_id": tenant_id,
            "gpus": gpus,
            "duration_min": duration_min,
            "weight": float(weight),
            "depends_on": list(depends_on),
            "depends_on_refs": list(depends_on_refs),
        })

    settings = get_settings()
    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = _submit_job_graph_tx(conn, norm, settings, now)
        except BaseException:
            # One atomic batch: a failed validation must leave neither
            # tenant weights nor queue state behind. (With
            # isolation_level=None the connection context manager does not
            # roll back on its own.)
            conn.rollback()
            raise
        conn.commit()
        return result


def _submit_job_graph_tx(
    conn: sqlite3.Connection,
    norm: list[dict[str, Any]],
    settings: Settings,
    now: str,
) -> dict[str, Any]:
    """Transaction body of :func:`submit_job_graph`; caller owns commit."""
    # Refs must be unique within the batch.
    seen_refs: set[str] = set()
    for item in norm:
        if item["ref"] in seen_refs:
            raise SchedulerError(
                f"duplicate ref '{item['ref']}' in jobs graph; each ref "
                "must be unique within the batch"
            )
        seen_refs.add(item["ref"])

    # Every item for the same tenant must agree on its weight; the agreed
    # value is the one tenant weight the batch will create/update to.
    weight_of: dict[str, float] = {}
    for item in norm:
        tid = item["tenant_id"]
        if tid in weight_of and weight_of[tid] != item["weight"]:
            raise SchedulerError(
                f"tenant '{tid}' has inconsistent weights in the same "
                f"graph: {weight_of[tid]} vs {item['weight']} "
                f"(ref '{item['ref']}')"
            )
        weight_of[tid] = item["weight"]

    # A job that could never run must never be queued, same as POST /jobs.
    for item in norm:
        if item["gpus"] > settings.cluster_gpus:
            raise SchedulerError(
                f"ref '{item['ref']}' requests {item['gpus']} GPUs which "
                f"exceeds cluster capacity {settings.cluster_gpus}; it "
                "could never run"
            )

    # Allocate the sequential ids up front so prerequisites can mix
    # in-batch refs with already-existing ids and the graph can be checked
    # before any row is written. Array order determines both id and FIFO
    # queue order (identical queued_at, id tie-break).
    existing_n = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs"
    ).fetchone()["n"]
    for index, item in enumerate(norm):
        item["job_id"] = f"J{existing_n + index + 1:06d}"
    id_by_ref = {item["ref"]: item["job_id"] for item in norm}

    # Resolve every prerequisite to a concrete job id. depends_on_refs
    # names in-batch items (forward/backward refs both allowed);
    # depends_on names ids that must already exist when the batch commits.
    prereqs_of: dict[str, list[str]] = {}
    for item in norm:
        resolved: list[str] = []
        for ref in item["depends_on_refs"]:
            if ref not in id_by_ref:
                raise SchedulerError(
                    f"ref '{item['ref']}' depends on unknown batch ref "
                    f"'{ref}'"
                )
            resolved.append(id_by_ref[ref])
        external = item["depends_on"]
        if external:
            placeholders = ",".join("?" for _ in external)
            found = {
                r["id"] for r in conn.execute(
                    f"SELECT id FROM jobs WHERE id IN ({placeholders})",
                    external,
                ).fetchall()
            }
            missing = [
                d for d in external
                if d not in found and d != item["job_id"]
            ]
            if missing:
                raise SchedulerError(
                    f"ref '{item['ref']}': unknown depends_on job id(s): "
                    + ", ".join(sorted(set(missing)))
                )
            resolved.extend(external)
        # Same normalised form as a single submission / revision:
        # deduplicated, sorted by job id.
        prereqs_of[item["job_id"]] = sorted(set(resolved))

    # Whole-graph cycle check: existing edges plus every new edge. The
    # existing graph is a DAG by construction; this also catches cycles
    # contained entirely inside the batch (including self-dependence via
    # own ref or own preallocated id) and cycles through existing jobs.
    cycle = _find_cycle(conn, extra_edges=prereqs_of)
    if cycle is not None:
        raise SchedulerError(
            "dependency cycle detected: " + " -> ".join(cycle)
        )

    # All checks passed: upsert each tenant once at the batch-agreed
    # weight. As with single submissions this never changes an existing
    # tenant's max_gpus/budget_gpu_min; it is also the only weight write,
    # so on rollback nothing about the tenant (or queue) changes.
    for tenant_id, weight in weight_of.items():
        if conn.execute(
            "SELECT 1 FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone() is None:
            conn.execute(
                """INSERT INTO tenants
                       (id, weight, deficit, max_gpus, aging_settled_at,
                        created_at, updated_at)
                   VALUES (?, ?, 0, NULL, NULL, ?, ?)""",
                (tenant_id, weight, now, now),
            )
        else:
            conn.execute(
                "UPDATE tenants SET weight = ?, updated_at = ? WHERE id = ?",
                (weight, now, tenant_id),
            )

    # Insert every job before any edge: forward refs mean a prerequisite
    # item may appear later in the array, and the FOREIGN KEY on
    # job_dependencies requires both endpoints to already exist.
    for item in norm:
        conn.execute(
            """INSERT INTO jobs
                   (id, tenant_id, gpus, duration_min, cost_gpu_min,
                    status, queued_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                item["job_id"], item["tenant_id"], item["gpus"],
                item["duration_min"],
                float(item["gpus"] * item["duration_min"]),
                QUEUED, now,
            ),
        )
    for item in norm:
        for prereq in prereqs_of[item["job_id"]]:
            conn.execute(
                "INSERT INTO job_dependencies (job_id, depends_on) "
                "VALUES (?, ?)",
                (item["job_id"], prereq),
            )
        # aging_settled_at is intentionally untouched, exactly like a
        # single submission: non-empty queues keep accruing wait time and
        # drained queues already carry NULL.

    jobs = [_get_job_or_404(conn, item["job_id"]) for item in norm]
    return {
        "ref_ids": {item["ref"]: item["job_id"] for item in norm},
        "jobs": jobs,
    }


def _get_job_or_404(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise NotFound(f"job '{job_id}' not found")
    return _job_dict(conn, row)


def _job_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "tenant_id": row["tenant_id"],
        "gpus": row["gpus"],
        "duration_min": row["duration_min"],
        "cost_gpu_min": round(row["cost_gpu_min"], 6),
        "status": row["status"],
        "depends_on": _depends_on(conn, row["id"]),
        # Present for QUEUED jobs: prerequisites that are not COMPLETED yet,
        # each annotated with its current status (QUEUED/RUNNING/CANCELLED).
        # Empty means every prerequisite is done and the job is schedulable.
        "blocked_by": (
            _unmet_prerequisites(conn, row["id"])
            if row["status"] == QUEUED else []
        ),
        "queued_at": row["queued_at"],
        "scheduled_at": row["scheduled_at"],
        "completed_at": row["completed_at"],
        "batch_id": row["batch_id"],
    }


def get_job(job_id: str) -> dict[str, Any]:
    with db.get_lock():
        return _get_job_or_404(db.get_conn(), job_id)


def cancel_job(job_id: str) -> dict[str, Any]:
    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"job '{job_id}' not found")
        if row["status"] in (COMPLETED, CANCELLED):
            raise Conflict(
                f"job '{job_id}' is already {row['status'].lower()}"
            )
        conn.execute(
            "UPDATE jobs SET status = ?, completed_at = ? WHERE id = ?",
            (CANCELLED, now, job_id),
        )
        if not _has_queued(conn, row["tenant_id"]):
            conn.execute(
                "UPDATE tenants SET aging_settled_at = NULL WHERE id = ?",
                (row["tenant_id"],),
            )
        return _get_job_or_404(conn, job_id)


def complete_job(job_id: str) -> dict[str, Any]:
    """Mark a RUNNING job COMPLETED, releasing its GPUs for future batches."""
    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"job '{job_id}' not found")
        if row["status"] != RUNNING:
            raise Conflict(
                f"job '{job_id}' is {row['status'].lower()}, only RUNNING "
                "jobs can complete"
            )
        conn.execute(
            "UPDATE jobs SET status = ?, completed_at = ? WHERE id = ?",
            (COMPLETED, now, job_id),
        )
        return _get_job_or_404(conn, job_id)


def revise_dependencies(
    job_id: str,
    depends_on: list[str] | None,
) -> dict[str, Any]:
    """Atomically replace a QUEUED job's prerequisite set.

    The supplied ``depends_on`` list substitutes every existing edge; an
    empty list detaches the job from all prerequisites. Ids are
    deduplicated and sorted exactly as at submission, and prerequisites may
    belong to any tenant (dependencies are cluster-global). Unknown ids,
    self-dependence and any cycle the replacement would close are rejected
    and leave the stored edges untouched (the whole revision is one write
    transaction that rolls back on error). Only QUEUED jobs can be revised:
    a dispatched or finished job's prerequisites are historical fact.

    Runs under the same process-wide lock and ``BEGIN IMMEDIATE``
    transaction style as batch scheduling, so the revision is atomic with
    respect to any concurrent batch/preview and the next preview or
    committed batch already judges the new edges. The job keeps its
    ``queued_at`` queue position and no tenant balance/aging state is
    touched; the job's own readiness clock (``ready_since``) is the one
    thing updated: a revision that releases a previously blocked job
    records the revision instant (aging then accrues from there, not from
    arrival), a revision that (re)blocks the job clears it, and a ready
    job that stays ready keeps its clock. The submission-time idempotency
    fingerprint (if any) is preserved — a replay of the original
    ``POST /jobs`` request returns this job's current object without
    restoring the original edges.
    """
    if depends_on is None:
        depends_on = []
    if not isinstance(depends_on, list) or any(
        not isinstance(d, str) or not d.strip() for d in depends_on
    ):
        raise SchedulerError(
            "depends_on must be a list of non-empty job id strings"
        )
    # The same normalised form as submission: deduplicated, sorted by id.
    depends_on = sorted(set(depends_on))

    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise NotFound(f"job '{job_id}' not found")
        if row["status"] != QUEUED:
            raise Conflict(
                f"job '{job_id}' is {row['status'].lower()}, only QUEUED "
                "jobs can have their dependencies revised"
            )

        # Reject unknown prerequisites (self is reported by the cycle check
        # below, exactly like at submission). Prerequisites may belong to
        # other tenants: dependencies are cluster-global, not tenant-scoped.
        if depends_on:
            placeholders = ",".join("?" for _ in depends_on)
            found = {
                r["id"] for r in conn.execute(
                    f"SELECT id FROM jobs WHERE id IN ({placeholders})",
                    depends_on,
                ).fetchall()
            }
            missing = [
                d for d in depends_on
                if d not in found and d != job_id
            ]
            if missing:
                raise SchedulerError(
                    f"unknown depends_on job id(s): {', '.join(missing)}"
                )

        # Reject anything that would close a cycle (including self
        # dependence), evaluated against the graph the replacement would
        # create: the job's current edges are substituted, not added to.
        cycle = _find_cycle(conn, replace_edges={job_id: depends_on})
        if cycle is not None:
            raise SchedulerError(
                "dependency cycle detected: " + " -> ".join(cycle)
            )

        # Readiness under the OLD edges, captured before the replacement:
        # whether the job is blocked, and if not, the moment its wait
        # became scheduler-facing.
        was_blocked = bool(_unmet_prerequisites(conn, job_id))
        was_ready_since = (
            None if was_blocked else _head_ready_since(conn, row)
        )
        conn.execute(
            "DELETE FROM job_dependencies WHERE job_id = ?", (job_id,)
        )
        for prereq in depends_on:
            conn.execute(
                "INSERT INTO job_dependencies (job_id, depends_on) VALUES (?, ?)",
                (job_id, prereq),
            )
        # Keep the job's readiness clock in sync with the new edges: a
        # revision that releases a previously blocked job starts its
        # scheduler-facing wait at the revision instant; a revision that
        # (re)blocks the job clears the clock; a ready job that stays
        # ready keeps the exact moment its wait became scheduler-facing
        # (stamped explicitly, since the replaced edges may no longer
        # allow deriving it).
        if _unmet_prerequisites(conn, job_id):
            conn.execute(
                "UPDATE jobs SET ready_since = NULL WHERE id = ?", (job_id,)
            )
        elif was_blocked:
            conn.execute(
                "UPDATE jobs SET ready_since = ? WHERE id = ?",
                (now, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET ready_since = ? WHERE id = ?",
                (was_ready_since.isoformat(), job_id),
            )
        return _get_job_or_404(conn, job_id)


# ---------------------------------------------------------------------------
# Dependency cancel cascade
# ---------------------------------------------------------------------------

def _cancel_closure(
    conn: sqlite3.Connection, target_id: str
) -> dict[str, list[str]]:
    """QUEUED jobs that cancelling ``target_id`` would permanently block.

    Breadth-first traversal of the reverse dependency graph (from a
    prerequisite to its dependents), starting at the target. Every
    non-COMPLETED node is expanded: a CANCELLED node can never complete and
    a QUEUED node reached here never will either (its chain back to the
    target guarantees a never-completing prerequisite), so the blockage
    propagates through both; a COMPLETED node is a satisfied prerequisite
    and stops the traversal. Returns a mapping of affected QUEUED job id ->
    one shortest dependency chain from the target (each job on the chain
    depends on the previous one); BFS plus id-sorted neighbour expansion
    makes the recorded chain deterministic.
    """
    dependents: dict[str, list[str]] = {}
    for r in conn.execute(
        "SELECT job_id, depends_on FROM job_dependencies"
    ).fetchall():
        dependents.setdefault(r["depends_on"], []).append(r["job_id"])
    status_of = {
        r["id"]: r["status"]
        for r in conn.execute("SELECT id, status FROM jobs").fetchall()
    }

    chain_of: dict[str, list[str]] = {target_id: [target_id]}
    queue: list[str] = [target_id]
    while queue:
        node = queue.pop(0)
        for nxt in sorted(dependents.get(node, [])):
            if nxt in chain_of or status_of.get(nxt) == COMPLETED:
                continue
            chain_of[nxt] = chain_of[node] + [nxt]
            queue.append(nxt)
    return {
        jid: chain
        for jid, chain in chain_of.items()
        if jid != target_id and status_of.get(jid) == QUEUED
    }


def _cancel_target_row(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    """Fetch the cancel target and enforce the shared admissibility rules."""
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise NotFound(f"job '{job_id}' not found")
    if row["status"] in (COMPLETED, CANCELLED):
        raise Conflict(
            f"job '{job_id}' is already {row['status'].lower()}"
        )
    return row


def cancel_impact(job_id: str) -> dict[str, Any]:
    """Preview the queued descendants a cancellation would permanently block.

    Pure read under the process-wide lock: nothing is written. The target
    must exist (404) and be QUEUED or RUNNING (409) — the same admissibility
    as cancelling it. Affected jobs are sorted by job id and each carries
    one shortest dependency chain from the target.
    """
    with db.get_lock():
        conn = db.get_conn()
        row = _cancel_target_row(conn, job_id)
        affected = _cancel_closure(conn, job_id)
        jobs = [
            _get_job_or_404(conn, jid) | {"chain": affected[jid]}
            for jid in sorted(affected)
        ]
        return {
            "job_id": row["id"],
            "status": row["status"],
            "affected_count": len(jobs),
            "affected_jobs": jobs,
        }


def cancel_cascade(job_id: str) -> dict[str, Any]:
    """Cancel a job and every still-QUEUED job its cancellation dooms.

    Runs under the same process-wide lock and ``BEGIN IMMEDIATE`` write
    transaction as single-job cancellation and dependency revision: the
    impact closure is recomputed against the locked state, then the target
    and every QUEUED job in it are cancelled in one atomic commit. A RUNNING
    target's GPUs are released for subsequent batches. Already-CANCELLED
    closure nodes are traversal-only and left untouched (their
    ``completed_at`` is not rewritten). Tenants whose queue drained get
    their aging clock reset, exactly as with a single cancellation.
    """
    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        _cancel_target_row(conn, job_id)
        affected = _cancel_closure(conn, job_id)
        cancelled_ids = sorted([job_id, *affected])
        placeholders = ",".join("?" for _ in cancelled_ids)
        tenant_ids = {
            r["tenant_id"]
            for r in conn.execute(
                f"SELECT tenant_id FROM jobs WHERE id IN ({placeholders})",
                cancelled_ids,
            ).fetchall()
        }
        conn.execute(
            f"UPDATE jobs SET status = ?, completed_at = ? "
            f"WHERE id IN ({placeholders})",
            [CANCELLED, now, *cancelled_ids],
        )
        # Reset the aging clock of every tenant whose queue drained, exactly
        # as a single cancellation does.
        for tenant_id in tenant_ids:
            if not _has_queued(conn, tenant_id):
                conn.execute(
                    "UPDATE tenants SET aging_settled_at = NULL WHERE id = ?",
                    (tenant_id,),
                )
        return {
            "job_id": job_id,
            "cancelled_count": len(cancelled_ids),
            "cancelled": cancelled_ids,
            "jobs": [_get_job_or_404(conn, jid) for jid in cancelled_ids],
        }


# ---------------------------------------------------------------------------
# GPU maintenance windows
# ---------------------------------------------------------------------------

def _maintenance_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "start_at": row["start_at"],
        "end_at": row["end_at"],
        "reserved_gpus": row["reserved_gpus"],
        "created_at": row["created_at"],
    }


def create_maintenance_window(
    start: datetime,
    end: datetime,
    reserved_gpus: int,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Persist a half-open ``[start, end)`` UTC GPU-maintenance window.

    Rejects an interval whose end is not later than its start and a
    reservation outside ``0..cluster_gpus``. Inside the write transaction
    the candidate is overlaid on the already-persisted windows and the
    peak combined reservation over the candidate's own half-open coverage
    is computed with the same sweep used at dispatch time: a peak
    exceeding the cluster's current capacity rejects the creation (the
    transaction rolls back, so nothing is persisted), while a peak
    exactly equal to the capacity is admissible. Existing RUNNING jobs
    are left untouched: a window only constrains future dispatch, it
    never preempts.
    """
    if not isinstance(reserved_gpus, int) or isinstance(reserved_gpus, bool):
        raise SchedulerError(
            "reserved_gpus must be an integer between 0 and cluster capacity"
        )
    settings = settings or get_settings()
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        raise SchedulerError("end_at must be later than start_at (UTC)")
    if reserved_gpus < 0 or reserved_gpus > settings.cluster_gpus:
        raise SchedulerError(
            f"reserved_gpus must be between 0 and cluster capacity "
            f"{settings.cluster_gpus}"
        )

    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        # Cumulative admission, checked against the locked state so a
        # concurrent creation cannot slip past it: over the candidate's
        # own half-open coverage, its reservation adds to every persisted
        # window overlapping the same instant (endpoint-touching windows
        # never overlap, so they never add). Rejecting here rolls the
        # transaction back: no row is written and no id is consumed.
        peak = _peak_reserved_gpus(
            _load_maintenance_windows(conn) + [(start, end, reserved_gpus)],
            start,
            end,
        )
        if peak > settings.cluster_gpus:
            raise SchedulerError(
                f"maintenance reservations over "
                f"[{start.isoformat()}, {end.isoformat()}) peak at {peak} "
                f"GPUs, exceeding cluster capacity {settings.cluster_gpus}"
            )
        cur = conn.execute(
            """INSERT INTO maintenance_windows
                   (start_at, end_at, reserved_gpus, created_at)
               VALUES (?, ?, ?, ?)""",
            (start.isoformat(), end.isoformat(), reserved_gpus, now),
        )
        row = conn.execute(
            "SELECT * FROM maintenance_windows WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()
        return _maintenance_dict(row)


def list_maintenance_windows(
    active_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """All maintenance windows in chronological order.

    With ``active_at``, only windows active at that UTC instant are returned
    (half-open: ``start_at <= t < end_at``).
    """
    with db.get_lock():
        conn = db.get_conn()
        if active_at is None:
            rows = conn.execute(
                "SELECT * FROM maintenance_windows ORDER BY start_at, id"
            ).fetchall()
        else:
            ts = _as_utc(active_at).isoformat()
            rows = conn.execute(
                """SELECT * FROM maintenance_windows
                    WHERE start_at <= ? AND end_at > ?
                    ORDER BY start_at, id""",
                (ts, ts),
            ).fetchall()
        return [_maintenance_dict(r) for r in rows]


def _load_maintenance_windows(conn: sqlite3.Connection) -> list[
    tuple[datetime, datetime, int]
]:
    """All windows as aware-UTC (start, end, reserved) tuples."""
    return [
        (
            _as_utc(_parse_iso(r["start_at"])),
            _as_utc(_parse_iso(r["end_at"])),
            int(r["reserved_gpus"]),
        )
        for r in conn.execute(
            "SELECT start_at, end_at, reserved_gpus "
            "FROM maintenance_windows ORDER BY id"
        ).fetchall()
    ]


def _peak_reserved_gpus(
    windows: list[tuple[datetime, datetime, int]],
    start: datetime,
    end: datetime,
) -> int:
    """Peak combined reservation of windows overlapping ``[start, end)``.

    Windows are half-open; a candidate interval is half-open too, so overlap
    is a non-empty ``[max(starts), min(ends))`` — windows that merely meet at
    an endpoint do not overlap and their reservations are never added.
    Reservations of simultaneously overlapping windows sum, and the sweep
    returns the largest such sum reached anywhere inside the interval
    (``0`` when nothing overlaps).
    """
    start = _as_utc(start)
    end = _as_utc(end)
    events: list[tuple[datetime, int]] = []
    for w_start, w_end, reserved in windows:
        lo = max(w_start, start)
        hi = min(w_end, end)
        if hi <= lo:
            continue  # empty half-open intersection -> not overlapping
        events.append((lo, reserved))
        events.append((hi, -reserved))
    if not events:
        return 0
    # At the same instant all ending reservations are removed before starting
    # reservations are added: touching windows (one ends exactly when another
    # begins) must not momentarily stack. Processing the +0 deltas too keeps
    # the running sum non-negative and the peak exactly 0 on touch-only gaps.
    events.sort(key=lambda e: (e[0], e[1]))
    current = peak = 0
    for _, delta in events:
        current += delta
        if current > peak:
            peak = current
    return peak


def _feasible_start_slots(
    windows: list[tuple[datetime, datetime, int]],
    start: datetime,
    end: datetime,
    duration: timedelta,
    headroom: int,
) -> list[tuple[datetime, datetime, bool, bool]]:
    """Feasible estimated-start instants inside the half-open ``[start, end)``.

    A start instant ``t`` is feasible when, throughout the whole predicted
    run ``[t, t + duration)``, the combined maintenance reservation never
    exceeds ``headroom`` (cluster capacity minus the fixed RUNNING baseline
    minus the job's own GPUs). Returns chronological ``(start, end,
    start_inclusive, end_inclusive)`` slots; a degenerate slot
    (``start == end``, both inclusive) is an isolated feasible instant
    wedged between two blocked regions.
    """
    # Only windows that can touch some run interval [t, t + duration) for a
    # start t inside the query range matter; clip them to that horizon.
    horizon_end = end + duration
    events: list[tuple[datetime, int]] = []
    for w_start, w_end, reserved in windows:
        lo = max(w_start, start)
        hi = min(w_end, horizon_end)
        if hi <= lo:
            continue  # empty half-open intersection -> not overlapping
        events.append((lo, reserved))
        events.append((hi, -reserved))
    events.sort(key=lambda e: (e[0], e[1]))

    # Sweep the reservation step function and keep the maximal half-open
    # segments [a, b) whose combined reservation exceeds the headroom.
    # Overlapping windows sum; windows merely touching at an endpoint never
    # stack, so adjacent over-headroom segments merge exactly.
    blocked_sources: list[tuple[datetime, datetime]] = []
    current = 0
    high_since: datetime | None = None
    i = 0
    while i < len(events):
        t = events[i][0]
        while i < len(events) and events[i][0] == t:
            current += events[i][1]
            i += 1
        # ``current`` is the combined reservation on [t, next event).
        if current > headroom:
            if high_since is None:
                high_since = t
        elif high_since is not None:
            blocked_sources.append((high_since, t))
            high_since = None
    if high_since is not None:
        blocked_sources.append((high_since, horizon_end))

    # A start t is blocked exactly when [t, t + duration) meets a segment
    # [a, b), i.e. when t lies in the OPEN interval (a - duration, b): a run
    # ending exactly at a (or starting exactly at b) merely touches the
    # reservation and stays feasible. Removing every such open interval from
    # the half-open query range [start, end) leaves the feasible slots.
    slots: list[tuple[datetime, datetime, bool, bool]] = []
    cursor = start
    for a, b in blocked_sources:
        u = a - duration
        v = b
        if v <= start or u >= end:
            continue  # blocked starts lie entirely outside the query range
        if u >= cursor:
            # u itself stays feasible (the blocked interval is open there);
            # u == cursor yields a degenerate slot: an isolated instant.
            slots.append((cursor, u, True, True))
        if v > cursor:
            # The blocked interval is open at v, so v itself is feasible and
            # the next slot starts there.
            cursor = v
    if cursor < end:
        slots.append((cursor, end, True, False))
    return slots


def job_maintenance_slots(
    job_id: str,
    start: datetime,
    end: datetime,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Feasible estimated-start slots for a QUEUED job over ``[start, end)``.

    The current RUNNING GPU usage is taken as a FIXED baseline held for the
    whole horizon (no future releases are speculated): a candidate start
    ``t`` is feasible iff at every instant of the predicted run
    ``[t, t + duration_min)`` the baseline plus the job's GPUs plus the
    combined maintenance reservation stays within cluster capacity. The
    answer is purely a resource-capacity view — tenant caps, budgets,
    balances, dependencies and other queued jobs are not consulted.
    """
    settings = settings or get_settings()
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        raise SchedulerError("end must be later than start (UTC)")
    with db.get_lock():
        conn = db.get_conn()
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise NotFound(f"job '{job_id}' not found")
        if row["status"] != QUEUED:
            raise Conflict(
                f"job '{job_id}' is {row['status'].lower()}; maintenance "
                "slots are only computed for QUEUED jobs"
            )
        used = int(conn.execute(
            "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = ?",
            (RUNNING,),
        ).fetchone()["n"])
        windows = _load_maintenance_windows(conn)

        gpus = int(row["gpus"])
        duration_min = int(row["duration_min"])
        headroom = settings.cluster_gpus - used - gpus
        raw_slots = (
            _feasible_start_slots(
                windows, start, end, timedelta(minutes=duration_min), headroom
            )
            if headroom >= 0 else []
        )
        slots = [
            {
                "start_at": s.isoformat(),
                "end_at": e.isoformat(),
                "start_inclusive": start_inclusive,
                "end_inclusive": end_inclusive,
            }
            for s, e, start_inclusive, end_inclusive in raw_slots
        ]
        return {
            "job_id": row["id"],
            "status": row["status"],
            "gpus": gpus,
            "duration_min": duration_min,
            "cluster_gpus": settings.cluster_gpus,
            # Fixed baseline: GPUs held by RUNNING jobs right now, assumed to
            # stay occupied across the whole query horizon.
            "running_baseline_gpus": used,
            "query_start_at": start.isoformat(),
            "query_end_at": end.isoformat(),
            # Earliest feasible estimated start, or null when no slot exists.
            "earliest_start_at": slots[0]["start_at"] if slots else None,
            "slots": slots,
        }


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def run_batch(
    now: datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Schedule the next batch of jobs and commit every effect.

    Returns the selection order, per-tenant balances after the batch, capacity
    usage and an explainable ledger of credit/debit events.
    """
    return _run_batch_transaction(now=now, settings=settings, persist=True)


def preview_batch(
    now: datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Dry-run :func:`run_batch` without persisting anything.

    The exact same batch transaction body executes (WDRR, aging, caps and
    dependency checks — there is no second algorithm), after which the
    transaction is rolled back: jobs stay QUEUED, balances/aging clocks are
    untouched, and no batch or ledger row survives (a rollback on error leaves
    no partial changes either). The returned snapshot is flagged
    ``preview: true``; everything else — selection order, post-batch GPU
    usage, balances and the full credit-event stream, including prospective
    batch/ledger row ids — is what the subsequently committed batch at the
    same reference instant on the same state will produce.
    """
    return _run_batch_transaction(now=now, settings=settings, persist=False)


def _run_batch_transaction(
    now: datetime | None,
    settings: Settings | None,
    persist: bool,
) -> dict[str, Any]:
    """Shared body of a real batch (``persist=True``) and a preview.

    Both modes execute byte-identical decisions inside one explicit
    transaction; only the boundary differs:

    * ``persist=True``  -> ``COMMIT`` (``POST /scheduler/batch``);
    * ``persist=False`` -> ``ROLLBACK`` (``POST /scheduler/preview``).

    Any exception rolls back (a no-op for a clean preview), so an aborted
    batch can never leave partial ledger/balance/job changes behind.
    """
    settings = settings or get_settings()
    if now is None:
        now = datetime.now(timezone.utc)
    now_str = _now_iso(_as_utc(now))
    with db.get_lock():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = _batch_tx_body(conn, settings, now_str, preview=not persist)
        except BaseException:
            conn.rollback()
            raise
        if persist:
            conn.commit()
        else:
            # Discard every tentative write: batch row, ledger entries, job
            # status transitions, balance and aging-clock updates.
            conn.rollback()
        return result


def _batch_tx_body(
    conn: sqlite3.Connection,
    settings: Settings,
    now_str: str,
    preview: bool = False,
) -> dict[str, Any]:
    """The scheduling algorithm itself; operates only on ``conn``.

    All writes are part of the caller's transaction, so the caller chooses
    commit vs. rollback. The algorithm never sees whether this is a preview.
    """
    used = conn.execute(
        "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = ?",
        (RUNNING,),
    ).fetchone()["n"]
    available = settings.cluster_gpus - used

    # Every selected job is predicted to start at the batch reference instant
    # and run for its duration_min: [now, now + duration_min). The maintenance
    # admission test below uses the peak reservation over that half-open
    # interval. Loaded once per batch; preview and commit share this read.
    now_dt = _as_utc(_parse_iso(now_str))
    windows = _load_maintenance_windows(conn)
    # GPUs handed out to jobs selected earlier in THIS batch; RUNNING usage
    # (``used``) is separate and is never reclaimed by a maintenance window.
    selected_gpus = 0

    cur = conn.execute(
        """INSERT INTO batches
               (created_at, now_ref, capacity_gpus, used_gpus_before,
                available_gpus, rounds, jobs_selected)
           VALUES (?, ?, ?, ?, ?, 0, 0)""",
        (now_str, now_str, settings.cluster_gpus, used,
         max(available, 0)),
    )
    batch_id = cur.lastrowid

    active = conn.execute(
        """SELECT DISTINCT j.tenant_id, t.weight, t.deficit,
                          t.max_gpus, t.budget_gpu_min, t.aging_settled_at
             FROM jobs j JOIN tenants t ON t.id = j.tenant_id
            WHERE j.status = ?
            ORDER BY j.tenant_id""",
        (QUEUED,),
    ).fetchall()

    if not active:
        conn.execute(
            "UPDATE batches SET rounds = 0, jobs_selected = 0 WHERE id = ?",
            (batch_id,),
        )
        return _batch_result(conn, batch_id, settings, preview)

    # Dependency readiness is recomputed from the *current* head of each
    # queue: only a COMPLETED prerequisite releases a job, and a tenant's
    # next head (revealed after a dispatch) may itself be blocked. A
    # tenant whose head-of-line job is blocked neither accrues aging nor
    # spends quantum: waiting on another job is not scheduler starvation,
    # and blocked batches must not hand out spendable credit.
    def head_of(tenant_id: str):
        return conn.execute(
            """SELECT * FROM jobs
                WHERE tenant_id = ? AND status = ?
                ORDER BY queued_at, id LIMIT 1""",
            (tenant_id, QUEUED),
        ).fetchone()

    def head_ready(head) -> bool:
        return head is not None and not _unmet_prerequisites(
            conn, head["id"]
        )

    active_ids = [t["tenant_id"] for t in active]
    ready_at_start = {
        tid for tid in active_ids if head_ready(head_of(tid))
    }

    # Per-tenant concurrent-GPU caps (NULL = unlimited). Dispatch also
    # respects each tenant's cap in addition to cluster capacity: the GPUs
    # a tenant already has RUNNING plus the head job's request may not
    # exceed its cap.
    caps: dict[str, int | None] = {
        t["tenant_id"]: t["max_gpus"] for t in active
    }

    # Rolling 24h GPU-minute budgets (NULL = unlimited). ``budget_spent`` is
    # the cost charged by earlier committed batches for jobs dispatched inside
    # (now-24h, now]; it is loaded once per batch and never changes during the
    # loop. ``selected_cost`` accumulates the cost of jobs selected earlier in
    # THIS batch: their scheduled_at is ``now`` (inside the window), and since
    # they are no longer QUEUED they never reappear in ``budget_spent``'s query,
    # so the running total counts each of them exactly once.
    budgets: dict[str, int | None] = {
        t["tenant_id"]: t["budget_gpu_min"] for t in active
    }
    budget_spent: dict[str, int] = _budget_spent_map(conn, now_dt)
    selected_cost: dict[str, int] = {}
    tenant_running: dict[str, int] = {}

    def running_gpus(tenant_id: str) -> int:
        """GPUs currently held by the tenant's RUNNING jobs."""
        row = conn.execute(
            "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs "
            "WHERE tenant_id = ? AND status = ?",
            (tenant_id, RUNNING),
        ).fetchone()
        return int(row["n"])

    for tid in active_ids:
        tenant_running[tid] = running_gpus(tid)

    def head_reservation(head) -> int:
        """Peak combined maintenance reservation overlapping the head job's
        predicted run interval [now, now + duration_min)."""
        end_dt = now_dt + timedelta(minutes=head["duration_min"])
        return _peak_reserved_gpus(windows, now_dt, end_dt)

    def tenant_can_fit(tenant_id: str, head, head_gpus: int, free_gpus: int) -> bool:
        """Head job must fit the cluster capacity left over after the peak
        maintenance reservation over its predicted run, the tenant's own
        concurrent-GPU cap, and the tenant's rolling 24h GPU-minute budget.

        The cluster test accounts for every allocation held throughout the
        candidate's run: pre-existing RUNNING jobs (``used``), jobs already
        selected in this batch (``selected_gpus``) and the candidate itself.
        ``free_gpus`` is the remaining *physical* headroom, i.e. the same
        total minus the running and already-selected usage.
        """
        reserved = head_reservation(head)
        if used + selected_gpus + head_gpus + reserved > settings.cluster_gpus:
            return False
        if head_gpus > free_gpus:
            return False
        cap = caps.get(tenant_id)
        if cap is not None and tenant_running[tenant_id] + head_gpus > cap:
            return False
        budget = budgets.get(tenant_id)
        if budget is not None:
            candidate_cost = int(head["gpus"]) * int(head["duration_min"])
            if budget_spent.get(tenant_id, 0) \
                    + selected_cost.get(tenant_id, 0) + candidate_cost > budget:
                return False
        return True

    # Aging is settled even when the cluster is full, so waiting tenants
    # keep accumulating starvation compensation across blocked batches.
    # --- Aging settlement at batch start -------------------------------
    # Compensation covers the wait since the head job truly became
    # schedulable: on first observation we settle from the readiness
    # instant (arrival, last prerequisite completion or unblocking
    # revision — see _head_ready_since), afterwards from the previous
    # settlement instant, whichever is LATER. A head job queued for 60
    # minutes whose prerequisite completed one minute before the batch
    # therefore earns one minute of credit, not sixty, even when no batch
    # ran (and could advance the checkpoint) during the blocked wait.
    # Same-instant retries grant nothing, so the endpoint cannot be farmed
    # via aging either.
    # Dependency-blocked tenants receive nothing, but their checkpoint is
    # advanced: once dependencies release them, waiting is measured from
    # the release point rather than the original submission.
    for t in active:
        if t["tenant_id"] not in ready_at_start:
            conn.execute(
                "UPDATE tenants SET aging_settled_at = ?, updated_at = ? "
                "WHERE id = ?",
                (now_str, now_str, t["tenant_id"]),
            )
            continue
        oldest = conn.execute(
            "SELECT MIN(queued_at) AS ts FROM jobs "
            "WHERE tenant_id = ? AND status = ?",
            (t["tenant_id"], QUEUED),
        ).fetchone()["ts"]
        base = t["aging_settled_at"] or oldest
        base_dt = _as_utc(_parse_iso(base))
        ready_dt = _head_ready_since(conn, head_of(t["tenant_id"]))
        if ready_dt > base_dt:
            base_dt = ready_dt
        waited_min = (
            _parse_iso(now_str) - base_dt
        ).total_seconds() / 60.0
        if waited_min < 0:
            waited_min = 0.0
        bonus = settings.aging_rate_gpu_min_per_min * waited_min
        if settings.aging_cap_gpu_min is not None:
            bonus = min(bonus, settings.aging_cap_gpu_min)
        if bonus >= _EPS:
            _credit(
                conn, t["tenant_id"], bonus, "AGING", now_str,
                batch_id=batch_id,
                detail=f"waited {waited_min:.3f} min",
            )
        conn.execute(
            "UPDATE tenants SET aging_settled_at = ?, updated_at = ? "
            "WHERE id = ?",
            (now_str, now_str, t["tenant_id"]),
        )

    if available <= 0:
        conn.execute(
            "UPDATE batches SET rounds = 0, jobs_selected = 0 WHERE id = ?",
            (batch_id,),
        )
        return _batch_result(conn, batch_id, settings, preview)

    # --- WDRR rounds ----------------------------------------------------
    # A tenant is visited in a round only while its head job FITS the
    # currently free capacity AND the tenant's own concurrent-GPU cap.
    # Fitting-but-unaffordable jobs keep accumulating quantum across rounds
    # until affordable. A tenant whose head job cannot physically fit —
    # blocked by cluster capacity OR by its tenant cap (running jobs
    # already filling it, or the head alone exceeding it) — accrues no
    # quantum this round: waiting for completions/cancellations is
    # compensated by aging, and this prevents callers farming quantum by
    # repeatedly invoking the endpoint. Other tenants are unaffected:
    # each tenant's eligibility is evaluated independently, so a capped
    # tenant is simply skipped while the rest still dispatch.
    #
    # Each round's grants are tentative above a commit watermark. The
    # moment a round produces a dispatch, every grant up to that point is
    # committed (it is ordinary DRR accrual). When the batch ends without
    # a trailing dispatch (e.g. max_rounds exhausted), the uncommitted
    # grants are rolled back so blocked calls leave balances unchanged.
    selected: list[str] = []
    run_index = 0
    rounds = 0
    free = available
    commit_mark = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS m FROM ledger"
    ).fetchone()["m"]

    def rollback_uncommitted() -> None:
        rows = conn.execute(
            "SELECT tenant_id, amount FROM ledger WHERE id > ?",
            (commit_mark,),
        ).fetchall()
        for r in rows:
            conn.execute(
                "UPDATE tenants SET deficit = deficit - ? WHERE id = ?",
                (r["amount"], r["tenant_id"]),
            )
        conn.execute("DELETE FROM ledger WHERE id > ?", (commit_mark,))

    while rounds < settings.max_rounds:
        fitting = []
        for tid in active_ids:
            h = head_of(tid)
            if (
                h is not None
                and head_ready(h)
                and tenant_can_fit(tid, h, h["gpus"], free)
            ):
                fitting.append(tid)
        if not fitting:
            break  # every queued job is capacity/cap/dependency-blocked

        rounds += 1
        dispatched_this_round = False

        for tenant_id in fitting:
            if free <= 0:
                break
            head = head_of(tenant_id)
            if head is None or not head_ready(head) or not tenant_can_fit(
                tenant_id, head, head["gpus"], free
            ):
                # Became ineligible after earlier tenants in this round
                # (free GPUs consumed, or the tenant used its cap headroom).
                continue

            weight = conn.execute(
                "SELECT weight FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchone()["weight"]
            quantum = settings.base_quantum_gpu_min * float(weight)
            _credit(
                conn, tenant_id, quantum, "QUANTUM", now_str,
                batch_id=batch_id, detail=f"round {rounds}",
            )

            bal = conn.execute(
                "SELECT deficit FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchone()["deficit"]
            cost = head["cost_gpu_min"]

            if bal + _EPS < cost:
                continue  # keep accruing in later rounds
            # Capacity, maintenance and cap invariants: never crossed.
            if not tenant_can_fit(tenant_id, head, head["gpus"], free):
                continue

            run_index += 1
            conn.execute(
                """UPDATE jobs
                      SET status = ?, scheduled_at = ?, batch_id = ?,
                          run_index = ?
                    WHERE id = ?""",
                (RUNNING, now_str, batch_id, run_index, head["id"]),
            )
            _credit(
                conn, tenant_id, -cost, "DISPATCH", now_str,
                batch_id=batch_id, job_id=head["id"],
                detail=f"{head['gpus']} GPU x {head['duration_min']} min",
            )
            selected.append(head["id"])
            free -= head["gpus"]
            selected_gpus += head["gpus"]
            tenant_running[tenant_id] += head["gpus"]
            # Charge the 24h rolling budget once, at dispatch; completion and
            # cancellation never refund it. In a preview this whole
            # transaction rolls back, so the charge never reaches the DB.
            selected_cost[tenant_id] = (
                selected_cost.get(tenant_id, 0)
                + int(head["gpus"]) * int(head["duration_min"])
            )
            dispatched_this_round = True

        if dispatched_this_round:
            # All accrual that led to this dispatch is now permanent.
            commit_mark = conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS m FROM ledger"
            ).fetchone()["m"]
        # Idle rounds are normal while tenants are still accruing toward
        # affordability (positive weights guarantee they get there); the
        # loop continues until dispatch or no job fits any longer.

    # Grants above the commit watermark never led to a dispatch (idle
    # trailing rounds, or a batch cut off by max_rounds): roll them back
    # so that invoking the endpoint while blocked leaves balances and the
    # ledger untouched.
    rollback_uncommitted()

    # Close aging windows for tenants whose queue drained; others keep
    # accruing wait time from this batch's reference instant.
    for tenant_id in active_ids:
        if _has_queued(conn, tenant_id):
            conn.execute(
                "UPDATE tenants SET aging_settled_at = ? WHERE id = ?",
                (now_str, tenant_id),
            )
        else:
            conn.execute(
                "UPDATE tenants SET aging_settled_at = NULL WHERE id = ?",
                (tenant_id,),
            )

    conn.execute(
        "UPDATE batches SET rounds = ?, jobs_selected = ? WHERE id = ?",
        (rounds, len(selected), batch_id),
    )
    return _batch_result(conn, batch_id, settings, preview)


def _batch_result(
    conn: sqlite3.Connection,
    batch_id: int,
    settings: Settings,
    preview: bool = False,
) -> dict[str, Any]:
    batch = conn.execute(
        "SELECT * FROM batches WHERE id = ?", (batch_id,)
    ).fetchone()
    jobs = conn.execute(
        """SELECT * FROM jobs WHERE batch_id = ? ORDER BY run_index""",
        (batch_id,),
    ).fetchall()
    used = conn.execute(
        "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = ?",
        (RUNNING,),
    ).fetchone()["n"]

    # Maintenance context at the batch reference instant. Every selected job
    # was predicted to run from that instant for its duration_min; annotate it
    # with the peak reservation over its own half-open interval.
    now_dt = _as_utc(_parse_iso(batch["now_ref"]))
    windows = _load_maintenance_windows(conn)
    reserved_at_start = _peak_reserved_gpus(
        windows, now_dt, now_dt + timedelta(microseconds=1)
    )

    def annotated_job(j: sqlite3.Row) -> dict[str, Any]:
        est_end = now_dt + timedelta(minutes=j["duration_min"])
        peak_reserved = _peak_reserved_gpus(windows, now_dt, est_end)
        return _job_dict(conn, j) | {
            "run_index": j["run_index"],
            "estimated_start_at": now_dt.isoformat(),
            "estimated_end_at": est_end.isoformat(),
            "max_maintenance_reserved_gpus": peak_reserved,
        }

    return {
        "batch_id": batch["id"],
        # False for a committed batch; True for a rolled-back preview whose
        # batch_id/ledger ids are only prospective (never persisted).
        "preview": preview,
        "created_at": batch["created_at"],
        "capacity_gpus": batch["capacity_gpus"],
        "used_gpus_before": batch["used_gpus_before"],
        "available_gpus_at_start": batch["available_gpus"],
        # GPUs reserved by maintenance windows active at the batch instant;
        # these are unavailable to a zero-duration probe and lower the room
        # for newly dispatched jobs (RUNNING jobs are never preempted).
        "maintenance_reserved_gpus_at_start": reserved_at_start,
        "effective_free_gpus_at_start": max(
            batch["available_gpus"] - reserved_at_start, 0
        ),
        "used_gpus_after": used,
        "free_gpus_after": settings.cluster_gpus - used,
        "rounds": batch["rounds"],
        "jobs_selected": len(jobs),
        "selection_order": [j["id"] for j in jobs],
        "selected_jobs": [annotated_job(j) for j in jobs],
        # Tenant budget figures are evaluated at this batch's reference
        # instant, so they match the very window the decision was made on.
        "tenants": _list_tenants(conn, window_end=now_dt),
        "events": _list_ledger(conn, batch_id=batch_id),
    }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def list_jobs(
    status: str | None = None,
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM jobs WHERE 1=1"
    params: list[Any] = []
    if status:
        valid = {QUEUED, RUNNING, COMPLETED, CANCELLED}
        if status not in valid:
            raise SchedulerError(f"status must be one of {sorted(valid)}")
        sql += " AND status = ?"
        params.append(status)
    if tenant_id:
        sql += " AND tenant_id = ?"
        params.append(tenant_id)
    sql += " ORDER BY queued_at, id"
    with db.get_lock():
        conn = db.get_conn()
        return [_job_dict(conn, r) for r in conn.execute(sql, params)]


def gpu_usage(
    start: datetime,
    end: datetime,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate GPU usage over a half-open ``[start, end)`` UTC window.

    Usage is computed from each job's *actual RUNNING interval*:

    * finished jobs (COMPLETED/CANCELLED while RUNNING): ``scheduled_at`` to
      ``completed_at``;
    * still-RUNNING jobs: ``scheduled_at`` truncated to the query cutoff.

    QUEUED jobs have never run and never contribute. Intervals intersect the
    query window left-closed/right-open, so a job starting exactly at ``end``
    or ending exactly at ``start`` contributes nothing.

    Returns per-tenant and global GPU-minutes (``gpus`` times the intersected
    duration in minutes) plus the number of overlapping RUNNING intervals.
    Every known tenant (or just ``tenant_id`` when filtered) appears, with
    zero values when none of its jobs overlaps the window.
    """
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        raise SchedulerError("end must be later than start (UTC)")

    with db.get_lock():
        conn = db.get_conn()
        if tenant_id is not None:
            if conn.execute(
                "SELECT 1 FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchone() is None:
                raise NotFound(f"tenant '{tenant_id}' not found")
            tenant_rows = conn.execute(
                "SELECT id FROM tenants WHERE id = ? ORDER BY id",
                (tenant_id,),
            ).fetchall()
        else:
            tenant_rows = conn.execute(
                "SELECT id FROM tenants ORDER BY id"
            ).fetchall()

        per_tenant: dict[str, dict[str, Any]] = {
            r["id"]: {"tenant_id": r["id"], "gpu_minutes": 0.0,
                      "overlapping_jobs": 0}
            for r in tenant_rows
        }

        # Only jobs that actually entered RUNNING state: their RUNNING interval
        # is [scheduled_at, completed_at), truncated to the query cutoff.
        rows = conn.execute(
            """SELECT id, tenant_id, gpus, status,
                      scheduled_at, completed_at
                 FROM jobs
                WHERE scheduled_at IS NOT NULL
                  AND (? IS NULL OR tenant_id = ?)
                ORDER BY id""",
            (tenant_id, tenant_id),
        ).fetchall()

        total_minutes = 0.0
        total_overlaps = 0
        for r in rows:
            ran_start = _parse_iso(r["scheduled_at"])
            if r["completed_at"] is not None:
                ran_end = _parse_iso(r["completed_at"])
            else:
                # Still RUNNING: the run is assumed to extend to the query
                # cutoff, never beyond it.
                ran_end = end

            lo = max(ran_start, start)
            hi = min(ran_end, end)
            if hi <= lo:
                continue  # empty half-open intersection -> not overlapping

            minutes = (hi - lo).total_seconds() / 60.0
            gpu_minutes = r["gpus"] * minutes
            entry = per_tenant[r["tenant_id"]]
            entry["gpu_minutes"] += gpu_minutes
            entry["overlapping_jobs"] += 1
            total_minutes += gpu_minutes
            total_overlaps += 1

        tenants = [
            {
                "tenant_id": t,
                "gpu_minutes": round(v["gpu_minutes"], 6),
                "overlapping_jobs": v["overlapping_jobs"],
            }
            for t, v in per_tenant.items()
        ]
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tenant_id": tenant_id,
            "global": {
                "gpu_minutes": round(total_minutes, 6),
                "overlapping_jobs": total_overlaps,
            },
            "tenants": tenants,
        }


def queue_overview() -> dict[str, Any]:
    """Return per-tenant FIFO queues with positions, weights and balances."""
    with db.get_lock():
        conn = db.get_conn()
        settings = get_settings()
        used = conn.execute(
            "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = ?",
            (RUNNING,),
        ).fetchone()["n"]

        tenants = conn.execute(
            _TENANT_SELECT + " ORDER BY t.id"
        ).fetchall()
        now_dt = _now_utc()
        spent_map = _budget_spent_map(conn, now_dt)
        groups = []
        for t in tenants:
            queued = conn.execute(
                """SELECT * FROM jobs WHERE tenant_id = ? AND status = ?
                   ORDER BY queued_at, id""",
                (t["id"], QUEUED),
            ).fetchall()
            if not queued:
                continue
            budget = t["budget_gpu_min"]
            spent = spent_map.get(t["id"], 0)
            groups.append({
                "tenant_id": t["id"],
                "weight": round(t["weight"], 6),
                "balance_gpu_min": round(t["deficit"], 6),
                "max_gpus": t["max_gpus"],
                "budget_gpu_min": budget,
                "budget_used_gpu_min": spent,
                "budget_remaining_gpu_min": (
                    None if budget is None else max(budget - spent, 0)
                ),
                "budget_window_start_at": (
                    now_dt - timedelta(hours=24)
                ).isoformat(),
                "budget_window_end_at": now_dt.isoformat(),
                "running_gpus": t["running_gpus"],
                "oldest_queued_at": queued[0]["queued_at"],
                "jobs": [
                    _job_dict(conn, j) | {"position": pos}
                    for pos, j in enumerate(queued, start=1)
                ],
            })

        total_queued = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = ?", (QUEUED,)
        ).fetchone()["n"]
        total_running = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = ?", (RUNNING,)
        ).fetchone()["n"]
        return {
            "cluster_gpus": settings.cluster_gpus,
            "used_gpus": used,
            "free_gpus": settings.cluster_gpus - used,
            "queued_jobs": total_queued,
            "running_jobs": total_running,
            "queues": groups,
        }


def scheduler_state() -> dict[str, Any]:
    with db.get_lock():
        conn = db.get_conn()
        settings = get_settings()
        used = conn.execute(
            "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = ?",
            (RUNNING,),
        ).fetchone()["n"]
        rows = conn.execute(_TENANT_SELECT + " ORDER BY t.id").fetchall()
        return {
            "cluster_gpus": settings.cluster_gpus,
            "used_gpus": used,
            "free_gpus": settings.cluster_gpus - used,
            "settings": {
                "base_quantum_gpu_min": settings.base_quantum_gpu_min,
                "aging_rate_gpu_min_per_min":
                    settings.aging_rate_gpu_min_per_min,
                "aging_cap_gpu_min": settings.aging_cap_gpu_min,
            },
            "tenants": [_tenant_dict(conn, r) for r in rows],
        }


def _list_ledger(
    conn: sqlite3.Connection,
    tenant_id: str | None = None,
    batch_id: int | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM ledger WHERE 1=1"
    params: list[Any] = []
    if tenant_id:
        sql += " AND tenant_id = ?"
        params.append(tenant_id)
    if batch_id is not None:
        sql += " AND batch_id = ?"
        params.append(batch_id)
    sql += " ORDER BY id ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": r["id"],
            "tenant_id": r["tenant_id"],
            "batch_id": r["batch_id"],
            "job_id": r["job_id"],
            "kind": r["kind"],
            "amount": round(r["amount"], 6),
            "balance_after": round(r["balance_after"], 6),
            "detail": r["detail"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def list_ledger(
    tenant_id: str | None = None,
    batch_id: int | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    if limit <= 0 or limit > 10_000:
        raise SchedulerError("limit must be between 1 and 10000")
    with db.get_lock():
        return _list_ledger(
            db.get_conn(), tenant_id=tenant_id, batch_id=batch_id, limit=limit
        )
