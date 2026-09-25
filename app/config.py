from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _get_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    # SQLite database file path (use "/data/scheduler.db" in the container).
    db_path: str
    # Total physical GPUs in the cluster; the scheduler never exceeds this.
    cluster_gpus: int
    # DRR quantum granted per unit of weight per round, measured in GPU-minutes
    # (job cost = requested GPUs * estimated duration minutes).
    base_quantum_gpu_min: float
    # Aging compensation: GPU-minutes of deficit credit granted per minute of
    # wait of a tenant's oldest queued job.
    aging_rate_gpu_min_per_min: float
    # Optional upper bound on the aging credit granted in a single batch.
    aging_cap_gpu_min: float | None
    # Safety stop for the per-batch round loop (normally terminates far sooner).
    max_rounds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        db_path=os.getenv("DB_PATH", "./data/scheduler.db"),
        cluster_gpus=_get_int("CLUSTER_GPUS", 8),
        base_quantum_gpu_min=_get_float("BASE_QUANTUM_GPU_MIN", 60.0),
        aging_rate_gpu_min_per_min=_get_float("AGING_RATE_GPU_MIN_PER_MIN", 1.0),
        aging_cap_gpu_min=_get_optional_float("AGING_CAP_GPU_MIN"),
        max_rounds=_get_int("MAX_ROUNDS", 10_000),
    )
