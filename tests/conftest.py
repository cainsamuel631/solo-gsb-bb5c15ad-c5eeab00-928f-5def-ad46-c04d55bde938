"""Shared pytest fixtures: fresh in-memory DB + deterministic settings."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def env_settings(monkeypatch):
    monkeypatch.setenv("DB_PATH", ":memory:")
    monkeypatch.setenv("CLUSTER_GPUS", "8")
    monkeypatch.setenv("BASE_QUANTUM_GPU_MIN", "60")
    monkeypatch.setenv("AGING_RATE_GPU_MIN_PER_MIN", "1")
    monkeypatch.delenv("AGING_CAP_GPU_MIN", raising=False)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def conn(env_settings):
    db.reset_for_tests()
    c = db.connect(":memory:")
    yield c
    db.reset_for_tests()


@pytest.fixture
def client(env_settings):
    db.reset_for_tests()
    with TestClient(app) as c:
        yield c
    db.reset_for_tests()


def iso(minutes: float) -> str:
    return at(minutes).isoformat()


def at(minutes: float) -> datetime:
    return EPOCH + timedelta(minutes=minutes)
