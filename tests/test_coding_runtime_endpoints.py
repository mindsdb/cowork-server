from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from coding_service_fakes import FakeEngine, service_with

from cowork.api.v1.endpoints import coding_runtime, health
from cowork.coding.control_models import ComputerCapabilities


def test_lease_long_poll_does_not_block_other_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = service_with(tmp_path, FakeEngine())
    computer, runtime_token = service.control.register_runtime(
        service.control.issue_registration_token(),
        "Build computer",
        ComputerCapabilities(platform="linux", architecture="test", runtime_version="test-runtime"),
    )
    polling = threading.Event()
    release = threading.Event()

    def blocked_acquire_lease(_computer_id: str) -> None:
        polling.set()
        release.wait(timeout=2)
        return None

    monkeypatch.setattr(service.remote, "acquire_lease", blocked_acquire_lease)
    monkeypatch.setattr(coding_runtime, "get_coding_service", lambda: service)
    app = FastAPI()
    app.include_router(coding_runtime.router, prefix="/api/v1/coding/runtime")
    app.include_router(health.router, prefix="/api/v1/health")
    lease_statuses: list[int] = []

    with TestClient(app) as client:
        def poll_lease() -> None:
            response = client.post(
                f"/api/v1/coding/runtime/computers/{computer.id}/lease",
                headers={"Authorization": f"Bearer {runtime_token}"},
                json={"protocol_version": "1.0", "wait_seconds": 1},
            )
            lease_statuses.append(response.status_code)

        poller = threading.Thread(target=poll_lease, name="lease-long-poll")
        poller.start()
        assert polling.wait(timeout=5)
        started = time.monotonic()
        health_response = client.get("/api/v1/health/")
        health_elapsed = time.monotonic() - started
        release.set()
        poller.join(timeout=5)

    assert health_response.status_code == 200
    assert health_elapsed < 0.5
    assert lease_statuses == [200]
