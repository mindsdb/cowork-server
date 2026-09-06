from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
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


def test_lease_long_polls_do_not_hold_threadpool_tokens_while_waiting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = service_with(tmp_path, FakeEngine())
    computer, runtime_token = service.control.register_runtime(
        service.control.issue_registration_token(),
        "Build computer",
        ComputerCapabilities(platform="linux", architecture="test", runtime_version="test-runtime"),
    )
    polls = threading.Semaphore(0)
    monkeypatch.setattr(service.remote, "acquire_lease", lambda _computer_id: polls.release())
    monkeypatch.setattr(coding_runtime, "get_coding_service", lambda: service)

    @asynccontextmanager
    async def two_threadpool_tokens(_app: FastAPI):
        anyio.to_thread.current_default_thread_limiter().total_tokens = 2
        yield

    app = FastAPI(lifespan=two_threadpool_tokens)
    app.include_router(coding_runtime.router, prefix="/api/v1/coding/runtime")
    app.include_router(health.router, prefix="/api/v1/health")
    lease_statuses: list[int] = []

    with TestClient(app) as client:
        def poll_lease() -> None:
            response = client.post(
                f"/api/v1/coding/runtime/computers/{computer.id}/lease",
                headers={"Authorization": f"Bearer {runtime_token}"},
                json={"protocol_version": "1.0", "wait_seconds": 2},
            )
            lease_statuses.append(response.status_code)

        pollers = [threading.Thread(target=poll_lease, name=f"lease-long-poll-{index}") for index in range(4)]
        for poller in pollers:
            poller.start()
        for _ in range(2):
            assert polls.acquire(timeout=5)
        started = time.monotonic()
        health_response = client.get("/api/v1/health/")
        health_elapsed = time.monotonic() - started
        for poller in pollers:
            poller.join(timeout=5)

    assert health_response.status_code == 200
    assert health_elapsed < 0.5
    assert lease_statuses == [200, 200, 200, 200]


def test_lease_authentication_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = service_with(tmp_path, FakeEngine())
    computer, runtime_token = service.control.register_runtime(
        service.control.issue_registration_token(),
        "Build computer",
        ComputerCapabilities(platform="linux", architecture="test", runtime_version="test-runtime"),
    )
    authenticate = service.control.authenticate_runtime
    on_event_loop: list[bool] = []

    def recording_authenticate(computer_id: str, token: str):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_event_loop.append(False)
        else:
            on_event_loop.append(True)
        return authenticate(computer_id, token)

    monkeypatch.setattr(service.control, "authenticate_runtime", recording_authenticate)
    monkeypatch.setattr(coding_runtime, "get_coding_service", lambda: service)
    app = FastAPI()
    app.include_router(coding_runtime.router, prefix="/api/v1/coding/runtime")

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/coding/runtime/computers/{computer.id}/lease",
            headers={"Authorization": f"Bearer {runtime_token}"},
            json={"protocol_version": "1.0", "wait_seconds": 0},
        )

    assert response.status_code == 200
    assert on_event_loop == [False]


def test_legacy_computer_id_in_registration_body_never_touches_an_existing_computer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = service_with(tmp_path, FakeEngine())
    control = service.control
    capabilities = ComputerCapabilities(
        platform="linux",
        architecture="test",
        runtime_version="test-runtime",
        protocol_versions=["1.0"],
    )
    existing, existing_token = control.register_runtime(control.issue_registration_token(), "Existing", capabilities)
    monkeypatch.setattr(coding_runtime, "get_coding_service", lambda: service)
    app = FastAPI()
    app.include_router(coding_runtime.router, prefix="/api/v1/coding/runtime")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/coding/runtime/register",
            json={
                "protocol_version": "1.0",
                "registration_token": control.issue_registration_token(),
                "computer_id": existing.id,
                "name": "Imposter",
                "capabilities": capabilities.model_dump(mode="json"),
            },
        )

    assert response.status_code == 200
    registered = response.json()["computer"]
    assert registered["id"] != existing.id
    assert registered["name"] == "Imposter"
    current = control.authenticate_runtime(existing.id, existing_token)
    assert current.name == existing.name
    assert current.registration_epoch == existing.registration_epoch
    assert {existing.id, registered["id"]} <= {item.id for item in control.list_computers().items}
