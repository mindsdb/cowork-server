from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from coding_service_fakes import CREDS, FakeEngine, repository, service_with

from cowork.coding.contracts import SessionCreateRequest, SessionStatus, TaskCapability
from cowork.coding.control_models import ComputerCapabilities, RuntimeCommand, RuntimeEvent
from cowork.coding.engines.registry import CodingEngineRegistry
from cowork.coding.project_models import ProjectCreateRequest, RepositoryResource
from cowork.coding.runtime_client import RuntimeIdentity
from cowork.coding.runtime_protocol import RuntimeLease
from cowork.coding.runtime_worker import CodeOnlyRuntime


class ControlPlaneRuntimeClient:
    """Exercise the runtime protocol in-process without bypassing its fences."""

    def __init__(self, service, computer_id: str) -> None:
        self.service = service
        self.server_url = "https://control.example.test"
        self.identity = RuntimeIdentity(computer_id, "runtime-secret", "Build computer")
        self._sequences: dict[str, int] = {}

    def heartbeat(self, active_run_count: int = 0) -> None:
        self.service.control.heartbeat(self.identity.computer_id, active_run_count)

    def lease(self, wait_seconds: float = 0) -> RuntimeLease | None:
        return self.service.remote.acquire_lease(self.identity.computer_id)

    def event(self, lease: RuntimeLease, kind: str, payload: dict[str, object] | None = None) -> None:
        sequence = self._sequences.get(lease.run.id, lease.run.last_event_seq) + 1
        self.deliver(RuntimeEvent(
            run_id=lease.run.id,
            computer_id=self.identity.computer_id,
            lease_id=lease.lease_id,
            epoch=lease.run.epoch,
            seq=sequence,
            kind=kind,
            payload=payload or {},
        ))
        self._sequences[lease.run.id] = sequence

    def deliver(self, event: RuntimeEvent) -> None:
        self.service.accept_runtime_event(event)

    def commands(self, lease: RuntimeLease) -> list[RuntimeCommand]:
        return self.service.control.claim_commands(
            lease.run.id,
            self.identity.computer_id,
            lease.lease_id,
            lease.run.epoch,
        )

    def acknowledge(
        self,
        lease: RuntimeLease,
        command: RuntimeCommand,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        self.service.control.acknowledge_command(
            lease.run.id,
            command.id,
            self.identity.computer_id,
            lease.lease_id,
            lease.run.epoch,
            result,
            error,
        )

    def inference_endpoint(self, lease: RuntimeLease) -> str:
        return f"{self.server_url}/api/v1/coding/runtime/runs/{lease.run.id}/inference"


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def connected_task(tmp_path: Path, engine: FakeEngine):
    """Create one portable task assigned to a connected computer running ``engine``."""
    source = repository(tmp_path)
    service = service_with(tmp_path, engine)
    registration = service.control.issue_registration_token()
    computer, _ = service.control.register_runtime(
        registration,
        "Build computer",
        ComputerCapabilities(
            platform="linux",
            architecture="test",
            runtime_version="test-runtime",
            agent_engines=["fake"],
            shells=["bash"],
            task_capabilities=[
                TaskCapability.review,
                TaskCapability.terminal,
                TaskCapability.project_actions,
                TaskCapability.slash_commands,
            ],
        ),
    )
    project = service.projects.create(ProjectCreateRequest(
        name="Portable project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url=str(source),
            local_path=str(source),
        )],
    ))
    task = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            computer_id=computer.id,
            prompt="Build remotely",
            engine_id="fake",
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    return service, computer, task


def worker_for(tmp_path: Path, engine: FakeEngine, client: ControlPlaneRuntimeClient) -> CodeOnlyRuntime:
    registry = CodingEngineRegistry()
    registry.register(engine)
    return CodeOnlyRuntime(tmp_path / "remote-runtime", client, registry, heartbeat_interval_seconds=0.01)


def test_connected_computer_lifecycle_runs_and_releases_end_to_end(tmp_path: Path) -> None:
    engine = FakeEngine()
    service, computer, task = connected_task(tmp_path, engine)
    runtime = worker_for(tmp_path, engine, ControlPlaneRuntimeClient(service, computer.id))
    worker = threading.Thread(target=runtime.run_once, name="remote-runtime-test", daemon=True)
    worker.start()

    wait_until(lambda: service.get_session(task.id).status == SessionStatus.completed)
    assert engine.prompts == ["Build remotely"]
    assert Path(engine.opened_workspaces[0]).is_relative_to(tmp_path / "remote-runtime")
    assert service.get_session(task.id).task_capabilities.files is False
    assert service.control.store.get_task(task.id).execution_project is not None

    event_count = service.get_session(task.id).event_count
    service.submit_turn(task.id, "/review", CREDS)
    wait_until(lambda: engine.reviews == 1)
    wait_until(lambda: service.get_session(task.id).event_count > event_count)
    wait_until(lambda: service.get_session(task.id).status == SessionStatus.completed)

    service.remote.release_workspace(service.get_session(task.id))
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert service.get_session(task.id).run_status == "completed"


def test_worker_redelivery_applies_a_workspace_event_whose_first_application_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine()
    service, computer, task = connected_task(tmp_path, engine)
    failures: list[int] = []
    accept_workspace = service.remote._accept_workspace

    def flaky_accept_workspace(run, event):
        if not failures:
            failures.append(event.seq)
            raise RuntimeError("workspace read model unavailable")
        accept_workspace(run, event)

    monkeypatch.setattr(service.remote, "_accept_workspace", flaky_accept_workspace)

    class RedeliveringClient(ControlPlaneRuntimeClient):
        """Mirror RemoteRuntimeClient: the same event id is resent after a server failure."""

        def deliver(self, event: RuntimeEvent) -> None:
            try:
                super().deliver(event)
            except RuntimeError:
                super().deliver(event)

    runtime = worker_for(tmp_path, engine, RedeliveringClient(service, computer.id))
    worker = threading.Thread(target=runtime.run_once, name="remote-runtime-redelivery", daemon=True)
    worker.start()
    wait_until(lambda: service.get_session(task.id).status == SessionStatus.completed)
    service.remote.release_workspace(service.get_session(task.id))
    worker.join(timeout=5)

    assert failures == [1]
    session = service.get_session(task.id)
    assert Path(session.workspace_path).is_relative_to(tmp_path / "remote-runtime")
    assert [item.title for item in service.events(task.id).items].count("Workspace prepared") == 1
    assert engine.prompts == ["Build remotely"]
    assert session.run_status == "completed"
