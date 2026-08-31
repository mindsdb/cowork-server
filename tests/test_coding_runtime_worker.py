from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

from coding_service_fakes import FakeEngine, repository

from cowork.coding.contracts import PermissionMode
from cowork.coding.control_models import CodeTask, RunStatus, RuntimeCommand, TaskRun
from cowork.coding.engines.registry import CodingEngineRegistry
from cowork.coding.project_models import CodeProject, RepositoryResource
from cowork.coding.runtime_protocol import RuntimeExecutionConfig, RuntimeLease
from cowork.coding.runtime_worker import (
    CodeOnlyRuntime,
    RuntimeIdentity,
    RuntimeClientError,
    load_runtime_identity,
    run_runtime_forever,
    save_runtime_identity,
)


class FakeRuntimeClient:
    def __init__(
        self,
        lease: RuntimeLease,
        command: RuntimeCommand,
        after_turn: list[RuntimeCommand] | None = None,
    ) -> None:
        self.server_url = "https://control.example.test"
        self.identity = RuntimeIdentity("remote-computer", "runtime-secret", "Remote")
        self._lease = lease
        self._commands = [command]
        self._after_turn = after_turn or []
        self.events: list[tuple[str, dict[str, object]]] = []
        self.acknowledged: list[str] = []
        self.results: dict[str, tuple[dict[str, object] | None, str | None]] = {}
        self.workspace_readmes: list[str] = []
        self.heartbeats = 0
        self.heartbeat_counts: list[int] = []
        self.calls: list[str] = []

    def heartbeat(self, active_run_count: int = 0) -> None:
        self.heartbeats += 1
        self.heartbeat_counts.append(active_run_count)

    def lease(self, wait_seconds: float = 0) -> RuntimeLease | None:
        lease, self._lease = self._lease, None
        return lease

    def event(self, _lease: RuntimeLease, kind: str, payload: dict[str, object] | None = None) -> None:
        self.events.append((kind, payload or {}))
        status = str((payload or {}).get("status") or "")
        self.calls.append(f"event:{kind}:{status}")
        if kind == "workspace":
            for item in (payload or {}).get("items", []):
                workspace_path = Path(str(item["workspace_path"]))
                self.workspace_readmes.append(
                    (workspace_path / "README.md").read_text(encoding="utf-8")
                )
        if kind == "turn_completed":
            self._commands.extend(self._after_turn)
            if not self._after_turn:
                self._queue_release(_lease)

    def commands(self, _lease: RuntimeLease) -> list[RuntimeCommand]:
        commands, self._commands = self._commands, []
        return commands

    def acknowledge(
        self,
        _lease: RuntimeLease,
        command: RuntimeCommand,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        self.acknowledged.append(command.id)
        self.calls.append(f"ack:{command.id}")
        self.results[command.id] = (result, error)
        if self._after_turn and all(item.id in self.acknowledged for item in self._after_turn):
            self._after_turn = []
            self._queue_release(_lease)

    def _queue_release(self, lease: RuntimeLease) -> None:
        if any(item.id == "command-release" for item in self._commands):
            return
        self._commands.append(RuntimeCommand(
            id="command-release",
            run_id=lease.run.id,
            epoch=lease.run.epoch,
            kind="release",
        ))

    def inference_endpoint(self, _lease: RuntimeLease) -> str:
        return "https://control.example.test/api/v1/coding/runtime/inference"


def test_runtime_identity_is_private_and_survives_restart(tmp_path: Path) -> None:
    identity = RuntimeIdentity("remote-computer", "runtime-secret", "Build computer")
    save_runtime_identity(tmp_path, "https://control.example.test/", identity)

    stored = load_runtime_identity(tmp_path)
    assert stored is not None
    assert stored.server_url == "https://control.example.test"
    assert stored.identity == identity
    assert (tmp_path / "runtime-identity.json").stat().st_mode & 0o777 == 0o600


def test_runtime_reconnects_with_bounded_backoff_after_transport_failure() -> None:
    class UnreachableRuntime:
        calls = 0

        def run_once(self, wait_seconds: float = 0) -> bool:
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("offline")
            raise RuntimeClientError("restarting", 503)

    delays: list[float] = []

    def stop_after_two_retries(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 2:
            raise StopIteration

    runtime = UnreachableRuntime()
    with pytest.raises(StopIteration):
        run_runtime_forever(runtime, sleep=stop_after_two_retries)

    assert runtime.calls == 2
    assert delays == [1.0, 2.0]


def test_runtime_does_not_retry_revoked_credentials() -> None:
    class RevokedRuntime:
        def run_once(self, wait_seconds: float = 0) -> bool:
            raise RuntimeClientError("revoked", 401)

    sleep = pytest.fail
    with pytest.raises(RuntimeClientError, match="revoked"):
        run_runtime_forever(RevokedRuntime(), sleep=sleep)


def test_code_only_runtime_prepares_a_portable_repo_and_runs_the_agent(tmp_path: Path) -> None:
    source = repository(tmp_path)
    project = CodeProject(
        id="portable-project",
        name="Portable project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url=str(source),
        )],
    )
    run = TaskRun(
        id="run-portable",
        task_id="task-portable",
        computer_id="remote-computer",
        status=RunStatus.preparing,
        lease_id="lease-portable",
    )
    lease = RuntimeLease(
        task=CodeTask(
            id="task-portable",
            title="Portable task",
            prompt="Build from the portable repository",
        ),
        run=run,
        lease_id="lease-portable",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=project,
        execution=RuntimeExecutionConfig(
            engine_id="fake",
            model="fake-model",
            permission_mode=PermissionMode.workspace,
        ),
    )
    command = RuntimeCommand(
        id="command-start",
        run_id=run.id,
        epoch=run.epoch,
        kind="start",
        payload={"prompt": "Implement the change"},
    )
    client = FakeRuntimeClient(lease, command)
    engine = FakeEngine()
    registry = CodingEngineRegistry()
    registry.register(engine)

    runtime = CodeOnlyRuntime(tmp_path / "runtime", client, registry)
    assert runtime.run_once()

    assert engine.prompts == ["Implement the change"]
    assert client.workspace_readmes == ["base\n"]
    assert client.acknowledged == [command.id, "command-release"]
    assert client.calls.index("ack:command-release") < client.calls.index("event:status:completed")
    statuses = [payload.get("status") for kind, payload in client.events if kind == "status"]
    assert statuses == ["ready", "running", "completed"]
    assert any(
        kind == "event" and (payload.get("event") or {}).get("text") == "done"
        for kind, payload in client.events
    )


def test_code_only_runtime_keeps_heartbeating_during_a_blocking_run(tmp_path: Path) -> None:
    run = TaskRun(
        id="run-heartbeat",
        task_id="task-heartbeat",
        computer_id="remote-computer",
        status=RunStatus.preparing,
        lease_id="lease-heartbeat",
    )
    lease = RuntimeLease(
        task=CodeTask(id="task-heartbeat", title="Heartbeat task", prompt="Build"),
        run=run,
        lease_id="lease-heartbeat",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=CodeProject(
            id="heartbeat-project",
            name="Heartbeat project",
            resources=[RepositoryResource(
                id="repo",
                name="Repo",
                source_url="https://example.test/repo.git",
            )],
        ),
        execution=RuntimeExecutionConfig(
            engine_id="fake",
            model="fake-model",
            permission_mode=PermissionMode.workspace,
        ),
    )
    client = FakeRuntimeClient(
        lease,
        RuntimeCommand(
            id="command-start",
            run_id=run.id,
            epoch=run.epoch,
            kind="start",
        ),
    )

    class BlockingRuntime(CodeOnlyRuntime):
        def _execute(self, _lease: RuntimeLease) -> None:
            deadline = time.monotonic() + 1
            while client.heartbeat_counts.count(1) < 3 and time.monotonic() < deadline:
                time.sleep(0.005)

    runtime = BlockingRuntime(
        tmp_path / "runtime",
        client,
        heartbeat_interval_seconds=0.01,
    )
    assert runtime.run_once()

    assert client.heartbeat_counts[0] == 0
    assert client.heartbeat_counts.count(1) >= 3


def test_code_only_runtime_serves_git_and_terminal_operations_after_a_turn(tmp_path: Path) -> None:
    source = repository(tmp_path)
    project = CodeProject(
        id="operations-project",
        name="Operations project",
        resources=[RepositoryResource(id="repo", name="Repo", source_url=str(source))],
    )
    run = TaskRun(
        id="run-operations",
        task_id="task-operations",
        computer_id="remote-computer",
        status=RunStatus.preparing,
        lease_id="lease-operations",
    )
    lease = RuntimeLease(
        task=CodeTask(id="task-operations", title="Operations task", prompt="Build"),
        run=run,
        lease_id="lease-operations",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=project,
        execution=RuntimeExecutionConfig(
            engine_id="fake",
            model="fake-model",
            permission_mode=PermissionMode.workspace,
        ),
    )
    operations = [
        RuntimeCommand(
            id="operation-git",
            run_id=run.id,
            epoch=run.epoch,
            kind="operation",
            payload={"operation": "git_states"},
        ),
        RuntimeCommand(
            id="operation-terminal",
            run_id=run.id,
            epoch=run.epoch,
            kind="operation",
            payload={
                "operation": "terminal_start",
                "terminal_id": "terminal-1",
                "cols": 100,
                "rows": 24,
                "shell": "auto",
            },
        ),
    ]
    client = FakeRuntimeClient(
        lease,
        RuntimeCommand(
            id="command-start",
            run_id=run.id,
            epoch=run.epoch,
            kind="start",
            payload={"prompt": "Build"},
        ),
        operations,
    )
    engine = FakeEngine()
    registry = CodingEngineRegistry()
    registry.register(engine)

    assert CodeOnlyRuntime(tmp_path / "runtime", client, registry).run_once()

    git_result, git_error = client.results["operation-git"]
    terminal_result, terminal_error = client.results["operation-terminal"]
    assert git_error is None
    assert git_result and len(git_result["items"]) == 1
    assert terminal_error is None
    assert terminal_result and terminal_result["status"] == "running"
    assert engine.terminal_process_ids


def test_code_only_runtime_reuses_the_workspace_for_follow_up_turns(tmp_path: Path) -> None:
    source = repository(tmp_path)
    project = CodeProject(
        id="follow-up-project",
        name="Follow-up project",
        resources=[RepositoryResource(id="repo", name="Repo", source_url=str(source))],
    )
    run = TaskRun(
        id="run-follow-up",
        task_id="task-follow-up",
        computer_id="remote-computer",
        status=RunStatus.preparing,
        lease_id="lease-follow-up",
    )
    lease = RuntimeLease(
        task=CodeTask(id="task-follow-up", title="Follow-up task", prompt="First turn"),
        run=run,
        lease_id="lease-follow-up",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=project,
        execution=RuntimeExecutionConfig(
            engine_id="fake",
            model="fake-model",
            permission_mode=PermissionMode.workspace,
        ),
    )
    first = RuntimeCommand(
        id="command-first",
        run_id=run.id,
        epoch=run.epoch,
        kind="start",
        payload={"prompt": "First turn"},
    )
    follow_up = RuntimeCommand(
        id="command-follow-up",
        run_id=run.id,
        epoch=run.epoch,
        kind="start",
        payload={
            "prompt": "/review",
            "engine_prompt": "/review",
            "command": "review",
            "goal_action": "",
            "goal_objective": None,
        },
    )
    client = FakeRuntimeClient(lease, first, [follow_up])
    engine = FakeEngine()
    registry = CodingEngineRegistry()
    registry.register(engine)

    assert CodeOnlyRuntime(tmp_path / "runtime", client, registry).run_once()

    assert engine.prompts == ["First turn"]
    assert engine.reviews == 1
    assert engine.existing_ids == [None]
    assert client.acknowledged == [first.id, follow_up.id, "command-release"]
    assert len([item for item in client.events if item[0] == "workspace"]) == 1


def test_code_only_runtime_routes_steering_and_cancellation_without_losing_claimed_commands(
    tmp_path: Path,
) -> None:
    source = repository(tmp_path)
    project = CodeProject(
        id="steer-project",
        name="Steer project",
        resources=[RepositoryResource(id="repo", name="Repo", source_url=str(source))],
    )
    run = TaskRun(
        id="run-steer",
        task_id="task-steer",
        computer_id="remote-computer",
        status=RunStatus.preparing,
        lease_id="lease-steer",
    )
    lease = RuntimeLease(
        task=CodeTask(id="task-steer", title="Steer task", prompt="Start"),
        run=run,
        lease_id="lease-steer",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=project,
        execution=RuntimeExecutionConfig(
            engine_id="fake",
            model="fake-model",
            permission_mode=PermissionMode.workspace,
        ),
    )
    start = RuntimeCommand(
        id="command-start",
        run_id=run.id,
        epoch=run.epoch,
        kind="start",
        payload={"prompt": "Start"},
    )
    steer = RuntimeCommand(
        id="command-steer",
        run_id=run.id,
        epoch=run.epoch,
        kind="steer",
        payload={"prompt": "Change direction"},
    )
    status = RuntimeCommand(
        id="command-status",
        run_id=run.id,
        epoch=run.epoch,
        kind="agent_command",
        payload={
            "prompt": "/status",
            "engine_prompt": "/status",
            "command": "status",
            "goal_action": "",
            "goal_objective": None,
        },
    )
    cancel = RuntimeCommand(
        id="command-cancel",
        run_id=run.id,
        epoch=run.epoch,
        kind="cancel",
    )
    client = FakeRuntimeClient(lease, start)
    client._commands.extend([steer, status, cancel])
    engine = FakeEngine(block_until_cancel=True)
    registry = CodingEngineRegistry()
    registry.register(engine)

    assert CodeOnlyRuntime(tmp_path / "runtime", client, registry).run_once()

    assert engine.steers == [("turn-1", "Change direction")]
    assert engine.cancels == ["turn-1"]
    assert {start.id, steer.id, status.id, cancel.id, "command-release"}.issubset(client.acknowledged)
    assert any(
        kind == "event" and (payload.get("event") or {}).get("title") == "Task status"
        for kind, payload in client.events
    )
    assert ("turn_completed", {"status": "cancelled"}) in client.events
