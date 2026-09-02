from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from coding_service_fakes import FakeEngine, FakeSession, repository

from cowork.coding.contracts import PermissionMode
from cowork.coding.control_models import (
    CodeTask,
    Computer,
    ComputerCapabilities,
    ExecutionWorkspace,
    RunStatus,
    RuntimeCommand,
    TaskRun,
)
from cowork.coding.engines.registry import CodingEngineRegistry
from cowork.coding.project_models import CodeProject, ProjectCommand, RepositoryResource
from cowork.coding.runtime_protocol import (
    ComputerRegistrationResponse,
    RuntimeExecutionConfig,
    RuntimeLease,
)
from cowork.coding import runtime_worker
from cowork.coding.runtime_worker import (
    CodeOnlyRuntime,
    RemoteRuntimeClient,
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
        self.mid_turn: list[RuntimeCommand] = []
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
        if kind == "status" and status == "running":
            self._commands.extend(self.mid_turn)
            self.mid_turn = []
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
        if command.run_id != _lease.run.id:
            raise RuntimeClientError("Runtime command not found", 404)
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


class FakeControlPlane:
    """Loopback HTTP stand-in for the runtime protocol's registration and idle loop."""

    RUNTIME_TOKEN = "runtime-token-issued-by-the-fake-control-plane"

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, object]]] = []
        self._changed = threading.Condition()
        control_plane = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server naming
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw) if raw else {}
                with control_plane._changed:
                    control_plane.requests.append((
                        self.path,
                        self.headers.get("Authorization") or "",
                        body,
                    ))
                    control_plane._changed.notify_all()
                if self.path.endswith("/register"):
                    payload = ComputerRegistrationResponse(
                        computer=Computer(
                            id="computer-subprocess",
                            name=str(body["name"]),
                            capabilities=ComputerCapabilities.model_validate(body["capabilities"]),
                        ),
                        runtime_token=control_plane.RUNTIME_TOKEN,
                    ).model_dump(mode="json")
                elif self.path.endswith("/lease"):
                    payload = None
                else:
                    payload = {}
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def wait_for(self, path_suffix: str, timeout: float, still_running=lambda: True) -> tuple[str, str, dict[str, object]]:
        deadline = time.monotonic() + timeout
        with self._changed:
            while True:
                match = next((item for item in self.requests if item[0].endswith(path_suffix)), None)
                if match is not None:
                    return match
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not still_running():
                    raise AssertionError(f"No request to {path_suffix!r}; saw {[item[0] for item in self.requests]}")
                self._changed.wait(min(remaining, 0.1))

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def test_runtime_cli_registers_a_computer_with_the_control_plane(tmp_path: Path) -> None:
    control_plane = FakeControlPlane()
    registration_token = "registration-token-with-at-least-32-chars"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cowork.coding.runtime_worker",
            "--server",
            control_plane.url,
            "--code",
            registration_token,
            "--name",
            "Subprocess computer",
            "--root",
            str(tmp_path / "root"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        try:
            _, _, registration = control_plane.wait_for(
                "/runtime/register",
                timeout=60,
                still_running=lambda: process.poll() is None,
            )
            _, heartbeat_auth, _ = control_plane.wait_for(
                "/heartbeat",
                timeout=30,
                still_running=lambda: process.poll() is None,
            )
        except AssertionError as exc:
            process.kill()
            _, stderr = process.communicate(timeout=10)
            raise AssertionError(f"{exc}\nworker stderr:\n{stderr}") from exc
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=10)
        control_plane.close()

    assert registration["registration_token"] == registration_token
    assert registration["name"] == "Subprocess computer"
    assert "computer_id" not in registration
    assert heartbeat_auth == f"Bearer {FakeControlPlane.RUNTIME_TOKEN}"
    stored = load_runtime_identity(tmp_path / "root")
    assert stored is not None
    assert stored.identity.computer_id == "computer-subprocess"


def test_runtime_identity_is_private_and_survives_restart(tmp_path: Path) -> None:
    identity = RuntimeIdentity("remote-computer", "runtime-secret", "Build computer")
    save_runtime_identity(tmp_path, "https://control.example.test/", identity)

    stored = load_runtime_identity(tmp_path)
    assert stored is not None
    assert stored.server_url == "https://control.example.test"
    assert stored.identity == identity
    assert (tmp_path / "runtime-identity.json").stat().st_mode & 0o777 == 0o600


def test_runtime_identity_is_created_owner_only_without_a_chmod(monkeypatch, tmp_path: Path) -> None:
    created: dict[str, tuple[int, int]] = {}
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *args, **kwargs):
        created[os.fspath(path)] = (flags, mode)
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "chmod", lambda *_args, **_kwargs: pytest.fail("mode must be set at creation"))
    monkeypatch.setattr(Path, "chmod", lambda *_args, **_kwargs: pytest.fail("mode must be set at creation"))

    save_runtime_identity(
        tmp_path,
        "https://control.example.test",
        RuntimeIdentity("remote-computer", "runtime-secret", "Build computer"),
    )

    ((flags, mode),) = [item for path, item in created.items() if path.startswith(str(tmp_path))]
    assert mode == 0o600
    assert flags & os.O_EXCL
    assert (tmp_path / "runtime-identity.json").stat().st_mode & 0o777 == 0o600


def test_runtime_client_redelivers_the_same_event_after_a_transient_failure() -> None:
    lease, _ = _lease_only("redelivery")
    statuses = iter([500, 200])
    posted: list[dict[str, object]] = []

    def control_plane(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(next(statuses), json={})

    delays: list[float] = []
    client = RemoteRuntimeClient(
        "https://control.example.test",
        RuntimeIdentity("remote-computer", "runtime-secret", "Remote"),
        httpx.Client(transport=httpx.MockTransport(control_plane)),
        sleep=delays.append,
    )

    client.event(lease, "status", {"status": "ready"})

    assert [(item["seq"], item["kind"]) for item in posted] == [(1, "status"), (1, "status")]
    assert posted[0]["id"] == posted[1]["id"]
    assert delays == [0.5]


def test_runtime_client_gives_up_redelivery_on_a_rejected_event() -> None:
    lease, _ = _lease_only("rejected")
    attempts: list[int] = []

    def control_plane(request: httpx.Request) -> httpx.Response:
        attempts.append(json.loads(request.content)["seq"])
        return httpx.Response(409, json={"detail": "Runtime event sequence is stale"})

    client = RemoteRuntimeClient(
        "https://control.example.test",
        RuntimeIdentity("remote-computer", "runtime-secret", "Remote"),
        httpx.Client(transport=httpx.MockTransport(control_plane)),
        sleep=lambda _delay: pytest.fail("a rejected event must not be redelivered"),
    )

    with pytest.raises(RuntimeClientError, match="stale") as raised:
        client.event(lease, "status", {"status": "ready"})

    assert raised.value.status_code == 409
    assert attempts == [1]


def test_runtime_client_advances_the_sequence_after_an_event_fails_every_delivery() -> None:
    lease, _ = _lease_only("gap")
    applied: list[tuple[int, str]] = []
    responses: list[int] = []

    def control_plane_whose_responses_were_lost(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["kind"] == "status":
            if not applied:
                applied.append((body["seq"], body["id"]))
            raise httpx.ConnectError("response lost")
        if body["seq"] <= applied[-1][0]:
            responses.append(409)
            return httpx.Response(409, json={"detail": "Runtime event sequence is stale"})
        applied.append((body["seq"], body["id"]))
        responses.append(200)
        return httpx.Response(200, json={})

    client = RemoteRuntimeClient(
        "https://control.example.test",
        RuntimeIdentity("remote-computer", "runtime-secret", "Remote"),
        httpx.Client(transport=httpx.MockTransport(control_plane_whose_responses_were_lost)),
        sleep=lambda _delay: None,
    )

    with pytest.raises(httpx.TransportError):
        client.event(lease, "status", {"status": "ready"})
    client.event(lease, "checkpoint", {"waiting": "start"})

    assert [seq for seq, _ in applied] == [1, 2]
    assert responses == [200]


def _lease_only(name: str) -> tuple[RuntimeLease, RuntimeCommand]:
    run = TaskRun(id=f"run-{name}", task_id=f"task-{name}", computer_id="remote-computer", lease_id=f"lease-{name}")
    lease = RuntimeLease(
        task=CodeTask(id=run.task_id, title=name, prompt="Start"),
        run=run,
        lease_id=f"lease-{name}",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        execution=RuntimeExecutionConfig(engine_id="fake", model="fake-model", permission_mode=PermissionMode.workspace),
    )
    return lease, RuntimeCommand(id="command-start", run_id=run.id, epoch=run.epoch, kind="start")


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


def test_code_only_runtime_runs_project_setup_before_the_agent(tmp_path: Path) -> None:
    source = repository(tmp_path)
    project = CodeProject(
        id="setup-project",
        name="Setup project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url=str(source),
            commands=[ProjectCommand(
                id="setup",
                label="Prepare dependencies",
                argv=["git", "status", "--short"],
                phase="setup",
            )],
        )],
    )
    run = TaskRun(
        id="run-setup",
        task_id="task-setup",
        computer_id="remote-computer",
        status=RunStatus.preparing,
        lease_id="lease-setup",
    )
    lease = RuntimeLease(
        task=CodeTask(id="task-setup", title="Setup task", prompt="Build"),
        run=run,
        lease_id="lease-setup",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=project,
        execution=RuntimeExecutionConfig(
            engine_id="fake",
            model="fake-model",
            permission_mode=PermissionMode.workspace,
        ),
    )
    command = RuntimeCommand(id="command-setup", run_id=run.id, epoch=1, kind="start")
    client = FakeRuntimeClient(lease, command)
    registry = CodingEngineRegistry()
    registry.register(FakeEngine())

    assert CodeOnlyRuntime(tmp_path / "runtime", client, registry).run_once()

    setup_event = next(
        payload["event"]
        for kind, payload in client.events
        if kind == "event" and (payload.get("event") or {}).get("title") == "Prepare dependencies"
    )
    assert setup_event["phase"] == "completed"
    assert client.calls.index("event:event:") < client.calls.index("event:status:ready")


def test_recovery_on_another_computer_prepares_new_workspaces(tmp_path: Path) -> None:
    source = repository(tmp_path)
    project = CodeProject(
        id="recovery-project",
        name="Recovery project",
        resources=[RepositoryResource(id="repo", name="Repo", source_url=str(source))],
    )
    run = TaskRun(
        id="run-recovery",
        task_id="task-recovery",
        computer_id="remote-computer",
        status=RunStatus.recovering,
        lease_id="lease-recovery",
        epoch=2,
    )
    lease = RuntimeLease(
        task=CodeTask(id="task-recovery", title="Recovery task", prompt="Resume"),
        run=run,
        lease_id="lease-recovery",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=project,
        execution=RuntimeExecutionConfig(
            engine_id="fake",
            model="fake-model",
            permission_mode=PermissionMode.workspace,
        ),
        workspaces=[ExecutionWorkspace(
            id="workspace-recovery",
            run_id=run.id,
            resource_id="repo",
            computer_id="previous-computer",
            path="/workspace/that-only-existed-on-the-previous-computer",
        )],
    )
    command = RuntimeCommand(id="command-recovery", run_id=run.id, epoch=2, kind="start")
    client = FakeRuntimeClient(lease, command)
    registry = CodingEngineRegistry()
    registry.register(FakeEngine())

    assert CodeOnlyRuntime(tmp_path / "runtime", client, registry).run_once()
    assert client.workspace_readmes == ["base\n"]


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


def test_remote_approval_timeout_returns_the_run_to_running(tmp_path: Path) -> None:
    run = TaskRun(
        id="run-approval-timeout",
        task_id="task-approval-timeout",
        computer_id="remote-computer",
        status=RunStatus.running,
        lease_id="lease-approval-timeout",
    )
    lease = RuntimeLease(
        task=CodeTask(id=run.task_id, title="Approval timeout", prompt="Build"),
        run=run,
        lease_id=run.lease_id or "lease-approval-timeout",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=CodeProject(
            id="approval-project",
            name="Approval project",
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
        RuntimeCommand(id="command-start", run_id=run.id, epoch=run.epoch, kind="start"),
    )
    runtime = CodeOnlyRuntime(
        tmp_path / "runtime",
        client,
        approval_timeout_seconds=0,
    )

    assert runtime._approval(lease, "command", {"title": "Run tests"}) == {
        "decision": "decline"
    }
    statuses = [payload.get("status") for kind, payload in client.events if kind == "status"]
    assert statuses == ["awaiting_approval", "running"]


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
    client.mid_turn = [steer, status, cancel]
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


def _remote_lease(tmp_path: Path, name: str) -> tuple[RuntimeLease, RuntimeCommand]:
    source = repository(tmp_path)
    run = TaskRun(
        id=f"run-{name}",
        task_id=f"task-{name}",
        computer_id="remote-computer",
        status=RunStatus.preparing,
        lease_id=f"lease-{name}",
    )
    lease = RuntimeLease(
        task=CodeTask(id=run.task_id, title=name, prompt="Start"),
        run=run,
        lease_id=f"lease-{name}",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=CodeProject(
            id=f"{name}-project",
            name=f"{name} project",
            resources=[RepositoryResource(id="repo", name="Repo", source_url=str(source))],
        ),
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
    return lease, start


def test_command_router_reports_a_failing_handler_and_still_acts_on_cancel(tmp_path: Path) -> None:
    lease, start = _remote_lease(tmp_path, "failing-steer")
    steer = RuntimeCommand(
        id="command-steer",
        run_id=lease.run.id,
        epoch=lease.run.epoch,
        kind="steer",
        payload={"prompt": "Change direction"},
    )
    cancel = RuntimeCommand(id="command-cancel", run_id=lease.run.id, epoch=lease.run.epoch, kind="cancel")
    client = FakeRuntimeClient(lease, start)
    client.mid_turn = [steer, cancel]
    engine = FakeEngine(block_until_cancel=True)
    engine.steer_error = True
    registry = CodingEngineRegistry()
    registry.register(engine)

    assert CodeOnlyRuntime(tmp_path / "runtime", client, registry).run_once()

    assert engine.steers == []
    assert engine.cancels == ["turn-1"]
    assert client.results[steer.id] == (None, "adapter rejected steer")
    assert client.results[cancel.id] == (None, None)
    assert ("turn_completed", {"status": "cancelled"}) in client.events


def test_release_claimed_during_a_turn_is_acknowledged_after_the_turn(tmp_path: Path) -> None:
    lease, start = _remote_lease(tmp_path, "early-release")
    release = RuntimeCommand(
        id="command-release-early",
        run_id=lease.run.id,
        epoch=lease.run.epoch,
        kind="release",
    )
    cancel = RuntimeCommand(id="command-cancel", run_id=lease.run.id, epoch=lease.run.epoch, kind="cancel")
    client = FakeRuntimeClient(lease, start)
    client.mid_turn = [release, cancel]
    registry = CodingEngineRegistry()
    registry.register(FakeEngine(block_until_cancel=True))

    assert CodeOnlyRuntime(tmp_path / "runtime", client, registry).run_once()

    assert client.acknowledged == [start.id, cancel.id, release.id]
    assert client.calls.index("event:turn_completed:cancelled") < client.calls.index(f"ack:{release.id}")
    assert client.calls.index(f"ack:{release.id}") < client.calls.index("event:status:completed")


def test_stale_turn_commands_between_turns_are_rejected_not_applied_to_the_next_turn(tmp_path: Path) -> None:
    lease, start = _remote_lease(tmp_path, "stale-cancel")
    stale_cancel = RuntimeCommand(id="command-stale-cancel", run_id=lease.run.id, epoch=lease.run.epoch, kind="cancel")
    stale_steer = RuntimeCommand(
        id="command-stale-steer",
        run_id=lease.run.id,
        epoch=lease.run.epoch,
        kind="steer",
        payload={"prompt": "Too late"},
    )
    client = FakeRuntimeClient(lease, stale_cancel)
    client._commands.extend([stale_steer, start])
    engine = FakeEngine()
    registry = CodingEngineRegistry()
    registry.register(engine)

    assert CodeOnlyRuntime(tmp_path / "runtime", client, registry).run_once()

    assert client.results[stale_cancel.id] == (None, "There is no active turn")
    assert client.results[stale_steer.id] == (None, "There is no active turn")
    assert client.calls.index(f"ack:{stale_cancel.id}") < client.calls.index("event:status:running")
    assert engine.cancels == []
    assert engine.steers == []
    assert ("turn_completed", {"status": "completed"}) in client.events


def test_unknown_and_orphaned_commands_are_acknowledged_with_an_error(tmp_path: Path) -> None:
    lease, start = _remote_lease(tmp_path, "unknown-kinds")
    prepare = RuntimeCommand(id="command-prepare", run_id=lease.run.id, epoch=lease.run.epoch, kind="prepare")
    checkpoint = RuntimeCommand(id="command-checkpoint", run_id=lease.run.id, epoch=lease.run.epoch, kind="checkpoint")
    orphan_approve = RuntimeCommand(
        id="command-orphan-approve",
        run_id=lease.run.id,
        epoch=lease.run.epoch,
        kind="approve",
        payload={"approvalId": "approval-that-already-timed-out", "decision": "approve_once"},
    )
    cancel = RuntimeCommand(id="command-cancel", run_id=lease.run.id, epoch=lease.run.epoch, kind="cancel")
    client = FakeRuntimeClient(lease, prepare)
    client._commands.append(start)
    client.mid_turn = [checkpoint, orphan_approve, cancel]
    registry = CodingEngineRegistry()
    registry.register(FakeEngine(block_until_cancel=True))

    assert CodeOnlyRuntime(tmp_path / "runtime", client, registry).run_once()

    unsupported = "The selected computer does not support that command"
    assert client.results[prepare.id] == (None, unsupported)
    assert client.results[checkpoint.id] == (None, unsupported)
    assert client.results[orphan_approve.id] == (None, "That approval is no longer pending")
    assert client.results[cancel.id] == (None, None)


class SequentialLeaseClient(FakeRuntimeClient):
    """Hand out several leases in turn and only the commands addressed to the active run."""

    def __init__(
        self,
        leases: list[RuntimeLease],
        commands: list[RuntimeCommand],
        mid_turn: dict[str, list[RuntimeCommand]],
    ) -> None:
        super().__init__(leases[0], commands[0])
        self._leases = list(leases)
        self._commands = list(commands)
        self._mid_turn = mid_turn

    def lease(self, wait_seconds: float = 0) -> RuntimeLease | None:
        return self._leases.pop(0) if self._leases else None

    def commands(self, lease: RuntimeLease) -> list[RuntimeCommand]:
        matching = [item for item in self._commands if item.run_id == lease.run.id]
        self._commands = [item for item in self._commands if item.run_id != lease.run.id]
        return matching

    def event(self, lease: RuntimeLease, kind: str, payload: dict[str, object] | None = None) -> None:
        if kind == "status" and (payload or {}).get("status") == "running":
            self.mid_turn = self._mid_turn.pop(lease.run.id, [])
        super().event(lease, kind, payload)


def test_release_redelivered_during_a_long_turn_does_not_leak_into_the_next_lease(tmp_path: Path) -> None:
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first_lease, first_start = _remote_lease(tmp_path / "first", "first")
    second_lease, second_start = _remote_lease(tmp_path / "second", "second")
    first_start = first_start.model_copy(update={"id": "start-first", "payload": {"prompt": "Start first"}})
    second_start = second_start.model_copy(update={"id": "start-second", "payload": {"prompt": "Start second"}})
    release = RuntimeCommand(id="release-first", run_id=first_lease.run.id, epoch=first_lease.run.epoch, kind="release")
    redelivered = release.model_copy(update={"delivery_count": 2})
    cancel = RuntimeCommand(id="cancel-first", run_id=first_lease.run.id, epoch=first_lease.run.epoch, kind="cancel")
    client = SequentialLeaseClient(
        [first_lease, second_lease],
        [first_start, second_start],
        {first_lease.run.id: [release, redelivered, cancel]},
    )
    engine = FakeEngine()
    registry = CodingEngineRegistry()
    registry.register(engine)
    runtime = CodeOnlyRuntime(tmp_path / "runtime", client, registry)

    engine.block_until_cancel = True
    assert runtime.run_once()
    engine.block_until_cancel = False
    assert runtime.run_once()

    assert engine.prompts == ["Start first", "Start second"]
    assert client.acknowledged.count(release.id) == 1
    assert client.acknowledged.index(second_start.id) > client.acknowledged.index(release.id)
    assert client.results["command-release"] == ({}, None)
    assert [payload.get("status") for kind, payload in client.events if kind == "status"].count("completed") == 2


def test_deferred_commands_survive_a_failed_claim(tmp_path: Path) -> None:
    lease, start = _remote_lease(tmp_path, "failed-claim")
    client = FakeRuntimeClient(lease, start)
    runtime = CodeOnlyRuntime(tmp_path / "runtime", client, CodingEngineRegistry())
    release = RuntimeCommand(id="command-release", run_id=lease.run.id, epoch=lease.run.epoch, kind="release")
    runtime._defer_command(release)
    claim = client.commands

    def failing_claim(_lease: RuntimeLease) -> list[RuntimeCommand]:
        client.commands = claim
        raise RuntimeClientError("restarting", 503)

    client.commands = failing_claim
    with pytest.raises(RuntimeClientError, match="restarting"):
        runtime._take_commands(lease)

    assert runtime._take_commands(lease) == [release, start]
    assert runtime._take_commands(lease) == []


def test_turn_completes_even_when_a_command_handler_hangs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lease, start = _remote_lease(tmp_path, "hanging-steer")
    steer = RuntimeCommand(
        id="command-hanging-steer",
        run_id=lease.run.id,
        epoch=lease.run.epoch,
        kind="steer",
        payload={"prompt": "Change direction"},
    )
    client = FakeRuntimeClient(lease, start)
    client.mid_turn = [steer]
    engine = FakeEngine()
    engine.block_until_release = True
    registry = CodingEngineRegistry()
    registry.register(engine)
    steer_started = threading.Event()
    release_steer = threading.Event()

    def hanging_steer(self: FakeSession, turn_id: str, prompt: str, attachments=()) -> None:
        steer_started.set()
        release_steer.wait(timeout=5)
        self.engine.steers.append((turn_id, prompt))

    monkeypatch.setattr(FakeSession, "steer", hanging_steer)
    monkeypatch.setattr(runtime_worker, "_COMMAND_ROUTER_JOIN_SECONDS", 0.05)
    runtime = CodeOnlyRuntime(tmp_path / "runtime", client, registry)
    worker = threading.Thread(target=runtime.run_once, daemon=True)
    worker.start()
    assert steer_started.wait(timeout=5)
    engine.release_events.set()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and ("turn_completed", {"status": "completed"}) not in client.events:
        time.sleep(0.01)
    assert ("turn_completed", {"status": "completed"}) in client.events
    assert steer.id not in client.acknowledged

    release_steer.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert engine.steers == [("turn-1", "Change direction")]
    assert steer.id in client.acknowledged


def test_router_emits_no_turn_checkpoint_after_the_turn_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lease, start = _remote_lease(tmp_path, "late-checkpoint")
    steer = RuntimeCommand(
        id="command-late-steer",
        run_id=lease.run.id,
        epoch=lease.run.epoch,
        kind="steer",
        payload={"prompt": "Change direction"},
    )
    client = FakeRuntimeClient(lease, start)
    client.mid_turn = [steer]
    engine = FakeEngine()
    engine.block_until_release = True
    registry = CodingEngineRegistry()
    registry.register(engine)
    steer_started = threading.Event()
    release_steer = threading.Event()

    def hanging_steer(self: FakeSession, turn_id: str, prompt: str, attachments=()) -> None:
        steer_started.set()
        release_steer.wait(timeout=5)
        self.engine.steers.append((turn_id, prompt))

    monkeypatch.setattr(FakeSession, "steer", hanging_steer)
    monkeypatch.setattr(runtime_worker, "_COMMAND_ROUTER_JOIN_SECONDS", 0.05)
    runtime = CodeOnlyRuntime(tmp_path / "runtime", client, registry)
    worker = threading.Thread(target=runtime.run_once, daemon=True)
    worker.start()
    assert steer_started.wait(timeout=5)
    engine.release_events.set()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and ("turn_completed", {"status": "completed"}) not in client.events:
        time.sleep(0.01)
    release_steer.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    completed_at = client.events.index(("turn_completed", {"status": "completed"}))
    assert ("checkpoint", {"activeTurn": "turn-1"}) not in client.events[completed_at:]
    assert steer.id in client.acknowledged


def test_approval_ids_are_unique_per_request(tmp_path: Path) -> None:
    lease, start = _remote_lease(tmp_path, "approval-ids")
    client = FakeRuntimeClient(lease, start)
    runtime = CodeOnlyRuntime(tmp_path / "runtime", client, approval_timeout_seconds=0)

    runtime._approval(lease, "command", {"title": "Run tests"})
    runtime._approval(lease, "command", {"title": "Run tests"})

    approval_ids = [str(payload["approvalId"]) for kind, payload in client.events if kind == "approval"]
    assert len(set(approval_ids)) == 2
    for approval_id in approval_ids:
        uuid.UUID(approval_id.removeprefix(f"approval-{lease.run.id}-"))
