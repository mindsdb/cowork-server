from __future__ import annotations

import base64
import importlib.util
import os
import re
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from cowork.coding.contracts import (
    CodingEvent,
    EngineCapabilities,
    EngineCommand,
    EventType,
    ExtensionInventory,
    RuntimePlatformStatus,
    TerminalShellPreference,
)
from cowork.coding.engines import codex_config, codex_events
from cowork.coding.engines.base import (
    ApprovalHandler,
    EngineCredentials,
    EngineInputReference,
    EngineSessionConfig,
    TerminalExitHandler,
    TerminalOutputHandler,
)
from cowork.coding.engines.codex_extensions import add_extension_response
from cowork.coding.processes import terminate_descendants
from cowork.coding.redaction import redact_text, sanitize
from cowork.common.settings.app_settings import get_app_settings

ADAPTER_VERSION = "1"

_EXPECTED_ACTIVE_TURN = re.compile(r"expected active turn id `([^`]+)`")


class CodexEngine:
    id = "codex"

    def capabilities(self) -> EngineCapabilities:
        available = importlib.util.find_spec("openai_codex") is not None
        return EngineCapabilities(
            id=self.id,
            label="Codex",
            adapter_version=ADAPTER_VERSION,
            available=available,
            reason=None if available else "The bundled Codex runtime is unavailable.",
            supports_steering=True,
            supports_approvals=True,
            supports_reasoning=True,
            supports_diff_events=True,
            supports_models=True,
            supports_terminal=True,
            commands=[
                EngineCommand(name="goal", label="Goal", description="View it alone, or set, edit, pause, resume, or clear a durable objective", argument_hint="set|edit|pause|resume|clear", action="goal"),
                EngineCommand(name="review", label="Review", description="Review the current working changes", action="turn"),
                EngineCommand(name="compact", label="Compact", description="Compact this task's Codex context", action="compact"),
                EngineCommand(name="status", label="Status", description="Show task, model, workspace, and goal status", action="status"),
                EngineCommand(name="permissions", label="Permissions", description="Change filesystem, approval, and network access", action="client", client_action="controls"),
                EngineCommand(name="model", label="Model", description="Choose the model for future turns", action="client", client_action="controls"),
                EngineCommand(name="reasoning", label="Reasoning", description="Choose reasoning effort and speed", action="client", client_action="controls"),
                EngineCommand(name="skills", label="Skills", description="Browse skills available to this task", action="client", client_action="skills"),
                EngineCommand(name="mcp", label="MCP", description="Inspect connected MCP servers and tools", action="client", client_action="mcp"),
                EngineCommand(name="init", label="Init", description="Create or improve AGENTS.md instructions", action="turn"),
                EngineCommand(name="fork", label="Fork", description="Fork this coding task from its current history", action="client", client_action="fork"),
                EngineCommand(name="processes", label="Processes", description="Show and control task-owned processes", action="client", client_action="terminal"),
            ],
        )

    def open_session(
        self,
        *,
        cowork_root: str,
        workspace: str,
        config: EngineSessionConfig,
        credentials: EngineCredentials,
        existing_session_id: str | None,
        approval_handler: ApprovalHandler,
    ) -> CodexEngineSession:
        return CodexEngineSession(
            cowork_root=Path(cowork_root),
            workspace=Path(workspace),
            config=config,
            credentials=credentials,
            existing_session_id=existing_session_id,
            approval_handler=approval_handler,
        )

    def discover_models(self, credentials: EngineCredentials) -> list[str]:
        if not credentials.minds_api_key:
            return []
        endpoint = codex_config.responses_base_url(credentials.minds_url)
        with httpx.Client(timeout=15.0) as client:
            response = client.get(
                f"{endpoint}/models",
                headers={"Authorization": f"Bearer {credentials.minds_api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        models: list[str] = []
        for row in payload.get("data", []) if isinstance(payload, dict) else []:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            if row.get("embedding") is True:
                continue
            # Discovery describes what the MindsHub Responses API can run, not
            # what the current wallet can start right now. Keep disabled rows
            # so Code can mirror Cowork's catalogue and explain "Needs
            # credits" instead of making paid models appear not to exist.
            models.append(row["id"])
        return models


class CodexEngineSession:
    def __init__(
        self,
        *,
        cowork_root: Path,
        workspace: Path,
        config: EngineSessionConfig,
        credentials: EngineCredentials,
        existing_session_id: str | None,
        approval_handler: ApprovalHandler,
    ) -> None:
        from openai_codex.client import CodexClient, CodexConfig

        if not credentials.minds_api_key:
            raise RuntimeError("MindsHub is not connected. Sign in or configure a MindsHub API key first.")
        # Thread IDs stored in CodingSession are only resumable from the Codex
        # home that owns their rollout files. Keep that home under Cowork's
        # persistent data root instead of inheriting the launching shell's
        # CODEX_HOME (or the user's unrelated Codex Desktop state).
        codex_home = codex_config.persistent_home(cowork_root)
        codex_home.mkdir(parents=True, exist_ok=True)
        self._config_path = codex_home / "config.toml"
        self._config_path.touch(exist_ok=True)
        endpoint = config.inference_base_url or codex_config.local_inference_base_url(get_app_settings().port)
        launch = codex_config.prepare_launch(config, workspace, endpoint)
        client_config = CodexConfig(
            cwd=str(workspace),
            # Codex talks only to Cowork's loopback inference proxy. Keep the
            # real credential in the server process, outside agent commands.
            env=codex_config.client_environment(
                codex_home,
                config.environment,
                config.inference_api_key or codex_config.LOCAL_PROXY_TOKEN,
            ),
            config_overrides=launch.config_overrides,
            client_name="mindshub_cowork",
            client_title="MindsHub Cowork",
        )
        self._client = CodexClient(config=client_config, approval_handler=approval_handler)
        self._workspace = workspace
        self._terminal_workspace = codex_config.terminal_workspace(
            cowork_root,
            config.session_id,
            workspace,
            config.workspace_label,
        )
        self._skill_roots = tuple(config.skill_roots)
        self._model = config.model
        self._reasoning_effort = config.reasoning_effort
        self._service_tier = config.service_tier
        self._personality = config.personality
        self._approval_policy = launch.approval_policy
        self._sandbox_policy = launch.sandbox_policy
        self._secrets = tuple(value for value in (credentials.minds_api_key,) if value)
        self._closed = threading.Event()
        self._cancel_watchdogs: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()
        self._terminal_handlers: dict[str, TerminalOutputHandler] = {}
        self._terminal_lock = threading.Lock()
        self._goal_states: dict[str, Any] = {}
        self._client.start()
        try:
            self._client.initialize()
            self._register_skill_roots()
            if existing_session_id:
                response = self._client.thread_resume(existing_session_id, launch.thread_params)
            else:
                response = self._client.thread_start(launch.thread_params)
        except Exception:
            self._client.close()
            raise
        self._session_id = response.thread.id
        self._notification_thread = threading.Thread(
            target=self._route_global_notifications,
            name=f"codex-notifications-{self._session_id[:8]}",
            daemon=True,
        )
        self._notification_thread.start()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_closed(self) -> bool:
        return self._closed.is_set()

    def start_turn(
        self,
        prompt: str,
        attachments: tuple[EngineInputReference, ...] = (),
    ) -> str:
        response = self._client.turn_start(
            self._session_id,
            codex_config.turn_input(prompt, attachments),
            params={
                "cwd": str(self._workspace),
                "model": self._model,
                "effort": self._reasoning_effort,
                "serviceTier": self._service_tier,
                "personality": self._personality,
                "approvalPolicy": self._approval_policy,
                "approvalsReviewer": "user",
                "sandboxPolicy": self._sandbox_policy,
                "summary": "concise",
            },
        )
        return response.turn.id

    def start_goal(self, objective: str) -> str:
        state, turn_id = self._client.start_goal_operation(self._session_id, objective)
        self._goal_states[turn_id] = state
        return turn_id

    def resume_goal(self) -> str:
        from openai_codex.generated.v2_all import ThreadGoalStatus

        goal = self.goal_status()
        if goal is None:
            raise RuntimeError("There is no goal to resume")
        status = codex_events.string(goal.get("status"))
        if status == "active":
            raise RuntimeError("The goal is already active")
        if status == "complete":
            raise RuntimeError("The goal is complete. Set a new objective to continue goal work")

        state = self._client.reserve_goal_operation(self._session_id)
        activated = False
        try:
            state.activate_turn_routing()
            self._client.thread_goal_set(self._session_id, status=ThreadGoalStatus.active)
            activated = True
            turn_id = state.wait_for_start(30.0)
            if not turn_id:
                raise RuntimeError("Timed out waiting for the resumed goal turn to start")
            self._goal_states[turn_id] = state
            return turn_id
        except BaseException:
            if activated:
                self._client.cancel_goal_operation(state)
            state.finish()
            self._client.unregister_goal_operation(state)
            raise

    def update_goal(self, action: str, objective: str | None = None) -> dict[str, Any] | None:
        if action == "clear":
            self._client.thread_goal_clear(self._session_id)
            return None
        current = self.goal_status()
        if current is None:
            raise RuntimeError("There is no goal to update")
        if action == "pause":
            self._client.pause_goal(self._session_id)
        elif action == "edit":
            if not objective:
                raise ValueError("Add the revised objective after /goal edit")
            self._client.thread_goal_set(self._session_id, objective=objective)
        else:
            raise ValueError(f"Unsupported goal action: {action}")
        return self.goal_status()

    def start_review(self) -> str:
        from openai_codex.generated.v2_all import ReviewStartResponse

        response = self._client.request(
            "review/start",
            {
                "threadId": self._session_id,
                "delivery": "inline",
                "target": {"type": "uncommittedChanges"},
            },
            response_model=ReviewStartResponse,
        )
        self._client.register_turn_notifications(response.turn.id)
        return response.turn.id

    def events(self, turn_id: str) -> Iterator[CodingEvent]:
        goal_state = self._goal_states.get(turn_id)
        if goal_state is not None:
            try:
                yield from self._goal_events(turn_id, goal_state)
            finally:
                self._finish_cancel_watchdog(turn_id)
            return
        try:
            while True:
                notification = self._client.next_turn_notification(turn_id)
                event = codex_events.map_codex_notification(notification.method, notification.payload)
                if event is not None:
                    event = codex_events.redact_event(event, self._secrets)
                    event.turn_id = event.turn_id or turn_id
                    yield event
                if notification.method == "turn/completed":
                    break
        finally:
            self._client.unregister_turn_notifications(turn_id)
            self._finish_cancel_watchdog(turn_id)

    def steer(
        self,
        turn_id: str,
        prompt: str,
        attachments: tuple[EngineInputReference, ...] = (),
    ) -> None:
        goal_state = self._goal_states.get(turn_id)
        expected_turn_id = goal_state.current_turn() if goal_state is not None else turn_id
        if not expected_turn_id:
            raise RuntimeError("The goal is between turns; queue this guidance for the next turn")
        turn_input = codex_config.turn_input(prompt, attachments)
        if goal_state is None:
            self._client.turn_steer(self._session_id, expected_turn_id, turn_input)
            return

        from openai_codex.errors import InvalidRequestError

        try:
            self._client.turn_steer(self._session_id, expected_turn_id, turn_input)
        except InvalidRequestError as exc:
            # A goal can roll into its next physical turn between reading the
            # routed state and sending guidance. Adopt the server's canonical
            # active id and retry once instead of presenting a spurious 500.
            match = _EXPECTED_ACTIVE_TURN.search(exc.message)
            active_turn_id = match.group(1) if match is not None else None
            if active_turn_id and active_turn_id != expected_turn_id:
                goal_state.resolve_active_turn(expected_turn_id, active_turn_id)
                try:
                    self._client.turn_steer(self._session_id, active_turn_id, turn_input)
                    return
                except InvalidRequestError as retry_exc:
                    raise RuntimeError(
                        "The goal advanced again before guidance arrived. Queue this instruction for the next turn."
                    ) from retry_exc
            raise RuntimeError(
                "The active goal could not accept guidance. Queue this instruction for the next turn."
            ) from exc

    def cancel(self, turn_id: str) -> None:
        # Arm the process-tree watchdog before the cooperative RPC. If the
        # app-server itself hangs while accepting the interrupt, Stop must
        # still converge instead of waiting forever to start its fallback.
        start_watchdog = False
        with self._cancel_lock:
            completed = self._cancel_watchdogs.get(turn_id)
            if completed is None:
                completed = threading.Event()
                self._cancel_watchdogs[turn_id] = completed
                start_watchdog = True
        if start_watchdog:
            threading.Thread(
                target=self._cancel_watchdog,
                args=(completed,),
                name="codex-cancel-watchdog",
                daemon=True,
            ).start()
        goal_state = self._goal_states.get(turn_id)
        if goal_state is not None:
            self._client.cancel_goal_operation(goal_state)
        else:
            self._client.turn_interrupt(self._session_id, turn_id)

    def compact(self) -> None:
        self._client.thread_compact(self._session_id)

    def goal_status(self) -> dict[str, Any] | None:
        from openai_codex.generated.v2_all import ThreadGoalGetResponse

        response = self._client.request(
            "thread/goal/get",
            {"threadId": self._session_id},
            response_model=ThreadGoalGetResponse,
        )
        if response.goal is None:
            return None
        dumped = response.goal.model_dump(by_alias=True, mode="json", exclude_none=True)
        return dumped if isinstance(dumped, dict) else None

    def extension_inventory(self) -> ExtensionInventory:
        from openai_codex.generated.v2_all import (
            AppsInstalledResponse,
            HooksListResponse,
            ListMcpServerStatusResponse,
            PluginInstalledResponse,
            SkillsListResponse,
        )

        inventory = ExtensionInventory(config_path=str(self._config_path))
        calls = (
            ("skills", "skills/list", {"cwds": [str(self._workspace)], "forceReload": True}, SkillsListResponse),
            ("mcp", "mcpServerStatus/list", {"threadId": self._session_id, "detail": "full"}, ListMcpServerStatusResponse),
            ("hooks", "hooks/list", {"cwds": [str(self._workspace)]}, HooksListResponse),
            ("apps", "app/installed", {"threadId": self._session_id, "forceRefresh": False}, AppsInstalledResponse),
            ("plugins", "plugin/installed", {"cwds": [str(self._workspace)]}, PluginInstalledResponse),
        )
        for kind, method, params, response_model in calls:
            try:
                response = self._client.request(method, params, response_model=response_model)
                add_extension_response(inventory, kind, response)
            except Exception as exc:  # noqa: BLE001 - one unavailable extension must not hide the rest.
                inventory.errors.append(f"{kind}: {redact_text(str(exc), self._secrets)[:1_000]}")
        return inventory

    def fork(self, workspace: str, additional_dirs: tuple[str, ...] = ()) -> str:
        sandbox_policy = dict(self._sandbox_policy)
        if sandbox_policy.get("type") == "workspaceWrite":
            sandbox_policy["writableRoots"] = [workspace, *additional_dirs]
        response = self._client.thread_fork(
            self._session_id,
            {
                "cwd": workspace,
                "model": self._model,
                "approvalPolicy": self._approval_policy,
                "approvalsReviewer": "user",
                "sandboxPolicy": sandbox_policy,
            },
        )
        return response.thread.id

    def platform_status(self) -> RuntimePlatformStatus:
        if os.name != "nt":
            return RuntimePlatformStatus(platform=sys.platform)
        from openai_codex.generated.v2_all import WindowsSandboxReadinessResponse

        response = self._client.request(
            "windowsSandbox/readiness",
            {},
            response_model=WindowsSandboxReadinessResponse,
        )
        return RuntimePlatformStatus(platform=sys.platform, windows_sandbox=codex_events.enum_value(response.status))

    def setup_windows_sandbox(self) -> RuntimePlatformStatus:
        if os.name != "nt":
            raise RuntimeError("Windows sandbox setup is only available on Windows")
        from openai_codex.generated.v2_all import WindowsSandboxSetupStartResponse

        response = self._client.request(
            "windowsSandbox/setupStart",
            {"mode": "elevated", "cwd": str(self._workspace)},
            response_model=WindowsSandboxSetupStartResponse,
        )
        status = self.platform_status()
        status.setup_started = response.started
        return status

    def _register_skill_roots(self) -> None:
        from openai_codex.generated.v2_all import SkillsExtraRootsSetResponse

        roots = [Path(root).expanduser() for root in getattr(self, "_skill_roots", [])]
        if not roots:
            # Backward compatibility for sessions created before task-scoped
            # skill snapshots existed. Keep the fallback inside Code's own
            # store; the general Cowork catalogue must never leak into Code.
            cowork_root = Path(get_app_settings().skill.root_dir).expanduser()
            code_root = cowork_root.parent / "code-skills"
            code_root.mkdir(parents=True, exist_ok=True)
            roots.append(code_root)
        user_root = codex_config.user_skills_root()
        if user_root.is_dir() and all(user_root.resolve() != root.resolve() for root in roots):
            roots.append(user_root)
        self._client.request(
            "skills/extraRoots/set",
            {"extraRoots": [str(root) for root in roots]},
            response_model=SkillsExtraRootsSetResponse,
        )

    def start_terminal(
        self,
        process_id: str,
        cols: int,
        rows: int,
        shell: TerminalShellPreference,
        output_handler: TerminalOutputHandler,
        exit_handler: TerminalExitHandler,
    ) -> None:
        if self.is_closed:
            raise RuntimeError("The coding runtime is no longer connected")
        with self._terminal_lock:
            if process_id in self._terminal_handlers:
                raise RuntimeError("Terminal process is already running")
            self._terminal_handlers[process_id] = output_handler
        worker = threading.Thread(
            target=self._run_terminal,
            args=(process_id, cols, rows, shell, exit_handler),
            name=f"codex-terminal-{process_id[:8]}",
            daemon=True,
        )
        try:
            worker.start()
        except Exception:
            with self._terminal_lock:
                self._terminal_handlers.pop(process_id, None)
            raise

    def write_terminal(self, process_id: str, data_base64: str) -> None:
        try:
            base64.b64decode(data_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Terminal input is not valid base64") from exc
        from openai_codex.generated.v2_all import CommandExecWriteResponse

        self._client.request(
            "command/exec/write",
            {"processId": process_id, "deltaBase64": data_base64},
            response_model=CommandExecWriteResponse,
        )

    def resize_terminal(self, process_id: str, cols: int, rows: int) -> None:
        from openai_codex.generated.v2_all import CommandExecResizeResponse

        self._client.request(
            "command/exec/resize",
            {"processId": process_id, "size": {"cols": cols, "rows": rows}},
            response_model=CommandExecResizeResponse,
        )

    def stop_terminal(self, process_id: str) -> None:
        from openai_codex.generated.v2_all import CommandExecTerminateResponse

        self._client.request(
            "command/exec/terminate",
            {"processId": process_id},
            response_model=CommandExecTerminateResponse,
        )

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        with self._cancel_lock:
            watchdogs = list(self._cancel_watchdogs.values())
            self._cancel_watchdogs.clear()
        for watchdog in watchdogs:
            watchdog.set()
        terminate_descendants(self._app_server_pid())
        self._client.close()

    def _cancel_watchdog(self, completed: threading.Event) -> None:
        if completed.wait(timeout=5.0) or self._closed.is_set():
            return
        # Interrupt is cooperative. If app-server or a child command does not
        # acknowledge it, tear down descendants so Stop cannot leave work
        # running invisibly after the UI reports cancellation.
        self.close()

    def _finish_cancel_watchdog(self, turn_id: str) -> None:
        with self._cancel_lock:
            watchdog = self._cancel_watchdogs.pop(turn_id, None)
        if watchdog is not None:
            watchdog.set()

    def _run_terminal(
        self,
        process_id: str,
        cols: int,
        rows: int,
        shell: TerminalShellPreference,
        exit_handler: TerminalExitHandler,
    ) -> None:
        from openai_codex.generated.v2_all import CommandExecResponse

        exit_code: int | None = None
        error: str | None = None
        try:
            command = codex_config.interactive_shell(shell)
            response = self._client.request(
                "command/exec",
                {
                    "command": command,
                    "cwd": str(self._terminal_workspace),
                    "env": codex_config.interactive_shell_environment(command, self._terminal_workspace),
                    "processId": process_id,
                    "sandboxPolicy": self._sandbox_policy,
                    "size": {"cols": cols, "rows": rows},
                    "streamStdin": True,
                    "streamStdoutStderr": True,
                    "disableOutputCap": True,
                    "disableTimeout": True,
                    "tty": True,
                },
                response_model=CommandExecResponse,
            )
            exit_code = response.exit_code
            self._emit_buffered_terminal_output(process_id, response.stdout, "stdout")
            self._emit_buffered_terminal_output(process_id, response.stderr, "stderr")
        except Exception as exc:  # noqa: BLE001 - SDK/background failures are surfaced through terminal state.
            if not self.is_closed:
                error = redact_text(str(exc), self._secrets)[:4_000] or "Terminal process failed"
        finally:
            with self._terminal_lock:
                self._terminal_handlers.pop(process_id, None)
            exit_handler(exit_code, error)

    def _goal_events(self, logical_turn_id: str, state: Any) -> Iterator[CodingEvent]:
        last_completion: CodingEvent | None = None
        try:
            while True:
                notification = self._client.next_goal_notification(state)
                if notification.method == "turn/completed":
                    mapped = codex_events.map_codex_notification(notification.method, notification.payload)
                    if mapped is not None:
                        last_completion = mapped
                    if state.is_finished():
                        break
                    continue
                if notification.method in {"thread/goal/updated", "thread/goal/cleared"}:
                    raw = codex_events.payload_dict(notification.payload)
                    goal = raw.get("goal") if isinstance(raw.get("goal"), dict) else {}
                    status = codex_events.string(goal.get("status")) or (
                        "cleared" if notification.method.endswith("cleared") else "active"
                    )
                    yield CodingEvent(
                        type=EventType.plan,
                        title=f"Goal {status.replace('Limited', ' limited')}",
                        text=codex_events.string(goal.get("objective")),
                        phase="completed" if status == "complete" else "progress",
                        turn_id=logical_turn_id,
                        data=sanitize(goal),
                    )
                    if state.is_finished():
                        break
                    continue
                mapped = codex_events.map_codex_notification(notification.method, notification.payload)
                if mapped is not None:
                    mapped = codex_events.redact_event(mapped, self._secrets)
                    mapped.turn_id = logical_turn_id
                    yield mapped
            terminal = last_completion or CodingEvent(
                type=EventType.session,
                title="Goal finished",
                phase="completed",
                data={"status": "completed"},
            )
            terminal.turn_id = logical_turn_id
            yield codex_events.redact_event(terminal, self._secrets)
        finally:
            self._goal_states.pop(logical_turn_id, None)
            state.finish()
            self._client.unregister_goal_operation(state)

    def _route_global_notifications(self) -> None:
        while not self.is_closed:
            try:
                notification = self._client.next_notification()
            except Exception:  # noqa: BLE001 - a closed SDK stream is terminal output completion.
                return
            if notification.method != "command/exec/outputDelta":
                continue
            raw = codex_events.payload_dict(notification.payload)
            process_id = codex_events.string(raw.get("processId") or raw.get("process_id"))
            with self._terminal_lock:
                handler = self._terminal_handlers.get(process_id)
            if handler is not None:
                handler(
                    codex_events.string(raw.get("deltaBase64") or raw.get("delta_base64")),
                    codex_events.string(raw.get("stream")),
                    bool(raw.get("capReached") or raw.get("cap_reached")),
                )

    def _emit_buffered_terminal_output(self, process_id: str, value: str, stream: str) -> None:
        if not value:
            return
        with self._terminal_lock:
            handler = self._terminal_handlers.get(process_id)
        if handler is not None:
            handler(base64.b64encode(value.encode()).decode(), stream, False)

    def _app_server_pid(self) -> int | None:
        process = getattr(self._client, "_proc", None)
        pid = getattr(process, "pid", None)
        return pid if isinstance(pid, int) and pid > 0 else None
