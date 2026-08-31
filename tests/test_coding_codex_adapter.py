from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cowork.coding.contracts import EventType, ExtensionInventory, PermissionMode
from cowork.coding.engines import codex as codex_module
from cowork.coding.engines import codex_config, codex_events
from cowork.coding.engines.base import (
    EngineCredentials,
    EngineInputReference,
    EngineSessionConfig,
)
from cowork.coding import shells
from cowork.coding.engines.codex_extensions import add_extension_response
from cowork.coding.redaction import sanitize


def test_mindshub_responses_base_url_is_normalized_once() -> None:
    assert codex_config.responses_base_url("https://api.mindshub.ai") == "https://api.mindshub.ai/v1"
    assert codex_config.responses_base_url("https://api.mindshub.ai/v1/") == "https://api.mindshub.ai/v1"


def test_permission_mode_maps_to_codex_approval_policy() -> None:
    assert codex_config.approval_policy(PermissionMode.read_only) == "on-request"
    assert codex_config.approval_policy(PermissionMode.supervised) == "on-request"
    assert codex_config.approval_policy(PermissionMode.workspace) == "never"
    assert codex_config.approval_policy(PermissionMode.full_access) == "never"
    assert codex_config.sandbox_mode(PermissionMode.read_only) == "read-only"
    assert codex_config.sandbox_mode(PermissionMode.supervised) == "workspace-write"
    assert codex_config.sandbox_mode(PermissionMode.full_access) == "danger-full-access"
    assert codex_config.sandbox_policy(PermissionMode.read_only) == {"type": "readOnly", "networkAccess": False}
    assert codex_config.sandbox_policy(PermissionMode.full_access) == {"type": "dangerFullAccess"}
    assert codex_config.sandbox_policy(PermissionMode.workspace) == {
        "type": "workspaceWrite",
        "networkAccess": False,
        "writableRoots": [],
    }


def test_codex_uses_the_loopback_inference_proxy() -> None:
    assert codex_config.local_inference_base_url(26866) == "http://127.0.0.1:26866/api/v1/coding/inference"


def test_codex_model_discovery_keeps_models_that_need_credits(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"id": "mindshub_air", "enabled": True},
                    {"id": "fable", "enabled": False},
                    {"id": "gpt-codex", "enabled": False},
                    {"id": "embed-small", "enabled": False, "embedding": True},
                ],
            }

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, url: str, *, headers: dict[str, str]) -> FakeResponse:
            assert url == "https://api.mindshub.ai/v1/models"
            assert headers == {"Authorization": "Bearer mdb_test"}
            return FakeResponse()

    monkeypatch.setattr(codex_module.httpx, "Client", lambda **_kwargs: FakeClient())

    models = codex_module.CodexEngine().discover_models(
        EngineCredentials(minds_url="https://api.mindshub.ai", minds_api_key="mdb_test"),
    )

    assert models == ["mindshub_air", "fable", "gpt-codex"]


def test_codex_child_environment_never_contains_the_real_mindshub_key() -> None:
    environment = codex_config.client_environment(Path("/tmp/codex-home"))
    assert environment == {
        "CODEX_HOME": "/tmp/codex-home",
        "MINDSHUB_CODEX_API_KEY": codex_config.LOCAL_PROXY_TOKEN,
    }
    assert len(codex_config.LOCAL_PROXY_TOKEN) >= 32
    assert "mdb_real-secret" not in environment.values()


def test_interactive_terminal_prefers_bash_over_the_users_shell(monkeypatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(
        shells,
        "_resolve_executable",
        lambda candidate: "/opt/homebrew/bin/bash" if candidate == "bash" else candidate,
    )

    assert codex_config.interactive_shell() == ["/opt/homebrew/bin/bash", "--login"]


def test_windows_skips_the_legacy_wsl_bash_launcher() -> None:
    assert shells._windows_bash_is_compatible(r"C:\Program Files\Git\bin\bash.exe") is True
    assert shells._windows_bash_is_compatible(r"C:\Windows\System32\bash.exe") is False


def test_bash_terminal_suppresses_the_misleading_macos_zsh_notice() -> None:
    assert codex_config.interactive_shell_environment(["/bin/bash", "--login"]) == {
        "BASH_SILENCE_DEPRECATION_WARNING": "1",
        "PREFIX": "",
        "npm_config_prefix": "",
    }
    assert codex_config.interactive_shell_environment(["/bin/zsh", "--login"]) == {
        "PREFIX": "",
        "npm_config_prefix": "",
    }


def test_interactive_terminal_does_not_inherit_the_apps_private_npm_prefix(
    monkeypatch,
) -> None:
    monkeypatch.setenv("npm_config_prefix", "/Users/example/.hermes/node")

    environment = codex_config.interactive_shell_environment(["/bin/zsh", "--login"])

    assert environment["npm_config_prefix"] == ""
    assert environment["PREFIX"] == ""


def test_terminal_workspace_uses_a_human_label_without_changing_the_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "7cf89628-023b-44e1-ab3d-498d8153cff4"
    workspace.mkdir(parents=True)

    alias = codex_config.terminal_workspace(
        tmp_path / "cowork-data",
        "7cf89628-023b-44e1-ab3d-498d8153cff4",
        workspace,
        "MindsHub Code QA",
    )

    assert alias.name == "MindsHub Code QA"
    assert alias.is_symlink()
    assert alias.resolve() == workspace.resolve()
    assert codex_config.interactive_shell_environment(["/bin/bash", "--login"], alias)["PWD"] == str(alias)


def test_codex_runtime_values_cannot_be_overridden_by_project_environment() -> None:
    environment = codex_config.client_environment(
        Path("/tmp/codex-home"),
        (("PROJECT_NAME", "atlas"), ("CODEX_HOME", "/wrong"), ("MINDSHUB_CODEX_API_KEY", "wrong")),
    )

    assert environment["PROJECT_NAME"] == "atlas"
    assert environment["CODEX_HOME"] == "/tmp/codex-home"
    assert environment["MINDSHUB_CODEX_API_KEY"] == codex_config.LOCAL_PROXY_TOKEN


def test_codex_runtime_home_is_persistent_and_isolated_from_shell_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "unrelated-user-home"))

    assert codex_config.persistent_home(tmp_path / "coding") == tmp_path / "coding" / "codex-home"
    assert codex_config.user_skills_root() == tmp_path / "unrelated-user-home" / "skills"


def test_turn_input_preserves_native_mentions_and_local_images() -> None:
    assert codex_config.turn_input(
        "Compare these files",
        (
            EngineInputReference(name="src/app.ts", path="/work/src/app.ts", kind="mention"),
            EngineInputReference(name="design.png", path="/tmp/design.png", kind="local_image"),
        ),
    ) == [
        {"type": "text", "text": "Compare these files"},
        {"type": "mention", "name": "src/app.ts", "path": "/work/src/app.ts"},
        {"type": "localImage", "path": "/tmp/design.png"},
    ]


def test_codex_launch_policy_is_resolved_once_for_client_and_thread() -> None:
    launch = codex_config.prepare_launch(
        EngineSessionConfig(
            model="fable",
            permission_mode=PermissionMode.workspace,
            reasoning_effort="high",
            service_tier="priority",
            personality="pragmatic",
            network_access=True,
            web_search=True,
            additional_dirs=("/extra",),
            developer_instructions="Use the project playbook.",
            session_id="task-123",
            cowork_root="/cowork-data",
        ),
        Path("/workspace"),
        "http://127.0.0.1:26866/api/v1/coding/inference",
    )

    assert launch.approval_policy == "never"
    assert launch.sandbox_policy == {
        "type": "workspaceWrite",
        "networkAccess": True,
        "writableRoots": ["/workspace", "/extra"],
    }
    assert 'model="fable"' in launch.config_overrides
    assert 'model_reasoning_effort="high"' in launch.config_overrides
    assert 'service_tier="priority"' in launch.config_overrides
    assert 'web_search="live"' in launch.config_overrides
    assert any(item.startswith("mcp_servers.mindshub_code.command=") for item in launch.config_overrides)
    assert any('"cowork.coding.integration_mcp","/cowork-data","task-123"' in item for item in launch.config_overrides)
    assert launch.thread_params["developerInstructions"] == "Use the project playbook."
    assert launch.thread_params["approvalPolicy"] == launch.approval_policy
    assert launch.thread_params["sandbox"] == "workspace-write"


def test_extension_inventory_normalizes_codex_skills_and_mcp_servers() -> None:
    inventory = ExtensionInventory()
    skill = SimpleNamespace(
        name="review",
        description="Review completed work",
        short_description=None,
        enabled=True,
        scope="user",
        path="/skills/review/SKILL.md",
    )
    add_extension_response(
        inventory,
        "skills",
        SimpleNamespace(data=[SimpleNamespace(skills=[skill])]),
    )
    server = SimpleNamespace(
        name="github",
        server_info=SimpleNamespace(name="github", title="GitHub", description="Repository tools"),
        auth_status="authenticated",
        tools={"issues": object(), "pulls": object()},
        resources=[],
    )
    add_extension_response(
        inventory,
        "mcp",
        SimpleNamespace(data=[server]),
    )

    assert inventory.skills[0].label == "review"
    assert inventory.skills[0].path == "/skills/review/SKILL.md"
    assert inventory.mcp_servers[0].label == "GitHub"
    assert inventory.mcp_servers[0].detail == "2 tools · 0 resources"


def test_code_and_user_skill_roots_use_current_codex_rpc_name(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object], object]] = []

    class FakeClient:
        def request(self, method: str, params: dict[str, object], *, response_model: object) -> None:
            calls.append((method, params, response_model))

    cowork_skill_root = tmp_path / "skills"
    code_skill_root = tmp_path / "code-skills"
    user_skill_root = tmp_path / "user-codex" / "skills"
    user_skill_root.mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(user_skill_root.parent))
    monkeypatch.setattr(
        codex_module,
        "get_app_settings",
        lambda: SimpleNamespace(skill=SimpleNamespace(root_dir=str(cowork_skill_root))),
    )
    engine_session = object.__new__(codex_module.CodexEngineSession)
    engine_session._client = FakeClient()

    engine_session._register_skill_roots()

    assert code_skill_root.is_dir()
    assert not cowork_skill_root.exists()
    assert calls[0][0] == "skills/extraRoots/set"
    assert calls[0][1] == {"extraRoots": [str(code_skill_root), str(user_skill_root)]}


def test_nested_codex_error_maps_to_safe_text() -> None:
    event = codex_events.map_codex_notification(
        "error",
        _Payload({"error": {"message": "Authorization: Bearer secret-value"}}),
    )
    assert event is not None
    assert event.text == "Authorization: Bearer [redacted]"


def test_event_redaction_covers_exact_secret_and_sensitive_fields() -> None:
    event = codex_events.map_codex_notification(
        "item/completed",
        _Payload({"item": {"id": "1", "type": "commandExecution", "command": "echo exact-key", "apiKey": "other"}}),
    )
    assert event is not None
    redacted = codex_events.redact_event(event, ("exact-key",))
    assert "exact-key" not in redacted.title
    assert redacted.data["apiKey"] == "[redacted]"


def test_turn_completion_preserves_terminal_status() -> None:
    event = codex_events.map_codex_notification(
        "turn/completed",
        _Payload({"turn": {"id": "turn-1", "status": "interrupted"}}),
    )
    assert event is not None
    assert event.data == {"status": "interrupted"}
    assert event.turn_id == "turn-1"


def test_plain_dict_notification_payload_is_supported() -> None:
    event = codex_events.map_codex_notification("item/agentMessage/delta", {"delta": "hello"})
    assert event is not None
    assert event.text == "hello"


def test_collaboration_items_map_to_visible_child_work() -> None:
    event = codex_events.map_codex_notification(
        "item/started",
        {"item": {"id": "worker-1", "type": "collabAgentToolCall", "prompt": "Audit the API boundary"}},
    )

    assert event is not None
    assert event.type == EventType.child_work
    assert event.title == "Audit the API boundary"
    assert event.item_id == "worker-1"


def test_nested_payload_sanitizer_has_a_global_budget() -> None:
    payload: object = "leaf"
    for _ in range(8):
        payload = [payload] * 100

    bounded = sanitize(payload)

    assert len(json.dumps(bounded)) < 300_000


def test_cancel_arms_watchdog_before_interrupt_rpc(monkeypatch) -> None:
    order: list[str] = []

    class FakeThread:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            order.append("watchdog")

    class FakeClient:
        def turn_interrupt(self, _session_id: str, _turn_id: str) -> None:
            order.append("interrupt")

    engine_session = object.__new__(codex_module.CodexEngineSession)
    engine_session._client = FakeClient()
    engine_session._session_id = "session-1"
    engine_session._cancel_watchdogs = {}
    engine_session._cancel_lock = codex_module.threading.Lock()
    engine_session._goal_states = {}
    monkeypatch.setattr(codex_module.threading, "Thread", FakeThread)

    engine_session.cancel("turn-1")
    watchdog = engine_session._cancel_watchdogs["turn-1"]
    engine_session.cancel("turn-1")

    assert order == ["watchdog", "interrupt", "interrupt"]
    assert engine_session._cancel_watchdogs["turn-1"] is watchdog


def test_resume_goal_registers_routing_before_reactivating_the_goal(monkeypatch) -> None:
    order: list[str] = []

    class FakeState:
        def activate_turn_routing(self) -> None:
            order.append("route")

        def wait_for_start(self, timeout: float) -> str:
            assert timeout == 30.0
            order.append("wait")
            return "goal-turn-2"

    state = FakeState()

    class FakeClient:
        def reserve_goal_operation(self, thread_id: str):
            assert thread_id == "session-1"
            order.append("reserve")
            return state

        def thread_goal_set(self, thread_id: str, *, status) -> None:
            assert thread_id == "session-1"
            assert status.value == "active"
            order.append("activate")

    engine_session = object.__new__(codex_module.CodexEngineSession)
    engine_session._client = FakeClient()
    engine_session._session_id = "session-1"
    engine_session._goal_states = {}
    monkeypatch.setattr(engine_session, "goal_status", lambda: {"objective": "Ship", "status": "paused"})

    assert engine_session.resume_goal() == "goal-turn-2"
    assert order == ["reserve", "route", "activate", "wait"]
    assert engine_session._goal_states["goal-turn-2"] is state


def test_goal_steer_retries_with_the_server_reported_active_turn() -> None:
    from openai_codex.errors import InvalidRequestError

    class FakeState:
        def __init__(self) -> None:
            self.turn_id = "stale-turn"

        def current_turn(self) -> str:
            return self.turn_id

        def resolve_active_turn(self, expected: str, active: str) -> None:
            assert expected == "stale-turn"
            self.turn_id = active

    class FakeClient:
        def __init__(self) -> None:
            self.turn_ids: list[str] = []

        def turn_steer(self, _thread_id: str, turn_id: str, _items) -> None:
            self.turn_ids.append(turn_id)
            if turn_id == "stale-turn":
                raise InvalidRequestError(
                    -32600,
                    "expected active turn id `active-turn` but found `stale-turn`",
                )

    state = FakeState()
    client = FakeClient()
    engine_session = object.__new__(codex_module.CodexEngineSession)
    engine_session._client = client
    engine_session._session_id = "session-1"
    engine_session._goal_states = {"logical-goal-turn": state}

    engine_session.steer("logical-goal-turn", "Finish validation")

    assert client.turn_ids == ["stale-turn", "active-turn"]
    assert state.turn_id == "active-turn"


def test_goal_updates_use_native_thread_lifecycle(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeClient:
        def pause_goal(self, thread_id: str) -> None:
            calls.append(("pause", thread_id))

        def thread_goal_set(self, thread_id: str, *, objective: str) -> None:
            calls.append(("edit", thread_id, objective))

        def thread_goal_clear(self, thread_id: str) -> None:
            calls.append(("clear", thread_id))

    engine_session = object.__new__(codex_module.CodexEngineSession)
    engine_session._client = FakeClient()
    engine_session._session_id = "session-1"
    statuses = iter([
        {"objective": "Ship", "status": "active"},
        {"objective": "Ship", "status": "paused"},
        {"objective": "Ship", "status": "paused"},
        {"objective": "Ship safely", "status": "paused"},
    ])
    monkeypatch.setattr(engine_session, "goal_status", lambda: next(statuses))

    assert engine_session.update_goal("pause")["status"] == "paused"
    assert engine_session.update_goal("edit", "Ship safely")["objective"] == "Ship safely"
    assert engine_session.update_goal("clear") is None
    assert calls == [
        ("pause", "session-1"),
        ("edit", "session-1", "Ship safely"),
        ("clear", "session-1"),
    ]


def test_terminal_write_waits_for_codex_process_registration(monkeypatch) -> None:
    from openai_codex.errors import InvalidRequestError

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def request(self, method: str, params: dict[str, object], *, response_model: object) -> object:
            assert method == "command/exec/write"
            assert params == {"processId": "process-1", "deltaBase64": "aGkK"}
            self.calls += 1
            if self.calls == 1:
                raise InvalidRequestError(-32600, "no active command/exec for process id process-1")
            return response_model.model_validate({})

    engine_session = object.__new__(codex_module.CodexEngineSession)
    engine_session._client = FakeClient()
    engine_session._closed = codex_module.threading.Event()
    monkeypatch.setattr(codex_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(codex_module, "_TERMINAL_WRITE_RETRY_SECONDS", 0.0)

    engine_session.write_terminal("process-1", "aGkK")

    assert engine_session._client.calls == 2


def test_terminal_write_does_not_retry_unrelated_rpc_failures(monkeypatch) -> None:
    from openai_codex.errors import InvalidRequestError

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def request(self, *_args, **_kwargs) -> None:
            self.calls += 1
            raise InvalidRequestError(-32600, "terminal input was rejected")

    engine_session = object.__new__(codex_module.CodexEngineSession)
    engine_session._client = FakeClient()
    engine_session._closed = codex_module.threading.Event()
    monkeypatch.setattr(codex_module, "_TERMINAL_WRITE_RETRY_SECONDS", 0.0)

    with pytest.raises(InvalidRequestError, match="terminal input was rejected"):
        engine_session.write_terminal("process-1", "aGkK")

    assert engine_session._client.calls == 1


class _Payload:
    def __init__(self, data: dict) -> None:
        self.data = data

    def model_dump(self, **_kwargs):
        return self.data
