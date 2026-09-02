from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cowork.coding.contracts import ExtensionInventory, PermissionMode
from cowork.coding.engines import codex as codex_module
from cowork.coding.engines import codex_config, codex_events
from cowork.coding.engines.base import EngineInputReference, EngineSessionConfig
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


def test_codex_child_environment_never_contains_the_real_mindshub_key() -> None:
    environment = codex_config.client_environment(Path("/tmp/codex-home"))
    assert environment == {
        "CODEX_HOME": "/tmp/codex-home",
        "MINDSHUB_CODEX_API_KEY": codex_config.LOCAL_PROXY_TOKEN,
    }
    assert len(codex_config.LOCAL_PROXY_TOKEN) >= 32
    assert "mdb_real-secret" not in environment.values()


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


def test_cowork_and_user_skill_roots_use_current_codex_rpc_name(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object], object]] = []

    class FakeClient:
        def request(self, method: str, params: dict[str, object], *, response_model: object) -> None:
            calls.append((method, params, response_model))

    skill_root = tmp_path / "skills"
    user_skill_root = tmp_path / "user-codex" / "skills"
    user_skill_root.mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(user_skill_root.parent))
    monkeypatch.setattr(
        codex_module,
        "get_app_settings",
        lambda: SimpleNamespace(skill=SimpleNamespace(root_dir=str(skill_root))),
    )
    engine_session = object.__new__(codex_module.CodexEngineSession)
    engine_session._client = FakeClient()

    engine_session._register_skill_roots()

    assert skill_root.is_dir()
    assert calls[0][0] == "skills/extraRoots/set"
    assert calls[0][1] == {"extraRoots": [str(skill_root), str(user_skill_root)]}


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


class _Payload:
    def __init__(self, data: dict) -> None:
        self.data = data

    def model_dump(self, **_kwargs):
        return self.data
