from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from cowork.coding.contracts import PermissionMode
from cowork.coding.engines import codex_config
from cowork.coding.engines.base import EngineSessionConfig
from cowork.coding.runtime_protocol import RuntimeExecutionConfig

INJECTED_EFFORT = 'high"\nmodel="x'


def test_runtime_execution_config_rejects_values_outside_the_codex_vocabulary() -> None:
    for field in ("reasoning_effort", "service_tier", "personality"):
        with pytest.raises(ValidationError):
            RuntimeExecutionConfig.model_validate({
                "engine_id": "codex",
                "model": "fable",
                "permission_mode": "workspace",
                field: INJECTED_EFFORT,
            })


@pytest.mark.parametrize("value", ["a\tb", "nul\x00byte", 'quote " and \\ slash', "line\nbreak\x7f"])
def test_toml_string_round_trips_control_characters(value: str) -> None:
    encoded = codex_config.toml_string(value)

    assert "\n" not in encoded
    assert "\x00" not in encoded
    assert tomllib.loads(f"value = {encoded}")["value"] == value


def test_every_codex_override_is_a_single_toml_assignment() -> None:
    launch = codex_config.prepare_launch(
        EngineSessionConfig(
            model="fable",
            permission_mode=PermissionMode.workspace,
            reasoning_effort=INJECTED_EFFORT,
            service_tier=INJECTED_EFFORT,
            personality=INJECTED_EFFORT,
            web_search=True,
        ),
        Path("/workspace"),
        "http://127.0.0.1:26866/api/v1/coding/inference",
    )

    assert all("\n" not in override for override in launch.config_overrides)
    parsed = tomllib.loads("\n".join(launch.config_overrides))
    assert parsed["model_reasoning_effort"] == INJECTED_EFFORT
    assert parsed["service_tier"] == INJECTED_EFFORT
    assert parsed["personality"] == INJECTED_EFFORT
    assert parsed["model"] == "fable"


def test_codex_retry_budget_is_bounded_on_the_mindshub_provider() -> None:
    launch = codex_config.prepare_launch(
        EngineSessionConfig(model="gpt", permission_mode=PermissionMode.supervised),
        Path("/tmp/workspace"),
        "http://127.0.0.1:1/api/v1/coding/inference",
    )

    assert "model_providers.mindshub.request_max_retries=1" in launch.config_overrides
    assert "model_providers.mindshub.stream_max_retries=2" in launch.config_overrides
    assert (codex_config.REQUEST_MAX_RETRIES + 1) * (codex_config.STREAM_MAX_RETRIES + 1) <= 6


def test_codex_sends_no_analytics_feedback_or_telemetry() -> None:
    launch = codex_config.prepare_launch(
        EngineSessionConfig(model="gpt", permission_mode=PermissionMode.supervised),
        Path("/tmp/workspace"),
        "http://127.0.0.1:1/api/v1/coding/inference",
    )

    for override in (
        "analytics.enabled=false",
        "feedback.enabled=false",
        'otel.exporter="none"',
        'otel.trace_exporter="none"',
        'otel.metrics_exporter="none"',
    ):
        assert override in launch.config_overrides


def test_the_child_env_blanks_an_inherited_codex_login() -> None:
    environment = codex_config.client_environment(
        Path("/tmp/codex-home"), (("CODEX_API_KEY", "project-key"), ("CODEX_ACCESS_TOKEN", "token")),
        inference_api_key="session-inference-token",
    )

    assert environment["CODEX_API_KEY"] == ""
    assert environment["CODEX_ACCESS_TOKEN"] == ""


def _launch(permission_mode: PermissionMode, *, writes_allowed: bool, task_mode: str = "build"):
    return codex_config.prepare_launch(
        EngineSessionConfig(
            model="gpt",
            permission_mode=permission_mode,
            task_mode=task_mode,
            writes_allowed=writes_allowed,
        ),
        Path("/workspace"),
        "http://127.0.0.1:1/api/v1/coding/inference",
    )


@pytest.mark.parametrize("mode", [PermissionMode.supervised, PermissionMode.workspace])
def test_held_writes_give_a_read_only_sandbox_that_asks_the_person(mode: PermissionMode) -> None:
    launch = _launch(mode, writes_allowed=False)

    assert launch.sandbox_policy["type"] == "readOnly"
    assert launch.thread_params["sandbox"] == "read-only"
    assert 'sandbox_mode="read-only"' in launch.config_overrides
    assert launch.approval_policy == "on-request"
    assert launch.approvals_reviewer == "user"


@pytest.mark.parametrize(
    ("mode", "reviewer"),
    [(PermissionMode.supervised, "user"), (PermissionMode.workspace, "auto_review")],
)
def test_allowed_writes_keep_the_workspace_write_sandbox(mode: PermissionMode, reviewer: str) -> None:
    launch = _launch(mode, writes_allowed=True)

    assert launch.sandbox_policy == {"type": "workspaceWrite", "networkAccess": False, "writableRoots": ["/workspace"]}
    assert launch.thread_params["sandbox"] == "workspace-write"
    assert launch.approvals_reviewer == reviewer


def test_full_access_ignores_held_writes() -> None:
    launch = _launch(PermissionMode.full_access, writes_allowed=False)

    assert launch.sandbox_policy == {"type": "dangerFullAccess"}
    assert launch.approval_policy == "never"


def test_plan_mode_stays_read_only_without_approvals_when_writes_are_held() -> None:
    launch = _launch(PermissionMode.supervised, writes_allowed=False, task_mode="plan")

    assert launch.sandbox_policy["type"] == "readOnly"
    assert launch.approval_policy == "never"


@pytest.mark.parametrize("permission_mode", list(PermissionMode))
def test_commands_the_engine_runs_get_a_blank_inference_key(permission_mode: PermissionMode) -> None:
    """The app-server reads its inference credential from its own environment
    and every command it starts inherits that environment. Codex 0.147 drops
    no *KEY* variable by default, so the launch blanks this one for commands:
    its shell_environment_policy `set` runs after inheritance and excludes.
    Checked against the bundled app-server: with this override a command/exec
    child sees MINDSHUB_CODEX_API_KEY empty, without it the full value."""
    launch = codex_config.prepare_launch(
        EngineSessionConfig(model="gpt", permission_mode=permission_mode),
        Path("/workspace"),
        "http://127.0.0.1:1/api/v1/coding/inference",
    )

    parsed = tomllib.loads("\n".join(launch.config_overrides))
    assert parsed["model_providers"]["mindshub"]["env_key"] == "MINDSHUB_CODEX_API_KEY"
    assert parsed["shell_environment_policy"] == {"set": {"MINDSHUB_CODEX_API_KEY": ""}}
    assert 'shell_environment_policy.set.MINDSHUB_CODEX_API_KEY=""' in launch.config_overrides


def _integration_launch() -> codex_config.CodexLaunchConfig:
    return codex_config.prepare_launch(
        EngineSessionConfig(
            model="gpt",
            permission_mode=PermissionMode.workspace,
            session_id="session-1",
            cowork_root="/data/coding",
            project_linked=True,
        ),
        Path("/workspace"),
        "http://127.0.0.1:1/api/v1/coding/inference",
    )


def test_the_integration_mcp_server_gets_the_store_paths_the_server_started_with(monkeypatch) -> None:
    """Codex starts an MCP server with a bare environment, so the paths that
    locate the server's stores go in its env table, from the launch values."""
    from types import MappingProxyType

    from cowork.common import launch_environment

    for name in codex_config.INTEGRATION_MCP_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launch_environment, "_captured", MappingProxyType({
        "COWORK_HOME": "/accounts/a1",
        "COWORK_VAULT_DIR": "/accounts/a1/orgs/o1/data-vault",
        "COWORK_SERVER_HEALTH_PROOF_KEY": "fake-not-a-path",
    }))

    parsed = tomllib.loads("\n".join(_integration_launch().config_overrides))

    assert parsed["mcp_servers"]["mindshub_code"]["env"] == {
        "COWORK_HOME": "/accounts/a1",
        "COWORK_VAULT_DIR": "/accounts/a1/orgs/o1/data-vault",
    }


def test_the_integration_mcp_server_gets_no_env_table_when_nothing_was_set(monkeypatch) -> None:
    from types import MappingProxyType

    from cowork.common import launch_environment

    for name in codex_config.INTEGRATION_MCP_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launch_environment, "_captured", MappingProxyType({}))

    parsed = tomllib.loads("\n".join(_integration_launch().config_overrides))

    assert "env" not in parsed["mcp_servers"]["mindshub_code"]
