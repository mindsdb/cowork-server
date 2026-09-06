from __future__ import annotations

import os
import re
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cowork.coding.contracts import PermissionMode, TerminalShellPreference
from cowork.coding.engines.base import EngineInputReference, EngineSessionConfig
from cowork.coding.shells import resolve_shell, shell_environment

# Process-local bearer credential for the inference proxy. A fixed string would
# turn the credential-injecting loopback endpoint into a confused deputy for
# any unrelated process on the machine. Rotation on every server start is fine:
# task runtimes are children of, and owned by, this process.
LOCAL_PROXY_TOKEN = secrets.token_urlsafe(32)

# Codex's local estimate can substantially under-count context forwarded
# through the MindsHub Responses proxy (notably cached input and large tool
# results). A conservative threshold makes compaction happen while the provider
# still has ample room for the compaction request and the next turn. This is a
# runtime safety limit, not the model's advertised context-window size.
DEFAULT_AUTO_COMPACT_TOKEN_LIMIT = 36_000
AUTO_COMPACT_TOKEN_LIMIT_ENV = "MINDSHUB_CODE_AUTO_COMPACT_TOKEN_LIMIT"


@dataclass(frozen=True)
class CodexLaunchConfig:
    """Fully resolved Codex policy at the engine-adapter boundary."""

    approval_policy: str
    sandbox_policy: dict[str, Any]
    config_overrides: tuple[str, ...]
    thread_params: dict[str, Any]


def persistent_home(cowork_root: Path) -> Path:
    """Keep Codex thread state with Cowork so task IDs survive app restarts."""
    return cowork_root / "codex-home"


def user_skills_root() -> Path:
    """Locate skills installed for the user's regular Codex environment."""
    codex_home = os.environ.get("CODEX_HOME")
    return (Path(codex_home).expanduser() if codex_home else Path.home() / ".codex") / "skills"


def client_environment(
    codex_home: Path,
    project_environment: tuple[tuple[str, str], ...] = (),
    inference_api_key: str = LOCAL_PROXY_TOKEN,
) -> dict[str, str]:
    """Build the child env without placing the user's MindsHub key in it."""
    environment = dict(project_environment)
    # Runtime-owned values win over project configuration. Otherwise a project
    # could accidentally point Codex at a different home or break the private
    # loopback credential used by the inference proxy.
    environment.update({
        "CODEX_HOME": str(codex_home),
        "MINDSHUB_CODEX_API_KEY": inference_api_key,
    })
    return environment


def turn_input(
    prompt: str,
    attachments: tuple[EngineInputReference, ...],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = [{"type": "text", "text": prompt}]
    for item in attachments:
        if item.kind == "local_image":
            items.append({"type": "localImage", "path": item.path})
        else:
            items.append({"type": "mention", "name": item.name, "path": item.path})
    return items


def approval_policy(permission_mode: PermissionMode) -> str:
    return "on-request" if permission_mode in {PermissionMode.read_only, PermissionMode.supervised} else "never"


def sandbox_mode(permission_mode: PermissionMode) -> str:
    if permission_mode == PermissionMode.read_only:
        return "read-only"
    if permission_mode == PermissionMode.full_access:
        return "danger-full-access"
    return "workspace-write"


def sandbox_policy(
    permission_mode: PermissionMode,
    workspace: Path | None = None,
    additional_dirs: tuple[str, ...] = (),
    network_access: bool = False,
) -> dict[str, Any]:
    if permission_mode == PermissionMode.read_only:
        return {"type": "readOnly", "networkAccess": network_access}
    if permission_mode == PermissionMode.full_access:
        return {"type": "dangerFullAccess"}
    writable_roots = [str(workspace)] if workspace is not None else []
    writable_roots.extend(additional_dirs)
    return {
        "type": "workspaceWrite",
        "networkAccess": network_access,
        "writableRoots": writable_roots,
    }


def interactive_shell(
    preference: TerminalShellPreference | str = TerminalShellPreference.auto,
) -> list[str]:
    """Compatibility seam for callers that resolve shells through Codex."""
    return resolve_shell(preference)


def interactive_shell_environment(
    command: list[str],
    working_directory: Path | None = None,
) -> dict[str, str]:
    return shell_environment(command, working_directory)


def terminal_workspace(
    cowork_root: Path,
    session_id: str,
    workspace: Path,
    label: str,
) -> Path:
    """Create a logical cwd whose basename is meaningful in shell prompts.

    Task isolation uses UUID-named directories. A managed symlink plus PWD
    lets ordinary shell prompts display a project/folder name while commands
    still operate on the exact same isolated files.
    """
    if not session_id:
        return workspace
    safe_label = re.sub(r"[^A-Za-z0-9._ -]+", "-", label).strip(" .-")[:64] or "Workspace"
    parent = cowork_root / "terminal-workspaces" / session_id
    alias = parent / safe_label
    try:
        parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(alias):
            if alias.is_symlink() and alias.resolve() == workspace.resolve():
                return alias
            if not alias.is_symlink():
                return workspace
            alias.unlink()
        alias.symlink_to(workspace.resolve(), target_is_directory=True)
        return alias
    except OSError:
        # Windows may disallow symlink creation without Developer Mode. The
        # terminal remains fully usable from its real isolated path.
        return workspace


def local_inference_base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/api/v1/coding/inference"


def responses_base_url(raw_url: str) -> str:
    base = (raw_url or "https://api.mindshub.ai").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def auto_compact_token_limit() -> int:
    """Return the preventive Codex compaction threshold.

    The environment override exists for deterministic runtime verification
    and emergency tuning without shipping a desktop update. Invalid values
    fail safe to the supported default instead of preventing Code Mode from
    launching.
    """
    raw_value = os.environ.get(AUTO_COMPACT_TOKEN_LIMIT_ENV)
    if raw_value is None:
        return DEFAULT_AUTO_COMPACT_TOKEN_LIMIT
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_AUTO_COMPACT_TOKEN_LIMIT
    return value if value > 0 else DEFAULT_AUTO_COMPACT_TOKEN_LIMIT


_TOML_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


def toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = _TOML_CONTROL_CHARACTER.sub(lambda match: f"\\u{ord(match.group()):04X}", escaped)
    return f'"{escaped}"'


def toml_array(values: tuple[str, ...]) -> str:
    return "[" + ",".join(toml_string(value) for value in values) + "]"


# Codex retries a failed request at two layers: `request_max_retries` HTTP
# attempts per stream attempt, and `stream_max_retries` stream attempts per turn.
# Its defaults (4 and 5) multiply to 30 requests against an upstream that is
# down. Every request already passes through the loopback proxy, so keep the
# product small: one HTTP retry, two stream reconnects.
REQUEST_MAX_RETRIES = 1
STREAM_MAX_RETRIES = 2


def prepare_launch(config: EngineSessionConfig, workspace: Path, endpoint: str) -> CodexLaunchConfig:
    """Translate Cowork runtime controls into one consistent Codex launch policy."""
    resolved_approval = approval_policy(config.permission_mode)
    resolved_sandbox = sandbox_mode(config.permission_mode)
    resolved_policy = sandbox_policy(
        config.permission_mode,
        workspace,
        config.additional_dirs,
        config.network_access,
    )
    overrides = [
        'model_provider="mindshub"',
        f"model={toml_string(config.model)}",
        'model_providers.mindshub.name="MindsHub Inference"',
        f"model_providers.mindshub.base_url={toml_string(endpoint)}",
        'model_providers.mindshub.env_key="MINDSHUB_CODEX_API_KEY"',
        'model_providers.mindshub.wire_api="responses"',
        f"model_providers.mindshub.request_max_retries={REQUEST_MAX_RETRIES}",
        f"model_providers.mindshub.stream_max_retries={STREAM_MAX_RETRIES}",
        f"model_auto_compact_token_limit={auto_compact_token_limit()}",
        f"approval_policy={toml_string(resolved_approval)}",
        f"sandbox_mode={toml_string(resolved_sandbox)}",
        f"web_search={toml_string('live' if config.web_search else 'disabled')}",
    ]
    if config.reasoning_effort:
        overrides.append(f"model_reasoning_effort={toml_string(config.reasoning_effort)}")
    if config.service_tier:
        overrides.append(f"service_tier={toml_string(config.service_tier)}")
    if config.personality and config.personality != "none":
        overrides.append(f"personality={toml_string(config.personality)}")
    if config.mcp_servers:
        for server in config.mcp_servers:
            safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", server.name)
            overrides.extend([
                f"mcp_servers.{safe_name}.command={toml_string(server.command)}",
                f"mcp_servers.{safe_name}.args={toml_array(server.args)}",
            ])
    elif config.session_id and config.cowork_root:
        overrides.extend([
            f"mcp_servers.mindshub_code.command={toml_string(sys.executable)}",
            "mcp_servers.mindshub_code.args=" + toml_array((
                "-m",
                "cowork.coding.integration_mcp",
                config.cowork_root,
                config.session_id,
            )),
        ])
    thread_params = {
        "cwd": str(workspace),
        "model": config.model,
        "modelProvider": "mindshub",
        "approvalPolicy": resolved_approval,
        "approvalsReviewer": "user",
        "sandbox": resolved_sandbox,
        "serviceTier": config.service_tier,
        "personality": config.personality,
    }
    if config.developer_instructions:
        thread_params["developerInstructions"] = config.developer_instructions
    return CodexLaunchConfig(
        approval_policy=resolved_approval,
        sandbox_policy=resolved_policy,
        config_overrides=tuple(overrides),
        thread_params=thread_params,
    )
