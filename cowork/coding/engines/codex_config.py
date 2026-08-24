from __future__ import annotations

import os
import secrets
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cowork.coding.contracts import PermissionMode
from cowork.coding.engines.base import EngineInputReference, EngineSessionConfig

# Process-local bearer credential for the inference proxy. A fixed string would
# turn the credential-injecting loopback endpoint into a confused deputy for
# any unrelated process on the machine. Rotation on every server start is fine:
# task runtimes are children of, and owned by, this process.
LOCAL_PROXY_TOKEN = secrets.token_urlsafe(32)


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
) -> dict[str, str]:
    """Build the child env without placing the user's MindsHub key in it."""
    environment = dict(project_environment)
    # Runtime-owned values win over project configuration. Otherwise a project
    # could accidentally point Codex at a different home or break the private
    # loopback credential used by the inference proxy.
    environment.update({
        "CODEX_HOME": str(codex_home),
        "MINDSHUB_CODEX_API_KEY": LOCAL_PROXY_TOKEN,
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


def interactive_shell() -> list[str]:
    if os.name == "nt":
        shell = shutil.which("pwsh") or shutil.which("powershell") or os.environ.get("COMSPEC") or "cmd.exe"
        return [shell]
    configured = os.environ.get("SHELL")
    shell = configured if configured and Path(configured).is_file() else None
    shell = shell or ("/bin/zsh" if Path("/bin/zsh").is_file() else "/bin/sh")
    return [shell, "-l"]


def local_inference_base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/api/v1/coding/inference"


def responses_base_url(raw_url: str) -> str:
    base = (raw_url or "https://api.mindshub.ai").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def toml_array(values: tuple[str, ...]) -> str:
    return "[" + ",".join(toml_string(value) for value in values) + "]"


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
        f'approval_policy="{resolved_approval}"',
        f'sandbox_mode="{resolved_sandbox}"',
        f'web_search="{"live" if config.web_search else "disabled"}"',
    ]
    if config.reasoning_effort:
        overrides.append(f'model_reasoning_effort="{config.reasoning_effort}"')
    if config.service_tier:
        overrides.append(f'service_tier="{config.service_tier}"')
    if config.personality and config.personality != "none":
        overrides.append(f'personality="{config.personality}"')
    if config.session_id and config.cowork_root:
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
