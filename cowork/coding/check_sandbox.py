"""Run project checks and commit hooks under the task's sandbox.

Checks and hooks run code the agent may have written, so they get the same
boundary as the agent, shaped for test runs: they read anywhere but write only
to the task's folders, temp and a per-project cache; they may listen on
localhost and use Unix sockets there; and outbound traffic follows the task's
network setting. Their environment is the server's minus Cowork's own
variables and anything credential-shaped.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cowork.coding.contracts import PermissionMode
from cowork.coding.engines.codex_config import toml_string
from cowork.coding.workspace import WorkspaceError

_PRIVATE_NAME = re.compile(r"(SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|_KEY$|^KEY$|_KEY_)", re.IGNORECASE)
_PRIVATE_PREFIXES = ("COWORK_", "ANTON_", "DATABASE_")
# Tools that otherwise write caches under the home folder, which the sandbox
# keeps read-only. A per-project cache keeps them working without letting a
# check poison a cache the person's own builds use.
_CACHE_VARIABLES = {
    "XDG_CACHE_HOME": "xdg",
    "UV_CACHE_DIR": "uv",
    "PIP_CACHE_DIR": "pip",
    "npm_config_cache": "npm",
    "YARN_CACHE_FOLDER": "yarn",
    "GOCACHE": "go-build",
    "PRE_COMMIT_HOME": "pre-commit",
}
SUPPORTED_COMMIT_HOOKS = ("pre-commit", "commit-msg")
HOOK_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class CheckPolicy:
    """The task's boundary, as it applies to checks and commit hooks."""

    sandboxed: bool
    network_access: bool
    cache_root: Path | None = None
    sandbox_home: Path | None = None

    @classmethod
    def for_task(
        cls,
        permission_mode: PermissionMode,
        network_access: bool,
        *,
        cache_root: Path | None = None,
        sandbox_home: Path | None = None,
    ) -> CheckPolicy:
        # Full access already lets the agent run anything outside the sandbox,
        # so its checks gain nothing from one.
        return cls(
            sandboxed=permission_mode != PermissionMode.full_access,
            network_access=network_access,
            cache_root=cache_root,
            sandbox_home=sandbox_home,
        )


def check_environment(
    base: Mapping[str, str],
    *,
    variables: Mapping[str, str] | None = None,
    ports: Mapping[str, int] | None = None,
    cache_root: Path | None = None,
) -> dict[str, str]:
    """The server's environment without what the agent's shell must not see."""
    environment = {
        name: value
        for name, value in base.items()
        if not _PRIVATE_NAME.search(name) and not name.upper().startswith(_PRIVATE_PREFIXES)
    }
    if cache_root is not None:
        for name, folder in _CACHE_VARIABLES.items():
            path = cache_root / folder
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
    # The project's own variables are an explicit choice, so they win.
    environment.update(variables or {})
    environment.update({name: str(value) for name, value in (ports or {}).items()})
    return environment


@dataclass(frozen=True)
class GitDirectories:
    """Where a repository keeps what a hook may change and what it must not."""

    common: Path
    private: Path

    def grants(self) -> dict[Path, str]:
        # Hooks may stage files and stash (the index, objects, refs and logs);
        # nothing that makes Git run a program or point somewhere else.
        grants: dict[Path, str] = {self.common: "write"}
        for name in ("config", "config.worktree", "hooks", "modules", "worktrees", "commondir"):
            grants[self.common / name] = "read"
        if self.private != self.common:
            grants[self.private] = "write"
            for name in ("config.worktree", "commondir", "gitdir"):
                grants[self.private / name] = "read"
        return grants


def sandbox_codex() -> Path:
    try:
        from codex_cli_bin import bundled_codex_path
    except ImportError as exc:
        raise WorkspaceError("The sandbox for project checks is not available on this computer") from exc
    return Path(bundled_codex_path())


def sandboxed_argv(
    argv: Sequence[str],
    policy: CheckPolicy,
    writable: Sequence[Path],
    *,
    git: GitDirectories | None = None,
    read_only: Sequence[Path] = (),
    codex: Path | None = None,
) -> list[str]:
    """Wrap a command so it runs inside the task's sandbox.

    ``read_only`` names paths inside a writable root that stay read-only, such
    as a repository's Git directory, which Codex protects only by default.
    """
    if not policy.sandboxed:
        return list(argv)
    temp = Path(tempfile.gettempdir()).resolve()
    roots = [Path(path).resolve() for path in writable]
    if policy.cache_root is not None:
        roots.append(policy.cache_root.resolve())
    filesystem: dict[str, str] = {":root": "read", ":slash_tmp": "write", str(temp): "write"}
    filesystem.update({str(root): "write" for root in roots})
    filesystem.update({str(Path(path).resolve()): "read" for path in read_only})
    if git is not None:
        filesystem.update({str(path.resolve()): access for path, access in git.grants().items()})
    sockets = ", ".join(f"{toml_string(str(path))}=\"allow\"" for path in (*roots, temp))
    network = f"enabled=true, allow_local_binding=true, unix_sockets={{{sockets}}}"
    if not policy.network_access:
        # Outbound traffic goes only through Codex's proxy, which allows no
        # domain; direct connections are refused by the sandbox itself.
        network += ", domains={}"
    files = ", ".join(f"{toml_string(path)}={toml_string(access)}" for path, access in filesystem.items())
    command = [
        str(codex or sandbox_codex()),
        "sandbox",
        "-c", 'default_permissions="checks"',
        "-c", f"permissions.checks.filesystem={{{files}}}",
        "-c", f"permissions.checks.network={{{network}}}",
    ]
    if not policy.network_access:
        command += ["-c", "features.network_proxy=true"]
    return [*command, "--", *argv]


def sandbox_environment(environment: Mapping[str, str], policy: CheckPolicy) -> dict[str, str]:
    """Keep the person's own Codex settings out of the sandbox launcher."""
    result = dict(environment)
    if policy.sandboxed and policy.sandbox_home is not None:
        policy.sandbox_home.mkdir(parents=True, exist_ok=True)
        result["CODEX_HOME"] = str(policy.sandbox_home)
    return result


def git_metadata(root: Path) -> list[Path]:
    """A folder's Git directories, which a check must not change."""
    paths = [root / ".git"]
    result = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir", "--git-common-dir"],
        cwd=str(root), capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        paths.extend((root / line).resolve() for line in result.stdout.splitlines() if line.strip())
    return paths


def git_directories(git_run, root: Path) -> GitDirectories:
    common = git_run(root, "rev-parse", "--git-common-dir").stdout.strip()
    private = git_run(root, "rev-parse", "--absolute-git-dir").stdout.strip()
    return GitDirectories(common=(root / common).resolve(), private=Path(private).resolve())


def hook_path(git_run, root: Path, name: str) -> Path | None:
    """The hook Git itself would run, honouring ``core.hooksPath``."""
    raw = git_run(root, "rev-parse", "--git-path", f"hooks/{name}").stdout.strip()
    path = Path(raw) if Path(raw).is_absolute() else root / raw
    if not path.is_file():
        return None
    if sys.platform != "win32" and not os.access(path, os.X_OK):
        # Git skips a hook it cannot execute, with only a hint.
        return None
    return path


def hook_argv(path: Path, *args: str) -> list[str]:
    if sys.platform != "win32":
        return [str(path), *args]
    # Git for Windows runs hooks through its bundled shell.
    shell = shutil.which("sh")
    if shell is None:
        raise WorkspaceError("Git's shell is not available to run this repository's commit hooks")
    return [shell, str(path), *args]


def run_sandboxed(
    argv: Sequence[str],
    *,
    cwd: Path,
    policy: CheckPolicy,
    environment: Mapping[str, str],
    writable: Sequence[Path],
    git: GitDirectories | None = None,
    timeout: int = HOOK_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        sandboxed_argv(argv, policy, writable, git=git),
        cwd=str(cwd),
        env=sandbox_environment(environment, policy),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        shell=False,
        timeout=timeout,
    )
