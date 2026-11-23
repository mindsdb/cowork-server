"""Run project checks and commit hooks under the task's sandbox.

Checks and hooks run code the agent may have written, so they get the same
effective write permission as the agent. Writable tasks get space for build
outputs and tool caches. Outbound traffic follows the task's network setting.
Only selected platform settings and explicit project variables are inherited.
"""

from __future__ import annotations

import os
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

# A new server variable stays private unless it is deliberately listed here.
# These names cover process launch, paths, locale and Git author identity on
# our supported platforms. Project-specific tools get overrides separately.
_PLATFORM_ENVIRONMENT = {
    "PATH", "PATHEXT", "HOME", "USER", "LOGNAME", "SHELL", "TMP", "TEMP", "TMPDIR",
    "SYSTEMROOT", "WINDIR", "COMSPEC", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "LC_TIME", "TZ",
    "JAVA_HOME", "EMAIL", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
}
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
    read_only: bool = False
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
            read_only=permission_mode == PermissionMode.read_only,
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
    """Platform settings plus the variables explicitly assigned to the task."""
    environment = {
        name: value
        for name, value in base.items()
        if name.upper() in _PLATFORM_ENVIRONMENT
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
    branch_ref: str | None = None

    def grants(self, *, committing: bool = False) -> dict[Path, str]:
        # A linked worktree shares its object database and branch refs with
        # other checkouts. Limit writes to staging data and this task's ref.
        # Git stash uses process-specific temporary index files in the
        # private directory. Keep shared refs and executable configuration
        # protected even when this is the repository's primary checkout.
        grants = {self.common: "read", self.private: "write", self.common / "objects": "write"}
        for name in ("config", "config.worktree", "hooks", "modules", "worktrees", "info",
                     "refs", "logs/refs", "packed-refs", "packed-refs.lock", "commondir", "gitdir", "HEAD"):
            grants[self.private / name] = "read"
        grants[self.common / "objects/info"] = "read"
        # Stash-backed formatters need the stash ref even in a linked worktree.
        refs = ["refs/stash"]
        if committing and self.branch_ref:
            refs.append(self.branch_ref)
        if committing:
            grants[self.private / "HEAD"] = "write"
        for ref in refs:
            for prefix in ("", "logs/"):
                path = self.common / (prefix + ref)
                grants[path] = "write"
                grants[path.with_name(path.name + ".lock")] = "write"
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
    committing: bool = False,
) -> list[str]:
    """Wrap a command so it runs inside the task's sandbox.

    ``read_only`` names paths inside a writable root that stay read-only, such
    as a repository's Git directory, which Codex protects only by default.
    """
    if not policy.sandboxed:
        return list(argv)
    temp = Path(tempfile.gettempdir()).resolve()
    filesystem: dict[str, str] = {":root": "read"}
    roots: list[Path] = []
    if not policy.read_only:
        roots = [Path(path).resolve() for path in writable]
        if policy.cache_root is not None:
            roots.append(policy.cache_root.resolve())
        filesystem.update({":slash_tmp": "write", str(temp): "write"})
        filesystem.update({str(root): "write" for root in roots})
        filesystem.update({str(Path(path).resolve()): "read" for path in read_only})
        if git is not None:
            for path, access in git.grants(committing=committing).items():
                if path.resolve() != path:
                    raise WorkspaceError("Cannot grant commit access through symbolic links in Git metadata")
                filesystem[str(path)] = access
    network = "enabled=true"
    if not policy.read_only:
        sockets = ", ".join(f'{toml_string(str(path))}="allow"' for path in (*roots, temp))
        network += f", allow_local_binding=true, unix_sockets={{{sockets}}}"
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
    ref = git_run(root, "symbolic-ref", "--quiet", "HEAD", check=False).stdout.strip()
    branch_ref = ref if ref.startswith("refs/heads/") and ".." not in ref.split("/") else None
    return GitDirectories(common=(root / common).resolve(), private=Path(private).resolve(), branch_ref=branch_ref)


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
