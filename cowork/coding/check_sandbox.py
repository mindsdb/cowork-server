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
from urllib.parse import urlsplit

from cowork.coding.contracts import PermissionMode
from cowork.coding.engines.codex_config import toml_string
from cowork.coding.workspace import WorkspaceError, _org_mode

# A new server variable stays private unless it is deliberately listed here.
# These names cover process launch, paths, locale and Git author identity on
# our supported platforms. Project-specific tools get overrides separately.
_PLATFORM_ENVIRONMENT = {
    "PATH", "PATHEXT", "HOME", "USER", "LOGNAME", "SHELL", "TMP", "TEMP", "TMPDIR",
    "SYSTEMROOT", "WINDIR", "COMSPEC", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "LC_TIME", "TZ",
    "EMAIL", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
    # Toolchain homes that test runners resolve from the environment.
    "JAVA_HOME", "GOPATH", "GOROOT", "CARGO_HOME", "RUSTUP_HOME", "NVM_DIR", "PYENV_ROOT",
    "VIRTUAL_ENV", "CONDA_PREFIX", "ANDROID_HOME", "ANDROID_SDK_ROOT",
    # Corporate networks: a CA bundle is a machine setting, not a secret.
    # Proxy URLs pass only without credentials, see _PROXY_VARIABLES.
    "NO_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS", "CURL_CA_BUNDLE",
}
_PROXY_VARIABLES = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
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
        or (name.upper() in _PROXY_VARIABLES and not _has_credentials(value))
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


# Files that make Git run a program or read its settings from somewhere else.
# A hook must not leave one behind for the next host-side Git command.
_GIT_CONTROL_FILES = ("config", "config.worktree", "commondir", "gitdir", "hooks", "modules")
_STASH_ENTRIES = ("stash", "stash.lock")
_GRANTS_FOLDERS_FOR_NEW_FILES = sys.platform.startswith("linux")


def _has_credentials(url: str) -> bool:
    """Whether a proxy URL carries a user name or password."""
    candidate = url if "://" in url else f"http://{url}"
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return True
    return "@" in parsed.netloc


@dataclass(frozen=True)
class GitDirectories:
    """Where a repository keeps what a hook may change and what it must not."""

    common: Path
    private: Path

    def grants(self) -> dict[Path, str]:
        # A linked worktree shares its object database and refs with other
        # checkouts, so a hook may write only staging data, new objects and
        # the stash. Only existing paths are listed: on Linux the sandbox
        # creates a stand-in for a missing path, which Git then misreads.
        grants = {self.common: "read", self.private: "write", self.common / "objects": "write"}
        protected = [self.common / "objects" / "info"]
        protected += [self.private / name for name in (*_GIT_CONTROL_FILES, "worktrees", "info", "refs", "logs/refs", "packed-refs", "HEAD")]
        # Only existing paths are listed on Linux, where the sandbox would
        # create a stand-in for a missing one. Elsewhere a missing control file
        # is listed too, so a sandboxed step cannot create it at all.
        grants.update({
            path: "read"
            for path in protected
            if path.exists() or (not _GRANTS_FOLDERS_FOR_NEW_FILES and path.name in _GIT_CONTROL_FILES)
        })
        # Stash-backed formatters such as lint-staged keep their backup in
        # refs/stash. Git writes a ref by renaming a lock file into place.
        # macOS and Windows can grant exactly those files. The Linux sandbox
        # cannot grant a file that does not exist yet, so there the refs
        # folder itself is writable while everything already in it, such as
        # branches, tags and notes, stays read-only.
        for refs in self._ref_folders():
            if _GRANTS_FOLDERS_FOR_NEW_FILES:
                grants[refs] = "write"
                grants.update({entry: "read" for entry in refs.iterdir() if entry.name not in _STASH_ENTRIES})
            else:
                grants.update({refs / name: "write" for name in _STASH_ENTRIES})
        return grants

    def snapshot(self) -> tuple[set[Path], set[Path]]:
        """Control files and top-level refs entries, to compare after a step."""
        files = {
            path
            for path in (self.private / name for name in _GIT_CONTROL_FILES)
            if path.exists() or path.is_symlink()
        }
        refs = {entry for folder in self._ref_folders() for entry in folder.iterdir() if entry.name not in _STASH_ENTRIES}
        return files, refs

    @property
    def task_owned(self) -> bool:
        """Whether the private Git directory belongs to a task worktree alone.

        A linked worktree's directory is written only by Git in that worktree.
        A local folder's Git directory is the person's repository, where
        another tool may be working at the same time.
        """
        return self.private != self.common

    def left_behind(self, before: tuple[set[Path], set[Path]]) -> tuple[set[Path], set[Path]]:
        """New control files in the task's Git directory, and new refs entries."""
        files, refs = self.snapshot()
        return files - before[0], refs - before[1]

    def _ref_folders(self) -> list[Path]:
        return [path for path in (self.common / "refs", self.common / "logs" / "refs") if path.is_dir()]


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
            for path, access in git.grants().items():
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
    if _org_mode():
        raise WorkspaceError("Local Code Project commands are not available on this deployment")
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
    if _org_mode():
        raise WorkspaceError("Local Code Project commands are not available on this deployment")
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
