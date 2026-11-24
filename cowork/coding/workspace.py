from __future__ import annotations

import difflib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING

from cowork.coding.contracts import (
    DiffFile,
    GitIdentity,
    GitState,
    PermissionMode,
    WorkspaceInspection,
    WorkspaceKind,
    WorkspaceReasonCode,
)
from cowork.coding.git_transport import ALLOWED_GIT_PROTOCOLS
from cowork.coding.local_copy import LocalCopyError, LocalCopyManager, force_remove
from cowork.coding.pending_removals import PendingRemovals
from cowork.coding.workspace_key import managed_key
from cowork.common.paths import O_NOFOLLOW, open_fd
from cowork.common.settings.app_settings import get_app_settings

if TYPE_CHECKING:
    from cowork.coding.check_sandbox import CheckPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageRoots:
    workspace: tuple[Path, ...]
    recovery: tuple[Path, ...]


class WorkspaceError(RuntimeError):
    """A workspace cannot be prepared or used as asked.

    ``reason`` names the cases a client words itself, in the person's
    language; ``folder`` is the project folder the message is about.
    """

    def __init__(
        self, message: str = "", *, reason: WorkspaceReasonCode | None = None, folder: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.folder = folder

    @classmethod
    def in_folder(cls, folder: str, exc: Exception) -> WorkspaceError:
        """``exc`` about one of a project's folders, named so the person knows which."""
        return cls(f"{folder}: {exc}", reason=getattr(exc, "reason", None), folder=folder)


class UnsafeGitMetadata(WorkspaceError):
    def __init__(self, common_dir: Path, detail: str) -> None:
        super().__init__(detail)
        self.common_dir = common_dir


@dataclass(frozen=True)
class RecordedCommit:
    common_dir: Path
    parent: str
    revision: str


class GitUnavailableError(WorkspaceError):
    """Git is not installed or cannot be found by this runtime."""


class GitIdentityMissingError(WorkspaceError):
    """Git has no author identity for commits on this computer.

    Raised before a commit is attempted so the caller can ask for a name and
    email instead of surfacing Git's rejection after a rolled-back attempt.
    """

    code = "git_identity_missing"

    def __init__(self, missing: list[str], detail: str = "") -> None:
        fields = " and ".join(field.removeprefix("user.") for field in missing) or "identity"
        super().__init__(
            f"Git needs your {fields} before it can commit on this computer. "
            "Set them once and the commit can be retried; nothing in the task is lost."
        )
        self.missing = list(missing)
        self.detail = detail


# A worktree is a Git checkout, so isolation needs a repository with a commit.
# Tasks on other folders work in them directly.
NOT_A_REPOSITORY = "This folder isn't a Git repository. Initialize Git to use a Worktree"
NO_COMMITS = "This repository has no commits yet. Make a first commit to use a Worktree"
# Review compares against a starting commit, so an in-place task without one
# works in the folder but has nothing to compare to.
NO_REVIEW_WITHOUT_GIT = "This folder isn't a Git repository, so the task's changes cannot be reviewed. Initialize Git to review them."
NO_REVIEW_WITHOUT_COMMITS = "This repository has no commits yet, so the task's changes cannot be reviewed. Make a first commit to review them."
WRITES_HELD = "Until you allow writes, the agent can read the folder but must ask before changing it."
MAX_DIFF_FILES = 250
MAX_TEXT_DIFF_BYTES = 2 * 1024 * 1024
MAX_TOTAL_DIFF_BYTES = 4 * 1024 * 1024
SNAPSHOT_TRUNCATED_MARKER = (
    "Snapshot truncated because this task has a large combined patch. Open the worktree to recover the rest.\n"
)
_TASK_ROOT_METADATA = (".DS_Store", "Thumbs.db", "desktop.ini")
_RELEASE_METADATA = "release.json"
# Ignored folders a worktree release may drop because tools recreate them on
# their own. Anything else ignored (dependencies, .env, build output) keeps
# the workspace in place.
_SELF_REBUILDING_CACHES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})


class ReleaseRefused(WorkspaceError):
    """A workspace cannot be released without losing something."""


def _write_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _org_mode() -> bool:
    return get_app_settings().tenancy_mode == "org"


@dataclass(frozen=True)
class PreparedWorkspace:
    source_path: Path
    workspace_path: Path
    kind: WorkspaceKind
    repository_root: Path | None
    base_revision: str | None
    source_dirty: bool
    warning: str | None


@dataclass(frozen=True)
class ApplyPlan:
    """A validated, typed handoff plan for one isolated workspace."""

    kind: WorkspaceKind
    git_patch: Path | None = None
    local_paths: tuple[str, ...] = ()

    @property
    def has_changes(self) -> bool:
        return self.git_patch is not None or bool(self.local_paths)

    def __post_init__(self) -> None:
        if self.kind == WorkspaceKind.git_worktree and self.local_paths:
            raise ValueError("a Git handoff plan cannot contain local-copy paths")
        if self.kind == WorkspaceKind.local_copy and self.git_patch is not None:
            raise ValueError("a local-copy handoff plan cannot contain a Git patch")


def _truncate_at_file_boundary(patch: Path, limit: int) -> int:
    """Cut a combined patch at the last whole-file diff that fits in ``limit`` bytes; return the kept size."""
    with patch.open("rb") as handle:
        head = handle.read(limit)
    boundary = head.rfind(b"\ndiff --git ")
    kept = boundary + 1 if boundary >= 0 else 0
    os.truncate(patch, kept)
    return kept


def _platform_config_args() -> tuple[str, ...]:
    """Per-invocation Git config the host platform needs.

    Windows paths are limited to 260 characters unless Git opts in, and deep
    dependency trees in a task worktree routinely exceed that. Passing it per
    command avoids editing the user's repository or global config.
    """
    if sys.platform == "win32":
        return ("-c", "core.longpaths=true")
    return ()


# Commands that read settings without running any program they name.
_SETTINGS_ONLY_COMMANDS = {"config", "rev-parse", "var"}
_FILTER_KEY = re.compile(r"^filter\..+\.(clean|smudge|process)$")


def names_task_file(command: str, root: Path) -> bool:
    """Whether a configured command runs a file inside ``root``.

    The setting itself may be the person's, but a script it names in the
    checkout is code the agent can edit.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    top = root.resolve()
    for token in tokens:
        if not token or token.startswith("-") or "%" in token:
            continue
        path = Path(os.path.expanduser(token))
        candidate = path if path.is_absolute() else top / path
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved.exists() and (resolved == top or top in resolved.parents):
            return True
    return False


def _host_safety_args(args: tuple[str, ...]) -> tuple[str, ...]:
    """Keep Git from running repository hooks in the server's process.

    A hook path can point into the task's own files (husky keeps hooks in
    ``.husky``), and Git runs some hooks, such as ``post-index-change``,
    from ordinary staging and status commands. Cowork runs the hooks it
    supports itself, inside the task's sandbox. Only hook discovery sees
    the repository's real hook path.
    """
    if args[:2] == ("rev-parse", "--git-path"):
        return ()
    # An fsmonitor setting names a program, which may live in the checkout.
    return ("-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false")


class GitRunner:
    """Shell-free Git boundary shared by macOS and Windows."""

    @staticmethod
    def _working_directory(path: Path) -> Path:
        """Resolve the user-selected local folder before giving it to Git.

        Code Mode intentionally supports repositories anywhere the desktop user
        can access.  The directory therefore cannot be restricted to one server
        root, but it must be a real directory before it crosses the process
        boundary.
        """
        try:
            resolved = path.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("Choose an available local folder") from exc
        if not resolved.is_dir():
            raise WorkspaceError("Choose an available local folder")
        return resolved

    def _filter_settings(self, cwd: Path) -> tuple[tuple[str, ...], bool]:
        """Disable checkout-based drivers and report whether other filters can still run."""
        listed = self.run(cwd, "config", "--null", "--get-regexp", r"^filter\.", check=False)
        if listed.returncode not in (0, 1):
            raise WorkspaceError(listed.stderr.strip() or "Could not read Git filter configuration")
        overrides: list[str] = []
        active = False
        for entry in listed.stdout.split("\0"):
            key, _, value = entry.partition("\n")
            if value and _FILTER_KEY.fullmatch(key):
                if names_task_file(value, cwd):
                    overrides += ["-c", f"{key}="]
                else:
                    active = True
        return tuple(overrides), active

    def run(
        self,
        cwd: Path,
        *args: str,
        check: bool = True,
        input_text: str | None = None,
        stdin: IO[bytes] | None = None,
        environment: dict[str, str] | None = None,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        if _org_mode():
            raise WorkspaceError("Local coding workspaces are not available on this deployment")
        working_directory = self._working_directory(cwd)
        try:
            child_environment = os.environ.copy()
            child_environment["GIT_TERMINAL_PROMPT"] = "0"
            child_environment.update(environment or {})
            # This is deliberately assigned after caller-provided values.
            # Project configuration cannot re-enable command-capable helpers
            # such as ``ext`` at the execution boundary.
            child_environment["GIT_ALLOW_PROTOCOL"] = ALLOWED_GIT_PROTOCOLS
            drivers, active_filters = (), False
            if args and args[0] not in _SETTINGS_ONLY_COMMANDS:
                drivers, active_filters = self._filter_settings(working_directory)
            sandboxed_read = args[:1] in {("status",), ("diff",)} and bool(
                drivers or active_filters or (working_directory / ".gitmodules").exists()
            )
            if sandboxed_read:
                # A read keeps the repository's filters, so a required one
                # still converts content. It runs them inside the sandbox.
                drivers = ()
            argv = ["git", *_platform_config_args(), *_host_safety_args(args), *drivers, *args]
            with ExitStack() as resources:
                output = subprocess.PIPE
                if sandboxed_read:
                    # A filter or a submodule can run project code. Outside a
                    # task operation, Git reads get no writes or network.
                    from cowork.coding import check_sandbox

                    if sys.platform == "win32":
                        # The offline account setup marker belongs to CODEX_HOME.
                        # Reuse it across reads to avoid repeating elevated setup.
                        home = Path(get_app_settings().coding.root_dir) / "check-sandbox"
                    else:
                        home = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix="cowork-git-read-")))
                    policy = check_sandbox.CheckPolicy.for_task(PermissionMode.read_only, False, sandbox_home=home)
                    child_environment = check_sandbox.check_environment(os.environ)
                    # Snapshot diffs use an isolated index and object store.
                    for name in ("GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
                        if environment and name in environment:
                            child_environment[name] = environment[name]
                    child_environment.update({
                        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": ALLOWED_GIT_PROTOCOLS,
                    })
                    child_environment = check_sandbox.sandbox_environment(child_environment, policy)
                    # Stream snapshot patches to the caller's output file. Git
                    # and its filters need no write access to the recovery folder.
                    options = args[:args.index("--")] if "--" in args else args
                    for option in options:
                        if args[0] == "diff" and option.startswith("--output="):
                            patch_path = working_directory / option.removeprefix("--output=")
                            output = resources.enter_context(patch_path.open("wb"))
                            argv.remove(option)
                            break
                    common = self.run(working_directory, "rev-parse", "--git-common-dir", check=False, environment=environment)
                    storage = check_sandbox.filter_storage((working_directory / common.stdout.strip()).resolve()) if common.returncode == 0 else []
                    argv = check_sandbox.sandboxed_argv(argv, policy, (), storage=storage)
                result = subprocess.run(
                    argv,
                    cwd=str(working_directory),  # lgtm[py/path-injection]
                    env=child_environment,
                    input=input_text,
                    stdin=stdin,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=output,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=timeout,
                    shell=False,
                )
        except FileNotFoundError as exc:
            raise GitUnavailableError("Git is not installed or is not available on PATH") from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorkspaceError(f"Git could not run: {exc}") from exc
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "Git command failed").strip()
            raise WorkspaceError(detail[:4_000])
        return result


class WorkspaceManager:
    def __init__(self, root: Path, git: GitRunner | None = None) -> None:
        self.root = root
        # Git worktrees and local folder copies share a per-task parent. Codex
        # goal turns receive only a thread-level workspace root, so this layout
        # lets one narrowly scoped root cover every folder in a Code Project.
        self.worktrees_root = root / "workspaces"
        self.legacy_worktrees_root = root / "worktrees"
        self.snapshots_root = root / "snapshots"
        self.git = git or GitRunner()
        self.local_copies = LocalCopyManager(root, self.worktrees_root)
        # Under the coding root so moving a task folder here is a same-disk rename.
        self.pending_removals = PendingRemovals(root / "pending-removal")
        self._mutation_lock = threading.RLock()
        self._refusals: dict[str, str] = {}
        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        self.snapshots_root.mkdir(parents=True, exist_ok=True)

    def inspect(self, raw_path: str) -> WorkspaceInspection:
        path = self._resolve_existing(raw_path)
        if path is None:
            return WorkspaceInspection(path=str(Path(raw_path).expanduser()), exists=False, is_directory=False, is_git=False)
        if not path.is_dir():
            return WorkspaceInspection(path=str(path), exists=True, is_directory=False, is_git=False)
        root = self._git_root(path)
        if root is None:
            return WorkspaceInspection(
                path=str(path),
                exists=True,
                is_directory=True,
                is_git=False,
            )
        status = self._status_lines(root)
        branch_result = self.git.run(root, "symbolic-ref", "--short", "-q", "HEAD", check=False)
        branch = branch_result.stdout.strip() or None
        revision = self._head_revision(root)
        return WorkspaceInspection(
            path=str(path),
            exists=True,
            is_directory=True,
            is_git=True,
            repository_root=str(root),
            branch=branch,
            revision=revision,
            dirty=bool(status),
            warning=(
                "The source repository has local changes. The task starts from the current HEAD in an isolated worktree; uncommitted source changes are not copied."
                if status and revision
                else None
            ),
        )

    def check_source(self, raw_path: str) -> None:
        """Reject a folder a worktree task cannot use, without the slower Git status work."""
        path = self._resolve_existing(raw_path)
        if path is None or not path.is_dir():
            raise WorkspaceError("Choose an existing local folder")
        root = self._git_root(path)
        if root is None:
            raise WorkspaceError(NOT_A_REPOSITORY, reason=WorkspaceReasonCode.not_a_repository)
        if self._head_revision(root) is None:
            raise WorkspaceError(NO_COMMITS, reason=WorkspaceReasonCode.no_commits)

    def initialize_git(self, raw_path: str) -> None:
        """Run ``git init`` in a folder that isn't in a Git repository yet.

        Nothing is committed: the user decides what the first commit holds, and
        a worktree is offered once there is one. Repositories already inside
        the folder stay separate repositories.
        """
        path = self._resolve_existing(raw_path)
        if path is None or not path.is_dir():
            raise WorkspaceError("Choose an existing local folder")
        if self._git_root(path) is not None:
            raise WorkspaceError("This folder is already in a Git repository")
        self.git.run(path, "init")

    def in_place_root(self, raw_path: str) -> Path:
        """Return the folder an in-place task on ``raw_path`` works in.

        Inside a Git repository that is the repository, so review and the
        one-task-per-folder check see all of it. Any other folder, including
        one holding several repositories, is worked in as it is.
        """
        path = self._resolve_existing(raw_path)
        if path is None or not path.is_dir():
            raise WorkspaceError("Choose an existing local folder")
        return self._git_root(path) or path

    def prepare_in_place(self, raw_path: str) -> PreparedWorkspace:
        """Point a task at the user's folder itself; nothing is created."""
        root = self.in_place_root(raw_path)
        inspection = self.inspect(str(root))
        if not inspection.is_git or not inspection.revision:
            return PreparedWorkspace(
                source_path=root,
                workspace_path=root,
                kind=WorkspaceKind.direct_folder,
                repository_root=root if inspection.is_git else None,
                base_revision=None,
                source_dirty=False,
                warning=NO_REVIEW_WITHOUT_COMMITS if inspection.is_git else NO_REVIEW_WITHOUT_GIT,
            )
        return PreparedWorkspace(
            source_path=root,
            workspace_path=root,
            kind=WorkspaceKind.direct_folder,
            repository_root=root,
            base_revision=inspection.revision,
            source_dirty=inspection.dirty,
            warning=(
                "The folder already had uncommitted changes. They appear in this task's review alongside the agent's."
                if inspection.dirty
                else None
            ),
        )

    def prepare(
        self,
        session_id: str,
        raw_path: str,
        base_branch: str | None = None,
        *,
        include_local_changes: bool = False,
    ) -> PreparedWorkspace:
        """Create an isolated worktree for a task."""
        with self._mutation_lock:
            inspection = self.inspect(raw_path)
            if not inspection.exists or not inspection.is_directory:
                raise WorkspaceError("Choose an existing local folder")
            if not inspection.is_git:
                raise WorkspaceError(NOT_A_REPOSITORY, reason=WorkspaceReasonCode.not_a_repository)

            repo = Path(inspection.repository_root or inspection.path)
            revision = inspection.revision
            if base_branch:
                revision = self.branch_revision(repo, base_branch)
                if revision is None:
                    raise WorkspaceError(f"Base branch is unavailable: {base_branch}")
            # Git cannot create a worktree until the repository has a commit.
            if revision is None:
                raise WorkspaceError(NO_COMMITS, reason=WorkspaceReasonCode.no_commits)

            worktree = self.worktrees_root / managed_key(session_id)
            if worktree.exists():
                raise WorkspaceError("A managed worktree already exists for this task")
            worktree.parent.mkdir(parents=True, exist_ok=True)
            self.git.run(repo, "worktree", "add", "--detach", str(worktree), revision)
            if include_local_changes and inspection.dirty:
                from cowork.coding.source_changes import copy_source_changes

                try:
                    copy_source_changes(self, repo, worktree, inspection.revision)
                except Exception:
                    self.git.run(repo, "worktree", "remove", "--force", str(worktree))
                    raise
            return PreparedWorkspace(
                source_path=repo,
                workspace_path=worktree,
                kind=WorkspaceKind.git_worktree,
                repository_root=repo,
                base_revision=revision,
                source_dirty=inspection.dirty,
                warning=None if include_local_changes else inspection.warning,
            )

    def git_state(
        self, source_path: str, workspace_path: str, checks: CheckPolicy | None = None, *, committed: bool = False,
    ) -> GitState:
        workspace = Path(workspace_path)
        root = self._git_root(workspace)
        if root is None:
            return GitState(is_git=False, worktree_path=str(workspace), source_path=source_path)
        branch_result = self.git.run(root, "symbolic-ref", "--short", "-q", "HEAD", check=False)
        branch = branch_result.stdout.strip() or None
        revision = self._head_revision(root)
        try:
            status_lines = self._status_lines(root, checks)
        except WorkspaceError:
            if not committed:
                raise
            # The commit is already recorded, so an unreadable status must not
            # report it as failed. A hook may have left changes behind, so the
            # state says the rest is unknown instead of clean.
            logger.warning("Could not read Git status after committing in %s", root, exc_info=True)
            status_lines = None
        return GitState(
            is_git=True,
            branch=branch,
            revision=revision,
            detached=branch is None,
            dirty=bool(status_lines),
            status_lines=status_lines or [],
            status_checked=status_lines is not None,
            worktree_path=str(root),
            source_path=source_path,
        )

    def fork(
        self,
        session_id: str,
        source_path: str,
        workspace_path: str,
        kind: WorkspaceKind,
        base_revision: str | None,
    ) -> PreparedWorkspace:
        with self._mutation_lock:
            source = Path(source_path)
            current = Path(workspace_path)
            # A legacy direct-folder task has no starting commit, so its fork
            # shares the folder. An in-place task's fork gets a worktree that
            # carries its changes; two tasks never edit the same checkout.
            if kind == WorkspaceKind.direct_folder and not base_revision:
                return PreparedWorkspace(
                    source_path=source,
                    workspace_path=current,
                    kind=kind,
                    repository_root=None,
                    base_revision=None,
                    source_dirty=False,
                    warning=None,
                )

            if kind == WorkspaceKind.local_copy:
                try:
                    prepared = self.local_copies.fork(session_id, source, current)
                except LocalCopyError as exc:
                    raise WorkspaceError(str(exc)) from exc
                return PreparedWorkspace(
                    source_path=prepared.source,
                    workspace_path=prepared.workspace,
                    kind=kind,
                    repository_root=None,
                    base_revision=None,
                    source_dirty=False,
                    warning=None,
                )

            target = self.worktrees_root / managed_key(session_id)
            if target.exists():
                raise WorkspaceError("A managed worktree already exists for this task")
            revision = self.git.run(current, "rev-parse", "HEAD").stdout.strip()
            self.git.run(source, "worktree", "add", "--detach", str(target), revision)
            try:
                snapshot = self._snapshot_changes(session_id, current, revision, "fork.patch")
                if snapshot is not None:
                    self._apply_patch(target, snapshot)
            except Exception:
                self.git.run(source, "worktree", "remove", "--force", str(target), check=False)
                self.git.run(source, "worktree", "prune", check=False)
                raise
            return PreparedWorkspace(
                source_path=source,
                workspace_path=target,
                kind=WorkspaceKind.git_worktree,
                repository_root=source,
                base_revision=base_revision,
                source_dirty=False,
                warning=None,
            )

    def diff(self, workspace_path: str, base_revision: str | None) -> list[DiffFile]:
        root = Path(workspace_path)
        if not base_revision:
            try:
                return self.local_copies.diff(root)
            except LocalCopyError:
                return []
        if self._git_root(root) is None:
            return []
        changes = self._changes_since(root, base_revision)
        working_status = {path: status for status, path in self._status_entries(root)}
        files: list[DiffFile] = []
        remaining_bytes = MAX_TOTAL_DIFF_BYTES
        for status, rel_path in changes[:MAX_DIFF_FILES]:
            path = root / rel_path
            staged, unstaged = self._review_status(working_status.get(rel_path))
            if remaining_bytes <= 0:
                files.append(
                    DiffFile(
                        path=rel_path,
                        status=status,
                        staged=staged,
                        unstaged=unstaged,
                        patch="Inline diff omitted because this task has a large combined patch. Open the worktree to review it.",
                    )
                )
                continue
            if status == "??":
                item = self._untracked_diff(root, path, rel_path).model_copy(
                    update={"staged": staged, "unstaged": unstaged}
                )
                files.append(item)
                remaining_bytes -= len(item.patch.encode("utf-8"))
                continue
            if self._tracked_file_is_large(root, path, rel_path, base_revision):
                item = DiffFile(
                    path=rel_path,
                    status=status,
                    staged=staged,
                    unstaged=unstaged,
                    patch="Large file changed. Open the worktree to review it.",
                    binary=True,
                )
                files.append(item)
                remaining_bytes -= len(item.patch)
                continue
            patch = self.git.run(
                root,
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--binary",
                "--unified=3",
                base_revision,
                "--",
                rel_path,
            ).stdout
            additions, deletions = self._counts(patch)
            item = DiffFile(
                path=rel_path,
                status=status,
                staged=staged,
                unstaged=unstaged,
                additions=additions,
                deletions=deletions,
                patch=patch,
                binary="GIT binary patch" in patch or "Binary files" in patch,
            )
            files.append(item)
            remaining_bytes -= len(item.patch.encode("utf-8"))
        omitted = len(changes) - MAX_DIFF_FILES
        if omitted > 0:
            files.append(
                DiffFile(
                    path=f"… {omitted} additional files",
                    status="…",
                    patch="Cowork limits the inline review surface. Open the worktree to inspect the remaining files.",
                )
            )
        return files

    def review_file_action(self, workspace_path: str, rel_path: str, action: str) -> None:
        """Apply one bounded Git review action to a task worktree file."""

        root = Path(workspace_path)
        path = self._validated_git_path(rel_path)
        if self._git_root(root) != root.resolve():
            raise WorkspaceError("Review actions require a Git task workspace")
        with self._mutation_lock:
            if action == "stage":
                self._add(root, "--", path)
                return
            if action == "unstage":
                result = self.git.run(root, "restore", "--staged", "--", path, check=False)
                if result.returncode != 0:
                    self.git.run(root, "reset", "HEAD", "--", path)
                return
            if action == "discard":
                status = dict((entry_path, entry_status) for entry_status, entry_path in self._status_entries(root)).get(path)
                if status == "??":
                    target = root / Path(*PurePosixPath(path).parts)
                    parent = target.parent.resolve()
                    try:
                        parent.relative_to(root.resolve())
                    except ValueError as exc:
                        raise WorkspaceError("The selected file is outside this task workspace") from exc
                    if target.is_dir() and not target.is_symlink():
                        raise WorkspaceError("Discard files individually")
                    target.unlink(missing_ok=True)
                    return
                self.git.run(root, "restore", "--source=HEAD", "--staged", "--worktree", "--", path)
                return
            raise WorkspaceError("Unsupported review action")

    @staticmethod
    def _validated_git_path(value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if value.startswith(("/", "\\")) or path.is_absolute() or not path.parts or path == PurePosixPath("."):
            raise WorkspaceError("Choose a file inside this task workspace")
        if any(part in {"", ".", ".."} for part in path.parts) or "\x00" in value:
            raise WorkspaceError("Choose a file inside this task workspace")
        return path.as_posix()

    @staticmethod
    def _review_status(status: str | None) -> tuple[bool, bool]:
        if not status:
            return False, False
        if status == "??":
            return False, True
        return status[0] not in {" ", "?"}, status[1] not in {" ", "?"}

    def _changes_since(self, root: Path, base_revision: str) -> list[tuple[str, str]]:
        """Return committed, staged, unstaged, and untracked paths vs base."""
        raw = self.git.run(
            root,
            "diff",
            "--name-status",
            "--find-renames",
            "-z",
            base_revision,
        ).stdout
        parts = raw.split("\0")
        changes: dict[str, str] = {}
        index = 0
        while index < len(parts):
            status = parts[index]
            index += 1
            if not status:
                continue
            if index >= len(parts):
                break
            path = parts[index]
            index += 1
            if status.startswith(("R", "C")):
                if index >= len(parts):
                    break
                path = parts[index]
                index += 1
            if path:
                changes[self._validated_git_path(path)] = status
        for status, path in self._status_entries(root):
            if status == "??":
                changes[path] = status
        return [(status, path) for path, status in changes.items()]

    def create_branch(self, workspace_path: str, name: str) -> GitState:
        with self._mutation_lock:
            root = Path(workspace_path)
            valid = self.git.run(root, "check-ref-format", "--branch", name, check=False)
            if valid.returncode != 0 or valid.stdout.strip() != name:
                raise WorkspaceError("Enter a valid Git branch name")
            self.git.run(root, "switch", "-c", name)
            return self.git_state(str(root), str(root))

    def branch_revision(self, repository_root: str | Path, name: str) -> str | None:
        """Resolve a validated branch name without allowing ref option injection."""
        root = Path(repository_root)
        valid = self.git.run(root, "check-ref-format", "--branch", name, check=False)
        if valid.returncode != 0 or valid.stdout.strip() != name:
            return None
        for prefix in ("refs/heads/", "refs/remotes/", "refs/remotes/origin/"):
            resolved = self.git.run(root, "rev-parse", "--verify", f"{prefix}{name}^{{commit}}", check=False)
            if resolved.returncode == 0:
                return resolved.stdout.strip()
        return None

    def commit(
        self, workspace_path: str, message: str, checks: CheckPolicy | None = None, *,
        workspace_kind: WorkspaceKind | None = None,
        on_recorded: Callable[[RecordedCommit], None] | None = None,
    ) -> GitState:
        """Commit everything in the workspace.

        Staging can run programs the repository names, such as a clean filter
        script in the checkout, so with ``checks`` it runs under the task's
        permissions, followed by the supported hooks. The commit itself is
        then recorded on the host with Git plumbing that never reads the
        working tree, so signing keeps the person's own environment.
        """
        with self._mutation_lock:
            root = Path(workspace_path)
            if not message.strip():
                raise WorkspaceError("Commit message cannot be empty")
            if checks is not None and checks.read_only:
                raise WorkspaceError("This task is read-only and cannot commit changes")
            # Before staging anything: a direct-folder task commits through
            # here without the project-level preflight.
            self.check_git_identity(str(root))
            self.check_no_operation_in_progress(str(root))
            with tempfile.TemporaryDirectory(prefix="cowork-commit-", ignore_cleanup_errors=True) as scratch:
                message_file = Path(scratch) / "COMMIT_EDITMSG"
                message_file.write_text(message.strip() + "\n", encoding="utf-8")
                new_objects = Path(scratch).resolve() / "objects"
                new_objects.mkdir()
                if checks is None:
                    self._add(root, "--all")
                else:
                    self._run_sandboxed_git_step(
                        root, checks, "Staging", ["git", *_platform_config_args(), "add", "--all"], [root], workspace_kind,
                        new_objects,
                    )
                if checks is not None:
                    self._run_commit_hooks(root, message_file, checks, workspace_kind, new_objects)
                record = self._record_commit(root, message_file.read_text(encoding="utf-8"))
                if on_recorded is not None:
                    on_recorded(record)
            return self.git_state(str(root), str(root), checks, committed=True)

    def check_no_operation_in_progress(self, workspace_path: str) -> None:
        """Refuse to commit over a merge, cherry-pick, revert or rebase Git is in the middle of.

        Cowork records a plain commit, which would drop the extra parent and
        leave Git still reporting the operation.
        """
        root = Path(workspace_path)
        for marker, operation in (
            ("MERGE_HEAD", "merge"),
            ("CHERRY_PICK_HEAD", "cherry-pick"),
            ("REVERT_HEAD", "revert"),
            ("rebase-merge", "rebase"),
            ("rebase-apply", "rebase"),
        ):
            raw = self.git.run(root, "rev-parse", "--git-path", marker).stdout.strip()
            path = Path(raw) if Path(raw).is_absolute() else root / raw
            if path.exists():
                raise WorkspaceError(f"A {operation} is in progress in {root.name}. Finish it with Git, then commit again.")

    def _record_commit(self, root: Path, message: str) -> RecordedCommit:
        """Commit the index without reading the working tree.

        Git runs from an empty folder with the repository named explicitly,
        so a signing program cannot pick up project code from the task's
        folder through its own module or plugin lookup.
        """
        git_dir = self.git.run(root, "rev-parse", "--absolute-git-dir").stdout.strip()
        with tempfile.TemporaryDirectory(prefix="cowork-record-", ignore_cleanup_errors=True) as neutral:
            return self._record_commit_in(Path(neutral), {"GIT_DIR": git_dir}, root, message)

    def _record_commit_in(self, cwd: Path, environment: dict[str, str], root: Path, message: str) -> RecordedCommit:
        def git(*args: str, **kwargs):
            return self.git.run(cwd, *args, environment=environment, **kwargs)

        message = git("stripspace", input_text=message).stdout
        if not message.strip():
            raise WorkspaceError("Commit message cannot be empty")
        index = self.git.run(root, "rev-parse", "--path-format=absolute", "--git-path", "index").stdout.strip()
        tree = self._write_tree(root, Path(index))
        parent = git("rev-parse", "--verify", "--quiet", "HEAD^{commit}", check=False).stdout.strip()
        if parent and tree == git("rev-parse", f"{parent}^{{tree}}").stdout.strip():
            raise WorkspaceError("There are no changes to commit")
        arguments = ["commit-tree", tree, *(("-p", parent) if parent else ())]
        signing = git("config", "--type=bool", "--get", "commit.gpgSign", check=False)
        if signing.returncode not in (0, 1):
            # Unset exits 1. Anything else is a setting Git itself would refuse
            # to commit with, so it must not become an unsigned commit.
            raise WorkspaceError(signing.stderr.strip() or "Could not read the commit signing setting")
        if signing.stdout.strip() == "true":
            self._refuse_task_signing_program(root)
            arguments.append("-S")
        revision = git(*arguments, input_text=message).stdout.strip()
        subject = message.splitlines()[0][:200]
        common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()).resolve()
        git("update-ref", "-m", f"commit: {subject}", "HEAD", revision, *((parent,) if parent else ()))
        return RecordedCommit(common, parent, revision)

    def _refuse_task_signing_program(self, root: Path) -> None:
        """Sign only with a program outside the task's own files."""
        for key in ("gpg.program", "gpg.openpgp.program", "gpg.x509.program", "gpg.ssh.program", "gpg.ssh.defaultKeyCommand"):
            value = self.git.run(root, "config", "--get", key, check=False).stdout.strip()
            if value and names_task_file(value, root):
                raise WorkspaceError(
                    f"Commit signing runs {value!r} from this task's folder, so Cowork won't sign with it. "
                    "Point the signing program outside the repository, or commit from a terminal."
                )

    def _run_commit_hooks(
        self, root: Path, message_file: Path, checks: CheckPolicy, workspace_kind: WorkspaceKind | None,
        new_objects: Path,
    ) -> None:
        from cowork.coding import check_sandbox

        hooks = {name: check_sandbox.hook_path(self.git.run, root, name) for name in check_sandbox.SUPPORTED_COMMIT_HOOKS}
        for name, path in hooks.items():
            if path is None:
                continue
            arguments = (str(message_file),) if name == "commit-msg" else ()
            self._run_sandboxed_git_step(
                root, checks, f"The {name} hook", check_sandbox.hook_argv(path, *arguments), [root, message_file.parent],
                workspace_kind, new_objects,
            )

    def _run_sandboxed_git_step(
        self, root: Path, checks: CheckPolicy, label: str, argv: list[str], writable: list[Path],
        workspace_kind: WorkspaceKind | None, new_objects: Path, *, environment: dict[str, str] | None = None,
    ) -> None:
        """Run a command that may change the index, under the task's permissions.

        The step writes new Git objects to ``new_objects`` and only reads the
        repository's own store. Cowork copies them in when the step ends.
        """
        from cowork.coding import check_sandbox

        index = self.git.run(root, "rev-parse", "--git-path", "index").stdout.strip()
        extra = environment or {}
        environment = check_sandbox.check_environment(os.environ, cache_root=checks.cache_root)
        environment.update({
            "GIT_INDEX_FILE": str((root / index).resolve()),
            **extra,
            "GIT_EDITOR": ":",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": ALLOWED_GIT_PROTOCOLS,
            # Git inside the sandbox runs no repository hook either.
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": os.devnull,
        })
        directories = check_sandbox.git_directories(self.git.run, root)
        environment.update({
            "GIT_OBJECT_DIRECTORY": str(new_objects),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(directories.common / "objects"),
        })
        # Local tasks may use linked worktrees the user created independently.
        task_owned = workspace_kind == WorkspaceKind.git_worktree and directories.private != directories.common
        before = directories.snapshot()
        result, failure = None, None
        try:
            result = check_sandbox.run_sandboxed(
                argv,
                cwd=root,
                policy=checks,
                environment=environment,
                writable=[*writable, new_objects],
                git=directories,
                timeout=check_sandbox.HOOK_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            failure = WorkspaceError(f"{label} did not finish in time")
            failure.__cause__ = exc
        except OSError as exc:
            failure = WorkspaceError(f"{label} could not run: {exc}")
            failure.__cause__ = exc
        # However the step ended, a new control file must not reach a
        # host-side Git command, where it would run code outside the sandbox.
        # A task worktree's own Git directory has no other writer, so what
        # appears there is removed. Anywhere else Cowork cannot tell who made
        # it, so it leaves it in place and stops before running Git again.
        planted, added_refs = directories.left_behind(before)
        if task_owned:
            for planted_path in planted:
                if planted_path.is_dir() and not planted_path.is_symlink():
                    force_remove(planted_path)
                else:
                    planted_path.unlink(missing_ok=True)
        unexplained = sorted(str(path) for path in added_refs | (set() if task_owned else planted))
        if unexplained:
            raise UnsafeGitMetadata(
                directories.common,
                f"Git files appeared while {label.lower()} ran, so nothing was committed: {', '.join(unexplained)}. "
                "Cowork left them in place. Remove any you didn't create, then commit again."
            ) from failure
        # Also after a failure: a formatter's stash may already name them.
        self._keep_new_objects(root, new_objects)
        if planted:
            raise WorkspaceError(f"{label} tried to change Git's configuration, so nothing was committed") from failure
        if failure is not None:
            raise failure
        if result.returncode != 0:
            output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
            raise WorkspaceError((f"{label} failed" + (f":\n{output}" if output else ""))[:4_000])

    def _add(self, root: Path, *args: str, environment: dict[str, str] | None = None) -> None:
        """Stage files outside a commit.

        A filter script in the task is code the agent can edit, so with one
        configured the staging runs sandboxed with the filter in place. Host
        Git would have to switch the filter off, which fails a required one.
        """
        scripted, _ = self.git._filter_settings(self.git._working_directory(root))
        if not scripted:
            self.git.run(root, "add", *args, environment=environment)
            return
        from cowork.coding.check_sandbox import CheckPolicy

        # No task is at hand here, so this is the narrowest policy that can stage.
        policy = CheckPolicy.for_task(PermissionMode.workspace, False, sandbox_home=self.root / "check-sandbox")
        with tempfile.TemporaryDirectory(prefix="cowork-stage-", ignore_cleanup_errors=True) as scratch:
            new_objects = Path(scratch).resolve() / "objects"
            new_objects.mkdir()
            self._run_sandboxed_git_step(
                root, policy, "Staging", ["git", *_platform_config_args(), "add", *args], [root], None, new_objects,
                environment=environment,
            )

    def _keep_new_objects(self, root: Path, new_objects: Path) -> None:
        """Copy the objects a sandboxed step wrote into the repository.

        Git repacks and unpacks them, which names each one by its real
        content, so a step can add objects but never replace or remove one.
        """
        if new_objects.is_symlink() or not new_objects.is_dir():
            raise WorkspaceError("The commit's new Git objects were moved, so nothing was committed")
        # An alternates file would pull in objects from another store.
        info = new_objects / "info"
        if info.is_symlink() or info.is_file():
            info.unlink()
        else:
            shutil.rmtree(info, ignore_errors=True)
        store = {"GIT_OBJECT_DIRECTORY": str(new_objects)}
        names = self.git.run(root, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)", environment=store).stdout
        if names.strip():
            with tempfile.TemporaryDirectory(prefix="cowork-objects-", ignore_cleanup_errors=True) as packing:
                pack = self.git.run(root, "pack-objects", "-q", str(Path(packing) / "new"), input_text=names, environment=store)
                with (Path(packing) / f"new-{pack.stdout.strip()}.pack").open("rb") as handle:
                    self.git.run(root, "unpack-objects", "-q", stdin=handle)
        force_remove(new_objects)
        new_objects.mkdir()

    def check_policy(self, owner_id: str, permission_mode: PermissionMode, network_access: bool) -> CheckPolicy:
        from cowork.coding.check_sandbox import CheckPolicy

        return CheckPolicy.for_task(
            permission_mode,
            network_access,
            cache_root=self.root / "check-caches" / managed_key(owner_id),
            sandbox_home=self.root / "check-sandbox",
        )

    def check_git_identity(self, workspace_path: str) -> None:
        """Raise GitIdentityMissingError unless Git can name an author for this repository.

        ``git var GIT_AUTHOR_IDENT`` and ``GIT_COMMITTER_IDENT`` apply exactly
        the rules ``git commit`` would (environment, repository, global and
        system configuration, and Git's own guess where it allows one), so this
        never blocks a commit Git would accept and never lets one through that
        Git would reject.
        """
        root = Path(workspace_path)
        # Git needs both identities; an environment that supplies only the
        # author's would pass an author-only probe and still fail the commit.
        for ident in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
            probe = self.git.run(root, "var", ident, check=False)
            if probe.returncode != 0 or not probe.stdout.strip():
                break
        else:
            return
        missing = [
            field
            for field in ("user.name", "user.email")
            if not self.git.run(root, "config", "--get", field, check=False).stdout.strip()
        ]
        raise GitIdentityMissingError(missing, detail=(probe.stderr or probe.stdout).strip()[:1_000])

    def git_identity(self) -> GitIdentity:
        """The globally configured author identity, unset fields as None."""
        home = Path.home()
        values: dict[str, str | None] = {}
        for field in ("name", "email"):
            result = self.git.run(home, "config", "--global", "--get", f"user.{field}", check=False)
            values[field] = result.stdout.strip() or None
        return GitIdentity(**values)

    def set_git_identity(self, name: str, email: str) -> GitIdentity:
        """Write the global author identity for whichever fields are still unset.

        Values already configured are kept, so a wrong guess can never
        overwrite an identity the user set up themselves.
        """
        current = self.git_identity()
        home = Path.home()
        if not current.name:
            self.git.run(home, "config", "--global", "user.name", name.strip())
        if not current.email:
            self.git.run(home, "config", "--global", "user.email", email.strip())
        return self.git_identity()

    def rollback_commit(self, workspace_path: str, record: RecordedCommit) -> None:
        """Remove a task commit while retaining its file changes for retry.

        Like ``git reset --mixed``, but without refreshing the index from the
        working tree, which can run a repository's filters on the host.
        """
        root = Path(workspace_path)
        self.git.run(root, "update-ref", "-m", "rollback: cowork commit", "HEAD", record.parent, record.revision)
        self.git.run(root, "read-tree", record.parent)

    def apply_to_source(self, session_id: str, source_path: str, workspace_path: str, base_revision: str | None) -> Path | None:
        with self._mutation_lock:
            plan = self.preflight_apply(session_id, source_path, workspace_path, base_revision)
            try:
                self.apply_checked(source_path, workspace_path, plan)
            except Exception:
                if plan.kind == WorkspaceKind.local_copy:
                    self.rollback_checked(source_path, workspace_path, plan)
                raise
            return plan.git_patch

    def preflight_apply(
        self,
        session_id: str,
        source_path: str,
        workspace_path: str,
        base_revision: str | None,
    ) -> ApplyPlan:
        if not base_revision:
            try:
                paths = self.local_copies.preflight(Path(source_path), Path(workspace_path))
                return ApplyPlan(kind=WorkspaceKind.local_copy, local_paths=tuple(paths))
            except LocalCopyError as exc:
                raise WorkspaceError(str(exc)) from exc
        source = Path(source_path)
        workspace = Path(workspace_path)
        snapshot = self._snapshot_changes(session_id, workspace, base_revision, "handoff.patch")
        if snapshot is None:
            return ApplyPlan(kind=WorkspaceKind.git_worktree)
        check = self._apply_patch(source, snapshot, "--check", check=False)
        if check.returncode != 0:
            detail = (check.stderr or check.stdout or "The changes conflict with the source working tree").strip()
            raise WorkspaceError(f"Handoff stopped before changing the source: {detail[:2_000]}")
        return ApplyPlan(kind=WorkspaceKind.git_worktree, git_patch=snapshot)

    def apply_checked(
        self,
        source_path: str,
        workspace_path: str,
        plan: ApplyPlan,
    ) -> None:
        if not plan.has_changes:
            return
        if plan.kind == WorkspaceKind.local_copy:
            try:
                self.local_copies.apply_checked(
                    Path(source_path),
                    Path(workspace_path),
                    list(plan.local_paths),
                )
            except LocalCopyError as exc:
                raise WorkspaceError(str(exc)) from exc
            return
        if plan.kind != WorkspaceKind.git_worktree or plan.git_patch is None:
            raise WorkspaceError("Invalid Git handoff plan")
        self._apply_patch(Path(source_path), plan.git_patch)

    def rollback_checked(
        self,
        source_path: str,
        workspace_path: str,
        plan: ApplyPlan,
    ) -> None:
        """Undo an applied handoff plan without discarding task workspace changes."""
        if not plan.has_changes:
            return
        if plan.kind == WorkspaceKind.local_copy:
            try:
                self.local_copies.rollback_checked(
                    Path(source_path),
                    Path(workspace_path),
                    list(plan.local_paths),
                )
            except LocalCopyError as exc:
                raise WorkspaceError(str(exc)) from exc
            return
        if plan.kind != WorkspaceKind.git_worktree or plan.git_patch is None:
            raise WorkspaceError("Invalid Git handoff plan")
        self._apply_patch(Path(source_path), plan.git_patch, "--reverse")

    def cleanup(
        self,
        session_id: str,
        source_path: str,
        workspace_path: str,
        kind: WorkspaceKind,
        base_revision: str | None,
        *,
        keep_recovery: bool = True,
    ) -> None:
        with self._mutation_lock:
            # Without recovery the task is being deleted, so its folders are
            # moved aside and removed in the background instead of holding the
            # request for as long as a large tree takes to remove.
            if kind == WorkspaceKind.local_copy:
                try:
                    self.local_copies.cleanup(
                        session_id,
                        Path(workspace_path),
                        keep_recovery=keep_recovery,
                        remove=None if keep_recovery else self._discard,
                    )
                except LocalCopyError as exc:
                    raise WorkspaceError(str(exc)) from exc
                return
            if kind != WorkspaceKind.git_worktree:
                return
            relative = managed_key(session_id)
            expected = {
                (self.worktrees_root / relative).resolve(),
                (self.legacy_worktrees_root / relative).resolve(),
            }
            actual = Path(workspace_path).resolve()
            if actual not in expected:
                raise WorkspaceError("Refusing to remove an unmanaged worktree path")
            source = Path(source_path)
            if not keep_recovery:
                self._discard(actual)
                if source.is_dir():
                    # The moved folder no longer exists where Git registered it,
                    # so prune drops the registration and frees the task branch.
                    # A failure only leaves a stale entry Git prunes itself later,
                    # which must not stop the task from being deleted.
                    self.git.run(source, "worktree", "prune", check=False)
                return
            if actual.exists():
                if base_revision and source.is_dir():
                    self._snapshot_changes(session_id, actual, base_revision, "cleanup.patch", truncate=True)
                if source.is_dir():
                    self.git.run(source, "worktree", "remove", "--force", str(actual))
                else:
                    # The user may move/delete the source checkout after the
                    # task was created. The managed path has already passed
                    # the ownership check above; removing it directly makes
                    # task deletion durable without touching arbitrary files.
                    shutil.rmtree(actual)
            if source.is_dir():
                self.git.run(source, "worktree", "prune")

    def usage_roots(self) -> UsageRoots:
        """Folders holding a ``<task>`` directory of each task's managed storage.

        These are what storage reporting measures, which is wider than what
        task cleanup prunes: recovery data outlives a task's workspace.
        """
        return UsageRoots(
            workspace=(
                # Folders waiting here belong to no task, so they count as removed.
                self.pending_removals.root,
                self.worktrees_root,
                self.legacy_worktrees_root,
                self.local_copies.legacy_copies_root,
                self.local_copies.baselines_root,
            ),
            recovery=(self.snapshots_root,),
        )

    def release(
        self,
        key: str,
        source_path: str,
        workspace_path: str,
        kind: WorkspaceKind,
    ) -> bool:
        """Remove a task workspace to reclaim disk, keeping what restores it.

        Returns False, leaving the workspace untouched, when it cannot be
        released safely; ``release_refusal`` then says why. Only worktrees
        are released. Their commits, staged state and working files (untracked
        included) are kept as commits pinned by a ref in the source
        repository, with the branch.
        """
        with self._mutation_lock:
            self._refusals.pop(key, None)
            try:
                if kind != WorkspaceKind.git_worktree:
                    raise ReleaseRefused("only worktrees can be released")
                return self._release_worktree(key, Path(source_path), workspace_path)
            except ReleaseRefused as refusal:
                self._refusals[key] = str(refusal)
                logger.info("Keeping the workspace %s: %s", key, refusal)
                return False

    def release_refusal(self, key: str) -> str | None:
        """Why the last release of ``key`` was refused, if it was."""
        return self._refusals.get(key)

    def _release_worktree(self, key: str, source: Path, workspace_path: str) -> bool:
        actual = self._managed_worktree(key, workspace_path)
        if not actual.exists():
            raise ReleaseRefused("the workspace folder is missing")
        if not source.is_dir():
            raise ReleaseRefused("the source repository is unavailable")
        head = self._head_revision(actual)
        if head is None:
            raise ReleaseRefused("the workspace has no commit to restore from")
        if self._has_nested_repository(actual):
            raise ReleaseRefused("a repository inside the workspace has history that cannot be saved")
        if self.git.run(actual, "ls-files", "--unmerged").stdout:
            raise ReleaseRefused("the workspace has unresolved merge conflicts")
        unsaved = self._unsaved_ignored(actual)
        if unsaved:
            raise ReleaseRefused(
                "ignored files would be lost (" + ", ".join(unsaved[:3]) + ("…" if len(unsaved) > 3 else "") + ")"
            )
        branch_result = self.git.run(actual, "symbolic-ref", "-q", "--short", "HEAD", check=False)
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
        index_commit, files_commit = self._save_worktree_state(actual, head)
        release_dir = self.snapshots_root / managed_key(key)
        release_dir.mkdir(parents=True, exist_ok=True)
        self.git.run(source, "update-ref", self._release_ref(key), files_commit)
        metadata = {
            "format": 2,
            "kind": WorkspaceKind.git_worktree.value,
            "head": head,
            "branch": branch,
            "index": index_commit,
            "files": files_commit,
        }
        try:
            _write_atomically(release_dir / _RELEASE_METADATA, json.dumps(metadata))
        except OSError as exc:
            self.git.run(source, "update-ref", "-d", self._release_ref(key), check=False)
            raise ReleaseRefused(f"the saved state could not be written: {exc}") from exc
        self.git.run(source, "worktree", "remove", "--force", str(actual))
        self.git.run(source, "worktree", "prune", check=False)
        return True

    def _unsaved_ignored(self, worktree: Path) -> list[str]:
        """Ignored entries a release would lose; caches that rebuild themselves are fine."""
        listed = self.git.run(
            worktree, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z",
        ).stdout
        return [
            path for path in filter(None, listed.split("\0"))
            if not any(part in _SELF_REBUILDING_CACHES for part in PurePosixPath(path.rstrip("/")).parts)
        ]

    def _write_tree(self, root: Path, index: Path) -> str:
        """Write an index as a tree without reading the working files.

        Writing an index makes Git re-read any file whose timestamp matches
        it, through the repository's filters. From an empty folder Git finds
        no such file, so no filter runs on the host. It works on a copy,
        because that rewrite would also make Git stop re-checking those
        files, and a later status or staging could then miss an edit.
        """
        git_dir = self.git.run(root, "rev-parse", "--absolute-git-dir").stdout.strip()
        with tempfile.TemporaryDirectory(prefix="cowork-tree-", ignore_cleanup_errors=True) as scratch:
            copy, neutral = Path(scratch) / "index", Path(scratch) / "empty"
            neutral.mkdir()
            if index.is_file():
                shutil.copy2(index, copy)
            return self.git.run(neutral, "write-tree", environment={"GIT_DIR": git_dir, "GIT_INDEX_FILE": str(copy)}).stdout.strip()

    def _save_worktree_state(self, worktree: Path, head: str) -> tuple[str, str]:
        """Commit the staged state and the working files without touching the task's index.

        Returns ``(index, files)``: a commit of the index whose parent is HEAD,
        and a commit of every working file Git would see, untracked included,
        whose parent is the index commit.
        """
        identity = {
            "GIT_AUTHOR_NAME": "Cowork", "GIT_AUTHOR_EMAIL": "cowork@localhost",
            "GIT_COMMITTER_NAME": "Cowork", "GIT_COMMITTER_EMAIL": "cowork@localhost",
        }
        index_path = Path(self.git.run(worktree, "rev-parse", "--path-format=absolute", "--git-path", "index").stdout.strip())
        with tempfile.TemporaryDirectory(prefix="cowork-release-") as temporary:
            private = Path(temporary) / "index"
            if index_path.is_file():
                shutil.copy2(index_path, private)
            environment = {"GIT_INDEX_FILE": str(private), "GIT_OPTIONAL_LOCKS": "0", **identity}
            index_tree = self._write_tree(worktree, private)
            index_commit = self.git.run(
                worktree, "commit-tree", index_tree, "-p", head, "-m", "Cowork: staged changes", environment=environment,
            ).stdout.strip()
            self._add(worktree, "--all", "--", ".", environment=environment)
            files_tree = self._write_tree(worktree, private)
            files_commit = self.git.run(
                worktree, "commit-tree", files_tree, "-p", index_commit, "-m", "Cowork: working files",
                environment=environment,
            ).stdout.strip()
        return index_commit, files_commit

    def restore(
        self,
        key: str,
        source_path: str,
        workspace_path: str,
        kind: WorkspaceKind,
    ) -> None:
        """Recreate a released workspace at its original path with its changes.

        Raises, keeping the saved state, when it cannot be rebuilt exactly.
        """
        with self._mutation_lock:
            if kind != WorkspaceKind.git_worktree:
                return
            actual = self._managed_worktree(key, workspace_path)
            release_dir = self.snapshots_root / managed_key(key)
            metadata_path = release_dir / _RELEASE_METADATA
            if not metadata_path.is_file():
                raise WorkspaceError("This task's workspace was removed and has no saved copy to restore")
            source = Path(source_path)
            if not source.is_dir():
                raise WorkspaceError(
                    f"The task's repository at {source} is unavailable. Put it back to reopen this task's workspace"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("format") != 2:
                raise WorkspaceError("This task's saved workspace is in a format this version cannot restore")
            branch = metadata.get("branch")
            if branch:
                self._check_branch_free(source, branch, metadata["head"])
            # An interrupted release can leave part of the folder behind. The
            # saved state is complete, so the leftover is replaced.
            if actual.exists():
                self.git.run(source, "worktree", "remove", "--force", str(actual), check=False)
                shutil.rmtree(actual, ignore_errors=True)
            if actual.exists():
                raise WorkspaceError("Part of this task's old workspace is still in use. Close programs using it and try again")
            self.git.run(source, "worktree", "prune", check=False)
            actual.parent.mkdir(parents=True, exist_ok=True)
            if branch and self.git.run(source, "rev-parse", "--verify", "-q", f"refs/heads/{branch}", check=False).returncode:
                # The branch was deleted while the task was released; nothing
                # else uses the name, so it is recreated where the task left it.
                self.git.run(source, "branch", branch, metadata["head"])
            target = branch or metadata["head"]
            self.git.run(source, "worktree", "add", *(() if branch else ("--detach",)), str(actual), target)
            try:
                # Files first, then the index on its own, so staged and
                # unstaged versions of a file both come back.
                self.git.run(actual, "read-tree", "-u", "--reset", f"{metadata['files']}^{{tree}}")
                self.git.run(actual, "read-tree", f"{metadata['index']}^{{tree}}")
            except Exception:
                self.git.run(source, "worktree", "remove", "--force", str(actual), check=False)
                self.git.run(source, "worktree", "prune", check=False)
                raise
            self.git.run(source, "update-ref", "-d", self._release_ref(key), check=False)
            metadata_path.unlink()

    def _check_branch_free(self, source: Path, branch: str, head: str) -> None:
        """Refuse to restore onto a branch that moved or that another checkout holds."""
        current = self.git.run(source, "rev-parse", "--verify", "-q", f"refs/heads/{branch}", check=False)
        if current.returncode == 0 and current.stdout.strip() != head:
            raise WorkspaceError(
                f"The task's branch {branch} has moved since its workspace was put away. "
                f"Move {branch} back to {head[:12]}, or rename it, to reopen this task"
            )
        for line in self.git.run(source, "worktree", "list", "--porcelain").stdout.splitlines():
            if line == f"branch refs/heads/{branch}":
                raise WorkspaceError(
                    f"The task's branch {branch} is checked out in another folder. "
                    "Switch that folder to another branch to reopen this task"
                )

    def is_released(self, key: str, kind: WorkspaceKind) -> bool:
        """Whether saved state exists to rebuild this workspace from."""
        if kind != WorkspaceKind.git_worktree:
            return False
        return (self.snapshots_root / managed_key(key) / _RELEASE_METADATA).is_file()

    def discard_release(self, key: str, source_path: str) -> None:
        """Forget a released workspace's saved state once its task is deleted."""
        with self._mutation_lock:
            release_dir = self.snapshots_root / managed_key(key)
            (release_dir / _RELEASE_METADATA).unlink(missing_ok=True)
            source = Path(source_path)
            if source.is_dir() and self._git_root(source) is not None:
                self.git.run(source, "update-ref", "-d", self._release_ref(key), check=False)

    def _has_nested_repository(self, worktree: Path) -> bool:
        """Whether a repository inside the worktree holds history a patch cannot carry.

        A nested repository's files and commits would be lost when the worktree
        is removed. One can sit anywhere, including in a tracked directory the
        parent still sees as clean, so every path Git lists and its parents are
        checked. A staged or committed nested repository is listed as a single
        gitlink path. Ignored directories are not checked, as their contents
        are never saved.
        """
        untracked = [path for status, path in self._status_entries(worktree) if status == "??"]
        tracked = [path for path in self.git.run(worktree, "ls-files", "-z").stdout.split("\0") if path]
        directories: set[PurePosixPath] = set()
        for path in (*untracked, *tracked):
            directories.add(PurePosixPath(path))
            directories.update(PurePosixPath(path).parents)
        directories.discard(PurePosixPath("."))
        return any((worktree / directory / ".git").exists() for directory in directories)

    def _managed_worktree(self, key: str, workspace_path: str) -> Path:
        relative = managed_key(key)
        expected = {
            (self.worktrees_root / relative).resolve(),
            (self.legacy_worktrees_root / relative).resolve(),
        }
        actual = Path(workspace_path).resolve()
        if actual not in expected:
            raise WorkspaceError("Refusing to change an unmanaged worktree path")
        return actual

    @staticmethod
    def _release_ref(key: str) -> str:
        return "refs/cowork/released/" + managed_key(key).as_posix()

    def discard_recovery(self, session_id: str) -> None:
        """Remove everything kept to recover a task, once the task is deleted.

        Recovery data outlives a released workspace on purpose, but a deleted
        task can never be reopened, so its folder would only fill the disk.
        """
        relative = managed_key(session_id)
        if len(relative.parts) != 1:
            raise WorkspaceError("Invalid task recovery key")
        with self._mutation_lock:
            task_dir = self.snapshots_root / relative
            if task_dir.is_symlink():
                task_dir.unlink()
            elif task_dir.is_dir():
                self._discard(task_dir)

    def _discard(self, path: Path) -> None:
        """Remove a managed folder in the background, or now if it cannot be moved aside."""
        if not self.pending_removals.discard(path):
            force_remove(path)

    def prune_task_root(self, session_id: str) -> None:
        """Remove an empty project-task parent without discarding task files."""
        relative = managed_key(session_id)
        if len(relative.parts) != 1:
            raise WorkspaceError("Invalid project task workspace key")
        managed_roots = (
            self.worktrees_root,
            self.legacy_worktrees_root,
            self.local_copies.legacy_copies_root,
            self.local_copies.baselines_root,
        )
        for workspace_root in managed_roots:
            task_root = workspace_root / relative
            if task_root.is_symlink() or not task_root.is_dir():
                continue
            for filename in _TASK_ROOT_METADATA:
                metadata = task_root / filename
                try:
                    if metadata.is_file() or metadata.is_symlink():
                        metadata.unlink()
                except OSError:
                    # A concurrent OS metadata write must not make task deletion fail.
                    pass
            try:
                task_root.rmdir()
            except OSError:
                # Preserve the root when it still contains any real task files.
                pass

    def _snapshot_changes(
        self,
        session_id: str,
        workspace: Path,
        base_revision: str,
        filename: str,
        *,
        truncate: bool = False,
    ) -> Path | None:
        untracked = [path for status, path in self._status_entries(workspace) if status == "??"]
        if untracked:
            # Intent-to-add makes Git include new files in the binary diff while
            # leaving their contents unstaged and the working copy unchanged.
            # Send NUL-delimited paths over stdin so Windows command-line limits
            # cannot break recovery for tasks with many or unusual filenames.
            self.git.run(
                workspace,
                "add",
                "-N",
                "--pathspec-from-file=-",
                "--pathspec-file-nul",
                input_text="\0".join(untracked) + "\0",
            )
        snapshot_dir = self.snapshots_root / session_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / filename
        temp = snapshot.with_suffix(".tmp")
        try:
            self.git.run(
                workspace, "diff", "--no-ext-diff", "--no-textconv", "--binary", f"--output={temp.resolve()}", base_revision,
            )
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        size = temp.stat().st_size
        if not size:
            temp.unlink()
            return None
        if truncate and size > MAX_TOTAL_DIFF_BYTES:
            size = _truncate_at_file_boundary(temp, MAX_TOTAL_DIFF_BYTES)
            snapshot.with_name(f"{filename}.truncated").write_text(SNAPSHOT_TRUNCATED_MARKER, encoding="utf-8")
            if not size:
                temp.unlink()
                return None
        os.replace(temp, snapshot)
        return snapshot

    def _apply_patch(
        self, cwd: Path, patch: Path, *flags: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        with patch.open("rb") as handle:
            return self.git.run(cwd, "apply", *flags, "--whitespace=nowarn", "-", check=check, stdin=handle)

    def _status_lines(self, root: Path, checks: CheckPolicy | None = None) -> list[str]:
        args = ("status", "--porcelain=v1", "--untracked-files=all")
        if checks is None:
            result = self.git.run(root, *args, environment={"GIT_OPTIONAL_LOCKS": "0"})
        else:
            # Status may invoke the same content filters as staging, before
            # and after the commit. It does not need to update the index.
            from cowork.coding import check_sandbox

            environment = check_sandbox.check_environment(os.environ, cache_root=checks.cache_root)
            environment.update({
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ALLOW_PROTOCOL": ALLOWED_GIT_PROTOCOLS,
            })
            try:
                result = check_sandbox.run_sandboxed(
                    ["git", *_platform_config_args(), *_host_safety_args(args), *args],
                    cwd=root, policy=checks, environment=environment, writable=[root],
                    read_only=check_sandbox.git_metadata(root),
                    storage=check_sandbox.filter_storage(check_sandbox.git_directories(self.git.run, root).common),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise WorkspaceError(f"Could not check Git status: {exc}") from exc
            if result.returncode:
                raise WorkspaceError((result.stderr or "Could not check Git status").strip()[:4_000])
        return result.stdout.splitlines()

    def source_change_paths(self, root: Path) -> list[str]:
        return [path for _, path in self._status_entries(root)]

    def _status_entries(self, root: Path) -> list[tuple[str, str]]:
        raw = self.git.run(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", environment={"GIT_OPTIONAL_LOCKS": "0"}).stdout
        parts = raw.split("\0")
        entries: list[tuple[str, str]] = []
        index = 0
        while index < len(parts):
            item = parts[index]
            index += 1
            if not item:
                continue
            status, path = item[:2], item[3:]
            # In porcelain v1 -z, rename/copy records carry the destination in
            # the current record and source path in the next NUL record.
            if ("R" in status or "C" in status) and index < len(parts):
                index += 1
            entries.append((status, self._validated_git_path(path)))
        return entries

    def _git_root(self, path: Path) -> Path | None:
        try:
            result = self.git.run(path, "rev-parse", "--show-toplevel", check=False)
        except GitUnavailableError:
            # Git is an enhancement for repository-backed folders, not a
            # prerequisite for using an ordinary local folder.
            return None
        if result.returncode != 0:
            return None
        return Path(result.stdout.strip()).resolve()

    def _head_revision(self, root: Path) -> str | None:
        result = self.git.run(root, "rev-parse", "--verify", "HEAD", check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    def _prepare_local_copy(self, session_id: str, source: Path) -> PreparedWorkspace:
        try:
            prepared = self.local_copies.prepare(session_id, source)
        except LocalCopyError as exc:
            raise WorkspaceError(str(exc)) from exc
        return PreparedWorkspace(
            source_path=source,
            workspace_path=prepared.workspace,
            kind=WorkspaceKind.local_copy,
            repository_root=None,
            base_revision=None,
            source_dirty=False,
            warning=None,
        )

    @staticmethod
    def _resolve_existing(raw_path: str) -> Path | None:
        try:
            path = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        return path

    @staticmethod
    def _counts(patch: str) -> tuple[int, int]:
        additions = sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
        return additions, deletions

    def _untracked_diff(self, root: Path, path: Path, rel_path: str) -> DiffFile:
        try:
            before = path.lstat()
            if stat.S_ISLNK(before.st_mode):
                target = os.readlink(path)
                return DiffFile(
                    path=rel_path,
                    status="??",
                    patch=f"Symbolic link → {target}",
                    binary=True,
                )
            if not stat.S_ISREG(before.st_mode):
                return DiffFile(path=rel_path, status="??", patch="Special file; open the worktree to review it.", binary=True)
            if before.st_size > MAX_TEXT_DIFF_BYTES:
                return DiffFile(path=rel_path, status="??", patch="Binary or large new file", binary=True)

            descriptor = open_fd(path, os.O_RDONLY | O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as handle:
                opened = os.fstat(handle.fileno())
                # On platforms without O_NOFOLLOW, also refuse a path swapped
                # between lstat and open before reading from its descriptor.
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    return DiffFile(path=rel_path, status="??", patch="File changed while Cowork was reviewing it.", binary=True)
                content = handle.read(MAX_TEXT_DIFF_BYTES + 1)
        except OSError as exc:
            return DiffFile(path=rel_path, status="??", patch=f"Unable to read file: {exc}")
        if b"\0" in content[:8_192] or len(content) > MAX_TEXT_DIFF_BYTES:
            return DiffFile(path=rel_path, status="??", patch="Binary or large new file", binary=True)
        text = content.decode("utf-8", errors="replace")
        patch = "".join(
            difflib.unified_diff(
                [],
                text.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=f"b/{rel_path}",
            )
        )
        additions, deletions = self._counts(patch)
        return DiffFile(path=rel_path, status="??", additions=additions, deletions=deletions, patch=patch)

    def _tracked_file_is_large(
        self,
        root: Path,
        path: Path,
        rel_path: str,
        base_revision: str,
    ) -> bool:
        try:
            if path.is_file() and path.stat().st_size > MAX_TEXT_DIFF_BYTES:
                return True
        except OSError:
            return True
        previous = self.git.run(
            root,
            "cat-file",
            "-s",
            f"{base_revision}:{rel_path}",
            check=False,
        )
        try:
            return previous.returncode == 0 and int(previous.stdout.strip()) > MAX_TEXT_DIFF_BYTES
        except ValueError:
            return False
