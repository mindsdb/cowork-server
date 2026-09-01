from __future__ import annotations

import difflib
import os
import shutil
import stat
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cowork.coding.contracts import (
    DiffFile,
    GitState,
    WorkspaceInspection,
    WorkspaceKind,
)
from cowork.coding.git_transport import ALLOWED_GIT_PROTOCOLS
from cowork.coding.local_copy import LocalCopyError, LocalCopyManager
from cowork.coding.workspace_key import managed_key
from cowork.common.settings.app_settings import get_app_settings


class WorkspaceError(RuntimeError):
    pass


MAX_DIFF_FILES = 250
MAX_TEXT_DIFF_BYTES = 2 * 1024 * 1024
MAX_TOTAL_DIFF_BYTES = 4 * 1024 * 1024
_TASK_ROOT_METADATA = (".DS_Store", "Thumbs.db", "desktop.ini")


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

    def run(
        self,
        cwd: Path,
        *args: str,
        check: bool = True,
        input_text: str | None = None,
        environment: dict[str, str] | None = None,
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
            result = subprocess.run(
                ["git", *args],
                # The directory is an explicit desktop capability selected by
                # the user and validated above. Git is fixed, arguments are an
                # argv array, and shell execution is disabled.
                cwd=str(working_directory),  # lgtm[py/path-injection]
                env=child_environment,
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=120,
                shell=False,
            )
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
        self._mutation_lock = threading.RLock()
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

    def prepare(
        self,
        session_id: str,
        raw_path: str,
        allow_direct_folder: bool,
        base_branch: str | None = None,
    ) -> PreparedWorkspace:
        with self._mutation_lock:
            inspection = self.inspect(raw_path)
            if not inspection.exists or not inspection.is_directory:
                raise WorkspaceError("Choose an existing local folder")
            source = Path(inspection.path)
            if not inspection.is_git:
                if not allow_direct_folder:
                    raise WorkspaceError("Local folder isolation was not enabled for this request")
                return self._prepare_local_copy(session_id, source)

            repo = Path(inspection.repository_root or inspection.path)
            revision = inspection.revision
            if base_branch:
                revision = self.branch_revision(repo, base_branch)
                if revision is None:
                    raise WorkspaceError(f"Base branch is unavailable: {base_branch}")
            # Git cannot create a detached worktree until the repository has a
            # commit. Preserve the same isolation guarantee with the existing
            # conflict-checkable folder-copy engine; no source mutation or
            # synthetic commit is required.
            if revision is None:
                return self._prepare_local_copy(session_id, repo)

            worktree = self.worktrees_root / managed_key(session_id)
            if worktree.exists():
                raise WorkspaceError("A managed worktree already exists for this task")
            worktree.parent.mkdir(parents=True, exist_ok=True)
            self.git.run(repo, "worktree", "add", "--detach", str(worktree), revision)
            return PreparedWorkspace(
                source_path=repo,
                workspace_path=worktree,
                kind=WorkspaceKind.git_worktree,
                repository_root=repo,
                base_revision=revision,
                source_dirty=inspection.dirty,
                warning=inspection.warning,
            )

    def git_state(self, source_path: str, workspace_path: str) -> GitState:
        workspace = Path(workspace_path)
        root = self._git_root(workspace)
        if root is None:
            return GitState(is_git=False, worktree_path=str(workspace), source_path=source_path)
        branch_result = self.git.run(root, "symbolic-ref", "--short", "-q", "HEAD", check=False)
        branch = branch_result.stdout.strip() or None
        revision = self._head_revision(root)
        status_lines = self._status_lines(root)
        return GitState(
            is_git=True,
            branch=branch,
            revision=revision,
            detached=branch is None,
            dirty=bool(status_lines),
            status_lines=status_lines,
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
            if kind == WorkspaceKind.direct_folder:
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
                    self.git.run(
                        target,
                        "apply",
                        "--whitespace=nowarn",
                        "-",
                        input_text=snapshot.read_text(encoding="utf-8"),
                    )
            except Exception:
                self.git.run(source, "worktree", "remove", "--force", str(target), check=False)
                self.git.run(source, "worktree", "prune", check=False)
                raise
            return PreparedWorkspace(
                source_path=source,
                workspace_path=target,
                kind=kind,
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
                self.git.run(root, "add", "--", path)
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
            if valid.returncode != 0:
                raise WorkspaceError("Enter a valid Git branch name")
            self.git.run(root, "switch", "-c", name)
            return self.git_state(str(root), str(root))

    def branch_revision(self, repository_root: str | Path, name: str) -> str | None:
        """Resolve a validated branch name without allowing ref option injection."""
        root = Path(repository_root)
        valid = self.git.run(root, "check-ref-format", "--branch", name, check=False)
        if valid.returncode != 0:
            return None
        resolved = self.git.run(root, "rev-parse", "--verify", name, check=False)
        return resolved.stdout.strip() if resolved.returncode == 0 else None

    def commit(self, workspace_path: str, message: str) -> GitState:
        with self._mutation_lock:
            root = Path(workspace_path)
            if not message.strip():
                raise WorkspaceError("Commit message cannot be empty")
            self.git.run(root, "add", "--all")
            self.git.run(root, "commit", "-m", message.strip())
            return self.git_state(str(root), str(root))

    def commit_checkpoint(self, workspace_path: str) -> str:
        root = Path(workspace_path)
        revision = self._head_revision(root)
        if revision is None:
            raise WorkspaceError("This task repository has no commit to checkpoint")
        return revision

    def rollback_commit(self, workspace_path: str, revision: str) -> None:
        """Remove a task commit while retaining its file changes for retry."""
        self.git.run(Path(workspace_path), "reset", "--mixed", revision)

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
        patch = snapshot.read_text(encoding="utf-8")
        check = self.git.run(source, "apply", "--check", "--whitespace=nowarn", "-", check=False, input_text=patch)
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
        self.git.run(
            Path(source_path),
            "apply",
            "--whitespace=nowarn",
            "-",
            input_text=plan.git_patch.read_text(encoding="utf-8"),
        )

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
        self.git.run(
            Path(source_path),
            "apply",
            "--reverse",
            "--whitespace=nowarn",
            "-",
            input_text=plan.git_patch.read_text(encoding="utf-8"),
        )

    def cleanup(
        self,
        session_id: str,
        source_path: str,
        workspace_path: str,
        kind: WorkspaceKind,
        base_revision: str | None,
    ) -> None:
        with self._mutation_lock:
            if kind == WorkspaceKind.local_copy:
                try:
                    self.local_copies.cleanup(session_id, Path(workspace_path))
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
            if actual.exists():
                if base_revision and source.is_dir():
                    self._snapshot_changes(session_id, actual, base_revision, "cleanup.patch")
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
        patch = self.git.run(workspace, "diff", "--binary", base_revision).stdout
        if not patch:
            return None
        snapshot_dir = self.snapshots_root / session_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / filename
        temp = snapshot.with_suffix(".tmp")
        temp.write_text(patch, encoding="utf-8")
        os.replace(temp, snapshot)
        return snapshot

    def _status_lines(self, root: Path) -> list[str]:
        return [line for line in self.git.run(root, "status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines() if line]

    def _status_entries(self, root: Path) -> list[tuple[str, str]]:
        raw = self.git.run(root, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout
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
        result = self.git.run(path, "rev-parse", "--show-toplevel", check=False)
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

            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
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
