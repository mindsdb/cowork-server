from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from cowork.coding.contracts import DiffFile
from cowork.coding.workspace_key import managed_key

MAX_LOCAL_DIFF_FILES = 250
MAX_LOCAL_TEXT_BYTES = 2 * 1024 * 1024


class LocalCopyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedLocalCopy:
    source: Path
    workspace: Path
    baseline: Path


class LocalCopyManager:
    """Isolate non-Git folders while retaining a conflict-checkable baseline."""

    def __init__(self, root: Path, workspace_root: Path | None = None) -> None:
        self.copies_root = workspace_root or root / "copies"
        self.legacy_copies_root = root / "copies"
        self.baselines_root = root / "baselines"
        self.recovery_root = root / "snapshots"
        for path in (self.copies_root, self.baselines_root, self.recovery_root):
            path.mkdir(parents=True, exist_ok=True)

    def prepare(self, key: str, source: Path) -> PreparedLocalCopy:
        relative = managed_key(key)
        workspace = self.copies_root / relative
        baseline = self.baselines_root / relative
        if workspace.exists() or baseline.exists():
            raise LocalCopyError("A managed copy already exists for this task folder")
        workspace.parent.mkdir(parents=True, exist_ok=True)
        baseline.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(source, baseline, symlinks=True)
            shutil.copytree(source, workspace, symlinks=True)
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            shutil.rmtree(baseline, ignore_errors=True)
            raise
        return PreparedLocalCopy(source=source, workspace=workspace, baseline=baseline)

    def fork(self, key: str, source: Path, current_workspace: Path) -> PreparedLocalCopy:
        relative = managed_key(key)
        workspace = self.copies_root / relative
        baseline = self.baselines_root / relative
        if workspace.exists() or baseline.exists():
            raise LocalCopyError("A managed copy already exists for this task folder")
        workspace.parent.mkdir(parents=True, exist_ok=True)
        baseline.parent.mkdir(parents=True, exist_ok=True)
        # A fork inherits both the parent's changes and its original comparison
        # point. Using the current workspace as the new baseline would make the
        # inherited changes disappear from review and handoff.
        parent_baseline = self._baseline_for(current_workspace)
        try:
            shutil.copytree(current_workspace, workspace, symlinks=True)
            shutil.copytree(parent_baseline, baseline, symlinks=True)
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            shutil.rmtree(baseline, ignore_errors=True)
            raise
        return PreparedLocalCopy(source=source, workspace=workspace, baseline=baseline)

    def diff(self, workspace: Path) -> list[DiffFile]:
        baseline = self._baseline_for(workspace)
        before = self._manifest(baseline)
        after = self._manifest(workspace)
        paths = sorted(set(before) | set(after))
        files: list[DiffFile] = []
        for relative in paths:
            old = before.get(relative)
            new = after.get(relative)
            if old == new:
                continue
            status = "A" if old is None else "D" if new is None else "M"
            patch, binary = self._patch(baseline / relative, workspace / relative, relative)
            additions = sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
            files.append(
                DiffFile(
                    path=relative,
                    status=status,
                    additions=additions,
                    deletions=deletions,
                    patch=patch,
                    binary=binary,
                )
            )
            if len(files) >= MAX_LOCAL_DIFF_FILES:
                files.append(DiffFile(path="… additional files", status="…", patch="Open the task workspace to review the remaining changes."))
                break
        return files

    def apply(self, source: Path, workspace: Path) -> list[str]:
        changed = self.preflight(source, workspace)
        return self.apply_checked(source, workspace, changed)

    def preflight(self, source: Path, workspace: Path) -> list[str]:
        baseline = self._baseline_for(workspace)
        before = self._manifest(baseline)
        current_source = self._manifest(source)
        task = self._manifest(workspace)
        changed = sorted(path for path in set(before) | set(task) if before.get(path) != task.get(path))
        conflicts = [path for path in changed if current_source.get(path) != before.get(path)]
        if conflicts:
            preview = ", ".join(conflicts[:5])
            suffix = "…" if len(conflicts) > 5 else ""
            raise LocalCopyError(f"Handoff stopped before changing the source; these files changed outside the task: {preview}{suffix}")
        return changed

    def apply_checked(self, source: Path, workspace: Path, changed: list[str]) -> list[str]:
        self._replace_changed(source, workspace, changed)
        return changed

    def rollback_checked(self, source: Path, workspace: Path, changed: list[str]) -> None:
        """Restore the pre-task versions of a checked handoff's paths."""
        self._replace_changed(source, self._baseline_for(workspace), changed)

    def _replace_changed(self, source: Path, desired: Path, changed: list[str]) -> None:
        desired_manifest = self._manifest(desired)
        # Remove deleted paths first, deepest first. This makes file↔directory
        # replacements deterministic instead of attempting to copy a file over
        # a still-populated directory (or create a directory over a file).
        removed = (relative for relative in changed if relative not in desired_manifest)
        for relative in sorted(removed, key=lambda value: (value.count("/"), value), reverse=True):
            target = self._safe_child(source, relative)
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()

        for relative in (item for item in changed if item in desired_manifest):
            target = self._safe_child(source, relative)
            desired_path = self._safe_child(desired, relative)
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            if desired_path.is_symlink():
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(os.readlink(desired_path))
            else:
                temp = target.with_name(f".{target.name}.cowork-tmp")
                shutil.copy2(desired_path, temp)
                os.replace(temp, target)

    def cleanup(self, key: str, workspace: Path) -> None:
        relative = managed_key(key)
        actual = workspace.resolve()
        expected_roots = (self.copies_root, self.legacy_copies_root)
        if actual not in {(root / relative).resolve() for root in expected_roots}:
            raise LocalCopyError("Refusing to remove an unmanaged task copy")
        baseline = self.baselines_root / relative
        recovery = self.recovery_root / relative / "local-copy"
        if workspace.exists() and self.diff(workspace):
            recovery.parent.mkdir(parents=True, exist_ok=True)
            if recovery.exists():
                shutil.rmtree(recovery)
            shutil.copytree(workspace, recovery, symlinks=True)
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(baseline, ignore_errors=True)

    def _baseline_for(self, workspace: Path) -> Path:
        actual = workspace.resolve()
        relative = next(
            (
                candidate
                for root in (self.copies_root, self.legacy_copies_root)
                if (candidate := self._relative_to(actual, root)) is not None
            ),
            None,
        )
        if relative is None:
            raise LocalCopyError("Task copy is outside the managed workspace root")
        baseline = self.baselines_root / relative
        if not baseline.is_dir():
            raise LocalCopyError("The task copy baseline is unavailable")
        return baseline

    @staticmethod
    def _relative_to(path: Path, root: Path) -> Path | None:
        try:
            return path.relative_to(root.resolve())
        except ValueError:
            return None

    @staticmethod
    def _manifest(root: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        if not root.is_dir():
            return result
        paths: list[Path] = []
        # Keep Git metadata in isolated copies so nested repositories remain
        # usable, but never scan it as source content for review or handoff.
        for directory, directories, filenames in os.walk(root, topdown=True, followlinks=False):
            current = Path(directory)
            retained: list[str] = []
            for name in sorted(directories):
                path = current / name
                if name == ".git":
                    continue
                if path.is_symlink():
                    paths.append(path)
                else:
                    retained.append(name)
            directories[:] = retained
            paths.extend(current / name for name in sorted(filenames) if name != ".git")

        for path in sorted(paths):
            relative = path.relative_to(root).as_posix()
            try:
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    result[relative] = f"link:{os.readlink(path)}"
                elif stat.S_ISREG(info.st_mode):
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    result[relative] = f"file:{info.st_mode & 0o777}:{digest.hexdigest()}"
            except OSError:
                result[relative] = "unreadable"
        return result

    @staticmethod
    def _patch(before: Path, after: Path, relative: str) -> tuple[str, bool]:
        def read(path: Path) -> list[str] | None:
            if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_LOCAL_TEXT_BYTES:
                return None
            data = path.read_bytes()
            if b"\0" in data[:8_192]:
                return None
            return data.decode("utf-8", errors="replace").splitlines(keepends=True)

        old = read(before) if before.exists() else []
        new = read(after) if after.exists() else []
        if old is None or new is None:
            return "Binary, linked, or large file changed. Open the task workspace to review it.", True
        return "".join(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{relative}" if before.exists() else "/dev/null",
                tofile=f"b/{relative}" if after.exists() else "/dev/null",
            )
        ), False

    @staticmethod
    def _safe_child(root: Path, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise LocalCopyError("A task change resolves outside its source folder")
        target = root / candidate
        try:
            # Resolve the parent, not the final component. The final component
            # may intentionally be a symlink that handoff needs to replace or
            # reproduce without following it outside the managed folder.
            target.parent.resolve(strict=False).relative_to(root.resolve())
        except ValueError as exc:
            raise LocalCopyError("A task change resolves outside its source folder") from exc
        return target
