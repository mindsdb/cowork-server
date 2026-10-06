from __future__ import annotations

import ctypes
import ctypes.util
import difflib
import errno
import hashlib
import json
import logging
import os
import shutil
import stat
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from cowork.coding.contracts import DiffFile
from cowork.coding.workspace_key import managed_key

logger = logging.getLogger(__name__)

MAX_LOCAL_DIFF_FILES = 250
MAX_LOCAL_TEXT_BYTES = 2 * 1024 * 1024


class LocalCopyError(RuntimeError):
    pass


class CloneUnavailable(OSError):
    """The filesystem cannot clone this tree, so a byte copy is needed instead."""


_libc: ctypes.CDLL | None = None


def _clone_tree(source: Path, target: Path) -> None:
    """Clone a whole directory tree copy-on-write in one call (APFS only).

    A 7 GB, 289k-file folder clones in about 9 s, where ``copytree`` takes
    about a minute, and the clone shares blocks with its source until either
    side writes. Raises CloneUnavailable when the platform, filesystem or
    volume pair cannot clone, so the caller can fall back to a byte copy.
    """
    global _libc
    if sys.platform != "darwin":
        raise CloneUnavailable(errno.ENOTSUP, "clonefile is macOS only")
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    if _libc.clonefile(os.fsencode(source), os.fsencode(target), 0) != 0:
        code = ctypes.get_errno()
        raise CloneUnavailable(code, os.strerror(code), str(source))


@contextmanager
def _writable(directory: Path) -> Iterator[None]:
    """Let the owner remove entries from a directory, then restore its mode.

    A clone keeps directory modes, so a source folder marked read-only yields
    a read-only directory the sweep cannot unlink from.
    """
    mode = os.lstat(directory).st_mode
    writable = mode | stat.S_IWUSR | stat.S_IXUSR
    if writable != mode:
        os.chmod(directory, stat.S_IMODE(writable))
    try:
        yield
    finally:
        if writable != mode:
            os.chmod(directory, stat.S_IMODE(mode))


def _force_remove(path: Path) -> None:
    """Remove a managed tree even where its directories are read-only."""

    def retry_writable(function, failed: str, _exc: BaseException) -> None:
        parent = os.path.dirname(failed)
        with suppress(OSError):
            os.chmod(parent, stat.S_IMODE(os.lstat(parent).st_mode) | stat.S_IWUSR | stat.S_IXUSR)
            function(failed)

    if path.exists() or path.is_symlink():
        shutil.rmtree(path, onexc=retry_writable)


@dataclass(frozen=True)
class _Entry:
    """A tree entry as review and handoff compare it; file content is read only on demand."""

    kind: str
    mode: int = 0
    size: int = 0
    mtime_ns: int = 0
    target: str = ""


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
            self._copy_tree(source, baseline)
            self._copy_tree(source, workspace)
        except Exception as exc:
            _force_remove(workspace)
            _force_remove(baseline)
            if isinstance(exc, OSError):
                raise self._copy_failure("The task folder could not be copied", exc) from exc
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
            self._copy_tree(current_workspace, workspace)
            self._copy_tree(parent_baseline, baseline)
        except Exception as exc:
            _force_remove(workspace)
            _force_remove(baseline)
            if isinstance(exc, OSError):
                raise self._copy_failure("The existing task copy could not be duplicated", exc) from exc
            raise
        return PreparedLocalCopy(source=source, workspace=workspace, baseline=baseline)

    def diff(self, workspace: Path) -> list[DiffFile]:
        baseline = self._baseline_for(workspace)
        before, after = self._manifests(baseline, workspace)
        files: list[DiffFile] = []
        for relative in self._changed(baseline, before, workspace, after):
            old = before.get(relative)
            new = after.get(relative)
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
        before, current_source, task = self._manifests(baseline, source, workspace)
        changed = self._changed(baseline, before, workspace, task)
        # A skipped entry is in no manifest, so a task file at its path reads as
        # a clean addition and would replace a live socket, pipe or device.
        occupied = self._occupied_by_special(source, changed)
        if occupied:
            preview = ", ".join(occupied[:5])
            suffix = "…" if len(occupied) > 5 else ""
            raise LocalCopyError(f"Handoff stopped before changing the source; a socket, pipe or device still occupies: {preview}{suffix}")
        conflicts = [
            path
            for path in changed
            if not self._same(path, baseline, before.get(path), source, current_source.get(path))
        ]
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
            try:
                self._copy_tree(workspace, recovery)
            except OSError as exc:
                raise self._copy_failure("The task copy could not be saved for recovery", exc) from exc
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(baseline, ignore_errors=True)

    def release(self, key: str, workspace: Path) -> bool:
        """Remove a task copy and its baseline, keeping only the task's changes.

        The saved state is the pre-task and task versions of each changed path,
        so a restore can rebuild both trees from the then-current source and
        the task's diff and handoff checks behave as before. Returns False,
        leaving everything in place, when the copy cannot be released safely.
        """
        relative = managed_key(key)
        actual = workspace.resolve()
        if actual not in {(root / relative).resolve() for root in (self.copies_root, self.legacy_copies_root)}:
            raise LocalCopyError("Refusing to release an unmanaged task copy")
        # Repository metadata is not part of the saved changes, so commits made
        # in a copied or task-created repository would be lost.
        if not actual.is_dir() or self._has_repository(actual):
            return False
        baseline = self._baseline_for(actual)
        before, after = self._manifests(baseline, actual)
        changed = self._changed(baseline, before, actual, after)
        if any(entry.kind == "unreadable" for entry in (*before.values(), *after.values())):
            return False
        release = self._release_dir(relative)
        staging = release.with_name(f"{release.name}.tmp")
        _force_remove(staging)
        try:
            staging.mkdir(parents=True)
            for source_root, side in ((baseline, "before"), (actual, "after")):
                manifest = before if side == "before" else after
                for path in changed:
                    if path in manifest:
                        self._copy_entry(self._safe_child(source_root, path), self._safe_child(staging / side, path))
            (staging / "changes.json").write_text(json.dumps({"paths": changed}), encoding="utf-8")
        except OSError as exc:
            _force_remove(staging)
            raise self._copy_failure("The task changes could not be saved before releasing the copy", exc) from exc
        _force_remove(release)
        os.replace(staging, release)
        _force_remove(actual)
        _force_remove(baseline)
        return True

    def restore(self, key: str, source: Path, workspace: Path) -> None:
        """Rebuild a released copy from the current source plus its saved changes."""
        relative = managed_key(key)
        release = self._release_dir(relative)
        changes = release / "changes.json"
        if not changes.is_file():
            raise LocalCopyError("This task's copy was removed and has no saved changes to restore")
        expected = (self.copies_root / relative).resolve()
        if workspace.resolve() != expected:
            raise LocalCopyError("Refusing to restore an unmanaged task copy")
        changed = list(json.loads(changes.read_text(encoding="utf-8"))["paths"])
        # An interrupted release can leave part of the copy behind. The saved
        # changes are complete, so the leftover is replaced.
        for leftover in (workspace, self.baselines_root / relative):
            _force_remove(leftover)
            if leftover.exists():
                raise LocalCopyError("Part of this task's old copy is still in use. Close programs using it and try again")
        prepared = self.prepare(key, source)
        try:
            self._replace_changed(prepared.baseline, release / "before", changed)
            self._replace_changed(prepared.workspace, release / "after", changed)
        except (OSError, LocalCopyError):
            _force_remove(prepared.workspace)
            _force_remove(prepared.baseline)
            raise
        _force_remove(release)

    def is_released(self, key: str) -> bool:
        return (self._release_dir(managed_key(key)) / "changes.json").is_file()

    def discard_release(self, key: str) -> None:
        _force_remove(self._release_dir(managed_key(key)))

    def _release_dir(self, relative: Path) -> Path:
        return self.recovery_root / relative / "local-release"

    @staticmethod
    def _has_repository(root: Path) -> bool:
        pending = [root]
        while pending:
            try:
                entries = list(os.scandir(pending.pop()))
            except OSError:
                return True
            for entry in entries:
                if entry.name == ".git":
                    return True
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
        return False

    @staticmethod
    def _copy_entry(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)

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

    @classmethod
    def _copy_tree(cls, source: Path, target: Path) -> None:
        try:
            _clone_tree(source, target)
            cls._remove_unsupported(target)
        except OSError as exc:
            logger.debug("Falling back to a byte copy of %s: %s", source, exc)
            # A failed clone or sweep can leave a partial tree behind, possibly
            # with read-only directories.
            _force_remove(target)
            shutil.copytree(source, target, symlinks=True, ignore=cls._skip_unsupported)

    @classmethod
    def _remove_unsupported(cls, root: Path) -> None:
        # A clone reproduces sockets, FIFOs and device nodes as inert entries.
        # Drop them so a cloned tree matches what the byte copy produces.
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                special = []
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif not entry.is_file(follow_symlinks=False) and not entry.is_symlink():
                        special.append(entry.path)
            if special:
                with _writable(directory):
                    for path in special:
                        os.unlink(path)

    @staticmethod
    def _copy_failure(subject: str, exc: OSError) -> LocalCopyError:
        # _call does not log a RuntimeError, so this warning is the only
        # operator record; the destination is managed, so report the source only.
        entries = exc.args[0] if exc.args and isinstance(exc.args[0], list) else []
        reported = [f"{item[0]}: {item[2]}" for item in entries[:5] if isinstance(item, tuple) and len(item) == 3]
        detail = "; ".join(reported) if reported else str(exc)
        logger.warning("%s: %s", subject, detail, exc_info=exc)
        return LocalCopyError(f"{subject}: {detail}")

    @staticmethod
    def _unsupported_mode(mode: int) -> bool:
        return not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode))

    @staticmethod
    def _skip_unsupported(directory: str, names: list[str]) -> set[str]:
        # Sockets, FIFOs and device nodes cannot be reproduced by a copy, and
        # _manifest already ignores them, so nothing skipped here is reviewable.
        skipped: set[str] = set()
        for name in names:
            try:
                mode = os.lstat(os.path.join(directory, name)).st_mode
            except OSError:
                # copytree collects per-entry failures; raising from inside its
                # ignore callback would abort the whole tree instead.
                continue
            if LocalCopyManager._unsupported_mode(mode):
                skipped.add(name)
        return skipped

    def _occupied_by_special(self, source: Path, changed: list[str]) -> list[str]:
        occupied: list[str] = []
        for relative in changed:
            try:
                mode = self._safe_child(source, relative).lstat().st_mode
            except (FileNotFoundError, NotADirectoryError):
                # Both mean nothing is at that path. NotADirectoryError is the
                # file-to-directory replacement the handoff already supports.
                continue
            except OSError as exc:
                # This gate exists to stop handoff destroying an entry it cannot
                # describe, so an entry it cannot read has to stop it too.
                raise LocalCopyError(
                    f"Handoff stopped before changing the source; {relative} could not be inspected: {exc}"
                ) from exc
            if self._unsupported_mode(mode):
                occupied.append(relative)
        return occupied

    @classmethod
    def _changed(
        cls,
        before_root: Path,
        before: dict[str, _Entry],
        after_root: Path,
        after: dict[str, _Entry],
    ) -> list[str]:
        return [
            relative
            for relative in sorted(set(before) | set(after))
            if not cls._same(relative, before_root, before.get(relative), after_root, after.get(relative))
        ]

    @classmethod
    def _same(
        cls,
        relative: str,
        left_root: Path,
        left: _Entry | None,
        right_root: Path,
        right: _Entry | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        if left.kind != right.kind or left.mode != right.mode or left.target != right.target:
            return False
        if left.kind != "file":
            return True
        if left.size != right.size:
            return False
        # Clones and copy2 keep modification times, so an untouched file
        # matches on stat and is never read. This is git's index heuristic.
        if left.mtime_ns == right.mtime_ns:
            return True
        return cls._digest(left_root / relative) == cls._digest(right_root / relative)

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return "unreadable"
        return digest.hexdigest()

    @classmethod
    def _manifests(cls, *roots: Path) -> list[dict[str, _Entry]]:
        # A walk is dominated by lstat calls, which release the GIL, so the
        # trees are walked side by side.
        with ThreadPoolExecutor(max_workers=len(roots)) as pool:
            return list(pool.map(cls._manifest, roots))

    @staticmethod
    def _manifest(root: Path) -> dict[str, _Entry]:
        """Stat every reviewable entry under ``root`` without reading content.

        Keep Git metadata in isolated copies so nested repositories remain
        usable, but never scan it as source content for review or handoff.
        A symlinked directory is one link entry and is never descended.
        """
        result: dict[str, _Entry] = {}
        if not root.is_dir():
            return result
        pending: list[tuple[str, str]] = [(str(root), "")]
        while pending:
            directory, prefix = pending.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                if entry.name == ".git":
                    continue
                relative = prefix + entry.name
                try:
                    if entry.is_symlink():
                        result[relative] = _Entry("link", target=os.readlink(entry.path))
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append((entry.path, relative + "/"))
                    elif entry.is_file(follow_symlinks=False):
                        info = entry.stat(follow_symlinks=False)
                        result[relative] = _Entry(
                            "file",
                            mode=info.st_mode & 0o777,
                            size=info.st_size,
                            mtime_ns=info.st_mtime_ns,
                        )
                except OSError:
                    result[relative] = _Entry("unreadable")
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
