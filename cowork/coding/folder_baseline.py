"""Review an in-place task on a folder without Git against how the folder started.

There is no history to compare with and no copy of the folder, so a task
records a baseline when it starts: every file's size and modification time,
plus a saved copy of each small text file. Review then lists what was added,
changed or deleted, with line diffs wherever the original was saved.
Dependency and build folders are left out, as they are too large to track and
are rarely what a task is reviewed for.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from cowork.coding.contracts import DiffFile
from cowork.coding.local_copy import MAX_LOCAL_DIFF_FILES, LocalCopyManager
from cowork.coding.workspace_key import managed_key

SKIPPED_DIRECTORIES = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".next", ".nuxt", ".turbo", "dist", "build", "target",
})
MAX_SAVED_FILE_BYTES = 512 * 1024
MAX_SAVED_TOTAL_BYTES = 64 * 1024 * 1024
_MANIFEST = "manifest.json"
_SAVED = "files"


class FolderBaselineError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Entry:
    kind: str
    mode: int = 0
    size: int = 0
    mtime_ns: int = 0
    target: str = ""
    saved: bool = False


class FolderBaselines:
    """Baselines for in-place tasks on folders without Git, one per task key."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def capture(self, key: str, folder: Path) -> None:
        """Record ``folder`` as it is now; replaces any earlier baseline for ``key``."""
        target = self._dir(key)
        self.discard(key)
        saved_root = target / _SAVED
        saved_root.mkdir(parents=True)
        budget = MAX_SAVED_TOTAL_BYTES
        manifest: dict[str, dict] = {}
        for relative, entry in _scan(folder).items():
            if entry.kind == "file" and entry.size <= MAX_SAVED_FILE_BYTES and entry.size <= budget:
                data = _read_text(folder / relative)
                if data is not None:
                    destination = saved_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    budget -= len(data)
                    entry = _Entry(**{**asdict(entry), "saved": True})
            manifest[relative] = asdict(entry)
        (target / _MANIFEST).write_text(json.dumps({"schema_version": 1, "entries": manifest}), encoding="utf-8")

    def exists(self, key: str) -> bool:
        return (self._dir(key) / _MANIFEST).is_file()

    def diff(self, key: str, folder: Path) -> list[DiffFile]:
        try:
            raw = json.loads((self._dir(key) / _MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FolderBaselineError("This task's starting snapshot is missing, so its changes cannot be reviewed") from exc
        before = {relative: _Entry(**value) for relative, value in raw["entries"].items()}
        after = _scan(folder)
        saved_root = self._dir(key) / _SAVED
        files: list[DiffFile] = []
        for relative in sorted(set(before) | set(after)):
            old, new = before.get(relative), after.get(relative)
            if not self._changed(relative, old, new, saved_root, folder):
                continue
            files.append(self._file(relative, old, new, saved_root, folder))
            if len(files) >= MAX_LOCAL_DIFF_FILES:
                files.append(DiffFile(path="… additional files", status="…", patch="Open the folder to review the remaining changes."))
                break
        return files

    def discard(self, key: str) -> None:
        target = self._dir(key)
        if target.exists():
            shutil.rmtree(target)

    def _dir(self, key: str) -> Path:
        return self.root / managed_key(key)

    @staticmethod
    def _changed(relative: str, old: _Entry | None, new: _Entry | None, saved_root: Path, folder: Path) -> bool:
        if old is None or new is None:
            return old is not new
        if old.kind != new.kind or old.mode != new.mode or old.target != new.target:
            return True
        if old.kind != "file" or (old.size == new.size and old.mtime_ns == new.mtime_ns):
            return False
        if old.saved:
            # A touched but identical file is not a change.
            return (saved_root / relative).read_bytes() != _read_bytes(folder / relative)
        return True

    @staticmethod
    def _file(relative: str, old: _Entry | None, new: _Entry | None, saved_root: Path, folder: Path) -> DiffFile:
        status = "A" if old is None else "D" if new is None else "M"
        if old is not None and not old.saved and old.kind == "file":
            return DiffFile(
                path=relative, status=status, binary=True,
                patch=(
                    "Large or binary file deleted." if new is None
                    else "Large or binary file changed. Its earlier version was not saved, so no line diff is available."
                ),
            )
        before = saved_root / relative if old is not None else saved_root / ".missing"
        patch, binary = LocalCopyManager._patch(before, folder / relative, relative)
        return DiffFile(
            path=relative,
            status=status,
            additions=sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")),
            deletions=sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")),
            patch=patch,
            binary=binary,
        )


def _scan(root: Path) -> dict[str, _Entry]:
    """Stat every reviewable entry under ``root``, skipping dependency and build folders."""
    result: dict[str, _Entry] = {}
    pending: list[tuple[str, str]] = [(str(root), "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            relative = prefix + entry.name
            try:
                if entry.is_symlink():
                    result[relative] = _Entry("link", target=os.readlink(entry.path))
                elif entry.is_dir(follow_symlinks=False):
                    if entry.name not in SKIPPED_DIRECTORIES:
                        pending.append((entry.path, relative + "/"))
                elif entry.is_file(follow_symlinks=False):
                    info = entry.stat(follow_symlinks=False)
                    result[relative] = _Entry("file", mode=info.st_mode & 0o777, size=info.st_size, mtime_ns=info.st_mtime_ns)
            except OSError:
                result[relative] = _Entry("unreadable")
    return result


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _read_text(path: Path) -> bytes | None:
    data = _read_bytes(path)
    if data is None or b"\0" in data[:8_192]:
        return None
    return data
