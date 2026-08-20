"""
This module defines the memory stores for canonical slot files on disk.
"""

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cowork.common.paths import opened_subdir_nofollow
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import TenantScope, scoped_user_storage_root
from cowork.harnesses.memory.registry import MemorySlot, SLOT_REGISTRY

PROJECT_SLOTS = (MemorySlot.RULES, MemorySlot.LESSONS)


class MemoryStore:
    def __init__(
        self,
        root: Path,
        *,
        nofollow_base: Path | None = None,
        nofollow_tail: tuple[str, ...] = (),
    ) -> None:
        self._root = root
        # When set, slot I/O opens the store directory by walking
        # `nofollow_base`/<tail> with O_NOFOLLOW, so a symlink planted at any
        # agent-reachable component (e.g. `<project>/.anton/memory`) cannot
        # redirect a read/write/delete into another org. When unset, `_root` is
        # trusted (cowork-controlled, e.g. the per-user global store under
        # COWORK_HOME, which the pod mounts read-only).
        self._nofollow_base = nofollow_base
        self._nofollow_tail = nofollow_tail

    @property
    def root(self) -> Path:
        """Resolved slot directory: the same layout anton's Hippocampus reads."""
        return self._root

    def _validate_slot(self, slot_id: MemorySlot) -> None:
        return

    def _filename(self, slot_id: MemorySlot | str) -> str:
        slot_id = MemorySlot(slot_id) if isinstance(slot_id, str) else slot_id
        self._validate_slot(slot_id)
        return SLOT_REGISTRY[slot_id].filename

    @contextmanager
    def _root_fd(self, *, create: bool) -> Iterator[int]:
        """A directory descriptor for the store root. With a nofollow base, no
        symlink in the `.anton`/`memory` chain can redirect the caller; slot
        files are then opened O_NOFOLLOW relative to it."""
        if self._nofollow_base is not None:
            with opened_subdir_nofollow(
                self._nofollow_base, *self._nofollow_tail, create=create
            ) as fd:
                yield fd
            return
        if create:
            self._root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            yield fd
        finally:
            os.close(fd)

    def read(self, slot_id: MemorySlot | str) -> str:
        name = self._filename(slot_id)
        try:
            with self._root_fd(create=False) as fd:
                sfd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                with open(sfd, encoding="utf-8") as f:
                    return f.read()
        except OSError:
            # Missing dir/file, or a symlink squatting the slot (O_NOFOLLOW ->
            # ELOOP): no readable slot content.
            return ""

    def write(self, slot_id: MemorySlot | str, content: str) -> None:
        name = self._filename(slot_id)
        with self._root_fd(create=True) as fd:
            # O_NOFOLLOW refuses a symlink squatting the slot name (writing
            # through it would land in another org); a real slot file opens
            # normally and is truncated.
            sfd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                0o600,
                dir_fd=fd,
            )
            with open(sfd, "w", encoding="utf-8") as f:
                f.write(content.rstrip() + "\n")

    def delete(self, slot_id: MemorySlot | str) -> None:
        name = self._filename(slot_id)
        try:
            with self._root_fd(create=False) as fd:
                st = os.lstat(name, dir_fd=fd)
                if stat.S_ISLNK(st.st_mode) or stat.S_ISREG(st.st_mode):
                    os.unlink(name, dir_fd=fd)
        except OSError:
            return


class GlobalMemoryStore(MemoryStore):
    """The ``global`` memory scope: identity, personal rules, cross-project lessons.

    Per-(org, user) in org mode, not per-org: anton overwrites identity by key, so
    a shared tier would let one member's turn replace another's (ADR-0002).
    In org mode an explicit ``root`` is inert — org-first resolution wins.
    """

    def __init__(
        self, root: Path | None = None, scope: TenantScope | None = None
    ) -> None:
        base = (
            root.expanduser()
            if root is not None
            else Path(get_app_settings().memory.root_dir).expanduser()
        )
        super().__init__(scoped_user_storage_root(base, scope, store="memory"))

    def list_slots(self) -> list[MemorySlot]:
        return list(SLOT_REGISTRY.keys())


class ProjectMemoryStore(MemoryStore):
    """The ``project`` memory scope: rules and lessons shared by everyone in the
    org who works in this project."""

    def __init__(self, project_path: Path) -> None:
        # `.anton` and `memory` sit under the agent-writable project tree, so pin
        # the store by an O_NOFOLLOW walk from the (trusted) project dir rather
        # than trusting the resolved path.
        super().__init__(
            Path(project_path) / ".anton" / "memory",
            nofollow_base=Path(project_path),
            nofollow_tail=(".anton", "memory"),
        )

    def _validate_slot(self, slot_id: MemorySlot) -> None:
        if slot_id not in PROJECT_SLOTS:
            raise ValueError(
                f"{slot_id.value} is not supported for project-scoped memory."
            )
