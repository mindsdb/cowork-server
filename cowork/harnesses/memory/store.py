"""
This module defines the memory stores for canonical slot files on disk.
"""

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cowork.common.paths import (
    PinnedDir,
    O_NOFOLLOW,
    dir_lstat,
    dir_open,
    dir_unlink,
    opened_subdir_nofollow,
    pinned_dir,
)
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import TenantScope, scoped_user_storage_root
from cowork.harnesses.memory.registry import MemorySlot, SLOT_REGISTRY

PROJECT_SLOTS = (MemorySlot.RULES, MemorySlot.LESSONS)

# Every slot filename is a fixed, hardcoded value in the registry; a slot id is a
# validated `MemorySlot` enum. `_filename` re-checks the resolved name against
# this closed set (and rejects any separator) so the value reaching os.open /
# os.unlink is provably one of a handful of constants, never attacker-shaped.
_KNOWN_SLOT_FILENAMES = frozenset(spec.filename for spec in SLOT_REGISTRY.values())


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
        name = SLOT_REGISTRY[slot_id].filename
        # Allowlist the resolved filename before it reaches a filesystem sink: it
        # is always a fixed registry constant, so anything else (or a separator)
        # is a bug, not a path to honour.
        if (
            name not in _KNOWN_SLOT_FILENAMES
            or os.sep in name
            or (os.altsep and os.altsep in name)
        ):
            raise ValueError(f"invalid memory slot filename: {name!r}")
        # basename strips any directory component, guaranteeing a bare filename
        # reaches os.open / os.unlink (it is one already; this is belt-and-braces
        # and the sanitiser static analysis recognises for CWE-23).
        return os.path.basename(name)

    @contextmanager
    def _root_fd(self, *, create: bool) -> Iterator[PinnedDir]:
        """A handle for the store root. With a nofollow base, no symlink in the
        `.anton`/`memory` chain can redirect the caller; slot files are then
        opened O_NOFOLLOW relative to it."""
        if self._nofollow_base is not None:
            with opened_subdir_nofollow(
                self._nofollow_base, *self._nofollow_tail, create=create
            ) as d:
                yield d
            return
        with pinned_dir(self._root, create=create) as d:
            yield d

    def read(self, slot_id: MemorySlot | str) -> str:
        name = self._filename(slot_id)
        try:
            with self._root_fd(create=False) as root:
                sfd = dir_open(root, name, os.O_RDONLY | O_NOFOLLOW)
                with open(sfd, encoding="utf-8") as f:
                    return f.read()
        except OSError:
            # Missing dir/file, or a symlink squatting the slot (O_NOFOLLOW ->
            # ELOOP): no readable slot content.
            return ""

    def read_checked(self, slot_id: MemorySlot | str) -> tuple[bool, str]:
        """Return verified existence and content, propagating unsafe/read errors.

        Authorization must distinguish a missing slot from a legacy slot that
        exists but cannot currently be read. Treating both as empty would let a
        member claim and overwrite an unreadable shared resource.
        """
        name = self._filename(slot_id)
        try:
            with self._root_fd(create=False) as root:
                sfd = dir_open(root, name, os.O_RDONLY | O_NOFOLLOW)
                # Keep valid legacy newline bytes intact for an exact
                # compensation snapshot. ``newline=None`` would translate
                # CRLF to LF before ``restore_exact`` can put it back.
                with open(sfd, encoding="utf-8", newline="") as f:
                    return True, f.read()
        except FileNotFoundError:
            return False, ""

    def write(self, slot_id: MemorySlot | str, content: str) -> None:
        name = self._filename(slot_id)
        with self._root_fd(create=True) as root:
            # O_NOFOLLOW refuses a symlink squatting the slot name (writing
            # through it would land in another org); a real slot file opens
            # normally and is truncated.
            sfd = dir_open(
                root,
                name,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_NOFOLLOW,
                0o600,
            )
            with open(sfd, "w", encoding="utf-8") as f:
                f.write(content.rstrip() + "\n")

    def restore_exact(
        self,
        slot_id: MemorySlot | str,
        content: str | bytes,
        *,
        existed: bool,
    ) -> None:
        """Restore a pre-mutation snapshot without normalizing its bytes.

        This is intentionally separate from ``write`` and only for failure
        compensation: normal writes retain Anton's canonical trailing-newline
        behavior, while rollback must reproduce the exact prior UTF-8 file.
        The same pinned, no-follow open protects every path component.
        """
        if not existed:
            self.delete(slot_id)
            return
        name = self._filename(slot_id)
        payload = content.encode("utf-8") if isinstance(content, str) else content
        with self._root_fd(create=True) as root:
            sfd = dir_open(
                root,
                name,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_NOFOLLOW,
                0o600,
            )
            with open(sfd, "wb") as f:
                f.write(payload)

    def delete(self, slot_id: MemorySlot | str) -> None:
        name = self._filename(slot_id)
        try:
            with self._root_fd(create=False) as root:
                st = dir_lstat(root, name)
                if stat.S_ISLNK(st.st_mode) or stat.S_ISREG(st.st_mode):
                    dir_unlink(root, name)
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
