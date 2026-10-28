"""The list of skills people share with everyone: ``<COWORK_SHARED_DIR>/shared/skills.json``.

Each entry names one person's own skill (cowork.managed.SharedSkill), which
everyone else on the deployment reads live from that person's store and
can't change, while its creator still can. Cowork writes the list itself,
when a person shares or hides a skill from their Skills page, creates one
shared, or deletes a shared one, and when an operator runs
``python -m cowork.ops.skills share|unshare``. Every one of those goes
through SkillService.share and SkillService.stop_sharing, which change the
list here alone (change_shared_skills).

The list sits beside every person's partition, in the shared root. No store
or partition is named ``shared``: partitions are UUID folders.

A missing list shares nothing. A malformed one shares nothing either: every
reader logs a warning naming the file, the server still starts, and the next
change replaces it with a valid list, starting from empty. A list the
filesystem refuses to read (EIO, EMFILE, EACCES) is not malformed: readers
share nothing while it lasts, and a change writes nothing and answers busy,
so a passing read error never drops anyone's share.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from cowork.common.paths import open_fd
from cowork.common.settings.app_settings import get_app_settings
from cowork.managed import (
    SharedRegistryError,
    SharedRegistryUnreadableError,
    SharedSkills,
)

logger = logging.getLogger(__name__)

# How long a change waits for another writer to finish before it answers busy.
SHARED_SKILLS_LOCK_TIMEOUT = 5.0
_LOCK_RETRY_SECONDS = 0.05

SKILL_SHARING_BUSY = "Sharing is busy right now. Try again in a moment."


class SharedSkillsBusyError(TimeoutError):
    """The list can't be changed right now: another writer held its lock for
    SHARED_SKILLS_LOCK_TIMEOUT, or the filesystem refused to read it."""


def shared_skills_file() -> Path:
    """The shared-skill list, in the shared root (COWORK_SHARED_DIR)."""
    return Path(get_app_settings().storage.shared_root) / "shared" / "skills.json"


def _lock_file(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def read_shared_skills() -> SharedSkills:
    """The shared-skill list, read for one request or turn: none when the
    file is missing, and none when it is malformed, with a warning naming
    the file (SharedRegistry.read's error names it)."""
    try:
        return SharedSkills.read(shared_skills_file())
    except SharedRegistryError as error:
        logger.warning("no skill is shared until the list is fixed or rewritten: %s", error)
        return SharedSkills(())


def change_shared_skills(*, change: Callable[[SharedSkills], SharedSkills]) -> SharedSkills:
    """Apply ``change`` to the shared-skill list under its lock and return
    the list it leaves.

    The lock is an flock on ``skills.json.lock`` beside the list. Every
    writer runs as Cowork's own user in the one container, so it covers
    them all, threads and worker processes alike, and the kernel releases
    it when a writer dies. A writer that can't take it within
    SHARED_SKILLS_LOCK_TIMEOUT raises SharedSkillsBusyError.

    The list is read after the lock is taken. A malformed one is read as
    empty, with a warning naming the file, and replaced. One the filesystem
    refuses to read raises SharedSkillsBusyError, with an error naming the
    file, and writes nothing: its entries may be valid. An exception from
    ``change`` writes nothing. Otherwise the list is written, at once
    (SharedRegistry.write), only when it changed or was malformed.
    """
    # Imported here: the desktop sidecar also runs on Windows, which has no
    # fcntl, and never shares a skill.
    import fcntl

    path = shared_skills_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = open_fd(_lock_file(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + SHARED_SKILLS_LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SharedSkillsBusyError(SKILL_SHARING_BUSY) from None
                time.sleep(_LOCK_RETRY_SECONDS)
        try:
            malformed = False
            try:
                current = SharedSkills.read(path)
            except SharedRegistryUnreadableError as error:
                logger.error("left the shared-skill list unchanged: %s", error)
                raise SharedSkillsBusyError(SKILL_SHARING_BUSY) from error
            except SharedRegistryError as error:
                logger.warning(
                    "replacing a malformed shared-skill list, which shared nothing: %s", error
                )
                current = SharedSkills(())
                malformed = True
            updated = change(current)
            if malformed or updated != current:
                updated.write(path)
            return updated
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
