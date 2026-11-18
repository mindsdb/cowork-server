"""A cloud turn's copy of its project's files.

The pod mounts its conversation's folder and not the project, so before each
cloud turn the server copies the project's ordinary files (the rule in
cowork.services.project_file_view, which routing uses too) into
``conversations/<id>/project-files/``. The pod can write that folder, so
every step there is pinned and never follows a link, and what the server
copied is recorded outside it, where the pod can't reach.
"""
from __future__ import annotations

import errno
import hashlib
import logging
import os
import stat
import time
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError

from cowork.common.logger import log_context
from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    cowork_home,
    dir_atomic_write,
    dir_lstat,
    dir_open,
    dir_rmdir,
    dir_scandir,
    open_pinned_child,
    pinned_dir,
    read_bounded,
    write_all,
)
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope, scoped_storage_root
from cowork.services.artifact_roots import CONVERSATIONS_DIRNAME, PROJECT_FILES_DIRNAME, REMOVED_DIRNAME
from cowork.services.files import (
    _make_dir,
    _open_workspace_folder,
    _own_folder,
    _restore_folder_mode,
    _secure_workspace_child,
    remove_conversation_workspace_dir,
)
from cowork.services.project_file_view import (
    ABSENT_ERRNOS,
    OrdinaryFile,
    WalkState,
    utf8_path,
    walk_ordinary_files,
)

logger = logging.getLogger(__name__)

#: The list a copy writes into the project-files folder, naming the files it
#: left out and why. A hidden name, so no project file can take it.
NOT_COPIED_FILENAME = ".not-copied.json"
#: The most left-out files the list names; it counts the rest.
MAX_NAMED_NOT_COPIED = 1000
#: The most bytes the list takes, its named files and the rest of it. anton
#: reads at most 1 MiB of it (_NOT_COPIED_MAX_BYTES in
#: anton/cloud_turn/session.py) and calls a longer list unreadable, and it
#: shows the agent at most 100 names anyway, so the list names files only
#: while they fit here and counts the rest.
MAX_NOT_COPIED_LIST_BYTES = 256 * 1024
#: The longest path the list names. anton refuses the whole list when one
#: path is longer, so a file with a longer path is counted, not named.
MAX_NOT_COPIED_PATH_CHARS = 4096
_COPY_CHUNK_BYTES = 1 << 20
#: The mode of every file in a copy.
_COPY_FILE_MODE = 0o600
#: How many entries one removal step examines, and for how long, before it
#: stops (_RemovalBudget). Removing what a copy should not hold, and deleting
#: what REMOVED_DIRNAME holds, each get this much per turn.
_REMOVAL_MAX_ENTRIES = 10_000
_REMOVAL_SECONDS = 2.0
#: Errors from opening a copy that mean something the pod put at its name (a
#: folder, a link, a pipe, a file it made unwritable), so the name is cleared
#: and the copy made afresh.
_REPLACEABLE_ERRNOS = frozenset({errno.ELOOP, errno.EISDIR, errno.ENXIO, errno.EACCES, errno.EPERM, errno.ETXTBSY})


class ProjectFileCopyLimits(NamedTuple):
    """How much of a project one conversation's copy holds, and how long one
    turn spends walking the project and copying (TurnQueueSettings.project_files_*)."""

    max_file_bytes: int
    max_files: int
    max_total_bytes: int
    seconds: float

    @classmethod
    def from_settings(cls) -> ProjectFileCopyLimits:
        settings = TurnQueueSettings()
        return cls(
            max_file_bytes=settings.project_files_max_file_bytes,
            max_files=settings.project_files_max_files,
            max_total_bytes=settings.project_files_max_total_bytes,
            seconds=settings.project_files_copy_seconds,
        )


class NotCopiedReason(StrEnum):
    """Why the list names a file. anton reads the list strictly
    (_NotCopiedFiles in anton/cloud_turn/session.py) and accepts these
    values alone, so a new reason ships in anton first."""

    TOO_LARGE = "too_large"
    TOO_MANY_FILES = "too_many_files"
    OVER_TOTAL_BYTES = "over_total_bytes"
    OUT_OF_TIME = "out_of_time"


class NotCopiedFile(BaseModel):
    #: The file's path below the project folder, ``/``-separated.
    path: str
    size: int
    reason: NotCopiedReason


class NotCopiedFiles(BaseModel):
    """The list a copy writes into the project-files folder as
    NOT_COPIED_FILENAME, so the agent can name the project files it did not
    get. Written only when a file was left out, the walk stopped early or the
    copy failed.

    anton reads it strictly (_NotCopiedFiles in anton/cloud_turn/session.py):
    it refuses a key it does not know and requires every one but
    ``copy_failed``, so the list is always written whole, defaults included,
    and a new key ships in anton first."""

    files: list[NotCopiedFile] = []
    #: Files left out and counted, not named: those past
    #: MAX_NAMED_NOT_COPIED or MAX_NOT_COPIED_LIST_BYTES, those whose path is
    #: longer than MAX_NOT_COPIED_PATH_CHARS, and those whose path is not
    #: valid UTF-8, which the list can't hold.
    unnamed: int = 0
    #: The walk stopped before it saw every file: the project holds more
    #: entries than one walk examines, folders deeper than it descends, or the
    #: turn's copy time ran out. Files past that point were neither copied
    #: nor named.
    listing_truncated: bool = False
    #: The copy could not finish, so the server emptied the folder.
    copy_failed: bool = False


class FileIdentity(BaseModel):
    """What a stat says about one file. The change time is the kernel's: no
    process can set it, so a copy the pod rewrote and then dated back still
    differs here."""

    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def of(cls, info: os.stat_result) -> FileIdentity:
        return cls(ino=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


class StagedCopy(BaseModel):
    """One project file the server copied, and its copy as the server left it."""

    source: FileIdentity
    staged: FileIdentity


class CopyRecord(BaseModel):
    """What the server copied into one conversation's project-files folder,
    kept in the project-file-copies store (copy_records_root), which no pod
    mounts and no route serves. A file is skipped as unchanged only when both
    its source and its copy still match the record."""

    #: By the file's path below the project folder, ``/``-separated.
    files: dict[str, StagedCopy] = {}


class ProjectFilesOutcome(StrEnum):
    #: The copy finished, whatever it left out.
    STAGED = "staged"
    #: The copy could not finish. The folder was emptied and its list says so.
    FAILED = "failed"
    #: Nothing was copied and the turn runs: the project's folder is outside
    #: the organization's projects store. ProjectService._repoint_if_stale
    #: leaves a project seeded before projects were kept per organization at
    #: its old folder while that folder holds content, and the turn's pod
    #: mounts the organization's default project for it
    #: (ResponsesHandler._remote_workspace), so a copy would land where no
    #: pod reads it.
    SKIPPED = "skipped"
    #: Nothing was kept: the conversation's folder went away while the copy
    #: ran, as a conversation delete removes it, so the copy wrote no record
    #: and made no folder again. Or the conversation was no longer in the
    #: project once the copy ended, so its staging removed what the copy made:
    #: the folder and the record of a deleted conversation, the copy of a
    #: moved one (stage_remote_workspace in cowork.handlers.responses).
    CONVERSATION_GONE = "conversation_gone"
    #: The turn must not run; ProjectFilesStaged.refusal says why.
    REFUSED = "refused"


class ProjectFilesRefusal(StrEnum):
    """Why a copy was refused (ProjectFilesOutcome.REFUSED)."""

    #: The conversation id is not a UUID, so it names no conversation folder.
    NOT_A_CONVERSATION = "not_a_conversation"
    #: The project's path ends in "", "." or "..", so it names no project
    #: folder, and the folder it does name could be the organization's root.
    NOT_A_PROJECT = "not_a_project"
    #: The copy could not finish and its folder could not be emptied, so the
    #: folder may still hold copies the project no longer backs.
    NOT_EMPTIED = "not_emptied"


#: The message of a turn's ``project_files_staged`` log line, by its outcome
#: (stage_remote_workspace in cowork.handlers.responses).
OUTCOME_LOG_MESSAGES: dict[ProjectFilesOutcome, str] = {
    ProjectFilesOutcome.STAGED: "Copied the project's files for a cloud turn",
    ProjectFilesOutcome.FAILED: "Could not copy the project's files for a cloud turn; its copy was emptied",
    ProjectFilesOutcome.SKIPPED: (
        "Did not copy the project's files for a cloud turn: the project's folder is outside "
        "the organization's projects"
    ),
    ProjectFilesOutcome.CONVERSATION_GONE: (
        "Kept none of the project's files for a cloud turn: its conversation was deleted or moved to "
        "another project while they were copied"
    ),
    ProjectFilesOutcome.REFUSED: "Refused the copy of the project's files for a cloud turn",
}


#: What the turn's error says for each refusal.
_REFUSAL_MESSAGES: dict[ProjectFilesRefusal, str] = {
    ProjectFilesRefusal.NOT_A_CONVERSATION: "its conversation id names no folder",
    ProjectFilesRefusal.NOT_A_PROJECT: "its project's path names no project folder",
    ProjectFilesRefusal.NOT_EMPTIED: "the copy could not finish and its folder could not be emptied",
}


class ProjectFilesStaged(NamedTuple):
    """What one turn's copy did, for its log line."""

    #: Files in the copy once staging ends, and their bytes.
    files: int
    bytes: int
    #: Files written this turn, and their bytes; the rest were unchanged.
    #: They count both passes when the folder was set aside.
    copied: int
    copied_bytes: int
    not_copied: int
    outcome: ProjectFilesOutcome
    elapsed_ms: int
    #: Why the copy was refused: set exactly when the outcome is
    #: ProjectFilesOutcome.REFUSED (_refused).
    refusal: ProjectFilesRefusal | None = None


def _nothing_staged(outcome: ProjectFilesOutcome, elapsed_ms: int = 0) -> ProjectFilesStaged:
    return ProjectFilesStaged(0, 0, 0, 0, 0, outcome=outcome, elapsed_ms=elapsed_ms)


def _refused(refusal: ProjectFilesRefusal, elapsed_ms: int = 0) -> ProjectFilesStaged:
    return _nothing_staged(ProjectFilesOutcome.REFUSED, elapsed_ms)._replace(refusal=refusal)


class ProjectFilesRefused(RuntimeError):
    """A cloud turn's copy of its project's files was refused
    (ProjectFilesOutcome.REFUSED), so the turn is refused rather than run on
    a folder that may hold copies its project no longer backs."""

    def __init__(self, conversation_id: str | UUID, refusal: ProjectFilesRefusal) -> None:
        super().__init__(
            f"the project files copy of conversation {conversation_id} was refused: {_REFUSAL_MESSAGES[refusal]}"
        )
        self.refusal = refusal


class _Copied(NamedTuple):
    staged: StagedCopy
    #: The copy holds the bytes the server wrote and nothing changed it while
    #: they were read back, so the record may trust it on a later turn.
    verified: bool


def copy_records_root(scope: TenantScope) -> Path:
    """The store that keeps each conversation's CopyRecord:
    ``<shared>/<org>/project-file-copies`` in org mode, beside the projects
    store, so outside every pod's mounts and every project file route."""
    return scoped_storage_root(cowork_home() / "project-file-copies", scope, store="project-file-copies")


def copy_record_path(scope: TenantScope, conversation_id: str | UUID) -> Path:
    """Where the conversation's CopyRecord is kept."""
    return copy_records_root(scope) / f"{UUID(str(conversation_id))}.json"


def remove_copy_record(path: Path) -> None:
    """Remove a deleted conversation's CopyRecord. Best-effort, logged."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning(
            "Could not remove a deleted conversation's project files record",
            exc_info=True,
            extra={"event": "project_files_record_not_removed", **log_context(conversation_id=path.stem)},
        )


def remove_deleted_conversations_copy(
    project_path: str | Path, conversation_id: str | UUID, *, scope: TenantScope,
) -> None:
    """Remove the conversation's folder and its CopyRecord, as its delete
    does (ConversationService.finalize_staged_conversation_delete), for a
    conversation deleted while a turn copied its project's files. A delete
    that lands before the copy opens the conversation's folder leaves the
    copy making that folder, its copies and its record again, and the copy's
    own check (_CopyPlace.still_there) passes on the folder it made. The
    caller is stage_remote_workspace in cowork.handlers.responses, once it
    finds the conversation's row gone. Best-effort, logged. Org mode only, on
    the file store's threads."""
    remove_copy_record(copy_record_path(scope, conversation_id))
    remove_conversation_workspace_dir(project_path, conversation_id)


def remove_project_files_copy(project_path: str | Path, conversation_id: str | UUID) -> bool:
    """Remove a conversation's copy of a project's files from that project's
    folder, for a conversation that moved to another project, and say whether
    it found the conversation's folder there and removed the copy. Its next
    turns copy the new project's files into the new project's folder, and
    nothing else removes the old copy before the old project is deleted. The
    callers are the move (ConversationService.update_conversation) and a
    turn's staging whose conversation moved while it copied
    (stage_remote_workspace in cowork.handlers.responses).

    The project-files folder is moved into REMOVED_DIRNAME and that folder
    deleted within one removal budget (_CopyFolders), as a turn deletes it;
    what the budget leaves waits for the old project's delete. Makes no
    conversation folder. Best-effort, logged. Org mode only, on the file
    store's threads."""
    try:
        conv_seg = str(UUID(str(conversation_id)))
    except (ValueError, TypeError):
        return False
    project = Path(project_path)
    if project.name in {"", ".", ".."}:
        return False
    try:
        with ExitStack() as stack:
            parent = stack.enter_context(pinned_dir(project.parent, nofollow_base=True))
            source = open_pinned_child(parent, project.name)
            stack.callback(source.close)
            conversations = open_pinned_child(source, CONVERSATIONS_DIRNAME)
            stack.callback(conversations.close)
            conversation = _open_workspace_folder(conversations, conv_seg)
            stack.callback(conversation.close)
            folders = _CopyFolders(conversation)
            stack.callback(folders.close)
            folders.remove(conversation, PROJECT_FILES_DIRNAME)
            folders.delete_removed(budget=_RemovalBudget.start())
    except (FileNotFoundError, NotADirectoryError):
        return False
    except (OSError, ValueError):
        logger.warning(
            "Could not remove a moved conversation's project files copy from the project it left",
            exc_info=True,
            extra={"event": "project_files_copy_not_removed", **log_context(conversation_id=conv_seg)},
        )
        return False
    return True


def stage_project_files(
    project_path: str | Path,
    conversation_id: str | UUID,
    *,
    scope: TenantScope,
    limits: ProjectFileCopyLimits | None = None,
) -> ProjectFilesStaged:
    """Copy the project's ordinary files into
    ``<project>/conversations/<conv>/project-files/``, mirroring their folders,
    for a cloud turn's pod, which mounts the conversation's folder only.

    A file whose source and copy both still match the conversation's
    CopyRecord is left as it is. A changed file is rewritten in place
    (truncated, never removed and made again), so a pod that holds the copy
    open keeps a valid handle. A copy whose project file is gone, or is now
    left out, is removed. Files over ``limits`` are left out and named in
    NOT_COPIED_FILENAME; a file whose path is not valid UTF-8 is left out
    and counted there, not named.

    ``limits.seconds`` bounds the walk and the copying: once it passes, the
    walk stops, a file being copied is removed and named as out of time, and
    the list says the walk stopped early. Copies the walk did not reach this
    turn are removed with the rest, since nothing checked them against the
    project, and a later turn copies them again. Removing those and saving
    the record run after the budget.

    Removing what the folder should not hold runs within a budget
    (_RemovalBudget). A folder is removed by moving it into REMOVED_DIRNAME,
    which each turn deletes within another budget after the copy, a level at
    a time, so no tree the pod builds there, however deep or wide, costs a
    turn more than its budgets. When the removal runs out of budget, the
    whole project-files folder is moved there and the copy runs again into a
    new one, with ``limits.seconds`` again, so the pod's leftovers never fail
    a turn.

    Fails closed: when the copy can't finish (a read or write error other
    than a file that went away mid-copy), the project-files folder is emptied
    and its list says the copy failed, so the turn never reads a copy the
    project no longer backs. When even that fails, or the conversation id or
    the project's path names no folder, the outcome is REFUSED
    (ProjectFilesRefusal says why) and the caller must not run the turn. A
    project whose folder is outside the scope's projects store copies
    nothing, touches nothing there and is not refused (SKIPPED); the caller's
    log line says so. Org mode only: the caller is the remote turn's staging
    (stage_remote_workspace in cowork.handlers.responses), on the file
    store's threads.

    A conversation deleted while its copy runs removes the conversation's
    folder, which this staging holds open. So once the copy is done, or has
    failed, the folder at the conversation's name must still be the one it
    opened; when it is not, the copy empties what it made through the folders
    it holds, writes no record and makes no folder again
    (CONVERSATION_GONE).
    """
    from cowork.services.projects import projects_root

    started = time.monotonic()

    def elapsed_ms() -> int:
        return round((time.monotonic() - started) * 1000)

    try:
        conv_seg = str(UUID(str(conversation_id)))
    except (ValueError, TypeError):
        return _refused(ProjectFilesRefusal.NOT_A_CONVERSATION)
    project = Path(project_path)
    # Every later step trusts this path: the record is keyed by the
    # conversation alone, and the copy reads whatever folder it names.
    # ``<root>/..`` has the projects root as its parent and names the
    # organization's own folder.
    if project.name in {"", ".", ".."}:
        logger.error(
            "refusing to copy project files from a path that names no project folder",
            extra=log_context(conversation_id=conv_seg),
        )
        return _refused(ProjectFilesRefusal.NOT_A_PROJECT)
    if project.parent != projects_root(scope=scope):
        return _nothing_staged(ProjectFilesOutcome.SKIPPED)
    limits = limits or ProjectFileCopyLimits.from_settings()
    try:
        staged = _stage_project_files(
            project, conv_seg, record_path=copy_record_path(scope, conv_seg), limits=limits,
            deadline=started + limits.seconds,
        )
    except _ConversationGone:
        # Emptying the copy here would make the deleted conversation's folder
        # again, and the copy's with it, where nothing would ever remove them.
        logger.warning(
            "A conversation's folder went away while its project files were copied; kept nothing",
            extra={"event": "project_files_conversation_gone", **log_context(conversation_id=conv_seg)},
        )
        return _nothing_staged(ProjectFilesOutcome.CONVERSATION_GONE, elapsed_ms())
    except Exception:
        logger.warning(
            "Could not copy a conversation's project files; emptying its copy",
            exc_info=True,
            extra={"event": "project_files_copy_failed", **log_context(conversation_id=conv_seg)},
        )
        if _empty_project_files_copy(project, conv_seg):
            return _nothing_staged(ProjectFilesOutcome.FAILED, elapsed_ms())
        return _refused(ProjectFilesRefusal.NOT_EMPTIED, elapsed_ms())
    return staged._replace(elapsed_ms=elapsed_ms())


def _stage_project_files(
    project: Path, conv_seg: str, *, record_path: Path, limits: ProjectFileCopyLimits, deadline: float,
) -> ProjectFilesStaged:
    record = _read_copy_record(record_path)
    with _opened_copy_folders(project, conv_seg) as place:
        if place is None:
            # No project folder, so no conversation folder inside it either.
            return _nothing_staged(ProjectFilesOutcome.STAGED)
        copy = _Copy(record=record, limits=limits, deadline=deadline)
        set_aside: _Copy | None = None
        try:
            try:
                _copy_pass(place, copy)
            except _RemovalBudgetSpent:
                # The pod left more in the folder than one turn removes: set
                # the folder aside whole and copy again into a new one, with
                # the copy's full time, since nothing copied so far is left.
                logger.warning(
                    "A conversation's project files copy holds too much to remove; setting it aside",
                    extra={"event": "project_files_copy_set_aside", **log_context(conversation_id=conv_seg)},
                )
                place.folders.set_aside()
                set_aside, copy = copy, _Copy(
                    record=record, limits=limits, deadline=time.monotonic() + limits.seconds,
                )
                _copy_pass(place, copy)
            place.folders.delete_removed(budget=_RemovalBudget.start())
        except Exception as error:
            # A copy into a folder the delete removed fails at its next
            # write, which is not a copy to empty again by its name.
            if not place.still_there():
                _empty_what_was_made(place.folders, conv_seg)
                raise _ConversationGone from error
            raise
        if not place.still_there():
            _empty_what_was_made(place.folders, conv_seg)
            raise _ConversationGone
    _write_copy_record(record_path, copy.new_record)
    copied, copied_bytes = copy.copied, copy.copied_bytes
    if set_aside is not None:
        # What this turn wrote, the pass it set aside included.
        copied += set_aside.copied
        copied_bytes += set_aside.copied_bytes
    return ProjectFilesStaged(
        files=len(copy.kept), bytes=copy.total, copied=copied, copied_bytes=copied_bytes,
        not_copied=len(copy.not_copied) + copy.unnameable, outcome=ProjectFilesOutcome.STAGED, elapsed_ms=0,
    )


def _copy_pass(place: _CopyPlace, copy: _Copy) -> None:
    """Copy each ordinary file of the project into the project-files folder,
    counting into ``copy``, then remove everything else there and write the
    not-copied list when something was left out. Raises _RemovalBudgetSpent,
    with the folder only partly pruned, when the removal runs out of its
    budget."""
    deadline = copy.deadline
    state = WalkState()
    with _MirrorFolders(place.folders) as mirror_folders, closing(
        walk_ordinary_files(place.source, state=state)
    ) as walk:
        for found in walk:
            if time.monotonic() >= deadline:
                # Stop the walk too, so a large project holds the slot no
                # longer than the budget.
                state.truncated = True
                break
            copy.add(found, mirror_folders)
    named = _named_not_copied(copy.not_copied, unnameable=copy.unnameable)
    not_copied = NotCopiedFiles(
        files=named,
        unnamed=len(copy.not_copied) - len(named) + copy.unnameable,
        listing_truncated=state.truncated,
    )
    keep = set(copy.kept)
    list_it = bool(copy.not_copied) or copy.unnameable > 0 or state.truncated
    if list_it:
        keep.add((NOT_COPIED_FILENAME,))
    _prune_folder(place.folders, keep=keep, budget=_RemovalBudget.start())
    if list_it:
        _write_in_place(place.folders, NOT_COPIED_FILENAME, not_copied.model_dump_json().encode())


def _named_not_copied(not_copied: list[NotCopiedFile], *, unnameable: int) -> list[NotCopiedFile]:
    """The left-out files the list names, in the order the walk found them:
    at most MAX_NAMED_NOT_COPIED, while the whole list, as written, fits in
    MAX_NOT_COPIED_LIST_BYTES, and none with a path longer than
    MAX_NOT_COPIED_PATH_CHARS. The list counts the rest."""
    # The list's own fields at their longest: ``unnamed`` is at most every
    # file left out, and ``false`` is longer than ``true``.
    used = len(NotCopiedFiles(unnamed=len(not_copied) + unnameable).model_dump_json().encode())
    named: list[NotCopiedFile] = []
    for entry in not_copied:
        if len(named) >= MAX_NAMED_NOT_COPIED:
            break
        if len(entry.path) > MAX_NOT_COPIED_PATH_CHARS:
            continue
        # One more for the comma between entries.
        size = len(entry.model_dump_json().encode()) + 1
        if used + size > MAX_NOT_COPIED_LIST_BYTES:
            break
        named.append(entry)
        used += size
    return named


class _RemovalBudgetSpent(Exception):
    """A removal step examined as many entries, or ran as long, as its
    _RemovalBudget allows."""


class _ConversationGone(Exception):
    """The conversation's folder a staging opened is no longer at the
    conversation's name (_CopyPlace.still_there)."""


@dataclass
class _RemovalBudget:
    """How much more one removal step may do: entries it may still examine,
    and the time it must stop by."""

    entries: int
    deadline: float

    @classmethod
    def start(cls) -> _RemovalBudget:
        return cls(entries=_REMOVAL_MAX_ENTRIES, deadline=time.monotonic() + _REMOVAL_SECONDS)

    def out_of_time(self) -> bool:
        return time.monotonic() >= self.deadline

    def spend(self) -> bool:
        """Count one entry examined: False, and nothing counted, once the
        budget is spent."""
        if self.entries <= 0 or self.out_of_time():
            return False
        self.entries -= 1
        return True


class _CopyFolders:
    """The folders one staging works in, all inside the conversation's
    folder, which the pod can write: the conversation's own, its
    project-files folder (``mirror``), and REMOVED_DIRNAME, opened when first
    needed. The caller must ``close()`` it."""

    def __init__(self, conversation: PinnedDir) -> None:
        self.conversation = conversation
        self.mirror = _secure_workspace_child(conversation, PROJECT_FILES_DIRNAME)
        self._removed: PinnedDir | None = None

    def close(self) -> None:
        self.mirror.close()
        self.close_removed()

    def removed(self) -> PinnedDir:
        if self._removed is None:
            self._removed = _secure_workspace_child(self.conversation, REMOVED_DIRNAME)
        return self._removed

    def remove(self, parent: PinnedDir, name: str) -> None:
        """Remove ``name`` from ``parent``: anything but a folder by its name
        alone (a link, never its target), and a folder by moving it, with
        whatever it holds, into REMOVED_DIRNAME under a new name, which takes
        one step however much the folder holds. delete_removed deletes it.

        The open REMOVED_DIRNAME may be gone by the time the folder moves:
        another staging of the conversation deleted it once it was empty, or
        the pod removed it. The move then fails as if ``name`` were gone, so
        a folder still at ``name`` is moved once more into REMOVED_DIRNAME
        made afresh, and a second failure raises."""
        try:
            if not stat.S_ISDIR(dir_lstat(parent, name).st_mode):
                _unlink_listed(parent, name)
                return
            # Moving a folder to another folder rewrites its "..", which needs
            # its owner's write access.
            _restore_folder_mode(parent, name)
        except FileNotFoundError:
            return
        for _attempt in range(2):
            try:
                _rename_listed(parent, name, self.removed(), uuid4().hex)
                return
            except FileNotFoundError:
                pass
            try:
                dir_lstat(parent, name)
            except FileNotFoundError:
                return
            self.close_removed()
        raise OSError(errno.ENOENT, "the removed folder went away twice while a folder moved into it", name)

    def set_aside(self) -> None:
        """Move the whole project-files folder into REMOVED_DIRNAME and make
        a new, empty one in its place."""
        self.remove(self.conversation, PROJECT_FILES_DIRNAME)
        fresh = _secure_workspace_child(self.conversation, PROJECT_FILES_DIRNAME)
        self.mirror.close()
        self.mirror = fresh

    def delete_removed(self, *, budget: _RemovalBudget) -> None:
        """Delete what REMOVED_DIRNAME holds until it is empty, then the
        folder itself, or until ``budget`` is spent; a later turn deletes the
        rest. Each folder there is emptied one level at a time: its files are
        removed and each folder in it is moved up into REMOVED_DIRNAME, so
        however deep a tree the pod built, the deletion holds two folders
        open and never recurses. Never raises: what the pod left there must
        not fail the turn, so an error is logged and the rest left for later.

        REMOVED_DIRNAME is listed a tenth of _REMOVAL_MAX_ENTRIES names at a
        time, and listing it spends nothing of ``budget``, so the budget goes
        to emptying what is listed: a removed folder whose top level holds
        more names than the budget still loses some on every turn."""
        try:
            try:
                dir_lstat(self.conversation, REMOVED_DIRNAME)
            except FileNotFoundError:
                return
            emptied = False
            while not emptied:
                emptied = True
                # self.removed() each time, never a handle kept from before:
                # a move into it may have made it afresh (remove).
                listed = _each_name(self.removed(), budget=budget, max_names=max(1, _REMOVAL_MAX_ENTRIES // 10))
                for name in listed:
                    emptied = False
                    self._empty_removed_entry(name, budget=budget)
            self.close_removed()
            dir_rmdir(self.conversation, REMOVED_DIRNAME)
        except _RemovalBudgetSpent:
            return
        except (OSError, ValueError):
            logger.warning(
                "Could not delete a conversation's removed project files",
                exc_info=True,
                extra={
                    "event": "project_files_removed_not_deleted",
                    **log_context(conversation_id=self.conversation.path.name),
                },
            )

    def close_removed(self) -> None:
        if self._removed is not None:
            self._removed.close()
            self._removed = None

    def _empty_removed_entry(self, name: str, *, budget: _RemovalBudget) -> None:
        removed = self.removed()
        try:
            if not stat.S_ISDIR(dir_lstat(removed, name).st_mode):
                _unlink_listed(removed, name)
                return
            folder = _open_workspace_folder(removed, name)
        except FileNotFoundError:
            return
        try:
            for child in _each_name(folder, budget=budget):
                self.remove(folder, child)
        finally:
            folder.close()
        try:
            # A move into the removed folder may have made it afresh.
            _rmdir_listed(self.removed(), name)
        except FileNotFoundError:
            pass


# The copy removes names a listing of a pinned folder returned, and the walk's
# own names, acting through the folder's descriptor. Such a name is one
# component, never "." or "..", but may hold a "\", which the shared store and
# the upload route take and the pod can make, and which dir_unlink,
# dir_rename and dir_rmdir refuse (is_child_name). Org mode runs on POSIX
# only, so every folder the copy opens has a descriptor.


def _descriptor(folder: PinnedDir) -> int:
    if folder.fd is None:
        raise NotImplementedError("the project files copy acts through pinned descriptors")
    return folder.fd


def _unlink_listed(folder: PinnedDir, name: str) -> None:
    os.unlink(name, dir_fd=_descriptor(folder))


def _rename_listed(source: PinnedDir, source_name: str, destination: PinnedDir, destination_name: str) -> None:
    os.rename(source_name, destination_name, src_dir_fd=_descriptor(source), dst_dir_fd=_descriptor(destination))


def _rmdir_listed(folder: PinnedDir, name: str) -> None:
    os.rmdir(name, dir_fd=_descriptor(folder))


class _CopyPlace(NamedTuple):
    #: The project folder the copy reads.
    source: PinnedDir
    folders: _CopyFolders
    #: The project's conversations folder, and the conversation's name in it.
    conversations: PinnedDir
    conv_seg: str

    def still_there(self) -> bool:
        """Whether the conversation's folder this staging opened is still the
        one at the conversation's name. A conversation delete removes it (or
        a project delete, with the conversations folder), and a folder made
        again there since is not this staging's either."""
        try:
            at_name = dir_lstat(self.conversations, self.conv_seg)
        except FileNotFoundError:
            return False
        held = os.fstat(_descriptor(self.folders.conversation))
        return (at_name.st_dev, at_name.st_ino) == (held.st_dev, held.st_ino)


def _empty_what_was_made(folders: _CopyFolders, conv_seg: str) -> None:
    """Empty the project-files folder and delete REMOVED_DIRNAME of a
    conversation's folder that went away while its copy ran, through the
    handles this staging holds, so whatever that folder now is holds no copy.
    A folder a delete removed is empty already. Never raises: logged."""
    try:
        _prune_folder(folders, keep=set(), budget=_RemovalBudget.start())
        folders.delete_removed(budget=_RemovalBudget.start())
    except (_RemovalBudgetSpent, OSError, ValueError):
        logger.warning(
            "Could not empty the project files copy of a conversation whose folder went away",
            exc_info=True,
            extra={"event": "project_files_gone_copy_not_emptied", **log_context(conversation_id=conv_seg)},
        )


@contextmanager
def _opened_copy_folders(project: Path, conv_seg: str) -> Iterator[_CopyPlace | None]:
    """The project folder and the conversation's copy folders (_CopyFolders),
    the project-files folder made when missing; None when there is no project
    folder. The conversations folder is the server's: the pod mounts its
    conversation's folder inside it and can't reach it. The conversation's
    folder is the pod's mount, so the pod can't replace it but can change its
    mode, and is opened as one of its folders (_open_workspace_folder)."""
    with ExitStack() as stack:
        try:
            parent = stack.enter_context(pinned_dir(project.parent, nofollow_base=True))
            source: PinnedDir | None = open_pinned_child(parent, project.name)
        except (FileNotFoundError, NotADirectoryError):
            source = None
        if source is None:
            yield None
            return
        stack.callback(source.close)
        conversations = _open_or_make_dir(source, CONVERSATIONS_DIRNAME)
        stack.callback(conversations.close)
        _make_dir(conversations, conv_seg)
        conversation = _open_workspace_folder(conversations, conv_seg)
        stack.callback(conversation.close)
        folders = _CopyFolders(conversation)
        stack.callback(folders.close)
        yield _CopyPlace(source, folders, conversations, conv_seg)


class _MirrorFolders:
    """The copy's folders on the way to the file being copied. The walk is
    depth first, so each folder is opened (and made, when missing) once, while
    the walk is inside it, and closed when the walk leaves it."""

    def __init__(self, folders: _CopyFolders) -> None:
        self.folders = folders
        self._open: list[tuple[str, PinnedDir]] = []

    def __enter__(self) -> _MirrorFolders:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def folder(self, parts: tuple[str, ...]) -> PinnedDir:
        """The copy's folder at ``parts`` below the project-files folder."""
        depth = 0
        while depth < min(len(parts), len(self._open)) and self._open[depth][0] == parts[depth]:
            depth += 1
        while len(self._open) > depth:
            self._open.pop()[1].close()
        for name in parts[depth:]:
            parent = self._open[-1][1] if self._open else self.folders.mirror
            self._open.append((name, _secure_workspace_child(parent, name)))
        return self._open[-1][1] if self._open else self.folders.mirror

    def close(self) -> None:
        while self._open:
            self._open.pop()[1].close()


class _Copy:
    """One turn's copy, file by file, in the order the walk finds them."""

    def __init__(self, *, record: CopyRecord, limits: ProjectFileCopyLimits, deadline: float) -> None:
        self.record = record
        self.limits = limits
        self.deadline = deadline
        self.new_record = CopyRecord()
        self.kept: list[tuple[str, ...]] = []
        self.not_copied: list[NotCopiedFile] = []
        #: Files left out because their path is not valid UTF-8: the list
        #: and the record are JSON, which can't hold such a name, so they are
        #: counted only (utf8_path).
        self.unnameable = 0
        self.total = 0
        self.copied = 0
        self.copied_bytes = 0

    def _leave_out(self, found: OrdinaryFile, size: int, reason: NotCopiedReason) -> None:
        self.not_copied.append(NotCopiedFile(path="/".join(found.parts), size=size, reason=reason))

    def add(self, found: OrdinaryFile, folders: _MirrorFolders) -> None:
        if not utf8_path(found.parts):
            self.unnameable += 1
            return None
        size = found.info.st_size
        if size > self.limits.max_file_bytes:
            return self._leave_out(found, size, NotCopiedReason.TOO_LARGE)
        if len(self.kept) >= self.limits.max_files:
            return self._leave_out(found, size, NotCopiedReason.TOO_MANY_FILES)
        if self.total + size > self.limits.max_total_bytes:
            return self._leave_out(found, size, NotCopiedReason.OVER_TOTAL_BYTES)
        previous = self.record.files.get("/".join(found.parts))
        if previous is None and time.monotonic() >= self.deadline:
            # Nothing to check, and no time to copy: no folder is opened.
            return self._leave_out(found, size, NotCopiedReason.OUT_OF_TIME)
        target = folders.folder(found.parts[:-1])
        if previous is not None and _unchanged(previous, source=found.info, target=target, name=found.name):
            self._keep(found, previous, record=True)
            return None
        if time.monotonic() >= self.deadline:
            return self._leave_out(found, size, NotCopiedReason.OUT_OF_TIME)
        outcome = _copy_file(
            found, target, folders=folders.folders,
            max_file_bytes=self.limits.max_file_bytes,
            room=self.limits.max_total_bytes - self.total,
            deadline=self.deadline,
        )
        if isinstance(outcome, NotCopiedReason):
            return self._leave_out(found, size, outcome)
        if outcome is not None:
            self._keep(found, outcome.staged, record=outcome.verified)
            self.copied += 1
            self.copied_bytes += outcome.staged.staged.size
        return None

    def _keep(self, found: OrdinaryFile, staged: StagedCopy, *, record: bool) -> None:
        if record:
            self.new_record.files["/".join(found.parts)] = staged
        self.kept.append(found.parts)
        self.total += staged.staged.size


def _unchanged(previous: StagedCopy, *, source: os.stat_result, target: PinnedDir, name: str) -> bool:
    """Whether the source is the one the record copied and the copy is still
    the file the server wrote: same inode, size, and modification and change
    times on both, and the copy still readable at its mode. The pod can
    change the mode while the server writes the copy, before the stat the
    record keeps, so the mode is checked here too."""
    if previous.source != FileIdentity.of(source):
        return False
    try:
        info = dir_lstat(target, name)
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == _COPY_FILE_MODE
        and FileIdentity.of(info) == previous.staged
    )


def _copy_file(
    found: OrdinaryFile, target: PinnedDir, *, folders: _CopyFolders, max_file_bytes: int, room: int,
    deadline: float,
) -> _Copied | NotCopiedReason | None:
    """Copy one project file into ``target`` under its own name. None when the
    file went away, changed size or stopped being an ordinary file since the
    walk saw it (its copy is then pruned); a reason when it no longer fits, or
    the budget ran out mid-copy (the partial copy is removed)."""
    try:
        src = dir_open(found.directory, found.name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    except OSError as error:
        if error.errno in ABSENT_ERRNOS:
            return None
        raise
    try:
        source = os.fstat(src)
        if not stat.S_ISREG(source.st_mode) or source.st_nlink != 1:
            return None
        if source.st_size > max_file_bytes:
            return NotCopiedReason.TOO_LARGE
        if source.st_size > room:
            return NotCopiedReason.OVER_TOTAL_BYTES
        dst = _open_copy_target(target, found.name, folders=folders)
        out_of_time = False
        try:
            digest = hashlib.sha256()
            written = 0
            for chunk in read_bounded(src, source.st_size, _COPY_CHUNK_BYTES):
                if time.monotonic() >= deadline:
                    out_of_time = True
                    break
                digest.update(chunk)
                write_all(dst, chunk)
                written += len(chunk)
            if not out_of_time:
                if written != source.st_size:
                    return None
                os.fsync(dst)
                checked = _read_back(dst, size=written, digest=digest.digest(), deadline=deadline)
                staged = checked if checked is not None else os.fstat(dst)
        finally:
            os.close(dst)
        if out_of_time:
            folders.remove(target, found.name)
            return NotCopiedReason.OUT_OF_TIME
        return _Copied(
            StagedCopy(source=FileIdentity.of(source), staged=FileIdentity.of(staged)),
            verified=checked is not None,
        )
    finally:
        os.close(src)


def _read_back(fd: int, *, size: int, digest: bytes, deadline: float) -> os.stat_result | None:
    """The copy's stat when it holds exactly the bytes the server wrote and
    nothing changed it while they were read back; None otherwise, or when the
    budget runs out first. The pod can write the copy while the server copies
    it, and a stat taken after such a write would record the pod's bytes as
    the server's: a copy that is not verified here is kept for this turn but
    left out of the record, so the next turn copies it again."""
    before = os.fstat(fd)
    if before.st_size != size:
        return None
    os.lseek(fd, 0, os.SEEK_SET)
    check = hashlib.sha256()
    # One byte past the size, so bytes the pod appended show.
    for chunk in read_bounded(fd, size + 1, _COPY_CHUNK_BYTES):
        if time.monotonic() >= deadline:
            return None
        check.update(chunk)
    after = os.fstat(fd)
    if FileIdentity.of(after) != FileIdentity.of(before) or check.digest() != digest:
        return None
    return after


def _open_copy_target(target: PinnedDir, name: str, *, folders: _CopyFolders) -> int:
    """A descriptor for reading and writing the copy ``name`` in ``target``,
    emptied.

    An existing copy that is a regular file with one name is emptied in place,
    keeping its inode for a pod that holds it open. Anything else the pod put
    there (a link, a folder, a pipe, a second name for one of its own files,
    a file it made unreadable) is removed (_CopyFolders.remove), never
    written through, and a new private file made. A name
    made between the check and the create (another staging of the same
    conversation, or the pod) is checked once more."""
    flags = os.O_RDWR | O_NOFOLLOW | O_NONBLOCK
    for _attempt in range(2):
        try:
            fd = dir_open(target, name, flags)
        except FileNotFoundError:
            pass
        except OSError as error:
            if error.errno not in _REPLACEABLE_ERRNOS:
                raise
            folders.remove(target, name)
        else:
            info = os.fstat(fd)
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                try:
                    os.ftruncate(fd, 0)
                    os.fchmod(fd, _COPY_FILE_MODE)
                except BaseException:
                    os.close(fd)
                    raise
                return fd
            os.close(fd)
            folders.remove(target, name)
        try:
            return dir_open(target, name, flags | os.O_CREAT | os.O_EXCL, _COPY_FILE_MODE)
        except FileExistsError:
            continue
    raise FileExistsError(errno.EEXIST, "a copy's name was taken twice while it was made", name)


def _write_in_place(folders: _CopyFolders, name: str, data: bytes) -> None:
    """Write ``data`` as the file ``name`` in the project-files folder."""
    fd = _open_copy_target(folders.mirror, name, folders=folders)
    try:
        write_all(fd, data)
    finally:
        os.close(fd)


def _folders_of(keep: set[tuple[str, ...]]) -> set[tuple[str, ...]]:
    """Every folder on the way to a path in ``keep``."""
    return {parts[:depth] for parts in keep for depth in range(1, len(parts))}


def _each_name(folder: PinnedDir, *, budget: _RemovalBudget, max_names: int | None = None) -> Iterator[str]:
    """Each name in ``folder``, for the caller to remove or keep, from a
    listing taken first: removing names while a listing of the same folder
    runs may skip others, on a network file store above all. Each name listed
    spends ``budget``, so a folder the pod filled is never read past it, and
    the names listed before it ran out are still handed out, so a later turn
    always gets further. Raises _RemovalBudgetSpent before the next name once
    the time is up, and after the last one when the listing was cut short.

    With ``max_names``, the listing spends nothing of ``budget`` and stops at
    that many names, for the caller to list the folder again once it removed
    them."""
    names: list[str] = []
    complete = True
    with dir_scandir(folder) as scan:
        for entry in scan:
            if max_names is not None:
                if len(names) >= max_names:
                    break
            elif not budget.spend():
                complete = False
                break
            names.append(entry.name)
    for name in names:
        if budget.out_of_time():
            raise _RemovalBudgetSpent
        yield name
    if not complete:
        raise _RemovalBudgetSpent


def _prune_folder(folders: _CopyFolders, *, keep: set[tuple[str, ...]], budget: _RemovalBudget) -> None:
    """Remove every entry below the project-files folder whose path is not
    in ``keep``, nor a folder on the way to one. A folder to remove is moved
    out whole (_CopyFolders.remove), so the prune descends only into folders
    on the way to a kept path, which the walk's depth bound keeps shallow. It
    does so from a list, not by recursion, with one of them open at a time.
    Each entry examined spends ``budget``, which raises _RemovalBudgetSpent
    once spent."""
    _own_folder(folders.mirror)
    on_the_way = _folders_of(keep)
    pending: list[tuple[str, ...]] = [()]
    while pending:
        prefix = pending.pop()
        try:
            folder = _open_copy_path(folders.mirror, prefix)
        except FileNotFoundError:
            continue
        try:
            for name in _each_name(folder, budget=budget):
                parts = (*prefix, name)
                if parts in keep:
                    continue
                if parts in on_the_way:
                    try:
                        is_folder = stat.S_ISDIR(dir_lstat(folder, name).st_mode)
                    except FileNotFoundError:
                        continue
                    if is_folder:
                        pending.append(parts)
                        continue
                folders.remove(folder, name)
        finally:
            if folder is not folders.mirror:
                folder.close()


def _open_copy_path(mirror: PinnedDir, parts: tuple[str, ...]) -> PinnedDir:
    """The copy's folder at ``parts`` below ``mirror``, or ``mirror`` itself
    for none, opening each folder on the way (_open_workspace_folder) and
    closing it once the next is open, or once opening the next fails, since
    the pod can remove or replace a folder on the way at any time. A new
    handle is the caller's to close."""
    current = mirror
    for name in parts:
        try:
            child = _open_workspace_folder(current, name)
        finally:
            if current is not mirror:
                current.close()
        current = child
    return current


def _open_or_make_dir(parent: PinnedDir, name: str) -> PinnedDir:
    _make_dir(parent, name)
    return open_pinned_child(parent, name)


#: How many times a copy that failed tries to empty its folder when the pod
#: took a folder's access away. The pod runs as the server's user, so it can
#: take that access away again right after the server gives it back
#: (_open_workspace_folder), and each try opens the folders afresh, which
#: gives it back once more.
_EMPTY_ATTEMPTS = 3


def _empty_project_files_copy(project: Path, conv_seg: str) -> bool:
    """Remove everything in the conversation's project-files folder, making
    the folder when missing, and write a list there that says the copy failed
    (NotCopiedFiles.copy_failed), so the agent tells the user rather than
    answering without the project's files. A folder that can't be emptied
    within its budget is set aside whole (_CopyFolders.set_aside). A try
    refused access is made again, up to _EMPTY_ATTEMPTS. True when that is
    done, or there is no project folder to copy from; False when it could
    not be, logged. The CopyRecord needs no change, since a removed copy
    never matches it."""
    for attempt in range(1, _EMPTY_ATTEMPTS + 1):
        try:
            _empty_project_files_copy_once(project, conv_seg)
        except PermissionError:
            if attempt < _EMPTY_ATTEMPTS:
                logger.warning(
                    "Access refused while emptying a conversation's project files copy; trying again",
                    exc_info=True,
                    extra={"event": "project_files_empty_retried", **log_context(conversation_id=conv_seg)},
                )
                continue
            logger.error(
                "Could not empty a conversation's project files copy",
                exc_info=True,
                extra={"event": "project_files_copy_not_emptied", **log_context(conversation_id=conv_seg)},
            )
            return False
        except Exception:
            logger.error(
                "Could not empty a conversation's project files copy",
                exc_info=True,
                extra={"event": "project_files_copy_not_emptied", **log_context(conversation_id=conv_seg)},
            )
            return False
        return True
    return False


def _empty_project_files_copy_once(project: Path, conv_seg: str) -> None:
    """One try of _empty_project_files_copy, which raises when it fails."""
    with _opened_copy_folders(project, conv_seg) as place:
        if place is None:
            return
        try:
            _prune_folder(place.folders, keep=set(), budget=_RemovalBudget.start())
        except (_RemovalBudgetSpent, OSError, ValueError):
            logger.warning(
                "Could not empty a conversation's project files copy; setting it aside",
                exc_info=True,
                extra={"event": "project_files_empty_set_aside", **log_context(conversation_id=conv_seg)},
            )
            place.folders.set_aside()
        _write_in_place(
            place.folders, NOT_COPIED_FILENAME, NotCopiedFiles(copy_failed=True).model_dump_json().encode(),
        )


def _read_copy_record(record_path: Path) -> CopyRecord:
    """The conversation's CopyRecord, or an empty one when there is none or it
    does not parse. Any other read error raises."""
    try:
        raw = record_path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return CopyRecord()
    try:
        return CopyRecord.model_validate_json(raw)
    except ValidationError:
        logger.warning(
            "Discarding a conversation's unreadable project files record",
            extra={"event": "project_files_record_unreadable", **log_context(conversation_id=record_path.stem)},
        )
        return CopyRecord()


def _write_copy_record(record_path: Path, record: CopyRecord) -> None:
    """Save the record. A record that could not be saved leaves the old one,
    which is safe: every copy rewritten since has a new change time, so the
    old record matches none of them, and they are copied again."""
    try:
        with pinned_dir(record_path.parent, create=True) as root:
            dir_atomic_write(root, record_path.name, content=record.model_dump_json().encode())
    except OSError:
        logger.warning(
            "Could not save a conversation's project files record",
            exc_info=True,
            extra={"event": "project_files_record_not_saved", **log_context(conversation_id=record_path.stem)},
        )
