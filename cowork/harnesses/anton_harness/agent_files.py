"""The files a turn's agent can read, found on this server's disk: the
conversation's project, and the files attached to the conversation."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path, PurePath, PurePosixPath
from uuid import UUID

from cowork.common.logger import get_logger, log_context
from cowork.common.paths import safe_join
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.units import run_db
from cowork.services.artifact_roots import CONVERSATIONS_DIRNAME
from cowork.services.files import FileService, attachment_purpose

logger = get_logger(__name__)

# The folder in a conversation's folder where FileService
# .stage_conversation_attachments copies the conversation's attachments, as
# `attachments/<file_id>/<name>`, for a cloud pod to read.
STAGED_ATTACHMENTS_DIRNAME = "attachments"

# Where a cloud pod's scratchpad runs when the controller gives it an
# ephemeral workspace: a copy of the conversation's folder
# (scratchpad-controller live_pod.EPHEMERAL_WORKSPACE).
EPHEMERAL_POD_WORKING_DIR = PurePosixPath("/workspace-ephemeral")

# Where anton runs a pod's scratchpad when the controller names no workspace
# (anton.cloud_turn.session.DEFAULT_CLOUD_WORKSPACE_PATH).
DEFAULT_POD_WORKING_DIR = PurePosixPath("/workspace")


@dataclass(frozen=True)
class Attachment:
    """A file attached to the conversation: its id in the server's files
    store, and where the store keeps it on this server's disk."""

    file_id: str
    stored_path: str


@dataclass(frozen=True)
class NamedFile:
    """The file a path the agent's code names turns out to be, as the web UI
    can open it: ``path`` is its project-relative POSIX path, and
    ``attachment_id`` is its files-store id when it is a file attached to the
    conversation. Exactly one is set."""

    path: str | None = None
    attachment_id: str | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.attachment_id is None):
            raise ValueError("a named file is a project path or an attachment, not both or neither")


@dataclass(frozen=True)
class AgentFiles:
    """What a turn's agent can read, as this server finds it.

    ``project_id`` is the project the turn runs in, and ``attachments`` the
    files attached to the conversation. A turn this server runs works in the
    project root. A cloud pod's turn works in the conversation's folder in
    the project, ``pod_folder``, which the pod names by any of
    ``pod_working_dirs``. Both are empty for a turn this server runs.
    """

    project_root: Path
    project_id: str | None = None
    attachments: tuple[Attachment, ...] = ()
    pod_working_dirs: tuple[PurePosixPath, ...] = ()
    pod_folder: PurePosixPath = field(default_factory=PurePosixPath)

    @classmethod
    def on_this_server(
        cls,
        project_root: str | Path,
        *,
        project_id: UUID | str | None = None,
        attachments: Iterable[Attachment] = (),
    ) -> AgentFiles:
        """A turn this server runs: its scratchpad works in the project root
        (AntonHarness._build_chat_session tells the agent so), and the agent
        reads an attachment where the files store keeps it."""
        return cls(
            project_root=Path(project_root),
            project_id=None if project_id is None else str(project_id),
            attachments=tuple(attachments),
        )

    @classmethod
    def in_cloud_pod(
        cls,
        project_root: str | Path,
        *,
        conversation_id: UUID | str,
        workspace_rel_path: str | None,
        project_id: UUID | str | None = None,
        attachments: Iterable[Attachment] = (),
    ) -> AgentFiles:
        """A turn a cloud pod runs, in the conversation's folder, where the
        server stages its attachments for the pod to read.

        ``workspace_rel_path`` is the project's path in the organization's
        tree as the turn's job carries it to the controller, or None when it
        is unknown. The controller mounts the conversation's folder in the pod
        at ``<shared mount root>/<workspace_rel_path>/conversations/<id>``, or
        copies it to an ephemeral workspace.
        """
        working_dirs = [EPHEMERAL_POD_WORKING_DIR, DEFAULT_POD_WORKING_DIR]
        if workspace_rel_path:
            working_dirs.insert(0, PurePosixPath(
                TurnQueueSettings().pod_shared_mount_root,
                workspace_rel_path.lstrip("/"),
                CONVERSATIONS_DIRNAME,
                str(conversation_id),
            ))
        return cls(
            project_root=Path(project_root),
            project_id=None if project_id is None else str(project_id),
            attachments=tuple(attachments),
            pod_working_dirs=tuple(working_dirs),
            pod_folder=PurePosixPath(CONVERSATIONS_DIRNAME, str(conversation_id)),
        )

    def named_file(self, literal: str) -> NamedFile | None:
        """The file ``literal`` names, read the way the agent's OS reads it:
        one of the conversation's attachments by its files-store id, or a file
        in the agent's working directory by its project path.

        None when it names no file now, a place outside the working directory
        that is no attachment, or a file a link leads out to. A failed lookup
        other than a missing file is logged.
        """
        if not literal:
            return None
        try:
            return self._named_file(literal)
        except (FileNotFoundError, NotADirectoryError, ValueError):
            return None
        except OSError:
            logger.warning(
                "Could not look up a file the agent named; its figure gets no file to open",
                exc_info=True,
                extra=log_context(project_id=self.project_id),
            )
            return None

    def _named_file(self, literal: str) -> NamedFile | None:
        """named_file, letting a failed lookup raise."""
        root = Path(os.path.realpath(self.project_root))
        if self.pod_working_dirs:
            parts = self._parts_in_pod_folder(literal)
            if parts is None:
                return None
            # Resolves links and refuses a folder outside the project.
            working_dir = safe_join(root, *self.pod_folder.parts)
            # Each `..` climbs from where the links before it lead, as the
            # pod's OS climbs, so the result is the file the pod read.
            target = Path(os.path.realpath(working_dir.joinpath(*parts)))
            attached = {
                working_dir.joinpath(
                    STAGED_ATTACHMENTS_DIRNAME, attachment.file_id, PurePath(attachment.stored_path).name,
                ): attachment.file_id
                for attachment in self.attachments
            }
        else:
            working_dir = root
            # An absolute literal replaces the project root, and links resolve
            # before each `..` climbs, as this server's OS reads the path.
            target = Path(os.path.realpath(self.project_root / literal))
            attached = {
                Path(os.path.realpath(attachment.stored_path)): attachment.file_id
                for attachment in self.attachments
            }
        if target in attached:
            named = NamedFile(attachment_id=attached[target])
        else:
            # Any other file must be in the working directory: a pod sees only
            # its conversation's folder, so a link into another member's
            # folder names a file the pod could not read.
            path = _posix_path_in(target, root=working_dir)
            if path is None:
                return None
            named = NamedFile(path=(self.pod_folder / path).as_posix())
        return named if stat.S_ISREG(os.stat(target).st_mode) else None

    def _parts_in_pod_folder(self, literal: str) -> tuple[str, ...] | None:
        """The literal's place in the pod's working directory as path parts,
        each ``..`` kept to climb after links resolve. None when it names a
        place outside the working directory: an absolute path elsewhere, which
        the pod reads from its own disk, or a ``..`` that climbs above it,
        where the pod sees nothing of the project."""
        seen = PurePosixPath(literal)
        if seen.is_absolute():
            working_dir = next((d for d in self.pod_working_dirs if seen.is_relative_to(d)), None)
            if working_dir is None:
                return None
            seen = seen.relative_to(working_dir)
        depth = 0
        for part in seen.parts:
            depth += -1 if part == ".." else 1
            if depth < 0:
                return None
        return seen.parts


def _posix_path_in(target: PurePath, *, root: PurePath) -> str | None:
    """``target``'s place in ``root`` with forward slashes on every OS, or
    None when it is outside ``root``. The web UI reads the path as POSIX and
    drops one with a backslash."""
    if not target.is_relative_to(root):
        return None
    return target.relative_to(root).as_posix()


def conversation_attachments(session: ScopedSession, *, conversation_id: UUID | str) -> tuple[Attachment, ...]:
    """The files attached to the conversation that the session's user owns:
    the ones the files API serves them and FileService
    .stage_conversation_attachments copies into a pod's folder."""
    rows = FileService(session).list_file_rows(attachment_purpose(str(conversation_id)))
    return tuple(Attachment(file_id=str(row.id), stored_path=row.path) for row in rows if row.path)


async def read_conversation_attachments(
    *, conversation_id: UUID | str, scope: TenantScope,
) -> tuple[Attachment, ...]:
    """conversation_attachments, read in a database unit of its own. A failed
    read is logged and lists none, so the turn runs on and its figures name
    no attached file."""
    try:
        return await run_db(partial(conversation_attachments, conversation_id=conversation_id), scope=scope)
    except Exception:
        logger.warning(
            "Could not read the conversation's attachments; its figures name no attached file",
            exc_info=True,
            extra=log_context(conversation_id=conversation_id),
        )
        return ()
