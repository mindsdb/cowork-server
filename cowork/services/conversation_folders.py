"""Working folders a user attaches to a desktop chat.

A working folder is a local directory the agent is told it may read and write
in, besides the chat's project. The scratchpad is not sandboxed, so these rules
decide what the agent is offered, not what it can reach. They are checked again
every time a folder is used, because the folder on disk can change after it was
attached.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.db.scoped import ScopedSession
from cowork.models.conversation import Conversation
from cowork.models.conversation_folder import ConversationFolder
from cowork.services.conversations import ConversationService

MAX_FOLDERS_PER_CONVERSATION = 16
_MAX_PATH_LENGTH = 1024

_NOT_AVAILABLE = "Working folders are only available in the desktop app"
_NOT_A_FOLDER = "Choose an existing local folder"
_APP_DATA = "Choose a folder that does not hold Cowork's own data"
_IN_PROJECT = "This folder is already part of the chat's project"


class FolderRefused(ValueError):
    """The folder cannot be a working folder; the message is shown to the user."""


class FolderAlreadyAttached(ValueError):
    """The chat already has this folder."""


class FolderLimitReached(ValueError):
    """The chat already has the maximum number of working folders."""


class FolderNotFound(LookupError):
    """No such chat for the caller, or no such folder on that chat."""


def _comparable(path: Path) -> Path:
    """`path` in the form two paths are compared in.

    macOS volumes are case-insensitive by default and `resolve()` keeps the
    caller's case, so `/Users/x/.COWORK` would otherwise slip past `.cowork`.
    """
    if sys.platform == "darwin":
        return Path(str(path).casefold())
    return path


def _overlaps(a: Path, b: Path) -> bool:
    """True when `a` and `b` are the same folder or one contains the other."""
    return a == b or b in a.parents or a in b.parents


def _store_roots() -> list[Path]:
    """Every directory Cowork keeps its own state in, resolved.

    Each store can be moved on its own, so `COWORK_HOME` alone does not cover
    the vault or the projects root.
    """
    settings = get_app_settings()
    raw = [
        cowork_home(),
        settings.project.root_dir,
        settings.file.root_dir,
        settings.skill.root_dir,
        settings.connector.vault_dir,
        settings.memory.root_dir,
        settings.coding.root_dir,
    ]
    roots: list[Path] = []
    for item in raw:
        try:
            roots.append(_comparable(Path(item).expanduser().resolve(strict=False)))
        except (OSError, RuntimeError):
            continue
    return roots


def _unavailable_reason() -> str | None:
    """Why this deployment cannot offer working folders, if it cannot.

    Checked before touching the filesystem: an org deployment does not run on
    the caller's machine, and a remote turn runs in a pod that never sees it.
    """
    if get_app_settings().tenancy_mode == "org" or TurnQueueSettings().is_remote:
        return _NOT_AVAILABLE
    return None


def resolve_folder(raw_path: str, project_path: str | None) -> Path:
    """The resolved folder `raw_path` names, if it may be a working folder.

    Inputs: the requested path and the chat's project folder. Output: the
    resolved absolute path. Raises `FolderRefused` with the user-facing reason
    otherwise. Reads the filesystem; writes nothing.
    """
    reason = _unavailable_reason()
    if reason is not None:
        raise FolderRefused(reason)
    path = Path(raw_path)
    if "\x00" in raw_path or not path.is_absolute():
        raise FolderRefused(_NOT_A_FOLDER)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FolderRefused(_NOT_A_FOLDER) from exc
    if not resolved.is_dir() or len(str(resolved)) > _MAX_PATH_LENGTH:
        raise FolderRefused(_NOT_A_FOLDER)
    candidate = _comparable(resolved)
    if any(_overlaps(candidate, root) for root in _store_roots()):
        raise FolderRefused(_APP_DATA)
    if project_path:
        try:
            project = _comparable(Path(project_path).resolve(strict=False))
        except (OSError, RuntimeError):
            project = None
        if project is not None and (candidate == project or project in candidate.parents):
            raise FolderRefused(_IN_PROJECT)
    return resolved


def folder_refusal(raw_path: str, project_path: str | None) -> str | None:
    """The user-facing reason `raw_path` cannot be used now, or None if it can.

    Same rules as `resolve_folder`; for callers that skip a folder rather than
    report an error, such as the agent prompt and the folder list.
    """
    try:
        resolve_folder(raw_path, project_path)
    except FolderRefused as exc:
        return str(exc)
    return None


class ConversationFolderService:
    """Attach, list and remove a chat's working folders.

    Every call first loads the chat through `ConversationService`, so a caller
    only ever reaches folders of a chat they own.
    """

    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _conversation(self, conversation_id: UUID) -> Conversation:
        try:
            return ConversationService(self.session).get_conversation(conversation_id)
        except ValueError as exc:
            raise FolderNotFound("Conversation not found") from exc

    def _rows(self, conversation_id: UUID) -> list[ConversationFolder]:
        return list(
            self.session.exec(
                self.session.select(ConversationFolder)
                .where(ConversationFolder.conversation_id == conversation_id)
                .order_by(ConversationFolder.created_at, ConversationFolder.id)
            ).all()
        )

    def list_folders(self, conversation_id: UUID) -> tuple[Conversation, list[ConversationFolder]]:
        """The chat and its working folders, oldest first. Raises `FolderNotFound`."""
        conversation = self._conversation(conversation_id)
        return conversation, self._rows(conversation.id)

    def get_folder(
        self, conversation_id: UUID, folder_id: UUID
    ) -> tuple[Conversation, ConversationFolder]:
        """One folder of the chat, matched on both ids. Raises `FolderNotFound`."""
        conversation = self._conversation(conversation_id)
        row = self.session.exec(
            self.session.select(ConversationFolder).where(
                ConversationFolder.id == folder_id,
                ConversationFolder.conversation_id == conversation.id,
            )
        ).first()
        if row is None:
            raise FolderNotFound("Folder not found")
        return conversation, row

    def add_folder(self, conversation_id: UUID, raw_path: str) -> ConversationFolder:
        """Attach `raw_path` to the chat and commit.

        Raises `FolderNotFound`, `FolderRefused`, `FolderAlreadyAttached` or
        `FolderLimitReached`. Two concurrent adds of different folders can both
        pass the limit check; the unique index only settles duplicates.
        """
        conversation = self._conversation(conversation_id)
        project_path = conversation.project.path if conversation.project else None
        resolved = resolve_folder(raw_path, project_path)
        existing = self._rows(conversation.id)
        key = _comparable(resolved)
        if any(_comparable(Path(row.path)) == key for row in existing):
            raise FolderAlreadyAttached("This folder is already attached to the chat")
        if len(existing) >= MAX_FOLDERS_PER_CONVERSATION:
            raise FolderLimitReached(
                f"A chat can have up to {MAX_FOLDERS_PER_CONVERSATION} working folders"
            )
        # Set here rather than by the column default, which SQLite stores to
        # the second: two folders added in one second would list in id order.
        row = ConversationFolder(
            conversation_id=conversation.id,
            path=str(resolved),
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(row)
        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            raise FolderAlreadyAttached("This folder is already attached to the chat") from exc
        self.session.refresh(row)
        return row

    def remove_folder(self, conversation_id: UUID, folder_id: UUID) -> None:
        """Detach one folder from the chat and commit. Raises `FolderNotFound`."""
        _conversation, row = self.get_folder(conversation_id, folder_id)
        self.session.delete(row)
        self.session.commit()
