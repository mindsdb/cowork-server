"""Recipient presentation and file reads. Never used by execution or mutation paths."""
from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from uuid import UUID

from sqlmodel import select

from cowork.common.paths import (
    O_NOFOLLOW, O_NONBLOCK, PinnedDir, dir_open, open_pinned_child, pinned_dir, read_bounded,
)
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import ScopedSession, unsafe_unscoped_session
from cowork.managed import reads_shared_registry
from cowork.models.project import Project
from cowork.schemas.project_files import ProjectFileListResponse, ProjectFileMetadata
from cowork.schemas.projects import ProjectResponse
from cowork.schemas.shared_resources import ProjectCapabilities, ResourceAttribution
from cowork.services.artifact_roots import SERVER_PROJECT_FOLDERS
from cowork.services.project_file_view import (
    MAX_LIST_ENTRIES, WalkState, ordinary_path, safe_component, utf8_path, walk_ordinary_files,
)
from cowork.shared_projects import SharedProject, read_shared_projects

TEXT_MAX_BYTES = 2 * 1024 * 1024
DOWNLOAD_MAX_BYTES = 100 * 1024 * 1024
# What a recipient sees as "not there": the walk skips such an entry, and the
# routes answer 404.
UNREADABLE_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EACCES, errno.EPERM})
#: Folder and file names a recipient's view hides at any depth, compared
#: case-folded. The ordinary-file rule hides the server's folders at the
#: project root only and memory nowhere (cowork.services.project_file_view);
#: a recipient's view keeps hiding them everywhere.
_RECIPIENT_HIDDEN_NAMES = SERVER_PROJECT_FOLDERS | {"memory"}


def _recipient_component(name: str) -> bool:
    """A name a recipient's view of a shared project shows: a safe one
    (cowork.services.project_file_view) with no ``:`` or ``\\`` either."""
    return safe_component(name) and not any(c in name for c in ("\\", ":"))


def _recipient_file_component(name: str) -> bool:
    """A name a recipient's view shows on the way to a file, at any depth,
    on top of the ordinary-file rule: a recipient's name
    (_recipient_component) that is none of _RECIPIENT_HIDDEN_NAMES and is
    valid UTF-8. A name that is not can't be sent in the listing's JSON, so
    it is left out, as the copy leaves it out of a cloud turn's files
    (utf8_path)."""
    return (
        _recipient_component(name)
        and name.casefold() not in _RECIPIENT_HIDDEN_NAMES
        and utf8_path((name,))
    )


def _recipient_path(parts: tuple[str, ...]) -> bool:
    """A path a recipient's view shows."""
    return ordinary_path(parts) and all(_recipient_file_component(part) for part in parts)


def _file_parts(path: str) -> tuple[str, ...]:
    parts = tuple(path.split("/"))
    if not _recipient_path(parts):
        raise FileNotFoundError("File not found")
    return parts


class SharedProjectFiles:
    def __init__(self, session: ScopedSession):
        self.session = session

    def _project(self, entry: SharedProject) -> Project | None:
        # Explicit read-only cross-partition query. No foreign row is passed to
        # ProjectService or any mutation/runtime code, whose scope stays intact.
        row = unsafe_unscoped_session(self.session).exec(select(Project).where(
            Project.id == entry.project_id,
            Project.org_id == entry.partition_id,
            Project.created_by == entry.user_id,
        )).first()
        if row is None or not _recipient_component(row.name):
            return None
        expected = Path(get_app_settings().storage.shared_root) / entry.partition_id / "projects" / row.name
        if Path(row.path) != expected:
            return None
        return row

    def list_projects(self) -> list[ProjectResponse]:
        if not reads_shared_registry(self.session.scope, resource=PersonalResource.PROJECTS):
            return []
        result = []
        for entry in read_shared_projects().root:
            if not entry.readable_by(self.session.scope):
                continue
            row = self._project(entry)
            if row is not None:
                result.append(ProjectResponse(
                    id=row.id, name=row.name, display_name=row.display_name,
                    path="", is_active=False, created_at=row.created_at,
                    modified_at=row.modified_at, created_by=row.created_by,
                    shared=True,
                    attribution=ResourceAttribution(created_by=None, last_modified_by=None, last_modified_at=None),
                    capabilities=ProjectCapabilities(can_rename=False, can_delete=False,
                        can_edit_instructions=False, can_run=False, can_write_files=False),
                ))
        return result

    @contextmanager
    def directory(self, project_id: UUID) -> Iterator[PinnedDir]:
        entry = next((entry for entry in read_shared_projects().root
                      if entry.project_id == project_id and entry.readable_by(self.session.scope)), None)
        row = self._project(entry) if entry else None
        if row is None:
            raise FileNotFoundError("File not found")
        # Pin every tier below the trusted shared root. No owner-controlled
        # directory or filename is followed through a symlink.
        with ExitStack() as stack:
            directory = stack.enter_context(pinned_dir(get_app_settings().storage.shared_root, nofollow_base=True))
            for name in (entry.partition_id, "projects", row.name):
                directory = open_pinned_child(directory, name)
                stack.callback(directory.close)
            # A delete/recreate or rename between row lookup and directory open
            # must not substitute the new project's files under the old grant.
            live = unsafe_unscoped_session(self.session).exec(select(Project.id).where(
                Project.id == entry.project_id, Project.org_id == entry.partition_id,
                Project.created_by == entry.user_id, Project.path == row.path,
            )).first()
            if live is None:
                raise FileNotFoundError("File not found")
            yield directory

    def list_files(self, project_id: UUID) -> ProjectFileListResponse:
        state = WalkState()
        with self.directory(project_id) as directory:
            files = [
                ProjectFileMetadata(path="/".join(found.parts), name=found.name,
                    size=found.info.st_size, modified=found.info.st_mtime, is_dir=False)
                # The walk never enters a folder a recipient can't see, so its
                # entry budget goes to the files they can.
                for found in walk_ordinary_files(
                    directory, state=state, max_entries=MAX_LIST_ENTRIES, unreadable=UNREADABLE_ERRNOS,
                    component=_recipient_file_component,
                )
                if _recipient_path(found.parts)
            ]
        return ProjectFileListResponse(files=sorted(files, key=lambda file: file.path), truncated=state.truncated)

    def read_file(self, project_id: UUID, path: str, *, limit: int) -> tuple[bytes, os.stat_result]:
        parts = _file_parts(path)
        with ExitStack() as stack:
            directory = stack.enter_context(self.directory(project_id))
            root = os.path.abspath(directory.path)
            target = os.path.abspath(os.path.join(root, *parts))
            if not target.startswith(root + os.sep):
                raise FileNotFoundError("File not found")
            # Use only the checked relative components. The pinned descriptors
            # and O_NOFOLLOW still enforce the filesystem's symlink boundary.
            parts = tuple(os.path.relpath(target, root).split(os.sep))
            for name in parts[:-1]:
                directory = open_pinned_child(directory, name)
                stack.callback(directory.close)
            fd = dir_open(directory, parts[-1], os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise FileNotFoundError("File not found")
            if info.st_size > limit:
                raise OverflowError("File too large")
            return b"".join(read_bounded(fd, info.st_size)), info
