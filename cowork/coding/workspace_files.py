from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from cowork.coding.contracts import CodingSession, TaskWorkspace
from cowork.coding.workspace_models import (
    WorkspaceEntry,
    WorkspaceEntryPage,
    WorkspaceFileContent,
    WorkspaceResource,
    WorkspaceResourcePage,
    WorkspaceSearchMatch,
    WorkspaceSearchPage,
)

EXCLUDED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
MAX_DIRECTORY_ENTRIES = 500
MAX_FILE_BYTES = 1024 * 1024
MAX_FILE_LINES = 400
MAX_SEARCH_FILES = 20_000
MAX_SEARCH_FILE_BYTES = 512 * 1024
MAX_SEARCH_RESULTS = 100


@dataclass(frozen=True)
class _WorkspaceRoot:
    resource_id: str
    name: str
    path: Path
    kind: str


class WorkspaceFileBrowser:
    """Read-only, resource-scoped access to one task's prepared workspaces."""

    def __init__(self, session: CodingSession) -> None:
        source_name = Path(session.source_path).name or "Folder"
        workspaces = list(session.workspaces) or [TaskWorkspace(
            folder_id="folder",
            # The isolated workspace directory is an internal task UUID. The
            # source folder is the stable, human-facing resource identity.
            folder_name=source_name,
            source_path=session.source_path,
            workspace_path=session.workspace_path,
            workspace_kind=session.workspace_kind,
            repository_root=session.repository_root,
            base_revision=session.base_revision,
            source_dirty=session.source_dirty,
        )]
        self._roots = {
            item.folder_id: _WorkspaceRoot(
                resource_id=item.folder_id,
                name=item.folder_name,
                path=Path(item.workspace_path).resolve(),
                kind="repository" if item.repository_root else "folder",
            )
            for item in workspaces
        }

    def resources(self) -> WorkspaceResourcePage:
        return WorkspaceResourcePage(items=[
            WorkspaceResource(id=root.resource_id, name=root.name, kind=root.kind)
            for root in self._roots.values()
        ])

    def entries(self, resource_id: str, relative_path: str = "") -> WorkspaceEntryPage:
        root, directory = self._resolve(resource_id, relative_path)
        if not directory.is_dir():
            raise ValueError("The selected workspace path is not a folder")
        items: list[WorkspaceEntry] = []
        children = sorted(
            (item for item in directory.iterdir() if item.name not in EXCLUDED_DIRECTORIES),
            key=lambda item: (not item.is_dir(), item.name.casefold(), item.name),
        )
        truncated = len(children) > MAX_DIRECTORY_ENTRIES
        for child in children[:MAX_DIRECTORY_ENTRIES]:
            try:
                resolved = child.resolve(strict=True)
                resolved.relative_to(root.path)
            except (FileNotFoundError, OSError, ValueError):
                continue
            is_directory = resolved.is_dir()
            if not is_directory and not resolved.is_file():
                continue
            child_relative = resolved.relative_to(root.path).as_posix()
            items.append(WorkspaceEntry(
                resource_id=root.resource_id,
                resource_name=root.name,
                path=child_relative,
                name=child.name,
                kind="directory" if is_directory else "file",
                size=None if is_directory else resolved.stat().st_size,
            ))
        return WorkspaceEntryPage(
            resource_id=resource_id,
            path=self._normalize(relative_path),
            items=items,
            truncated=truncated,
        )

    def file(
        self,
        resource_id: str,
        relative_path: str,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> WorkspaceFileContent:
        root, path = self._resolve(resource_id, relative_path)
        if not path.is_file():
            raise ValueError("The selected workspace path is not a file")
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("This file is too large to preview in Code Mode")
        if b"\0" in data:
            raise ValueError("Binary files cannot be previewed in Code Mode")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("This file is not UTF-8 text and cannot be previewed") from exc
        lines = text.splitlines()
        line_count = len(lines)
        if line_count == 0:
            normalized = path.relative_to(root.path).as_posix()
            return WorkspaceFileContent(
                resource_id=root.resource_id,
                resource_name=root.name,
                path=normalized,
                name=path.name,
                content="",
                content_hash=hashlib.sha256(data).hexdigest(),
                line_count=0,
                line_start=0,
                line_end=0,
                truncated=False,
            )
        start = max(1, line_start or 1)
        if start > max(1, line_count):
            raise ValueError("The requested line is outside this file")
        requested_end = line_end or min(line_count, start + MAX_FILE_LINES - 1)
        end = min(line_count, requested_end, start + MAX_FILE_LINES - 1)
        content = "\n".join(lines[start - 1:end])
        if text.endswith("\n") and end == line_count:
            content += "\n"
        normalized = path.relative_to(root.path).as_posix()
        return WorkspaceFileContent(
            resource_id=root.resource_id,
            resource_name=root.name,
            path=normalized,
            name=path.name,
            content=content,
            content_hash=hashlib.sha256(data).hexdigest(),
            line_count=line_count,
            line_start=start,
            line_end=end,
            truncated=start > 1 or end < line_count,
        )

    def search(
        self,
        query: str,
        resource_id: str | None = None,
        limit: int = 60,
    ) -> WorkspaceSearchPage:
        needle = query.strip().casefold()
        if not needle:
            return WorkspaceSearchPage(items=[])
        roots = [self._root(resource_id)] if resource_id else list(self._roots.values())
        requested = max(1, min(limit, MAX_SEARCH_RESULTS))
        matches: list[WorkspaceSearchMatch] = []
        visited = 0
        truncated = False
        for root in roots:
            for current_root, dirs, files in os.walk(root.path, followlinks=False):
                dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRECTORIES)
                for filename in sorted(files):
                    visited += 1
                    if visited > MAX_SEARCH_FILES:
                        truncated = True
                        return WorkspaceSearchPage(items=matches, truncated=truncated)
                    path = Path(current_root, filename)
                    try:
                        resolved = path.resolve(strict=True)
                        relative = resolved.relative_to(root.path).as_posix()
                    except (FileNotFoundError, OSError, ValueError):
                        continue
                    if needle in relative.casefold():
                        matches.append(self._match(root, relative, match_kind="path"))
                        if len(matches) >= requested:
                            return WorkspaceSearchPage(items=matches, truncated=True)
                        continue
                    try:
                        if resolved.stat().st_size > MAX_SEARCH_FILE_BYTES:
                            continue
                        data = resolved.read_bytes()
                        if b"\0" in data:
                            continue
                        text = data.decode("utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    for line_number, line in enumerate(text.splitlines(), start=1):
                        if needle not in line.casefold():
                            continue
                        matches.append(self._match(
                            root,
                            relative,
                            line=line_number,
                            preview=" ".join(line.strip().split())[:240],
                            match_kind="content",
                        ))
                        break
                    if len(matches) >= requested:
                        return WorkspaceSearchPage(items=matches, truncated=True)
        return WorkspaceSearchPage(items=matches, truncated=truncated)

    def absolute_path(self, resource_id: str, relative_path: str) -> Path:
        return self._resolve(resource_id, relative_path)[1]

    def _root(self, resource_id: str | None) -> _WorkspaceRoot:
        if not resource_id or resource_id not in self._roots:
            raise ValueError("This workspace resource is not available to the task")
        return self._roots[resource_id]

    def _resolve(self, resource_id: str, relative_path: str) -> tuple[_WorkspaceRoot, Path]:
        root = self._root(resource_id)
        normalized = self._normalize(relative_path)
        candidate = root.path.joinpath(*PurePosixPath(normalized).parts) if normalized else root.path
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root.path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ValueError("This workspace path is unavailable") from exc
        return root, resolved

    @staticmethod
    def _normalize(relative_path: str) -> str:
        value = relative_path.replace("\\", "/").strip("/")
        parts = PurePosixPath(value).parts if value else ()
        if any(part in {".", ".."} for part in parts):
            raise ValueError("Workspace paths must stay inside the selected resource")
        return PurePosixPath(*parts).as_posix() if parts else ""

    @staticmethod
    def _match(
        root: _WorkspaceRoot,
        path: str,
        *,
        line: int | None = None,
        preview: str = "",
        match_kind: Literal["path", "content"],
    ) -> WorkspaceSearchMatch:
        return WorkspaceSearchMatch(
            resource_id=root.resource_id,
            resource_name=root.name,
            path=path,
            name=PurePosixPath(path).name,
            line=line,
            preview=preview,
            match_kind=match_kind,
        )
