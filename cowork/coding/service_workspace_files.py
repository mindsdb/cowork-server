from __future__ import annotations

from cowork.coding.context import workspace_files
from cowork.coding.contracts import CodingSession
from cowork.coding.workspace_files import WorkspaceFileBrowser
from cowork.coding.workspace_models import (
    WorkspaceEntryPage,
    WorkspaceFileContent,
    WorkspaceResourcePage,
    WorkspaceSearchPage,
)


class CodingWorkspaceFilesOperations:
    """Read-only workspace navigation shared by local Code sessions."""

    def get_session(self, session_id: str) -> CodingSession:  # pragma: no cover - mixin contract
        raise NotImplementedError

    def workspace_files(self, session_id: str, query: str = "", limit: int = 40) -> list[dict[str, str]]:
        return workspace_files(self._local_file_session(session_id), query, limit)

    def workspace_resources(self, session_id: str) -> WorkspaceResourcePage:
        return WorkspaceFileBrowser(self._local_file_session(session_id)).resources()

    def workspace_entries(
        self,
        session_id: str,
        resource_id: str,
        path: str = "",
    ) -> WorkspaceEntryPage:
        return WorkspaceFileBrowser(self._local_file_session(session_id)).entries(resource_id, path)

    def workspace_file(
        self,
        session_id: str,
        resource_id: str,
        path: str,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> WorkspaceFileContent:
        return WorkspaceFileBrowser(self._local_file_session(session_id)).file(
            resource_id,
            path,
            line_start,
            line_end,
        )

    def workspace_search(
        self,
        session_id: str,
        query: str,
        resource_id: str | None = None,
        limit: int = 60,
    ) -> WorkspaceSearchPage:
        return WorkspaceFileBrowser(self._local_file_session(session_id)).search(query, resource_id, limit)

    def _local_file_session(self, session_id: str) -> CodingSession:
        session = self.get_session(session_id)
        if not session.computer_is_local or not session.task_capabilities.files:
            raise RuntimeError(
                "Task files stay on the computer running this task. Open Code Mode on that computer to browse them."
            )
        return session
