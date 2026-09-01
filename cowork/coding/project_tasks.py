from __future__ import annotations

from cowork.coding.contracts import CodingEvent, EventType, WorkspaceKind
from cowork.coding.delivery import ProjectDeliveryService
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.operation_types import EventEmitter, GetSession, MaintenanceSession
from cowork.coding.project_models import (
    DraftPullRequestRequest,
    PullRequestActionRequest,
    PullRequestStatus,
)
from cowork.coding.project_service import CodeProjectService
from cowork.coding.project_workspaces import ProjectWorkspaceManager
from cowork.coding.store import CodingStore
from cowork.coding.workspace import WorkspaceError, WorkspaceManager


class ProjectTaskOperations:
    """Git, validation, handoff, and pull-request operations for project tasks."""

    def __init__(
        self,
        *,
        get_session: GetSession,
        maintenance_session: MaintenanceSession,
        emit: EventEmitter,
        store: CodingStore,
        workspaces: WorkspaceManager,
        project_workspaces: ProjectWorkspaceManager,
        projects: CodeProjectService,
        delivery: ProjectDeliveryService,
    ) -> None:
        self.get_session = get_session
        self.maintenance_session = maintenance_session
        self.emit = emit
        self.store = store
        self.workspaces = workspaces
        self.project_workspaces = project_workspaces
        self.projects = projects
        self.delivery = delivery

    def git_state(self, session_id: str):
        session = self.get_session(session_id)
        if session.workspaces:
            return self.project_workspaces.git_states(session.workspaces)[0]
        return self.workspaces.git_state(session.source_path, session.workspace_path)

    def git_states(self, session_id: str):
        session = self.get_session(session_id)
        if session.workspaces:
            return self.project_workspaces.git_states(session.workspaces)
        return [self.workspaces.git_state(session.source_path, session.workspace_path)]

    def diff(self, session_id: str):
        session = self.get_session(session_id)
        if session.workspaces:
            return self.project_workspaces.diff(session.workspaces)
        return self.workspaces.diff(session.workspace_path, session.base_revision)

    def create_branch(self, session_id: str, name: str):
        with self.maintenance_session(
            session_id,
            "Wait for the active turn to finish before changing task branches",
        ) as session:
            if session.workspaces:
                states = []
                for workspace in session.workspaces:
                    if workspace.workspace_kind != WorkspaceKind.git_worktree:
                        continue
                    state = self.workspaces.create_branch(workspace.workspace_path, name)
                    states.append(state.model_copy(update={
                        "folder_id": workspace.folder_id,
                        "folder_name": workspace.folder_name,
                    }))
                if not states:
                    raise WorkspaceError("This project does not contain a Git folder")
                self.emit(
                    session.id,
                    CodingEvent(type=EventType.session, title="Branches created", text=name, phase="completed"),
                )
                return states[0]
            if session.base_revision is None:
                raise WorkspaceError("This task is not using a Git worktree")
            state = self.workspaces.create_branch(session.workspace_path, name)
            self.emit(
                session.id,
                CodingEvent(type=EventType.session, title="Branch created", text=name, phase="completed"),
            )
            return state

    def commit(self, session_id: str, message: str):
        with self.maintenance_session(
            session_id,
            "Wait for the active turn to finish before committing task changes",
        ) as session:
            if session.workspaces:
                states = self.project_workspaces.commit(session.workspaces, message)
                if not states:
                    raise WorkspaceError("This project does not contain a Git folder")
                self.emit(
                    session.id,
                    CodingEvent(
                        type=EventType.session,
                        title="Changes committed",
                        text=message.strip(),
                        phase="completed",
                    ),
                )
                return states[0]
            if session.base_revision is None:
                raise WorkspaceError("This task is not using a Git worktree")
            state = self.workspaces.commit(session.workspace_path, message)
            self.emit(
                session.id,
                CodingEvent(
                    type=EventType.session,
                    title="Changes committed",
                    text=message.strip(),
                    phase="completed",
                ),
            )
            return state

    def apply_to_source(self, session_id: str) -> dict[str, str | None]:
        with self.maintenance_session(
            session_id,
            "Wait for the active turn to finish before applying task changes",
        ) as session:
            if session.workspaces:
                applied = self.project_workspaces.apply(session.id, session.workspaces)
                text = (
                    f"Applied reviewed changes across {len(applied)} project folders."
                    if applied else "No changes to apply."
                )
                self.emit(
                    session.id,
                    CodingEvent(type=EventType.session, title="Handoff complete", text=text, phase="completed"),
                )
                return {"status": "applied" if applied else "no_changes", "snapshot": None}
            if session.workspace_kind == WorkspaceKind.direct_folder:
                raise WorkspaceError("This legacy task uses its source folder directly and has no isolated handoff")
            snapshot = self.workspaces.apply_to_source(
                session.id,
                session.source_path,
                session.workspace_path,
                session.base_revision,
            )
            text = (
                "Applied the task diff to the source working tree. A recovery patch was saved first."
                if snapshot is not None else "No changes to apply."
            )
            self.emit(
                session.id,
                CodingEvent(type=EventType.session, title="Handoff complete", text=text, phase="completed"),
            )
            return {
                "status": "no_changes" if snapshot is None else "applied",
                "snapshot": str(snapshot) if snapshot else None,
            }

    def validate_project(self, session_id: str) -> list[dict]:
        with self.maintenance_session(
            session_id,
            "Wait for the active turn to finish before running project checks",
        ) as session:
            if not session.project_id or not session.workspaces:
                return []
            project = self.projects.get(session.project_id)
            results = self.project_workspaces.run_commands(
                project,
                session.workspaces,
                "validate",
                session.allocated_ports,
            )
            for result in results:
                self.emit(
                    session.id,
                    CodingEvent(
                        type=EventType.command,
                        title=result.label,
                        text=result.output,
                        phase="completed" if result.return_code == 0 else "failed",
                        data={
                            "folderId": result.folder_id,
                            "returnCode": result.return_code,
                            "phase": "validate",
                        },
                    ),
                )
            return [result.__dict__ for result in results]

    def delivery_plan(
        self,
        session_id: str,
        integrations: DeveloperIntegrationService | None = None,
    ):
        session = self.get_session(session_id)
        if not session.project_id or not session.workspaces:
            raise WorkspaceError("This task is not linked to a Code Project")
        project = self.projects.get(session.project_id)
        return self.delivery.plan(session, project, integrations)

    def create_draft_pull_requests(
        self,
        session_id: str,
        request: DraftPullRequestRequest,
        integrations: DeveloperIntegrationService,
    ):
        with self.maintenance_session(
            session_id,
            "Wait for the active turn to finish before publishing draft pull requests",
        ) as session:
            if not session.project_id or not session.workspaces:
                raise WorkspaceError("This task is not linked to a Code Project")
            project = self.projects.get(session.project_id)
            records = self.delivery.create_draft_pull_requests(session, project, request, integrations)
            self.store.update_session(
                session.id,
                lambda current: current.deliveries.extend(records),
            )
            published = sum(item.status == "published" for item in records)
            failed = sum(item.status == "failed" for item in records)
            self.emit(
                session.id,
                CodingEvent(
                    type=EventType.session,
                    title="Draft pull requests prepared",
                    text=(
                        f"Created {published} draft pull request(s)."
                        + (f" {failed} folder(s) need attention." if failed else "")
                    ),
                    phase="failed" if failed and not published else "completed",
                ),
            )
            return records

    def pull_request_action(
        self,
        session_id: str,
        request: PullRequestActionRequest,
        integrations: DeveloperIntegrationService,
    ) -> PullRequestStatus:
        with self.maintenance_session(
            session_id,
            "Wait for the active turn to finish before updating a pull request",
        ) as session:
            if not session.project_id:
                raise WorkspaceError("This task is not linked to a Code Project")
            project = self.projects.get(session.project_id)
            status = integrations.pull_request_action(project, request)
            self.emit(
                session.id,
                CodingEvent(
                    type=EventType.session,
                    title="Pull request updated",
                    text={
                        "ready": "Marked ready for review.",
                        "merge": "Merged on GitHub.",
                        "resolve_thread": "Resolved a GitHub review thread.",
                    }[request.action],
                    phase="completed",
                ),
            )
            return status
