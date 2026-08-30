from __future__ import annotations

from collections.abc import Callable

from cowork.coding.contracts import CodingSession, DeliveryRecord
from cowork.coding.control_service import ControlPlaneService
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import (
    CodeProject,
    DeliveryPlan,
    DeliveryPlanItem,
    DraftPullRequestRequest,
    PullRequestActionRequest,
    PullRequestStatus,
)
from cowork.coding.project_service import CodeProjectService
from cowork.coding.project_tasks import ProjectTaskOperations
from cowork.coding.redaction import redact_text
from cowork.coding.remote_execution import RemoteExecutionCoordinator
from cowork.coding.store import CodingStore


class TaskDeliveryService:
    """Coordinate local and remote delivery without moving OAuth credentials to runtimes."""

    def __init__(
        self,
        store: CodingStore,
        control: ControlPlaneService,
        projects: CodeProjectService,
        local: ProjectTaskOperations,
        remote: RemoteExecutionCoordinator,
        get_session: Callable[[str], CodingSession],
    ) -> None:
        self.store = store
        self.control = control
        self.projects = projects
        self.local = local
        self.remote = remote
        self.get_session = get_session

    def plan(
        self,
        session_id: str,
        integrations: DeveloperIntegrationService | None = None,
    ) -> DeliveryPlan:
        session = self.get_session(session_id)
        if not self.remote.is_remote(session):
            return self.local.delivery_plan(session_id, integrations)
        project = self._project(session)
        plan = DeliveryPlan.model_validate(self.remote.operation(
            session,
            "delivery_plan",
            {"deliveries": [item.model_dump(mode="json") for item in session.deliveries]},
        ))
        plan.integrations = integrations.statuses(project) if integrations else []
        if integrations:
            self._load_pull_request_statuses(project, plan, integrations)
        return plan

    def create_drafts(
        self,
        session_id: str,
        request: DraftPullRequestRequest,
        integrations: DeveloperIntegrationService,
    ) -> list[DeliveryRecord]:
        session = self.get_session(session_id)
        if not self.remote.is_remote(session):
            records = self.local.create_draft_pull_requests(session_id, request, integrations)
            self._sync(session_id)
            return records
        self.remote.require_idle(session, "publishing draft pull requests")
        if not request.confirmed:
            raise RuntimeError("Confirm draft pull-request creation before publishing branches")
        project = self._project(session)
        requested = {item.folder_id: item for item in request.drafts}
        records: list[DeliveryRecord] = []
        for item in self.plan(session_id).items:
            if item.status in {"no_changes", "published"} or (requested and item.folder_id not in requested):
                continue
            if item.status != "ready":
                records.append(self._failed(item, item.detail))
                continue
            draft = requested.get(item.folder_id)
            try:
                self.remote.operation(
                    session,
                    "push_branch",
                    {"folder_id": item.folder_id},
                    timeout=180,
                )
                external_url = integrations.create_draft_pull_request(
                    project,
                    repository_url=item.remote_url or "",
                    title=draft.title if draft else request.title,
                    body=draft.body if draft else request.body,
                    head=item.task_branch or "",
                    base_branch=item.base_branch or "",
                    connection_name=request.connection_name,
                )
                records.append(self._published(item, external_url, request.connection_name))
            except Exception as exc:  # noqa: BLE001 - preserve partial multi-repository delivery.
                records.append(self._failed(item, redact_text(str(exc))[:2_000]))
        self.store.update_session(session.id, lambda current: current.deliveries.extend(records))
        self._sync(session.id)
        return records

    def record(self, session_id: str, delivery: DeliveryRecord) -> DeliveryRecord:
        self.store.update_session(session_id, lambda current: current.deliveries.append(delivery))
        self._sync(session_id)
        return delivery

    def pull_request_action(
        self,
        session_id: str,
        request: PullRequestActionRequest,
        integrations: DeveloperIntegrationService,
    ) -> PullRequestStatus:
        return self.local.pull_request_action(session_id, request, integrations)

    def _sync(self, session_id: str) -> None:
        session = self.store.load_session(session_id)
        if not session.task_id:
            return
        task = self.control.store.get_task(session.task_id)
        task.deliveries = list(session.deliveries)
        self.control.store.save_task(task)

    def _project(self, session: CodingSession) -> CodeProject:
        if not session.project_id:
            raise RuntimeError("This task is not linked to a Code Project")
        return self.projects.get(session.project_id)

    @staticmethod
    def _load_pull_request_statuses(
        project: CodeProject,
        plan: DeliveryPlan,
        integrations: DeveloperIntegrationService,
    ) -> None:
        for item in plan.items:
            if item.status != "published" or not item.external_url:
                continue
            try:
                item.pull_request_status = integrations.pull_request_status(
                    project,
                    item.external_url,
                    item.connection_name,
                )
            except Exception as exc:  # noqa: BLE001 - one PR status must not hide other repositories.
                item.status_error = redact_text(str(exc))[:2_000]

    @staticmethod
    def _published(item: DeliveryPlanItem, external_url: str, connection_name: str | None) -> DeliveryRecord:
        return DeliveryRecord(
            provider="github",
            action="draft_pull_request",
            target_url=item.remote_url or "",
            status="published",
            external_url=external_url,
            detail="Draft pull request created",
            folder_id=item.folder_id,
            folder_name=item.folder_name,
            base_branch=item.base_branch,
            task_branch=item.task_branch,
            connection_name=connection_name,
        )

    @staticmethod
    def _failed(item: DeliveryPlanItem, detail: str) -> DeliveryRecord:
        return DeliveryRecord(
            provider="github",
            action="draft_pull_request",
            target_url=item.remote_url or item.workspace_path,
            status="failed",
            detail=detail,
            folder_id=item.folder_id,
            folder_name=item.folder_name,
            base_branch=item.base_branch,
            task_branch=item.task_branch,
        )
