from __future__ import annotations

from typing import TYPE_CHECKING

from cowork.coding.contracts import DeliveryRecord
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import (
    DeliveryPlan,
    DraftPullRequestRequest,
    PublishRequest,
    PullRequestActionRequest,
    PullRequestStatus,
    SourceActionRequest,
)

if TYPE_CHECKING:
    from cowork.coding.task_delivery import TaskDeliveryService


class CodingDeliveryOperations:
    """Stable coding-service façade for task delivery and source lifecycle work."""

    task_delivery: TaskDeliveryService

    def delivery_plan(
        self,
        session_id: str,
        integrations: DeveloperIntegrationService | None = None,
    ) -> DeliveryPlan:
        return self.task_delivery.plan(session_id, integrations)

    def create_draft_pull_requests(
        self,
        session_id: str,
        request: DraftPullRequestRequest,
        integrations: DeveloperIntegrationService,
    ) -> list[DeliveryRecord]:
        return self.task_delivery.create_drafts(session_id, request, integrations)

    def record_delivery(self, session_id: str, delivery: DeliveryRecord) -> DeliveryRecord:
        return self.task_delivery.record(session_id, delivery)

    def publish_task_update(
        self,
        session_id: str,
        request: PublishRequest,
        integrations: DeveloperIntegrationService,
    ) -> DeliveryRecord:
        return self.task_delivery.publish_update(session_id, request, integrations)

    def complete_task_source(
        self,
        session_id: str,
        request: SourceActionRequest,
        integrations: DeveloperIntegrationService,
    ) -> DeliveryRecord:
        return self.task_delivery.complete_source(session_id, request, integrations)

    def pull_request_action(
        self,
        session_id: str,
        request: PullRequestActionRequest,
        integrations: DeveloperIntegrationService,
    ) -> PullRequestStatus:
        return self.task_delivery.pull_request_action(session_id, request, integrations)
