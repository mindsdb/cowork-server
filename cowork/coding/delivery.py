from __future__ import annotations

import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from cowork.coding.contracts import CodingSession, TaskWorkspace, WorkspaceKind
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import (
    CodeProject,
    DeliveryPlan,
    DeliveryPlanItem,
    DeliveryRecord,
    DraftPullRequestRequest,
    DraftPullRequestSpec,
)
from cowork.coding.workspace import GitRunner, WorkspaceError

_REVISION = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")


class ProjectDeliveryService:
    """Plan and perform explicit, ordered multi-repository delivery."""

    def __init__(self, git: GitRunner | None = None) -> None:
        self.git = git or GitRunner()

    def plan(
        self,
        session: CodingSession,
        project: CodeProject | None = None,
        integrations: DeveloperIntegrationService | None = None,
    ) -> DeliveryPlan:
        return self.plan_workspaces(
            session.workspaces,
            session.deliveries,
            project,
            integrations,
        )

    def plan_workspaces(
        self,
        workspaces: Sequence[TaskWorkspace],
        deliveries: list[DeliveryRecord],
        project: CodeProject | None = None,
        integrations: DeveloperIntegrationService | None = None,
    ) -> DeliveryPlan:
        """Build review/delivery state where the workspaces actually live."""

        items: list[DeliveryPlanItem] = []
        published = {
            str(item.folder_id): item
            for item in deliveries
            if item.action == "draft_pull_request" and item.status == "published" and item.folder_id
        }
        for workspace in workspaces:
            if workspace.workspace_kind != WorkspaceKind.git_worktree:
                continue
            state = self._state(workspace.workspace_path)
            remote = self.git.run(
                self._path(workspace.workspace_path),
                "remote",
                "get-url",
                "origin",
                check=False,
            ).stdout.strip() or None
            branch = state[0] or workspace.task_branch
            base_branch = workspace.base_branch or self._source_branch(workspace.source_path)
            commits = self._commit_count(workspace.workspace_path, workspace.base_revision)
            existing = published.get(workspace.folder_id)
            pull_request_status = None
            status_error = None
            if existing:
                status, detail = "published", "Draft pull request created"
                if project and integrations and existing.external_url:
                    try:
                        pull_request_status = integrations.pull_request_status(
                            project,
                            existing.external_url,
                            existing.connection_name,
                        )
                    except WorkspaceError as exc:
                        status_error = str(exc)[:2_000]
            elif state[1]:
                status, detail = "needs_commit", "Commit this folder's changes before creating its draft pull request"
            elif commits == 0:
                status, detail = "no_changes", "No committed task changes"
            elif not remote:
                status, detail = "unavailable", "Add an origin remote before creating a draft pull request"
            elif not branch or not base_branch:
                status, detail = "unavailable", "The task or base branch could not be determined"
            else:
                status, detail = "ready", ""
            items.append(
                DeliveryPlanItem(
                    folder_id=workspace.folder_id,
                    folder_name=workspace.folder_name,
                    workspace_path=workspace.workspace_path,
                    remote_url=remote,
                    base_branch=base_branch,
                    task_branch=branch,
                    head_revision=state[2],
                    status=status,
                    detail=detail,
                    external_url=existing.external_url if existing else None,
                    connection_name=existing.connection_name if existing else None,
                    pull_request_status=pull_request_status,
                    status_error=status_error,
                )
            )
        integration_statuses = integrations.statuses(project) if project and integrations else []
        return DeliveryPlan(items=items, integrations=integration_statuses)

    def create_draft_pull_requests(
        self,
        session: CodingSession,
        project: CodeProject,
        request: DraftPullRequestRequest,
        integrations: DeveloperIntegrationService,
    ) -> list[DeliveryRecord]:
        if not request.confirmed:
            raise WorkspaceError("Confirm draft pull-request creation before publishing branches")
        records: list[DeliveryRecord] = []
        requested = {item.folder_id: item for item in request.drafts}
        for item in self.plan(session).items:
            if item.status in {"no_changes", "published"}:
                continue
            if requested and item.folder_id not in requested:
                continue
            if item.status != "ready":
                records.append(self._failed(item, item.detail))
                continue
            draft = requested.get(item.folder_id)
            changed = confirmation_mismatch(item, draft)
            if changed:
                records.append(self._failed(item, changed))
                continue
            try:
                push = integrations.git_push_credentials(
                    project,
                    item.remote_url or "",
                    request.connection_name,
                )
                self.push_commit(
                    item.workspace_path,
                    item.head_revision or "",
                    item.task_branch or "",
                    push.remote_url,
                    push.environment,
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
                records.append(
                    DeliveryRecord(
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
                        connection_name=request.connection_name,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - preserve partial delivery results across repositories.
                records.append(self._failed(item, str(exc)[:2_000]))
        return records

    def push_commit(
        self,
        workspace_path: str,
        revision: str,
        branch: str,
        remote_url: str,
        environment: dict[str, str],
    ) -> None:
        """Push one exact commit to one branch from a repository Cowork owns.

        The push carries the GitHub credential, so it runs outside the task's
        repository: hook managers keep their scripts in the task's files, and
        repository settings (URL rewrites, proxies, credential helpers) could
        send the credential elsewhere. A fresh bare repository borrows the
        task's objects through ``alternates``, so nothing is copied.
        """
        if not _REVISION.fullmatch(revision):
            raise WorkspaceError("The commit to publish could not be determined")
        root = self._path(workspace_path)
        if self.git.run(root, "check-ref-format", "--branch", branch, check=False).returncode != 0:
            raise WorkspaceError("The task branch name is not a valid Git branch")
        common = self.git.run(root, "rev-parse", "--git-common-dir").stdout.strip()
        objects = (root / common).resolve() / "objects"
        with tempfile.TemporaryDirectory(prefix="cowork-publish-", ignore_cleanup_errors=True) as scratch:
            clean = Path(scratch) / "repository.git"
            no_hooks = Path(scratch) / "no-hooks"
            clean.mkdir()
            no_hooks.mkdir()
            # An empty template keeps hooks from init.templateDir out too.
            self.git.run(clean, "init", "--quiet", "--bare", "--template=", ".")
            (clean / "objects" / "info").mkdir(parents=True, exist_ok=True)
            (clean / "objects" / "info" / "alternates").write_text(f"{objects}\n", encoding="utf-8")
            if self.git.run(clean, "cat-file", "-e", f"{revision}^{{commit}}", check=False).returncode != 0:
                raise WorkspaceError("The commit to publish is no longer in the task's repository")
            # Without the pre-push hook Git LFS never uploads file contents,
            # and the branch would land with pointer files in their place.
            if self.git.run(
                clean, "grep", "--quiet", "-e", "filter=lfs", revision, "--", ":(glob)**/.gitattributes",
                check=False,
            ).returncode == 0:
                raise WorkspaceError(
                    "This repository uses Git LFS, which Code mode can't publish yet. "
                    "Push the task branch yourself, then create the pull request."
                )
            self.git.run(
                clean,
                "-c",
                f"core.hooksPath={no_hooks}",
                "push",
                "--no-verify",
                remote_url,
                f"{revision}:refs/heads/{branch}",
                environment=environment,
            )

    def _state(self, workspace_path: str) -> tuple[str | None, bool, str | None]:
        root = self._path(workspace_path)
        branch = self.git.run(root, "branch", "--show-current", check=False).stdout.strip() or None
        dirty = bool(self.git.run(root, "status", "--porcelain=v1", "--untracked-files=all").stdout.strip())
        head = self.git.run(root, "rev-parse", "--verify", "--quiet", "HEAD", check=False).stdout.strip()
        return branch, dirty, head if _REVISION.fullmatch(head) else None

    def _source_branch(self, source_path: str) -> str | None:
        branch = self.git.run(
            self._path(source_path),
            "symbolic-ref",
            "--short",
            "-q",
            "HEAD",
            check=False,
        ).stdout.strip()
        return branch or None

    def _commit_count(self, workspace_path: str, base_revision: str | None) -> int:
        if not base_revision:
            return 0
        result = self.git.run(
            self._path(workspace_path),
            "rev-list",
            "--count",
            f"{base_revision}..HEAD",
            check=False,
        )
        return int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip().isdigit() else 0

    @staticmethod
    def _path(value: str) -> Path:
        return Path(value)

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


def confirmation_mismatch(item: DeliveryPlanItem, confirmed: DraftPullRequestSpec | None) -> str | None:
    """Why a folder no longer matches what the person confirmed, if it doesn't."""
    if confirmed is None:
        return None
    pinned = (
        (confirmed.head_revision, item.head_revision),
        (confirmed.task_branch, item.task_branch),
        (confirmed.remote_url, item.remote_url),
    )
    if any(expected is not None and expected != current for expected, current in pinned):
        return "This folder changed after you confirmed its draft pull request. Review it and confirm again."
    return None
