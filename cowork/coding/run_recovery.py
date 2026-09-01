from __future__ import annotations

from cowork.coding.control_models import (
    Computer,
    ComputerStatus,
    RecoveryOption,
    RecoveryPlan,
    RunStatus,
    TaskRun,
    WorkspaceStatus,
)
from cowork.coding.control_store import ControlPlaneStore
from cowork.coding.project_models import RepositoryResource
from cowork.coding.run_state import transition_run


class NoEligibleComputer(RuntimeError):
    pass


def build_recovery_plan(
    store: ControlPlaneStore,
    run: TaskRun,
    eligible_computers: list[Computer],
) -> RecoveryPlan:
    """Describe recovery honestly: restore in place or recreate from Git."""

    task = store.get_task(run.task_id)
    if task.execution_project is None:
        try:
            original = store.get_computer(run.computer_id)
        except KeyError:
            return RecoveryPlan(run_id=run.id)
        options = [_option(original, preserves_changes=True)] if original.status == ComputerStatus.online else []
        return RecoveryPlan(run_id=run.id, options=options)

    options = [
        _option(computer, preserves_changes=computer.id == run.computer_id)
        for computer in eligible_computers
    ]
    if options and not any(option.recommended for option in options):
        options[0].recommended = True
    return RecoveryPlan(run_id=run.id, options=options)


def recover_run(
    store: ControlPlaneStore,
    run_id: str,
    target: Computer,
    *,
    allow_recreate: bool,
) -> TaskRun:
    """Fence the old execution and prepare an explicit restore or recreation."""

    if target.status != ComputerStatus.online:
        raise NoEligibleComputer("Choose an online computer to resume this task")

    def fence(run: TaskRun) -> None:
        moving = target.id != run.computer_id
        if moving:
            task = store.get_task(run.task_id)
            project = task.execution_project
            if project is None or any(
                not isinstance(resource, RepositoryResource) or not resource.source_url
                for resource in project.resources
            ):
                raise NoEligibleComputer(
                    "This task includes resources that only exist on its original computer"
                )
            if not allow_recreate:
                raise NoEligibleComputer(
                    "Moving this task starts a fresh working copy and requires confirmation"
                )
        run.computer_id = target.id
        run.epoch += 1
        run.lease_id = None
        run.lease_expires_at = None
        run.last_event_seq = 0
        run.last_event_id = None
        run.checkpoint = {}
        run.workspace_resume_mode = "recreate" if moving else "restore"
        run.recovery_count += 1
        transition_run(run, RunStatus.recovering)

    saved = store.update_run(run_id, fence)

    if saved.workspace_resume_mode == "recreate":
        for workspace in store.list_workspaces(saved.id):
            workspace.computer_id = target.id
            workspace.status = WorkspaceStatus.pending
            workspace.path = ""
            workspace.workspace_kind = None
            workspace.base_revision = None
            workspace.task_branch = None
            workspace.detail = "Recreating an isolated working copy from the task snapshot"
            store.save_workspace(workspace)
    return saved


def _option(computer: Computer, *, preserves_changes: bool) -> RecoveryOption:
    return RecoveryOption(
        computer=computer,
        mode="restore" if preserves_changes else "recreate",
        preserves_workspace_changes=preserves_changes,
        recommended=preserves_changes,
        detail=(
            "Resume the saved working copy and its current changes."
            if preserves_changes
            else "Create a fresh isolated working copy from the task's saved repository definitions. "
            "Changes that were not pushed are not carried over."
        ),
    )
