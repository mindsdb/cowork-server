"""Choose the workspace mode a new task runs in.

The server owns this choice so a desktop and a web runtime computer resolve
the same request the same way, and a client can leave the mode unset.
"""

from __future__ import annotations

from cowork.coding.contracts import WorkspaceMode

# In-place tasks need the user's own checkout, which only this computer has.
# The mode is part of the contract already, but preparation cannot run a task
# in place yet, so no computer offers it.
_LOCAL_MODES: tuple[WorkspaceMode, ...] = (WorkspaceMode.worktree,)
_RUNTIME_MODES: tuple[WorkspaceMode, ...] = (WorkspaceMode.worktree,)
_LOCAL_DEFAULT = WorkspaceMode.worktree
_RUNTIME_DEFAULT = WorkspaceMode.worktree


def supported_workspace_modes(*, runs_locally: bool, org_mode: bool) -> tuple[WorkspaceMode, ...]:
    """Modes the computer running a task can prepare.

    A web runtime computer has no user checkout to work in, and on a local
    computer org mode refuses local workspaces altogether.
    """
    return _LOCAL_MODES if runs_locally and not org_mode else _RUNTIME_MODES


def resolve_workspace_mode(
    requested: WorkspaceMode | None,
    project_default: WorkspaceMode | None,
    *,
    runs_locally: bool,
    org_mode: bool,
) -> WorkspaceMode:
    """Return the mode for a new task.

    An explicit request must be honoured or refused; a project default is a
    preference, so a computer that cannot honour it falls back to its own
    default.
    """
    supported = supported_workspace_modes(runs_locally=runs_locally, org_mode=org_mode)
    if requested is not None:
        if requested not in supported:
            raise ValueError(_unsupported_message(requested, runs_locally=runs_locally, org_mode=org_mode))
        return requested
    if project_default is not None and project_default in supported:
        return project_default
    return _LOCAL_DEFAULT if runs_locally and not org_mode else _RUNTIME_DEFAULT


def _unsupported_message(mode: WorkspaceMode, *, runs_locally: bool, org_mode: bool) -> str:
    if mode == WorkspaceMode.in_place:
        if org_mode or not runs_locally:
            return "Tasks on another computer run in an isolated worktree; choose Worktree"
        return "Running a task in the original folder is not available yet; choose Worktree"
    return f"This computer cannot prepare a {mode.value} workspace"
