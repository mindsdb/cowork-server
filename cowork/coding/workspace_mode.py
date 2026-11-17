"""Choose the workspace mode a new task runs in.

The server owns this choice so a desktop and a web runtime computer resolve
the same request the same way, and a client can leave the mode unset.
"""

from __future__ import annotations

import sys

from cowork.coding.contracts import WorkspaceMode, WorkspaceModeOptions

# An in-place task relies on Codex's sandbox to keep writes inside the user's
# folder: Seatbelt on macOS and Landlock on Linux. Its Windows sandbox has not
# been verified for a task in the user's own folder, so Windows keeps
# worktrees until it is.
_IN_PLACE_PLATFORMS = frozenset({"darwin", "linux"})
# Like Claude Code and the Codex app, a task works in the user's folder unless
# it asks for a worktree; isolation is opt-in.


def supported_workspace_modes(
    *,
    runs_locally: bool,
    org_mode: bool,
    platform: str | None = None,
) -> tuple[WorkspaceMode, ...]:
    """Modes the computer running a task can prepare for it.

    A web runtime computer has no user checkout to work in, and on a local
    computer org mode refuses local workspaces altogether.
    """
    if runs_locally and not org_mode and (platform or sys.platform) in _IN_PLACE_PLATFORMS:
        return (WorkspaceMode.worktree, WorkspaceMode.in_place)
    return (WorkspaceMode.worktree,)


def resolve_workspace_mode(
    requested: WorkspaceMode | None,
    project_default: WorkspaceMode | None,
    *,
    runs_locally: bool,
    org_mode: bool,
    platform: str | None = None,
) -> WorkspaceMode:
    """Return the mode for a new task.

    An explicit request must be honoured or refused; a project default is a
    preference, so a computer that cannot honour it falls back to its own
    default.
    """
    supported = supported_workspace_modes(runs_locally=runs_locally, org_mode=org_mode, platform=platform)
    if requested is not None:
        if requested not in supported:
            raise ValueError(_unsupported_message(requested, runs_locally=runs_locally, org_mode=org_mode))
        return requested
    if project_default is not None and project_default in supported:
        return project_default
    return WorkspaceMode.in_place if WorkspaceMode.in_place in supported else WorkspaceMode.worktree


def workspace_mode_options(
    project_default: WorkspaceMode | None,
    *,
    runs_locally: bool,
    org_mode: bool,
    platform: str | None = None,
) -> WorkspaceModeOptions:
    """What a client may offer for a new task, resolved the way creation resolves it."""
    context = {"runs_locally": runs_locally, "org_mode": org_mode, "platform": platform}
    supported = supported_workspace_modes(**context)
    return WorkspaceModeOptions(
        supported=list(supported),
        default=resolve_workspace_mode(None, project_default, **context),
        in_place_unavailable_reason=(
            None
            if WorkspaceMode.in_place in supported
            else _unsupported_message(
                WorkspaceMode.in_place, runs_locally=runs_locally, org_mode=org_mode,
            )
        ),
    )


def without_in_place(options: WorkspaceModeOptions, reason: str) -> WorkspaceModeOptions:
    """``options`` for a folder that cannot be worked in directly."""
    if WorkspaceMode.in_place not in options.supported:
        return options
    return _without(options, WorkspaceMode.in_place, in_place_unavailable_reason=reason)


def without_worktree(
    options: WorkspaceModeOptions, reason: str, *, can_initialize_git: bool = False,
) -> WorkspaceModeOptions:
    """``options`` for a folder a worktree cannot be made from."""
    if WorkspaceMode.worktree not in options.supported:
        return options.model_copy(update={"can_initialize_git": can_initialize_git})
    return _without(
        options, WorkspaceMode.worktree, worktree_unavailable_reason=reason, can_initialize_git=can_initialize_git,
    )


def _without(options: WorkspaceModeOptions, mode: WorkspaceMode, **update: object) -> WorkspaceModeOptions:
    supported = [item for item in options.supported if item != mode]
    default = options.default if options.default in supported else (supported[0] if supported else None)
    return options.model_copy(update={"supported": supported, "default": default, **update})


def _unsupported_message(mode: WorkspaceMode, *, runs_locally: bool, org_mode: bool) -> str:
    if mode == WorkspaceMode.in_place:
        if org_mode or not runs_locally:
            return "Tasks on another computer run in an isolated worktree; choose Worktree"
        return "Running a task in the original folder is not available on this platform yet; choose Worktree"
    return f"This computer cannot prepare a {mode.value} workspace"
