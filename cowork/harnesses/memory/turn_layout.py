"""Where one in-process turn's learned memory lives.

anton's Cortex reads and writes two learned tiers, global and project. Which
folders those are depends on who asked the turn:

- The desktop, and a deployment that does not trust identity headers: the base
  global tier and the project's ``.anton/memory``, as always.
- A person on a deployment that trusts identity headers: their own tiers where
  memory is per person, and the deployment's shared global tier with its tier
  for the turn's project where memory is shared (GlobalMemoryStore and
  project_memory_store resolve them from the scope).
- A turn with no person on such a deployment, such as an operator call under
  audit, and every turn there while memory is off
  (cowork.harnesses.memory.policy): two empty folders the turn removes when it
  ends, and no learning at all, so it neither reads nor writes anyone's
  memory. The operator's managed rules still reach it, through the Cortex the
  harness builds.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import TenantScope
from cowork.harnesses.memory.store import GlobalMemoryStore, project_memory_store


class TurnMemoryLayout(BaseModel):
    """The two learned tiers one turn hands anton, and whether it may learn."""

    model_config = ConfigDict(frozen=True)

    global_dir: Path
    project_dir: Path
    # False for a turn with no person, and while memory is off: its tiers are
    # empty scratch folders, and the turn runs with memory off and no
    # background passes.
    learns: bool


def person_turn_memory(
    *, scope: TenantScope | None, project_path: Path, project_id: UUID | str
) -> TurnMemoryLayout | None:
    """The tiers of the person who asked the turn, or None when there is none.

    A deployment that trusts identity headers has a person exactly when the
    turn's scope names a user. Without one (the operator view, or no scope
    bound at all) the turn must not touch the base tiers, which belong to no
    person, nor any partition's; the caller gives it empty_turn_memory instead.
    Everywhere else the scope resolves the tiers as the memory API does.
    """
    if get_app_settings().trust_identity_headers and (scope is None or not scope.user_id):
        return None
    return TurnMemoryLayout(
        global_dir=GlobalMemoryStore(scope=scope).root,
        project_dir=project_memory_store(
            scope=scope, project_path=project_path, project_id=project_id
        ).root,
        learns=True,
    )


def empty_turn_memory(scratch_dir: Path) -> TurnMemoryLayout:
    """Two empty tiers under ``scratch_dir``, for a turn with no person or
    with memory off.

    The caller owns ``scratch_dir`` and removes it when the turn ends, so
    nothing anton writes here outlives the turn.
    """
    layout = TurnMemoryLayout(
        global_dir=scratch_dir / "global",
        project_dir=scratch_dir / "project",
        learns=False,
    )
    layout.global_dir.mkdir(parents=True, exist_ok=True)
    layout.project_dir.mkdir(parents=True, exist_ok=True)
    return layout
