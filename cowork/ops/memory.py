"""Learned memory: ``python -m cowork.ops.memory print|clear``.

Where memory is kept per person (COWORK_PERSONAL_RESOURCES lists memory), both
actions act on one person's memory and need ``--organization-id`` and
``--user-id``. ``print`` shows what the person's agent learned, as their
Memories page lists it (MemoryService.list_memory_sync): their global tier,
then each project tier of theirs, with the folder each one is in. ``clear``
removes all of it with cowork.services.memory.reset_person_memory, the one
reset for a person's memory.

Where memory is shared, both act on the deployment's one store and take
neither argument, so a clear meant for one person can never clear everyone's.
``print`` shows the shared global tier, then each project tier, titled by its
project's id alone. ``clear`` removes them with
cowork.services.memory.clear_deployment_memory, which touches no person's tier
and nothing of the managed folder. A link where the store's ``projects/``
folder should be stops either one with an error, before anything is read or
removed.

``print`` ends with "Nothing learned." when every tier is empty, and ``clear``
can't be undone. While memory is off (MEMORY_ENABLED false), ``print`` shows
nothing learned and says the stored memory is kept, and ``clear`` refuses, so
the kept memory stays as it is. Both check their arguments first.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import unit_session
from cowork.harnesses.memory.policy import MEMORY_OFF_KEPT, resolve_memory_policy
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import (
    GlobalMemoryStore,
    deployment_memory_root,
    nested_project_memory_store,
    project_memory_store,
)
from cowork.models.project import Project
from cowork.ops import (
    OperatorCommandError,
    Person,
    add_person_arguments,
    person_from_arguments,
    personal_scope,
    require_trusted_identity,
    run_operator_command,
)
from cowork.schemas.memory import MemoryResponse, MemoryScope
from cowork.services.memory import (
    MemoryService,
    PersonMemoryResetError,
    clear_deployment_memory,
    reset_person_memory,
)

MEMORY_OFF_PRINT = f"{MEMORY_OFF_KEPT} No turn reads or learns memory, so nothing learned is shown."
MEMORY_OFF_CLEAR = f"{MEMORY_OFF_KEPT} Clearing it is refused."


class MemoryTarget(BaseModel):
    """What one memory command acts on, checked against the deployment."""

    model_config = ConfigDict(frozen=True)

    # None for the deployment's one shared store.
    person: Person | None
    # MEMORY_ENABLED is false (cowork.harnesses.memory.policy).
    off: bool


class MemoryTier(BaseModel):
    """One memory tier and what each of its slots holds."""

    model_config = ConfigDict(frozen=True)

    title: str
    path: Path
    slots: dict[MemorySlot, str]

    @property
    def learned(self) -> bool:
        return any(content.strip() for content in self.slots.values())


class MemoryPrintout(BaseModel):
    model_config = ConfigDict(frozen=True)

    # None for the deployment's one shared store.
    person: Person | None
    tiers: list[MemoryTier]

    def render(self) -> str:
        if self.person is None:
            lines = ["Shared memory of this deployment"]
        else:
            lines = [f"Memory of user {self.person.user_id}, space {self.person.partition_id}"]
        learned = [tier for tier in self.tiers if tier.learned]
        for tier in learned:
            lines.append(f"{tier.title}, in {tier.path}")
            for slot, content in tier.slots.items():
                if content.strip():
                    lines.append(f"  {slot.value}:")
                    lines.extend(f"    {line}" for line in content.strip().splitlines())
        if not learned:
            lines.append("Nothing learned.")
        return "\n".join(lines)


class ListedSlots(NamedTuple):
    """A Memories listing's slots, by tier."""

    global_slots: dict[MemorySlot, str]
    project_slots: dict[UUID, dict[MemorySlot, str]]


def _listed_slots(*, items: list[MemoryResponse]) -> ListedSlots:
    global_slots = {
        item.category: item.content for item in items if item.scope == MemoryScope.global_
    }
    project_slots: dict[UUID, dict[MemorySlot, str]] = {}
    for item in items:
        if item.scope == MemoryScope.project and item.project_id is not None:
            project_slots.setdefault(item.project_id, {})[item.category] = item.content
    return ListedSlots(global_slots=global_slots, project_slots=project_slots)


def _memory_target(*, arguments: argparse.Namespace, action: str) -> MemoryTarget:
    """Check the person arguments against how the deployment keeps memory.

    Memory kept per person needs both; memory shared takes neither, so a clear
    meant for one person can never become a clear of the shared store.
    """
    require_trusted_identity()
    # memory_enabled is operator-pinned, so every scope on this deployment
    # resolves the one value its turns read.
    policy = resolve_memory_policy()
    named = arguments.organization_id is not None or arguments.user_id is not None
    if policy.shared:
        if named:
            raise OperatorCommandError(
                "This deployment keeps memory shared (COWORK_PERSONAL_RESOURCES leaves "
                f"out memory), so {action} acts on its one store and takes no "
                "--organization-id or --user-id."
            )
        return MemoryTarget(person=None, off=policy.off)
    if arguments.organization_id is None or arguments.user_id is None:
        raise OperatorCommandError(
            "This deployment keeps memory per person (COWORK_PERSONAL_RESOURCES lists "
            f"memory), so {action} needs --organization-id and --user-id."
        )
    return MemoryTarget(person=person_from_arguments(arguments), off=policy.off)


def read_memory(person: Person | None) -> MemoryPrintout:
    """What the person's agent learned, or, with no person, what the
    deployment's shared store holds, as the Memories listing reads it."""
    if person is None:
        return _read_shared_memory()
    scope = personal_scope(person)
    with unit_session(scope=scope) as session:
        items = MemoryService(session).list_memory_sync()
        projects = {
            project.id: project
            for project in session.exec(session.select(Project)).all()
        }
        project_paths = {
            project_id: project_memory_store(
                scope=scope, project_path=Path(project.path), project_id=project_id
            ).root
            for project_id, project in projects.items()
        }
        project_names = {project_id: project.name for project_id, project in projects.items()}

    listed = _listed_slots(items=items)
    tiers = [
        MemoryTier(
            title="Global",
            path=GlobalMemoryStore(scope=scope).root,
            slots=listed.global_slots,
        ),
        *(
            MemoryTier(
                title=f"Project {project_names[project_id]} ({project_id})",
                path=project_paths[project_id],
                slots=slots,
            )
            for project_id, slots in sorted(
                listed.project_slots.items(), key=lambda entry: project_names[entry[0]]
            )
        ),
    ]
    return MemoryPrintout(person=person, tiers=tiers)


def _read_shared_memory() -> MemoryPrintout:
    """The deployment's shared store, each project tier titled by its id: the
    tier may belong to a project of someone else's, whose name stays theirs."""
    root = deployment_memory_root()
    try:
        with unit_session(scope=LOCAL_SCOPE) as session:
            items = MemoryService(session).list_memory_sync()
    except OSError as error:
        raise OperatorCommandError(
            f"Could not read the deployment's shared memory in {root}: {error}. "
            "A link where its projects/ folder should be is refused, never followed."
        ) from None
    listed = _listed_slots(items=items)
    tiers = [
        MemoryTier(title="Global", path=root, slots=listed.global_slots),
        *(
            MemoryTier(
                title=f"Project {project_id}",
                path=nested_project_memory_store(scope=LOCAL_SCOPE, project_id=project_id).root,
                slots=slots,
            )
            for project_id, slots in sorted(
                listed.project_slots.items(), key=lambda entry: str(entry[0])
            )
        ),
    ]
    return MemoryPrintout(person=None, tiers=tiers)


def print_memory(target: MemoryTarget) -> str:
    if target.off:
        return MEMORY_OFF_PRINT
    return read_memory(target.person).render()


def clear_memory(target: MemoryTarget) -> str:
    if target.off:
        raise OperatorCommandError(MEMORY_OFF_CLEAR)
    if target.person is None:
        try:
            cleared = clear_deployment_memory()
        except OSError as error:
            raise OperatorCommandError(
                f"Stopped clearing the deployment's shared memory in {deployment_memory_root()}: "
                f"{error}. A link where its projects/ folder should be is refused, never "
                "followed, and stops the clear before anything is removed."
            ) from None
        if not cleared.removed:
            return f"The deployment has no shared memory to clear, in {cleared.root}."
        return f"Cleared the deployment's shared memory, in {cleared.root}."
    person = target.person
    try:
        reset = reset_person_memory(personal_scope(person))
    except PersonMemoryResetError as error:
        raise OperatorCommandError(str(error)) from None
    if not reset.removed:
        return f"User {person.user_id} has no memory to clear, in {reset.root}."
    return f"Cleared the memory of user {person.user_id}, in {reset.root}."


_ACTIONS: dict[str, Callable[[MemoryTarget], str]] = {
    "print": print_memory,
    "clear": clear_memory,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cowork.ops.memory",
        description=(
            "Print or clear learned memory: one person's where memory is kept per "
            "person, the deployment's one store where it is shared."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)
    add_person_arguments(
        actions.add_parser("print", help="print what the agent learned"),
        required=False,
    )
    add_person_arguments(
        actions.add_parser("clear", help="remove everything the agent learned"),
        required=False,
    )
    arguments = parser.parse_args(argv)
    return run_operator_command(
        lambda: _ACTIONS[arguments.action](_memory_target(arguments=arguments, action=arguments.action))
    )


if __name__ == "__main__":
    sys.exit(main())
