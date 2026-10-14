"""One person's memory: ``python -m cowork.ops.memory print|clear``.

``print`` shows what the person's agent learned, as their Memories page lists
it (MemoryService.list_memory_sync): their global tier, then each project tier
of theirs, with the folder each one is in. It ends with "Nothing learned."
when every tier is empty.

``clear`` removes all of it with cowork.services.memory.reset_person_memory,
the one reset for a person's memory. It can't be undone.

Both need memory kept per person (COWORK_PERSONAL_RESOURCES lists memory).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import TenantScope
from cowork.db.units import unit_session
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import GlobalMemoryStore, project_memory_store
from cowork.models.project import Project
from cowork.ops import (
    OperatorCommandError,
    Person,
    add_person_arguments,
    person_from_arguments,
    personal_scope,
    run_operator_command,
)
from cowork.schemas.memory import MemoryScope
from cowork.services.memory import MemoryService, PersonMemoryResetError, reset_person_memory


class MemoryTier(BaseModel):
    """One tier of a person's memory and what each of its slots holds."""

    model_config = ConfigDict(frozen=True)

    title: str
    path: Path
    slots: dict[MemorySlot, str]

    @property
    def learned(self) -> bool:
        return any(content.strip() for content in self.slots.values())


class MemoryPrintout(BaseModel):
    model_config = ConfigDict(frozen=True)

    person: Person
    tiers: list[MemoryTier]

    def render(self) -> str:
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


def _require_personal_memory(scope: TenantScope) -> None:
    if not scope.partitioned_for(PersonalResource.MEMORY):
        raise OperatorCommandError(
            "This deployment keeps memory shared (COWORK_PERSONAL_RESOURCES leaves "
            "out memory), so no person has memory of their own."
        )


def read_memory(person: Person) -> MemoryPrintout:
    scope = personal_scope(person)
    _require_personal_memory(scope)
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

    global_slots = {
        item.category: item.content for item in items if item.scope == MemoryScope.global_
    }
    project_slots: dict[UUID, dict[MemorySlot, str]] = {}
    for item in items:
        if item.scope == MemoryScope.project and item.project_id is not None:
            project_slots.setdefault(item.project_id, {})[item.category] = item.content
    tiers = [
        MemoryTier(
            title="Global",
            path=GlobalMemoryStore(scope=scope).root,
            slots=global_slots,
        ),
        *(
            MemoryTier(
                title=f"Project {project_names[project_id]} ({project_id})",
                path=project_paths[project_id],
                slots=slots,
            )
            for project_id, slots in sorted(
                project_slots.items(), key=lambda entry: project_names[entry[0]]
            )
        ),
    ]
    return MemoryPrintout(person=person, tiers=tiers)


def clear_memory(person: Person) -> str:
    scope = personal_scope(person)
    _require_personal_memory(scope)
    try:
        reset = reset_person_memory(scope)
    except PersonMemoryResetError as error:
        raise OperatorCommandError(str(error)) from None
    if not reset.removed:
        return f"User {person.user_id} has no memory to clear, in {reset.root}."
    return f"Cleared the memory of user {person.user_id}, in {reset.root}."


_ACTIONS: dict[str, Callable[[Person], str]] = {
    "print": lambda person: read_memory(person).render(),
    "clear": clear_memory,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cowork.ops.memory",
        description="Print or clear one person's memory.",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    add_person_arguments(actions.add_parser("print", help="print what the person's agent learned"))
    add_person_arguments(
        actions.add_parser("clear", help="remove everything the person's agent learned")
    )
    arguments = parser.parse_args(argv)
    person = person_from_arguments(arguments)
    return run_operator_command(lambda: _ACTIONS[arguments.action](person))


if __name__ == "__main__":
    sys.exit(main())
