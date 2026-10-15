"""One person's skills: ``python -m cowork.ops.skills share|unshare|snapshot``.

``share --skill <slug> --managed-dir <folder>`` shares one of the person's
skills with everyone, read-only: it adds the skill to the shared-skill
registry, ``<folder>/shared/skills.json`` (cowork.managed.SharedSkills).
``<folder>`` is the managed folder, writable by the caller; cowork-server
reads the same registry read-only. Nothing is copied: everyone else reads the
skill live from the person's own store and can't change it, and the person
still edits, disables and deletes it as their own (SkillService). Its name
stays while it is shared.

``unshare --skill <slug> --managed-dir <folder>`` removes the person's entry
for that skill from the registry. The person's skill stays in their store.
Both take ``--registry-stdout``, which prints the new registry instead of
writing it (cowork.ops).

``snapshot [--project <name>]`` lists the skills the person's agent sees in a
project (``general`` unless named), shared ones included, in anton's order of
precedence, and prints one sha256 over them: the sha256 of the lines
``<sha256 of the file>  <slug>/<path in the skill>``, one per file of each
skill, sorted by slug and path, as ``sha256sum`` prints them. Every file the
agent can read counts, helper scripts and reference files included; a skill's
``stats.json``, its hidden files and links don't, as for the files a turn
hands the agent (cowork.services.skills._skill_wire_files).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cowork.db.personal_resources import PersonalResource
from cowork.managed import SharedSkill, SharedSkills, managed_folder, readonly_skill_dirs
from cowork.ops import (
    REGISTRY_ON_STDOUT,
    OperatorCommandError,
    Person,
    RegistryChange,
    add_person_arguments,
    add_registry_stdout_argument,
    finish_registry_change,
    managed_folder_at,
    person_from_arguments,
    personal_scope,
    read_registry,
    run_operator_command,
)
from cowork.services.projects import GENERAL_PROJECT
from cowork.services.skills import (
    SkillService,
    agent_skill_names,
    is_builtin_skill,
    linked_turn_skills,
    shared_turn_skills,
)

SkillSource = Literal["own", "host", "managed", "shared", "anton"]


class SnapshotFile(BaseModel):
    """One file of a skill: its path inside the skill's folder, and its sha256."""

    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str


class SnapshotSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str
    source: SkillSource
    files: tuple[SnapshotFile, ...]


class SkillSnapshot(BaseModel):
    """The skills one person's agent sees in one project."""

    model_config = ConfigDict(frozen=True)

    person: Person
    project: str
    skills: list[SnapshotSkill]

    @property
    def sha256(self) -> str:
        manifest = "".join(
            f"{file.sha256}  {skill.slug}/{file.path}\n"
            for skill in sorted(self.skills, key=lambda skill: skill.slug)
            for file in sorted(skill.files, key=lambda file: file.path)
        )
        return hashlib.sha256(manifest.encode("utf-8")).hexdigest()

    def render(self) -> str:
        lines = [
            f"Skills user {self.person.user_id}'s agent sees in project "
            f"{self.project}, space {self.person.partition_id}:"
        ]
        lines.extend(
            f"{skill.slug}  ({skill.source})"
            for skill in sorted(self.skills, key=lambda skill: skill.slug)
        )
        lines.append(f"sha256 {self.sha256}")
        return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _skill_files(skill_dir: Path) -> tuple[SnapshotFile, ...]:
    """Every file of one skill the agent can read, by path inside the skill.

    Leaves out what cowork.services.skills._skill_wire_files leaves out of a
    turn: ``stats.json``, the store's private usage sidecar, which changes as
    the skill is used; hidden files; and links. Unlike a turn's files, binary
    and large files count, since the agent reads those too.
    """
    files: list[SnapshotFile] = []
    for child in sorted(skill_dir.rglob("*")):
        if not child.is_file() or child.is_symlink():
            continue
        relative = child.relative_to(skill_dir).as_posix()
        if relative == "stats.json" or any(part.startswith(".") for part in relative.split("/")):
            continue
        files.append(SnapshotFile(path=relative, sha256=_file_sha256(child)))
    return tuple(files)


def snapshot_skills(*, person: Person, project: str) -> SkillSnapshot:
    """The skills a turn of the person's in ``project`` hands anton.

    The turn's own skills folder first (link_turn_skills, which seeds the
    packaged builtins into the person's store, as their first turn does),
    then anton's read-only roots in its order: the harness's own skills, the
    managed skills, the skills others share with the person (each read live
    from its creator's store), then anton's packaged ones. A slug is taken
    from the first place that has it, as anton's SkillStore takes it.
    """
    from anton.core.memory.skills import SkillStore

    from cowork.harnesses.anton_harness.harness import HOST_SKILLS_DIR

    service = SkillService(personal_scope(person))
    service.ensure_builtin_skills()
    found: dict[str, SnapshotSkill] = {}
    for skill in linked_turn_skills(service=service, project_name=project):
        found[skill.slug] = SnapshotSkill(
            slug=skill.slug, source="own", files=_skill_files(skill.path)
        )
    managed = managed_folder()
    readonly: list[tuple[SkillSource, list[Path]]] = [
        ("host", list(readonly_skill_dirs(HOST_SKILLS_DIR).values())),
        ("managed", [] if managed is None else list(managed.skill_dirs().values())),
        ("shared", [skill.path for skill in shared_turn_skills(service=service, project_name=project)]),
        ("anton", list(readonly_skill_dirs(SkillStore().builtin_root).values())),
    ]
    for source, skill_dirs in readonly:
        for skill_dir in skill_dirs:
            if skill_dir.name not in found:
                found[skill_dir.name] = SnapshotSkill(
                    slug=skill_dir.name, source=source, files=_skill_files(skill_dir)
                )
    return SkillSnapshot(person=person, project=project, skills=list(found.values()))


def _refuse_links(skill_dir: Path) -> None:
    """A link in a shared skill would hand every person whatever it points
    at, so the person's skill must hold none when it is shared."""
    for directory, subdirectories, files in os.walk(skill_dir):
        for name in [*subdirectories, *files]:
            entry = Path(directory) / name
            if entry.is_symlink():
                raise OperatorCommandError(
                    f"The skill holds a link, {entry}. Remove it before sharing the skill."
                )


def share_skill(*, person: Person, slug: str, managed_dir: Path) -> RegistryChange[SharedSkill]:
    scope = personal_scope(person)
    if not scope.partitioned_for(PersonalResource.SKILLS):
        raise OperatorCommandError(
            "COWORK_PERSONAL_RESOURCES leaves skills shared, so everyone already uses every skill."
        )
    path = managed_folder_at(managed_dir).shared_skills_file
    registry = read_registry(registry_type=SharedSkills, path=path)
    if is_builtin_skill(slug):
        raise OperatorCommandError(
            f"{slug} is a packaged built-in skill, which every person already has."
        )
    if slug in agent_skill_names():
        raise OperatorCommandError(
            f"Every person's agent already has a skill named {slug}, so the share would reach "
            "nobody. Rename the skill before sharing it."
        )
    managed = managed_folder()
    if managed is not None and slug in managed.skill_slugs():
        # Everyone reads the operator's own skill first, so the share would
        # reach nobody.
        raise OperatorCommandError(f"The managed folder already provides a skill named {slug}.")
    service = SkillService(scope)
    try:
        complete = service.has_complete_skill(slug)
    except ValueError as error:
        raise OperatorCommandError(str(error)) from None
    if not complete:
        raise OperatorCommandError(f"User {person.user_id} has no skill named {slug}.")
    _refuse_links(service._skill_dir(slug))
    for entry in registry.root:
        if entry.skill != slug:
            continue
        if entry.partition_id == person.partition_id:
            raise OperatorCommandError(f"Skill {slug} of user {person.user_id} is already shared.")
        # Everyone else reads the first entry for a name, so a second one
        # would reach nobody.
        raise OperatorCommandError(f"User {entry.user_id} already shares a skill named {slug}.")
    added = SharedSkill(user_id=person.user_id, partition_id=person.partition_id, skill=slug)
    return RegistryChange(
        entries=(added,), registry=path, updated=SharedSkills((*registry.root, added))
    )


def unshare_skill(*, person: Person, slug: str, managed_dir: Path) -> RegistryChange[SharedSkill]:
    path = managed_folder_at(managed_dir).shared_skills_file
    entries = read_registry(registry_type=SharedSkills, path=path).root
    removed = tuple(
        entry
        for entry in entries
        if entry.partition_id == person.partition_id and entry.skill == slug
    )
    if not removed:
        raise OperatorCommandError(f"User {person.user_id} shares no skill named {slug}.")
    kept = tuple(entry for entry in entries if entry not in removed)
    return RegistryChange(entries=removed, registry=path, updated=SharedSkills(kept))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cowork.ops.skills",
        description=(
            "Share one person's skill with everyone, stop sharing it, or print the "
            "skills their agent sees."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)
    share = actions.add_parser("share", help="share the person's skill with everyone, read-only")
    unshare = actions.add_parser(
        "unshare", help="stop sharing it; the person's skill stays theirs"
    )
    for action in (share, unshare):
        add_person_arguments(action)
        action.add_argument("--skill", required=True, help="the skill's name, its slug")
        action.add_argument(
            "--managed-dir",
            required=True,
            type=Path,
            help="the managed folder, writable: the registry is its shared/skills.json",
        )
        add_registry_stdout_argument(action)
    snapshot = actions.add_parser(
        "snapshot", help="print the skills the person's agent sees, with one sha256"
    )
    add_person_arguments(snapshot)
    snapshot.add_argument(
        "--project", default=GENERAL_PROJECT, help="the project's name (default: general)"
    )
    arguments = parser.parse_args(argv)
    person = person_from_arguments(arguments)

    def command() -> str:
        if arguments.action == "snapshot":
            return snapshot_skills(person=person, project=arguments.project).render()
        if arguments.action == "share":
            change = share_skill(
                person=person, slug=arguments.skill, managed_dir=arguments.managed_dir
            )
            target = REGISTRY_ON_STDOUT if arguments.registry_stdout else change.registry
            message = "\n".join(
                f"Shared skill {entry.skill} of user {person.user_id} with everyone, "
                f"read-only: added it to {target}."
                for entry in change.entries
            )
        else:
            change = unshare_skill(
                person=person, slug=arguments.skill, managed_dir=arguments.managed_dir
            )
            target = REGISTRY_ON_STDOUT if arguments.registry_stdout else change.registry
            message = "\n".join(
                f"Stopped sharing skill {entry.skill} of user {person.user_id}: removed it from "
                f"{target}. Their own skill stays in their store."
                for entry in change.entries
            )
        return finish_registry_change(
            change, message=message, registry_stdout=arguments.registry_stdout
        )

    return run_operator_command(command)


if __name__ == "__main__":
    sys.exit(main())
