"""One person's skills: ``python -m cowork.ops.skills share|snapshot``.

``share --skill <slug> --managed-dir <folder>`` moves one of the person's
skills into the operator's managed skills: it copies the skill's folder into
``<folder>/<slug>``, then deletes the person's own copy. ``<folder>`` is a
copy of the managed ``skills/`` folder the caller can write, which the caller
then places in the managed folder itself, read-only for everyone, the
author included.

``snapshot [--project <name>]`` lists the skills the person's agent sees in a
project (``general`` unless named), in anton's order of precedence, and
prints one sha256 over them: the sha256 of the lines ``<sha256 of SKILL.md>
<slug>/SKILL.md``, one per skill, sorted by slug, as ``sha256sum`` prints
them.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

from anton.core.tools.skill_format import SKILL_FILE
from pydantic import BaseModel, ConfigDict

from cowork.managed import managed_folder
from cowork.ops import (
    OperatorCommandError,
    Person,
    add_person_arguments,
    person_from_arguments,
    personal_scope,
    run_operator_command,
)
from cowork.services.projects import GENERAL_PROJECT
from cowork.services.skills import SkillService, is_builtin_skill, linked_turn_skills

SkillSource = Literal["own", "host", "managed", "anton"]


class SnapshotSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str
    source: SkillSource
    sha256: str


class SkillSnapshot(BaseModel):
    """The skills one person's agent sees in one project."""

    model_config = ConfigDict(frozen=True)

    person: Person
    project: str
    skills: list[SnapshotSkill]

    @property
    def sha256(self) -> str:
        manifest = "".join(
            f"{skill.sha256}  {skill.slug}/{SKILL_FILE}\n"
            for skill in sorted(self.skills, key=lambda skill: skill.slug)
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


class SharedSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str
    destination: Path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readonly_skill_dirs(root: Path) -> list[Path]:
    """The skill folders anton reads from one read-only root: every
    directory holding a SKILL.md file."""
    if not root.is_dir():
        return []
    return [
        entry
        for entry in sorted(root.iterdir())
        if entry.is_dir() and (entry / SKILL_FILE).is_file()
    ]


def snapshot_skills(*, person: Person, project: str) -> SkillSnapshot:
    """The skills a turn of the person's in ``project`` hands anton.

    The turn's own skills folder first (link_turn_skills, which seeds the
    packaged builtins into the person's store, as their first turn does),
    then anton's read-only roots in its order: the harness's own skills, the
    managed skills, then anton's packaged ones. A slug is taken from the
    first place that has it, as anton's SkillStore takes it.
    """
    from anton.core.memory.skills import SkillStore

    from cowork.harnesses.anton_harness.harness import HOST_SKILLS_DIR

    service = SkillService(personal_scope(person))
    service.ensure_builtin_skills()
    found: dict[str, SnapshotSkill] = {}
    for skill in linked_turn_skills(service=service, project_name=project):
        found[skill.slug] = SnapshotSkill(
            slug=skill.slug, source="own", sha256=_file_sha256(skill.path / SKILL_FILE)
        )
    managed = managed_folder()
    readonly_roots: list[tuple[SkillSource, Path]] = [
        ("host", HOST_SKILLS_DIR),
        *([] if managed is None else [("managed", managed.skills_dir)]),
        ("anton", SkillStore().builtin_root),
    ]
    for source, root in readonly_roots:
        for skill_dir in _readonly_skill_dirs(root):
            if skill_dir.name not in found:
                found[skill_dir.name] = SnapshotSkill(
                    slug=skill_dir.name,
                    source=source,
                    sha256=_file_sha256(skill_dir / SKILL_FILE),
                )
    return SkillSnapshot(person=person, project=project, skills=list(found.values()))


def _refuse_links(skill_dir: Path) -> None:
    """A link in a shared skill would hand every person whatever it points
    at, so the person's copy must hold none."""
    for directory, subdirectories, files in os.walk(skill_dir):
        for name in [*subdirectories, *files]:
            entry = Path(directory) / name
            if entry.is_symlink():
                raise OperatorCommandError(
                    f"The skill holds a link, {entry}. Remove it before sharing the skill."
                )


def share_skill(*, person: Person, slug: str, managed_dir: Path) -> SharedSkill:
    if not managed_dir.is_dir():
        raise OperatorCommandError(f"{managed_dir} is not a folder.")
    if is_builtin_skill(slug):
        raise OperatorCommandError(
            f"{slug} is a packaged built-in skill, which every person already has."
        )
    managed = managed_folder()
    if managed is not None and slug in managed.skill_slugs():
        raise OperatorCommandError(f"The managed folder already provides a skill named {slug}.")
    service = SkillService(personal_scope(person))
    try:
        complete = service.has_complete_skill(slug)
    except ValueError as error:
        raise OperatorCommandError(str(error)) from None
    if not complete:
        raise OperatorCommandError(f"User {person.user_id} has no skill named {slug}.")
    source = service._skill_dir(slug)
    _refuse_links(source)
    destination = managed_dir / slug
    if destination.exists() or destination.is_symlink():
        raise OperatorCommandError(f"{destination} already exists.")

    # Copied under a name no skill can have, so the person's copy is deleted
    # before the shared one is in place under the skill's name.
    staged = managed_dir / f".{slug}.sharing-{uuid4().hex}"
    shutil.copytree(source, staged, symlinks=True)
    try:
        service.delete_skill(slug)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    staged.rename(destination)
    return SharedSkill(slug=slug, destination=destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cowork.ops.skills",
        description="Share one person's skill with everyone, or print the skills their agent sees.",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    share = actions.add_parser("share", help="move the person's skill into the managed skills")
    add_person_arguments(share)
    share.add_argument("--skill", required=True, help="the skill's name, its slug")
    share.add_argument(
        "--managed-dir",
        required=True,
        type=Path,
        help="the writable copy of the managed skills folder the skill is copied into",
    )
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
        shared = share_skill(person=person, slug=arguments.skill, managed_dir=arguments.managed_dir)
        return (
            f"Moved skill {shared.slug} of user {person.user_id} into {shared.destination}. "
            "The person's own copy is deleted."
        )

    return run_operator_command(command)


if __name__ == "__main__":
    sys.exit(main())
