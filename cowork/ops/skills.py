"""One person's skills: ``python -m cowork.ops.skills share|unshare|snapshot``.

``share --skill <slug>`` shares one of the person's skills with everyone,
read-only, on their behalf: it adds the skill to the shared-skill list,
``<COWORK_SHARED_DIR>/shared/skills.json`` (cowork.shared_skills), with the
checks and refusal texts the person's own share from their Skills page gets
(SkillService.share). Nothing is copied: everyone else reads the skill live
from the person's own store and can't change it, and the person still edits,
disables and deletes it as their own. Its name stays while it is shared. The
person can hide it again from their Skills page.

``unshare --skill <slug>`` removes the person's entry for that skill from the
list (SkillService.stop_sharing), whether or not the skill still exists. The
person's skill stays in their store.

Both run as the user Cowork runs as, inside its container, and refuse root:
Cowork writes the list and its lock file itself, and a folder or lock file
root creates there would stop every later write.

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

from cowork.managed import managed_folder, readonly_skill_dirs
from cowork.ops import (
    OperatorCommandError,
    Person,
    add_person_arguments,
    person_from_arguments,
    personal_scope,
    run_operator_command,
)
from cowork.services.projects import GENERAL_PROJECT
from cowork.services.skills import (
    SharedSkillsBusyError,
    SkillNotFoundError,
    SkillService,
    SkillSharingError,
    linked_turn_skills,
    shared_turn_skills,
)
from cowork.shared_skills import shared_skills_file

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
        ("host", list(readonly_skill_dirs(HOST_SKILLS_DIR).readable.values())),
        ("managed", [] if managed is None else list(managed.skill_dirs().values())),
        ("shared", [skill.path for skill in shared_turn_skills(service=service, project_name=project)]),
        ("anton", list(readonly_skill_dirs(SkillStore().builtin_root).readable.values())),
    ]
    for source, skill_dirs in readonly:
        for skill_dir in skill_dirs:
            if skill_dir.name not in found:
                found[skill_dir.name] = SnapshotSkill(
                    slug=skill_dir.name, source=source, files=_skill_files(skill_dir)
                )
    return SkillSnapshot(person=person, project=project, skills=list(found.values()))


def _refuse_root() -> None:
    """Cowork writes the shared-skill list, so a share or unshare runs as
    Cowork's own user: a ``shared/`` folder or lock file root creates would
    make every later write of Cowork's fail."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise OperatorCommandError(
            "Run this as the user Cowork runs as, not root: Cowork itself writes "
            f"{shared_skills_file()}, and a folder or lock file root creates there would "
            "stop it."
        )


def share_skill(*, person: Person, slug: str) -> str:
    """Share the person's skill ``slug`` with everyone, read-only, with the
    checks their own share gets (SkillService.share), and say so."""
    _refuse_root()
    service = SkillService(personal_scope(person))
    try:
        added = service.share(slug)
    except SkillNotFoundError:
        raise OperatorCommandError(f"User {person.user_id} has no skill named {slug}.") from None
    except (SkillSharingError, SharedSkillsBusyError, ValueError) as error:
        raise OperatorCommandError(str(error)) from None
    if not added:
        raise OperatorCommandError(f"Skill {slug} of user {person.user_id} is already shared.")
    return (
        f"Shared skill {slug} of user {person.user_id} with everyone, read-only: "
        f"added it to {shared_skills_file()}."
    )


def unshare_skill(*, person: Person, slug: str) -> str:
    """Stop sharing the person's skill ``slug`` (SkillService.stop_sharing),
    whether or not the skill still exists, and say so."""
    _refuse_root()
    service = SkillService(personal_scope(person))
    try:
        removed = service.stop_sharing(slug)
    except (SkillSharingError, SharedSkillsBusyError) as error:
        raise OperatorCommandError(str(error)) from None
    if not removed:
        raise OperatorCommandError(f"User {person.user_id} shares no skill named {slug}.")
    return (
        f"Stopped sharing skill {slug} of user {person.user_id}: removed it from "
        f"{shared_skills_file()}. Their own skill stays in their store."
    )


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
            return share_skill(person=person, slug=arguments.skill)
        return unshare_skill(person=person, slug=arguments.skill)

    return run_operator_command(command)


if __name__ == "__main__":
    sys.exit(main())
