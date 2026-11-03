"""Operator-only project sharing: share, unshare, and list live grants."""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from typing import Literal
from uuid import UUID

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.models.project import Project
from cowork.ops import (
    OperatorCommandError, Person, add_person_arguments, person_from_arguments,
    personal_scope, run_operator_command,
)
from cowork.services.projects import ProjectService
from cowork.shared_projects import SharedProject, SharedProjects, change_shared_projects, shared_projects_file


def _scope(person: Person) -> TenantScope:
    scope = personal_scope(person)
    if not scope.personal or not scope.partitioned_for(PersonalResource.PROJECTS):
        raise OperatorCommandError("Selective project sharing requires personal projects on a trusted-identity local deployment.")
    return scope


def _change(change: Callable[[SharedProjects], SharedProjects]) -> SharedProjects:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise OperatorCommandError("Run this command as the user Cowork runs as, not root.")
    try:
        return change_shared_projects(change)
    except (OSError, ValueError, TimeoutError) as error:
        raise OperatorCommandError(f"Project shares were left unchanged: {error}") from error


def share_project(*, person: Person, project: str, audience: Literal["everyone"] | tuple[UUID, ...]) -> str:
    scope = _scope(person)
    with get_open_session() as raw:
        try:
            row = ProjectService(ScopedSession(raw, scope)).get_project_by_name(project)
        except ValueError as error:
            raise OperatorCommandError("The owner has no project with that name.") from error
        if row.created_by != person.user_id:
            raise OperatorCommandError("The named user does not own this project.")
        entry = SharedProject(organization_id=person.organization_id, user_id=person.user_id,
            partition_id=person.partition_id, project_id=row.id, audience=audience)
        _change(lambda registry: SharedProjects(tuple(
            item for item in registry.root if item.project_id != entry.project_id
        ) + (entry,)))
        recipients = "everyone" if entry.audience == "everyone" else ", ".join(map(str, entry.audience))
        return f"Shared project {row.name} ({row.id}), owner {person.user_id}, with {recipients}. Recipients can read files; only the owner can run agents or change the project."


def unshare_project(*, person: Person, project: str | None = None, project_id: UUID | None = None) -> str:
    scope = _scope(person)
    if project_id is None:
        if project is None:
            raise OperatorCommandError("Name the project or its immutable id.")
        with get_open_session() as raw:
            try:
                project_id = ProjectService(ScopedSession(raw, scope)).get_project_by_name(project).id
            except ValueError as error:
                raise OperatorCommandError("Project not found; use --project-id to revoke a deleted project's grant.") from error
    def revoke(registry: SharedProjects) -> SharedProjects:
        kept = tuple(item for item in registry.root if not (
            item.project_id == project_id and item.partition_id == person.partition_id and item.user_id == person.user_id
        ))
        if len(kept) == len(registry.root):
            raise OperatorCommandError(
                f"No share of project {project_id} owned by user {person.user_id}; nothing changed. "
                "The list command names each share's project and owner."
            )
        return SharedProjects(kept)

    _change(revoke)
    return f"Stopped sharing project {project_id} of user {person.user_id}. The owner's project is unchanged."


def list_shared_projects() -> str:
    settings = get_app_settings()
    if (settings.tenancy_mode != "local" or not settings.trust_identity_headers
            or PersonalResource.PROJECTS not in settings.personal_resources):
        return "Selective project sharing is inactive; project access follows the deployment's resource policy."
    try:
        registry = SharedProjects.read(shared_projects_file())
    except ValueError as error:
        raise OperatorCommandError(str(error)) from error
    lines = ["Projects shared read-only (agent execution stays with the owner):"]
    with get_open_session() as session:
        for entry in registry.root:
            row = session.get(Project, entry.project_id)
            audience = "everyone" if entry.audience == "everyone" else ", ".join(map(str, entry.audience))
            if row is None or row.org_id != entry.partition_id or row.created_by != entry.user_id:
                # Grants nothing (the recipient routes apply the same match), but
                # stays listed so the operator can revoke it by these IDs.
                lines.append(f"  (no such project; grants nothing) ({entry.project_id}) | owner {entry.user_id} | audience {audience}")
                continue
            lines.append(f"  {row.display_name or row.name} ({row.id}) | owner {entry.user_id} | audience {audience}")
    if len(lines) == 1:
        lines.append("  none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    share = commands.add_parser("share")
    add_person_arguments(share)
    share.add_argument("--project", required=True)
    audience = share.add_mutually_exclusive_group(required=True)
    audience.add_argument("--everyone", action="store_true")
    audience.add_argument("--recipient", type=UUID, action="append")
    unshare = commands.add_parser("unshare")
    add_person_arguments(unshare)
    selector = unshare.add_mutually_exclusive_group(required=True)
    selector.add_argument("--project")
    selector.add_argument("--project-id", type=UUID)
    commands.add_parser("list")
    args = parser.parse_args(argv)
    if args.action == "list":
        return run_operator_command(list_shared_projects)
    person = person_from_arguments(args)
    if args.action == "share":
        recipients = "everyone" if args.everyone else tuple(dict.fromkeys(args.recipient))
        return run_operator_command(lambda: share_project(person=person, project=args.project, audience=recipients))
    return run_operator_command(lambda: unshare_project(person=person, project=args.project, project_id=args.project_id))


if __name__ == "__main__":
    sys.exit(main())
