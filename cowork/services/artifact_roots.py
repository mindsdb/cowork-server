"""Resolving which artifacts root belongs to the caller.

This is the ONE place that branches on tenancy mode for artifact storage.

Org mode resolves roots from the database through a `ScopedSession`, so tenant
isolation comes from the query layer the project already relies on — a project
row belonging to another organization is simply not returned (`ScopedSession.get`
compares `row.org_id` to the scope). Nothing here trusts a path supplied by the
client, because the artifact HTTP surface has no way to tell which organization a
filesystem path belongs to.

Desktop keeps the pre-existing filesystem scan: one user per machine, so the scan
IS the authorization boundary there. It still resolves a project by id when given
one, so both modes address artifacts the same way.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from cowork.db.scoped import ScopedSession
from cowork.services.artifacts import (
    ProjectArtifacts,
    _org_mode,
    _scan_artifact_dirs,
)

_ARTIFACTS_SUBPATH = (".anton", "artifacts")


def _artifacts_base(project_path: str) -> Path:
    return Path(project_path).joinpath(*_ARTIFACTS_SUBPATH)


def _source_for(project) -> ProjectArtifacts:
    return ProjectArtifacts(
        base=_artifacts_base(project.path),
        project_id=str(project.id),
        project_name=project.name,
    )


def artifacts_sources_for_scope(session: ScopedSession) -> list[ProjectArtifacts]:
    """Every artifacts root the caller's organization owns.

    Used by the unparameterized artifacts list, which the frontend calls with no
    project filter. In local mode this falls back to the filesystem scan so the
    desktop list is unchanged.
    """
    if not _org_mode():
        return artifacts_sources_for_scan()

    from cowork.services.projects import ProjectService

    return [_source_for(p) for p in ProjectService(session).list_projects()]


def artifacts_source_for_project(session: ScopedSession, project_id: UUID) -> ProjectArtifacts:
    """The caller's own project by id. Raises ValueError for anything else —
    including another organization's project, which the scoped read does not
    return at all.

    Works in BOTH modes: `ProjectService.get_project` is a plain scoped read and
    resolves fine on desktop too. That is deliberate — without it the desktop
    branch would have to ignore `project_id`, and a slug-addressed delete would
    then act on whichever project happened to sort first.
    """
    from cowork.services.projects import ProjectService

    return _source_for(ProjectService(session).get_project(project_id))


def artifacts_sources_for_scan() -> list[ProjectArtifacts]:
    """Desktop: the registered `.anton/artifacts` dirs found by scanning the
    projects root.

    `project_id` is None here — desktop cards stay path-addressed. The directory
    name IS the project name: `create_project` builds both from one sanitized
    string, and a rename moves the directory and updates the row together
    (services/projects.py).
    """
    return [
        ProjectArtifacts(base=base, project_id=None, project_name=base.parent.parent.name)
        for base in _scan_artifact_dirs()
    ]
