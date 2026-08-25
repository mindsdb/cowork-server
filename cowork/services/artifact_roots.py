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

Both modes now isolate artifacts per conversation on disk (see
`conversation_artifacts_base`/`_project_artifact_bases` below) — desktop used to
share one project-wide folder across every conversation, which let a concurrent
sibling conversation's new artifact get misattributed to the wrong turn's
before/after diff (ENG-1933). The project-wide folder is kept as a fallback root
everywhere artifacts are listed/served/published, so anything written before this
change stays reachable.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from cowork.db.scoped import ScopedSession
from cowork.services.artifacts import (
    CONVERSATIONS_DIRNAME,
    ProjectArtifacts,
    _ARTIFACTS_SUBPATH,
    _artifact_roots_for_project_dir,
    _org_mode,
    _registered_project_dirs,
)


def conversation_artifacts_base(project_path: str, conversation_id) -> Path:
    """The artifacts root one turn writes into: its own conversation's
    folder, in both modes."""
    return (
        Path(project_path)
        / CONVERSATIONS_DIRNAME
        / str(conversation_id)
    ).joinpath(*_ARTIFACTS_SUBPATH)


def _project_artifact_bases(project_path: str) -> list[Path]:
    """Every artifacts root belonging to one project: the legacy project-
    wide folder (pre-existing artifacts, in either mode) plus one per
    conversation that has actually written something — the directory only
    exists once a turn has created it, so an absent `conversations/` dir
    means the project has no per-conversation artifacts yet and is not an
    error."""
    return _artifact_roots_for_project_dir(Path(project_path))


def _sources_for(project) -> list[ProjectArtifacts]:
    """One `ProjectArtifacts` per root. They all carry the SAME project identity:
    a conversation is where the bytes happen to live, not a thing the client
    addresses artifacts by, so cards stay project-addressed in both modes."""
    return [
        ProjectArtifacts(
            base=base,
            project_id=str(project.id),
            project_name=project.name,
        )
        for base in _project_artifact_bases(project.path)
    ]


def artifacts_sources_for_scope(session: ScopedSession) -> list[ProjectArtifacts]:
    """Every artifacts root the caller's organization owns.

    Used by the unparameterized artifacts list, which the frontend calls with no
    project filter. In local mode this falls back to the filesystem scan so the
    desktop list is unchanged.
    """
    if not _org_mode():
        return artifacts_sources_for_scan()

    from cowork.services.projects import ProjectService

    return [
        source
        for project in ProjectService(session).list_projects()
        for source in _sources_for(project)
    ]


def artifacts_sources_for_project(session: ScopedSession, project_id: UUID) -> list[ProjectArtifacts]:
    """The caller's own project by id. Raises ValueError for anything else —
    including another organization's project, which the scoped read does not
    return at all.

    A LIST, not one root: in both modes a project's artifacts are spread across
    its conversations (plus, on desktop, the legacy project-wide folder), so a
    caller that addresses by slug has to look in each.

    Works in BOTH modes: `ProjectService.get_project` is a plain scoped read and
    resolves fine on desktop too. That is deliberate — without it the desktop
    branch would have to ignore `project_id`, and a slug-addressed delete would
    then act on whichever project happened to sort first.
    """
    from cowork.services.projects import ProjectService

    return _sources_for(ProjectService(session).get_project(project_id))


def artifacts_sources_for_scan() -> list[ProjectArtifacts]:
    """Desktop: every registered project's artifact roots (legacy +
    per-conversation) found by scanning the projects root.

    `project_id` is None here — desktop cards stay path-addressed. The directory
    name IS the project name: `create_project` builds both from one sanitized
    string, and a rename moves the directory and updates the row together
    (services/projects.py). Walking `_registered_project_dirs()` directly (rather
    than inferring the project dir back out of each root's ancestors) is what
    keeps this correct now that a root can sit two or four segments below its
    project dir depending on whether it's the legacy folder or a conversation's.
    """
    return [
        ProjectArtifacts(base=base, project_id=None, project_name=project_dir.name)
        for project_dir in _registered_project_dirs()
        for base in _artifact_roots_for_project_dir(project_dir)
    ]
