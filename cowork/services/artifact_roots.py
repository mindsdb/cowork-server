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

#: Org mode only. The agent's workspace on the cloud is one conversation, not the
#: project: scratchpad-controller mounts `<project>/conversations/<conversation_id>`
#: at the pod's workspace root and anton writes artifacts under
#: `<workspace>/.anton/artifacts` exactly as it does on the desktop, so the tree
#: gains a segment cowork-server has to know about. The isolation is deliberate on
#: the controller's side — the workspace lands on the scratchpad's `sys.path`, so a
#: project-wide mount would let a cell in one conversation plant a module that
#: imports in a co-user's pod (see live_pod.py).
#:
#: Desktop has no such segment: there the workspace IS the project directory, every
#: conversation shares one artifacts folder, and nothing below changes.
CONVERSATIONS_DIRNAME = "conversations"


def _artifacts_base(project_path: str) -> Path:
    return Path(project_path).joinpath(*_ARTIFACTS_SUBPATH)


def conversation_artifacts_base(project_path: str, conversation_id) -> Path:
    """The artifacts root one org-mode turn writes into.

    Local mode ignores `conversation_id` and returns the project-wide root, so a
    caller can hand its conversation id over unconditionally.
    """
    if not _org_mode():
        return _artifacts_base(project_path)
    return (
        Path(project_path)
        / CONVERSATIONS_DIRNAME
        / str(conversation_id)
    ).joinpath(*_ARTIFACTS_SUBPATH)


def _project_artifact_bases(project_path: str) -> list[Path]:
    """Every artifacts root belonging to one project.

    One on the desktop. In org mode, one per conversation that has actually
    written something — the directory only exists once a pod has mounted it, so an
    absent `conversations/` dir means the project has no cloud artifacts yet and
    is not an error.
    """
    if not _org_mode():
        return [_artifacts_base(project_path)]
    conversations = Path(project_path) / CONVERSATIONS_DIRNAME
    try:
        children = sorted(conversations.iterdir())
    except OSError:
        return []
    return [child.joinpath(*_ARTIFACTS_SUBPATH) for child in children if child.is_dir()]


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

    A LIST, not one root: in org mode a project's artifacts are spread across its
    conversations, so a caller that addresses by slug has to look in each. Desktop
    always yields exactly one.

    Works in BOTH modes: `ProjectService.get_project` is a plain scoped read and
    resolves fine on desktop too. That is deliberate — without it the desktop
    branch would have to ignore `project_id`, and a slug-addressed delete would
    then act on whichever project happened to sort first.
    """
    from cowork.services.projects import ProjectService

    return _sources_for(ProjectService(session).get_project(project_id))


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
