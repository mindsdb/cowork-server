"""Resolving which artifacts root belongs to the caller.

This is the ONE place that branches on tenancy mode for artifact storage.

Org mode resolves roots from the database through a `ScopedSession`, so tenant
isolation comes from the query layer the project already relies on — a project
row belonging to another organization is simply not returned (`ScopedSession.get`
compares `row.org_id` to the scope). Nothing here trusts a path supplied by the
client, because the artifact HTTP surface has no way to tell which organization a
filesystem path belongs to.

The org filter is not the whole answer, because a project is shared by the whole
organization while the conversation workspaces under it are private to whoever
created them. So org mode also drops the workspaces of other members, and this
module is the only place that can: the artifact routes never see a conversation
id.

Artifacts themselves are the exception to that privacy (ENG-2056): the product
decision is project-wide visibility — every task in a project, across the org's
members, shares the project's artifacts. New artifacts land in the PROJECT-level
base (`<project>/.anton/artifacts`), which every project member may read; only
the LEGACY per-conversation bases keep the owner filter, so artifacts written
before the layout change stay visible to whoever created them without suddenly
exposing the rest of a member's old workspace tree.

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
#: at the pod's workspace root. The isolation is deliberate on the controller's
#: side — the workspace lands on the scratchpad's `sys.path`, so a project-wide
#: mount would let a cell in one conversation plant a module that imports in a
#: co-user's pod (see live_pod.py).
#:
#: ARTIFACTS no longer live under this segment (ENG-2056): the controller mounts
#: the PROJECT-level `.anton/artifacts` separately at `/project-artifacts` (off
#: `sys.path`, so the workspace isolation above still holds) and anton writes
#: there via `ANTON_CLOUD_ARTIFACTS_ROOT`, so sibling tasks in a project share
#: artifacts. Artifacts written before that change still sit at
#: `conversations/<id>/.anton/artifacts` and are read as legacy roots.
#:
#: Desktop has no such segment: there the workspace IS the project directory, every
#: conversation shares one artifacts folder, and nothing below changes.
CONVERSATIONS_DIRNAME = "conversations"


def _artifacts_base(project_path: str) -> Path:
    return Path(project_path).joinpath(*_ARTIFACTS_SUBPATH)


def project_artifacts_base(project_path: str) -> Path:
    """The project-level artifacts root — where NEW artifacts land in BOTH modes.

    ENG-2056: on the cloud, scratchpad-controller mounts this directory at
    `/project-artifacts` in the pod and anton writes artifacts there, so every
    task in a project shares one artifacts tree exactly like the desktop. The
    per-conversation layout `conversation_artifacts_base` describes is legacy —
    still read, never written.
    """
    return _artifacts_base(project_path)


def conversation_artifacts_base(project_path: str, conversation_id) -> Path:
    """LEGACY (pre-ENG-2056): the artifacts root an org-mode turn used to write
    into. New artifacts land in `project_artifacts_base`; this layout is kept
    only so artifacts written before the change stay reachable.

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


def _conversation_id(child: Path) -> UUID | None:
    """The conversation a workspace directory belongs to, or None when the name
    is not one. Every directory this module writes is `str(conversation_id)`
    (see `conversation_artifacts_base`), so a name that will not parse names no
    conversation and therefore has no owner to check against."""
    try:
        return UUID(child.name)
    except ValueError:
        return None


def _project_artifact_bases(project_path: str, session: ScopedSession) -> list[Path]:
    """Every artifacts root of one project the caller is allowed to read.

    One on the desktop. In org mode (ENG-2056): the PROJECT-level base first —
    the primary location, where every new artifact lands and which every member
    of the project may read (project-wide visibility is the product intent; org
    isolation still holds because `project_path` comes from a scoped DB read
    that never returns another organization's project) — followed by one legacy
    per-conversation base for each conversation of the caller's own that wrote
    something before the layout change. An absent legacy directory means no
    pre-change artifacts and is not an error.

    The legacy bases keep the owner filter: a conversation workspace is private
    to whoever created it, so a member's workspaces must not appear in another
    member's roots. That check happens here rather than at the route because no
    artifact route ever receives a conversation id: clients address artifacts by
    project and slug, and the conversation only exists as the directory name.
    Filtering here covers the list, the delete, and anything else that resolves
    roots.

    A legacy directory is skipped unless it names a conversation the caller
    owns, and that covers a name which is not a conversation id at all. The
    sibling gate on project files (`_conversation_workspace_ok`) treats such a
    name as a shared file instead, because it guards a tree where shared files
    really do sit beside the workspaces. Nothing shares this one.
    """
    if not _org_mode():
        return [_artifacts_base(project_path)]
    bases = [_artifacts_base(project_path)]
    conversations = Path(project_path) / CONVERSATIONS_DIRNAME
    try:
        children = sorted(conversations.iterdir())
    except OSError:
        return bases
    children = [child for child in children if child.is_dir()]
    if session.scope.org_mode:
        from cowork.services.conversations import ConversationService

        candidates = {child: _conversation_id(child) for child in children}
        owned = ConversationService(session).owned_ids(
            cid for cid in candidates.values() if cid is not None
        )
        children = [child for child, cid in candidates.items() if cid in owned]
    return bases + [child.joinpath(*_ARTIFACTS_SUBPATH) for child in children]


def _sources_for(session: ScopedSession, project) -> list[ProjectArtifacts]:
    """One `ProjectArtifacts` per root. They all carry the SAME project identity:
    a conversation is where the bytes happen to live, not a thing the client
    addresses artifacts by, so cards stay project-addressed in both modes."""
    return [
        ProjectArtifacts(
            base=base,
            project_id=str(project.id),
            project_name=project.name,
        )
        for base in _project_artifact_bases(project.path, session)
    ]


def artifacts_sources_for_scope(session: ScopedSession) -> list[ProjectArtifacts]:
    """Every artifacts root the caller can read, across their organization's
    projects. In org mode that is each project's shared base (ENG-2056) plus the
    caller's own legacy conversation workspaces.

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
        for source in _sources_for(session, project)
    ]


def artifacts_sources_for_project(session: ScopedSession, project_id: UUID) -> list[ProjectArtifacts]:
    """The caller's own project by id. Raises ValueError for anything else —
    including another organization's project, which the scoped read does not
    return at all.

    A LIST, not one root: in org mode a project's artifacts sit in the shared
    project base plus the caller's own legacy conversation roots (ENG-2056), so
    a caller that addresses by slug has to look in each. Desktop always yields
    exactly one.

    Works in BOTH modes: `ProjectService.get_project` is a plain scoped read and
    resolves fine on desktop too. That is deliberate — without it the desktop
    branch would have to ignore `project_id`, and a slug-addressed delete would
    then act on whichever project happened to sort first.
    """
    from cowork.services.projects import ProjectService

    return _sources_for(session, ProjectService(session).get_project(project_id))


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
