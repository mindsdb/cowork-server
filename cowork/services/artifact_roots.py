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

import logging
import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.services.artifacts import (
    ProjectArtifacts,
    _org_mode,
    _scan_artifact_dirs,
)

logger = logging.getLogger(__name__)

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


def _is_real_directory(path: Path) -> bool:
    """True only for a directory entry, never a link to one."""
    try:
        return stat.S_ISDIR(path.stat(follow_symlinks=False).st_mode)
    except OSError:
        return False


def _storage_components_are_safe(workspace: Path, *, may_be_absent: bool) -> bool:
    """Reject a link in the writable ``.anton/artifacts`` chain.

    This is an early discovery filter.  The identity service repeats the
    guarantee atomically by reopening both components relative to a pinned
    project directory, so replacing either entry after this check is refused at
    use time too.
    """
    for component in (workspace / _ARTIFACTS_SUBPATH[0], workspace.joinpath(*_ARTIFACTS_SUBPATH)):
        try:
            mode = component.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            return may_be_absent
        except OSError:
            return False
        if not stat.S_ISDIR(mode):
            return False
    return True


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


@dataclass(frozen=True)
class ProjectRow:
    """The columns root discovery reads from a project row, as plain data, so
    the file store step that uses them holds no session."""

    id: UUID
    name: str
    path: str
    external: bool


@dataclass(frozen=True)
class RootsPlan:
    """What the database says about the artifacts roots a caller may read.

    Built in a database unit, then expanded on the file store by
    `discover_roots`, so no pooled connection is held while a stat waits on
    the file server.

    `projects` are expanded into their shared base and, in org mode, their
    legacy conversation roots. `scan` asks for the desktop scan of
    `scan_scope`'s projects root, and `adopted` are desktop projects in a
    folder the user chose, outside that root. `owner_filter` keeps only the
    legacy conversation roots of `owned`, the conversations the caller owns,
    which the same database unit reads.
    """

    projects: tuple[ProjectRow, ...] = ()
    scan: bool = False
    scan_scope: TenantScope | None = None
    adopted: tuple[ProjectRow, ...] = ()
    owner_filter: bool = False
    owned: frozenset[UUID] = frozenset()


@dataclass(frozen=True)
class _FoundRoot:
    source: ProjectArtifacts
    # Set for a legacy conversation root: the conversation its directory name
    # names, or None when the name is not a conversation id.
    legacy: bool = False
    conversation_id: UUID | None = None


@dataclass(frozen=True)
class DiscoveredRoots:
    """The roots `discover_roots` found, in discovery order, before the owner
    filter, with the plan's owner filter and owned conversations."""

    roots: tuple[_FoundRoot, ...]
    owner_filter: bool
    owned: frozenset[UUID] = frozenset()

    def sources(self) -> list[ProjectArtifacts]:
        """The roots the caller may read. With the owner filter, a legacy root
        stays only when its conversation is in `owned`, which covers a
        directory name that is not a conversation id at all."""
        return [
            root.source
            for root in self.roots
            if not (self.owner_filter and root.legacy)
            or (root.conversation_id is not None and root.conversation_id in self.owned)
        ]

    def unfiltered(self) -> list[ProjectArtifacts]:
        """Every root, a co-member's legacy roots included. NOT an access
        decision: see `plan_project_roots`'s `include_other_members`."""
        return [root.source for root in self.roots]


def _project_row(service, project) -> ProjectRow:
    return ProjectRow(
        id=project.id,
        name=project.name,
        path=project.path,
        # The same truth as the scope resolver's: a serve URL for an adopted
        # folder cannot be rediscovered by scanning, and this is the resolver
        # the desktop rail reaches (it addresses artifacts by project id).
        external=service.directory_is_external(project),
    )


def _readable_project_rows(service, projects) -> tuple[ProjectRow, ...]:
    """`projects`' rows, leaving out a project that names a folder in another
    person's partition (`ProjectService.sits_in_another_partition`)."""
    rows: list[ProjectRow] = []
    for project in projects:
        if service.sits_in_another_partition(project):
            logger.warning(
                "artifacts: a project names a folder in another person's partition, so its "
                "artifacts are left out",
                extra={
                    "event": "artifact_folder_refused",
                    "reason": "other_partition",
                    "project_id": str(project.id),
                },
            )
            continue
        rows.append(_project_row(service, project))
    return tuple(rows)


def plan_project_roots(
    session: ScopedSession, project_id: UUID, *, include_other_members: bool = False
) -> RootsPlan:
    """The caller's own project by id. Raises ValueError for anything else,
    including another organization's project, which the scoped read does not
    return at all.

    Works in BOTH modes: `ProjectService.get_project` is a plain scoped read and
    resolves fine on desktop too. That is deliberate: without it the desktop
    branch would have to ignore `project_id`, and a slug-addressed delete would
    then act on whichever project happened to sort first.

    The legacy roots keep the owner filter: a conversation workspace is private
    to whoever created it, so a member's workspaces must not appear in another
    member's roots. That check happens in root discovery rather than at the
    route because no artifact route ever receives a conversation id: clients
    address artifacts by project and slug, and the conversation only exists as
    the directory name. Filtering here covers the list, the delete, and
    anything else that resolves roots.

    `include_other_members` drops that filter and is NOT an access decision: it
    only widens the search so a co-member's artifact can be found by id, and
    every caller must then check the owner's per-artifact grant
    (`artifact_draft_review.draft_review_allows`). It exists because a review
    route receives an artifact id and no conversation id, so there is nothing
    else to look the folder up by. Never reachable from the artifacts list or
    from any mutation.
    """
    from cowork.services.projects import ProjectService

    service = ProjectService(session)
    owner_filter = not include_other_members and session.scope.org_mode
    return RootsPlan(
        projects=_readable_project_rows(service, (service.get_project(project_id),)),
        owner_filter=owner_filter,
        owned=owned_conversation_ids(session) if owner_filter else frozenset(),
    )


def plan_scope_roots(session: ScopedSession) -> RootsPlan:
    """Every artifacts root the caller can read, across their organization's
    projects. In org mode that is each project's shared base (ENG-2056) plus the
    caller's own legacy conversation workspaces. In a person's partition it is
    each of that person's projects, read from the database like org mode, so
    every card carries its project id; the scan of the partition's projects
    root would leave cards path-addressed.

    Used by the unparameterized artifacts list, which the frontend calls with no
    project filter. In local mode, and for a scope that keeps projects shared,
    this falls back to the filesystem scan so the desktop list is unchanged.
    """
    from cowork.services.projects import ProjectService

    service = ProjectService(session)
    if not (_org_mode() or session.scope.partitioned_for(PersonalResource.PROJECTS)):
        # Desktop projects pointed at a folder the user chose. The scan only
        # sees direct children of the projects root, so an adopted folder is
        # invisible to it and has to come from its row.
        return RootsPlan(
            scan=True,
            scan_scope=session.scope,
            adopted=tuple(
                _project_row(service, project)
                for project in service.list_projects()
                if service.directory_is_external(project)
            ),
        )
    return RootsPlan(
        projects=_readable_project_rows(service, service.list_projects()),
        owner_filter=session.scope.org_mode,
        owned=owned_conversation_ids(session) if session.scope.org_mode else frozenset(),
    )


def _project_roots(project: ProjectRow) -> list[_FoundRoot]:
    """One project's roots. One on the desktop. In org mode (ENG-2056): the
    PROJECT-level base first, the primary location, where every new artifact
    lands and which every member of the project may read (project-wide
    visibility is the product intent; org isolation still holds because the
    row comes from a scoped database read that never returns another
    organization's project), followed by one legacy per-conversation base for
    each conversation workspace that holds pre-change artifacts.

    A workspace with no `.anton/artifacts` holds no artifacts and is not a
    root. Counting it would make every lookup in the project index an empty
    directory, and most conversation workspaces have none.

    Every root carries the SAME project identity: a conversation is where the
    bytes happen to live, not a thing the client addresses artifacts by, so
    cards stay project-addressed in both modes.
    """
    project_path = Path(project.path)

    def root(base: Path, **found) -> _FoundRoot:
        return _FoundRoot(
            source=ProjectArtifacts(
                base=base,
                project_id=str(project.id),
                project_name=project.name,
                trusted_anchor=project_path,
                root_parts=base.relative_to(project_path).parts,
                external=project.external,
            ),
            **found,
        )

    roots = [root(_artifacts_base(project.path))]
    if not _org_mode():
        return roots
    conversations = project_path / CONVERSATIONS_DIRNAME
    if not _is_real_directory(conversations):
        # No legacy per-conversation dir at all is common and not an error:
        # the project-level base is still a valid, primary root.
        return roots
    try:
        children = sorted(conversations.iterdir())
    except OSError:
        return roots
    return roots + [
        root(child.joinpath(*_ARTIFACTS_SUBPATH), legacy=True, conversation_id=_conversation_id(child))
        for child in children
        if _is_real_directory(child) and _storage_components_are_safe(child, may_be_absent=False)
    ]


def _adopted_root(project: ProjectRow) -> _FoundRoot | None:
    """A desktop project in a folder the user chose.

    `project_name` is the row's name, not the folder's basename. Those were the
    same string for every project the scan can see, and they are not once a
    folder is adopted: adopting `~/Documents/notes` while `notes` is taken
    gives a row named `notes-2` over a directory named `notes`.
    """
    project_path = Path(project.path)
    # Same refusal as the scan: a linked root must not become an
    # authorization source.
    if not _storage_components_are_safe(project_path, may_be_absent=False):
        return None
    base = _artifacts_base(project.path)
    if not _is_real_directory(base):
        return None
    return _FoundRoot(
        source=ProjectArtifacts(
            base=base,
            # None keeps desktop cards path-addressed, exactly as the scan
            # leaves them; adopting a folder must not change addressing.
            project_id=None,
            project_name=project.name,
            trusted_anchor=project_path,
            root_parts=_ARTIFACTS_SUBPATH,
            external=True,
        )
    )


def discover_roots(plan: RootsPlan) -> DiscoveredRoots:
    """Expand `plan` on the file store. Opens no database session, so it can run
    on a file store thread (cowork.db.units.run_efs)."""
    roots: list[_FoundRoot] = []
    if plan.scan:
        roots += [_FoundRoot(source=source) for source in artifacts_sources_for_scan(scope=plan.scan_scope)]
        roots += [found for project in plan.adopted if (found := _adopted_root(project)) is not None]
    for project in plan.projects:
        roots += _project_roots(project)
    return DiscoveredRoots(roots=tuple(roots), owner_filter=plan.owner_filter, owned=plan.owned)


def readable_sources(plan: RootsPlan) -> list[ProjectArtifacts]:
    """The roots `plan` names that the caller may read. File store work only:
    the plan already holds the owner filter's answer, so async code runs this
    inside a file store call (cowork.db.units.run_efs), with the work on the
    roots it finds."""
    return discover_roots(plan).sources()


def owned_conversation_ids(session: ScopedSession) -> frozenset[UUID]:
    """Every conversation the caller owns, in one query, for a plan's owner
    filter."""
    from cowork.services.conversations import ConversationService

    return frozenset(ConversationService(session).owned_ids())


def artifacts_sources_for_scope(session: ScopedSession) -> list[ProjectArtifacts]:
    """Every artifacts root the caller can read; see `plan_scope_roots`."""
    return readable_sources(plan_scope_roots(session))


def artifacts_sources_for_project(
    session: ScopedSession, project_id: UUID, *, include_other_members: bool = False
) -> list[ProjectArtifacts]:
    """The caller's own project's roots by id; see `plan_project_roots`.

    A LIST, not one root: in org mode a project's artifacts sit in the shared
    project base plus the caller's own legacy conversation roots (ENG-2056), so
    a caller that addresses by slug has to look in each. Desktop always yields
    exactly one.
    """
    return readable_sources(
        plan_project_roots(session, project_id, include_other_members=include_other_members)
    )


def artifacts_sources_for_scan(*, scope: TenantScope | None) -> list[ProjectArtifacts]:
    """Desktop: the registered `.anton/artifacts` dirs found by scanning
    ``scope``'s projects root — the unkeyed root for LOCAL_SCOPE, and a
    person's partition root where their projects are personal.

    `project_id` is None here — desktop cards stay path-addressed. The directory
    name IS the project name: `create_project` builds both from one sanitized
    string, and a rename moves the directory and updates the row together
    (services/projects.py).
    """
    sources: list[ProjectArtifacts] = []
    for base in _scan_artifact_dirs(scope=scope):
        # ``_scan_artifact_dirs`` skips a linked root too; this list repeats
        # the check because it is an authorization source, so a linked root
        # never becomes one. Ordinary desktop directories keep the same shape.
        if not _storage_components_are_safe(base.parent.parent, may_be_absent=False):
            continue
        project_path = base.parent.parent
        sources.append(
            ProjectArtifacts(
                base=base,
                project_id=None,
                project_name=project_path.name,
                trusted_anchor=project_path,
                root_parts=_ARTIFACTS_SUBPATH,
            )
        )
    return sources
