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
from collections.abc import Collection
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.services.artifact_identity import may_not_read
from cowork.services.artifacts import (
    ProjectArtifacts,
    _org_mode,
    _scan_artifact_dirs,
)

if TYPE_CHECKING:
    from cowork.services.artifact_locations import Location
    from cowork.services.artifact_sort_times import NewestFolders
    from cowork.services.held_artifact_roots import HeldRoot

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
#: Top-level project folders that are Cowork's own, not the user's files.
SERVER_PROJECT_FOLDERS = frozenset({CONVERSATIONS_DIRNAME, "skills"})


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


def _is_real_directory(path: Path, *, strict: bool = False) -> bool:
    """True only for a directory entry, never a link to one. With `strict`, a
    stat error other than FileNotFoundError, NotADirectoryError or one that
    says the server may not read the entry (may_not_read) raises instead of
    answering False: what the server may not read it can never serve, so
    leaving it out hides nothing a strict count must see."""
    try:
        return stat.S_ISDIR(path.stat(follow_symlinks=False).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        if strict and not may_not_read(exc=exc):
            raise
        return False


def _storage_components_are_safe(workspace: Path, *, may_be_absent: bool, strict: bool = False) -> bool:
    """Reject a link in the writable ``.anton/artifacts`` chain.

    This is an early discovery filter.  The identity service repeats the
    guarantee atomically by reopening both components relative to a pinned
    project directory, so replacing either entry after this check is refused at
    use time too. With `strict`, a stat error other than FileNotFoundError,
    NotADirectoryError or one that says the server may not read the entry
    (may_not_read) raises instead of answering False.
    """
    for component in (workspace / _ARTIFACTS_SUBPATH[0], workspace.joinpath(*_ARTIFACTS_SUBPATH)):
        try:
            mode = component.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            return may_be_absent
        except NotADirectoryError:
            return False
        except OSError as exc:
            if strict and not may_not_read(exc=exc):
                raise
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


def workspace_conversation_id(name: str) -> UUID | None:
    """The conversation a workspace directory named `name` belongs to, or None
    when the name is not one. Every directory this module writes is
    `str(conversation_id)` (see `conversation_artifacts_base`), so only that
    canonical spelling names a conversation. A name that will not parse names
    none, and neither does another spelling of an id: an uppercase spelling
    is another folder on a case-sensitive store, and one without hyphens or
    with braces is another folder on every store, which the held rows would
    never reach, since a row's root is rebuilt from `str(conversation_id)`
    (recorded_source). Neither has an owner to check against."""
    try:
        conversation_id = UUID(name)
    except ValueError:
        return None
    return conversation_id if str(conversation_id) == name else None


def _conversation_id(child: Path) -> UUID | None:
    """The conversation the workspace directory `child` belongs to
    (workspace_conversation_id)."""
    return workspace_conversation_id(child.name)


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
    which the same database unit reads. `partition` is the caller's partition
    of the shared root (`projects.partition_of`): a project whose folder sits
    in another one is left out of `projects` when the plan is built, so no
    reader of the plan reaches it, and again when the plan is expanded.
    `held`, which only the artifacts routes' plans set (`with_held_roots`),
    names the roots of `projects` known to hold an artifact folder, so
    expanding the plan opens nothing on the file store; None finds the roots
    there. `newest`, which only the list's plans set, holds the newest
    sort-time rows of the roots `held` names for the projects outside `walk`
    (artifact_sort_times.NewestFolders), and names the projects of `walk`
    among `newest.unread`. The list then opens only the folders of the
    rows, and reads every folder of the projects in `newest.unread`, which
    have no rows to trust. When the rows were cut at the query's limit and
    the folders at least as new as the last row kept build fewer cards than
    the list shows, it reads every folder of the plan's roots instead; only
    the folders newer than that row count when the run of rows at its time
    was too long to read (NewestFolders.ties_read).
    """

    projects: tuple[ProjectRow, ...] = ()
    partition: str | None = None
    scan: bool = False
    scan_scope: TenantScope | None = None
    adopted: tuple[ProjectRow, ...] = ()
    owner_filter: bool = False
    owned: frozenset[UUID] = frozenset()
    # Set beside `held` (with_held_roots): the projects of `projects` whose
    # roots the held-root backfill could not read, or that moved or went
    # away during its walk (the unread list of its sentinel,
    # held_artifact_roots.HeldRootSentinel). That backfill wrote none of
    # their rows. A turn end, the project files PUT or the self-heal may
    # have written some since, and `held` holds those, but they may not
    # name every root, so `discover_roots` finds these projects' roots on
    # the file store, as for a plan without `held`.
    walk: frozenset[UUID] = frozenset()
    held: frozenset[HeldRoot] | None = None
    # Set by the artifacts list's plans, beside `held`, once the sort times'
    # backfill has written its sentinel (artifact_sort_times.newest_folders):
    # `rows`, the newest rows of the held roots of the projects outside
    # `walk`, newest first; `complete`, false when the query's limit cut
    # them; `tied`, the rest of a run of tied times that cut landed in, and
    # `ties_read`, whether that run was read whole; `unread`, the projects
    # whose every folder the list reads: those the sort-time sentinel lists
    # (SortTimeSentinel.unread), which may have no rows and have none in
    # `rows`, and those of `walk`. None reads every folder of the roots the
    # plan expands to.
    newest: NewestFolders | None = None


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


def scoped_project_row(session: ScopedSession, *, project_id: UUID) -> ProjectRow | None:
    """`project_id`'s row as plain data, or None when the caller's scope has
    no such project."""
    from cowork.services.projects import ProjectNotFoundError, ProjectService

    service = ProjectService(session)
    try:
        return _project_row(service, service.get_project(project_id))
    except ProjectNotFoundError:
        return None


def _rows_in_partition(*, service, projects, partition: str | None) -> tuple[ProjectRow, ...]:
    """`projects`' rows, less one that names a folder in another partition
    (`left_out_for_another_partition`). Compares paths against the shared
    root resolved at startup, read once for every project, so a database unit
    may run it."""
    from cowork.services.projects import resolved_shared_root

    shared_root = resolved_shared_root() if partition is not None else None
    rows = (_project_row(service, project) for project in projects)
    return tuple(
        row
        for row in rows
        if not left_out_for_another_partition(row, partition=partition, shared_root=shared_root)
    )


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
    from cowork.services.projects import ProjectService, partition_of

    service = ProjectService(session)
    owner_filter = not include_other_members and session.scope.org_mode
    partition = partition_of(session.scope)
    return RootsPlan(
        projects=_rows_in_partition(service=service, projects=(service.get_project(project_id),), partition=partition),
        partition=partition,
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
    from cowork.services.projects import ProjectService, partition_of

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
    partition = partition_of(session.scope)
    return RootsPlan(
        projects=_rows_in_partition(service=service, projects=service.list_projects(), partition=partition),
        partition=partition,
        owner_filter=session.scope.org_mode,
        owned=owned_conversation_ids(session) if session.scope.org_mode else frozenset(),
    )


def with_held_roots(session: ScopedSession, plan: RootsPlan) -> RootsPlan:
    """`plan` with the roots of its projects the held_artifact_roots rows
    name (held_artifact_roots.held_root_rows), for the artifacts list, and
    the projects whose roots their backfill could not read (the unread list
    of its sentinel, held_artifact_roots.read_sentinel), which it still
    walks. Unchanged for a desktop scan, a scope that keeps no rows, or
    before their backfill has written its sentinel. A database step: run it
    in a database unit."""
    from cowork.services.artifact_locations import keeps_locations
    from cowork.services.held_artifact_roots import held_root_rows, read_sentinel

    if plan.scan or not keeps_locations(session.scope):
        return plan
    # The one read of the sentinel, before the rows: `walk` comes from it.
    # Another replica's pass over the unread projects commits a project's
    # rows, then the sentinel without it: whenever it lands after this read,
    # the project is still in `walk` and found on the file store, never left
    # both unheld and unwalked.
    sentinel = read_sentinel(session)
    if sentinel is None:
        return plan
    project_ids = [project.id for project in plan.projects]
    held = held_root_rows(session, project_ids=project_ids)
    return replace(plan, held=held, walk=frozenset(sentinel.unread).intersection(project_ids))


def _parts_below(path: Path, anchor: Path) -> tuple[str, ...]:
    """``path.relative_to(anchor).parts``, compared by parts, case-sensitive:
    on POSIX paths the same answer, and the same ValueError for a path outside
    ``anchor``, without its walk over every parent, which a listing pays once
    per project. Every caller passes a base built under ``anchor``."""
    parts, anchor_parts = path.parts, anchor.parts
    if parts[: len(anchor_parts)] != anchor_parts:
        raise ValueError(f"{path} is not below {anchor}")
    return parts[len(anchor_parts) :]


def _project_source(project: ProjectRow, base: Path) -> ProjectArtifacts:
    """A root of `project`, reopened from the project's directory one component
    at a time."""
    project_path = Path(project.path)
    return ProjectArtifacts(
        base=base,
        project_id=str(project.id),
        project_name=project.name,
        trusted_anchor=project_path,
        root_parts=_parts_below(base, project_path),
        external=project.external,
    )


def recorded_source(project: ProjectRow, conversation_id: UUID | None) -> ProjectArtifacts:
    """The one root a location row names (cowork.services.artifact_locations):
    `project`'s shared root, or a conversation's legacy root. Built without
    listing anything; `project` is a row the caller may read, and
    `conversation_id` is a UUID, so it is one path component."""
    if conversation_id is None:
        return _project_source(project, _artifacts_base(project.path))
    base = (Path(project.path) / CONVERSATIONS_DIRNAME / str(UUID(str(conversation_id)))).joinpath(*_ARTIFACTS_SUBPATH)
    return _project_source(project, base)


def held_root_of(source: ProjectArtifacts) -> HeldRoot | None:
    """The root `source` is, as the rows name it: its project, and its
    conversation for a legacy root. None for a root no row describes: a
    desktop root, which has no project id, or a legacy root whose folder
    name is no conversation id."""
    from cowork.services.held_artifact_roots import HeldRoot

    if source.project_id is None:
        return None
    parts = tuple(source.root_parts)
    if parts == _ARTIFACTS_SUBPATH:
        return HeldRoot(UUID(source.project_id), None)
    if len(parts) == 4 and parts[0] == CONVERSATIONS_DIRNAME and parts[2:] == _ARTIFACTS_SUBPATH:
        conversation_id = _conversation_id(Path(parts[1]))
        return HeldRoot(UUID(source.project_id), conversation_id) if conversation_id is not None else None
    return None


def recorded_location(source: ProjectArtifacts, folder: Path) -> Location | None:
    """The location row for `folder`, found below `source`, or None for a root
    no row describes (held_root_of). A folder name that is not valid UTF-8
    still gets its location, so it counts toward a conflict;
    record_checked_folders never stores it in a row."""
    from cowork.services.artifact_locations import Location

    root = held_root_of(source)
    return Location(root.project_id, root.conversation_id, folder.name) if root is not None else None


def project_roots(
    project: ProjectRow, *, owned: frozenset[UUID] | None = None, strict: bool = False
) -> list[_FoundRoot]:
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

    `owned`, when given, keeps only the workspaces whose names it holds,
    checked by name before any stat, so another member's workspace is never
    statted, and an empty `owned` lists no conversations folder at all.

    A conversations directory or workspace that fails to read is left out,
    unless `strict`: then the error raises, for a caller that must know it
    saw every root (the location backfill).
    """
    project_path = Path(project.path)

    def root(base: Path, **found) -> _FoundRoot:
        return _FoundRoot(source=_project_source(project, base), **found)

    roots = [root(_artifacts_base(project.path))]
    if not _org_mode():
        return roots
    if owned is not None and not owned:
        # No workspace can pass the owner filter.
        return roots
    conversations = project_path / CONVERSATIONS_DIRNAME
    if not _is_real_directory(conversations, strict=strict):
        # No legacy per-conversation dir at all is common and not an error:
        # the project-level base is still a valid, primary root.
        return roots
    try:
        children = sorted(conversations.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return roots
    except OSError:
        if strict:
            raise
        return roots
    if owned is not None:
        # By name first, so a co-member's workspace is never statted.
        children = [child for child in children if _conversation_id(child) in owned]
    return roots + [
        root(child.joinpath(*_ARTIFACTS_SUBPATH), legacy=True, conversation_id=_conversation_id(child))
        for child in children
        if _is_real_directory(child, strict=strict)
        and _storage_components_are_safe(child, may_be_absent=False, strict=strict)
    ]


def _held_project_roots(
    project: ProjectRow, *, held: Collection[UUID | None], owned: frozenset[UUID] | None
) -> list[_FoundRoot]:
    """The roots of `project` that its held rows name, `held` being their
    conversations (None for the shared root), in `project_roots`' order: the
    shared root, then the legacy roots by conversation id. Opens nothing on
    the file store. `owned`, when given, keeps only the legacy roots of those
    conversations, as `project_roots` does by name."""
    roots = [_FoundRoot(source=recorded_source(project, None))] if None in held else []
    legacy = sorted(
        (conversation_id for conversation_id in held
         if conversation_id is not None and (owned is None or conversation_id in owned)),
        key=str,
    )
    return roots + [
        _FoundRoot(source=recorded_source(project, conversation_id), legacy=True, conversation_id=conversation_id)
        for conversation_id in legacy
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


def left_out_for_another_partition(
    project: ProjectRow, *, partition: str | None, shared_root: Path | None
) -> bool:
    """Whether `project` names a folder in another person's partition
    (`projects.sits_in_another_partition`), logged when it does: its
    artifacts are left out."""
    from cowork.services.projects import sits_in_another_partition

    if not sits_in_another_partition(project.path, partition=partition, shared_root=shared_root):
        return False
    logger.warning(
        "artifacts: a project names a folder in another person's partition, so its "
        "artifacts are left out",
        extra={
            "event": "artifact_folder_refused",
            "reason": "other_partition",
            "project_id": str(project.id),
        },
    )
    return True


def discover_roots(plan: RootsPlan, *, own_only: bool = False, strict: bool = False) -> DiscoveredRoots:
    """Expand `plan` on the file store. Opens no database session, so it can run
    on a file store thread (cowork.db.units.run_efs).

    With `own_only` and the plan's owner filter, another member's legacy
    workspaces are skipped by name before any stat (`project_roots`), for a
    caller that reads only `sources()`; `unfiltered()` then holds no
    co-member's root either. Without it, every workspace is found, for a
    lookup's scan, which reads `unfiltered()`.

    With `strict`, a project root that fails to list or stat raises
    (project_roots) instead of being left out. A plan with `held` lists and
    stats nothing for a project outside `walk`: its roots are the ones
    `held` names (_held_project_roots). A project in `walk` has its roots
    found on the file store, as without `held`."""
    from cowork.services.projects import resolved_shared_root

    owned = plan.owned if own_only and plan.owner_filter else None
    # Read once for every project the plan names, not once per project.
    shared_root = resolved_shared_root() if plan.partition is not None and plan.projects else None
    roots: list[_FoundRoot] = []
    held_by_project: dict[UUID, set[UUID | None]] = {}
    for project_id, conversation_id in plan.held or ():
        held_by_project.setdefault(project_id, set()).add(conversation_id)
    if plan.scan:
        roots += [_FoundRoot(source=source) for source in artifacts_sources_for_scan(scope=plan.scan_scope)]
        roots += [found for project in plan.adopted if (found := _adopted_root(project)) is not None]
    for project in plan.projects:
        if left_out_for_another_partition(project, partition=plan.partition, shared_root=shared_root):
            continue
        if plan.held is not None and project.id not in plan.walk:
            roots += _held_project_roots(project, held=held_by_project.get(project.id, set()), owned=owned)
        else:
            roots += project_roots(project, owned=owned, strict=strict)
    return DiscoveredRoots(roots=tuple(roots), owner_filter=plan.owner_filter, owned=plan.owned)


def readable_sources(plan: RootsPlan) -> list[ProjectArtifacts]:
    """The roots `plan` names that the caller may read. File store work only:
    the plan already holds the owner filter's answer, so async code runs this
    inside a file store call (cowork.db.units.run_efs), with the work on the
    roots it finds. Never stats another member's workspace."""
    return discover_roots(plan, own_only=True).sources()


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
