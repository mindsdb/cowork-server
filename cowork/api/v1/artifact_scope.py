"""Project-scoped artifact resolution shared by artifact HTTP endpoints."""
from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar
from uuid import UUID

from fastapi import HTTPException, status

from cowork.db.scoped import ScopedSession, TenantScope, tenant_scope_required
from cowork.db.units import run_db, run_efs
from cowork.services.artifact_identity import (
    FolderCheck,
    FolderInRoot,
    HeldFolder,
    artifact_folders_holding,
    folders_holding_strictly,
    resolve_recorded_folder,
)
from cowork.services.artifact_locations import (
    Location,
    LocationRow,
    find_location,
    keeps_locations,
    record_checked_folders,
)
from cowork.services.artifact_roots import (
    DiscoveredRoots,
    RootsPlan,
    artifacts_sources_for_project,
    artifacts_sources_for_scope,
    discover_roots,
    plan_project_roots,
    plan_scope_roots,
    recorded_location,
    recorded_source,
    scoped_project_row,
)
from cowork.services.artifacts import ProjectArtifacts, _org_mode

if TYPE_CHECKING:
    from cowork.services.projects import ProjectService

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _project_dir_matches(source, wanted: str) -> bool:
    project_dir = source.base.parent.parent
    try:
        resolved = str(project_dir.resolve(strict=False))
    except OSError:
        resolved = ""
    return wanted in (str(project_dir), resolved)


def artifact_sources_for_request(
    session,
    project_id: UUID | None,
    project_path: str | None,
):
    """Return only artifact roots authorized by this request's project scope.

    ``project_id`` is honored in both tenancy modes. Organization requests
    reject client filesystem paths; desktop paths only select from roots the
    server already discovered and never become filesystem inputs themselves.
    """
    if _org_mode() and project_path is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_path is not accepted in org deployments; use project_id",
        )
    if project_id is not None:
        # Even a typed FastAPI UUID originates at the request boundary. Recover
        # the equal UUID object from the scoped project catalog before any root
        # discovery so downstream paths depend only on server-owned records.
        project_id = scoped_project_id_for_request(session, str(project_id))
        try:
            return artifacts_sources_for_project(session, project_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unknown project",
            ) from exc
    sources = artifacts_sources_for_scope(session)
    if _org_mode():
        return sources
    if project_path is not None:
        wanted = os.path.normpath(os.path.expanduser(project_path))
        sources = [source for source in sources if _project_dir_matches(source, wanted)]
    return sources


def scoped_project_id_for_request(session: ScopedSession, project_ref: str) -> UUID:
    """Recover a request-selected project id from the scoped server catalog.

    Callers that will discover filesystem roots must use the returned database
    value, not the request spelling.  The client value is parsed and compared
    in memory only, so it cannot flow into path construction even though both
    values represent the same UUID.
    """
    try:
        wanted_project_id = UUID(project_ref).hex
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project",
        ) from exc

    from cowork.services.projects import ProjectService

    service = ProjectService(session)
    project_id = next(
        (
            project.id
            for project in service.list_projects()
            if secrets.compare_digest(project.id.hex, wanted_project_id)
        ),
        None,
    )
    if project_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown project",
        )
    _refuse_a_persons_project_in_the_operator_view(service, project_id)
    return project_id


def _refuse_a_persons_project_in_the_operator_view(service: ProjectService, project_id: UUID) -> None:
    """The operator view (no person, so LOCAL_SCOPE) lists every project row,
    but a card's serveUrl names its project by name alone, which the serve
    route looks up in the unkeyed projects root, and a delete by id would act
    on that person's partition root. A person's project, a row with an org_id,
    is refused the operator view so neither a list nor a delete reaches it.

    Placed here, in the one project-id resolver every artifact route shares, so
    the list, the delete by id, and the drafts/workspace/review routes all
    refuse it alike. Returns early on the desktop (no tenant scope required)
    and where the scope partitions (hosted, a person), so only the operator
    view pays the extra row read."""
    if not tenant_scope_required() or service.session.scope.partitioned:
        return
    if service.get_project(project_id).org_id is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "This project is a person's own; the operator view lists only "
                "the deployment's artifacts."
            ),
        )


def _plan_for_project_ref(session: ScopedSession, *, project_ref: str) -> RootsPlan:
    """The database half of root discovery for one request's project, run as
    a unit (cowork.db.units.run_db)."""
    if project_ref == "local":
        if _org_mode():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")
        return plan_scope_roots(session)
    # Recover the project id from a tenant-scoped database row before artifact
    # root discovery.  The request value is used only for an in-memory equality
    # check; the UUID passed into the filesystem-facing source resolver is the
    # value the server loaded from its own database.
    project_id = scoped_project_id_for_request(session, project_ref)
    try:
        return plan_project_roots(session, project_id)
    except ValueError as exc:
        # The row can disappear between the scoped inventory and the second
        # scoped read performed by artifact root discovery.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown project",
        ) from exc


def _canonical_or_400(artifact_id: str) -> str:
    from cowork.services.artifact_identity import canonical_artifact_id

    # Normalized here, at the edge of the service layer, and not only inside
    # `resolve_artifact_folder`: everything downstream — the location row, the
    # identity index, the revision journal, the draft preview — takes this
    # value, and every caller of this module should be able to see that the
    # string was rejected unless it parses as a UUID. Callers over HTTP are
    # already gated by the route's `UUID` path type; the desktop delete path
    # reaches here with its own `UUID(slug)` check, and a future caller gets
    # the same guarantee for free.
    try:
        return canonical_artifact_id(artifact_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact identity",
        ) from exc


class ResolvedArtifact(NamedTuple):
    """An artifact folder the caller may at least review.

    `is_own` false means the bytes sit in a co-member's private conversation
    workspace and the only reason resolution succeeded is the owner's
    draft-review grant on that artifact: the caller is a reviewer, and no
    route may let them mutate anything.
    """

    source: ProjectArtifacts
    folder: Path
    metadata: dict
    is_own: bool


class ReviewedRead(NamedTuple, Generic[T]):
    """An artifact the caller may at least review, and what a route's read of
    it returned."""

    artifact: ResolvedArtifact
    value: T


def _read_nothing(_artifact: ResolvedArtifact) -> None:
    return None


class _Lookup(NamedTuple):
    """The database half of one artifact lookup: the request's roots plan,
    and the artifact's location row, if any, with the one root that row
    names, whether the caller reads that root as their own, and whether the
    row's project is the one the request names."""

    plan: RootsPlan
    row: LocationRow | None = None
    source: ProjectArtifacts | None = None
    own: bool = False
    in_plan: bool = False

    @property
    def serves(self) -> bool:
        """Whether the row's folder may answer the request: it is in the
        request's project, and the id is not conflicted."""
        return self.in_plan and not self.row.conflicted


def _lookup_for_request(session: ScopedSession, *, project_ref: str, artifact_id: str) -> _Lookup:
    """Read the plan and the location row as one unit (cowork.db.units.run_db).

    The row is only a hint. It answers the request only when its project is
    the one the request names and the id is not conflicted. Its root is
    rebuilt from that project's row, and a legacy conversation root counts as
    the caller's own only when the conversation is theirs; otherwise the
    folder is a co-member's and needs the owner's review grant, the same as a
    scan that finds it. A row naming another project of the organization is
    opened too, never served, so the lookup learns whether its folder still
    holds the id."""
    plan = _plan_for_project_ref(session, project_ref=project_ref)
    row = find_location(session, artifact_id)
    if row is None:
        return _Lookup(plan)
    location = row.location
    project = next((p for p in plan.projects if p.id == location.project_id), None)
    in_plan = project is not None
    if project is None:
        project = scoped_project_row(session, project_id=location.project_id)
        if project is None:
            return _Lookup(plan, row)
    own = (
        location.conversation_id is None
        or not plan.owner_filter
        or location.conversation_id in plan.owned
    )
    return _Lookup(plan, row, recorded_source(project, location.conversation_id), own, in_plan)


class _Opened(NamedTuple):
    """What opening a location row's folder found, and the route's read when
    that folder answers the request."""

    check: FolderCheck
    reviewed: ReviewedRead | None = None


def _open_recorded(
    lookup: _Lookup, *, artifact_id: str, org_id: str | None, read: Callable[[ResolvedArtifact], T],
) -> _Opened:
    """The file store half of a lookup by its location row: one root, one
    folder, its metadata, then `read` when that folder answers the request.
    Runs on a file store thread (run_efs)."""
    from cowork.services.artifact_draft_review import draft_review_allows

    recorded = resolve_recorded_folder(lookup.source, lookup.row.location.slug, artifact_id)
    if recorded.check is not FolderCheck.HOLDS or not lookup.serves:
        return _Opened(recorded.check)
    if lookup.own:
        artifact = ResolvedArtifact(lookup.source, recorded.folder, recorded.metadata, True)
    elif draft_review_allows(recorded.folder, org_id):
        artifact = ResolvedArtifact(lookup.source, recorded.folder, recorded.metadata, False)
    else:
        # No grant on a co-member's folder: the scan decides, as it did
        # before rows existed, so a refusal reads as a missing artifact.
        return _Opened(recorded.check)
    return _Opened(recorded.check, ReviewedRead(artifact, read(artifact)))


class _Scanned(NamedTuple):
    """What a scan found: every folder of the plan's roots that holds the id,
    or None when a root or folder could not be read and nothing may be
    recorded, then the caller's answer, the route's read or the refusal to
    raise once those folders are recorded."""

    folders: tuple[Location, ...] | None
    reviewed: ReviewedRead | None = None
    refusal: HTTPException | None = None


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")


def _duplicated() -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Artifact identity is duplicated")


def _reviewable_among(
    matches: list[HeldFolder], *, own: list[ProjectArtifacts], org_id: str | None,
) -> ResolvedArtifact:
    """What a scan answers from `matches`, every folder of the plan's roots
    that holds the id: the one in the caller's own roots; else, in org mode,
    a co-member's behind the owner's review grant. Two folders in the roots
    that decide are a 409, and none the caller may read a 404."""
    mine = [match for match in matches if match.source in own]
    if len(mine) > 1:
        raise _duplicated()
    if mine:
        return ResolvedArtifact(mine[0].source, mine[0].folder, mine[0].metadata, True)
    if not _org_mode() or not matches:
        raise _not_found()
    if len(matches) > 1:
        raise _duplicated()

    from cowork.services.artifact_draft_review import draft_review_allows

    held = matches[0]
    if not draft_review_allows(held.folder, org_id):
        raise _not_found()
    return ResolvedArtifact(held.source, held.folder, held.metadata, False)


def _scan_found(
    found: DiscoveredRoots,
    *,
    artifact_id: str,
    org_id: str | None,
    read: Callable[[ResolvedArtifact], T],
) -> _Scanned:
    """A scan of the discovered roots: every folder there that holds the id,
    a co-member's legacy roots included, the caller's answer from those, then
    `read` on it."""
    matches = artifact_folders_holding(found.unfiltered(), artifact_id=artifact_id)
    folders = _locations(matches)
    try:
        artifact = _reviewable_among(matches, own=found.sources(), org_id=org_id)
    except HTTPException as refusal:
        return _Scanned(folders, refusal=refusal)
    return _Scanned(folders, reviewed=ReviewedRead(artifact, read(artifact)))


def _locations(held: list[HeldFolder] | list[FolderInRoot]) -> tuple[Location, ...]:
    """The location rows for folders found below the lookup's roots, without
    the folders of roots no row describes."""
    return tuple(
        location for item in held
        if (location := recorded_location(item.source, item.folder)) is not None
    )


def _strictly_held(plan: RootsPlan, *, artifact_id: str) -> tuple[Location, ...] | None:
    """The folders holding the id, found again past every cache before a lone
    folder gets an unconflicted row. The scan's discovery leaves out a root
    that fails to list or stat, and its identity cache a folder that fails to
    read, so a second folder holding the id can be missing from the scan,
    and the row would then serve one of two folders. None when a root or
    folder fails to read again: the lookup records nothing, and the next one
    scans."""
    try:
        roots = discover_roots(plan, strict=True).unfiltered()
        return _locations(folders_holding_strictly(roots, artifact_id=artifact_id))
    except OSError:
        return None


def _replaced_row(lookup: _Lookup, opened: _Opened | None) -> Location | None:
    """The row a scan may move or drop: one whose folder is stale, and only
    for a caller who reads the row's root as their own."""
    if opened is not None and opened.check is FolderCheck.STALE and lookup.own:
        return lookup.row.location
    return None


class _Found(NamedTuple):
    """What a lookup's file store call found: what opening the row's folder
    found, when the lookup had a row to open, and the scan, which ran unless
    that folder answered the request (`opened.reviewed`)."""

    opened: _Opened | None
    scanned: _Scanned | None = None


def _open_or_scan(
    lookup: _Lookup,
    *,
    artifact_id: str,
    org_id: str | None,
    read: Callable[[ResolvedArtifact], T],
    keeps_rows: bool,
) -> _Found:
    """The file store half of a lookup: the row's folder (_open_recorded),
    then, unless it answered, the plan's roots (_scan_found). When rows are
    kept and the scan found one folder that would get a fresh unconflicted
    row, because the id has no row or the caller replaces a stale one, the
    folders are counted again strictly (_strictly_held). Runs on a file store
    thread (run_efs), so the lookup waits for a file store slot once."""
    opened = None
    if lookup.source is not None:
        opened = _open_recorded(lookup, artifact_id=artifact_id, org_id=org_id, read=read)
        if opened.reviewed is not None:
            return _Found(opened)
    scanned = _scan_found(discover_roots(lookup.plan), artifact_id=artifact_id, org_id=org_id, read=read)
    writes_a_fresh_row = keeps_rows and (lookup.row is None or _replaced_row(lookup, opened) is not None)
    if writes_a_fresh_row and len(scanned.folders) == 1:
        scanned = scanned._replace(folders=_strictly_held(lookup.plan, artifact_id=artifact_id))
    return _Found(opened, scanned)


async def _record(
    artifact_id: str, *, folders: tuple[Location, ...], replaces: Location | None, scope: TenantScope,
) -> None:
    """Record the folders a scan found holding the artifact
    (record_checked_folders). Best-effort: the request already has its
    answer, and the next lookup scans again."""
    try:
        await run_db(
            partial(record_checked_folders, artifact_id=artifact_id, folders=folders, replaces=replaces),
            scope=scope,
        )
    except Exception:
        logger.warning("could not record an artifact's location", exc_info=True)


async def _review(
    scope: TenantScope, *, project_ref: str, artifact_id: str, read: Callable[[ResolvedArtifact], T],
) -> ReviewedRead[T]:
    """The lookup behind review_artifact_for_request and
    review_artifact_and_read: the location row's folder when the row answers
    the request, else a scan of the project's roots, whose folders are then
    recorded unless they leave the row as it is."""
    artifact_id = _canonical_or_400(artifact_id)
    lookup = await run_db(
        partial(_lookup_for_request, project_ref=project_ref, artifact_id=artifact_id), scope=scope
    )
    keeps_rows = keeps_locations(scope)
    found = await run_efs(
        partial(
            _open_or_scan, lookup, artifact_id=artifact_id, org_id=scope.org_id, read=read, keeps_rows=keeps_rows,
        ),
        scope=scope,
    )
    opened, scanned = found
    if scanned is None:
        return opened.reviewed
    seen: tuple[Location, ...] = ()
    if opened is not None and opened.check is FolderCheck.HOLDS and lookup.in_plan:
        # A folder of this project holds the id, whether or not the caller
        # may be served it, so the scan's count includes it.
        seen = (lookup.row.location,)
    replaces = _replaced_row(lookup, opened)
    folders = tuple(dict.fromkeys(seen + (scanned.folders or ())))
    # With no stale row to replace, a scan that found no folder, or only the
    # one the row names, leaves the row as it is. So does a scan that could
    # not read every folder.
    unchanged = scanned.folders is None or (
        replaces is None
        and (not folders or (lookup.row is not None and folders == (lookup.row.location,)))
    )
    if keeps_rows and not unchanged:
        await _record(artifact_id, folders=folders, replaces=replaces, scope=scope)
    if scanned.refusal is not None:
        raise scanned.refusal
    return scanned.reviewed


async def review_artifact_for_request(
    scope: TenantScope, project_ref: str, artifact_id: str
) -> ResolvedArtifact:
    """Resolve an artifact the caller may at least review, and say which it is.

    An artifact without a grant is a 404 rather than a 403: the private case
    must not be distinguishable from a missing artifact.

    In org tenancy the artifact's location row names its folder, and the
    lookup opens only that one: the caller's own, or a co-member's behind the
    owner's review grant. Without a row that answers, or when its folder no
    longer holds the artifact, the lookup lists the project's roots: the
    caller's own folder answers, else a co-member's behind the grant, and two
    folders in the roots that decide are a 409. That scan reads every folder
    of the project holding the id and records them
    (cowork.services.artifact_locations.record_checked_folders): a row names
    the id's folder only when one folder holds it, and an id two folders hold
    is marked conflicted, so its lookups keep listing the roots.

    One database unit reads the project row, the conversations the caller
    owns and the location row (run_db), then one file store call opens the
    row's folder or lists the roots (run_efs): no pooled connection is held
    while a stat waits on the file server, the event loop never waits for
    either, and the lookup waits for a file store slot once. A scan whose
    folders can change the row records them in another database unit after
    that call.
    """
    return (await _review(scope, project_ref=project_ref, artifact_id=artifact_id, read=_read_nothing)).artifact


async def review_artifact_and_read(
    scope: TenantScope,
    project_ref: str,
    artifact_id: str,
    *,
    read: Callable[[ResolvedArtifact], T],
) -> ReviewedRead[T]:
    """`review_artifact_for_request`, with the route's own blocking ``read`` of
    the artifact run in the lookup's file store call, so the route waits for a
    file store slot once, and its read is never refused a slot after the
    lookup's work. A database unit the route runs after this, such as the
    review entry's capabilities read, can still answer 503 when no connection
    frees within POOL_TIMEOUT.

    Only for a route whose authorization ends with the review: ``read`` runs
    for a reviewer as well as the owner, so it may read only what a reviewer
    may see. An owner-only route checks ownership in a database unit after
    the lookup, and reads in a file store call of its own after that check.
    """
    return await _review(scope, project_ref=project_ref, artifact_id=artifact_id, read=read)
