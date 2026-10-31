"""Project-scoped artifact resolution shared by artifact HTTP endpoints."""
from __future__ import annotations

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
from cowork.services.artifact_roots import (
    RootsPlan,
    artifacts_sources_for_project,
    artifacts_sources_for_scope,
    discover_roots,
    plan_project_roots,
    plan_scope_roots,
)
from cowork.services.artifacts import ProjectArtifacts, _org_mode

if TYPE_CHECKING:
    from cowork.services.projects import ProjectService

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


def _resolve_in(sources, artifact_id: str):
    from cowork.services.artifact_identity import (
        ArtifactIdentityConflict,
        canonical_artifact_id,
        resolve_artifact_folder,
    )

    # Normalized here, at the edge of the service layer, and not only inside
    # `resolve_artifact_folder`: everything downstream — the identity index, the
    # revision journal, the draft preview — takes this value, and every caller
    # of this module should be able to see that the string was rejected unless
    # it parses as a UUID. Callers over HTTP are already gated by the route's
    # `UUID` path type; the desktop delete path reaches here with its own
    # `UUID(slug)` check, and a future caller gets the same guarantee for free.
    try:
        artifact_id = canonical_artifact_id(artifact_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact identity",
        ) from exc

    try:
        return resolve_artifact_folder(sources, artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ArtifactIdentityConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
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


def _reviewable_in(plan: RootsPlan, artifact_id: str, *, org_id: str | None) -> ResolvedArtifact:
    """The file store half of `review_artifact_for_request`: the roots, the
    owner filter, the folder and, for a reviewer, the owner's grant, all on
    the calling thread."""
    found = discover_roots(plan)
    try:
        return ResolvedArtifact(*_resolve_in(found.sources(), artifact_id), True)
    except HTTPException as exc:
        if exc.status_code != status.HTTP_404_NOT_FOUND or not _org_mode():
            raise

    from cowork.services.artifact_draft_review import draft_review_allows

    source, folder, metadata = _resolve_in(found.unfiltered(), artifact_id)
    if not draft_review_allows(folder, org_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return ResolvedArtifact(source, folder, metadata, False)


async def review_artifact_for_request(
    scope: TenantScope, project_ref: str, artifact_id: str
) -> ResolvedArtifact:
    """Resolve an artifact the caller may at least review, and say which it is.

    Own roots are tried first, so an owner never pays for the wider scan, and an
    artifact without a grant is a 404 rather than a 403: the private case must
    not be distinguishable from a missing artifact.

    One database unit reads the project row and the conversations the caller
    owns (run_db), then one file store call does the rest (run_efs): no pooled
    connection is held while a stat waits on the file server, the event loop
    never waits for either, and the lookup waits for a file store slot once.
    """
    plan = await run_db(partial(_plan_for_project_ref, project_ref=project_ref), scope=scope)
    return await run_efs(partial(_reviewable_in, plan, artifact_id, org_id=scope.org_id), scope=scope)


class ReviewedRead(NamedTuple, Generic[T]):
    """An artifact the caller may at least review, and what a route's read of
    it returned."""

    artifact: ResolvedArtifact
    value: T


def _reviewable_and_read(
    plan: RootsPlan,
    artifact_id: str,
    *,
    org_id: str | None,
    read: Callable[[ResolvedArtifact], T],
) -> ReviewedRead[T]:
    """`_reviewable_in`, then `read` on the artifact it found, on the calling
    thread. `read` runs only once `_reviewable_in` returned, so never for a
    caller it refused."""
    artifact = _reviewable_in(plan, artifact_id, org_id=org_id)
    return ReviewedRead(artifact, read(artifact))


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
    plan = await run_db(partial(_plan_for_project_ref, project_ref=project_ref), scope=scope)
    return await run_efs(
        partial(_reviewable_and_read, plan, artifact_id, org_id=scope.org_id, read=read),
        scope=scope,
    )
