"""Project-scoped artifact resolution shared by artifact HTTP endpoints."""
from __future__ import annotations

import os
import secrets
from uuid import UUID

from fastapi import HTTPException, status

from cowork.db.scoped import ScopedSession
from cowork.services.artifact_roots import (
    artifacts_sources_for_project,
    artifacts_sources_for_scan,
    artifacts_sources_for_scope,
)
from cowork.services.artifacts import _org_mode


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
    if _org_mode():
        return artifacts_sources_for_scope(session)

    sources = artifacts_sources_for_scan()
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

    project_id = next(
        (
            project.id
            for project in ProjectService(session).list_projects()
            if secrets.compare_digest(project.id.hex, wanted_project_id)
        ),
        None,
    )
    if project_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown project",
        )
    return project_id


def _sources_for_project_ref(
    session: ScopedSession, project_ref: str, *, include_other_members: bool = False
):
    if project_ref == "local":
        if _org_mode():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")
        return artifacts_sources_for_scan()
    # Recover the project id from a tenant-scoped database row before artifact
    # root discovery.  The request value is used only for an in-memory equality
    # check; the UUID passed into the filesystem-facing source resolver is the
    # value the server loaded from its own database.
    project_id = scoped_project_id_for_request(session, project_ref)
    try:
        return artifacts_sources_for_project(
            session,
            project_id,
            include_other_members=include_other_members,
        )
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


def workspace_artifact_for_request(session: ScopedSession, project_ref: str, artifact_id: str):
    """Resolve an artifact identity only inside roots authorized for the caller."""
    return _resolve_in(_sources_for_project_ref(session, project_ref), artifact_id)


def review_artifact_for_request(session: ScopedSession, project_ref: str, artifact_id: str):
    """Resolve an artifact the caller may at least review, and say which it is.

    Returns `(source, folder, metadata, is_own)`. `is_own` false means the bytes
    sit in a co-member's private conversation workspace and the only reason this
    call succeeded is the owner's draft-review grant on that artifact — the
    caller is a reviewer and no route may let them mutate anything.

    Own roots are tried first, so an owner never pays for the wider scan, and an
    artifact without a grant is a 404 rather than a 403: the private case must
    not be distinguishable from a missing artifact.
    """
    try:
        source, folder, metadata = workspace_artifact_for_request(
            session, project_ref, artifact_id
        )
        return source, folder, metadata, True
    except HTTPException as exc:
        if exc.status_code != status.HTTP_404_NOT_FOUND or not _org_mode():
            raise

    from cowork.services.artifact_draft_review import draft_review_allows

    source, folder, metadata = _resolve_in(
        _sources_for_project_ref(session, project_ref, include_other_members=True),
        artifact_id,
    )
    scope = getattr(session, "scope", None)
    if not draft_review_allows(folder, getattr(scope, "org_id", None)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return source, folder, metadata, False
