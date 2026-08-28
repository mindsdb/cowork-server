"""Project-scoped artifact resolution shared by artifact HTTP endpoints."""
from __future__ import annotations

import os
import re
from uuid import UUID

from fastapi import HTTPException, status

#: The only shape an artifact identity may have by the time it selects a folder:
#: 32 lowercase hex characters, anchored. `canonical_artifact_id` already
#: guarantees it (it returns `UUID(value).hex`), so this is a second, explicit
#: statement of the same rule rather than a new one — worth its line because
#: this is the point where a request-supplied string starts choosing a path, and
#: an allowlist is checkable at a glance by a reader and by a static analyzer.
_CANONICAL_ARTIFACT_ID = re.compile(r"\A[0-9a-f]{32}\Z")

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


def _sources_for_project_ref(
    session: ScopedSession, project_ref: str, *, include_other_members: bool = False
):
    if project_ref == "local":
        if _org_mode():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")
        return artifacts_sources_for_scan()
    try:
        project_id = UUID(project_ref)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project") from exc
    if include_other_members:
        try:
            return artifacts_sources_for_project(session, project_id, include_other_members=True)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unknown project",
            ) from exc
    return artifact_sources_for_request(session, project_id, None)


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
    if not _CANONICAL_ARTIFACT_ID.fullmatch(artifact_id):
        # Unreachable through `canonical_artifact_id` above; kept so the
        # allowlist, not a UUID constructor's side effect, is what the code says
        # about which strings may reach the resolver.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact identity",
        )

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
