"""Artifacts API endpoints.

Ported from cowork/server/routes/artifacts.py. Provides listing,
preview, iframe mount, open-in-OS, and reveal-in-finder for
agent-produced artifacts.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import re
import secrets
import stat as stat_module
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session

from cowork.services.product_permissions import require_product_permission
from cowork.db.scoped import (
    ScopedSession,
    ScopedSessionDep,
    TenantScope,
    get_scoped_session,
    get_tenant_scope,
)
from cowork.db.session import get_session
from cowork.principal import Principal, get_principal
from cowork.api.v1.permissions import AuthenticatedInOrgMode, DesktopOnly, OpenByDesign, require
from cowork.api.v1.artifact_preview import (
    artifact_response_headers,
    html_preview_response,
    wants_comment_layer,
    wants_html_preview,
)
from cowork.api.v1.artifact_scope import (
    artifact_sources_for_request,
    scoped_project_id_for_request,
)
from cowork.common.logger import log_context
from cowork.common.paths import dir_scandir, dir_stat, open_pinned_child
from cowork.api.v1.endpoints.live_apps import LiveAppRoute
from cowork.schemas.live_apps import (
    LiveAppCopyRef,
    LiveAppDuplicateRequest,
    LiveAppShareRequest,
    LiveAppShareResponse,
    SharedLiveAppResponse,
)
from cowork.services.artifact_roots import (
    artifacts_sources_for_project,
    artifacts_sources_for_scan,
)
from cowork.services.artifacts import (
    BackendPreviewRefused,
    ExecutionRefused,
    _org_mode,
    _project_artifacts_base,
    artifact_id_for_folder as _artifact_id_for_folder,
    delete_artifact as _delete_artifact,
    delete_artifact_from_source as _delete_artifact_from_source,
    list_artifacts as _list_artifacts,
    mount_preview,
    preview_artifact as _preview_artifact,
    preview_mount_for,
    resolve_artifact_path,
    reveal_in_file_manager,
)
from cowork.services.projects import ProjectService

if TYPE_CHECKING:
    from cowork.services.live_apps import CopyOfShare

logger = logging.getLogger(__name__)

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


class _PathBody(BaseModel):
    path: str


@dataclass(frozen=True)
class _ArtifactDeleteRef:
    """A canonical identity or a validated legacy folder name."""

    artifact_id: str | None = None
    legacy_slug: str | None = None


_LEGACY_SLUG = re.compile(r"[^/\\\x00]{1,255}\Z")


def _artifact_delete_ref(value: str) -> _ArtifactDeleteRef:
    """Validate the polymorphic ``/{slug}`` compatibility parameter.

    Current clients send an artifact UUID. Older desktop clients send the
    artifact folder's direct-child name, so that form cannot be narrowed to a
    UUID without breaking them. It can still be made a genuine single segment:
    separators, NUL, dot segments and overlong filesystem names are refused
    before any artifacts directory is inspected.
    """
    try:
        return _ArtifactDeleteRef(artifact_id=UUID(value).hex)
    except (ValueError, TypeError, AttributeError):
        pass
    if value in {".", ".."} or _LEGACY_SLUG.fullmatch(value) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact identity",
        )
    return _ArtifactDeleteRef(legacy_slug=value)


def _artifact_folder_name(source, folder: Path) -> str:
    """Reduce an identity result to one direct child of its authorized root."""
    try:
        relative = folder.relative_to(Path(source.base))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found",
        ) from exc
    if len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found",
        )
    return relative.name


def _legacy_artifact_for_sources(sources, folder_name: str):
    """Find an exact, real directory entry below an authorized pinned root."""
    from cowork.services.artifact_identity import opened_artifact_root

    for source in sources:
        try:
            with opened_artifact_root(source) as root:
                with dir_scandir(root) as entries:
                    for entry in entries:
                        if entry.name != folder_name:
                            continue
                        try:
                            if entry.is_symlink() or not entry.is_dir(
                                follow_symlinks=False
                            ):
                                break
                        except OSError:
                            break
                        return source, root.path / entry.name
        except (OSError, ValueError):
            continue
    return None, None


# ``no-cache`` mirrors the FileResponse headers used elsewhere so a rebuilt
# artifact is always re-fetched.
# The two functions below hold the logic; the routes under them are thin adapters.
# Keeping them separate means the tenancy behavior is testable without FastAPI's
# Query sentinels standing in for the defaults.


def _artifact_cards(session, sources) -> list[dict]:
    # Owner-side fields are stripped inside `card_for_folder`, not here: inline chat
    # cards use the same builder and would otherwise still carry the plaintext
    # password.
    cards = _list_artifacts(sources)
    from cowork.services.artifact_export import card_export_source
    from cowork.services.artifact_ownership import resolve_artifact_owners
    from cowork.services.artifact_permissions import artifact_capabilities

    # Capabilities are a property of each ARTIFACT: in org mode a project root is
    # shared by every member (ENG-2056), so each card's owner is resolved on its
    # own. Owners are read in one query per root (the unfiltered listing spans
    # every project of the organization, so one query per project root).
    by_base = {Path(source.base): source for source in sources}
    cards_by_base: dict[Path, list[tuple[str, dict]]] = {}
    for card in cards:
        folder = Path(str(card.get("folder") or ""))
        if folder.parent in by_base:
            cards_by_base.setdefault(folder.parent, []).append((folder.name, card))
    for base, entries in cards_by_base.items():
        source = by_base[base]
        resolutions = resolve_artifact_owners(session, source, [slug for slug, _ in entries])
        for slug, card in entries:
            card["capabilities"] = artifact_capabilities(
                session,
                source,
                slug,
                resolution=resolutions[slug],
                export_source=card_export_source(card),
            )
    _mark_received_live_apps(session, cards_by_base, by_base, cards)
    return cards


def _mark_received_live_apps(session, cards_by_base, by_base, cards: list[dict]) -> None:
    """With live apps on, give each card ``receivedFrom``: the owner's email
    for a copy the caller duplicated from a share, from the server's own
    record of the copy (cowork.services.live_apps.received_from), else None."""
    from cowork.code_sandbox import live_apps_on

    scope = getattr(session, "scope", None)
    if not live_apps_on() or scope is None or not scope.personal:
        return
    from cowork.services.live_apps import received_from

    received = received_from(session, partition_id=scope.org_id)
    for card in cards:
        card["receivedFrom"] = None
    for base, entries in cards_by_base.items():
        project_id = by_base[base].project_id
        if not project_id:
            continue
        for slug, card in entries:
            owner_email = received.get((UUID(str(project_id)), slug))
            if owner_email:
                card["receivedFrom"] = {"email": owner_email}


def _scoped_project_sources(session, project_id: UUID | str):
    """Select roots with a server-loaded project identity, never the request.

    ``basename`` is intentionally applied even though FastAPI already typed the
    route value as a UUID. Snyk recognizes this standard path-leaf operation as
    a sanitizer, while it does not model FastAPI's coercion or the server-catalog
    equality boundary below. Equality with the original preserves UUID semantics
    instead of silently accepting a path whose last component happens to be one.

    ``scoped_project_id_for_request`` first enumerates the tenant-scoped project
    catalog and uses the client UUID only as an in-memory equality key. Artifact
    root discovery then receives the matching database value. Keeping this
    boundary beside the two filesystem consumers below makes the dataflow
    explicit to both reviewers and static analysis.
    """
    raw_project_ref = str(project_id)
    project_ref = os.path.basename(raw_project_ref)
    if project_ref != raw_project_ref:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project",
        )
    server_project_id = scoped_project_id_for_request(session, project_ref)
    try:
        sources = artifacts_sources_for_project(session, server_project_id)
    except ValueError as exc:
        # The project can disappear between catalog recovery and root discovery.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown project",
        ) from exc
    return server_project_id, sources


def _scoped_project_cards(session, project_ref: str) -> list[dict]:
    """Build cards for one basename-sanitized project selector.

    HTTP adapters sanitize the selector before entering this filesystem-facing
    helper. ``_scoped_project_sources`` then recovers the equal UUID object from
    the scoped server catalog, so root discovery still receives server-owned
    identity rather than the request string.
    """
    server_project_id, sources = _scoped_project_sources(session, project_ref)
    return _artifact_cards(session, sources)


def _all_artifact_cards(session) -> list[dict]:
    """Build the unfiltered scoped list without accepting any HTTP selector."""
    sources = artifact_sources_for_request(session, None, None)
    return _artifact_cards(session, sources)


def artifacts_for_request(
    session, *, project_id: UUID | None = None, project_path: str | None = None
) -> list[dict]:
    # Preserve the service's existing precedence: org mode rejects a filesystem
    # path even when a project UUID is also present; desktop prefers that UUID.
    if project_path is not None and _org_mode():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_path is not accepted in org deployments; use project_id",
        )
    if project_id is not None:
        raw_project_ref = str(project_id)
        project_ref = os.path.basename(raw_project_ref)
        if project_ref != raw_project_ref:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid project",
            )
        return _scoped_project_cards(session, project_ref)
    if project_path is not None:
        return _desktop_artifacts_for_project_path(session, project_path)
    return _all_artifact_cards(session)


def _desktop_source_path_keys(source) -> set[str]:
    """Exact legacy path spellings for one server-discovered desktop source."""
    project_dir = Path(source.base).parent.parent
    candidates = {os.path.normpath(str(project_dir))}
    try:
        candidates.add(os.path.normpath(str(project_dir.resolve(strict=False))))
    except OSError:
        pass
    return candidates


@dataclass(frozen=True)
class _DesktopSourceCatalogEntry:
    """One server-discovered source and its accepted legacy path spelling."""

    path_key: str
    project_name: str
    source: object


def _desktop_registered_path_catalog(
    *, scope: TenantScope | None
) -> tuple[_DesktopSourceCatalogEntry, ...]:
    """Build the complete legacy source catalog of ``scope``'s projects
    without an HTTP selector."""
    return tuple(
        _DesktopSourceCatalogEntry(
            path_key=path_key,
            project_name=os.path.basename(str(Path(source.base).parent.parent)),
            source=source,
        )
        for source in artifacts_sources_for_scan(scope=scope)
        for path_key in sorted(_desktop_source_path_keys(source))
    )


def _desktop_registered_source(
    catalog: tuple[_DesktopSourceCatalogEntry, ...],
    requested: str,
    project_name: str,
):
    """Select the exact server source for a registered legacy path.

    This selector performs comparisons only: all Path operations happened in
    ``_desktop_registered_path_catalog`` before the HTTP path entered the
    operation. Returning the catalog's source, rather than rescanning by leaf
    name, preserves exact-path semantics when two roots share a basename.
    """
    for entry in catalog:
        if secrets.compare_digest(
            entry.path_key, requested
        ) and secrets.compare_digest(entry.project_name, project_name):
            return entry.source
    return None


def _desktop_artifacts_for_project_path(session, project_path: str) -> list[dict]:
    """Select one desktop project's response by its legacy path.

    ``project_path`` remains supported because older desktop builds address the
    list this way. A complete source catalog is built first, then the request is
    used only as an equality key. It is never joined or opened. The exact source
    retained in that catalog keeps ``project_id=None``, preserving local draft
    URLs and card addressing.
    """
    catalog = _desktop_registered_path_catalog(scope=session.scope)
    if not project_path or "\x00" in project_path:
        return []
    requested = os.path.normpath(os.path.expanduser(project_path))
    project_name = os.path.basename(requested)
    if not project_name:
        return []
    source = _desktop_registered_source(catalog, requested, project_name)
    if source is None:
        return []
    return _artifact_cards(session, [source])


_BLANK_ARTIFACT_STATUS = {
    "publishedUrl": "",
    "modified": False,
    "accessMode": "public",
    "accessProtected": False,
    "accessEmails": [],
    "orgAllowed": False,
}


def _status_from_card(card: dict) -> dict:
    """The non-secret status subset exposed by the focus-refresh endpoint."""
    return {
        "publishedUrl": card.get("publishedUrl", ""),
        "modified": bool(card.get("modified")),
        "accessMode": card.get("accessMode", "public"),
        "accessProtected": bool(card.get("accessProtected")),
        "accessEmails": card.get("accessEmails", []),
        "orgAllowed": bool(card.get("orgAllowed")),
        "artifactKey": card.get("artifactKey", ""),
    }


def _normalized_lookup_key(value: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.expanduser(value)))


def _server_path_keys(candidate: Path, source_base: Path) -> tuple[set[str], str | None]:
    """Lookup spellings derived exclusively from a server-discovered path."""
    absolute = {_normalized_lookup_key(str(candidate))}
    try:
        absolute.add(_normalized_lookup_key(str(candidate.resolve(strict=False))))
    except OSError:
        pass
    # macOS exposes temporary/project trees through both /var and /private/var.
    for key in tuple(absolute):
        private_prefix = f"{os.sep}private{os.sep}"
        if key.startswith(private_prefix):
            absolute.add(key[len(f"{os.sep}private"):])
    try:
        relative = _normalized_lookup_key(str(candidate.relative_to(source_base)))
    except ValueError:
        relative = None
    return absolute, relative


def _matches_server_path(
    candidate: Path,
    source_base: Path,
    requested: str,
    relative_requested: str,
) -> bool:
    absolute, relative = _server_path_keys(candidate, source_base)
    return requested in absolute or (
        relative is not None and relative_requested == relative
    )


def _pinned_artifact_match(
    directory,
    logical_folder: Path,
    source_base: Path,
    requested: str,
    relative_requested: str,
    *,
    allow_directory: bool = True,
) -> Path | None:
    """Return an exactly matched real file (or the artifact root) by scanning."""
    if allow_directory and _matches_server_path(
        logical_folder, source_base, requested, relative_requested
    ):
        return logical_folder
    try:
        with dir_scandir(directory) as entries:
            discovered = sorted(entries, key=lambda entry: entry.name)
    except OSError:
        return None
    for entry in discovered:
        try:
            mode = entry.stat(follow_symlinks=False).st_mode
        except OSError:
            continue
        logical = logical_folder / entry.name
        if stat_module.S_ISREG(mode):
            if _matches_server_path(
                logical, source_base, requested, relative_requested
            ):
                return logical
            continue
        if not stat_module.S_ISDIR(mode):
            continue
        try:
            child = open_pinned_child(directory, entry.name)
        except OSError:
            continue
        try:
            matched = _pinned_artifact_match(
                child,
                logical,
                source_base,
                requested,
                relative_requested,
                allow_directory=False,
            )
        finally:
            child.close()
        if matched is not None:
            return matched
    return None


def _pinned_loose_status(directory, logical_file: Path) -> dict:
    """Status for a metadata-less legacy file using its pinned parent record."""
    from cowork.services.artifacts import (
        _load_published_map_pinned,
        _published_access_for,
        _published_url_for,
    )

    published_map = _load_published_map_pinned(directory)
    return _status_from_card(
        {
            "publishedUrl": _published_url_for(
                directory.path,
                logical_file,
                published_map=published_map,
            ),
            "modified": False,
            **_published_access_for(
                directory.path,
                logical_file,
                published_map=published_map,
            ),
        }
    )


def _pinned_loose_match(
    directory,
    logical_directory: Path,
    source_base: Path,
    requested: str,
    relative_requested: str,
) -> dict | None:
    """Find an exact real file below a metadata-less legacy directory."""
    try:
        with dir_scandir(directory) as entries:
            discovered = sorted(entries, key=lambda entry: entry.name)
    except OSError:
        return None
    for entry in discovered:
        try:
            mode = entry.stat(follow_symlinks=False).st_mode
        except OSError:
            continue
        logical = logical_directory / entry.name
        if stat_module.S_ISREG(mode):
            if _matches_server_path(
                logical, source_base, requested, relative_requested
            ):
                return _pinned_loose_status(directory, logical)
            continue
        if not stat_module.S_ISDIR(mode):
            continue
        try:
            child = open_pinned_child(directory, entry.name)
        except OSError:
            continue
        try:
            result = _pinned_loose_match(
                child,
                logical,
                source_base,
                requested,
                relative_requested,
            )
        finally:
            child.close()
        if result is not None:
            return result
    return None


def _desktop_artifact_status_for_path(path: str, *, scope: TenantScope | None) -> dict:
    """Match a legacy path only against entries scanned from ``scope``'s roots.

    The request is reduced to a string lookup key. Every card and folder key is
    built first from server-discovered ``ProjectArtifacts`` roots, so no
    request-derived value is ever opened, joined to a root, or passed to a
    path-oriented status helper. Only the one exactly matched artifact is then
    built into a card, preserving legacy loose-file status without eagerly
    hashing every artifact on each window-focus poll.
    """
    if not path or "\x00" in path:
        return dict(_BLANK_ARTIFACT_STATUS)
    requested = _normalized_lookup_key(path)
    relative_requested = requested
    artifacts_prefix = f"artifacts{os.sep}"
    if relative_requested.startswith(artifacts_prefix):
        relative_requested = relative_requested[len(artifacts_prefix):]

    from cowork.services.artifact_identity import (
        _opened_child_directory,
        opened_artifact_root,
    )
    from cowork.services.artifacts import card_for_folder

    for source in artifacts_sources_for_scan(scope=scope):
        source_base = Path(source.base)
        try:
            with opened_artifact_root(source) as root:
                with dir_scandir(root) as entries:
                    discovered = sorted(entries, key=lambda entry: entry.name)
                for entry in discovered:
                    try:
                        mode = entry.stat(follow_symlinks=False).st_mode
                    except OSError:
                        continue
                    logical = root.path / entry.name
                    if stat_module.S_ISREG(mode):
                        if _matches_server_path(
                            logical,
                            source_base,
                            requested,
                            relative_requested,
                        ):
                            return _pinned_loose_status(root, logical)
                        continue
                    if not stat_module.S_ISDIR(mode):
                        continue
                    try:
                        with _opened_child_directory(root, entry.name) as folder:
                            try:
                                metadata_stat = dir_stat(
                                    folder,
                                    "metadata.json",
                                    follow_symlinks=False,
                                )
                                is_artifact = stat_module.S_ISREG(
                                    metadata_stat.st_mode
                                )
                            except OSError:
                                is_artifact = False
                            if not is_artifact:
                                loose = _pinned_loose_match(
                                    folder,
                                    logical,
                                    source_base,
                                    requested,
                                    relative_requested,
                                )
                                if loose is not None:
                                    return loose
                                continue
                            matched = _pinned_artifact_match(
                                folder,
                                logical,
                                source_base,
                                requested,
                                relative_requested,
                            )
                            if matched is None:
                                continue
                            card = card_for_folder(
                                logical,
                                scope=scope,
                                project_id=source.project_id,
                                project_name=source.project_name,
                                _pinned_folder=folder,
                                _pinned_root=root,
                            )
                            if card is not None:
                                return _status_from_card(card)
                            if matched != logical:
                                return _pinned_loose_status(folder, matched)
                            return dict(_BLANK_ARTIFACT_STATUS)
                    except (OSError, ValueError):
                        continue
        except (OSError, ValueError):
            continue
    return dict(_BLANK_ARTIFACT_STATUS)


# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope.
@router.get("/", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def list_artifacts(
    session: ScopedSessionDep,
    project_id: UUID | None = Query(default=None),
    project_path: str | None = Query(default=None, max_length=4096),
):
    if project_path is not None:
        if _org_mode():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="project_path is not accepted in org deployments; use project_id",
            )
        # Local requests that carry both parameters have always preferred the
        # UUID. Do not let the ignored compatibility field enter resolution.
        if project_id is None:
            catalog = await run_in_threadpool(
                _desktop_registered_path_catalog, scope=session.scope
            )
            if not project_path or "\x00" in project_path:
                return []
            requested = os.path.normpath(os.path.expanduser(project_path))
            # Snyk Code recognizes basename as a path-traversal sanitizer. Keep
            # it directly at the HTTP boundary. Both request-derived values are
            # consumed solely by equality checks; the card builder receives the
            # exact server-created source retained in the complete catalog.
            project_name = os.path.basename(requested)
            if not project_name:
                return []
            source = _desktop_registered_source(catalog, requested, project_name)
            if source is None:
                return []
            return await run_in_threadpool(_artifact_cards, session, [source])
    if project_id is not None:
        raw_project_ref = str(project_id)
        # Although FastAPI has already parsed this as UUID, make the recognized
        # taint barrier explicit before the value enters root discovery.
        project_ref = os.path.basename(raw_project_ref)
        if project_ref != raw_project_ref:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid project",
            )
        return await run_in_threadpool(_scoped_project_cards, session, project_ref)
    return await run_in_threadpool(_all_artifact_cards, session)



# AuthenticatedInOrgMode, declared explicitly: same ScopedSessionDep
# fail-closed reasoning as list_artifacts above.
@router.delete(
    "/{slug}", status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def delete_artifact_by_slug(
    slug: str,
    # Keep the dependency explicit: SAST treats the Annotated alias as an HTTP
    # parameter, then incorrectly taints the server-owned artifact roots.
    session: ScopedSession = Depends(get_scoped_session),
    project_id: UUID = Query(...),
    # D7: an org admin may delete an artifact whose owner is unknown.
    principal: Principal | None = Depends(get_principal),
):
    ref = _artifact_delete_ref(slug)
    return await delete_artifact_for_request(
        session, ref, project_id=project_id, principal=principal
    )


async def delete_artifact_for_request(
    session, slug: str | _ArtifactDeleteRef, *, project_id: UUID, principal: Principal | None = None
) -> None:
    """Delete one artifact of one project. Unpublishes first; a failed unpublish
    leaves the folder in place and surfaces the error."""
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.services.artifact_publish_key import PublishKey
    from cowork.services.publish import _resolve_publish_endpoint

    # This helper is also called directly by internal/test code, so retain the
    # same concrete parsing boundary the HTTP adapter applies above. In
    # particular, no request-derived string can select a project path.
    await require_product_permission(session.scope, "artifact.manage")

    project_id = UUID(str(project_id))

    # New clients address deletion by artifact id. Keep the slug fallback for
    # older desktop clients, but never use it when the caller supplied a UUID:
    # two conversations in one project can legitimately produce the same slug.
    ref = slug if isinstance(slug, _ArtifactDeleteRef) else _artifact_delete_ref(slug)
    server_project_id, sources = _scoped_project_sources(session, project_id)
    source = None
    folder = None
    if ref.artifact_id is not None:
        # Keep UUID validation and pass the basename output to the resolver:
        # SAST does not recognize UUID parsing or dataclass fields as a path
        # sanitizer. Parsing first rejects traversal instead of truncating it.
        artifact_id = os.path.basename(UUID(ref.artifact_id).hex)
        project_ref = os.path.basename(str(server_project_id))
        # Resolved through the review path so a reviewer who was granted access
        # to this draft is told they cannot delete it, instead of being told it
        # does not exist. Without a grant it still 404s — `require_artifact_owner`
        # below is what refuses, the resolver only decides visibility.
        from cowork.api.v1.artifact_scope import review_artifact_for_request

        source, folder, _metadata, _is_own = review_artifact_for_request(
            session, project_ref, artifact_id
        )
    else:
        # The legacy name is compared with entries discovered under each
        # authorized root. It is never joined onto the root, and symlinked
        # artifact folders are not eligible for mutation.
        assert ref.legacy_slug is not None
        source, folder = _legacy_artifact_for_sources(sources, ref.legacy_slug)
    if source is None or folder is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    folder_name = _artifact_folder_name(source, folder)

    from cowork.services.artifact_ownership import resolve_artifact_owner
    from cowork.services.artifact_permissions import (
        may_delete_ownerless_artifact,
        require_artifact_owner,
    )

    # One resolution for both decisions, so they cannot disagree.
    resolution = resolve_artifact_owner(session, source, folder_name)
    # D7: the only thing an unknown owner grants anyone is an org admin's delete.
    admin_delete = may_delete_ownerless_artifact(
        session, source, folder_name, principal, resolution=resolution
    )
    if not admin_delete:
        require_artifact_owner(session, source, folder_name, resolution=resolution)
    expected_artifact_id = artifact_id if ref.artifact_id is not None else None
    publish_url, api_key = _resolve_publish_endpoint(get_user_settings())
    if _org_mode():
        # `ScopedSession.scope` is the wrapper's own attribute — the same one
        # ProjectService reads for `scoped_storage_root`.
        scope = session.scope
        if not scope or not scope.org_id or not scope.user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tenant in scope")
        # Unpublish acts on the viewer, and the viewer scopes by the token's owner,
        # so the credential has to be the acting user's - not a stored provider key
        # (org deployments have none).
        from cowork.services.artifact_autopublish import _active_workspace_id

        api_key = await PublishKey(
            scope.user_id, scope.org_id, min_ttl_s=120.0, workspace_id=_active_workspace_id(scope)
        ).get()
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not mint a publish credential",
            )
        from cowork.services.artifact_access import (
            ArtifactAccessUnavailable,
            revoke_draft_review_access,
        )
        from cowork.services.artifact_authorization_identity import existing_authorization_key
        try:
            if expected_artifact_id is None:
                expected_artifact_id = await run_in_threadpool(
                    _artifact_id_for_folder, source, folder_name
                )
            canonical_key = await run_in_threadpool(
                existing_authorization_key,
                expected_artifact_id,
                scope,
                # The admin is not the identity's owner; passing their id would
                # refuse the delete over an alias bound to the original creator.
                owner_user_id=None if admin_delete else str(scope.user_id),
            )
            # A historical grant may predate the SQL alias. The auth delete
            # checks its existing owner binding and never creates one. An
            # ownerless artifact without an alias never had a draft rule
            # provisioned (provisioning needs an owner), so there is nothing
            # to revoke on the admin path.
            if canonical_key is not None or not admin_delete:
                authorization_id = canonical_key.split("/", 1)[1] if canonical_key else expected_artifact_id
                await revoke_draft_review_access(authorization_id, scope)
        except ArtifactAccessUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artifact not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    try:
        await run_in_threadpool(
            _delete_artifact_from_source,
            source,
            folder_name,
            expected_artifact_id=expected_artifact_id,
            api_key=api_key,
            publish_url=publish_url,
            scope=session.scope,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except OSError as e:
        # A root/folder replaced by a symlink is deliberately indistinguishable
        # from an artifact that disappeared between resolution and mutation.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found") from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not delete artifact") from e

    from cowork.services.artifact_ownership import forget_artifact_owner

    # After the bytes are gone: a later artifact with the same slug starts with
    # its own owner (a no-op outside org mode). Best-effort; a stale row is
    # replaced on rekey and never grants anything to a folder that does not
    # exist. Synchronous on purpose: it uses the request session, which must
    # not cross threads, and it is one indexed lookup plus one delete.
    try:
        forget_artifact_owner(
            session, source, folder_name, actor_id=str(session.scope.user_id)
        )
    except Exception:
        logger.warning(
            "Could not drop the owner of a deleted artifact",
            exc_info=True,
            extra=log_context(project_id=source.project_id, artifact_slug=folder_name),
        )


# The routes below (through delete_artifact_endpoint) are DesktopOnly, which
# IS the require_local_tenancy guard: it 403s them outright in org mode, so they
# only ever run in local tenancy. There the request's scope is LOCAL_SCOPE, or,
# on a deployment that trusts identity headers, a person's partition, so each
# route takes the scope and resolves a path only inside that scope's artifacts
# roots: another person's path answers 404, as an unknown one does. /proxy is
# the exception and says so at its own definition. A comment covering a range
# of routes only holds while the range does; each route carries the
# declaration itself, so inserting one here cannot inherit this paragraph by
# accident.
@router.get("/status", dependencies=[Depends(require(DesktopOnly))])
async def artifact_status(
    scope: ScopeDep, path: str = Query(..., min_length=1, max_length=4096)
):
    # Cheap published/modified/access read for the preview viewer's in-place
    # refresh. Never raises for an unknown path — returns the blank default.
    return _desktop_artifact_status_for_path(path, scope=scope)


@router.get("/preview", dependencies=[Depends(require(DesktopOnly))])
async def preview_artifact(scope: ScopeDep, path: str = Query(...)):
    try:
        artifact = resolve_artifact_path(path, scope=scope)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        return _preview_artifact(artifact)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read artifact") from e


@router.post("/preview-mount", dependencies=[Depends(require(DesktopOnly))])
async def preview_mount_endpoint(req: _PathBody, request: Request, scope: ScopeDep):
    try:
        artifact = resolve_artifact_path(req.path, scope=scope)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        payload = await mount_preview(artifact, scope=scope)
    except BackendPreviewRefused as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    if payload.get("kind") == "proxy":
        # Build the absolute proxy URL from the incoming request. Using
        # scheme+netloc means the iframe loads through the same host
        # the client used to reach us — works equally for desktop
        # (127.0.0.1:port) and cloud (reverse-proxy origin).
        token = payload["token"]
        payload["proxyUrl"] = (
            f"{request.url.scheme}://{request.url.netloc}"
            f"/api/v1/artifacts/proxy/{token}/"
        )
    return payload


@router.get("/preview-asset/{token}/{rel_path:path}", dependencies=[Depends(require(DesktopOnly))])
async def preview_asset(token: str, rel_path: str, request: Request, scope: ScopeDep):
    parent = preview_mount_for(token, scope=scope)
    if parent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview mount has expired or is unknown")
    try:
        target = (parent / rel_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid asset path") from exc
    try:
        target.relative_to(parent)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Asset is outside the artifact directory")
    if not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    if wants_html_preview(media_type, request):
        # Offload the (potentially large) synchronous read so it doesn't stall
        # the event loop / other in-flight SSE streams — this endpoint is async.
        resp = await run_in_threadpool(
            html_preview_response, target, comments=wants_comment_layer(request)
        )
        if resp is not None:
            return resp
    return FileResponse(target, media_type=media_type, headers=artifact_response_headers(media_type))


@router.get(
    "/serve/{project_name}/{file_path:path}",
    dependencies=[Depends(require(DesktopOnly))],
)
def serve_artifact_file(
    project_name: str,
    file_path: str,
    request: Request,
    session: ScopedSessionDep,
):
    """Serve a file from `<project>/.anton/artifacts/<file_path>` over
    HTTP. Stateless, origin-relative, frame-able so the in-app iframe
    and new-tab open both work in web deployments."""
    base = _project_artifacts_base(project_name, session)
    if base is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")
    try:
        target = (base / file_path).resolve()
        target.relative_to(base.resolve())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact path") from exc
    if not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    # This endpoint is a sync `def`, so FastAPI already runs it in a threadpool
    # — the blocking read here doesn't touch the event loop.
    if wants_html_preview(media_type, request):
        resp = html_preview_response(target, comments=wants_comment_layer(request))
        if resp is not None:
            return resp
    return FileResponse(target, media_type=media_type, headers=artifact_response_headers(media_type))


@router.post("/open", dependencies=[Depends(require(DesktopOnly))])
async def open_artifact(req: _PathBody, scope: ScopeDep):
    from cowork.services.artifacts import _org_mode, _NO_EXEC_DETAIL
    # In org mode this always refuses; see _org_mode's docstring in services/artifacts.py.
    # A person's partition refuses too: the server is not that person's
    # machine, so opening a file runs a program on a host they don't sit at.
    # Both refuse before the path resolves, so the answer says nothing about it.
    if _org_mode() or scope.personal:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_NO_EXEC_DETAIL)
    try:
        artifact = resolve_artifact_path(req.path, scope=scope)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        subprocess.run(["open", str(artifact)], check=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not open artifact") from exc
    return {"status": "ok", "path": str(artifact)}


def _resolve_reveal_path(path: str, session: ScopedSession) -> Path:
    try:
        return resolve_artifact_path(path, scope=session.scope)
    except FileNotFoundError:
        pass
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    if not path or "\x00" in path or path.lstrip().startswith("~"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")

    normalized = os.path.normpath(path.strip())
    if (
        not normalized
        or normalized == "."
        or normalized.startswith("..")
        or normalized == ".."
        or os.path.isabs(normalized)
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")

    try:
        rel = Path(normalized)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path") from exc

    # Fallback resolution only accepts project-relative paths.
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")

    for project in ProjectService(session).list_projects():
        project_root = Path(project.path).resolve()
        try:
            resolved = (project_root / rel).resolve()
            resolved.relative_to(project_root)
        except Exception:
            continue
        if resolved.exists():
            return resolved
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Path is not in a known project or artifact directory")


@router.post("/reveal", dependencies=[Depends(require(DesktopOnly))])
async def reveal_artifact(req: _PathBody, session: ScopedSessionDep):
    if session.scope.personal:
        # Same refusal as /open, before the path resolves: the server's file
        # manager is not on a person's machine.
        from cowork.services.artifacts import _NO_EXEC_DETAIL

        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_NO_EXEC_DETAIL)
    target = _resolve_reveal_path(req.path, session)
    try:
        reveal_in_file_manager(target)
    except ExecutionRefused as exc:
        # reveal_in_file_manager's own _org_mode() refusal (services/artifacts.py) is a
        # deployment policy, not a failure to reveal the file: surface it as 403 with its
        # detail, matching open_artifact, instead of falling into the generic 500 below.
        # Caught by its own type rather than as RuntimeError, so a genuine RuntimeError
        # from the platform call underneath is still reported as the 500 it is.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not reveal artifact") from exc
    return {"status": "ok", "path": str(target)}


@router.api_route(
    "/proxy/{token}/{rel_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    dependencies=[Depends(require(OpenByDesign))],
)
async def proxy(token: str, rel_path: str, request: Request, scope: ScopeDep):
    """HTTP forwarder for fullstack-artifact previews.

    Streams the request to the artifact's backend running on
    `127.0.0.1:<metadata.json port>`, injects CORS, strips hop-by-hop
    headers. See `cowork.services.preview_proxy` for the body.

    OpenByDesign, not `DesktopOnly` like its siblings: the token is the
    real guard here, not tenancy. `_PREVIEW_MOUNTS` (see
    `cowork.services.artifacts.preview_mount_for`) is only ever populated by
    `/preview-mount`, which IS `DesktopOnly` — so in org mode
    this dict is permanently empty and any token 404s here regardless, by
    construction rather than by a declared guard. The token resolves only
    inside the caller's own artifacts roots, as on `/preview-asset`.
    """
    from cowork.services.preview_proxy import proxy_artifact_request
    return await proxy_artifact_request(token, rel_path, request, scope=scope)


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require(DesktopOnly))])
def delete_artifact_endpoint(scope: ScopeDep, path: str = Query(...)):
    try:
        from cowork.services.publish import (
            desktop_artifact_and_base,
            desktop_publish_credential,
        )

        found = desktop_artifact_and_base(path, scope=scope)
        # Credential resolved after the path so an unresolvable artifact still
        # reports 404 rather than "configure your API key". Only needed at all
        # because delete unpublishes first.
        api_key, publish_url = desktop_publish_credential()
        _delete_artifact(
            found.artifact, artifacts_base=found.artifacts_base,
            api_key=api_key, publish_url=publish_url, scope=scope,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        # An owner refusal (409) or unavailable ownership (503) keeps its status.
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not delete artifact") from e


# ── live apps: share and duplicate ──────────────────────────────────────────
# Only with live apps on (cowork.code_sandbox.live_apps_on), which create_app
# allows only on a trusted-identity deployment with personal projects; 404
# everywhere else, before the request is read (LiveAppRoute). Their own
# router, mounted at /artifacts beside this one.

live_app_router = APIRouter(route_class=LiveAppRoute)

_SHARE_ONE_WAY = "Choose one address, or everyone on this stack."
_NOT_AN_EMAIL = "Enter a valid email address."
_OWN_ADDRESS = "That's your own address."
_NOT_A_LIVE_APP = "Only full-stack apps can be duplicated this way."
_NO_SUCH_ARTIFACT = "No artifact by that name in this project."
_NOT_CONFIRMED = "Confirm the disclaimer first."
_NO_LONGER_SHARED = "You can no longer duplicate this app."
_NO_SUCH_PROJECT = "No project by that id."
_SHARE_GONE = "That permission no longer exists."
_ALREADY_COPIED = "You already have a copy of this app. Open it, or delete it to duplicate the app again."
_NO_EMAIL = "Live apps need an email address on your account. Ask the operator to add one."

_LIVE_APP_ROUTE = [Depends(require(AuthenticatedInOrgMode))]


def _live_app_caller(principal: Principal | None):
    """The caller of a live-app route that changes something: 403 for a
    caller with no verified email (live_app_person), who can neither share
    nor be shared with."""
    from cowork.services.live_apps import live_app_person

    person = live_app_person(principal)
    if person is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_NO_EMAIL)
    return person


def _project_dir(session: ScopedSession, *, project_id: UUID) -> Path | None:
    """Project ``project_id``'s folder in the session's own scope, or None."""
    from cowork.services.projects import ProjectNotFoundError

    try:
        return Path(ProjectService(session).get_project(project_id).path)
    except ProjectNotFoundError:
        return None


async def _require_own_live_app(principal: Principal | None, *, project_id: UUID, slug: str) -> None:
    """Refuse unless ``slug`` in the caller's own project ``project_id`` is a
    full-stack app: 404 for a project outside their space or a missing
    artifact, 400 for another type, 422 for a folder the copy rule refuses."""
    from cowork.db.scoped import scope_from_principal
    from cowork.db.units import run_db
    from cowork.services.live_apps import LiveAppCopyRefused, LiveAppGone, NotALiveApp, read_app_metadata

    project_dir = await run_db(partial(_project_dir, project_id=project_id), scope=scope_from_principal(principal))
    if project_dir is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NO_SUCH_PROJECT)
    try:
        await run_in_threadpool(read_app_metadata, project_dir, slug)
    except LiveAppGone:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NO_SUCH_ARTIFACT) from None
    except NotALiveApp:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_NOT_A_LIVE_APP) from None
    except LiveAppCopyRefused as refusal:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(refusal)) from None


def _share_response(share) -> LiveAppShareResponse:
    return LiveAppShareResponse(
        share_id=share.share_id,
        slug=share.slug,
        email=share.recipient_email,
        everyone=share.recipient_email is None,
        shared_at=share.shared_at,
    )


@live_app_router.post("/live-app-share", dependencies=_LIVE_APP_ROUTE)
async def share_live_app(
    body: LiveAppShareRequest, principal: Principal | None = Depends(get_principal)
) -> LiveAppShareResponse:
    """Share one of the caller's full-stack apps with one colleague's address,
    or with everyone on the stack. Sharing it the same way again returns the
    same share."""
    from cowork.db.scoped import SYSTEM_SCOPE
    from cowork.db.units import run_db
    from cowork.services.live_apps import is_email, normalized_email, record_share

    owner = _live_app_caller(principal)
    if (body.email is None) == (not body.everyone):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_SHARE_ONE_WAY)
    recipient_email = None
    if body.email is not None:
        recipient_email = normalized_email(body.email)
        if not is_email(recipient_email):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_NOT_AN_EMAIL)
        if recipient_email == owner.email:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_OWN_ADDRESS)
    await _require_own_live_app(principal, project_id=body.project_id, slug=body.slug)
    share = await run_db(
        partial(
            record_share,
            owner=owner,
            project_id=body.project_id,
            slug=body.slug,
            recipient_email=recipient_email,
        ),
        scope=SYSTEM_SCOPE,
    )
    return _share_response(share)


@live_app_router.get("/live-app-shares", dependencies=_LIVE_APP_ROUTE)
async def live_app_shares(
    project_id: Annotated[UUID, Query(alias="projectId")],
    slug: Annotated[str, Query(min_length=1, max_length=255)],
    principal: Principal | None = Depends(get_principal),
) -> list[LiveAppShareResponse]:
    """Every share the caller made of their app ``slug`` in project
    ``projectId``, newest first, for the share dialog's "Stop sharing" list."""
    from cowork.db.scoped import SYSTEM_SCOPE
    from cowork.db.units import run_db
    from cowork.services.live_apps import live_app_person, shares_of_app

    owner = live_app_person(principal)
    if owner is None:
        return []
    await _require_own_live_app(principal, project_id=project_id, slug=slug)
    shares = await run_db(partial(shares_of_app, owner=owner, project_id=project_id, slug=slug), scope=SYSTEM_SCOPE)
    return [_share_response(share) for share in shares]


@live_app_router.delete("/live-app-share/{share_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=_LIVE_APP_ROUTE)
async def withdraw_live_app_share(share_id: UUID, principal: Principal | None = Depends(get_principal)) -> None:
    """Stop sharing: remove one of the caller's own shares. The recipients'
    lists drop it at once; copies already made stay theirs."""
    from cowork.db.scoped import SYSTEM_SCOPE
    from cowork.db.units import run_db
    from cowork.services.live_apps import delete_share

    owner = _live_app_caller(principal)
    if not await run_db(partial(delete_share, owner=owner, share_id=share_id), scope=SYSTEM_SCOPE):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_SHARE_GONE)


@live_app_router.get("/shared-live-apps", dependencies=_LIVE_APP_ROUTE)
async def shared_live_apps(
    session: ScopedSession = Depends(get_scoped_session),
    principal: Principal | None = Depends(get_principal),
) -> list[SharedLiveAppResponse]:
    """The apps other people let the caller duplicate, by address or for
    everyone, newest first, each once, with the caller's copy from each while
    its folder is there. An app whose owner deleted it, or whose folder no
    longer holds a full-stack app the server can read, drops out."""
    from cowork.db.scoped import SYSTEM_SCOPE, scope_for_identity
    from cowork.db.units import run_db
    from cowork.services.live_apps import (
        LiveAppCopyRefused,
        LiveAppGone,
        copies_of_shares,
        live_app_person,
        read_app_metadata,
        shared_text,
        shares_with,
    )

    recipient = live_app_person(principal)
    if recipient is None:
        return []
    shares = await run_db(partial(shares_with, recipient=recipient), scope=SYSTEM_SCOPE)
    copies = await run_db(partial(copies_of_shares, recipient=recipient), scope=SYSTEM_SCOPE)
    listed = []
    for share in shares:
        owner_scope = scope_for_identity(org_id=share.owner_partition_id, user_id=share.owner_user_id)
        project_dir = await run_db(partial(_project_dir, project_id=share.project_id), scope=owner_scope)
        if project_dir is None:
            continue
        try:
            metadata = await run_in_threadpool(read_app_metadata, project_dir, share.slug)
        except (LiveAppGone, LiveAppCopyRefused):
            continue
        kept = await _kept_copy_of(session, copies.get(share.share_id, []))
        listed.append(
            SharedLiveAppResponse(
                share_id=share.share_id,
                slug=share.slug,
                name=shared_text(metadata.name) or share.slug,
                description=shared_text(metadata.description),
                owner_email=share.owner_email,
                everyone=share.everyone,
                shared_at=share.shared_at,
                your_copy=None if kept is None else LiveAppCopyRef(project_id=kept.project_id, slug=kept.slug),
            )
        )
    return listed


async def _kept_copy_of(session: ScopedSession, copies: list[CopyOfShare]) -> CopyOfShare | None:
    """The caller's copy among ``copies`` whose folder is still there, in a
    project of theirs (``session``'s scope)."""
    from cowork.services.live_apps import kept_copy

    project_dirs = {}
    for copy in copies:
        project_dirs[copy.project_id] = await run_in_threadpool(_project_dir, session, project_id=copy.project_id)
    return await run_in_threadpool(kept_copy, project_dirs, copies)


@live_app_router.post("/live-app-duplicate", dependencies=_LIVE_APP_ROUTE)
async def duplicate_live_app(
    body: LiveAppDuplicateRequest,
    session: ScopedSession = Depends(get_scoped_session),
    principal: Principal | None = Depends(get_principal),
) -> dict:
    """Copy an app someone let the caller duplicate into a project of the
    caller's own, after the caller confirmed the disclaimer, and answer the
    copy's card. One copy per person per share: while the caller's copy from
    this share still has its folder, another call answers 409."""
    from cowork.db.scoped import SYSTEM_SCOPE, scope_for_identity
    from cowork.db.units import run_db
    from cowork.services.artifact_ownership import record_artifact_owner
    from cowork.services.live_apps import (
        LiveAppCopyRefused,
        LiveAppGone,
        copies_of_shares,
        forget_copy,
        read_shared_app,
        record_copy,
        remove_app_copy,
        share_for_recipient,
        write_app_copy,
    )

    recipient = _live_app_caller(principal)
    if not body.confirmed:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_NOT_CONFIRMED)
    share = await run_db(
        partial(share_for_recipient, share_id=body.share_id, recipient=recipient), scope=SYSTEM_SCOPE
    )
    if share is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NO_LONGER_SHARED)
    copies = await run_db(partial(copies_of_shares, recipient=recipient), scope=SYSTEM_SCOPE)
    if await _kept_copy_of(session, copies.get(share.share_id, [])) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_ALREADY_COPIED)
    owner_scope = scope_for_identity(org_id=share.owner_partition_id, user_id=share.owner_user_id)
    owner_dir = await run_db(partial(_project_dir, project_id=share.project_id), scope=owner_scope)
    if owner_dir is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NO_LONGER_SHARED)
    target_dir = await run_in_threadpool(_project_dir, session, project_id=body.project_id)
    if target_dir is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NO_SUCH_PROJECT)
    try:
        app = await run_in_threadpool(read_shared_app, owner_dir, share.slug)
    except LiveAppGone:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NO_LONGER_SHARED) from None
    except LiveAppCopyRefused as refusal:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(refusal)) from None
    slug = await run_in_threadpool(write_app_copy, target_dir, app)
    try:
        await run_db(
            partial(record_copy, recipient=recipient, project_id=body.project_id, slug=slug, share=share),
            scope=SYSTEM_SCOPE,
        )
        # Writes no row where live apps run: create_app allows them only in
        # local tenancy, where the copy is the caller's by the project it
        # lands in. The one helper for that row, should live apps ever run
        # in org mode.
        await run_in_threadpool(
            record_artifact_owner, session, body.project_id, slug, recipient.user_id, action="live_app_duplicate"
        )
    except Exception:
        # A copy without its received mark would reach the agent as the
        # person's own code, so it doesn't stay.
        await run_in_threadpool(remove_app_copy, target_dir, slug)
        try:
            await run_db(
                partial(forget_copy, recipient=recipient, project_id=body.project_id, slug=slug), scope=SYSTEM_SCOPE
            )
        except Exception:
            logger.warning(
                "Live app copy mark not removed",
                exc_info=True,
                extra={
                    "event": "live_app_copy_mark_unremoved",
                    **log_context(project_id=body.project_id, artifact_slug=slug),
                },
            )
        raise
    cards = await run_in_threadpool(_scoped_project_cards, session, str(body.project_id))
    card = next((card for card in cards if Path(str(card.get("folder") or "")).name == slug), None)
    if card is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="The copy has no card")
    return card
