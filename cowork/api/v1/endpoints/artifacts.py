"""Artifacts API endpoints.

Ported from cowork/server/routes/artifacts.py. Provides listing,
preview, iframe mount, open-in-OS, and reveal-in-finder for
agent-produced artifacts.
"""
from __future__ import annotations

import mimetypes
import os
import re
import stat as stat_module
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session

from cowork.db.scoped import ScopedSession, ScopedSessionDep
from cowork.db.session import get_session
from cowork.api.v1.endpoints.guards import require_local_tenancy
from cowork.api.v1.artifact_preview import (
    NO_CACHE_HEADERS,
    html_with_comment_layer,
    wants_comment_layer,
)
from cowork.api.v1.artifact_scope import artifact_sources_for_request
from cowork.common.paths import dir_scandir, dir_stat, open_pinned_child
from cowork.services.artifact_roots import artifacts_sources_for_scan
from cowork.services.artifacts import (
    ExecutionRefused,
    _org_mode,
    _project_artifacts_base,
    artifact_id_for_folder as _artifact_id_for_folder,
    delete_artifact as _delete_artifact,
    delete_artifact_from_source as _delete_artifact_from_source,
    get_preview_mount,
    list_artifacts as _list_artifacts,
    mount_preview,
    preview_artifact as _preview_artifact,
    resolve_artifact_path,
    reveal_in_file_manager,
)
from cowork.services.projects import ProjectService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


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
    from cowork.services.artifact_permissions import artifact_capabilities

    # Capabilities are a property of the ROOT, not of the artifact: in org mode
    # they come from the owning conversation, and every artifact under one root
    # shares it. Derived once per root, so a project with 50 artifacts across 50
    # conversations costs 50 conversation reads instead of 50 per card.
    by_base = {Path(source.base): source for source in sources}
    capabilities_by_base: dict[Path, dict] = {}
    for card in cards:
        base = Path(str(card.get("folder") or "")).parent
        source = by_base.get(base)
        if source is None:
            continue
        if base not in capabilities_by_base:
            capabilities_by_base[base] = artifact_capabilities(session, source)
        card["capabilities"] = capabilities_by_base[base]
    return cards


def artifacts_for_request(
    session, *, project_id: UUID | None = None, project_path: str | None = None
) -> list[dict]:
    sources = artifact_sources_for_request(session, project_id, project_path)
    return _artifact_cards(session, sources)


def _desktop_artifacts_for_project_path(session, project_path: str) -> list[dict]:
    """Select one desktop project's already-built response by its legacy path.

    ``project_path`` remains supported because older desktop builds address the
    list this way. The request string is never used to construct or open a
    path: every filesystem-backed card is built first from roots discovered by
    the server, then the normalized request is only a lookup key into those
    completed responses. In particular, the selected ``ProjectArtifacts``
    keeps ``project_id=None``, preserving local draft URLs and card addressing.
    """
    cards_by_path: dict[str, list[dict]] = {}
    for source in artifacts_sources_for_scan():
        cards = _artifact_cards(session, [source])
        project_dir = Path(source.base).parent.parent
        candidates = {os.path.normpath(str(project_dir))}
        try:
            candidates.add(os.path.normpath(str(project_dir.resolve(strict=False))))
        except OSError:
            pass
        for candidate in candidates:
            cards_by_path.setdefault(candidate, cards)

    if not project_path or "\x00" in project_path:
        return []
    requested = os.path.normpath(os.path.expanduser(project_path))
    return cards_by_path.get(requested, [])


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


def _desktop_artifact_status_for_path(path: str) -> dict:
    """Match a legacy path only against entries scanned from discovered roots.

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

    for source in artifacts_sources_for_scan():
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


@router.get("/")
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
            return _desktop_artifacts_for_project_path(session, project_path)
    return artifacts_for_request(session, project_id=project_id)



@router.delete("/{slug}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_artifact_by_slug(
    slug: str,
    session: ScopedSessionDep,
    project_id: UUID = Query(...),
):
    ref = _artifact_delete_ref(slug)
    return await delete_artifact_for_request(session, ref, project_id=project_id)


async def delete_artifact_for_request(
    session, slug: str | _ArtifactDeleteRef, *, project_id: UUID
) -> None:
    """Delete one artifact of one project. Unpublishes first; a failed unpublish
    leaves the folder in place and surfaces the error."""
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.services.artifact_publish_key import PublishKey
    from cowork.services.publish import _resolve_publish_endpoint

    # New clients address deletion by artifact id. Keep the slug fallback for
    # older desktop clients, but never use it when the caller supplied a UUID:
    # two conversations in one project can legitimately produce the same slug.
    ref = slug if isinstance(slug, _ArtifactDeleteRef) else _artifact_delete_ref(slug)
    sources = artifact_sources_for_request(session, project_id, None)
    source = None
    folder = None
    if ref.artifact_id is not None:
        # Resolved through the review path so a reviewer who was granted access
        # to this draft is told they cannot delete it, instead of being told it
        # does not exist. Without a grant it still 404s — `require_artifact_owner`
        # below is what refuses, the resolver only decides visibility.
        from cowork.api.v1.artifact_scope import review_artifact_for_request

        source, folder, _metadata, _is_own = review_artifact_for_request(
            session, str(project_id), ref.artifact_id
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

    from cowork.services.artifact_permissions import require_artifact_owner

    require_artifact_owner(session, source)
    expected_artifact_id = ref.artifact_id
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
        api_key = await PublishKey(scope.user_id, scope.org_id, min_ttl_s=120.0).get()
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not mint a publish credential",
            )
        from cowork.services.artifact_access import (
            ArtifactAccessUnavailable,
            revoke_draft_review_access,
        )
        try:
            if expected_artifact_id is None:
                expected_artifact_id = await run_in_threadpool(
                    _artifact_id_for_folder, source, folder_name
                )
            await revoke_draft_review_access(expected_artifact_id)
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


@router.get("/status", dependencies=[Depends(require_local_tenancy)])
async def artifact_status(path: str = Query(..., min_length=1, max_length=4096)):
    # Cheap published/modified/access read for the preview viewer's in-place
    # refresh. Never raises for an unknown path — returns the blank default.
    return _desktop_artifact_status_for_path(path)


@router.get("/preview", dependencies=[Depends(require_local_tenancy)])
async def preview_artifact(path: str = Query(...)):
    try:
        artifact = resolve_artifact_path(path)
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


class _ExportBody(BaseModel):
    path: str
    format: str  # 'pdf' | 'docx' | 'html'


@router.post("/export", dependencies=[Depends(require_local_tenancy)])
async def export_artifact_endpoint(req: _ExportBody):
    """Convert a document artifact (markdown/HTML) to PDF/Word/HTML, writing
    the result into the same artifact folder. Returns the new file's path so
    the client can open or download it.

    The route-level `require_local_tenancy` is broader than the pdf/docx refusal
    inside: it takes `req.path`, an absolute server path, and nothing in an org
    deployment can say which organization that path belongs to. The inner check
    stays because it is the one the direct-call tests exercise, and because it
    documents WHY those two formats are unsafe even where a path is trusted.
    """
    from fastapi.concurrency import run_in_threadpool

    from cowork.services.artifact_export import ExportError, export_artifact
    from cowork.services.artifacts import _org_mode

    fmt = (req.format or "").lower().lstrip(".")
    if fmt in ("pdf", "docx") and _org_mode():
        # Both converters resolve URIs referenced in the source HTML to embed
        # them in the output: xhtml2pdf fetches <img>/<link>/@import for PDF;
        # htmldocx's image handling calls urllib.request.urlopen for docx. The
        # source HTML is artifact content from the shared filesystem, written
        # by any org's agent, so this is an SSRF and local-file-read primitive
        # (including other orgs' trees) into a file the requester downloads.
        # Plain HTML export does no such resolution, so it stays available.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="PDF and Word export are not available on this deployment. Export to HTML instead.",
        )
    try:
        source = resolve_artifact_path(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        out = await run_in_threadpool(export_artifact, source, req.format)
    except ExportError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Export failed") from e
    return {"path": str(out), "filename": out.name}


@router.post("/preview-mount", dependencies=[Depends(require_local_tenancy)])
async def preview_mount_endpoint(req: _PathBody, request: Request):
    try:
        artifact = resolve_artifact_path(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        payload = await mount_preview(artifact)
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


@router.get("/preview-asset/{token}/{rel_path:path}", dependencies=[Depends(require_local_tenancy)])
async def preview_asset(token: str, rel_path: str, request: Request):
    parent = get_preview_mount(token)
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
    if wants_comment_layer(media_type, request):
        # Offload the (potentially large) synchronous read so it doesn't stall
        # the event loop / other in-flight SSE streams — this endpoint is async.
        resp = await run_in_threadpool(html_with_comment_layer, target)
        if resp is not None:
            return resp
    return FileResponse(target, media_type=media_type, headers=NO_CACHE_HEADERS)


@router.get("/serve/{project_name}/{file_path:path}", dependencies=[Depends(require_local_tenancy)])
def serve_artifact_file(project_name: str, file_path: str, request: Request):
    """Serve a file from `<project>/.anton/artifacts/<file_path>` over
    HTTP. Stateless, origin-relative, frame-able so the in-app iframe
    and new-tab open both work in web deployments."""
    base = _project_artifacts_base(project_name)
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
    if wants_comment_layer(media_type, request):
        resp = html_with_comment_layer(target)
        if resp is not None:
            return resp
    return FileResponse(target, media_type=media_type, headers=NO_CACHE_HEADERS)


@router.post("/open", dependencies=[Depends(require_local_tenancy)])
async def open_artifact(req: _PathBody):
    from cowork.services.artifacts import _org_mode, _NO_EXEC_DETAIL
    # In org mode this always refuses; see _org_mode's docstring in services/artifacts.py.
    if _org_mode():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_NO_EXEC_DETAIL)
    try:
        artifact = resolve_artifact_path(req.path)
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
        return resolve_artifact_path(path)
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


@router.post("/reveal", dependencies=[Depends(require_local_tenancy)])
async def reveal_artifact(req: _PathBody, session: ScopedSessionDep):
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
)
async def proxy(token: str, rel_path: str, request: Request):
    """HTTP forwarder for fullstack-artifact previews.

    Streams the request to the artifact's backend running on
    `127.0.0.1:<metadata.json port>`, injects CORS, strips hop-by-hop
    headers. See `cowork.services.preview_proxy` for the body.
    """
    from cowork.services.preview_proxy import proxy_artifact_request
    return await proxy_artifact_request(token, rel_path, request)


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_local_tenancy)])
def delete_artifact_endpoint(path: str = Query(...)):
    try:
        from cowork.services.publish import (
            desktop_artifact_and_base,
            desktop_publish_credential,
        )

        artifact, artifacts_base = desktop_artifact_and_base(path)
        # Credential resolved after the path so an unresolvable artifact still
        # reports 404 rather than "configure your API key". Only needed at all
        # because delete unpublishes first.
        api_key, publish_url = desktop_publish_credential()
        _delete_artifact(
            artifact, artifacts_base=artifacts_base,
            api_key=api_key, publish_url=publish_url,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not delete artifact") from e
