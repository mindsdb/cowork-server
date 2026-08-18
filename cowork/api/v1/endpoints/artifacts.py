"""Artifacts API endpoints.

Ported from cowork/server/routes/artifacts.py. Provides listing,
preview, iframe mount, open-in-OS, and reveal-in-finder for
agent-produced artifacts.
"""
from __future__ import annotations

import mimetypes
import os
import subprocess
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sqlmodel import Session

from cowork.db.scoped import ScopedSession, ScopedSessionDep
from cowork.db.session import get_session
from cowork.api.v1.endpoints.guards import require_local_tenancy
from cowork.services.artifact_roots import (
    artifacts_source_for_project as _source_for_project,
    artifacts_sources_for_scan as _sources_for_scan,
    artifacts_sources_for_scope as _sources_for_scope,
)
from cowork.services.comments_layer import ACTIVATION_PARAM, inject_layer
from cowork.services.artifacts import (
    ExecutionRefused,
    _project_artifacts_base,
    artifact_status as _artifact_status,
    delete_artifact as _delete_artifact,
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


# ``no-cache`` mirrors the FileResponse headers used elsewhere so a rebuilt
# artifact is always re-fetched.
_NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}


def _wants_comment_layer(media_type: str, request: Request) -> bool:
    """The comment marker layer is injected only into the top-level HTML
    document, and only when the renderer opts in via the activation query flag.
    Asset requests and flag-less loads stream untouched."""
    return media_type == "text/html" and ACTIVATION_PARAM in request.query_params


def _html_with_layer(target: Path):
    """Read an HTML file and return an HTMLResponse with the marker layer
    injected, or None when it can't be read as text (caller falls back to a
    plain FileResponse)."""
    try:
        html = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return HTMLResponse(inject_layer(html), headers=_NO_CACHE)


def _org_mode() -> bool:
    from cowork.common.settings.app_settings import get_app_settings

    return get_app_settings().tenancy_mode == "org"


def _sources(session, project_id: UUID | None, project_path: str | None):
    """The artifact roots this request may read.

    `project_id` is honored in BOTH modes. That is not cosmetic: the delete handler
    acts on `sources[0]`, so a desktop branch that ignored it would delete a
    same-named slug from whichever project sorted first — and inline chat cards
    carry a `project_id` in every mode, so that path is reachable from the UI.

    Org mode additionally refuses `project_path`: a filesystem path from the client
    carries no tenant, and these endpoints have no other way to tell which
    organization it belongs to.
    """
    if _org_mode() and project_path is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_path is not accepted in org deployments; use project_id",
        )

    if project_id is not None:
        try:
            return [_source_for_project(session, project_id)]
        except ValueError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")

    if _org_mode():
        return _sources_for_scope(session)

    sources = _sources_for_scan()
    if project_path is not None:
        # The client value is normalized as a STRING and never resolved through the
        # filesystem. Nothing here opens it — it only selects from roots the server
        # discovered itself, so the result is always a subset of the scan — but
        # `Path.resolve()` on untrusted input is a filesystem access driven by that
        # input, which is a path-injection sink whether or not it is reachable
        # (CodeQL py/path-injection). Comparing against both the raw and the
        # resolved form of each server-side root keeps symlinked project dirs
        # matching; those paths are ours, so resolving them is fine.
        wanted = os.path.normpath(os.path.expanduser(project_path))
        sources = [s for s in sources if _project_dir_matches(s, wanted)]
    return sources


def _project_dir_matches(source, wanted: str) -> bool:
    project_dir = source.base.parent.parent
    try:
        resolved = str(project_dir.resolve(strict=False))
    except OSError:
        resolved = ""
    return wanted in (str(project_dir), resolved)


# The two functions below hold the logic; the routes under them are thin adapters.
# Keeping them separate means the tenancy behavior is testable without FastAPI's
# Query sentinels standing in for the defaults.


def artifacts_for_request(
    session, *, project_id: UUID | None = None, project_path: str | None = None
) -> list[dict]:
    # Owner-side fields are stripped inside `card_for_folder`, not here: inline chat
    # cards use the same builder and would otherwise still carry the plaintext
    # password.
    return _list_artifacts(_sources(session, project_id, project_path))


@router.get("/")
async def list_artifacts(
    session: ScopedSessionDep,
    project_id: UUID | None = Query(default=None),
    project_path: str | None = Query(default=None),
):
    return artifacts_for_request(session, project_id=project_id, project_path=project_path)


@router.delete("/{slug}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_artifact_by_slug(
    slug: str,
    session: ScopedSessionDep,
    project_id: UUID = Query(...),
):
    return await delete_artifact_for_request(session, slug, project_id=project_id)


async def delete_artifact_for_request(session, slug: str, *, project_id: UUID) -> None:
    """Delete one artifact of one project. Unpublishes first; a failed unpublish
    leaves the folder in place and surfaces the error."""
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.services.artifact_publish_key import PublishKey
    from cowork.services.publish import _resolve_publish_endpoint

    base = _sources(session, project_id, None)[0].base
    folder = base / slug
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
    try:
        await run_in_threadpool(
            _delete_artifact, folder,
            artifacts_base=base, api_key=api_key, publish_url=publish_url,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not delete artifact") from e


@router.get("/status", dependencies=[Depends(require_local_tenancy)])
async def artifact_status(path: str = Query(...)):
    # Cheap published/modified/access read for the preview viewer's in-place
    # refresh. Never raises for an unknown path — returns the blank default.
    return _artifact_status(path)


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
    if _wants_comment_layer(media_type, request):
        # Offload the (potentially large) synchronous read so it doesn't stall
        # the event loop / other in-flight SSE streams — this endpoint is async.
        resp = await run_in_threadpool(_html_with_layer, target)
        if resp is not None:
            return resp
    return FileResponse(target, media_type=media_type, headers=_NO_CACHE)


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
    if _wants_comment_layer(media_type, request):
        resp = _html_with_layer(target)
        if resp is not None:
            return resp
    return FileResponse(target, media_type=media_type, headers=_NO_CACHE)


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
