"""Artifact-comments proxy endpoints (renderer -> cowork-server -> inference).

Mounted at /api/v1/artifact-comments. The renderer calls these without auth; the
proxy attaches the user's MindsHub credential upstream (see comments_proxy).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from cowork.services.comments_proxy import forward_comments_rest, forward_comments_stream
from cowork.services.local_artifact_comments import handle_local_comments, local_comments_stream

router = APIRouter()


def _is_desktop_artifact(user_dir: str) -> bool:
    if user_dir != "artifact":
        return False
    from cowork.common.settings.app_settings import get_app_settings

    return get_app_settings().tenancy_mode != "org"


@router.get("/{user_dir}/{report_id}/stream")
async def comments_stream(user_dir: str, report_id: str, request: Request):
    # SSE — registered before the catch-all so it isn't swallowed by {subpath:path}.
    if _is_desktop_artifact(user_dir):
        return local_comments_stream(report_id)
    return await forward_comments_stream(request, user_dir, report_id)


@router.api_route(
    "/{user_dir}/{report_id}/{subpath:path}",
    methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
)
async def comments_rest(user_dir: str, report_id: str, subpath: str, request: Request):
    # threads (list/create/edit/delete), replies (add/edit/delete), status.
    if _is_desktop_artifact(user_dir):
        return await handle_local_comments(request, report_id, subpath)
    return await forward_comments_rest(request, user_dir, report_id, subpath)
