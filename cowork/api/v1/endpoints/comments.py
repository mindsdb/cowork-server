"""Artifact-comments proxy endpoints (renderer -> cowork-server -> inference).

Mounted at /api/v1/artifact-comments. The renderer calls these without auth; the
proxy attaches the user's MindsHub credential upstream (see comments_proxy).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from cowork.services.comments_proxy import forward_comments_rest, forward_comments_stream

router = APIRouter()


@router.get("/{user_dir}/{report_id}/stream")
async def comments_stream(user_dir: str, report_id: str, request: Request):
    # SSE — registered before the catch-all so it isn't swallowed by {subpath:path}.
    return await forward_comments_stream(request, user_dir, report_id)


@router.api_route(
    "/{user_dir}/{report_id}/{subpath:path}",
    methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
)
async def comments_rest(user_dir: str, report_id: str, subpath: str, request: Request):
    # threads (list/create/edit/delete), replies (add/edit/delete), status.
    return await forward_comments_rest(request, user_dir, report_id, subpath)
