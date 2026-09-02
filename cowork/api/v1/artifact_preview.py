"""Shared HTTP presentation helpers for artifact preview endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse

from cowork.services.comments_layer import ACTIVATION_PARAM, inject_layer

NO_CACHE_HEADERS = {"Cache-Control": "no-cache, must-revalidate"}


def wants_comment_layer(media_type: str, request: Request) -> bool:
    """Whether this top-level HTML request opted into review markers."""
    return media_type == "text/html" and ACTIVATION_PARAM in request.query_params


def html_with_comment_layer(target: Path) -> HTMLResponse | None:
    """Return injected HTML, or ``None`` when the file is not UTF-8 text."""
    try:
        html = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return HTMLResponse(inject_layer(html), headers=NO_CACHE_HEADERS)
