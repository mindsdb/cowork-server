"""Shared HTTP presentation helpers for artifact preview endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse

from cowork.services.comments_layer import ACTIVATION_PARAM, inject_layer

NO_CACHE_HEADERS = {"Cache-Control": "no-cache, must-revalidate"}

# Artifact HTML is project-generated and can carry attacker-influenced
# content (e.g. an agent turn that processed a malicious web page, PR, or
# cloned repo); it is rendered here without sanitization by design, since
# arbitrary preview is the feature. The in-app preview iframe already
# constrains this with its own `sandbox` attribute (no `allow-same-origin`),
# but that only applies when the content is framed — ArtifactViewer's "open
# in browser" opens this same URL as a direct, top-level navigation, which no
# client-side sandbox attribute can touch. A CSP `sandbox` response header
# enforces the same restrictions regardless of how the page is loaded, so it
# also covers that path. Mirrors the non-proxy iframe sandbox value in
# ArtifactViewerBody.jsx (cowork repo) for consistency.
HTML_SANDBOX_CSP = "sandbox allow-scripts allow-popups allow-forms allow-modals"


def wants_comment_layer(media_type: str, request: Request) -> bool:
    """Whether this top-level HTML request opted into review markers."""
    return media_type == "text/html" and ACTIVATION_PARAM in request.query_params


def artifact_response_headers(media_type: str) -> dict[str, str]:
    """Cache headers for any artifact response; HTML responses additionally
    get the sandbox CSP above, since only those can carry executable script."""
    if media_type == "text/html":
        return {**NO_CACHE_HEADERS, "Content-Security-Policy": HTML_SANDBOX_CSP}
    return NO_CACHE_HEADERS


def html_with_comment_layer(target: Path) -> HTMLResponse | None:
    """Return injected HTML, or ``None`` when the file is not UTF-8 text."""
    try:
        html = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return HTMLResponse(inject_layer(html), headers=artifact_response_headers("text/html"))
