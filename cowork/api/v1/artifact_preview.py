"""Shared HTTP presentation helpers for artifact preview endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse

from cowork.services.comments_layer import ACTIVATION_PARAM
from cowork.services.preview_html import prepare_preview_html

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


def wants_comment_layer(request: Request) -> bool:
    """Whether this top-level HTML request opted into review markers.

    Callers only reach here after already establishing the response is
    text/html, so that check is not repeated.
    """
    return ACTIVATION_PARAM in request.query_params


# Mirrors the string forms Pydantic's bool coercion accepts for a `Query()`
# parameter. Anything else -- garbage, or the empty string FastAPI itself
# would 422 on -- is treated as "no", the same as the key being absent.
_TRUE_DOWNLOAD_VALUES = frozenset({"1", "true", "yes", "on", "y", "t"})


def wants_download(request: Request) -> bool:
    """Whether this request asked for the raw file instead of the preview.

    `/serve` and `/preview-asset` used to gate this on mere key presence
    (`"download" not in request.query_params`), so `?download=0` suppressed
    the shim there while the `/drafts` route -- which parses `download` as a
    real `Query(bool)` -- kept injecting it for the same query string. One
    predicate, with the same truthiness `Query(bool)` gives, keeps the three
    routes agreeing on what `?download=0` means.
    """
    raw = request.query_params.get("download")
    return raw is not None and raw.strip().lower() in _TRUE_DOWNLOAD_VALUES


def artifact_response_headers(media_type: str) -> dict[str, str]:
    """Cache headers for any artifact response; HTML responses additionally
    get the sandbox CSP above, since only those can carry executable script."""
    if media_type == "text/html":
        return {**NO_CACHE_HEADERS, "Content-Security-Policy": HTML_SANDBOX_CSP}
    return NO_CACHE_HEADERS


def html_preview_response(target: Path, *, comments: bool) -> HTMLResponse | None:
    """Return the prepared HTML, or ``None`` when the file is not UTF-8 text."""
    try:
        html = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return HTMLResponse(
        prepare_preview_html(html, comments=comments),
        headers=artifact_response_headers("text/html"),
    )
