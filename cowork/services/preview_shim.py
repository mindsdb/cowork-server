"""Inject the in-frame preview shim as a document's first script.

Mirrors comments_layer.inject_layer, which appends its own script before
</body>. This one has to go first instead: it substitutes the storage APIs a
generated page reads during top-level initialisation, so anything running
before it has already thrown.
"""
from __future__ import annotations

import re
from pathlib import Path

from cowork.services.comments_layer import script_tag

SHIM_JS = Path(__file__).with_name("preview_shim.js").read_text(encoding="utf-8")

_LINE_OFFSET_TOKEN = "__LINE_OFFSET__"
_HEAD_OPEN_RE = re.compile(r"<head(?:\s[^>]*)?>", re.IGNORECASE)
_HTML_OPEN_RE = re.compile(r"<html(?:\s[^>]*)?>", re.IGNORECASE)
_DOCTYPE_RE = re.compile(r"<!doctype[^>]*>", re.IGNORECASE)


def _script_tag(shim_js: str, line_offset: int) -> str:
    return script_tag(shim_js.replace(_LINE_OFFSET_TOKEN, str(line_offset)))


def _build_markup(shim_js: str) -> str:
    """Render the shim's <script> tag with its own line offset filled in.

    The offset a later inline script shifts by is the number of newlines this
    injection adds, so it is measured on the rendered tag rather than
    maintained by hand.
    """
    return _script_tag(shim_js, _script_tag(shim_js, 0).count("\n"))


# Built once: SHIM_JS is a constant, and inject_shim runs in the hot preview path.
SHIM_MARKUP = _build_markup(SHIM_JS)


def inject_shim(html: str) -> str:
    """Return ``html`` with the shim as its first script.

    Placement, in order of preference: inside ``<head>``, after ``<html>``,
    after the doctype, at the very start. The doctype rule is not cosmetic —
    a script in front of it switches the document to quirks mode and changes
    the page's layout.

    These patterns match the FIRST occurrence, unlike comments_layer.inject_layer's
    LAST ``</body>``: a literal ``<head>`` earlier in the document — inside a
    comment, or a JavaScript string — mis-targets the insertion, and in the
    string case the injected ``</script>`` terminates the page's own script.
    inject_layer can anchor on the last match because it only needs to land
    somewhere before the document ends; this function has to land before the
    page's own scripts run, which the first-match risk is accepted for.
    """
    for pattern in (_HEAD_OPEN_RE, _HTML_OPEN_RE, _DOCTYPE_RE):
        match = pattern.search(html)
        if match:
            return html[: match.end()] + SHIM_MARKUP + html[match.end() :]
    return SHIM_MARKUP + html
