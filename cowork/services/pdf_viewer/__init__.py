"""The pdf.js viewer that shows PDFs embedded in sandboxed artifact frames.

Browsers do not run their built-in PDF viewer in a sandboxed document, and no
sandbox keyword turns it back on. embed.js (spliced into the frame runtime and
the preview shim) points <embed>, <object> and <iframe> elements that show a
PDF of the page's own host at a document that renders it with pdf.js;
page.html is that document for a frame that navigates to a PDF itself.

The folder is shared byte for byte by mindshub_services (published artifacts,
the shell layer) and cowork-server (draft previews). manifest.txt lists every
file with its sha256, and both repositories pin the digest of manifest.txt in
a test. Only the standard library is used, so the module imports the same way
in both.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

DIR = Path(__file__).resolve().parent
ASSET_DIR = "_anton_pdf"
# First path segment, under an artifact's root, of the files served from here.
ASSET_PREFIX = ASSET_DIR + "/"
MANIFEST_NAME = "manifest.txt"
PDF_TYPE = "application/pdf"
# Sec-Fetch-Dest of a frame navigating to a PDF: it gets the viewer page.
FRAME_DESTS = frozenset({"iframe", "frame"})
# The frame runtime and the preview shim carry this comment once; embed.js
# takes its place.
EMBEDS_MARKER = "/* pdf-embeds */"

# Only these types are served. With nosniff, a module served with another
# type does not run.
_CONTENT_TYPES = {
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".css": "text/css",
    ".wasm": "application/wasm",
    ".ttf": "font/ttf",
    ".pfb": "application/octet-stream",
    ".bcmap": "application/octet-stream",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
    ".png": "image/png",
}


# Listed in manifest.txt but never fetched by a browser: embed.js is spliced
# into the frame runtime and the preview shim as source text.
_NOT_SERVED = frozenset({"embed.js"})


def _read_manifest() -> tuple[str, dict[str, str]]:
    lines = (DIR / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    files = {}
    for line in lines[1:]:
        digest, name = line.split("  ", 1)
        files[name] = digest
    return lines[0], files


PDFJS_VERSION, FILES = _read_manifest()
EMBED_JS = (DIR / "embed.js").read_text(encoding="utf-8")
PAGE_HTML = (DIR / "page.html").read_text(encoding="utf-8")


def splice_embeds(script: str, name: str) -> str:
    """script (the file called name) with embed.js in place of its marker."""
    if script.count(EMBEDS_MARKER) != 1:
        raise ValueError(f"{name} must carry the pdf-embeds marker exactly once")
    return script.replace(EMBEDS_MARKER, EMBED_JS)


def wants_viewer_page(content_type: str, dest: str, *, document: bool = False) -> bool:
    """Whether a request for a file of content_type made with Sec-Fetch-Dest
    dest gets the viewer page instead of the PDF: a frame navigating to it,
    and with ``document`` also a top-level load (the draft host serves top
    documents itself, sandboxed, so a popup cannot show the PDF either)."""
    return content_type == PDF_TYPE and (dest in FRAME_DESTS or (document and dest == "document"))


def manifest_digest() -> str:
    """sha256 of manifest.txt, the value both repositories pin."""
    return hashlib.sha256((DIR / MANIFEST_NAME).read_bytes()).hexdigest()


def asset_name(rel_path: str) -> str | None:
    """The file name inside the asset folder when rel_path points into it."""
    return rel_path[len(ASSET_PREFIX):] if rel_path.startswith(ASSET_PREFIX) else None


def asset(name: str) -> tuple[bytes, str] | None:
    """(bytes, Content-Type) of a served file, or None.

    Only files listed in manifest.txt with a known type are served, so a name
    from a request can never reach another file; license files and
    manifest.txt itself are not served.
    """
    if name not in FILES or name in _NOT_SERVED:
        return None
    content_type = _CONTENT_TYPES.get(PurePosixPath(name).suffix)
    if content_type is None:
        return None
    return (DIR / name).read_bytes(), content_type


def viewer_page(rel_path: str, *, draft: bool) -> str:
    """page.html for the PDF at rel_path, relative to the artifact root.

    The asset folder is addressed relative to the page, so no credential goes
    into the HTML.
    """
    assets = "../" * rel_path.count("/") + ASSET_PREFIX
    return PAGE_HTML.replace("%ASSETS%", assets).replace("%DRAFT_ATTR%", " data-anton-pdf-draft" if draft else "")
