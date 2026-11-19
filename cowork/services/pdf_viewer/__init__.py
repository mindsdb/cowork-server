"""The pdf.js viewer that shows PDFs embedded in sandboxed artifact frames.

Browsers do not run their built-in PDF viewer in a sandboxed document, and no
sandbox keyword turns it back on. embed.js (spliced into the frame runtime and
the preview shim) points <embed>, <object> and <iframe> elements that show a
PDF of the page's own host at a document that renders it with pdf.js;
page.html is that document for a frame that navigates to a PDF itself.

The folder is shared byte for byte by mindshub_services (published artifacts,
the shell layer) and cowork-server (draft previews); both repositories pin
source_digest() in a test. pdf.js itself is not committed: fetch_pdfjs.py puts
it into pdfjs/ when the layer or the wheel is built. Only the standard library
is used, so the module imports the same way in both.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

DIR = Path(__file__).resolve().parent
# Filled by fetch_pdfjs.py; not committed.
PDFJS_DIR = DIR / "pdfjs"
ASSET_DIR = "_anton_pdf"
# First path segment, under an artifact's root, of the files served from here.
ASSET_PREFIX = ASSET_DIR + "/"
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


# The viewer's own files a browser fetches; every other served name is a
# pdf.js file in PDFJS_DIR. embed.js is spliced into the frame runtime and the
# preview shim as source text.
_OWN_ASSETS = frozenset({"viewer.mjs", "viewer.css"})


def _pdfjs_files() -> frozenset[str]:
    if not PDFJS_DIR.is_dir():
        return frozenset()
    return frozenset(
        path.relative_to(PDFJS_DIR).as_posix()
        for path in PDFJS_DIR.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )


# Empty until fetch_pdfjs.py has run: the viewer then shows its error state.
PDFJS_FILES = _pdfjs_files()
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


def source_digest() -> str:
    """sha256 over the committed files of the folder, the value both
    repositories pin; pdfjs/ is left out, fetch_pdfjs.py pins it."""
    lines = []
    for path in sorted(DIR.rglob("*"), key=lambda p: p.relative_to(DIR).as_posix()):
        rel = path.relative_to(DIR)
        # pdfjs/ and the folders fetch_pdfjs.py unpacks into (left behind by
        # an interrupted run) are not committed.
        top = rel.parts[0]
        if top == PDFJS_DIR.name or top.startswith(".pdfjs-") or "__pycache__" in rel.parts:
            continue
        if path.is_file():
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel.as_posix()}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def asset_name(rel_path: str) -> str | None:
    """The file name inside the asset folder when rel_path points into it."""
    return rel_path[len(ASSET_PREFIX):] if rel_path.startswith(ASSET_PREFIX) else None


def asset(name: str) -> tuple[bytes, str] | None:
    """(bytes, Content-Type) of a served file, or None.

    Only names from a fixed set (the viewer's own files, the files found in
    PDFJS_DIR at import) with a known type are served, so a name from a
    request can never reach another file; license files are not served.
    """
    if name in _OWN_ASSETS:
        path = DIR / name
    elif name in PDFJS_FILES:
        path = PDFJS_DIR / name
    else:
        return None
    content_type = _CONTENT_TYPES.get(PurePosixPath(name).suffix)
    if content_type is None:
        return None
    try:
        return path.read_bytes(), content_type
    except OSError:
        # pdfjs/ was fetched again after this process imported the list.
        return None


def viewer_page(rel_path: str, *, draft: bool) -> str:
    """page.html for the PDF at rel_path, relative to the artifact root.

    The asset folder is addressed relative to the page, so no credential goes
    into the HTML.
    """
    assets = "../" * rel_path.count("/") + ASSET_PREFIX
    return PAGE_HTML.replace("%ASSETS%", assets).replace("%DRAFT_ATTR%", " data-anton-pdf-draft" if draft else "")
