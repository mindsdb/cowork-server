"""The owner's editor that rides along with a shared doc's page.

While a doc is shared, and its owner can edit it from the link
(``cowork.docs.cloud``), the page loads the owner's editor from two files beside
it. They are copied in only then: about 600 KB in every doc would fill project
folders, and a project can be someone's git repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from cowork.db.scoped import tenant_scope_required

from .model import Doc
from .sharing import is_shared
from .store import atomic_write, write_html

# Built by the app (`npm run build:doc-editor`) and handed over as
# COWORK_DOC_EDITOR_DIR. Without it the page is read-only.
EDITOR_FILES = ("doc-editor.js", "doc-editor.css")
EDITOR_DIR_ENV = "COWORK_DOC_EDITOR_DIR"


def link_editing_available() -> bool:
    """The owner edits a shared doc from its link only on a one-person desktop.

    The live copy goes to MindsHub under the app's own credential, and the
    owner's edits come back to this machine. A deployment serving several
    people (org tenancy, or trusted identity headers) has no single owner to
    act for.
    """
    return not tenant_scope_required()


def _editor_source() -> Path | None:
    raw = os.environ.get(EDITOR_DIR_ENV, "").strip()
    if not raw:
        return None
    src = Path(raw)
    return src if all((src / name).is_file() for name in EDITOR_FILES) else None


def install_editor(folder: Path) -> bool:
    """Put the editor's files beside doc.html (when they differ); True when they are there."""
    src = _editor_source()
    if src is not None:
        for name in EDITOR_FILES:
            data = (src / name).read_bytes()
            dest = folder / name
            try:
                same = dest.read_bytes() == data
            except OSError:
                same = False
            if not same:
                atomic_write(dest, data)
    return all((folder / name).is_file() for name in EDITOR_FILES)


def remove_editor(folder: Path) -> None:
    for name in EDITOR_FILES:
        try:
            (folder / name).unlink()
        except FileNotFoundError:
            pass


def render_page(folder: Path, doc: Doc, *, shared: bool | None = None) -> None:
    """Write doc.html from ``doc``, with the owner's editor while the doc is shared.

    ``shared`` None reads the folder's publish record. Publishing passes True:
    the page is rendered for the upload, before the record says it is shared.
    """
    if shared is None:
        shared = is_shared(folder)
    editor = shared and link_editing_available() and install_editor(folder)
    if not editor:
        remove_editor(folder)
    write_html(folder, doc, editor=editor)
