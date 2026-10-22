"""Render a Markdown artifact to HTML for export.

The output is untrusted HTML: Python-Markdown passes raw HTML through and turns
``![x](url)`` into ``<img>``, so ``artifact_inline.inline_artifact_html`` runs
after it.
"""
from __future__ import annotations

import html

import markdown
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.sane_lists import SaneListExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension


def render_markdown(*, text: str) -> str:
    """Convert Markdown *text* to an HTML fragment, without sanitizing it.

    The extensions are passed as instances rather than names, so no
    entry-point lookup can load a different extension, and none of them reads
    a file or the network. A ``Markdown`` instance holds per-document state,
    so each call builds its own.

    Python-Markdown parses nested lists recursively, so a few hundred levels
    raise ``RecursionError``. Such a document comes back as
    its own text in a ``pre`` block, so the export still holds every word.
    """
    renderer = markdown.Markdown(
        extensions=[
            TableExtension(),
            FencedCodeExtension(),
            SaneListExtension(),
            TocExtension(),
        ],
        output_format="html",
    )
    try:
        return renderer.convert(text)
    except RecursionError:
        return f"<pre>{html.escape(text, quote=False)}</pre>"
