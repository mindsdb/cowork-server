"""Render a Markdown artifact to HTML for export.

The output is untrusted HTML: Python-Markdown passes raw HTML through and turns
``![x](url)`` into ``<img>``, so ``artifact_inline.inline_artifact_html`` runs
after it.
"""
from __future__ import annotations

import html
import re

import markdown
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.sane_lists import SaneListExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension
from markdown.util import ETX, STX

# The longest run of backticks or tildes a fence may open with. Python-Markdown
# looks for each unclosed fence's closer through the rest of the document, so
# a ladder of ever-longer fences, which no real document needs, holds the
# whole process for about a minute in 2 MiB of Markdown. The cap refuses
# only the ladder: an unclosed fence of any length still makes that search,
# so many short unclosed fences still stall a conversion.
MAX_FENCE_LENGTH = 32
# A line that opens a longer fence, after at most three spaces.
_LONG_FENCE = re.compile(
    r"^ {0,3}(?:`{%d,}|~{%d,})" % (MAX_FENCE_LENGTH + 1, MAX_FENCE_LENGTH + 1), re.MULTILINE
)


class MarkdownRefused(Exception):
    """Markdown that ``render_markdown`` refuses, before or while Python-Markdown reads it."""


def render_markdown(*, text: str) -> str:
    """Convert Markdown *text* to an HTML fragment, without sanitizing it.

    The extensions are passed as instances rather than names, so no
    entry-point lookup can load a different extension, and none of them reads
    a file or the network. A ``Markdown`` instance holds per-document state,
    so each call builds its own.

    Each heading gets the id ``TocExtension`` gives it, so an ``#fragment``
    link in an HTML or PDF export lands where it does in a published copy of
    the same Markdown. There is no table of contents: a ``[TOC]`` marker stays
    as text.

    Python-Markdown parses nested lists recursively, so a few hundred levels
    raise ``RecursionError``. Such a document comes back as
    its own text in a ``pre`` block, so the export still holds every word.

    Raises ``MarkdownRefused`` when a line opens a fence longer than
    ``MAX_FENCE_LENGTH``, and when Python-Markdown raises ``ValueError``, as
    ``TocExtension`` does for a repeated heading whose number runs past 4300
    digits. Nothing here bounds how long a conversion takes: many short
    unclosed fences, many repeated headings, or a paragraph of backtick runs
    that never close can hold it for minutes. The export converts in a
    process it kills at its wall clock (``cowork.services.export_runner``).
    """
    _refuse_long_fence(text=text)
    renderer = markdown.Markdown(
        extensions=[
            TableExtension(),
            FencedCodeExtension(),
            SaneListExtension(),
            TocExtension(marker=""),
        ],
        output_format="html",
    )
    try:
        return renderer.convert(text)
    except RecursionError:
        return f"<pre>{html.escape(text, quote=False)}</pre>"
    except ValueError as exc:
        raise MarkdownRefused("Python-Markdown can't convert this document") from exc


def _refuse_long_fence(*, text: str) -> None:
    # Python-Markdown drops STX and ETX and reads a lone CR as a line end
    # before it looks for fences, so the check reads the text the same way.
    normalized = text.replace(STX, "").replace(ETX, "").replace("\r", "\n")
    if _LONG_FENCE.search(normalized):
        raise MarkdownRefused(f"a fence longer than {MAX_FENCE_LENGTH} characters")
