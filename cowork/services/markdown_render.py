"""Render a Markdown artifact to HTML for export.

The output is untrusted HTML: Python-Markdown passes raw HTML through and turns
``![x](url)`` into ``<img>``, so ``artifact_inline.inline_artifact_html`` runs
after it.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as etree

import markdown
from markdown.extensions import Extension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.sane_lists import SaneListExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import IDCOUNT_RE, render_inner_html, slugify
from markdown.treeprocessors import Treeprocessor

_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
# A tag in a heading's rendered HTML. It can't run past another "<", so a
# heading full of unclosed "<" still strips in linear time.
_TAG = re.compile(r"<[^<>]*>")
# A longer numeric suffix is part of the id's stem. ``int()`` is quadratic in
# the digits and refuses more than 4300 of them.
_MAX_COUNT_DIGITS = 32


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
            _HeadingIdExtension(),
        ],
        output_format="html",
    )
    try:
        return renderer.convert(text)
    except RecursionError:
        return f"<pre>{html.escape(text, quote=False)}</pre>"


class _HeadingIdExtension(Extension):
    def extendMarkdown(self, md: markdown.Markdown) -> None:
        # The priority TocExtension's tree processor runs at: after inline
        # parsing, before the unescape step.
        md.treeprocessors.register(_HeadingIds(md), "heading_ids", 5)


class _HeadingIds(Treeprocessor):
    """Give each heading the id ``TocExtension`` would, in linear time.

    An ``#fragment`` link in an HTML or PDF export lands on these ids, and a
    published copy of the same Markdown, which uses ``TocExtension``, carries
    the same ones. ``TocExtension`` itself counts up from ``_1`` past every
    id a repeated heading already took, so n identical headings cost n²
    lookups. Its table of contents goes too, so a ``[TOC]`` marker stays as
    text.
    """

    def run(self, root: etree.Element) -> None:
        taken = {element.get("id") for element in root.iter() if element.get("id")}
        # Each ``(stem, n)`` a search walked past maps to the count it stopped
        # at: every count from n up to there was taken.
        skips: dict[tuple[str, int], int] = {}
        for element in root.iter():
            if element.tag not in _HEADINGS or "id" in element.attrib:
                continue
            name = html.unescape(_TAG.sub("", render_inner_html(element, self.md)))
            element.set("id", _unique_id(candidate=slugify(name, "-"), taken=taken, skips=skips))


def _unique_id(*, candidate: str, taken: set[str], skips: dict[tuple[str, int], int]) -> str:
    """*candidate*, or the free ``stem_n`` that ``toc.unique`` picks for it.

    ``unique`` counts n up from the candidate's own numeric suffix plus one,
    or from 1, until ``stem_n`` is free. Ids are only ever added, so a count
    found taken stays taken, and ``skips`` jumps over every run of taken
    counts a search has walked before.
    """
    if candidate and candidate not in taken:
        taken.add(candidate)
        return candidate
    match = IDCOUNT_RE.match(candidate)
    if match is not None and len(match.group(2)) <= _MAX_COUNT_DIGITS:
        stem, count = match.group(1), int(match.group(2)) + 1
    else:
        stem, count = candidate, 1
    walked = []
    while f"{stem}_{count}" in taken:
        walked.append(count)
        count = skips.get((stem, count), count + 1)
    for passed in walked:
        skips[(stem, passed)] = count
    identifier = f"{stem}_{count}"
    taken.add(identifier)
    return identifier
