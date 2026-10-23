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
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
# A longer numeric suffix is part of the id's stem. ``int()`` is quadratic in
# the digits and refuses more than 4300 of them.
_MAX_COUNT_DIGITS = 32


class MarkdownRefused(Exception):
    """Markdown that ``render_markdown`` refuses before Python-Markdown reads it."""


def render_markdown(*, text: str) -> str:
    """Convert Markdown *text* to an HTML fragment, without sanitizing it.

    The extensions are passed as instances rather than names, so no
    entry-point lookup can load a different extension, and none of them reads
    a file or the network. A ``Markdown`` instance holds per-document state,
    so each call builds its own.

    Python-Markdown parses nested lists recursively, so a few hundred levels
    raise ``RecursionError``. Such a document comes back as
    its own text in a ``pre`` block, so the export still holds every word.

    Raises ``MarkdownRefused`` when a line opens a fence longer than
    ``MAX_FENCE_LENGTH``. Nothing else bounds how long a conversion takes:
    many short unclosed fences, or a paragraph of backtick runs that never
    close, can still hold it for minutes or more.
    """
    _refuse_long_fence(text=text)
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


def _refuse_long_fence(*, text: str) -> None:
    # Python-Markdown drops STX and ETX and reads a lone CR as a line end
    # before it looks for fences, so the check reads the text the same way.
    normalized = text.replace(STX, "").replace(ETX, "").replace("\r", "\n")
    if _LONG_FENCE.search(normalized):
        raise MarkdownRefused(f"a fence longer than {MAX_FENCE_LENGTH} characters")


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
    lookups. This differs from it in two ways: a numeric suffix longer than
    ``_MAX_COUNT_DIGITS`` is part of the stem, and there is no table of
    contents, so a ``[TOC]`` marker stays as text.
    """

    def run(self, root: etree.Element) -> None:
        taken = {element.get("id") for element in root.iter() if element.get("id")}
        # Each ``(stem, n)`` a search walked past maps to the count it stopped
        # at: every count from n up to there was taken.
        skips: dict[tuple[str, int], int] = {}
        for element in root.iter():
            if element.tag not in _HEADINGS or "id" in element.attrib:
                continue
            name = html.unescape(_strip_tags(text=render_inner_html(element, self.md)))
            element.set("id", _unique_id(candidate=slugify(name, "-"), taken=taken, skips=skips))


def _strip_tags(*, text: str) -> str:
    """What ``toc.strip_tags`` returns for *text*, in linear time.

    It drops each ``<!--`` through the next ``-->``, then each ``<`` through
    the next ``>``, then collapses whitespace as ``str.split`` finds it.
    ``strip_tags`` itself copies the whole string after each one it drops,
    so a heading of n tags costs n² copying.
    """
    text = _drop_comments(text=text)
    kept, position = [], 0
    # Nothing before the first "<" holds one, so dropping a tag can't join
    # what was kept and what follows into a new one.
    while (start := text.find("<", position)) != -1 and (end := text.find(">", start)) != -1:
        kept.append(text[position:start])
        position = end + 1
    kept.append(text[position:])
    return " ".join("".join(kept).split())


def _drop_comments(*, text: str) -> str:
    """*text* without the comments ``toc.strip_tags`` drops, in linear time.

    ``strip_tags`` searches the joined string from its start after each
    comment, so a ``<``, ``<!`` or ``<!-`` kept just before a comment can
    open another with what follows it. Each search here tries the last three
    kept characters first. Kept text is held as slices of *text*, so giving
    those characters back copies nothing.
    """
    kept: list[tuple[int, int]] = []
    position = 0
    while True:
        tail = _kept_tail(text=text, kept=kept)
        joined = (tail + text[position : position + 3]).find("<!--")
        if 0 <= joined < len(tail):
            # Its "-->" can start in the kept characters too, as in "<!-->".
            held = tail[joined:]
            closer = (held + text[position : position + 2]).find("-->")
            if closer != -1:
                end = position + closer + 3 - len(held)
            elif (closer := text.find("-->", position)) != -1:
                end = closer + 3
            else:
                break
            _give_back(kept=kept, count=len(held))
        elif (opener := text.find("<!--", position)) != -1 and (
            closer := text.find("-->", opener)
        ) != -1:
            if opener > position:
                kept.append((position, opener))
            end = closer + 3
        else:
            break
        position = end
    return "".join(text[start:end] for start, end in kept) + text[position:]


def _kept_tail(*, text: str, kept: list[tuple[int, int]]) -> str:
    """The last three characters the slices in *kept* hold, or all of them."""
    tail = ""
    for start, end in reversed(kept):
        tail = text[max(start, end - 3 + len(tail)) : end] + tail
        if len(tail) == 3:
            break
    return tail


def _give_back(*, kept: list[tuple[int, int]], count: int) -> None:
    """Drop the last *count* characters the slices in *kept* hold."""
    while count:
        start, end = kept.pop()
        if end - start > count:
            kept.append((start, end - count))
            return
        count -= end - start


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
