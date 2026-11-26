"""Code blocks on a doc's page: Pygments tokens with the editor's highlight.js classes.

``fixtures/docs_code_highlight/contract.json`` is shared with cowork
(components/markdown/hljs.js); see its README.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import pytest
from pygments.lexers import get_lexer_by_name

from cowork.docs import highlight as hl

_CONTRACT_BYTES = (Path(__file__).parent / "fixtures" / "docs_code_highlight" / "contract.json").read_bytes()
# The same digest is pinned in cowork's components/markdown/hljs.test.js.
_CONTRACT_SHA256 = "727358beaa992a4df007fa34334fa52877b73e63c5c500c97331d96f73dc06cd"
_CONTRACT = json.loads(_CONTRACT_BYTES)
_PIECE = re.compile(r'<span class="hljs-([\w-]+)">|</span>|[^<]+')


def _read(markup: str) -> tuple[str, list[tuple[str, int, int, int]]]:
    """The text ``markup`` shows, and each span as (class, start, end, depth)."""
    text, spans, open_ = "", [], []
    for m in _PIECE.finditer(markup):
        if m.group(1):
            open_.append((m.group(1), len(text), len(open_)))
        elif m.group(0) == "</span>":
            cls, start, depth = open_.pop()
            spans.append((cls, start, len(text), depth))
        else:
            text += html.unescape(m.group(0))
    assert not open_
    return text, spans


def _class_of(text: str, spans: list[tuple[str, int, int, int]], sub: str) -> str | None:
    start = text.find(sub)
    assert start >= 0, sub
    end = start + len(sub)
    holding = sorted((s for s in spans if s[1] <= start and s[2] >= end), key=lambda s: -s[3])
    if holding:
        return holding[0][0]
    return "straddles a span" if any(s[1] < end and s[2] > start for s in spans) else None


def test_the_fixture_is_the_copy_cowork_pins():
    assert hashlib.sha256(_CONTRACT_BYTES).hexdigest() == _CONTRACT_SHA256


def test_the_languages_limit_and_palette_are_the_contracts():
    assert hl.LANGUAGES == {k: (v["name"], v["pygments"]) for k, v in _CONTRACT["languages"].items()}
    assert hl.MAX_CHARS == _CONTRACT["max_chars"]
    assert hl.COLORS == _CONTRACT["colors"]
    for _, lexer in hl.LANGUAGES.values():
        get_lexer_by_name(lexer)


@pytest.mark.parametrize("spelling", sorted(_CONTRACT["anchors"]))
def test_the_main_tokens_take_the_editors_classes(spelling):
    anchor = _CONTRACT["anchors"][spelling]
    name, markup = hl.highlight(anchor["code"], spelling)
    assert name == _CONTRACT["languages"][spelling]["name"]
    text, spans = _read(markup)
    assert text == anchor["code"]
    assert [[sub, _class_of(text, spans, sub)] for sub, _ in anchor["expect"]] == anchor["expect"]


@pytest.mark.parametrize("language", ["python", "py", "PY", " py "])
def test_a_spelling_reads_in_any_case_and_around_spaces(language):
    assert hl.highlight("x = 1", language)[0] == "python"


@pytest.mark.parametrize("code, language", [
    ("x = 1", "java"),
    ("x = 1", "svg"),
    ("x = 1", ""),
    ("x = 1", None),
    ("a = 1\r\nb = 2", "python"),
    ("a = 1\rb = 2", "python"),
    (chr(0xFEFF) + "x = 1", "python"),
    ("x" * (hl.MAX_CHARS + 1), "python"),
])
def test_what_the_page_cannot_colour_as_the_editor_does_stays_plain(code, language):
    assert hl.highlight(code, language) is None


def test_a_block_right_at_the_limit_is_highlighted():
    assert hl.highlight("x" * hl.MAX_CHARS, "python") is not None


def test_the_code_is_escaped_and_reads_back_whole():
    code = "\n\n<script>alert('x & y')</script>\n\n"
    _, markup = hl.highlight(code, "html")
    assert "<script" not in markup
    assert _read(markup)[0] == code


def test_a_last_line_without_a_newline_is_kept():
    # Pygments' session lexer (console) drops a last line that has no newline.
    code = "$ ls -la\nfile.txt"
    assert _read(hl.highlight(code, "shell")[1])[0] == code


def test_neighbouring_tokens_of_one_class_share_a_span():
    # Pygments splits 4px into a number and a unit; hljs gives one number.
    assert '<span class="hljs-number">4px</span>' in hl.highlight("a{margin:4px}", "css")[1]


def test_the_cache_is_bounded():
    assert hl._render.cache_info().maxsize == 128


def test_the_css_gives_each_class_its_colour_and_background():
    css = hl.code_css("light")
    assert ".hljs-keyword{color:rgb(124, 58, 175)}" in css
    assert ".hljs-addition{color:rgb(20, 121, 79)}" in css
    assert ".hljs-addition{background:rgba(20, 121, 79, 0.12)}" in css
    assert ".hljs-comment{color:var(--ink-4)}" in css
