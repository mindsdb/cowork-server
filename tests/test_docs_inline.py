"""Inline Markdown on the page: the cases shared with the editor, tags that always nest, and linear time."""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pytest

from cowork.docs.render import inline_html

# The editor (cowork docModel.inlineToHtml) runs an identical copy of this file.
_SHARED = json.loads((Path(__file__).parent / "fixtures" / "docs_inline" / "cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _SHARED["cases"], ids=[c["name"] for c in _SHARED["cases"]])
def test_a_shared_case_renders_as_the_editor_renders_it(case):
    assert inline_html(case["text"]) == case["html"]


_TAG = re.compile(r"<(/?)(\w+)[^>]*>")


def _nests(markup: str) -> bool:
    stack: list[str] = []
    for closing, tag in _TAG.findall(markup):
        if not closing:
            stack.append(tag)
        elif not stack or stack.pop() != tag:
            return False
    return not stack


# Pieces that open, close or break marks, and characters CommonMark counts as white space or punctuation.
_PIECES = [
    "*", "**", "***", "_", "__", "~", "~~", "`", "\\", "\\*", "\\_", "\\`", "[", "]", "(", ")", "](", "](http://x)",
    "](https://", "](mailto:a)", "{{", "}}", "{{date:", "{{ status :", "2026-10-21", " ", "\n", "\r", " ",
    " ", "　", "a", "b c", "é", "1", ".", "&", "<", '"', "\U0001F600", "\x00",
]


def test_any_text_renders_to_tags_that_nest_and_nothing_else():
    """A stray close tag (``<em>a <a href=…>b</em></a>``) broke the page and the exports that parse it."""
    rng = random.Random(3667)
    for _ in range(5000):
        text = "".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 24)))
        out = inline_html(text)
        assert _nests(out), repr(text)
        assert {tag for _, tag in _TAG.findall(out)} <= {"em", "strong", "s", "code", "a", "span"}, repr(text)
        assert "<" not in _TAG.sub("", out), repr(text)


@pytest.mark.parametrize("text", [
    "*a " * 33_334,
    "_a " * 33_334,
    "[" * 100_000,
    "[a](http://" * 9_091,
    "{{date:" * 14_286,
    "{{date:a" + " " * 100_000,
    "*a " * 33_334 + "b* " * 33_334,
    "[" * 50_000 + "[a](http://x)" * 4_000,
    "[" * 50_000 + "](http://a" * 5_000,
    "*_" * 50_000,
    "*a**" * 25_000,
    "a* " * 33_334 + " *a" * 33_334,
], ids=[
    "italic-star", "italic-underscore", "link-words", "link-address", "chip-value", "chip-spaces",
    "nested", "brackets-before-links", "brackets-before-addresses", "mixed-marks", "rule-of-3",
    "closers-before-openers",
])
def test_adversarial_marks_render_in_linear_time(text):
    """A doc write renders every block under the doc lock; quadratic readings took 5-25 s on inputs like these."""
    start = time.perf_counter()
    inline_html(text)
    assert time.perf_counter() - start < 1.0
