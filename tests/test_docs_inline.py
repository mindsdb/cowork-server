"""Inline Markdown on the page: the cases shared with the editor, tags that always nest, and linear time."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pytest

from cowork.docs.render import inline_html
from tests._linear_time import MAX_LINEAR_GROWTH, growth

# The editor (cowork docModel.inlineToHtml) runs an identical copy of this file.
_CASES_BYTES = (Path(__file__).parent / "fixtures" / "docs_inline" / "cases.json").read_bytes()
_SHARED = json.loads(_CASES_BYTES)
# The same digest is pinned in cowork's docModel.inline.test.js.
_CASES_SHA256 = "6297ddd5c762e12e18f2353dfce5869093267e0458001f22d71ad7256bd5cf0c"


def test_the_shared_cases_are_the_pinned_copy():
    """A change to the cases stops here until its digest is pinned: the reminder to copy the file to cowork
    and pin the same digest there. Neither suite reads the other repo's copy."""
    assert hashlib.sha256(_CASES_BYTES).hexdigest() == _CASES_SHA256


@pytest.mark.parametrize("case", _SHARED["cases"], ids=[c["name"] for c in _SHARED["cases"]])
def test_a_shared_case_renders_as_the_editor_renders_it(case):
    assert inline_html(case["text"]) == case["html"]


_TAG = re.compile(r"<(/?)(\w+)[^>]*>")


def _nests(markup: str) -> bool:
    stack: list[str] = []
    for closing, tag in _TAG.findall(markup):
        if tag == "br":
            continue
        if not closing:
            stack.append(tag)
        elif not stack or stack.pop() != tag:
            return False
    return not stack


# Pieces that open, close or break marks, and characters CommonMark counts as white space or punctuation.
_PIECES = [
    "*", "**", "***", "_", "__", "~", "~~", "`", "\\", "\\*", "\\_", "\\`", "[", "]", "(", ")", "](", "](http://x)",
    "](https://", "](mailto:a)", "{{", "}}", "{{date:", "{{ status :", "2026-10-21", " ", "\n", "\r", "\u00a0",
    "\u2028", "\u3000", "a", "b c", "é", "1", ".", "&", "<", '"', "\U0001F600", "\x00",
]


def test_any_text_renders_to_tags_that_nest_and_nothing_else():
    """A stray close tag (``<em>a <a href=…>b</em></a>``) broke the page and the exports that parse it."""
    rng = random.Random(3667)
    for _ in range(5000):
        text = "".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 24)))
        out = inline_html(text)
        assert _nests(out), repr(text)
        assert {tag for _, tag in _TAG.findall(out)} <= {"em", "strong", "s", "code", "a", "span", "br"}, repr(text)
        assert "<" not in _TAG.sub("", out), repr(text)


@pytest.mark.parametrize(("make", "n"), [
    (lambda n: "*a " * n, 33_334),
    (lambda n: "_a " * n, 33_334),
    (lambda n: "[" * n, 100_000),
    (lambda n: "[a](http://" * n, 9_091),
    (lambda n: "{{date:" * n, 14_286),
    (lambda n: "{{date:a" + " " * n, 100_000),
    (lambda n: "*a " * n + "b* " * n, 16_667),
    (lambda n: "[" * n + "[a](http://x)" * (n // 12), 50_000),
    (lambda n: "[" * n + "](http://a" * (n // 10), 50_000),
    (lambda n: "*_" * n, 50_000),
    (lambda n: "*a**" * n, 25_000),
    (lambda n: "a* " * n + " *a" * n, 16_667),
    (lambda n: " *a" * n + "b" + "*" * n, 25_000),
    (lambda n: "*" * n + "a" + "b* " * n, 25_000),
], ids=[
    "italic-star", "italic-underscore", "link-words", "link-address", "chip-value", "chip-spaces",
    "nested", "brackets-before-links", "brackets-before-addresses", "mixed-marks", "rule-of-3",
    "closers-before-openers", "one-run-opens-many", "one-run-closes-many",
])
def test_adversarial_marks_render_in_linear_time(make, n):
    """A doc write renders every block under the doc lock; quadratic readings took 5-25 s on 100k characters
    like these, and rebuilding a long run's tags on every pairing took 10 s at 320k marks."""
    assert growth(inline_html, small=make(n // 10), large=make(n)).ratio < MAX_LINEAR_GROWTH
