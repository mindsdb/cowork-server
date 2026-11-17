"""Inline Markdown on the page: the cases shared with the editor, tags that always nest, and linear time."""
from __future__ import annotations

import gc
import hashlib
import json
import random
import re
import time
from pathlib import Path

import pytest

from cowork.docs.render import inline_html

# The editor (cowork docModel.inlineToHtml) runs an identical copy of this file.
_CASES_FILE = Path(__file__).parent / "fixtures" / "docs_inline" / "cases.json"
_SHARED = json.loads(_CASES_FILE.read_text(encoding="utf-8"))
# The same digest is pinned in cowork's docModel.inline.test.js.
_CASES_SHA256 = "9d979b16ea87cc93abe4e00acdd19f1927c2db08bdd0cd290aeb6b440970603f"


def test_the_shared_cases_are_the_copy_the_editor_runs():
    """Changed here alone, the page and the editor drift apart unseen: copy the file to cowork and pin the new digest in both."""
    assert hashlib.sha256(_CASES_FILE.read_bytes()).hexdigest() == _CASES_SHA256


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
        assert {tag for _, tag in _TAG.findall(out)} <= {"em", "strong", "s", "code", "a", "span", "br"}, repr(text)
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


def _seconds(text: str) -> float:
    """The best of three runs, without the collector, whose passes grow with the rest of the test session's heap."""
    best = float("inf")
    gc.disable()
    try:
        for _ in range(3):
            start = time.perf_counter()
            inline_html(text)
            best = min(best, time.perf_counter() - start)
    finally:
        gc.enable()
    return best


@pytest.mark.parametrize("make", [
    lambda n: " *a" * n + "b" + "*" * n,
    lambda n: "*" * n + "a" + "b* " * n,
], ids=["one-run-opens-many", "one-run-closes-many"])
def test_a_long_run_paired_many_times_renders_in_linear_time(make):
    """Rebuilding a run's tags on every pairing grew with the square of the run (10 s at 320k marks)."""
    assert _seconds(make(80_000)) < 8 * _seconds(make(20_000))
