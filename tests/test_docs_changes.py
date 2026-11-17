"""Version-history summaries: a block's words read in linear time, exactly as the one-regex-per-mark reading did."""
from __future__ import annotations

import copy
import random
import re
import time

import pytest

from cowork.docs.changes import _EMPHASIS, _plain, describe

# _plain as it was when links and chips were one regex each. These regexes are
# quadratic on marks that never close; the scans must read every text the same way.
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_CHIP = re.compile(r"\{\{(?:date:)?([^}]*)\}\}")


def _regex_plain(text: object) -> str:
    t = _LINK.sub(r"\1", str(text or ""))
    t = _CHIP.sub(lambda m: m.group(1).removeprefix("pending:").strip(), t)
    return " ".join(_EMPHASIS.sub("", t).split())


# Pieces that open, close or break links and chips, and the marks and spaces _plain drops.
_PIECES = [
    "[", "]", "(", ")", "](", "[a", "a]", "](x)", "[]", "()", "{", "}", "{{", "}}", "{{date:", "date:", "pending:",
    "{{pending:", "{{ date :", "*", "**", "`", "~", "~~", " ", "  ", "\n", "\t", " ", "a", "b c", "é", "1",
]


def test_the_scans_read_any_text_as_the_regexes_did():
    rng = random.Random(3667)
    for _ in range(20_000):
        text = "".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 16)))
        assert _plain(text) == _regex_plain(text), repr(text)


@pytest.mark.parametrize(("text", "plain"), [
    ("[Q3 plan](https://x.io/q3) is **due**", "Q3 plan is due"),
    ("[a](x) and [b](y)", "a and b"),
    ("[a[b](x)", "a[b"),
    ("[a]b](x)", "[a]b](x)"),
    ("[a] [b](x)", "[a] b"),
    ("[](x)", ""),
    ("[a]()", "a"),
    ("[x](a(b)c)", "xc)"),
    ("[a](x", "[a](x"),
    ("{{date: 2026-10-21 }}", "2026-10-21"),
    ("{{pending: write the intro}}", "write the intro"),
    ("{{date:pending: x}}", "x"),
    ("{{status: done}}", "status: done"),
    ("{{a{{b}}", "a{{b"),
    ("{{a}", "{{a}"),
    ("{{a} b}} {{c}}", "{{a} b}} c"),
    ("[{{a}}](x)", "a"),
    ("{{[a](x)}}", "a"),
    ("**bold** `code` ~~gone~~ ~x~", "bold code gone ~x~"),
    ("a\n\n b", "a b"),
    (None, ""),
    (0, ""),
    (72, "72"),
])
def test_links_chips_and_marks_read_as_before(text, plain):
    assert _plain(text) == plain == _regex_plain(text)


@pytest.mark.parametrize("text", [
    "[" * 100_000,
    "[" * 100_000 + "]",
    "[](" * 100_000,
    "{{" * 50_000,
    "{{" * 50_000 + "}",
    "{{date:" * 14_286,
], ids=["link-words", "link-no-address", "link-address", "chip-value", "chip-one-brace", "chip-date"])
def test_marks_that_never_close_are_read_in_linear_time(text):
    """The regexes took 7-25 s on each of these."""
    start = time.perf_counter()
    _plain(text)
    assert time.perf_counter() - start < 1.0


def test_a_summary_of_blocks_full_of_open_marks_is_quick():
    old = {"title": "Plan", "tabs": [{"id": "t1", "title": "Plan", "blocks": [
        {"id": "b1", "type": "paragraph", "text": "[" * 100_000},
        {"id": "b2", "type": "paragraph", "text": "a"},
    ]}]}
    new = copy.deepcopy(old)
    new["tabs"][0]["blocks"][0]["text"] = "{{" * 50_000
    new["tabs"][0]["blocks"][1]["text"] = "[](" * 100_000
    start = time.perf_counter()
    summary = describe(old, new)
    assert time.perf_counter() - start < 1.0
    assert summary == "Edited “{{{{{{{{{{{{{{{{{{{{{{{{{{{{…”, and 1 more change"
