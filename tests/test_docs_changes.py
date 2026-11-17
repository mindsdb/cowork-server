"""Version-history summaries: a block's words read in linear time, exactly as the one-regex-per-mark reading did."""
from __future__ import annotations

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


@pytest.mark.parametrize(("opener", "tail"), [
    ("[", ""), ("[", "]"), ("[](", ""), ("{{", ""), ("{{", "}"), ("{{date:", ""),
], ids=["link-words", "link-no-address", "link-address", "chip-value", "chip-one-brace", "chip-date"])
def test_marks_that_never_close_are_read_in_linear_time(opener, tail):
    """The regexes took 7-26 s on 100k characters of these. A scan that restarts at every
    opener passes at 100k but takes seconds at 1M; the shorter text goes first, so a regex fails fast."""
    for size in (100_000, 1_000_000):
        text = opener * (size // len(opener)) + tail
        start = time.perf_counter()
        _plain(text)
        assert time.perf_counter() - start < 1.0, f"{size:,} characters"


def _table_doc(columns: list[str], rows: list[list[str]]) -> dict:
    return {"title": "Plan", "tabs": [{"id": "t1", "title": "Plan", "blocks": [{
        "id": "b1", "type": "table",
        "columns": [{"id": f"c{n}", "name": name} for n, name in enumerate(columns)],
        "rows": [{"id": f"r{n}", "cells": {f"c{k}": v for k, v in enumerate(cells)}} for n, cells in enumerate(rows)],
    }]}]}


@pytest.mark.parametrize(("columns", "before", "after", "summary"), [
    (["Channel", "Budget"], ["", "72"], ["Paid search", "72"], "Named a row “Paid search”"),
    (["Channel", "Budget"], ["Paid search", "72"], ["", "72"], "Cleared a row's name"),
    (["Channel", "Budget", "Signups"], ["", "72", "1"], ["", "80", "1"], "Budget 72 → 80"),
    (["Channel", "Budget", "Signups"], ["[Paid](x)", "72", "1"], ["[Paid](x)", "80", "2"],
     "Paid · Budget 72 → 80, and 1 more cell"),
])
def test_a_cell_names_its_row_and_column(columns, before, after, summary):
    assert describe(_table_doc(columns, [before]), _table_doc(columns, [after])) == summary


def test_a_summary_of_a_table_with_long_names_is_quick():
    """Each changed cell's line names its row and column; reading those names per cell took 4 s here."""
    columns = [f"{n} " + "a" * 20_000 for n in range(40)]
    old = _table_doc(columns, [[f"Row {r}"] + ["1"] * 39 for r in range(250)])
    new = _table_doc(columns, [[f"Row {r}"] + ["2"] * 39 for r in range(250)])
    start = time.perf_counter()
    summary = describe(old, new)
    assert time.perf_counter() - start < 1.0
    assert summary.endswith(f", and {250 * 39 - 1} more cells")
