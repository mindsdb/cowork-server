"""Markdown lines read in linear time, exactly as the one-regex-per-line reading did."""
from __future__ import annotations

import contextlib
import random
import re
import time

import pytest

from cowork.docs.markdown import _read_heading, _read_pending, _read_table, parse_markdown
from cowork.docs.model import MAX_TABLE_COLS, MAX_TABLE_ROWS, DocError

# The regexes these lines were read with. A run of spaces inside the line made
# them quadratic (cubic for an empty pending line); the readers must read every
# line the same way.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_PENDING = re.compile(r"^\s*\{\{\s*pending\s*:\s*(.*?)\s*\}\}\s*$", re.I)
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")

# Characters both \s and str.strip() treat as spaces, and one neither does (U+200B).
_SPACES = [" ", "  ", "\t", "\r", *map(chr, (0x0B, 0x0C, 0x1C, 0x85, 0xA0, 0x2028, 0x3000, 0x200B))]
# "pending" written with letters re.I also reads as i: dotless i and dotted capital I.
_ODD_PENDING = ["pend" + chr(0x131) + "ng", "PEND" + chr(0x130) + "NG"]


def _blocks(reader, *lines: str):
    read = reader(list(lines), 0, False)
    return read[0] if read else None


def _regex_heading(line: str):
    m = _HEADING.match(line)
    return [{"type": "heading", "level": min(3, len(m.group(1))), "text": m.group(2)}] if m else None


def _regex_pending(line: str):
    m = _PENDING.match(line)
    return [{"type": "pending", "intent": m.group(1)}] if m else None


def _lines(seed: int, heads: list[str], pieces: list[str], count: int = 5000):
    """Random lines: one of ``heads``, then up to ten ``pieces``. A line never holds a newline."""
    rng = random.Random(seed)
    for _ in range(count):
        yield rng.choice(heads) + "".join(rng.choice(pieces) for _ in range(rng.randint(0, 10)))


def test_strip_removes_exactly_what_backslash_s_matches():
    every = "".join(map(chr, range(0x110000)))
    assert re.findall(r"\s", every) == [c for c in every if not c.strip()]


def test_headings_read_as_the_regex_did():
    heads = ["#", "##", "###", "######", "#######", "# ", "## ", " #", "a", ""]
    pieces = ["#", "##", "a", "b c", "C#", "é", "}", *_SPACES]
    for line in _lines(1, heads, pieces):
        assert _blocks(_read_heading, line) == _regex_heading(line), repr(line)


def test_pending_lines_read_as_the_regex_did():
    heads = ["{{pending:", "{{ pending :", " {{PENDING:", "{{Pending ", "{{pending", "{pending:", "{{", "",
             *("{{" + word + ":" for word in _ODD_PENDING)]
    pieces = ["{", "{{", "}", "}}", "}}}", ":", "a", "b c", "pending", *_SPACES]
    for line in _lines(2, heads, pieces):
        assert _blocks(_read_pending, line) == _regex_pending(line), repr(line)


def test_table_separators_read_as_the_regex_did():
    heads = ["", "|", " |", ":", "-", "--", "|--", "| :--- |"]
    pieces = ["|", "-", "--", ":", "x", "|--", *_SPACES]
    for line in _lines(3, heads, pieces):
        assert (_blocks(_read_table, "| a |", line) is not None) == bool(_TABLE_SEP.match(line)), repr(line)


@pytest.mark.parametrize(("md", "blocks"), [
    ("# Title", [("heading", 1, "Title")]),
    ("## Learning C#", [("heading", 2, "Learning C#")]),
    ("### Done ###  ", [("heading", 3, "Done")]),
    ("#### a ## ##", [("heading", 3, "a ##")]),
    ("# a # b", [("heading", 1, "a # b")]),
    ("# ##", [("heading", 1, "##")]),
    ("#" + chr(0xA0) + "a" + chr(0xA0) + "#", [("heading", 1, "a")]),
    ("#Title", [("paragraph", None, "#Title")]),
    ("####### seven", [("paragraph", None, "####### seven")]),
    ("{{pending: the chart}}", [("pending", None, "the chart")]),
    ("  {{ PENDING : a }} }}  ", [("pending", None, "a }}")]),
    ("{{pending: a}}}", [("pending", None, "a}")]),
    ("{{pending:}}", [("pending", None, "Writing…")]),
    ("{{pending: a} ", [("paragraph", None, "{{pending: a}")]),
])
def test_lines_parse_as_before(md, blocks):
    got = [(b["type"], b.get("level"), b.get("intent", b.get("text"))) for b in parse_markdown(md)]
    assert got == blocks


@pytest.mark.parametrize("sep", ["|:--|--:|", "  | -- | :---: |  ", "--|--", "| --- |---|"])
def test_spaced_separators_still_make_tables(sep):
    assert [b["type"] for b in parse_markdown(f"| a | b |\n{sep}\n| 1 | 2 |")] == ["table"]


def _table(cols: int, rows: int) -> str:
    return "\n".join(["|" + " x |" * cols, "|" + " --- |" * cols, *(["|" + " 1 |" * cols] * rows)])


def test_a_table_at_the_limits_reads_and_one_past_them_is_refused_as_before():
    (table,) = parse_markdown(_table(MAX_TABLE_COLS, MAX_TABLE_ROWS))
    assert (len(table["columns"]), len(table["rows"])) == (MAX_TABLE_COLS, MAX_TABLE_ROWS)
    for cols, rows, message in [(MAX_TABLE_COLS + 1, 1, f"at most {MAX_TABLE_COLS} columns"),
                                (MAX_TABLE_COLS, MAX_TABLE_ROWS + 1, f"at most {MAX_TABLE_ROWS} rows")]:
        with pytest.raises(DocError, match=message) as refused:
            parse_markdown(_table(cols, rows))
        assert refused.value.code == "bad_table"


# HTTP create takes Markdown up to this length.
_LONGEST = 2_000_000


@pytest.mark.parametrize("md", [
    "# a" + " " * _LONGEST + "b",
    "{{pending:" + " " * _LONGEST,
    "{{pending: a" + " " * _LONGEST + "b",
    "| a |\n" + " " * _LONGEST + "x",
    "| a |\n|--" + " " * _LONGEST + "x",
    "|" * (_LONGEST // 2) + "\n|--|\n" + "|\n" * (_LONGEST // 4),
    "|" * (_LONGEST // 2) + "\n|--|",
    "| a |\n|--|\n" + "| b |\n" * (_LONGEST // 6),
], ids=["heading", "pending-empty", "pending", "separator-indent", "separator-tail", "wide-table", "wide-header",
        "long-table"])
def test_hostile_markdown_parses_in_linear_time(md):
    """One doc write parses its Markdown under the doc lock. The regexes took hours
    on these lines, the wide table ran out of memory, and the wide header and the
    long table took 5-10 s. The bound leaves room for coverage on a slow runner."""
    start = time.perf_counter()
    with contextlib.suppress(DocError):
        parse_markdown(md)
    assert time.perf_counter() - start < 2.0
