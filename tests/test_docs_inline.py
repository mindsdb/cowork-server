"""Inline Markdown on the page: marks read in linear time, exactly as the one-regex-per-mark reading did."""
from __future__ import annotations

import html
import random
import re
import time

import pytest

from cowork.docs.render import _date_label, inline_html

# inline_html as it was when every mark was one regex. Italic, links and chips
# are scanned now (these regexes are quadratic on marks that never close); the
# scans must read every text the same way.
_ESCAPE_OR_CODE = re.compile(r"\\([*`~\[\]{}_\\])|`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])|(?<![_\w])_(?!\s)(.+?)(?<!\s)_(?![_\w])")
_STRIKE = re.compile(r"~~(.+?)~~")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+|mailto:[^)\s]+)\)")
_CHIP = re.compile(r"\{\{\s*(date|status|user)\s*:\s*([^}]+?)\s*\}\}")
_HELD = re.compile(r"\x00(\d+)\x00")


def _regex_inline_html(text: str) -> str:
    held: list[str] = []

    def hold(markup: str) -> str:
        held.append(markup)
        return f"\x00{len(held) - 1}\x00"

    def chip(m: re.Match[str]) -> str:
        kind, value = m.group(1), m.group(2)
        return f'<span class="chip chip-{kind}">{_date_label(value) if kind == "date" else value}</span>'

    s = text.replace("\x00", "")
    s = _ESCAPE_OR_CODE.sub(
        lambda m: hold(html.escape(m.group(1)) if m.group(1) is not None else f"<code>{html.escape(m.group(2))}</code>"),
        s,
    )
    s = _LINK.sub(lambda m: hold(f'<a href="{html.escape(m.group(2))}">') + m.group(1) + hold("</a>"), s)
    s = html.escape(s, quote=False)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _ITALIC.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", s)
    s = _STRIKE.sub(r"<s>\1</s>", s)
    s = _CHIP.sub(chip, s)
    return _HELD.sub(lambda m: held[int(m.group(1))], s)


# Pieces that open, close or break marks, and the characters the regexes' \s, \w and . treat apart.
_PIECES = [
    "*", "**", "_", "__", "~", "~~", "`", "\\", "\\*", "\\_", "[", "]", "(", ")", "](", "](http://", "](https://",
    "](https:", "](mailto:", "{", "}", "{{", "}}", "{{date:", "{{status:", "{{ user :", ":", " ", "  ", "\n", "\t",
    "\r", "\u00a0", "\u2028", "\x1c", "a", "b c", "é", "1", "&", "<", "x.com/_p_", "2026-10-21", "\x00", "\x000\x00",
]


def test_the_scans_read_any_text_as_the_regexes_did():
    rng = random.Random(3667)
    for _ in range(5000):
        text = "".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 14)))
        assert inline_html(text) == _regex_inline_html(text), repr(text)


@pytest.mark.parametrize(("text", "html_out"), [
    ("*a **b** c*", "<em>a <strong>b</strong> c</em>"),
    ("**a *b* c**", "<strong>a <em>b</em> c</strong>"),
    ("~~a *b* c~~", "<s>a <em>b</em> c</s>"),
    ("*a *b**", "<em>a *b*</em>"),
    ("*a* and _b_", "<em>a</em> and <em>b</em>"),
    ("*a* b* c*", "<em>a</em> b* c*"),
    ("_a _b_ c_", "<em>a _b</em> c_"),
    ("_a_b_", "<em>a_b</em>"),
    ("*a\nb*", "*a\nb*"),
    ("a * b * c", "a * b * c"),
    ("*[x](https://x.com/*y*)*", '<em><a href="https://x.com/*y*">x</a></em>'),
    ("[[a](http://x.io)", '<a href="http://x.io">[a</a>'),
    ("[a](http://x.io) [b](mailto:me@x.io)", '<a href="http://x.io">a</a> <a href="mailto:me@x.io">b</a>'),
    ("[a](http://x y)", "[a](http://x y)"),
    ("[a]](http://x.io)", "[a]](http://x.io)"),
    ("[](http://x.io)", "[](http://x.io)"),
    ("[a](http://)", "[a](http://)"),
    ("[a](mailto:)", "[a](mailto:)"),
    ("[a](https:x)", "[a](https:x)"),
    ("{{ date : 2026-10-21 }}", '<span class="chip chip-date">Oct 21, 2026</span>'),
    ("{{status: in progress  }} and {{user:me}}",
     '<span class="chip chip-status">in progress</span> and <span class="chip chip-user">me</span>'),
    ("{{status:   }}", '<span class="chip chip-status"> </span>'),
    ("{{user: {{status: done}}", '<span class="chip chip-user">{{status: done</span>'),
    ("{{date:a}", "{{date:a}"),
])
def test_nested_and_unclosed_marks_render_as_before(text, html_out):
    assert inline_html(text) == html_out


@pytest.mark.parametrize("text", [
    "*a " * 33_334,
    "_a " * 33_334,
    "[" * 100_000,
    "[a](http://" * 9_091,
    "{{date:" * 14_286,
    "{{date:a" + " " * 100_000,
], ids=["italic-star", "italic-underscore", "link-words", "link-address", "chip-value", "chip-spaces"])
def test_marks_that_never_close_render_in_linear_time(text):
    """A doc write renders every block under the doc lock; the regexes took 5-25 s on each of these."""
    start = time.perf_counter()
    inline_html(text)
    assert time.perf_counter() - start < 1.0
