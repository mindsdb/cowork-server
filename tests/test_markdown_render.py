"""Markdown renders to HTML that the inliner sanitizes afterwards."""
from __future__ import annotations

import re
import time

import markdown
import pytest
from markdown.extensions.toc import TocExtension, strip_tags

from cowork.services import markdown_render
from cowork.services.markdown_render import MAX_FENCE_LENGTH, MarkdownRefused, render_markdown
from tests._no_fetch import no_fetch, recording_server


def test_a_table_renders_as_an_html_table():
    html = render_markdown(text="| a | b |\n|---|---|\n| 1 | 2 |\n")

    assert "<table>" in html
    assert "<th>a</th>" in html
    assert "<td>2</td>" in html


def test_fenced_code_keeps_its_whitespace_and_escapes_its_text():
    html = render_markdown(text="```python\nif a < b:\n    pass\n```\n")

    assert '<pre><code class="language-python">if a &lt; b:\n    pass\n</code></pre>' in html


def test_ordered_and_bulleted_lists_stay_separate():
    html = render_markdown(text="1. one\n2. two\n\n* three\n")

    assert "<ol>\n<li>one</li>\n<li>two</li>\n</ol>" in html
    assert "<ul>\n<li>three</li>\n</ul>" in html


def test_void_elements_render_as_html_not_xhtml():
    assert render_markdown(text="a  \nb\n\n---\n") == "<p>a<br>\nb</p>\n<hr>"


def test_headings_get_ids():
    assert '<h2 id="results">Results</h2>' in render_markdown(text="## Results\n")


def _heading_ids(*, html: str) -> list[str]:
    return re.findall(r'<h[1-6] id="([^"]*)"', html)


# A published copy renders with TocExtension, so a #fragment link must land
# on the same id in both.
@pytest.mark.parametrize(
    "headings",
    [
        ["Results", "Results", "Results"],
        ["a", "a", "a_1", "a_1", "a"],
        ["a_5", "a_5", "a", "a"],
        ["", "", "_1"],
        ["The *big* result", "Use `a<b`", "Fish &amp; chips", "1 < 2 > 0", "žlutý"],
        ["Note <em>raw</em> html", "Note raw html"],
        ["a_99999999999", "a_99999999999"],
        ["Results <!-- old -> new -->", "Results"],
        ["a <?x <y ?> b"],
        ["a <![CDATA[ <b ]]> c"],
        ["a\u2028b", "a\u2029b", "a\x85b", "a\u1680b"],
    ],
)
def test_heading_ids_match_what_toc_extension_gives(headings):
    text = "".join(f"## {heading}\n\n" for heading in headings)
    oracle = markdown.Markdown(extensions=[TocExtension()], output_format="html")

    assert _heading_ids(html=render_markdown(text=text)) == _heading_ids(html=oracle.convert(text))


# toc.strip_tags searches from the start again after each comment it drops,
# so a "<", "<!" or "<!-" kept before one can open another with what follows.
@pytest.mark.parametrize(
    "text",
    [
        "<!<!-- x -->-- a > b -->",
        "<<!-- -->!-- --> x",
        "<!-<!-- -->-> y",
        "<<<!-- -->!-- -->!-- --> z",
        "a <!--> b",
        "a <!---> b",
        "a <!-- b",
        "a <b c",
        "a <b> <!-- c > d --> e",
        "a\u2028\x85\u1680 b",
    ],
)
def test_tags_strip_as_toc_strip_tags_strips_them(text):
    assert markdown_render._strip_tags(text=text) == strip_tags(text)


# Each shape is linear for the port; toc.strip_tags copies the whole string
# after each tag or comment it drops, and a cascade gives back one kept
# character per comment.
@pytest.mark.parametrize(
    "text",
    [
        "<b>" * 200_000,
        "<!---->" * 200_000,
        "<" * 200_000,
        "<!-- x" * 200_000,
        "<" * 200_000 + "<!-- -->" + "!-- -->" * 200_000,
    ],
    ids=["tags", "comments", "unclosed-tags", "unclosed-comments", "cascade"],
)
def test_tags_strip_in_linear_time(text):
    started = time.perf_counter()
    markdown_render._strip_tags(text=text)

    assert time.perf_counter() - started < 1.0


def _render_seconds(*, text: str) -> float:
    started = time.perf_counter()
    render_markdown(text=text)
    return time.perf_counter() - started


@pytest.mark.parametrize("heading", ["a", "a_100"])
def test_repeated_headings_take_no_longer_than_distinct_ones(heading):
    """TocExtension counted past every id a repeat took, so 20,000 repeats took minutes."""
    count = 20_000
    distinct = _render_seconds(text="".join(f"# h{index}\n\n" for index in range(count)))
    repeated = _render_seconds(text=f"# {heading}\n\n" * count)

    assert repeated < 3 * distinct + 0.5


def test_raw_html_and_image_references_pass_through_unsanitized():
    """Sanitizing is the inliner's job; the renderer hands it everything."""
    html = render_markdown(
        text=(
            '<div onclick="steal()">raw</div>\n\n'
            "![chart](http://169.254.169.254/latest/meta-data)\n"
        )
    )

    assert '<div onclick="steal()">raw</div>' in html
    assert 'src="http://169.254.169.254/latest/meta-data"' in html


@pytest.mark.parametrize(("quote", "marker"), [("", "* "), ("", "- "), ("", "1. "), ("> ", "* ")])
def test_nesting_too_deep_to_parse_comes_back_as_escaped_text(quote, marker):
    text = "".join(
        quote + "    " * depth + marker + f"<b>{depth}</b>\n" for depth in range(400)
    )

    html = render_markdown(text=text)

    assert html.startswith("<pre>")
    assert html.endswith("</pre>")
    assert "&lt;b&gt;399&lt;/b&gt;" in html
    assert "<b>" not in html


@pytest.mark.parametrize(
    "text",
    [
        "`" * (MAX_FENCE_LENGTH + 1) + "\ncode\n" + "`" * (MAX_FENCE_LENGTH + 1) + "\n",
        "~" * (MAX_FENCE_LENGTH + 1) + "python\n",
        "intro\n\n   " + "`" * (MAX_FENCE_LENGTH + 1) + "\n",
        "intro\r" + "~" * (MAX_FENCE_LENGTH + 1) + "\r",
        "intro\n\x02" + "`" * (MAX_FENCE_LENGTH + 1) + "\n",
        "intro\n\x03" + "`" * (MAX_FENCE_LENGTH + 1) + "\n",
        # A ladder of ever-longer fences, none of them closed.
        "".join("~" * length + "\n" for length in range(3, 200)),
    ],
    ids=["backticks", "tildes", "indented-three", "carriage-returns", "after-stx", "after-etx", "ladder"],
)
def test_a_fence_longer_than_the_cap_is_refused(text):
    with pytest.raises(MarkdownRefused):
        render_markdown(text=text)


@pytest.mark.parametrize(
    "text",
    [
        "`" * MAX_FENCE_LENGTH + "\ncode\n" + "`" * MAX_FENCE_LENGTH + "\n",
        "~" * MAX_FENCE_LENGTH + "\ncode\n" + "~" * MAX_FENCE_LENGTH + "\n",
        # Four spaces make an indented code block, not a fence.
        "    " + "`" * (MAX_FENCE_LENGTH + 1) + "\n",
    ],
    ids=["backticks", "tildes", "indented-four"],
)
def test_a_fence_at_the_cap_renders(text):
    assert render_markdown(text=text)


def test_rendering_makes_no_network_or_process_call():
    with recording_server() as server:
        with no_fetch() as recorder:
            render_markdown(
                text=(
                    f"[TOC]\n\n# Title\n\n![x]({server.url}/img.png)\n\n"
                    f'<img src="{server.url}/raw.png">\n\n[link]({server.url}/page)\n'
                )
            )
        assert recorder.calls == []
        assert server.hits() == []
