"""Markdown renders to HTML that the inliner sanitizes afterwards."""
from __future__ import annotations

import re
import time

import markdown
import pytest
from markdown.extensions.toc import TocExtension

from cowork.services.markdown_render import render_markdown
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
    ],
)
def test_heading_ids_match_what_toc_extension_gives(headings):
    text = "".join(f"## {heading}\n\n" for heading in headings)
    oracle = markdown.Markdown(extensions=[TocExtension()], output_format="html")

    assert _heading_ids(html=render_markdown(text=text)) == _heading_ids(html=oracle.convert(text))


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
