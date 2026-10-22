"""Markdown renders to HTML that the inliner sanitizes afterwards."""
from __future__ import annotations

import pytest

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
