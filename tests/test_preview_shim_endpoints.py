"""Every HTML preview response carries the shim, with or without comments.

The shim is not opt-in the way the comment layer is: a page without the
activation flag needs its storage substitutes just as much.
"""

from __future__ import annotations

from cowork.services.preview_html import prepare_preview_html

_HTML = "<html><head></head><body><h1>Report</h1></body></html>"


def test_shim_is_present_without_comments():
    out = prepare_preview_html(_HTML, comments=False)
    assert "anton-preview" in out
    assert "anton-comments" not in out


def test_comments_layer_rides_on_top_of_the_shim():
    out = prepare_preview_html(_HTML, comments=True)
    assert out.index("anton-preview") < out.index("anton-comments")
    assert out.index("anton-comments") < out.index("</body>")
