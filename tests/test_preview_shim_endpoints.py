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


def test_proxy_root_document_gets_the_shim():
    # Fullstack previews never reach the file endpoints above; the proxy is
    # their equivalent injection point. Storage is native there, so the shim's
    # guard skips the substitutes — the error reporter is what it is for.
    import inspect

    from cowork.services import preview_proxy

    source = inspect.getsource(preview_proxy)
    assert "prepare_preview_html" in source
    assert "inject_layer(" not in source


def test_services_do_not_import_from_the_api_layer():
    # preview_proxy is the first services module that needed the composition,
    # and cowork/services/ holds no imports from cowork.api today. Keep it that
    # way: the reverse edge is where import cycles start.
    import inspect

    from cowork.services import preview_proxy

    assert "from cowork.api" not in inspect.getsource(preview_proxy)
