"""Every HTML preview response carries the shim, with or without comments.

The shim is not opt-in the way the comment layer is: a page without the
activation flag needs its storage substitutes just as much.
"""

from __future__ import annotations

import pytest

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


@pytest.mark.asyncio
@pytest.mark.parametrize("query, expect_comments", [(b"", False), (b"__antonComments=1", True)])
async def test_proxy_root_document_gets_the_shim(tmp_path, monkeypatch, query, expect_comments):
    # Fullstack previews never reach the file endpoints above; the proxy is
    # their equivalent injection point. Storage is native there, so the shim's
    # guard skips the substitutes — the error reporter is what it is for.
    # Driven through proxy_artifact_request with a stub upstream, so the test
    # proves the injection reaches the response rather than that a name
    # appears in the module's source.
    from starlette.requests import Request as StarletteRequest

    from cowork.services import artifacts, preview_proxy

    class _Upstream:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8", "content-length": "60"}

        async def aread(self):
            return b'<html><head><meta name="api-base" content=""></head><body>hi</body></html>'

        async def aclose(self):
            pass

    class _Client:
        def build_request(self, method, url, headers, content):
            return (method, url)

        async def send(self, upstream_req, stream):
            return _Upstream()

    monkeypatch.setattr(preview_proxy, "_org_mode", lambda: False)
    monkeypatch.setattr(preview_proxy, "get_proxy_client", _Client)
    root = tmp_path / "artifact_root"
    root.mkdir()
    (root / "metadata.json").write_text('{"port": 4242}', encoding="utf-8")
    token = "shim-proxy-test-token"
    artifacts._PREVIEW_MOUNTS[token] = root
    try:
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        request = StarletteRequest({
            "type": "http", "method": "GET", "path": "/",
            "headers": [], "query_string": query,
        }, receive)
        response = await preview_proxy.proxy_artifact_request(token, "", request)
    finally:
        del artifacts._PREVIEW_MOUNTS[token]

    assert response.status_code == 200
    body = response.body.decode("utf-8")
    assert "anton-preview" in body
    assert ("anton-comments" in body) is expect_comments
    assert 'content="/api/v1/artifacts/proxy/shim-proxy-test-token"' in body
    # The upstream's stale length (60 bytes) must not survive the patch; the
    # framework recomputes it from the injected body.
    assert response.headers["content-length"] == str(len(response.body))
