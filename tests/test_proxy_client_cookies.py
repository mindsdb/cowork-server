from __future__ import annotations

import httpx
import pytest

from cowork.common import http_client


@pytest.fixture
def proxy_client(monkeypatch):
    monkeypatch.setattr(http_client, "_client", None)
    return http_client.get_proxy_client()


def test_a_cookie_set_for_one_visitor_is_not_sent_for_the_next(proxy_client):
    """The client is shared by every visitor, so a cookie an artifact backend
    sets for one of them must not ride along on another visitor's request."""
    first = httpx.Response(
        200,
        headers={"set-cookie": "browser_id=visitor-a; Path=/"},
        request=httpx.Request("POST", "http://127.0.0.1:5000/api/vote"),
    )
    proxy_client.cookies.extract_cookies(first)

    second = proxy_client.build_request("POST", "http://127.0.0.1:5000/api/vote")

    assert "cookie" not in second.headers


def test_the_visitors_own_cookie_header_is_still_forwarded(proxy_client):
    request = proxy_client.build_request(
        "GET", "http://127.0.0.1:5000/api/poll", headers={"cookie": "browser_id=visitor-b"}
    )

    assert request.headers["cookie"] == "browser_id=visitor-b"
