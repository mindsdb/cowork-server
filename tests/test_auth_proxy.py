"""auth_proxy._relay against real httpx.AsyncClient shapes it hasn't been
exercised against before: an empty-body 2xx (disconnect's 204) and an error
body with a `detail` field. Every other proxy_* function already goes
through this same relay, monkeypatched away at the call site in the other
test files - this one tests the relay itself.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from cowork.common.settings.app_settings import OAuthSettings
from cowork.services.connectors.oauth import auth_proxy
from tests._fakes import FakeRequest


class _Resp:
    def __init__(self, status_code, text="", json_body=None):
        self.status_code = status_code
        self.text = text
        self._json_body = json_body

    def json(self):
        if self._json_body is None:
            raise ValueError("no body")
        return self._json_body


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def __call__(self, *a, **k):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def request(self, method, url, params=None, json=None, headers=None):
        return self._resp


def _settings():
    return OAuthSettings(_env_file=None, AUTH_SERVICE_BASE_URL="https://auth.internal")


@pytest.mark.asyncio
async def test_proxy_delete_204_empty_body_does_not_raise(monkeypatch):
    monkeypatch.setattr(auth_proxy, "httpx", type("_H", (), {"AsyncClient": _FakeClient(_Resp(204))}))

    result = await auth_proxy.proxy_delete("gmail", "work", FakeRequest(), _settings())

    assert result is None


@pytest.mark.asyncio
async def test_proxy_delete_propagates_auths_404_detail(monkeypatch):
    resp = _Resp(404, text='{"detail": "No gmail connection named \'work\' for this user."}',
                 json_body={"detail": "No gmail connection named 'work' for this user."})
    monkeypatch.setattr(auth_proxy, "httpx", type("_H", (), {"AsyncClient": _FakeClient(resp)}))

    with pytest.raises(HTTPException) as exc_info:
        await auth_proxy.proxy_delete("gmail", "work", FakeRequest(), _settings())

    assert exc_info.value.status_code == 404
    assert "work" in exc_info.value.detail


@pytest.mark.asyncio
async def test_proxy_remove_picked_file_calls_auths_route(monkeypatch):
    seen = {}

    class _Recording(_FakeClient):
        async def request(self, method, url, params=None, json=None, headers=None):
            seen.update(method=method, url=url, params=params)
            return self._resp

    remaining = [{"id": "p2", "projects": ["beta"]}]
    resp = _Resp(200, text="{...}", json_body={"files": remaining})
    monkeypatch.setattr(auth_proxy, "httpx", type("_H", (), {"AsyncClient": _Recording(resp)}))

    result = await auth_proxy.proxy_remove_picked_file("notion", "ws-a", "page/1", "alpha", FakeRequest(), _settings())

    assert seen == {
        "method": "DELETE",
        "url": "https://auth.internal/v1/oauth/notion/ws-a/picked-files/page%2F1",
        "params": {"project": "alpha"},
    }
    assert result == remaining
