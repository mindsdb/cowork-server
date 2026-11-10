"""Video calls (prototype): the server mints one Nuance session token per call
with the organization's key, and the session is fixed by constants."""
from __future__ import annotations

import json

import httpx
import pytest

import cowork.services.facetime as facetime
from cowork.common.settings.app_settings import get_app_settings
from cowork.server import create_app

KEY = "nk_test_key"


@pytest.fixture
def nuance(monkeypatch):
    """Nuance's API, answering with `nuance.answer`; records each request."""
    seen: list[httpx.Request] = []

    class Fake:
        answer = httpx.Response(201, json={"token": "st_abc", "expires_at": "2026-11-10T04:05:49Z", "warnings": []})

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return Fake.answer

    monkeypatch.setattr(facetime, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    Fake.seen = seen
    return Fake


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("COWORK_NUANCE_API_KEY", KEY)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


async def _call(method: str, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, f"/api/v1/facetime{path}")


async def test_no_key_no_video_calls(monkeypatch, nuance):
    monkeypatch.delenv("COWORK_NUANCE_API_KEY", raising=False)
    get_app_settings.cache_clear()

    assert (await _call("GET", "/status")).json() == {"available": False}
    refused = await _call("POST", "/token")
    assert refused.status_code == 409 and refused.json()["detail"]["code"] == "not_configured"
    assert nuance.seen == []


async def test_mints_a_token_with_the_fixed_session(keyed, nuance):
    assert (await _call("GET", "/status")).json() == {"available": True}

    minted = await _call("POST", "/token")

    assert minted.status_code == 200, minted.text
    assert minted.json() == {"token": "st_abc", "expires_at": "2026-11-10T04:05:49Z"}
    [request] = nuance.seen
    assert str(request.url) == "https://api.nuancelabs.ai/session-tokens"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["personality"] == {"persona_id": "sam", "system_prompt": facetime.PERSONA_PROMPT}
    delegation = body["interface_setup"]["delegation"]
    assert delegation["type"] == "responses"
    assert delegation["responses"]["instructions"] == facetime.BACKEND_INSTRUCTIONS
    assert [t["name"] for t in delegation["responses"]["tools"]] == ["ask_cowork"]
    assert body["end_user_id"] == facetime.end_user_id(None, None)


def test_the_person_is_named_by_an_opaque_id():
    one = facetime.end_user_id("org-1", "jorge@example.com")
    assert one.startswith("cw_") and "jorge" not in one and len(one) <= 128
    assert one == facetime.end_user_id("org-1", "jorge@example.com")
    assert one != facetime.end_user_id("org-2", "jorge@example.com")


@pytest.mark.parametrize(
    ("status", "body", "expected_status", "code", "words"),
    [
        (429, {"detail": "x", "code": "credits_spent"}, 429, "credits_spent", "used up for today"),
        (409, {"detail": "x", "code": "concurrency_limit", "active": 1, "limit": 1}, 409, "concurrency_limit", "Another video call"),
        (
            403,
            {"detail": "x", "code": "policy_curfew", "retry_after_seconds": 7200, "error": {"code": "policy_curfew"}},
            403, "policy_curfew", "paused for you",
        ),
        (401, {"detail": "bad key", "code": "not_authenticated"}, 424, "not_authenticated", "Couldn't start"),
        (500, {"detail": "boom"}, 424, "refused", "Couldn't start"),
    ],
)
async def test_a_refusal_says_why_in_words_for_the_person(keyed, nuance, status, body, expected_status, code, words):
    nuance.answer = httpx.Response(status, json=body)

    refused = await _call("POST", "/token")

    assert refused.status_code == expected_status
    detail = refused.json()["detail"]
    assert detail["code"] == code and words in detail["message"]
    assert "bad key" not in detail["message"] and KEY not in refused.text
