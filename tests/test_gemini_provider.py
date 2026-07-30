"""Gemini BYOK provider error handling (ENG-1145).

Google's OpenAI-compatible endpoint has two footguns these tests pin: its CHAT
errors arrive as a single-element JSON ARRAY (``[{"error": {"message": ...}}]``)
rather than a bare object, and a bad key returns 400 (not 401/403). Both made
the real failure — a retired default model id — surface as an opaque status code.
"""
import asyncio

import cowork.services.providers as providers
from cowork.services.providers import (
    GEMINI_BASE_URL,
    _provider_error_message,
    validate_openai_compatible,
)


class _Resp:
    def __init__(self, status_code, body=None, *, raises=False):
        self.status_code = status_code
        self._body = body
        self._raises = raises

    def json(self):
        if self._raises:
            raise ValueError("not json")
        return self._body


def _client_for(resp):
    """A fake httpx.AsyncClient whose post returns `resp`."""

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            return resp

    return _Client


# ── _provider_error_message: array vs object bodies ───────────────────


def test_error_message_from_object_body():
    resp = _Resp(404, {"error": {"message": "models/foo is not found."}})
    assert _provider_error_message(resp) == "models/foo is not found."


def test_error_message_from_gemini_array_body():
    # The shape that broke ENG-1145: message nested inside a top-level array.
    resp = _Resp(404, [{"error": {"message": "no longer available to new users"}}])
    assert _provider_error_message(resp) == "no longer available to new users"


def test_error_message_from_bare_message_and_unparseable():
    assert _provider_error_message(_Resp(400, {"message": "bad"})) == "bad"
    assert _provider_error_message(_Resp(500, raises=True)) is None
    assert _provider_error_message(_Resp(404, [])) is None


# ── validate_openai_compatible: Gemini-shaped failures ────────────────


def test_validate_surfaces_gemini_404_message(monkeypatch):
    resp = _Resp(404, [{"error": {
        "message": "This model models/gemini-2.5-flash is no longer available to new users.",
        "status": "NOT_FOUND",
    }}])
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_for(resp))

    result = asyncio.run(
        validate_openai_compatible("real-key", GEMINI_BASE_URL, "gemini-2.5-flash")
    )
    assert result["ok"] is False
    # The real reason reaches the user instead of a bare "HTTP 404".
    assert "no longer available to new users" in result["error"]


def test_validate_maps_gemini_bad_key_400_to_invalid_key(monkeypatch):
    # Google returns 400 (not 401/403) for a bad key; it must still read as an
    # invalid key, not an opaque "HTTP 400".
    resp = _Resp(400, [{"error": {"message": "Please pass a valid API key"}}])
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_for(resp))

    result = asyncio.run(
        validate_openai_compatible("bad", GEMINI_BASE_URL, "gemini-3.6-flash")
    )
    assert result == {"ok": False, "error": "Invalid API key"}


def test_validate_ok_on_200(monkeypatch):
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_for(_Resp(200, {})))
    result = asyncio.run(
        validate_openai_compatible("k", GEMINI_BASE_URL, "gemini-3.6-flash")
    )
    assert result == {"ok": True}
