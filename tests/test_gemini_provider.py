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
    _is_auth_error,
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


def test_validate_400_permission_message_is_not_invalid_key(monkeypatch):
    # A key that is fine but lacks model permission — Google returns 400 with a
    # message that CONTAINS "API key" but is not a bad key. It must surface
    # verbatim, not be relabeled "Invalid API key" and send the user to
    # regenerate a good key (ENG-1145 review).
    resp = _Resp(400, [{"error": {
        "message": "The API key does not have permission to use this model.",
    }}])
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_for(resp))
    result = asyncio.run(
        validate_openai_compatible("real-key", GEMINI_BASE_URL, "gemini-3.6-flash")
    )
    assert result["ok"] is False
    assert result["error"] != "Invalid API key"
    assert "does not have permission" in result["error"]


def test_validate_400_quota_message_is_not_invalid_key(monkeypatch):
    # Out of quota — also contains "API key" — must not read as a bad key.
    resp = _Resp(400, [{"error": {
        "message": "Quota exceeded for this API key. Upgrade your plan.",
    }}])
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_for(resp))
    result = asyncio.run(
        validate_openai_compatible("real-key", GEMINI_BASE_URL, "gemini-3.6-flash")
    )
    assert result["ok"] is False
    assert result["error"] != "Invalid API key"
    assert "Quota exceeded" in result["error"]


def test_is_auth_error_matrix():
    # The shared bad-key rule (must stay in step with cowork's provider-error.ts).
    assert _is_auth_error(401, None)
    assert _is_auth_error(403, "anything")
    # Gemini's real bad-key 400.
    assert _is_auth_error(400, "API key not valid. Please pass a valid API key.")
    # 400s that merely mention the key but are NOT bad keys.
    assert not _is_auth_error(400, "The API key does not have permission to use this model.")
    assert not _is_auth_error(400, "Quota exceeded for this API key. Upgrade your plan.")
    # A model-not-found message is not auth at all.
    assert not _is_auth_error(400, "models/gemini-2.5-flash is no longer available")
    assert not _is_auth_error(400, None)


def test_validate_ok_on_200(monkeypatch):
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_for(_Resp(200, {})))
    result = asyncio.run(
        validate_openai_compatible("k", GEMINI_BASE_URL, "gemini-3.6-flash")
    )
    assert result == {"ok": True}
