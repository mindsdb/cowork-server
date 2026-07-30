"""Gemini BYOK provider handling (ENG-1145).

Google's OpenAI-compatible endpoint is the source of two footguns these tests
pin: its CHAT errors arrive as a single-element JSON ARRAY
(``[{"error": {"message": ...}}]``) rather than a bare object, and a bad key
returns 400 (not 401/403). Both made the real failure — a retired default model
id — surface as an opaque status code. `fetch_gemini_models` is the live picker
source that replaces the hardcoded gemini-2.5-* list that 404s for new users.
"""
import asyncio

import cowork.services.providers as providers
from cowork.services.providers import (
    GEMINI_BASE_URL,
    _GEMINI_NATIVE_MODELS_URL,
    _provider_error_message,
    fetch_gemini_models,
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


def _client_for(resp, *, capture=None):
    """A fake httpx.AsyncClient whose get/post return `resp`.

    `resp` may be a single response or a callable(url) → response (so a test can
    vary the reply). `capture`, if given, records {url, params, headers} per call.
    """

    def _pick(url):
        return resp(url) if callable(resp) else resp

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, headers=None):
            if capture is not None:
                capture.append({"url": url, "params": params, "headers": headers})
            return _pick(url)

        async def post(self, url, headers=None, json=None):
            if capture is not None:
                capture.append({"url": url, "params": None, "headers": headers})
            return _pick(url)

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


# ── fetch_gemini_models: live picker source ───────────────────────────

# A native models.list payload spanning chat + specialized families. Only the
# generateContent-capable, non-multimodal-output models may reach the picker.
_NATIVE_MODELS = {"models": [
    {"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
    {"name": "models/gemini-3.1-pro-preview", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},        # embeddings
    {"name": "models/gemini-2.5-flash-image", "supportedGenerationMethods": ["generateContent"]}, # image OUT
    {"name": "models/gemini-2.5-flash-preview-tts", "supportedGenerationMethods": ["generateContent"]},  # TTS
    {"name": "models/veo-3.0-generate", "supportedGenerationMethods": ["predictLongRunning"]},    # video
    {"name": "models/gemini-live-2.5-flash", "supportedGenerationMethods": ["bidiGenerateContent"]},  # Live
]}


def test_fetch_gemini_models_keeps_only_chat_capable_and_normalizes(monkeypatch):
    providers._gemini_models_cache.clear()
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_for(_Resp(200, _NATIVE_MODELS)))

    ids = asyncio.run(fetch_gemini_models("real-key"))

    # models/ prefix stripped; image/TTS/video/Live/embedding all excluded even
    # though several advertise generateContent for non-text output.
    assert ids == ["gemini-3.6-flash", "gemini-3.1-pro-preview"]


def test_fetch_gemini_models_uses_native_endpoint_and_key_header(monkeypatch):
    providers._gemini_models_cache.clear()
    captured: list[dict] = []
    monkeypatch.setattr(
        providers.httpx, "AsyncClient", _client_for(_Resp(200, _NATIVE_MODELS), capture=captured)
    )

    asyncio.run(fetch_gemini_models("real-key"))

    call = captured[0]
    assert call["url"] == _GEMINI_NATIVE_MODELS_URL
    # Native metadata carries capability data; authed by x-goog-api-key, not Bearer.
    assert call["headers"].get("x-goog-api-key") == "real-key"
    assert "Authorization" not in call["headers"]


def test_fetch_gemini_models_cache_is_per_key(monkeypatch):
    providers._gemini_models_cache.clear()
    # Reply depends on which key's fingerprint asked — simulate two accounts by
    # varying the response, then assert the cache never bleeds A's list to B.
    state = {"payload": {"models": [
        {"name": "models/gemini-A", "supportedGenerationMethods": ["generateContent"]},
    ]}}
    monkeypatch.setattr(
        providers.httpx, "AsyncClient", _client_for(lambda _url: _Resp(200, state["payload"]))
    )

    assert asyncio.run(fetch_gemini_models("key-A")) == ["gemini-A"]
    state["payload"] = {"models": [
        {"name": "models/gemini-B", "supportedGenerationMethods": ["generateContent"]},
    ]}
    # Different key → different cache entry → fresh fetch, not A's cached list.
    assert asyncio.run(fetch_gemini_models("key-B")) == ["gemini-B"]
    # A stays cached under its own fingerprint.
    assert asyncio.run(fetch_gemini_models("key-A")) == ["gemini-A"]


def test_fetch_gemini_models_none_on_http_error(monkeypatch):
    providers._gemini_models_cache.clear()
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_for(_Resp(403, {"error": {}})))
    assert asyncio.run(fetch_gemini_models("bad-key")) is None


def test_fetch_gemini_models_none_without_key():
    assert asyncio.run(fetch_gemini_models("")) is None
