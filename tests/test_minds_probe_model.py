"""Regression tests for ENG-577 — the MindsHub connection probe must send a
model the router actually accepts.

MindsHub's router resolves only the canonical ``latest:<alias>`` form; a bare
alias like ``haiku`` matches no pattern and is rejected with an uncaught HTTP
500, which surfaced as "Provider is currently unreachable (HTTP 500)" on the
Settings page even though chat worked. ``canonical_minds_model`` prefixes bare
aliases, and ``ping_provider`` applies it to whatever model it probes with.
"""

import pytest

from cowork.common.settings.app_settings import CODING_MODEL_DEFAULTS
from cowork.services import providers
from cowork.services.providers import canonical_minds_model, ping_provider


class TestCanonicalMindsModel:
    def test_bare_alias_gets_latest_prefix(self):
        assert canonical_minds_model("haiku") == "latest:haiku"
        assert canonical_minds_model("sonnet") == "latest:sonnet"

    def test_bare_alias_with_whitespace(self):
        assert canonical_minds_model("  haiku ") == "latest:haiku"

    def test_already_canonical_unchanged(self):
        assert canonical_minds_model("latest:haiku") == "latest:haiku"

    def test_concrete_id_unchanged(self):
        # Live `/v1/models` ids and full model names must pass through as-is —
        # prefixing them would break routing.
        assert canonical_minds_model("claude-haiku-4-5-20251001") == "claude-haiku-4-5-20251001"

    def test_none_and_empty(self):
        assert canonical_minds_model(None) == ""
        assert canonical_minds_model("") == ""

    def test_the_probe_default_is_bare_so_would_500_without_normalization(self):
        # Guards the regression at its source: if the default ever becomes a
        # canonical/concrete id this test still passes, but today it's bare and
        # MUST be normalized before hitting the router.
        default = CODING_MODEL_DEFAULTS["minds_cloud"]
        assert canonical_minds_model(default).startswith("latest:") or ":" in default


class _CapturingResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.content = b"{}"

    def json(self):
        return {}


class _CapturingClient:
    """Minimal async httpx.AsyncClient stand-in that records the posted JSON."""

    captured: dict | None = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        _CapturingClient.captured = json
        return _CapturingResponse(200)


@pytest.mark.asyncio
async def test_ping_minds_cloud_sends_canonical_model_for_bare_fallback(monkeypatch):
    monkeypatch.setattr(providers.httpx, "AsyncClient", _CapturingClient)
    _CapturingClient.captured = None

    # No explicit model → falls back to the bare default; must be normalized.
    status, detail = await ping_provider(
        {"type": "minds-cloud", "apiKey": "k", "mindsUrl": "https://api.mindshub.ai"}
    )

    assert status == "ok", detail
    assert _CapturingClient.captured is not None
    assert _CapturingClient.captured["model"] == "latest:haiku"


@pytest.mark.asyncio
async def test_ping_minds_cloud_preserves_explicit_concrete_model(monkeypatch):
    monkeypatch.setattr(providers.httpx, "AsyncClient", _CapturingClient)
    _CapturingClient.captured = None

    await ping_provider(
        {
            "type": "minds-cloud",
            "apiKey": "k",
            "mindsUrl": "https://api.mindshub.ai",
            "model": "latest:sonnet",
        }
    )

    assert _CapturingClient.captured["model"] == "latest:sonnet"
