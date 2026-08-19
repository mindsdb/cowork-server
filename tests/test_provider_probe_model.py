"""The MindsHub connectivity probe must use a tier-universal model (ENG-576).

MindsHub gates paid models per plan tier: a free-tier key gets a 403 for
haiku/sonnet/etc. The Settings health probe (`ping_provider`) and onboarding
validation (`validate_minds`) used to POST `CODING_MODEL_DEFAULTS["minds_cloud"]`
= "haiku" (paid) → free-tier accounts saw "MindsHub failed its last test" /
"Invalid API key" even though chat worked on mindshub_air. Both must now probe
`mindshub_air` (the free baseline, present in every tier).
"""
import asyncio

import cowork.services.providers as providers
from cowork.services.providers import (
    MINDS_PROBE_MODEL,
    is_minds_host,
    ping_provider,
    validate_minds,
    validate_provider,
)


class _CapturingClient:
    """Fake httpx.AsyncClient that records the JSON body of the probe POST."""

    captured: dict = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        _CapturingClient.captured = {"url": url, "json": json}
        return _Resp(200)

    async def get(self, url, headers=None):
        return _Resp(200)


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code

    def json(self):
        return {"choices": [{"message": {"content": "pong"}}]}


def _patch(monkeypatch):
    _CapturingClient.captured = {}
    monkeypatch.setattr(providers.httpx, "AsyncClient", _CapturingClient)


def test_probe_model_is_tier_universal():
    # Guard the constant itself — this is the whole fix.
    assert MINDS_PROBE_MODEL == "mindshub_air"


def test_ping_provider_probes_universal_model(monkeypatch):
    _patch(monkeypatch)
    status, _ = asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": "mdb_x"}))
    assert status == "ok"
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_ping_provider_ignores_configured_paid_model(monkeypatch):
    # Even if a (paid) model is passed, the connectivity probe uses the
    # universal one — the dot reflects reachability, not model availability.
    _patch(monkeypatch)
    asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": "mdb_x", "model": "sonnet"}))
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_validate_minds_probes_universal_model(monkeypatch):
    _patch(monkeypatch)
    result = asyncio.run(validate_minds("mdb_x", "https://api.mindshub.ai"))
    assert result.get("ok") is True
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_ping_provider_missing_key_still_fails_fast(monkeypatch):
    _patch(monkeypatch)
    status, detail = asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": ""}))
    assert status == "fail" and "key" in detail.lower()


def test_ping_minds_cloud_surfaces_provider_message(monkeypatch):
    # minds-cloud is the one provider routed through _chat_probe (real chat
    # completions), so its failures carry the gateway's actionable reason
    # (wallet/allowance/model). The dot detail must show it, not a bare
    # "HTTP 429" (ENG-1145 review, ENG-576).
    class _FailResp:
        status_code = 429

        def json(self):
            return {"error": {"message": "Wallet allowance exhausted. Top up to continue."}}

    class _FailClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            return _FailResp()

    monkeypatch.setattr(providers.httpx, "AsyncClient", _FailClient)
    status, detail = asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": "mdb_x"}))
    assert status == "fail"
    assert "HTTP 429" in detail
    assert "Wallet allowance exhausted" in detail


# ── The omitted-model default on a MindsHub host ──────────────────────
#
# Onboarding validates the MindsHub key through validate_provider. The generic
# openai-compatible default ("gpt-5.5") is not a MindsHub alias, so it 404s
# there, and the recommended MindsHub model is paid, so it 402s for an account
# whose wallet is empty. Both come back looking like a bad key, which routed a
# brand-new user to bring-your-own-key holding a valid MindsHub key.


def test_omitted_model_on_minds_host_probes_free_model(monkeypatch):
    _patch(monkeypatch)
    result = asyncio.run(
        validate_provider("openai-compatible", "mdb_x", "https://api.mindshub.ai/v1", None)
    )
    assert result.get("ok") is True
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_empty_model_on_minds_host_probes_free_model(monkeypatch):
    # The client sends `model` as an optional field, so an empty string arrives
    # as often as a missing key. Both mean "no model was chosen".
    _patch(monkeypatch)
    asyncio.run(validate_provider("openai-compatible", "mdb_x", "https://api.mindshub.ai/v1", ""))
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_explicit_model_on_minds_host_is_sent_as_asked(monkeypatch):
    # The negative case that keeps the default honest: a user validating one
    # specific model must not be told a different model passed.
    _patch(monkeypatch)
    asyncio.run(
        validate_provider("openai-compatible", "mdb_x", "https://api.mindshub.ai/v1", "sonnet")
    )
    assert _CapturingClient.captured["json"]["model"] == "sonnet"


def test_omitted_model_off_minds_host_keeps_generic_default(monkeypatch):
    # A real openai-compatible endpoint is unchanged by this.
    _patch(monkeypatch)
    asyncio.run(
        validate_provider("openai-compatible", "sk_x", "https://api.openai.com/v1", None)
    )
    assert _CapturingClient.captured["json"]["model"] == "gpt-5.5"


def test_is_minds_host_matches_the_host_not_a_substring():
    for url in (
        "https://api.mindshub.ai/v1",
        "https://api.staging.mindshub.ai",
        "https://api-pr-12.dev.mindshub.ai/v1",
        "https://mindshub.ai",
        "https://mdb.ai/api/v1",
        "https://llm.mdb.ai",
        "api.mindshub.ai/v1",  # no scheme, as a stored setting can be
    ):
        assert is_minds_host(url) is True, url

    for url in (
        "",
        None,
        "https://api.openai.com/v1",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        # A lookalike domain and a redirect-style parameter both defeat a
        # substring test, which is why this compares the parsed hostname.
        "https://mindshub.ai.example.test/v1",
        "https://evil-mindshub.ai/v1",
        "https://example.test/r?u=https://api.mindshub.ai/v1",
    ):
        assert is_minds_host(url) is False, url
