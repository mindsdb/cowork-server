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
from cowork.services.providers import MINDS_PROBE_MODEL, ping_provider, validate_minds


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
