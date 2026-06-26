"""Regression tests for `_resolve_coding` minds-cloud resolution (ENG-436).

The scratchpad must resolve a minds-cloud user's coding LLM from the
dedicated ``minds_api_key`` / ``minds_url`` slots — NOT the
``openai_api_key`` slot. Before the fix, ``minds-cloud`` fell into the
``else`` branch and read ``openai_api_key``, which is why login had to
copy the minds key into the OpenAI slot (clobbering a user's real
OpenAI key). These tests guard that the scratchpad now resolves minds
natively, mirroring ``_make_provider``.
"""
from types import SimpleNamespace

from cowork.services import scratchpad_runtime


def _fake_settings(**overrides):
    base = dict(
        coding_provider="",
        coding_model="",
        anthropic_api_key=None,
        openai_api_key=None,
        openai_base_url=None,
        minds_api_key=None,
        minds_url="https://api.mindshub.ai",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _patch_settings(monkeypatch, settings):
    # `_resolve_coding` does `from anton.config.settings import AntonSettings`
    # at call time, so patching the attribute on that module is picked up.
    monkeypatch.setattr("anton.config.settings.AntonSettings", lambda: settings)


def test_minds_cloud_resolves_from_minds_slot_not_openai(monkeypatch):
    # minds-cloud user who ALSO has a real OpenAI key set — the real key
    # must be left alone; the scratchpad uses the minds slot.
    _patch_settings(
        monkeypatch,
        _fake_settings(
            coding_provider="minds-cloud",
            coding_model="latest:haiku",
            minds_api_key="mdb_minds_key",
            minds_url="https://api.mindshub.ai",
            openai_api_key="sk-proj-real-user-key",
            openai_base_url="https://api.openai.com/v1",
        ),
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    # Presented to anton's scratchpad as openai-compatible (the string it
    # understands) — NOT "minds-cloud", which anton would route to Anthropic.
    assert provider == "openai-compatible"
    assert api_key == "mdb_minds_key"
    assert api_key != "sk-proj-real-user-key"          # NOT the OpenAI slot
    assert base_url == "https://api.mindshub.ai/v1"    # host-aware: /v1 for mindshub


def test_minds_cloud_uses_api_v1_for_legacy_mdb_ai(monkeypatch):
    _patch_settings(
        monkeypatch,
        _fake_settings(
            coding_provider="minds-cloud",
            minds_api_key="mdb_minds_key",
            minds_url="https://mdb.ai",
        ),
    )
    _provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert api_key == "mdb_minds_key"
    assert base_url == "https://mdb.ai/api/v1"          # legacy host → /api/v1


def test_openai_reads_openai_key_and_does_not_inherit_base_slot(monkeypatch):
    # Direct OpenAI reads the openai_api_key slot but must NOT inherit the
    # shared openai_base_url slot — base is empty so anton uses the SDK default
    # (api.openai.com). (Previously openai read the shared slot, which is the
    # contamination bug being fixed.)
    _patch_settings(
        monkeypatch,
        _fake_settings(
            coding_provider="openai",
            openai_api_key="sk-openai",
            openai_base_url="https://api.openai.com/v1",
        ),
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "openai"
    assert api_key == "sk-openai"
    assert base_url == ""  # no inherited base → anton defaults to api.openai.com


def test_openai_ignores_contaminated_base_slot(monkeypatch):
    # The trap: a user configured MindsHub (leaving openai_base_url pointed at
    # MindsHub), then switched to OpenAI BYOK. Their OpenAI key must NOT be
    # routed to MindsHub.
    _patch_settings(
        monkeypatch,
        _fake_settings(
            coding_provider="openai",
            openai_api_key="sk-proj-real-openai",
            openai_base_url="https://api.mindshub.ai/v1",  # stale, contaminated
        ),
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "openai"
    assert api_key == "sk-proj-real-openai"
    assert "mindshub" not in base_url        # key is NOT misrouted to MindsHub
    assert base_url == ""


def test_gemini_routes_to_google_as_openai_compatible(monkeypatch):
    # Gemini reads the shared openai key slot but must target Google's endpoint
    # (not OpenAI, not a contaminated slot) and be presented as
    # openai-compatible so the scratchpad uses OpenAIProvider, not Anthropic.
    _patch_settings(
        monkeypatch,
        _fake_settings(
            coding_provider="gemini",
            openai_api_key="AIza-gemini-key",
            openai_base_url="https://api.mindshub.ai/v1",  # contaminated; must be ignored
        ),
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "openai-compatible"   # NOT "gemini" → avoids AnthropicProvider
    assert api_key == "AIza-gemini-key"
    assert base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_openai_compatible_keeps_its_own_base(monkeypatch):
    # openai-compatible is the one provider that legitimately owns the base slot.
    _patch_settings(
        monkeypatch,
        _fake_settings(
            coding_provider="openai-compatible",
            openai_api_key="sk-compat",
            openai_base_url="https://my-proxy.example.com/v1",
        ),
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "openai-compatible"
    assert api_key == "sk-compat"
    assert base_url == "https://my-proxy.example.com/v1"


def test_anthropic_uses_own_slot_no_base(monkeypatch):
    _patch_settings(
        monkeypatch,
        _fake_settings(
            coding_provider="anthropic",
            anthropic_api_key="sk-ant",
            openai_base_url="https://api.mindshub.ai/v1",  # must be ignored
        ),
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "anthropic"
    assert api_key == "sk-ant"
    assert base_url == ""


def test_explicit_coding_api_key_wins(monkeypatch):
    # An explicitly passed key short-circuits all slot resolution.
    _patch_settings(
        monkeypatch,
        _fake_settings(coding_provider="minds-cloud", minds_api_key="mdb_minds_key"),
    )
    _provider, _model, api_key, _base = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="explicit-key", coding_base_url=""
    )
    assert api_key == "explicit-key"
