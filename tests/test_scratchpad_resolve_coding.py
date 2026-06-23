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


def test_openai_provider_unchanged(monkeypatch):
    # Regression guard: non-minds providers still read the OpenAI slot.
    _patch_settings(
        monkeypatch,
        _fake_settings(
            coding_provider="openai",
            openai_api_key="sk-openai",
            openai_base_url="https://api.openai.com/v1",
        ),
    )
    _provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert api_key == "sk-openai"
    assert base_url == "https://api.openai.com/v1"


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
