"""Regression tests for `_resolve_coding` — the scratchpad backend launcher's
per-provider key/base resolution.

`_resolve_coding` reads from cowork's authoritative UserSettings (which owns the
dedicated gemini / openai-compatible key slots), NOT AntonSettings. So:
  - a minds-cloud user resolves their real minds key + host, never the
    ``openai_api_key`` / shared ``openai_base_url`` slots (ENG-436);
  - openai/gemini never inherit a stale (contaminated) shared base slot;
  - a dedicated-only gemini / openai-compatible key — invisible to AntonSettings,
    which carries only the three legacy slots — still resolves (the gap
    SailingSF flagged on #111).
"""
from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.services import scratchpad_runtime


def _patch_user_settings(monkeypatch, **kw):
    # `_resolve_coding` does `from ...user_settings import get_user_settings` at
    # call time, so patching the attribute on that module is picked up.
    settings = UserSettings(**kw)
    monkeypatch.setattr(
        "cowork.common.settings.user_settings.get_user_settings",
        lambda: settings,
    )


def test_minds_cloud_resolves_from_minds_slot_not_openai(monkeypatch):
    # minds-cloud user who ALSO has a real OpenAI key set — the real key
    # must be left alone; the scratchpad uses the minds slot.
    _patch_user_settings(
        monkeypatch,
        coding_provider=Provider.MINDS_CLOUD,
        coding_model="latest:haiku",
        minds_api_key=SecretStr("mdb_minds_key"),
        minds_url="https://api.mindshub.ai",
        openai_api_key=SecretStr("sk-proj-real-user-key"),
        openai_base_url="https://api.openai.com/v1",
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
    _patch_user_settings(
        monkeypatch,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb_minds_key"),
        minds_url="https://mdb.ai",
    )
    _provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert api_key == "mdb_minds_key"
    assert base_url == "https://mdb.ai/api/v1"          # legacy host → /api/v1


def test_openai_reads_openai_key_and_does_not_inherit_base_slot(monkeypatch):
    # Direct OpenAI reads the openai_api_key slot but must NOT inherit the
    # shared openai_base_url slot — base is empty so anton uses the SDK default
    # (api.openai.com).
    _patch_user_settings(
        monkeypatch,
        coding_provider=Provider.OPENAI,
        openai_api_key=SecretStr("sk-openai"),
        openai_base_url="https://api.openai.com/v1",
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
    _patch_user_settings(
        monkeypatch,
        coding_provider=Provider.OPENAI,
        openai_api_key=SecretStr("sk-proj-real-openai"),
        openai_base_url="https://api.mindshub.ai/v1",  # stale, contaminated
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "openai"
    assert api_key == "sk-proj-real-openai"
    assert "mindshub" not in base_url        # key is NOT misrouted to MindsHub
    assert base_url == ""


def test_gemini_routes_to_google_as_openai_compatible(monkeypatch):
    # Gemini on the shared openai key slot (fallback) must target Google's
    # endpoint (not OpenAI, not a contaminated slot) and be presented as
    # openai-compatible so the scratchpad uses OpenAIProvider, not Anthropic.
    _patch_user_settings(
        monkeypatch,
        coding_provider=Provider.GEMINI,
        openai_api_key=SecretStr("AIza-gemini-key"),
        openai_base_url="https://api.mindshub.ai/v1",  # contaminated; must be ignored
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "openai-compatible"   # NOT "gemini" → avoids AnthropicProvider
    assert api_key == "AIza-gemini-key"
    assert base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_dedicated_gemini_key_resolves_without_shared_openai(monkeypatch):
    # SailingSF #111-B: a gemini-only user with NO shared openai key. The
    # dedicated slot is invisible to AntonSettings (3 legacy slots only), so the
    # old AntonSettings-based path handed it an empty key. UserSettings sees it.
    _patch_user_settings(
        monkeypatch,
        coding_provider=Provider.GEMINI,
        gemini_api_key=SecretStr("AIza-dedicated"),
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "openai-compatible"
    assert api_key == "AIza-dedicated"
    assert base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_openai_compatible_keeps_its_own_base(monkeypatch):
    # openai-compatible is the one provider that legitimately owns the base slot.
    _patch_user_settings(
        monkeypatch,
        coding_provider=Provider.OPENAI_COMPATIBLE,
        openai_compatible_api_key=SecretStr("sk-compat"),
        openai_base_url="https://my-proxy.example.com/v1",
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "openai-compatible"
    assert api_key == "sk-compat"
    assert base_url == "https://my-proxy.example.com/v1"


def test_anthropic_uses_own_slot_no_base(monkeypatch):
    _patch_user_settings(
        monkeypatch,
        coding_provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("sk-ant"),
        openai_base_url="https://api.mindshub.ai/v1",  # must be ignored
    )
    provider, _model, api_key, base_url = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="", coding_base_url=""
    )
    assert provider == "anthropic"
    assert api_key == "sk-ant"
    assert base_url == ""


def test_explicit_coding_api_key_wins(monkeypatch):
    # An explicitly passed key short-circuits all slot resolution.
    _patch_user_settings(
        monkeypatch,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb_minds_key"),
    )
    _provider, _model, api_key, _base = scratchpad_runtime._resolve_coding(
        coding_provider="", coding_model="", coding_api_key="explicit-key", coding_base_url=""
    )
    assert api_key == "explicit-key"
