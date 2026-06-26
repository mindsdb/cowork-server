"""Direct wiring tests for build_llm_client._make_provider — the test gap
flagged on #111.

_make_provider is the main-agent counterpart of scratchpad _resolve_coding: it
builds an anton provider per role with the per-provider key and base URL. These
tests assert that wiring without hitting the network by stubbing anton's
provider classes and capturing the constructor kwargs:

  - openai/gemini NEVER inherit the shared openai_base_url slot (no misrouting);
  - gemini targets Google and reads the shared openai key via the fallback;
  - openai-compatible uses its dedicated key + its own base;
  - anthropic gets no base_url kwarg (its SDK has no such arg).
"""
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.services.providers import GEMINI_BASE_URL


@pytest.fixture
def build(monkeypatch):
    """Return a `build(settings) -> (client, calls)` helper.

    `calls` maps "openai"/"anthropic" → list of constructor kwarg dicts, in the
    order build_llm_client built them (planning first, then coding)."""
    calls: dict[str, list[dict]] = {}

    def _capture(kind):
        def _factory(**kw):
            calls.setdefault(kind, []).append(kw)
            return MagicMock(name=f"{kind}Provider")
        return _factory

    # build_llm_client imports these inside the function, so patching the module
    # attribute is picked up at call time.
    monkeypatch.setattr("anton.core.llm.openai.OpenAIProvider", _capture("openai"))
    monkeypatch.setattr(
        "anton.core.llm.anthropic.AnthropicProvider", _capture("anthropic")
    )

    def _build(settings: UserSettings):
        monkeypatch.setattr(
            "cowork.common.settings.user_settings.get_user_settings",
            lambda: settings,
        )
        from cowork.services.providers import build_llm_client

        client = build_llm_client()
        return client, calls

    return _build


def test_gemini_targets_google_with_shared_key_fallback(build):
    # gemini relying on the shared openai key (no dedicated slot) + a stale
    # contaminated base slot that must be ignored.
    settings = UserSettings(
        planning_provider=Provider.GEMINI,
        coding_provider=Provider.GEMINI,
        openai_api_key=SecretStr("AIza-shared"),
        openai_base_url="https://api.mindshub.ai/v1",  # contaminated; must be ignored
    )
    _client, calls = build(settings)
    assert "anthropic" not in calls
    kw = calls["openai"][0]
    assert kw["api_key"] == "AIza-shared"
    assert kw["base_url"] == GEMINI_BASE_URL  # Google, NOT the contaminated slot


def test_openai_never_inherits_contaminated_base(build):
    settings = UserSettings(
        planning_provider=Provider.OPENAI,
        coding_provider=Provider.OPENAI,
        openai_api_key=SecretStr("sk-openai"),
        openai_base_url="https://api.mindshub.ai/v1",  # contaminated; must be ignored
    )
    _client, calls = build(settings)
    kw = calls["openai"][0]
    assert kw["api_key"] == "sk-openai"
    assert kw["base_url"] is None  # SDK default host, never the shared slot


def test_openai_compatible_uses_dedicated_key_and_own_base(build):
    settings = UserSettings(
        planning_provider=Provider.OPENAI_COMPATIBLE,
        coding_provider=Provider.OPENAI_COMPATIBLE,
        planning_model="my-model",
        coding_model="my-coding-model",
        openai_compatible_api_key=SecretStr("sk-compat"),
        openai_api_key=SecretStr("sk-openai-should-not-win"),
        openai_base_url="https://my-proxy.example.com/v1",
    )
    _client, calls = build(settings)
    kw = calls["openai"][0]
    assert kw["api_key"] == "sk-compat"  # dedicated slot, not shared openai
    assert kw["base_url"] == "https://my-proxy.example.com/v1"


def test_anthropic_gets_no_base_url_kwarg(build):
    settings = UserSettings(
        planning_provider=Provider.ANTHROPIC,
        coding_provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("sk-ant"),
        openai_base_url="https://api.mindshub.ai/v1",  # must be ignored
    )
    _client, calls = build(settings)
    assert "openai" not in calls
    kw = calls["anthropic"][0]
    assert kw["api_key"] == "sk-ant"
    assert "base_url" not in kw  # AnthropicProvider takes no base_url kwarg


def test_openai_compatible_without_base_raises(build):
    # Defense-in-depth: config_status flags an empty OC base, but callers don't
    # all gate on config_ready, so the build site must refuse rather than let
    # OpenAIProvider default to api.openai.com (which would leak the BYO key).
    settings = UserSettings(
        planning_provider=Provider.OPENAI_COMPATIBLE,
        coding_provider=Provider.OPENAI_COMPATIBLE,
        planning_model="m",
        coding_model="m",
        openai_compatible_api_key=SecretStr("sk-compat"),
        # openai_base_url intentionally unset
    )
    with pytest.raises(ValueError, match="base URL"):
        build(settings)


def test_minds_cloud_uses_minds_key_and_derived_base(build):
    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
        openai_api_key=SecretStr("sk-openai-should-not-win"),
    )
    _client, calls = build(settings)
    kw = calls["openai"][0]
    assert kw["api_key"] == "mdb-key"  # minds slot, not the OpenAI slot
    assert kw["base_url"] == "https://api.mindshub.ai/v1"
