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

from anton.core.llm.openai import OpenAIProvider as _RealOpenAIProvider
from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.services.providers import GEMINI_BASE_URL


@pytest.fixture
def build(monkeypatch):
    """Return a `build(settings) -> (client, calls)` helper.

    `calls` maps "openai"/"anthropic" → list of constructor kwarg dicts, in the
    order build_llm_client built them. When the installed anton's LLMClient
    accepts a router role, build_llm_client constructs that one first, so the
    list may start with a router call before planning/coding — tests index
    `[-1]` (always the coding call) rather than `[0]` so they don't depend on
    whether the router role happened to resolve to the same provider."""
    calls: dict[str, list[dict]] = {}

    def _capture(kind):
        def _factory(**kw):
            calls.setdefault(kind, []).append(kw)
            return MagicMock(name=f"{kind}Provider")
        return _factory

    # build_llm_client imports these inside the function, so patching the module
    # attribute is picked up at call time.
    _fake_openai = _capture("openai")
    # _make_provider calls OpenAIProvider.resolve_web_flavor(...) (ENG-1359),
    # whose body references the FLAVOR_* constants via the module-global name
    # `OpenAIProvider` — which this monkeypatch just repointed at `_fake_openai`.
    # So the fake needs both the real staticmethod and the real constants it reads.
    _fake_openai.resolve_web_flavor = _RealOpenAIProvider.resolve_web_flavor
    _fake_openai.FLAVOR_OPENAI = _RealOpenAIProvider.FLAVOR_OPENAI
    _fake_openai.FLAVOR_MINDS_PASSTHROUGH = _RealOpenAIProvider.FLAVOR_MINDS_PASSTHROUGH
    _fake_openai.FLAVOR_OPENAI_COMPATIBLE_GENERIC = (
        _RealOpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC
    )
    monkeypatch.setattr("anton.core.llm.openai.OpenAIProvider", _fake_openai)
    monkeypatch.setattr(
        "anton.core.llm.anthropic.AnthropicProvider", _capture("anthropic")
    )

    def _build(settings: UserSettings, effort_override=None):
        monkeypatch.setattr(
            "cowork.common.settings.user_settings.get_user_settings",
            lambda: settings,
        )
        from cowork.services.providers import build_llm_client

        client = build_llm_client(effort_override=effort_override)
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
    kw = calls["openai"][-1]
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
    kw = calls["openai"][-1]
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
    kw = calls["openai"][-1]
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
    kw = calls["anthropic"][-1]
    assert kw["api_key"] == "sk-ant"
    assert "base_url" not in kw  # AnthropicProvider takes no base_url kwarg


def test_missing_key_error_names_the_actual_provider(build):
    # gemini/openai-compatible go through the OpenAIProvider branch but the
    # "not configured" message must name the real provider, not "OpenAI".
    settings = UserSettings(
        planning_provider=Provider.GEMINI,
        coding_provider=Provider.GEMINI,
        # no key anywhere → no fallback either
    )
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        build(settings)


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
    kw = calls["openai"][-1]
    assert kw["api_key"] == "mdb-key"  # minds slot, not the OpenAI slot
    assert kw["base_url"] == "https://api.mindshub.ai/v1"


# ── Reasoning effort follows the model, not the role (ENG-1632) ────────

def test_effort_travels_when_resolution_keeps_the_stored_model(build):
    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
        coding_model="haiku",
        coding_reasoning_effort="high",
    )
    _client, calls = build(settings)
    assert calls["openai"][-1].get("reasoning_effort") == "high"


def test_effort_dropped_when_wallet_fallback_swaps_the_model(build):
    # A wallet-locked coding pin resolves to the first enabled model; the
    # stored effort was chosen for the pinned model and may not exist on the
    # substitute — it must not travel (the gateway 400s an unsupported level).
    import json as _json

    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
        coding_model="haiku",
        coding_reasoning_effort="high",
        minds_model_enabled=_json.dumps({"mindshub_air": True, "haiku": False}),
    )
    _client, calls = build(settings)
    assert "reasoning_effort" not in calls["openai"][-1]


def test_effort_survives_when_no_model_row_is_stored(build):
    # Pin for the apply_model_defaults ↔ _effort_for coupling: a user with NO
    # coding_model row keeps their reasoning effort only because the validator
    # pre-fills the stored field, making stored == resolved. If the "collapse
    # the redundant enabled-aware branch" idea from ENG-1632 ever removes that
    # pre-fill, this goes red instead of every no-row user silently losing
    # their effort setting.
    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
        coding_reasoning_effort="high",
    )
    assert settings.coding_model is not None  # the validator pre-fill
    _client, calls = build(settings)
    assert calls["openai"][-1].get("reasoning_effort") == "high"


def test_keyless_local_endpoint_routes_to_its_base(build):
    """A local model server needs no API key — only a reachable base URL.

    Treated as unconfigured, the resolver walks past openai-compatible to the
    first provider that does have a key (MindsHub first), so prompts meant for
    a machine on the user's own network are sent to the hosted gateway instead.
    """
    settings = UserSettings(
        planning_provider=Provider.OPENAI_COMPATIBLE,
        coding_provider=Provider.OPENAI_COMPATIBLE,
        router_provider=Provider.OPENAI_COMPATIBLE,
        planning_model="qwen/qwen3.5-9b",
        coding_model="qwen/qwen3.5-9b",
        router_model="qwen/qwen3.5-9b",
        minds_api_key=SecretStr("mdb_abc"),  # signed in, but not the endpoint
        openai_base_url="http://192.168.1.100:1234/v1",
    )
    _client, calls = build(settings)
    for kw in calls["openai"]:
        assert kw["base_url"] == "http://192.168.1.100:1234/v1"
        assert kw["api_key"]  # the SDK requires some string
    assert "anthropic" not in calls


def test_keyless_local_endpoint_reports_ready(build):
    settings = UserSettings(
        planning_provider=Provider.OPENAI_COMPATIBLE,
        coding_provider=Provider.OPENAI_COMPATIBLE,
        planning_model="qwen/qwen3.5-9b",
        coding_model="qwen/qwen3.5-9b",
        minds_api_key=SecretStr("mdb_abc"),
        openai_base_url="http://192.168.1.100:1234/v1",
    )
    status = settings.config_status
    assert status["provider"] == Provider.OPENAI_COMPATIBLE.value
    assert status["config_ready"] is True
    assert status["config_error"] is None


def test_keyless_openai_compatible_without_base_still_gates(build):
    """No key and no base URL is genuinely unconfigured — it must not read as
    ready, and must never quietly become a hosted-gateway turn."""
    settings = UserSettings(
        planning_provider=Provider.OPENAI_COMPATIBLE,
        coding_provider=Provider.OPENAI_COMPATIBLE,
        openai_base_url="",
    )
    assert settings._has_key(Provider.OPENAI_COMPATIBLE) is False
    assert settings.config_status["config_ready"] is False


# ── Per-task effort override (ENG-1940) ─────────────────────────────────

def test_effort_override_wins_over_stored_role_effort(build):
    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
        coding_model="haiku",
        coding_reasoning_effort="low",
    )
    _client, calls = build(settings, effort_override="high")
    assert calls["openai"][-1].get("reasoning_effort") == "high"


def test_effort_override_applies_even_when_stored_effort_would_be_dropped(build):
    # The stale-model guard (_effort_for) must not suppress an explicit
    # per-task override the way it suppresses a stale persisted choice.
    import json as _json

    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
        coding_model="haiku",
        coding_reasoning_effort="low",
        minds_model_enabled=_json.dumps({"mindshub_air": True, "haiku": False}),
    )
    _client, calls = build(settings, effort_override="high")
    assert calls["openai"][-1].get("reasoning_effort") == "high"


def test_no_effort_override_falls_back_to_existing_behavior(build):
    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
        coding_model="haiku",
        coding_reasoning_effort="high",
    )
    _client, calls = build(settings)  # no effort_override — default None
    assert calls["openai"][-1].get("reasoning_effort") == "high"
