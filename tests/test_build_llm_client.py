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
import inspect
import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from anton.core.llm.openai import OpenAIProvider as _RealOpenAIProvider
from anton.core.llm.provider import ProviderAuthError
from pydantic import SecretStr

from cowork.common.settings import runtime_credential
from cowork.common.settings.app_settings import ROUTER_MODEL_DEFAULTS
from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.services import providers
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
    # Production capability-gates new Anton kwargs from the callable signature.
    # Preserve the real constructor contract on this kwargs-capturing fake.
    _fake_openai.__signature__ = inspect.signature(_RealOpenAIProvider)
    monkeypatch.setattr("anton.core.llm.openai.OpenAIProvider", _fake_openai)
    monkeypatch.setattr(
        "anton.core.llm.anthropic.AnthropicProvider", _capture("anthropic")
    )

    def _build(settings: UserSettings, effort_override=None, model_override=None):
        monkeypatch.setattr(
            "cowork.common.settings.user_settings.get_user_settings",
            lambda: settings,
        )
        from cowork.services.providers import build_llm_client

        client = build_llm_client(effort_override=effort_override, model_override=model_override)
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
    assert "api_key_provider" not in kw


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
    assert "api_key_provider" not in kw


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
    assert "api_key_provider" not in kw


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
    assert "api_key_provider" not in kw


@pytest.mark.parametrize("model", [None, "picked"])
def test_missing_key_error_names_the_actual_provider(build, model):
    # gemini/openai-compatible go through the OpenAIProvider branch but the
    # "not configured" message must name the real provider, not "OpenAI".
    settings = UserSettings(
        planning_provider=Provider.GEMINI,
        coding_provider=Provider.GEMINI,
        # no key anywhere → no fallback either
    )
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        build(settings, model_override=model)


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


def test_static_minds_cloud_key_stays_static(build, monkeypatch):
    monkeypatch.setattr(runtime_credential, "get_minds_credential", lambda: None)
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
    assert "api_key_provider" not in kw


def test_org_mode_minds_cloud_key_stays_static(build, monkeypatch):
    monkeypatch.setattr(
        runtime_credential,
        "get_app_settings",
        lambda: SimpleNamespace(tenancy_mode="org"),
    )
    monkeypatch.setattr(runtime_credential, "_minds_credential", "desktop-token")
    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("per-turn-key"),
        minds_url="https://api.mindshub.ai",
    )
    _client, calls = build(settings)
    kw = calls["openai"][-1]
    assert kw["api_key"] == "per-turn-key"
    assert "api_key_provider" not in kw


@pytest.mark.asyncio
async def test_local_minds_cloud_provider_rereads_runtime_credential(build, monkeypatch):
    current = {"credential": "token-A"}
    monkeypatch.setattr(
        runtime_credential,
        "get_minds_credential",
        lambda: current["credential"],
    )

    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("token-A"),
        minds_url="https://api.mindshub.ai",
    )
    _client, calls = build(settings)
    provider_calls = len(calls["openai"])
    assert provider_calls == 3  # router, planning, and coding
    credential_providers = [kw["api_key_provider"] for kw in calls["openai"]]

    assert all(kw["api_key"] == "token-A" for kw in calls["openai"])
    assert [await provider() for provider in credential_providers] == [
        "token-A",
        "token-A",
        "token-A",
    ]

    current["credential"] = "token-B"
    assert [await provider() for provider in credential_providers] == [
        "token-B",
        "token-B",
        "token-B",
    ]
    assert len(calls["openai"]) == provider_calls

    current["credential"] = None
    for provider in credential_providers:
        with pytest.raises(ProviderAuthError):
            await provider()


def test_old_anton_keeps_construction_time_credential_and_warns(
    build, monkeypatch, caplog
):
    """An allowed older Anton has no supplier kwarg, so do not pass it."""
    calls: list[dict] = []

    class _OldOpenAIProvider:
        FLAVOR_MINDS_PASSTHROUGH = _RealOpenAIProvider.FLAVOR_MINDS_PASSTHROUGH

        def __init__(
            self,
            api_key=None,
            base_url=None,
            flavor=None,
            reasoning_effort=None,
        ):
            calls.append(
                {
                    "api_key": api_key,
                    "base_url": base_url,
                    "flavor": flavor,
                    "reasoning_effort": reasoning_effort,
                }
            )

    monkeypatch.setattr(
        "anton.core.llm.openai.OpenAIProvider", _OldOpenAIProvider
    )
    monkeypatch.setattr(
        runtime_credential, "get_minds_credential", lambda: "token-A"
    )
    settings = UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("token-A"),
        minds_url="https://api.mindshub.ai",
    )

    with caplog.at_level("WARNING", logger="cowork.services.providers"):
        build(settings)

    assert len(calls) == 3  # router, planning, and coding still construct
    assert all(call["api_key"] == "token-A" for call in calls)
    assert caplog.text.count("construction-time credential") == 1


@pytest.mark.asyncio
async def test_runtime_supplier_falls_back_when_anton_lacks_typed_auth_error(
    monkeypatch,
):
    from cowork.services import providers

    monkeypatch.setattr(runtime_credential, "get_minds_credential", lambda: None)
    monkeypatch.setitem(
        sys.modules, "anton.core.llm.provider", ModuleType("anton.core.llm.provider")
    )

    with pytest.raises(ConnectionError) as err:
        await providers._current_runtime_minds_credential()

    assert type(err.value) is ConnectionError
    assert str(err.value).startswith("Invalid API key")


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


@pytest.mark.parametrize(
    "model,effort,expected_effort",
    [
        ("picked", None, None),
        ("picked", "low", "low"),
        ("old", None, "high"),
        ("old", "low", "low"),
    ],
)
def test_model_override_keeps_only_applicable_effort(build, model, effort, expected_effort):
    settings = UserSettings(
        planning_provider=Provider.OPENAI, coding_provider=Provider.OPENAI,
        router_provider=Provider.OPENAI, openai_api_key=SecretStr("test-key"),
        planning_model="old", coding_model="old", router_model="old",
        planning_reasoning_effort="high", coding_reasoning_effort="high",
        router_reasoning_effort="none",
    )
    client, calls = build(settings, model_override=model, effort_override=effort)
    assert (client.planning_model, client.coding_model, client.router_model) == (model,) * 3
    # The router never takes the composer's pick. Nor, here, its own effort:
    # on direct OpenAI the gate runs gpt-5.5-mini on the router's provider,
    # not "old" (see test_router_effort_is_sent_only_where_the_gate_runs_the_router_model).
    assert [kw.get("reasoning_effort") for kw in calls["openai"]] == [
        None, expected_effort, expected_effort
    ]
    assert (settings.planning_model, settings.coding_model, settings.router_model) == ("old",) * 3


@pytest.mark.parametrize("effort", [None, "low"])
def test_model_override_preserves_other_provider_roles_and_credentials(build, effort):
    settings = UserSettings(
        planning_provider=Provider.OPENAI, openai_api_key=SecretStr("openai-key"),
        coding_provider=Provider.ANTHROPIC, router_provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("anthropic-key"),
        planning_model="old", coding_model="claude-coding", router_model="claude-router",
        coding_reasoning_effort="high",
    )
    client, calls = build(settings, model_override="picked-openai", effort_override=effort)
    assert client.planning_model == "picked-openai"
    assert client.coding_model == "claude-coding"
    assert client.router_model == "claude-router"
    assert calls["openai"][0]["api_key"] == "openai-key"
    assert all(kw["api_key"] == "anthropic-key" for kw in calls["anthropic"])
    assert calls["anthropic"][-1]["reasoning_effort"] == "high"


# ── The router's own reasoning effort ───────────────────────────────────

def _openai_compatible(**kw) -> UserSettings:
    """All three roles on one openai_compatible endpoint, each with its model."""
    return UserSettings(
        planning_provider=Provider.OPENAI_COMPATIBLE,
        coding_provider=Provider.OPENAI_COMPATIBLE,
        router_provider=Provider.OPENAI_COMPATIBLE,
        planning_model="gpt-5.6-sol",
        coding_model="gpt-5.6-luna",
        router_model="gpt-5.6-luna",
        openai_compatible_api_key=SecretStr("sk-compat"),
        openai_base_url="https://example-resource.openai.azure.com/openai/v1",
        **kw,
    )


def test_router_effort_reaches_the_router_and_only_the_router(build):
    # A reasoning router sends "none" so the gate's one function tool is
    # accepted on chat completions; planning and coding keep their own.
    _client, calls = build(_openai_compatible(router_reasoning_effort="none"))
    router, planning, coding = calls["openai"]
    assert router["reasoning_effort"] == "none"
    assert "reasoning_effort" not in planning
    assert "reasoning_effort" not in coding


@pytest.mark.parametrize("router_effort", [None, ""])
def test_an_unset_or_empty_router_effort_is_omitted(build, router_effort):
    # A model that doesn't reason refuses any reasoning_effort, so an unset
    # value, or the empty one a deployment passes for an unset variable,
    # sends no field at all.
    _client, calls = build(_openai_compatible(router_reasoning_effort=router_effort))
    assert "reasoning_effort" not in calls["openai"][0]


def test_the_composer_effort_never_replaces_the_router_effort(build):
    # The per-task pick is chosen for the chat model; the router keeps its own.
    _client, calls = build(
        _openai_compatible(router_reasoning_effort="none"), effort_override="high"
    )
    router, planning, coding = calls["openai"]
    assert router["reasoning_effort"] == "none"
    assert planning["reasoning_effort"] == coding["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("provider", "router_model", "expected"),
    [
        # The gate runs claude-haiku-4-5, which takes no effort at all.
        (Provider.ANTHROPIC, "claude-sonnet-4-6", None),
        # The gate runs gpt-5.5-mini.
        (Provider.OPENAI, "gpt-5.6-luna", None),
        # The router pick is the provider's default, so the gate runs it too.
        (Provider.OPENAI, ROUTER_MODEL_DEFAULTS["openai"], "low"),
    ],
)
def test_router_effort_is_sent_only_where_the_gate_runs_the_router_model(
    build, provider, router_model, expected
):
    # The route gate runs on the router's provider, but with
    # resolved_gate_model: off openai_compatible, the provider's default router
    # model. An effort chosen for another router pick must not reach it.
    settings = UserSettings(
        planning_provider=provider, coding_provider=provider, router_provider=provider,
        anthropic_api_key=SecretStr("sk-ant"), openai_api_key=SecretStr("sk-openai"),
        router_model=router_model, router_reasoning_effort="low",
    )
    assert (settings.resolved_gate_model == router_model) is (expected is not None)
    _client, calls = build(settings)
    assert calls[provider.value][0].get("reasoning_effort") == expected


def test_router_effort_stays_with_the_model_it_was_chosen_for(build):
    # No Anthropic key, so the router resolves onto OpenAI's default, which the
    # gate runs too. The stored effort was chosen for the Anthropic pick, so
    # the stale-model guard keeps it off.
    settings = UserSettings(
        _env_file=None,
        planning_provider=Provider.OPENAI, coding_provider=Provider.OPENAI,
        router_provider=Provider.ANTHROPIC, anthropic_api_key=None, minds_api_key=None,
        openai_api_key=SecretStr("sk-openai"),
        router_model="claude-sonnet-4-6", router_reasoning_effort="max",
    )
    assert settings.resolved_router_provider is Provider.OPENAI
    assert settings.resolved_gate_model == settings.resolved_router_model
    _client, calls = build(settings)
    assert "reasoning_effort" not in calls["openai"][0]


# ── COWORK_OPENAI_COMPATIBLE_API: planning and coding on the Responses API ──

def test_switch_moves_planning_and_coding_to_the_responses_flavor(
    build, openai_compatible_api, anton_responses_ready
):
    openai_compatible_api("responses")
    anton_responses_ready()
    client, calls = build(_openai_compatible())
    router, planning, coding = calls["openai"]
    assert planning["flavor"] == _RealOpenAIProvider.FLAVOR_OPENAI
    assert coding["flavor"] == _RealOpenAIProvider.FLAVOR_OPENAI
    # The router stays on chat completions, as a provider of its own. A router
    # build error would hand the role to the coding provider, which is now on
    # the Responses path, without logging anything.
    assert "flavor" not in router
    assert client.router_provider is not client.coding_provider


def test_switch_covers_a_keyless_openai_compatible_endpoint(
    build, openai_compatible_api, anton_responses_ready
):
    openai_compatible_api("responses")
    anton_responses_ready()
    settings = UserSettings(
        planning_provider=Provider.OPENAI_COMPATIBLE,
        coding_provider=Provider.OPENAI_COMPATIBLE,
        router_provider=Provider.OPENAI_COMPATIBLE,
        planning_model="m", coding_model="m", router_model="m",
        openai_base_url="http://192.168.1.100:1234/v1",
    )
    _client, calls = build(settings)
    router, planning, coding = calls["openai"]
    assert planning["api_key"] == f"{Provider.OPENAI_COMPATIBLE.value}-no-auth"
    assert planning["flavor"] == coding["flavor"] == _RealOpenAIProvider.FLAVOR_OPENAI
    assert "flavor" not in router


def test_switch_leaves_gemini_on_chat_completions(
    build, openai_compatible_api, anton_responses_ready
):
    # Gemini shares the openai_compatible build branch; its endpoint is
    # Google's, which the switch does not name.
    openai_compatible_api("responses")
    anton_responses_ready()
    settings = UserSettings(
        planning_provider=Provider.GEMINI,
        coding_provider=Provider.GEMINI,
        router_provider=Provider.GEMINI,
        gemini_api_key=SecretStr("AIza-key"),
    )
    _client, calls = build(settings)
    assert all("flavor" not in kw for kw in calls["openai"])


def test_switch_waits_for_an_anton_whose_responses_path_is_ready(
    build, openai_compatible_api, anton_responses_ready
):
    # An anton released before its Responses path was fixed reports no
    # readiness. Pinned on the stand-in, so the lock's anton doesn't decide it.
    openai_compatible_api("responses")
    anton_responses_ready(False)
    _client, calls = build(_openai_compatible())
    assert all("flavor" not in kw for kw in calls["openai"])


def _provider_warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == "cowork.services.providers" and r.levelno == logging.WARNING
    ]


def test_a_switch_the_installed_anton_cannot_serve_is_logged_once(
    build, openai_compatible_api, anton_responses_ready, monkeypatch, caplog
):
    # Planning and coding then stay on chat completions, where an effort other
    # than none refuses every tool call. The line names the cause once per
    # process, however many clients are built.
    monkeypatch.setattr(providers, "_warned_responses_transport_missing", False, raising=False)
    openai_compatible_api("responses")
    anton_responses_ready(False)

    with caplog.at_level(logging.WARNING, logger="cowork.services.providers"):
        build(_openai_compatible())
        _client, calls = build(_openai_compatible())

    (line,) = _provider_warnings(caplog)
    assert "COWORK_OPENAI_COMPATIBLE_API=responses" in line
    assert "RESPONSES_TRANSPORT_READY" in line
    assert "chat completions" in line
    assert all("flavor" not in kw for kw in calls["openai"])


@pytest.mark.parametrize(
    ("api", "ready"),
    [(None, False), ("chat_completions", False), ("responses", True)],
)
def test_nothing_is_logged_when_the_switch_is_off_or_served(
    build, openai_compatible_api, anton_responses_ready, monkeypatch, caplog, api, ready
):
    monkeypatch.setattr(providers, "_warned_responses_transport_missing", False, raising=False)
    openai_compatible_api(api)
    anton_responses_ready(ready)

    with caplog.at_level(logging.WARNING, logger="cowork.services.providers"):
        build(_openai_compatible())

    assert _provider_warnings(caplog) == []


@pytest.mark.parametrize("api", [None, "chat_completions"])
def test_openai_compatible_stays_on_chat_completions_unless_switched(
    build, openai_compatible_api, anton_responses_ready, api
):
    openai_compatible_api(api)
    anton_responses_ready()
    _client, calls = build(_openai_compatible())
    assert all("flavor" not in kw for kw in calls["openai"])


def _pin(monkeypatch, *, token="session-jwt", key="mdb_turn_a"):
    """A desktop that sent its mounted organization and holds a key for it."""
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(runtime_credential, "_minds_credential", token)
    monkeypatch.setattr(runtime_credential, "_organization_id", "org-a")
    monkeypatch.setattr(
        runtime_credential,
        "_inference_key",
        runtime_credential.InferenceKey(
            value=key,
            organization_id="org-a",
            instance_id="desktop-1",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        if key
        else None,
    )


def _minds_settings(key="session-jwt"):
    return UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr(key),
        minds_url="https://api.mindshub.ai",
    )


@pytest.mark.asyncio
async def test_a_pinned_desktop_bills_its_turn_key_not_the_session_token(build, monkeypatch):
    _pin(monkeypatch)

    _client, calls = build(_minds_settings())

    assert all(kw["api_key"] == "mdb_turn_a" for kw in calls["openai"])
    # A web switch hands over a token for another org; the key doesn't follow it.
    monkeypatch.setattr(runtime_credential, "_minds_credential", "token-now-org-b")
    assert [await kw["api_key_provider"]() for kw in calls["openai"]] == ["mdb_turn_a"] * 3


def test_a_pinned_desktop_without_a_key_refuses_to_build(build, monkeypatch):
    _pin(monkeypatch, key=None)

    with pytest.raises(ValueError, match="not configured"):
        build(_minds_settings())


def test_inference_key_str_falls_back_to_settings_without_a_hand_over():
    from cowork.common.settings.user_settings import inference_api_key_str

    assert inference_api_key_str(_minds_settings("stored-key"), Provider.MINDS_CLOUD) == "stored-key"


def test_inference_key_str_pins_only_minds_cloud(monkeypatch):
    from cowork.common.settings.user_settings import inference_api_key_str

    _pin(monkeypatch)
    settings = UserSettings(minds_api_key=SecretStr("session-jwt"), openai_api_key=SecretStr("sk-openai"))

    assert inference_api_key_str(settings, Provider.MINDS_CLOUD) == "mdb_turn_a"
    assert inference_api_key_str(settings, Provider.OPENAI) == "sk-openai"


def test_inference_key_str_is_empty_when_pinned_without_a_key(monkeypatch):
    from cowork.common.settings.user_settings import inference_api_key_str

    _pin(monkeypatch, key=None)

    assert inference_api_key_str(_minds_settings(), Provider.MINDS_CLOUD) == ""
