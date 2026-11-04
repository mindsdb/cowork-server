"""build_llm_client's sidekick role (Chimera).

The sidekick reaches anton only when the turn asks for it AND the installed
LLMClient takes the kwargs, guarded exactly as the router's are: the uv.lock
anton predates them, and passing one it does not declare would raise on every
turn. These tests stand a recording LLMClient in for anton's, with and without
the new parameters, and capture each provider's constructor kwargs.
"""
import inspect
from unittest.mock import MagicMock

import pytest
from anton.core.llm.openai import OpenAIProvider as _RealOpenAIProvider
from pydantic import SecretStr

from cowork.common.settings.app_settings import CODING_MODEL_DEFAULTS
from cowork.common.settings.user_settings import Provider, UserSettings


class _ClientWithSidekick:
    """An LLMClient as the Chimera anton declares it."""

    def __init__(
        self, *, planning_provider, planning_model, coding_provider, coding_model,
        router_provider=None, router_model=None, sidekick_provider=None, sidekick_model=None,
        main_cache_ttl=None, max_tokens=8192,
    ):
        self.kwargs = dict(
            planning_provider=planning_provider, planning_model=planning_model,
            coding_provider=coding_provider, coding_model=coding_model,
            router_provider=router_provider, router_model=router_model,
            sidekick_provider=sidekick_provider, sidekick_model=sidekick_model,
            main_cache_ttl=main_cache_ttl,
        )


class _ClientWithoutSidekick:
    """An LLMClient as the uv.lock anton declares it: router, no sidekick."""

    def __init__(
        self, *, planning_provider, planning_model, coding_provider, coding_model,
        router_provider=None, router_model=None, max_tokens=8192,
    ):
        self.kwargs = dict(
            planning_provider=planning_provider, planning_model=planning_model,
            coding_provider=coding_provider, coding_model=coding_model,
            router_provider=router_provider, router_model=router_model,
        )


@pytest.fixture
def build(monkeypatch):
    """``build(settings, client_cls, **kw) -> (client, provider_calls)``."""
    calls: list[dict] = []

    def _fake_openai(**kw):
        calls.append(kw)
        return MagicMock(name="OpenAIProvider", kwargs=kw)

    for name in (
        "FLAVOR_OPENAI", "FLAVOR_MINDS_PASSTHROUGH", "FLAVOR_OPENAI_COMPATIBLE_GENERIC",
    ):
        setattr(_fake_openai, name, getattr(_RealOpenAIProvider, name))
    _fake_openai.RESPONSES_TRANSPORT_READY = getattr(_RealOpenAIProvider, "RESPONSES_TRANSPORT_READY", False)
    _fake_openai.__signature__ = inspect.signature(_RealOpenAIProvider)
    monkeypatch.setattr("anton.core.llm.openai.OpenAIProvider", _fake_openai)

    def _anthropic(**kw):
        calls.append({"anthropic": True, **kw})
        return MagicMock(name="AnthropicProvider", kwargs=kw)

    monkeypatch.setattr("anton.core.llm.anthropic.AnthropicProvider", _anthropic)

    def _build(settings, client_cls, **kw):
        monkeypatch.setattr("anton.core.llm.client.LLMClient", client_cls)
        from cowork.services.providers import build_llm_client

        calls.clear()
        return build_llm_client(settings=settings, **kw), calls

    return _build


def _minds(**kw) -> UserSettings:
    return UserSettings(**{
        "_env_file": None,
        "planning_provider": Provider.MINDS_CLOUD,
        "coding_provider": Provider.MINDS_CLOUD,
        "router_provider": Provider.MINDS_CLOUD,
        "sidekick_provider": Provider.MINDS_CLOUD,
        "minds_api_key": SecretStr("mdb-key"),
        "minds_url": "https://api.mindshub.ai",
        **kw,
    })


def test_no_sidekick_unless_the_turn_asks_for_one(build):
    client, calls = build(_minds(), _ClientWithSidekick)
    assert client.kwargs["sidekick_provider"] is None
    assert client.kwargs["sidekick_model"] is None
    assert client.kwargs["main_cache_ttl"] is None
    # router, planning, coding: no fourth provider built.
    assert len(calls) == 3


def test_a_chimera_turn_gets_the_sidekick_settings(build):
    settings = _minds(sidekick_model="kimi", sidekick_reasoning_effort="low")
    client, calls = build(settings, _ClientWithSidekick, sidekick=True)
    assert client.kwargs["sidekick_model"] == "kimi"
    assert client.kwargs["sidekick_provider"].kwargs["reasoning_effort"] == "low"
    assert client.kwargs["sidekick_provider"].kwargs["flavor"] == _RealOpenAIProvider.FLAVOR_MINDS_PASSTHROUGH
    # Built before planning and coding, so the coding provider stays the last.
    router, sidekick, planning, coding = calls
    assert client.kwargs["coding_provider"].kwargs is coding


def test_an_anton_without_the_sidekick_role_gets_no_kwarg(build):
    # The uv.lock anton: passing sidekick_* would TypeError every turn.
    client, calls = build(_minds(), _ClientWithoutSidekick, sidekick=True, main_cache_ttl="1h")
    assert "sidekick_provider" not in client.kwargs
    assert len(calls) == 3


def test_the_pseudo_models_sidekick_runs_on_the_planning_provider(build):
    # The pair's sidekick is a model from the planning provider's catalogue,
    # whatever the sidekick settings name.
    settings = _minds(
        sidekick_provider=Provider.ANTHROPIC, anthropic_api_key=SecretStr("sk-ant"),
    )
    client, _calls = build(
        settings, _ClientWithSidekick,
        model_override="opus", sidekick=True, sidekick_model_override="gpt-sol",
    )
    assert client.kwargs["planning_model"] == "opus"
    assert client.kwargs["sidekick_model"] == "gpt-sol"
    assert client.kwargs["sidekick_provider"].kwargs["base_url"].startswith("https://api.mindshub.ai")


def test_the_composer_pick_never_reaches_the_sidekick(build):
    client, _calls = build(
        _minds(), _ClientWithSidekick, model_override="opus", effort_override="high", sidekick=True,
    )
    assert client.kwargs["planning_model"] == "opus"
    assert client.kwargs["sidekick_model"] == CODING_MODEL_DEFAULTS["minds_cloud"]
    assert "reasoning_effort" not in client.kwargs["sidekick_provider"].kwargs


def test_the_sidekick_effort_stays_with_the_model_it_was_chosen_for(build):
    # The stored effort was chosen for kimi; the pair's sidekick is another model.
    settings = _minds(sidekick_model="kimi", sidekick_reasoning_effort="low")
    client, _calls = build(
        settings, _ClientWithSidekick, sidekick=True, sidekick_model_override="gpt-sol",
    )
    assert "reasoning_effort" not in client.kwargs["sidekick_provider"].kwargs


def test_the_cache_ttl_is_passed_only_to_a_client_that_takes_it(build):
    client, _calls = build(_minds(), _ClientWithSidekick, sidekick=True, main_cache_ttl="1h")
    assert client.kwargs["main_cache_ttl"] == "1h"


def test_a_sidekick_that_cannot_be_built_leaves_the_turn_running(build):
    # An openai-compatible sidekick with no base URL: building it raises, and
    # the turn runs on its other roles without Chimera rather than failing.
    settings = _minds(
        sidekick_provider=Provider.OPENAI_COMPATIBLE,
        sidekick_model="local-model",
        openai_compatible_api_key=SecretStr("sk-compat"),
        openai_base_url="",
    )
    assert settings.resolved_sidekick_provider is Provider.OPENAI_COMPATIBLE
    client, calls = build(settings, _ClientWithSidekick, sidekick=True)
    assert client.kwargs["sidekick_provider"] is None
    assert client.kwargs["planning_model"] == settings.resolved_planning_model
    assert len(calls) == 3
