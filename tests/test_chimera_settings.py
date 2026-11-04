"""Chimera's settings: the sidekick role (shaped like the router's) and the
switches that turn Chimera on.

Everything is off by default, so a fresh install builds every turn exactly as
before; the sidekick resolves like the other roles, wallet-aware, starting on
the coding model; and only the sidekick's provider rides the .env seed.
"""
import json

import pytest
from pydantic import SecretStr

from cowork.common.settings.app_settings import (
    CHIMERA_MAIN_MODEL_DEFAULT,
    CHIMERA_SIDEKICK_MODEL_DEFAULT,
    CODING_MODEL_DEFAULTS,
    MINDS_FREE_MODEL,
)
from cowork.common.settings.user_settings import (
    ENV_ALIAS_TO_SETTING,
    SETTING_ENV_ALIASES,
    Provider,
    UserSettings,
    setting_is_operator_pinned,
    setting_is_org_scoped,
)


def _settings(**kw) -> UserSettings:
    return UserSettings(_env_file=None, **kw)


# ── Defaults ─────────────────────────────────────────────────────────

def test_chimera_is_off_by_default():
    settings = _settings()
    assert settings.chimera_enabled is False
    assert settings.sidekick_reasoning_effort is None


def test_the_pseudo_models_pair_defaults_to_the_contract_pair():
    settings = _settings()
    assert settings.chimera_main_model == CHIMERA_MAIN_MODEL_DEFAULT == "opus"
    assert settings.chimera_sidekick_model == CHIMERA_SIDEKICK_MODEL_DEFAULT == "gpt-luna-6"


@pytest.mark.parametrize(
    "provider", [Provider.ANTHROPIC, Provider.OPENAI, Provider.GEMINI, Provider.MINDS_CLOUD]
)
def test_an_unset_sidekick_model_starts_on_the_coding_default(provider):
    settings = _settings(sidekick_provider=provider)
    assert settings.sidekick_model == CODING_MODEL_DEFAULTS[provider.value]


def test_the_sidekick_default_follows_its_own_provider_not_the_coding_one():
    # The router's default reads coding_provider (a known wart); the sidekick
    # does not repeat it.
    settings = _settings(coding_provider=Provider.ANTHROPIC, sidekick_provider=Provider.OPENAI)
    assert settings.sidekick_model == CODING_MODEL_DEFAULTS["openai"]


def test_an_explicit_sidekick_model_is_kept():
    assert _settings(sidekick_model="kimi").sidekick_model == "kimi"


# ── Resolution ───────────────────────────────────────────────────────

def test_the_sidekick_provider_resolves_to_a_keyed_one():
    settings = _settings(
        sidekick_provider=Provider.ANTHROPIC,
        anthropic_api_key=None,
        minds_api_key=None,
        openai_api_key=SecretStr("sk-openai"),
    )
    assert settings.resolved_sidekick_provider is Provider.OPENAI
    # Switched provider: the provider's default, never the other provider's model.
    assert settings.resolved_sidekick_model == CODING_MODEL_DEFAULTS["openai"]


def test_a_locked_sidekick_pin_falls_back_to_an_affordable_model():
    settings = _settings(
        sidekick_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb"),
        sidekick_model="opus",
        minds_model_enabled=json.dumps({MINDS_FREE_MODEL: True, "opus": False}),
    )
    assert settings.resolved_sidekick_model == MINDS_FREE_MODEL


def test_the_catalog_can_declare_the_sidekick_default():
    settings = _settings(
        sidekick_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb"),
        minds_role_defaults=json.dumps({"sidekick": "gpt-sol"}),
        minds_model_enabled=json.dumps({MINDS_FREE_MODEL: True, "gpt-sol": True}),
    )
    assert settings.sidekick_model == "gpt-sol"
    assert settings.resolved_sidekick_model == "gpt-sol"


# ── Env and scope ────────────────────────────────────────────────────

def test_only_the_sidekick_provider_rides_the_env_seed():
    assert SETTING_ENV_ALIASES["sidekick_provider"] == "ANTON_SIDEKICK_PROVIDER"
    assert ENV_ALIAS_TO_SETTING["ANTON_SIDEKICK_PROVIDER"] == "sidekick_provider"
    # A model is CLI-only (ENG-739), and a stale CLI line must not switch on a
    # paid second agent for every turn.
    for key in (
        "sidekick_model", "sidekick_reasoning_effort", "chimera_enabled",
        "chimera_main_model", "chimera_sidekick_model",
    ):
        assert key not in SETTING_ENV_ALIASES, key


def test_the_field_name_env_var_switches_chimera_on(monkeypatch):
    # The by-field-name path (local tenancy) is how an operator or an eval sets
    # it without a settings row.
    monkeypatch.setenv("CHIMERA_ENABLED", "true")
    monkeypatch.setenv("SIDEKICK_MODEL", "kimi")
    settings = UserSettings()
    assert settings.chimera_enabled is True
    assert settings.sidekick_model == "kimi"


def test_the_sidekick_is_scoped_like_the_router():
    assert setting_is_org_scoped("sidekick_provider") is True
    assert setting_is_org_scoped("sidekick_model") is False
    for key in (
        "sidekick_provider", "sidekick_model", "sidekick_reasoning_effort",
        "chimera_enabled", "chimera_main_model", "chimera_sidekick_model",
    ):
        assert setting_is_operator_pinned(key) is True, key


def test_an_empty_pseudo_model_pair_is_refused():
    with pytest.raises(ValueError):
        UserSettings.model_validate({"chimera_main_model": ""})
