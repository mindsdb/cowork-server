"""Readiness in org mode, where no provider key is stored by design.

Cloud mints a short-lived per-turn credential (turnqueue/auth_keys) and hands it
to the pod, so a cloud tenant has nothing in ``minds_api_key`` and never should.
Gating ``config_ready`` on a stored key therefore reported every cloud org as
unconfigured, which sent it to onboarding — and onboarding's only way forward is
writing ``planning_provider``/``coding_provider``, which are admin-owned, so a
non-admin member got a 403 and could never reach the app.
"""

import json

import pytest

from cowork.common.settings import user_settings as us
from cowork.common.settings.app_settings import (
    CODING_MODEL_DEFAULTS,
    MINDS_FREE_MODEL,
    PLANNING_MODEL_DEFAULTS,
    ROUTER_MODEL_DEFAULTS,
)
from cowork.common.settings.user_settings import Provider, UserSettings


# UserSettings reads bare env vars (ANTHROPIC_API_KEY, …) and the .env chain, so
# a developer's exported key would otherwise make a provider look configured and
# win the readiness resolver — passing locally, failing in CI, or vice versa.
_KEY_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "MINDS_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "PLANNING_PROVIDER",
    "CODING_PROVIDER",
)


def _mode(monkeypatch, mode: str) -> None:
    """Set tenancy_mode via the real settings (env + cache clear), so every
    module's get_app_settings() agrees — user_settings, app_settings, and
    providers all read the same source."""
    for name in _KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COWORK_TENANCY_MODE", mode)
    us.get_app_settings.cache_clear()


def _settings(**kw) -> UserSettings:
    """No .env chain: the file would reintroduce the keys _mode just cleared."""
    return UserSettings(_env_file=None, **kw)


@pytest.fixture
def org_mode(monkeypatch):
    _mode(monkeypatch, "org")
    yield
    us.get_app_settings.cache_clear()


@pytest.fixture
def local_mode(monkeypatch):
    _mode(monkeypatch, "local")
    yield
    us.get_app_settings.cache_clear()


class TestOrgModeDefaults:
    def test_org_mode_defaults_to_minds_cloud(self, org_mode):
        s = _settings()
        assert s.planning_provider is Provider.MINDS_CLOUD
        assert s.coding_provider is Provider.MINDS_CLOUD
        assert s.router_provider is Provider.MINDS_CLOUD

    def test_desktop_still_defaults_to_anthropic(self, local_mode):
        s = _settings()
        assert s.planning_provider is Provider.ANTHROPIC
        assert s.coding_provider is Provider.ANTHROPIC


class TestOrgModeReadiness:
    def test_ready_with_nothing_stored(self, org_mode):
        """The case that stranded every cloud org on the onboarding screen."""
        cs = _settings().config_status

        assert cs["config_ready"] is True
        assert cs["config_error"] is None
        assert cs["provider"] == Provider.MINDS_CLOUD.value

    def test_models_resolve_to_the_minds_defaults(self, org_mode):
        s = _settings()

        # Readiness must imply build_llm_client can actually run: both roles
        # resolve, or the turn throws despite reading as ready.
        assert s.resolved_planning_model == "mindshub_air"
        assert s.resolved_coding_model == "mindshub_air"

    def test_desktop_with_nothing_stored_is_still_unconfigured(self, local_mode):
        """The bypass is org-mode only — desktop must keep asking for a key."""
        cs = _settings().config_status

        assert cs["config_ready"] is False
        assert "API key" in cs["config_error"]

    def test_byok_without_a_key_falls_back_to_minds_in_org_mode(self, org_mode):
        """BYOK in cloud is deferred: an org that selected a BYOK provider but
        stored no key falls back to the managed MindsHub provider (ready) rather
        than blocking — minds-cloud is always keyed in org mode (per-turn mint)."""
        cs = _settings(
            planning_provider=Provider.ANTHROPIC,
            coding_provider=Provider.ANTHROPIC,
        ).config_status

        assert cs["config_ready"] is True
        assert cs["provider"] == Provider.MINDS_CLOUD.value


_PLANNING = PLANNING_MODEL_DEFAULTS["minds_cloud"]
_CODING = CODING_MODEL_DEFAULTS["minds_cloud"]
_ROUTER = ROUTER_MODEL_DEFAULTS["minds_cloud"]


class TestOrgModeCreditAwareDefaults:
    """An org with credit gets the canonical defaults; anything short of
    positive evidence in the availability map stays on MINDS_FREE_MODEL."""

    PAID = json.dumps({MINDS_FREE_MODEL: True, _PLANNING: True, _CODING: True, _ROUTER: True})
    FREE = json.dumps({MINDS_FREE_MODEL: True, _PLANNING: False, _CODING: False, _ROUTER: False})

    def test_org_with_credits_gets_premium_defaults(self, org_mode):
        s = _settings(minds_model_enabled=self.PAID)
        assert s.resolved_planning_model == _PLANNING
        assert s.resolved_coding_model == _CODING
        assert s.resolved_router_model == _ROUTER

    def test_org_without_credits_stays_on_free_model(self, org_mode):
        s = _settings(minds_model_enabled=self.FREE)
        assert s.resolved_planning_model == MINDS_FREE_MODEL
        assert s.resolved_coding_model == MINDS_FREE_MODEL
        assert s.resolved_router_model == MINDS_FREE_MODEL

    def test_org_cold_start_stays_on_free_model(self, org_mode):
        # Empty map (fetch never ran) is not evidence of credit — free-first.
        s = _settings()
        assert s.resolved_router_model == MINDS_FREE_MODEL

    def test_org_default_missing_from_map_stays_free(self, org_mode):
        # Unlike desktop (missing = available), org needs the default itself
        # marked payable; a gateway that stops listing it downgrades to free.
        s = _settings(minds_model_enabled=json.dumps({MINDS_FREE_MODEL: True, _PLANNING: True}))
        assert s.resolved_planning_model == _PLANNING
        assert s.resolved_coding_model == MINDS_FREE_MODEL

    def test_org_explicit_stored_choice_survives_the_default_fill(self, org_mode):
        # apply_model_defaults only fills a None model at construction time —
        # an explicitly stored value (whatever it is) is never touched there,
        # regardless of what resolved_planning_model later does with it.
        s = _settings(minds_model_enabled=self.FREE, planning_model="opus")
        assert s.planning_model == "opus"

    def test_org_explicit_choice_of_an_available_model_is_not_rewritten(self, org_mode):
        # A real, currently-enabled MindsHub pick is left alone by the
        # wallet-aware fallback below — only a locked/foreign id gets swapped.
        s = _settings(minds_model_enabled=self.PAID, planning_model=_PLANNING)
        assert s.resolved_planning_model == _PLANNING

    def test_org_explicit_choice_of_a_locked_or_foreign_model_falls_back(self, org_mode):
        # wallet_aware (ENG-1632 follow-up): planning gets the same fallback
        # as coding/router — an id absent from a non-empty map (e.g. a BYOK
        # id like "opus" surviving a MindsHub SSO sign-in, or a locked
        # MindsHub model) resolves to the first enabled model instead of
        # 404ing the gateway on every turn.
        s = _settings(minds_model_enabled=self.FREE, planning_model="opus")
        assert s.resolved_planning_model == MINDS_FREE_MODEL


def test_ambient_provider_keys_do_not_override_minds(monkeypatch, org_mode):
    """The live bug: the pod carries ANTHROPIC_API_KEY/OPENAI_API_KEY in its env,
    so the resolver picked Anthropic and the UI showed it instead of MindsHub.
    Org mode must resolve to MindsHub regardless of ambient provider keys."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-ambient")
    cs = UserSettings().config_status  # reads the ambient env, like the pod

    assert cs["provider"] == Provider.MINDS_CLOUD.value
    assert cs["model"] == "mindshub_air"
    assert cs["config_ready"] is True


class TestUserSettingsIgnoresProcessEnvInOrgMode:
    """A shared cloud server injects provider secrets as process env vars
    (ANTHROPIC_API_KEY, ...). Per-user UserSettings must not read them in org
    mode, or every tenant sees another config's keys as connected and the
    Agent-Models UI resolves to a BYOK provider (staging bug). Desktop/local
    keeps env reading so the standalone CLI / .env-first flow still work.
    """

    def test_org_mode_ignores_bare_provider_env(self, monkeypatch, org_mode):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-leak")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-leak")
        s = UserSettings(_env_file=None)
        assert s.anthropic_api_key is None
        assert s.openai_api_key is None
        # Default provider stays MindsHub, not the leaked BYOK key.
        assert s.planning_provider == Provider.MINDS_CLOUD

    def test_local_mode_still_reads_provider_env(self, monkeypatch, local_mode):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-local")
        s = UserSettings(_env_file=None)
        assert s.anthropic_api_key is not None
        assert s.anthropic_api_key.get_secret_value() == "sk-ant-local"
