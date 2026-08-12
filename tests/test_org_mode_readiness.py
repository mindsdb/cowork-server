"""Readiness in org mode, where no provider key is stored by design.

Cloud mints a short-lived per-turn credential (turnqueue/auth_keys) and hands it
to the pod, so a cloud tenant has nothing in ``minds_api_key`` and never should.
Gating ``config_ready`` on a stored key therefore reported every cloud org as
unconfigured, which sent it to onboarding — and onboarding's only way forward is
writing ``planning_provider``/``coding_provider``, which are admin-owned, so a
non-admin member got a 403 and could never reach the app.
"""

import pytest

from cowork.common.settings import user_settings as us
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
    """Override only tenancy_mode. A stub object won't do — several field
    factories in UserSettings read other app settings (channels_harness, the
    tool budgets), so the real settings must stay intact."""
    for name in _KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    real = us.get_app_settings()
    patched = real.model_copy(update={"tenancy_mode": mode})
    monkeypatch.setattr(us, "get_app_settings", lambda: patched)


def _settings(**kw) -> UserSettings:
    """No .env chain: the file would reintroduce the keys _mode just cleared."""
    return UserSettings(_env_file=None, **kw)


@pytest.fixture
def org_mode(monkeypatch):
    _mode(monkeypatch, "org")


@pytest.fixture
def local_mode(monkeypatch):
    _mode(monkeypatch, "local")


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
        assert s.resolved_planning_model == "sonnet"
        assert s.resolved_coding_model == "haiku"

    def test_desktop_with_nothing_stored_is_still_unconfigured(self, local_mode):
        """The bypass is org-mode only — desktop must keep asking for a key."""
        cs = _settings().config_status

        assert cs["config_ready"] is False
        assert "API key" in cs["config_error"]

    def test_byok_provider_in_org_mode_still_needs_its_key(self, org_mode):
        """Scoped to MindsHub: only minds-cloud credentials are minted per turn,
        so an org that deliberately selected a BYOK provider is not ready until
        it supplies that provider's key."""
        cs = _settings(
            planning_provider=Provider.ANTHROPIC,
            coding_provider=Provider.ANTHROPIC,
        ).config_status

        assert cs["config_ready"] is False
