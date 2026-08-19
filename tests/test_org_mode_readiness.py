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
from cowork.common.settings.app_settings import MINDS_FREE_MODEL
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


class TestOrgModeCreditAwareDefaults:
    """Org mode needs positive evidence in the availability map before it hands
    out the canonical default, and falls back to MINDS_FREE_MODEL otherwise.

    Every minds-cloud role default IS MINDS_FREE_MODEL now, so the two arms of
    that branch return the same string for every real map: an org resolves to the
    free model whether its wallet can pay or not. That makes a fixture built from
    the role defaults useless here — ``{MINDS_FREE_MODEL: True, _PLANNING: False}``
    is one key, not two, and every assertion becomes
    ``mindshub_air == mindshub_air``. So the map cases below assert the single
    outcome they actually have, and the branch itself is pinned separately by
    ``test_org_needs_positive_evidence_for_a_non_free_default``, which is the only
    test here that can still tell the two arms apart.
    """

    # A funded wallet and no map at all: every role lands on the free model,
    # because that is what the canonical default is now.
    @pytest.mark.parametrize(
        "enabled_map",
        [
            pytest.param(json.dumps({MINDS_FREE_MODEL: True, "sonnet": True}), id="funded"),
            pytest.param(None, id="cold-start"),
        ],
    )
    def test_org_resolves_every_role_to_the_free_model(self, org_mode, enabled_map):
        s = _settings(**({"minds_model_enabled": enabled_map} if enabled_map else {}))
        assert s.resolved_planning_model == MINDS_FREE_MODEL
        assert s.resolved_coding_model == MINDS_FREE_MODEL
        assert s.resolved_router_model == MINDS_FREE_MODEL

    # Once the map stops vouching for the free model the three roles diverge, and
    # the org branch is not what decides it for two of them.
    @pytest.mark.parametrize(
        "enabled_map",
        [
            pytest.param(json.dumps({MINDS_FREE_MODEL: False, "sonnet": True}), id="drained"),
            pytest.param(json.dumps({"sonnet": True}), id="default-unlisted"),
        ],
    )
    def test_org_planning_holds_the_free_model_while_the_aux_roles_move(
        self, org_mode, enabled_map
    ):
        """Planning goes through the org branch; coding and router do not.

        The org arm of ``_enabled_aware_default`` has no first-enabled fallback, so
        planning stays on MINDS_FREE_MODEL even when the map says it is not
        callable. Coding and router resolve through ENG-1632's wallet-aware path
        instead, which does substitute the first enabled alias. So a drained org
        runs its two invisible roles on a paid model while the visible one points
        at a model the map has disabled. That is the fully-drained account
        ENG-1652 puts out of scope, pinned here so the split is on the record
        rather than discovered again.
        """
        s = _settings(minds_model_enabled=enabled_map)
        assert s.resolved_planning_model == MINDS_FREE_MODEL
        assert s.resolved_coding_model == "sonnet"
        assert s.resolved_router_model == "sonnet"

    @pytest.mark.parametrize("listed", [True, False])
    def test_org_needs_positive_evidence_for_a_non_free_default(
        self, org_mode, monkeypatch, listed
    ):
        """The branch, tested where it still has two answers.

        Point the planning default at a paid alias and the org arm becomes
        observable again: listed-and-enabled keeps it, anything less falls back to
        the free model. This is the test that fails if someone collapses the
        branch to an unconditional return, and it is the one to delete or rewrite
        if org is ever given a non-free default of its own.
        """
        monkeypatch.setitem(us.PLANNING_MODEL_DEFAULTS, "minds_cloud", "sonnet")
        s = _settings(minds_model_enabled=json.dumps({MINDS_FREE_MODEL: True, "sonnet": listed}))
        assert s.resolved_planning_model == ("sonnet" if listed else MINDS_FREE_MODEL)

    def test_org_explicit_choice_is_never_rewritten(self, org_mode):
        s = _settings(
            minds_model_enabled=json.dumps({MINDS_FREE_MODEL: False}), planning_model="opus"
        )
        assert s.resolved_planning_model == "opus"


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
