"""Tests for provider_api_key — per-provider key resolution with fallback.

gemini and openai-compatible now have dedicated key slots but fall back to the
shared openai_api_key when their slot is empty, so existing single-key configs
keep working with no migration. This is the key half of the shared-slot
isolation (the base-URL half is provider_base_url).
"""

from pydantic import SecretStr

from cowork.common.settings.user_settings import (
    Provider,
    UserSettings,
    provider_api_key,
)


def _settings(**kw):
    return UserSettings(**kw)


def _val(secret):
    return secret.get_secret_value() if secret else None


class TestProviderApiKey:
    def test_openai_reads_own_slot(self):
        s = _settings(openai_api_key=SecretStr("sk-openai"))
        assert _val(provider_api_key(s, Provider.OPENAI)) == "sk-openai"

    def test_anthropic_reads_own_slot(self):
        s = _settings(anthropic_api_key=SecretStr("sk-ant"))
        assert _val(provider_api_key(s, Provider.ANTHROPIC)) == "sk-ant"

    def test_minds_reads_own_slot(self):
        s = _settings(minds_api_key=SecretStr("mdb-key"))
        assert _val(provider_api_key(s, Provider.MINDS_CLOUD)) == "mdb-key"

    def test_gemini_uses_dedicated_slot_when_set(self):
        s = _settings(
            gemini_api_key=SecretStr("AIza-gemini"),
            openai_api_key=SecretStr("sk-openai"),
        )
        # Dedicated slot wins; the OpenAI key is NOT used.
        assert _val(provider_api_key(s, Provider.GEMINI)) == "AIza-gemini"

    def test_gemini_falls_back_to_openai_when_unset(self):
        # Existing config: only the shared slot is set → gemini keeps working.
        s = _settings(openai_api_key=SecretStr("sk-shared"))
        assert _val(provider_api_key(s, Provider.GEMINI)) == "sk-shared"

    def test_openai_compatible_uses_dedicated_slot_when_set(self):
        s = _settings(
            openai_compatible_api_key=SecretStr("sk-compat"),
            openai_api_key=SecretStr("sk-openai"),
        )
        assert _val(provider_api_key(s, Provider.OPENAI_COMPATIBLE)) == "sk-compat"

    def test_openai_compatible_falls_back_to_openai_when_unset(self):
        s = _settings(openai_api_key=SecretStr("sk-shared"))
        assert _val(provider_api_key(s, Provider.OPENAI_COMPATIBLE)) == "sk-shared"

    def test_openai_does_not_fall_back(self):
        # openai has no fallback — only its own slot.
        s = _settings(gemini_api_key=SecretStr("AIza"))
        assert provider_api_key(s, Provider.OPENAI) is None

    def test_all_unset_returns_none(self):
        assert provider_api_key(_settings(), Provider.GEMINI) is None


class TestConfigStatusFallback:
    def test_gemini_planning_on_shared_key_reads_as_configured(self):
        # Regression guard: a gemini user relying on the shared openai key must
        # still report config_ready (was broken when config_status read the
        # dedicated slot directly).
        s = _settings(
            planning_provider=Provider.GEMINI,
            openai_api_key=SecretStr("sk-shared"),
        )
        assert s.config_status["config_ready"] is True

    def test_gemini_planning_with_no_key_reads_as_not_configured(self):
        s = _settings(planning_provider=Provider.GEMINI)
        assert s.config_status["config_ready"] is False


class TestResolverIsolation:
    """Readiness resolver (#93) + dedicated key slots (#113): a user who
    configured ONLY a gemini / openai-compatible key while planning_provider
    still points at a keyless provider must resolve to that provider and read
    as ready. Guards the bug that emerges from combining the two."""

    def test_gemini_only_key_resolves_when_planning_is_keyless(self):
        s = _settings(
            planning_provider=Provider.ANTHROPIC,   # keyless
            gemini_api_key=SecretStr("AIza-only"),
        )
        assert s.resolved_planning_provider == Provider.GEMINI
        assert s.config_status["config_ready"] is True
        # The resolved MODEL must be non-None — otherwise build_llm_client gets
        # model=None and throws despite config_ready=True (the live bug).
        assert s.resolved_planning_model == "gemini-2.5-pro"
        assert s.resolved_coding_model == "gemini-2.5-flash"

    def test_switch_to_oc_does_not_misroute_the_default_model(self):
        # Realistic case: planning_provider=anthropic (keyless), only an OC key.
        # apply_model_defaults fills planning_model with anthropic's DEFAULT
        # (claude-sonnet-4-6). The resolver switches to OC — but must NOT hand
        # that Claude id to the OC endpoint. resolved model is None → not ready
        # → "select a model" (rather than silently misrouting).
        s = _settings(
            planning_provider=Provider.ANTHROPIC,   # keyless
            openai_compatible_api_key=SecretStr("sk-compat-only"),
        )
        assert s.planning_model == "claude-sonnet-4-6"   # filled by apply_model_defaults
        assert s.resolved_planning_provider == Provider.OPENAI_COMPATIBLE
        assert s.resolved_planning_model is None          # NOT the Claude id
        cs = s.config_status
        assert cs["config_ready"] is False
        assert "model" in cs["config_error"].lower()

    def test_switch_to_gemini_uses_gemini_default_not_original(self):
        # Switch to gemini (which HAS a default) must use the gemini default,
        # never the original provider's model.
        s = _settings(
            planning_provider=Provider.ANTHROPIC,   # keyless; planning_model→claude default
            gemini_api_key=SecretStr("AIza"),
        )
        assert s.resolved_planning_provider == Provider.GEMINI
        assert s.resolved_planning_model == "gemini-2.5-pro"
        assert s.resolved_coding_model == "gemini-2.5-flash"
        assert s.config_status["config_ready"] is True

    def test_legacy_shared_openai_key_still_resolves(self):
        # Legacy single-key config (only the shared openai slot) keeps working
        # via the fallback — resolver finds OPENAI.
        s = _settings(
            planning_provider=Provider.ANTHROPIC,   # keyless
            openai_api_key=SecretStr("sk-shared"),
        )
        assert s.resolved_planning_provider == Provider.OPENAI
        assert s.config_status["config_ready"] is True

    def test_nothing_configured_returns_preferred_and_not_ready(self):
        s = _settings(planning_provider=Provider.ANTHROPIC)
        assert s.resolved_planning_provider == Provider.ANTHROPIC
        assert s.config_status["config_ready"] is False

    def test_openai_compatible_key_but_no_model_is_not_ready(self):
        # openai-compatible has a key but no model and no canonical default →
        # build_llm_client would get model=None. config_ready must be False so
        # we don't claim "ready" for a client that throws on first use.
        s = _settings(
            planning_provider=Provider.OPENAI_COMPATIBLE,
            planning_model=None,
            openai_compatible_api_key=SecretStr("sk-compat"),
        )
        assert s.resolved_planning_provider == Provider.OPENAI_COMPATIBLE
        assert s.resolved_planning_model is None
        cs = s.config_status
        assert cs["config_ready"] is False
        assert "model" in cs["config_error"].lower()

    def test_openai_compatible_fully_configured_is_ready(self):
        # Key + planning model + coding model + base URL all set → ready.
        s = _settings(
            planning_provider=Provider.OPENAI_COMPATIBLE,
            planning_model="my-model",
            coding_provider=Provider.OPENAI_COMPATIBLE,
            coding_model="my-coding-model",
            openai_compatible_api_key=SecretStr("sk-compat"),
            openai_base_url="https://my-host/v1",
        )
        assert s.config_status["config_ready"] is True

    def test_coding_model_gate_blocks_ready(self):
        # planning is fully fine (anthropic + key); coding resolves to OC with a
        # key but no coding model and no canonical default → build_llm_client
        # builds BOTH roles and would hand coding_model=None to the provider.
        # config_ready must be False (the coding axis of the planning fix).
        s = _settings(
            planning_provider=Provider.ANTHROPIC,
            anthropic_api_key=SecretStr("sk-ant"),
            coding_provider=Provider.OPENAI_COMPATIBLE,
            coding_model=None,
            openai_compatible_api_key=SecretStr("sk-compat"),
            openai_base_url="https://my-host/v1",
        )
        assert s.resolved_planning_model  # planning side fine
        assert s.resolved_coding_model is None
        cs = s.config_status
        assert cs["config_ready"] is False
        assert "coding" in cs["config_error"].lower()

    def test_openai_compatible_no_base_is_not_ready(self):
        # OC fully modeled but no base URL — provider_base_url returns None (it
        # must NOT silently fall back to api.openai.com, which would leak the BYO
        # key to OpenAI), so config_status surfaces it instead of reading ready.
        s = _settings(
            planning_provider=Provider.OPENAI_COMPATIBLE,
            planning_model="my-model",
            coding_provider=Provider.OPENAI_COMPATIBLE,
            coding_model="my-coding-model",
            openai_compatible_api_key=SecretStr("sk-compat"),
            # openai_base_url intentionally unset
        )
        cs = s.config_status
        assert cs["config_ready"] is False
        assert "base url" in cs["config_error"].lower()


class TestCheckConfiguredGeminiOnly:
    """A gemini-only user (dedicated key, no shared openai key) must read as
    configured — /configured gates app startup (App.tsx)."""

    _SLOTS = (
        "minds_api_key", "anthropic_api_key", "openai_api_key",
        "gemini_api_key", "openai_compatible_api_key",
    )

    def _clear(self, svc):
        for k in self._SLOTS:
            try:
                svc.delete_setting(k)
            except Exception:
                pass

    def test_gemini_only_is_configured(self):
        from cowork.api.v1.endpoints.settings import check_configured
        from cowork.db.session import get_open_session
        from cowork.services.settings import SettingService

        session = get_open_session()
        svc = SettingService(session)
        self._clear(svc)
        try:
            svc.upsert_setting("gemini_api_key", "AIza-only")
            res = check_configured(session)
            assert res["configured"] is True
            assert res["provider"] == "gemini"
        finally:
            self._clear(svc)

    def test_openai_compatible_only_is_configured(self):
        from cowork.api.v1.endpoints.settings import check_configured
        from cowork.db.session import get_open_session
        from cowork.services.settings import SettingService

        session = get_open_session()
        svc = SettingService(session)
        self._clear(svc)
        try:
            svc.upsert_setting("openai_compatible_api_key", "sk-compat-only")
            res = check_configured(session)
            assert res["configured"] is True
            assert res["provider"] == "openai-compatible"
        finally:
            self._clear(svc)
