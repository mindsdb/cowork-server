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
