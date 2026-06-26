"""Unit tests for provider_base_url — the single source of truth for each
provider's inference base URL, shared by build_llm_client._make_provider
(main agent) and scratchpad _resolve_coding.

Core invariant: the shared openai_base_url slot is owned ONLY by
openai-compatible. openai and gemini never inherit it, so a stale value left
by a prior provider setup can't misroute another provider's API key.
"""

from cowork.services.providers import GEMINI_BASE_URL, provider_base_url

CONTAMINATED = "https://api.mindshub.ai/v1"  # e.g. left behind by a MindsHub setup


class TestProviderBaseUrl:
    def test_anthropic_uses_sdk_default(self):
        assert provider_base_url("anthropic", openai_base_url=CONTAMINATED) is None

    def test_openai_never_inherits_shared_slot(self):
        # Even with a contaminated slot, direct OpenAI uses the SDK default.
        assert provider_base_url("openai", openai_base_url=CONTAMINATED) is None

    def test_gemini_always_targets_google(self):
        # Ignores the shared slot entirely — always Google's endpoint.
        assert provider_base_url("gemini", openai_base_url=CONTAMINATED) == GEMINI_BASE_URL
        assert "googleapis.com" in GEMINI_BASE_URL

    def test_openai_compatible_owns_the_slot(self):
        assert (
            provider_base_url("openai-compatible", openai_base_url="https://proxy/v1")
            == "https://proxy/v1"
        )

    def test_openai_compatible_falls_back_to_openai_when_empty(self):
        assert (
            provider_base_url("openai-compatible", openai_base_url="")
            == "https://api.openai.com/v1"
        )

    def test_minds_cloud_derives_from_minds_slot(self):
        assert (
            provider_base_url("minds-cloud", minds_url="https://api.mindshub.ai")
            == "https://api.mindshub.ai/v1"
        )
        assert (
            provider_base_url("minds-cloud", minds_url="https://mdb.ai")
            == "https://mdb.ai/api/v1"
        )

    def test_underscore_enum_values_normalized(self):
        # Provider enum values use underscores (e.g. "minds_cloud").
        assert provider_base_url("minds_cloud", minds_url="https://api.mindshub.ai") == "https://api.mindshub.ai/v1"
        assert provider_base_url("openai_compatible", openai_base_url="https://p/v1") == "https://p/v1"
