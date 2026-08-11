"""Regression guard for ENG-1359: a cowork-server-built LLM client must route
web_search/web_fetch natively on MindsHub and BYOK OpenAI.

Before the fix, ``build_llm_client`` constructed every ``OpenAIProvider``
without a ``flavor``, so it defaulted to ``FLAVOR_OPENAI_COMPATIBLE_GENERIC``
and ``native_web_tools()`` returned an empty set — anton then registered a
fallback ``web_search`` tool with no Exa/Brave credential behind it, which
always fails.
"""

from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
import cowork.services.providers as providers


def _patch_settings(monkeypatch, settings: UserSettings):
    monkeypatch.setattr(
        "cowork.common.settings.user_settings.get_user_settings",
        lambda *a, **k: settings,
    )


class TestWebSearchFlavorRouting:
    def test_minds_cloud_routes_web_search_natively(self, monkeypatch):
        settings = UserSettings(
            planning_provider=Provider.MINDS_CLOUD,
            coding_provider=Provider.MINDS_CLOUD,
            minds_api_key=SecretStr("mdb-key"),
        )
        _patch_settings(monkeypatch, settings)

        client = providers.build_llm_client()

        assert client.planning_provider.native_web_tools() == {
            "web_search",
            "web_fetch",
        }
        assert client.coding_provider.native_web_tools() == {
            "web_search",
            "web_fetch",
        }

    def test_byok_openai_routes_web_search_natively(self, monkeypatch):
        settings = UserSettings(
            planning_provider=Provider.OPENAI,
            coding_provider=Provider.OPENAI,
            openai_api_key=SecretStr("sk-openai"),
        )
        _patch_settings(monkeypatch, settings)

        client = providers.build_llm_client()

        assert client.planning_provider.native_web_tools() == {
            "web_search",
            "web_fetch",
        }

    def test_openai_compatible_third_party_stays_generic(self, monkeypatch):
        # A non-Minds openai-compatible endpoint has no native web search —
        # must NOT be upgraded to a flavor it doesn't support.
        settings = UserSettings(
            planning_provider=Provider.OPENAI_COMPATIBLE,
            coding_provider=Provider.OPENAI_COMPATIBLE,
            openai_compatible_api_key=SecretStr("sk-proxy"),
            openai_base_url="https://my-proxy.internal/v1",
        )
        _patch_settings(monkeypatch, settings)

        client = providers.build_llm_client()

        assert client.planning_provider.native_web_tools() == set()
