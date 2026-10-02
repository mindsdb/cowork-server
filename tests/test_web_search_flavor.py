"""Web-tool routing for clients built by ``build_llm_client``.

The flavor an ``OpenAIProvider`` is constructed with decides whether anton
passes web_search / web_fetch as a native provider capability or registers its
own handler-dispatched fallback (which needs an Exa/Brave key Cowork never asks
for). It is not a web-tools-only knob: ``FLAVOR_OPENAI`` also switches the
transport to the Responses API. MindsHub's passthrough serves native web tools
over chat.completions; direct OpenAI gets them on the Responses API once the
installed anton's Responses transport is ready.
"""

from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
import cowork.services.providers as providers

WEB_TOOLS = {"web_search", "web_fetch"}


def _patch_settings(monkeypatch, settings: UserSettings):
    monkeypatch.setattr(
        "cowork.common.settings.user_settings.get_user_settings",
        lambda *a, **k: settings,
    )


class TestWebSearchFlavorRouting:
    def test_minds_cloud_routes_web_tools_natively(self, monkeypatch):
        settings = UserSettings(
            planning_provider=Provider.MINDS_CLOUD,
            coding_provider=Provider.MINDS_CLOUD,
            minds_api_key=SecretStr("mdb-key"),
        )
        _patch_settings(monkeypatch, settings)

        client = providers.build_llm_client()

        assert client.planning_provider.native_web_tools() == WEB_TOOLS
        assert client.coding_provider.native_web_tools() == WEB_TOOLS

    def test_minds_cloud_native_on_a_self_hosted_gateway(self, monkeypatch):
        # The flavor is stated by the branch, not sniffed from the host, so a
        # gateway whose URL doesn't spell "mindshub.ai" still gets native web
        # tools instead of silently losing search.
        settings = UserSettings(
            planning_provider=Provider.MINDS_CLOUD,
            coding_provider=Provider.MINDS_CLOUD,
            minds_api_key=SecretStr("mdb-key"),
            minds_url="https://staging-gateway.internal",
        )
        _patch_settings(monkeypatch, settings)

        client = providers.build_llm_client()

        assert client.planning_provider.native_web_tools() == WEB_TOOLS

    def test_byok_openai_routes_web_tools_natively_on_responses(self, monkeypatch):
        # Direct OpenAI runs on the Responses API (FLAVOR_OPENAI), which
        # carries OpenAI's native web search.
        settings = UserSettings(
            planning_provider=Provider.OPENAI,
            coding_provider=Provider.OPENAI,
            openai_api_key=SecretStr("sk-openai"),
        )
        _patch_settings(monkeypatch, settings)

        client = providers.build_llm_client()

        # An anton without the Responses marker keeps chat.completions, where
        # direct OpenAI has no native web tools.
        from anton.core.llm.openai import OpenAIProvider

        ready = getattr(OpenAIProvider, "RESPONSES_TRANSPORT_READY", False) is True
        expected = WEB_TOOLS if ready else set()
        assert client.planning_provider.native_web_tools() == expected
        assert client.coding_provider.native_web_tools() == expected

    def test_openai_compatible_third_party_is_generic(self, monkeypatch):
        # A third-party openai-compatible endpoint has no native web search and
        # must not be upgraded to a flavor it doesn't implement.
        settings = UserSettings(
            planning_provider=Provider.OPENAI_COMPATIBLE,
            coding_provider=Provider.OPENAI_COMPATIBLE,
            openai_compatible_api_key=SecretStr("sk-proxy"),
            openai_base_url="https://my-proxy.internal/v1",
        )
        _patch_settings(monkeypatch, settings)

        client = providers.build_llm_client()

        assert client.planning_provider.native_web_tools() == set()
