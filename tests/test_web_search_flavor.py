"""Web-tool routing for clients built by ``build_llm_client``.

The flavor an ``OpenAIProvider`` is constructed with decides whether anton
passes web_search / web_fetch as a native provider capability or registers its
own handler-dispatched fallback (which needs an Exa/Brave key Cowork never asks
for). It is not a web-tools-only knob: ``FLAVOR_OPENAI`` also switches the
transport to the Responses API, so only MindsHub — whose passthrough serves
native web tools over chat.completions — opts in here by default. An
openai_compatible deployment that sets COWORK_OPENAI_COMPATIBLE_API=responses
moves its planning and coding roles onto ``FLAVOR_OPENAI``.
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

    def test_byok_openai_stays_on_chat_completions(self, monkeypatch):
        # Direct OpenAI keeps the generic flavor: the flavor that would enable
        # its native web tools also moves the transport to the Responses API,
        # which loses truncation reporting, tool_result images and trace
        # headers. Native search here waits on those gaps closing in anton.
        settings = UserSettings(
            planning_provider=Provider.OPENAI,
            coding_provider=Provider.OPENAI,
            openai_api_key=SecretStr("sk-openai"),
        )
        _patch_settings(monkeypatch, settings)

        client = providers.build_llm_client()

        assert client.planning_provider.native_web_tools() == set()

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

    def test_openai_compatible_on_the_responses_switch_reports_native_web_tools(
        self, monkeypatch, openai_compatible_api, anton_responses_ready
    ):
        # The real anton provider on the Responses flavor reports both web
        # tools as native, which is why the harness turns them off on this
        # path (test_switched_path_web_tools.py). The router stays generic.
        openai_compatible_api("responses")
        anton_responses_ready()
        settings = UserSettings(
            planning_provider=Provider.OPENAI_COMPATIBLE,
            coding_provider=Provider.OPENAI_COMPATIBLE,
            router_provider=Provider.OPENAI_COMPATIBLE,
            planning_model="gpt-5.6-sol",
            coding_model="gpt-5.6-luna",
            router_model="gpt-5.6-luna",
            openai_compatible_api_key=SecretStr("sk-proxy"),
            openai_base_url="https://my-proxy.internal/v1",
        )
        _patch_settings(monkeypatch, settings)

        client = providers.build_llm_client()

        assert client.planning_provider.native_web_tools() == WEB_TOOLS
        assert client.coding_provider.native_web_tools() == WEB_TOOLS
        assert client.router_provider.native_web_tools() == set()
