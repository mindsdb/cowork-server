"""Hosted web search stays off on the switched openai_compatible path.

On anton's Responses flavor, either ChatSessionConfig web flag becomes OpenAI's
hosted web_search tool. That tool reads the web from the provider's side,
outside the deployment's egress gateway (on Azure it runs on Bing). So when
COWORK_OPENAI_COMPATIBLE_API puts an openai_compatible planning provider on
that path, both sessions built on the planning role, a chat turn's and a
connector credential probe's, turn both flags off. Every other session keeps
anton's defaults: MindsHub keeps its native web tools, and direct OpenAI keeps
what its flavor provides.
"""
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.harnesses.anton_harness import harness

# A local address, so a request that escapes the stubs below is refused at
# once rather than sent anywhere.
_BASE_URL = "http://127.0.0.1:9/v1"

_SETTINGS = {
    Provider.OPENAI_COMPATIBLE: dict(
        openai_compatible_api_key=SecretStr("sk-compat"),
        openai_base_url=_BASE_URL,
        planning_model="gpt-5.6-sol",
        coding_model="gpt-5.6-luna",
        router_model="gpt-5.6-luna",
    ),
    Provider.OPENAI: dict(openai_api_key=SecretStr("sk-openai")),
    Provider.MINDS_CLOUD: dict(
        minds_api_key=SecretStr("mdb-key"), minds_url="https://api.mindshub.ai"
    ),
}

_TOOL = {
    "name": "scratchpad",
    "description": "Run Python.",
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}},
}


def _settings(provider: Provider) -> UserSettings:
    return UserSettings(
        _env_file=None,
        planning_provider=provider,
        coding_provider=provider,
        router_provider=provider,
        episodic_memory=False,
        **_SETTINGS[provider],
    )


async def _session_config(monkeypatch, settings: UserSettings):
    """One turn's ChatSessionConfig, built by the real harness."""
    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.services.conversations import ConversationService

    monkeypatch.setattr(
        "cowork.common.settings.user_settings.get_user_settings", lambda: settings
    )
    # Capture the config rather than a session: no scratchpad, no connectors.
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", None)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")
    with get_open_session() as db:
        conversation = ConversationService(
            ScopedSession(db, LOCAL_SCOPE)
        ).create_conversation(topic="web tools")
        config, _, _ = await harness.AntonHarness()._build_chat_session(conversation)
    return config


@pytest.mark.asyncio
async def test_switched_planning_request_carries_no_hosted_web_search(
    monkeypatch, openai_compatible_api, anton_responses_ready
):
    from cowork.common.chat_session import build_chat_session

    openai_compatible_api("responses")
    anton_responses_ready()
    requests = []

    async def _completed():
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(usage=None, status="completed", model="gpt-5.6-sol"),
        )

    async def responses_create(_self, **kwargs):
        requests.append(kwargs)
        return _completed()

    async def chat_create(_self, **kwargs):
        raise AssertionError("the planning call went to chat completions")

    monkeypatch.setattr("openai.resources.responses.AsyncResponses.create", responses_create)
    monkeypatch.setattr(
        "openai.resources.chat.completions.AsyncCompletions.create", chat_create
    )

    config = await _session_config(monkeypatch, _settings(Provider.OPENAI_COMPATIBLE))
    session = build_chat_session(config)
    try:
        # The call every agent-loop step makes; the session adds its native
        # web tools to it.
        async for _event in session.plan_stream_with_recovery(
            system="You are Cowork.",
            tools=[_TOOL],
            messages_factory=lambda: [
                {"role": "user", "content": "Search the web for today's top headline."}
            ],
        ):
            pass
    finally:
        await session.close()

    (request,) = requests
    assert request["model"] == "gpt-5.6-sol"
    assert {"type": "web_search"} not in request["tools"]
    assert [tool.get("name") for tool in request["tools"]] == ["scratchpad"]
    assert (config.web_search_enabled, config.web_fetch_enabled) == (False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "api", "ready"),
    [
        # The switch at its default.
        (Provider.OPENAI_COMPATIBLE, None, True),
        # Asked for, but the installed anton's Responses path isn't ready, so
        # planning stays on chat completions and keeps its fallback web tools.
        (Provider.OPENAI_COMPATIBLE, "responses", False),
        # The switch names openai_compatible only.
        (Provider.OPENAI, "responses", True),
        (Provider.MINDS_CLOUD, "responses", True),
    ],
)
async def test_web_tools_keep_their_defaults_off_the_switched_path(
    monkeypatch, openai_compatible_api, anton_responses_ready, provider, api, ready
):
    openai_compatible_api(api)
    anton_responses_ready(ready)

    config = await _session_config(monkeypatch, _settings(provider))
    await config.llm_client.aclose()

    assert (config.web_search_enabled, config.web_fetch_enabled) == (True, True)


async def _probe_session_config(monkeypatch, settings: UserSettings):
    """The ChatSessionConfig a connector credential probe builds."""
    from cowork.services.connectors import probe as probe_module

    monkeypatch.setattr(
        "cowork.common.settings.user_settings.get_user_settings", lambda *a, **k: settings
    )
    configs = []

    def _capture(config):
        configs.append(config)
        raise RuntimeError("config captured")

    monkeypatch.setattr(probe_module, "build_chat_session", _capture)
    probe = probe_module.CredentialProbe(
        engine="postgres", credentials={"password": "hunter2"}, llm_client=None, workspace=None
    )
    [event async for event in probe.run()]
    (config,) = configs
    return config


@pytest.mark.asyncio
async def test_switched_probe_session_runs_without_web_tools(
    monkeypatch, openai_compatible_api, anton_responses_ready
):
    # A connector's credential probe runs a turn on the same planning role.
    openai_compatible_api("responses")
    anton_responses_ready()

    config = await _probe_session_config(monkeypatch, _settings(Provider.OPENAI_COMPATIBLE))

    assert (config.web_search_enabled, config.web_fetch_enabled) == (False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "api"),
    [(Provider.OPENAI_COMPATIBLE, None), (Provider.OPENAI, "responses")],
)
async def test_probe_session_keeps_web_tool_defaults_off_the_switched_path(
    monkeypatch, openai_compatible_api, anton_responses_ready, provider, api
):
    openai_compatible_api(api)
    anton_responses_ready()

    config = await _probe_session_config(monkeypatch, _settings(provider))

    assert (config.web_search_enabled, config.web_fetch_enabled) == (True, True)
