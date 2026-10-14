"""MODEL-2: a composer selection must reach actual provider requests."""

import asyncio
from types import SimpleNamespace
import pytest
from anton.config.settings import AntonSettings
from anton.core.llm.provider import LLMResponse, StreamComplete
from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.harnesses.anton_harness import harness


def test_runtime_context_uses_actual_client_models():
    a = AntonSettings(_env_file=None)
    client = SimpleNamespace(
        planning_model="picked-model", coding_model="coding-model", router_model="router-model"
    )
    applied = harness._apply_client_models(a, client)

    assert a.planning_model == "picked-model"
    assert a.coding_model == "coding-model"
    assert a.router_model == "router-model"
    assert set(applied) == {"planning_model", "coding_model", "router_model"}


def test_older_client_without_router_keeps_existing_router_setting():
    a = AntonSettings(_env_file=None)
    before = a.router_model
    applied = harness._apply_client_models(
        a, SimpleNamespace(planning_model="picked", coding_model="picked")
    )
    assert applied == ["planning_model", "coding_model"]
    assert a.router_model == before


def test_skew_guard_skips_fields_the_pinned_anton_does_not_have():
    """Version skew must degrade, never crash — anton is a git dep pinned to
    branch="main" (see _overlay_user_settings's docstring for why); pydantic
    raises ValueError on setattr of an unknown field."""
    from pydantic_settings import BaseSettings

    class PinnedAnton(BaseSettings):
        model_config = {"env_prefix": "ANTON_", "extra": "ignore"}
        planning_model: str = "default-planning"

    old = PinnedAnton()
    applied = harness._apply_client_models(
        old, SimpleNamespace(planning_model="picked-model", coding_model="picked-model")
    )

    assert applied == ["planning_model"]
    assert old.planning_model == "picked-model"


@pytest.mark.asyncio
async def test_session_model_selection_reaches_provider_and_stays_per_turn(monkeypatch):
    """Real session configuration + real LLMClient, with only external calls stubbed.

    Rebuild one conversation after a model switch, and issue requests from
    old/new/default sessions together to catch shared-settings contamination.
    """
    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.services.conversations import ConversationService

    saved = UserSettings(
        _env_file=None,
        planning_provider=Provider.OPENAI,
        coding_provider=Provider.OPENAI,
        router_provider=Provider.OPENAI,
        planning_model="saved-planning", coding_model="saved-coding", router_model="saved-router",
        openai_api_key=SecretStr("test-only-key"),
        episodic_memory=False,
    )
    before = saved.model_dump()
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: saved)
    # Capture the real config without starting scratchpad processes or reading connectors.
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", None)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")
    requests = []

    async def complete(_self, **kwargs):
        requests.append(kwargs["model"])
        await asyncio.sleep(0)
        return LLMResponse(content="ok")

    async def stream(_self, **kwargs):
        yield StreamComplete(await complete(_self, **kwargs))

    monkeypatch.setattr("anton.core.llm.openai.OpenAIProvider.complete", complete)
    monkeypatch.setattr("anton.core.llm.openai.OpenAIProvider.stream", stream)

    configs = []
    with get_open_session() as db:
        service = ConversationService(ScopedSession(db, LOCAL_SCOPE))
        conversation = service.create_conversation(topic="MODEL-2")
        other = service.create_conversation(topic="MODEL-2 defaults")
        try:
            for conv, model in ((conversation, "pick-A"), (conversation, "pick-B"), (other, None)):
                config, _, _ = await harness.AntonHarness()._build_chat_session(conv, model=model)
                configs.append(config)
                for role in ("planning", "coding", "router"):
                    assert getattr(config.settings, f"{role}_model") == (
                        model or f"saved-{role}"
                    )

            async def request_all_roles(config):
                client = config.llm_client
                # Desktop responses stream; code and compaction use complete().
                events = [event async for event in client.plan_stream(system="", messages=[])]
                assert len(events) == 1
                await client.code(system="", messages=[])
                await client.summarize(system="", messages=[])

            await asyncio.gather(*(request_all_roles(config) for config in configs))
            assert sorted(requests) == sorted(
                ["pick-A"] * 3 + ["pick-B"] * 3 + ["saved-planning", "saved-coding", "saved-router"]
            )
            assert saved.model_dump() == before
        finally:
            for config in configs:
                # The locked Anton predates LLMClient.aclose(). Close the SDK
                # clients directly so this test also runs with that version.
                client = config.llm_client
                for provider in (client.planning_provider, client.coding_provider, client.router_provider):
                    await provider._client.close()
