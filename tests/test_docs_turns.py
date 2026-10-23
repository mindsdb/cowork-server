"""Which turns get the doc tools: only those whose client shows docs.

The tools carry the docs section of the system prompt. Registered for an app
whose renderer predates docs, they had the agent write docs that app has no
way to show, so a request turns them on with ``docs: true``, and a request
sent with a doc open or asking for a doc implies it.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

import cowork.handlers.responses as responses_mod
from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.docs.prompt import DOC_PROMPT
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
from cowork.harnesses.anton_harness import harness
from cowork.harnesses.anton_harness.doc_tools import DOC_TOOL_NAMES
from cowork.server import create_app
from cowork.services.conversations import ConversationService
from cowork.streaming import registry

from _fakes import PausedHarness

DOC_CONTEXT = {"artifactId": "a" * 32, "title": "Plan", "mode": "edit"}


async def _chat_session_config(monkeypatch, **kwargs):
    """One chat turn's ChatSessionConfig, built by the real harness (see test_shared_verifier_latch)."""
    settings = UserSettings(
        _env_file=None,
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        router_provider=Provider.MINDS_CLOUD,
        episodic_memory=False,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
    )
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: settings)
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", None)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")
    with get_open_session() as db:
        conversation = ConversationService(ScopedSession(db, LOCAL_SCOPE)).create_conversation(topic="docs")
        return (await harness.AntonHarness()._build_chat_session(conversation, **kwargs)).session


@pytest.mark.asyncio
@pytest.mark.parametrize("docs", [False, True])
async def test_the_doc_tools_and_their_prompt_are_there_only_when_the_client_shows_docs(monkeypatch, docs):
    config = await _chat_session_config(monkeypatch, **({"docs": True} if docs else {}))
    names = {tool.name for tool in config.tools}
    prompts = "".join(tool.prompt or "" for tool in config.tools)
    assert (set(DOC_TOOL_NAMES) <= names) is docs
    assert names.isdisjoint(DOC_TOOL_NAMES) is not docs
    assert (DOC_PROMPT in prompts) is docs


@pytest.fixture(autouse=True)
def _forget_turns():
    yield
    registry.reset()


class _SeesDocs(PausedHarness):
    """Records whether each turn was told the client shows docs, and answers at once."""

    def __init__(self) -> None:
        super().__init__()
        self.release.set()
        self.docs: list[bool] = []

    def stream_response(self, **kwargs):
        self.docs.append(kwargs.get("docs"))
        return super().stream_response(**kwargs)


async def _ask(monkeypatch, stub: _SeesDocs, *, stream: bool, **body) -> httpx.Response:
    async def decide(**_kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    async def collected(self, stream, conversation_id, model, original_content):
        return {"ok": True}

    monkeypatch.setattr(responses_mod, "decide_route", decide)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: stub)
    monkeypatch.setattr(responses_mod.ResponsesHandler, "_collect", collected)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await asyncio.wait_for(
            client.post(
                "/api/v1/responses/",
                json={"input": "hi", "stream": stream, "conversation": str(uuid4()), **body},
            ),
            timeout=10,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("body,shows_docs", [
    ({}, False),
    ({"docs": False}, False),
    ({"docs": True}, True),
    ({"output": "doc"}, True),
    ({"doc_context": DOC_CONTEXT}, True),
])
async def test_the_request_tells_the_turn_whether_its_client_shows_docs(monkeypatch, stream, body, shows_docs):
    stub = _SeesDocs()

    answered = await _ask(monkeypatch, stub, stream=stream, **body)

    assert answered.status_code == 200, answered.text
    assert stub.docs == [shows_docs]
