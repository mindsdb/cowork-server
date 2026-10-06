from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import cowork.services.artifact_autopublish as autopublish
import cowork.services.task_objects as task_objects
from cowork.handlers.responses import ResponsesHandler
from cowork.harnesses.anton_harness import harness
from cowork.harnesses.anton_harness.harness import _tool_message_kwargs
from cowork.harnesses.base import ChannelContext


@dataclass
class _Config:
    tool_messages: bool = False


@dataclass
class _OldConfig:
    other: int = 0


def test_a_ui_turn_renders_tool_messages():
    assert _tool_message_kwargs(_Config, None) == {"tool_messages": True}


def test_a_channel_turn_does_not():
    ctx = ChannelContext(channel_type="slack")
    assert _tool_message_kwargs(_Config, ctx) == {"tool_messages": False}


def test_an_anton_without_the_field_gets_nothing():
    assert _tool_message_kwargs(_OldConfig, None) == {}


def test_a_text_only_caller_does_not():
    assert _tool_message_kwargs(_Config, None, False) == {"tool_messages": False}


def test_the_session_builder_passes_it():
    source = inspect.getsource(harness.AntonHarness._build_chat_session)
    assert "**_tool_message_kwargs(ChatSessionConfig, channel_context, renders_tool_messages)" in source


def test_the_non_streaming_api_declares_no_tool_messages():
    """It returns only the collected answer text, so a tool's message would
    reach nobody while the agent is told the user already saw it."""
    source = inspect.getsource(ResponsesHandler.handle)
    assert "renders_tool_messages=False" in source


class _FakeSession:
    async def turn_stream(self, user_input, *, turn_id=None):
        return
        yield  # noqa: marks this an async generator


def test_stream_response_forwards_the_flag(monkeypatch):
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_a, **_k: (set(), {}))
    monkeypatch.setattr(task_objects, "index_turn_artifacts", lambda *_a, **_k: ([], set(), None))
    monkeypatch.setattr(task_objects, "cards_for_slugs", lambda *_a, **_k: [])

    async def _no_autopublish(*_a, **_k):
        return set()

    monkeypatch.setattr(autopublish, "autopublish_project_artifacts", _no_autopublish)
    received = []

    async def _fake_build(self, conversation, **kwargs):
        received.append(kwargs["renders_tool_messages"])
        return _FakeSession(), None, None

    monkeypatch.setattr(harness.AntonHarness, "_build_chat_session", _fake_build)
    conversation = SimpleNamespace(
        id="conv-1", project_id="proj-1", project=SimpleNamespace(path="/tmp", name="tmp")
    )

    async def _drain(**extra):
        return [
            event
            async for event in harness.AntonHarness().stream_response(
                conversation=conversation, input=[{"type": "text", "text": "hi"}], **extra,
            )
        ]

    asyncio.run(_drain())
    asyncio.run(_drain(renders_tool_messages=False))
    assert received == [True, False]
