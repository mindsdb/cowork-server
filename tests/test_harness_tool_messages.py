"""`tool_messages` is opt-in: only the cowork UI renders a tool's message to
the user. Channel bots and the non-streaming API get only the answer text."""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import cowork.services.artifact_autopublish as autopublish
import cowork.services.task_objects as task_objects
from cowork.handlers.responses import ResponsesHandler
from cowork.harnesses.anton_harness import harness


class _FakeSession:
    async def turn_stream(self, user_input, *, turn_id=None):
        return
        yield  # noqa: marks this an async generator


def test_stream_response_forwards_the_flag_and_defaults_it_off(monkeypatch):
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_a, **_k: (set(), {}))
    monkeypatch.setattr(task_objects, "index_turn_artifacts", lambda *_a, **_k: ([], set(), None))
    monkeypatch.setattr(task_objects, "cards_for_slugs", lambda *_a, **_k: [])

    async def _no_autopublish(*_a, **_k):
        return set()

    monkeypatch.setattr(autopublish, "autopublish_project_artifacts", _no_autopublish)
    received = []

    async def _fake_build(self, conversation, **kwargs):
        received.append(kwargs["tool_messages"])
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
    asyncio.run(_drain(tool_messages=True))
    assert received == [False, True]


def test_only_the_ui_turn_opts_in():
    assert "tool_messages=True" in inspect.getsource(ResponsesHandler._run_turn)
    assert "tool_messages" not in inspect.getsource(ResponsesHandler.handle)


def test_the_session_builder_degrades_on_an_older_anton():
    source = inspect.getsource(harness.AntonHarness._build_chat_session)
    assert "**supported_kwargs(ChatSessionConfig, tool_messages=tool_messages)" in source
