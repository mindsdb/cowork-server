"""The eval trace-tags pass-through (PR #125).

Asserts the AntonHarness forwards `trace_tags` / `trace_metadata` into
`ChatSession.turn_stream` when the installed anton supports them, tolerates an
older anton whose `turn_stream` lacks the kwargs (no TypeError — the blocker
the deployed PyPI/main anton would otherwise hit), and that Hermes accepts the
kwargs so the handler can forward them uniformly.
"""

import asyncio
import inspect
from types import SimpleNamespace

import cowork.services.task_objects as task_objects
from cowork.harnesses.anton_harness.harness import AntonHarness


class _SessionWithTraceKwargs:
    """Stand-in for an anton ChatSession that supports the trace kwargs."""

    def __init__(self):
        self.received = {}

    async def turn_stream(self, user_input, *, turn_id=None,
                          trace_tags=None, trace_metadata=None):
        self.received = {"trace_tags": trace_tags, "trace_metadata": trace_metadata}
        return
        yield  # noqa: marks this an async generator


class _SessionWithoutTraceKwargs:
    """An older anton ChatSession predating the trace kwargs."""

    async def turn_stream(self, user_input, *, turn_id=None):
        return
        yield  # noqa: marks this an async generator


def _run_harness(monkeypatch, session):
    monkeypatch.setattr(task_objects, "snapshot_artifact_slugs", lambda *_a, **_k: set())
    monkeypatch.setattr(task_objects, "finalize_turn_artifacts", lambda *_a, **_k: [])

    async def _fake_build(self, conversation, disabled_connections, channel_context=None):
        return session, None, None

    monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)

    conversation = SimpleNamespace(
        id="conv-1", project_id="proj-1", project=SimpleNamespace(path="/tmp")
    )

    async def _drain():
        return [
            event
            async for event in AntonHarness().stream_response(
                conversation=conversation,
                input=[{"type": "text", "text": "hi"}],
                trace_tags=["eval", "eval_run:r1"],
                trace_metadata={"eval_run_id": "r1"},
            )
        ]

    return asyncio.run(_drain())


def test_forwards_trace_kwargs_when_anton_supports_them(monkeypatch):
    session = _SessionWithTraceKwargs()
    _run_harness(monkeypatch, session)
    assert session.received == {
        "trace_tags": ["eval", "eval_run:r1"],
        "trace_metadata": {"eval_run_id": "r1"},
    }


def test_tolerates_anton_without_trace_kwargs(monkeypatch):
    # Must not raise TypeError: unexpected keyword argument 'trace_tags'.
    _run_harness(monkeypatch, _SessionWithoutTraceKwargs())


def test_hermes_stream_response_accepts_trace_kwargs():
    from cowork.harnesses.hermes_harness.harness import HermesHarness

    params = inspect.signature(HermesHarness.stream_response).parameters
    assert "trace_tags" in params and "trace_metadata" in params
