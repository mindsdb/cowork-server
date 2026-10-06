"""Shared test doubles (importing conftest directly double-loads it)."""

import asyncio


class FakeRequest:
    """Minimal Request stub for calling route handlers directly. `headers` is
    empty unless a test sets it (only the org-mode bearer path reads it)."""

    def __init__(self, headers: dict | None = None) -> None:
        self.headers = headers or {}


class PausedHarness:
    """get_harness() stand-in whose answer waits on ``release`` mid-stream.

    ``answering`` is set once a turn has written its first frame and is
    waiting on the model; ``started`` counts the turns that reached the model.
    """

    id = "stub"

    def __init__(self) -> None:
        self.answering = asyncio.Event()
        self.release = asyncio.Event()
        self.started = 0

    def stream_response(self, **kwargs):
        self.started += 1
        return None

    async def formatter(self, stream, model, event_sink):
        from cowork.streaming import sse_frame

        yield sse_frame("response.created", {"type": "response.created"})
        self.answering.set()
        await self.release.wait()
        delta = {"type": "response.output_text.delta", "delta": "ok"}
        event_sink(delta["type"], delta)
        yield sse_frame(delta["type"], delta)
        yield sse_frame("response.completed", {"type": "response.completed", "response": {"output": []}})


def inline_run_db(session):
    """A run_db stand-in for producer tests whose services are fakes: it runs
    each unit on the event loop with ``session``, so nothing reaches a
    database."""

    async def run_db(fn, *, scope):
        return fn(session)

    return run_db


class PausedModel:
    """Stands in for anton's ChatSession under the real AntonHarness, set with
    ``monkeypatch.setattr(harness, "build_chat_session", model.build)``.

    Each turn waits on ``release``, then answers "ok". ``answering`` is set
    while a turn waits, and ``turns`` counts the turns that reached it.
    ``before_answer``, an async callable given the session, runs first, so a
    test can call one of the turn's tools or leave a compaction or a touched
    artifact for the harness's end-of-turn writes.
    """

    def __init__(self, *, before_answer=None) -> None:
        self.answering = asyncio.Event()
        self.release = asyncio.Event()
        self.turns = 0
        self.before_answer = before_answer

    def build(self, config):
        return _PausedChatSession(self, config)


class _PausedChatSession:
    def __init__(self, model: PausedModel, config) -> None:
        self._model = model
        self.config = config
        self.history = list(config.initial_history or [])
        self.artifacts_touched: set[str] = set()
        self.last_compaction = None

    async def turn_stream(self, user_input, **_kwargs):
        from anton.core.llm.provider import StreamTextDelta

        self._model.turns += 1
        if self._model.before_answer is not None:
            await self._model.before_answer(self)
        self._model.answering.set()
        await self._model.release.wait()
        self.history.append({"role": "assistant", "content": "ok"})
        yield StreamTextDelta(text="ok")
