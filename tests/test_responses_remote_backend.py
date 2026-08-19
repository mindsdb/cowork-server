"""Streaming producer selection: COWORK_TURN_BACKEND=remote routes the turn
through the Redis-backed remote producer instead of the in-process detached
run; unset/"inprocess" must stay byte-identical to today.

Built without __init__ (same pattern as tests/test_turn_errors.py) so no
DB/harness setup is needed - the DB-touching pieces (_remote_history and
the persistence layer inside _produce_remote) are stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

import cowork.handlers.responses as responses_mod
from cowork.handlers.responses import ResponsesHandler
from cowork.streaming.registry import TurnLifecycle


class _FakeScope:
    org_id = "org-123"
    user_id = "user-456"


class _FakeScoped:
    scope = _FakeScope()


class _FakeBuffer:
    async def append(self, *args, **kwargs):
        pass

    async def close(self, *args, **kwargs):
        pass


def _handler() -> ResponsesHandler:
    handler = object.__new__(ResponsesHandler)
    handler.scoped = _FakeScoped()
    return handler


def _kwargs(**overrides) -> dict:
    base = dict(
        conv_id=uuid4(),
        harness_input=[{"type": "text", "text": "hello there"}],
        original_content="hello there",
        model="anton",
        disabled=None,
        harness_name="anton",
        harness_id="anton",
        buffer=object(),
        trace_tags=None,
        trace_metadata=None,
    )
    base.update(overrides)
    return base


def test_remote_backend_selected(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")

    called = {}

    handler = _handler()

    def _boom(**kwargs):
        raise AssertionError("in-process _produce must not run when backend=remote")

    handler._produce = _boom

    def fake_produce_remote(**kwargs):
        called.update(kwargs)
        return "remote-sentinel"

    handler._produce_remote = fake_produce_remote

    kwargs = _kwargs()
    result = handler._select_producer(**kwargs)

    assert result == "remote-sentinel"
    assert called["conv_id"] == kwargs["conv_id"]
    assert called["input_text"] == "hello there"
    assert called["original_content"] == "hello there"
    assert called["model"] == "anton"
    assert called["harness_id"] == "anton"
    assert called["buffer"] is kwargs["buffer"]


def _remote_handler(monkeypatch, saved):
    """Handler wired for _produce_remote with the DB layer faked out."""
    handler = _handler()
    handler.principal = object()
    handler._remote_history = lambda session, conv_id: []

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return object()

        def save_user_message(self, conv_id, content, *, pending=False):
            saved["user"] = content
            saved["user_pending"] = pending
            msg = SimpleNamespace(id=uuid4())
            saved["user_id"] = msg.id
            return msg

        def finalize_pending(self, conv_id, message_id=None):
            saved["finalized"] = True
            saved["finalized_id"] = message_id

        def save_assistant_turn(self, conv_id, text, events, harness=None, tool_rows=None):
            saved["assistant"] = text
            saved["events"] = events
            saved["harness"] = harness

    class FakeSession:
        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "ScopedSession", lambda s, scope: FakeSession())
    monkeypatch.setattr(responses_mod, "get_open_session", lambda: None)
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    return handler


@pytest.mark.asyncio
async def test_produce_remote_persists_pending_user_then_finalizes_and_saves_assistant(monkeypatch):
    # ENG-1231: the producer persists the user message (pending) as its first
    # action — inside the coroutine, so a registry-deduped duplicate start whose
    # coroutine is discarded never writes a row — then on terminal finalizes the
    # flag (rejoining LLM history) and persists the assistant turn.
    saved = {}
    handler = _remote_handler(monkeypatch, saved)

    async def fake_replies(**kwargs):
        yield "turn_delta", {"text": "he"}
        yield "turn_delta", {"text": "llo"}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    await handler._produce_remote(
        conv_id=uuid4(), input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=_FakeBuffer(),
    )

    assert saved["user"] == "hi"
    assert saved["user_pending"] is True   # persisted in-flight at producer start
    assert saved["finalized"] is True      # flag cleared on terminal
    # Finalize is scoped to THIS turn's row, so a completing turn can't absorb a
    # pending row stranded by an earlier crashed turn into history.
    assert saved["finalized_id"] == saved["user_id"]
    assert saved["assistant"] == "hello"
    assert saved["harness"] == "anton"
    # Events are the formatter's sink payloads (richer dicts), ending with the
    # completed response carrying the collected text.
    completed = saved["events"][-1]
    assert completed["type"] == "response.completed"
    assert completed["response"]["output"][0]["content"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_produce_remote_persists_on_failure(monkeypatch):
    saved = {}
    handler = _remote_handler(monkeypatch, saved)

    async def fake_replies(**kwargs):
        yield "turn_delta", {"text": "partial"}
        # The real producer attaches the classified (code, message) it yields.
        yield "turn_failed", {"error": "RuntimeError: boom",
                              "code": "anton_error", "message": "An unexpected error occurred."}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    await handler._produce_remote(
        conv_id=uuid4(), input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=_FakeBuffer(),
    )

    assert saved["user"] == "hi"
    assert saved["user_pending"] is True
    assert saved["finalized"] is True
    assert saved["finalized_id"] == saved["user_id"]
    assert saved["assistant"] == "partial"
    # Persisted events mirror the streamed response.failed frame.
    assert saved["events"][-1] == {"type": "response.failed", "code": "anton_error",
                                   "error": "An unexpected error occurred."}


@pytest.mark.asyncio
async def test_produce_remote_pending_persist_failure_does_not_clear_all_pending(monkeypatch):
    # ENG-1231 hardening: if the pending user persist itself raises before its id
    # is captured, this turn owns no pending row — persist() must NOT fall back to
    # finalize_pending(conv, None), which would clear (and fold into history) a
    # pending row stranded by an earlier crashed turn.
    saved = {}
    handler = _remote_handler(monkeypatch, saved)

    def raising_save_user_message(self, conv_id, content, *, pending=False):
        raise RuntimeError("db down")

    monkeypatch.setattr(
        responses_mod.ConversationService, "save_user_message", raising_save_user_message,
    )

    async def fake_replies(**kwargs):
        raise AssertionError("turn must not run when the user persist failed")
        yield  # makes this an async generator, like the real producer

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    await handler._produce_remote(
        conv_id=uuid4(), input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=_FakeBuffer(),
    )

    # No row was persisted for this turn, so finalize must not have run at all —
    # in particular never the clear-all (message_id=None) form.
    assert "finalized" not in saved
    assert saved.get("finalized_id") is None


@pytest.mark.parametrize("backend_env", [None, "inprocess"])
def test_inprocess_backend_selected_by_default(monkeypatch, backend_env):
    if backend_env is None:
        monkeypatch.delenv("COWORK_TURN_BACKEND", raising=False)
    else:
        monkeypatch.setenv("COWORK_TURN_BACKEND", backend_env)

    called = {}

    def fake_replies(**kwargs):
        raise AssertionError("remote producer must not run for the inprocess backend")

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    handler = _handler()

    def fake_produce(**kwargs):
        called.update(kwargs)
        return "inprocess-sentinel"

    handler._produce = fake_produce

    kwargs = _kwargs()
    result = handler._select_producer(**kwargs)

    assert result == "inprocess-sentinel"
    # Verbatim pass-through - none of _produce's existing args are dropped
    # or reordered by the new selection branch. `lifecycle` rides along on top:
    # the selector supplies one when the caller did not, so a directly-called
    # producer still has a discard flag to read.
    assert isinstance(called.pop("lifecycle"), TurnLifecycle)
    assert called == kwargs


@pytest.mark.asyncio
async def test_discarded_remote_turn_persists_nothing(monkeypatch):
    """A turn delete cancels the producer (registry.discard) after truncating
    history — the remote path must drop the turn instead of writing rows into a
    conversation that no longer has room for them, and must not close (i.e.
    recreate) the buffer file the delete just removed."""
    import asyncio

    saved = {}
    handler = _remote_handler(monkeypatch, saved)

    class _ClosableBuffer:
        def __init__(self):
            self.closed = None

        async def append(self, *args, **kwargs):
            pass

        async def close(self, reason, extra=None):
            self.closed = reason

    started = asyncio.Event()

    async def fake_replies(**kwargs):
        started.set()
        await asyncio.sleep(3600)
        yield  # never reached; makes this an async generator

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    lifecycle = TurnLifecycle()
    buffer = _ClosableBuffer()
    task = asyncio.create_task(handler._produce_remote(
        conv_id=uuid4(), input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=buffer, lifecycle=lifecycle,
    ))
    await asyncio.wait_for(started.wait(), timeout=5)

    lifecycle.discarded = True
    task.cancel()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)

    assert "assistant" not in saved and "finalized" not in saved
    assert buffer.closed is None


@pytest.mark.asyncio
async def test_produce_remote_streams_desktop_step_vocabulary(monkeypatch):
    """Full remote pipeline: turn_step replies come out as the same
    response.in_progress thought frames the desktop formatter emits, the
    created frame carries the injected conversation_id/harness, rounds are
    separated by a paragraph break, and the steps land in the persisted
    events log for replay."""
    import json

    saved = {}
    handler = _remote_handler(monkeypatch, saved)
    handler._remote_memory = lambda session, conv_id: None

    async def fake_replies(**kwargs):
        yield "turn_step", {"step": "tool_start", "id": "t1", "name": "scratchpad"}
        yield "turn_step", {"step": "tool_end", "id": "t1",
                            "args": '{"name":"cell","one_line_description":"Adds","code":"1+1"}'}
        yield "turn_delta", {"text": "Preamble."}
        yield "turn_step", {"step": "round_end", "stop_reason": "end_turn",
                            "had_tool_calls": False}
        yield "turn_delta", {"text": "The answer."}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    frames = []

    class RecBuffer:
        async def append(self, kind, record):
            frames.append(record["sse"])

        async def close(self, reason):
            frames.append(f"CLOSE:{reason}")

    conv_id = uuid4()
    await handler._produce_remote(
        conv_id=conv_id, input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=RecBuffer(),
    )

    def payload(frame):
        return json.loads(frame.split("data: ", 1)[1])

    assert frames[0].startswith("event: response.created\n")
    created = payload(frames[0])
    assert created["conversation_id"] == str(conv_id)
    assert created["harness"] == "anton"

    roles = [payload(f)["thought_role"] for f in frames
             if f.startswith("event: response.in_progress")]
    assert "thought.scratchpad.start" in roles
    assert "thought.scratchpad.end" in roles

    deltas = [payload(f)["delta"] for f in frames
              if f.startswith("event: response.output_text.delta")]
    assert deltas[0] == "Preamble."
    assert deltas[1] == "\n\nThe answer."  # round break, not glued text

    assert frames[-1] == "CLOSE:completed"
    completed = payload(frames[-2])
    assert completed["response"]["output"][0]["content"][0]["text"] == "Preamble.\n\nThe answer."

    assert saved["assistant"] == "Preamble.\n\nThe answer."
    assert any(e.get("thought_role") == "thought.scratchpad.start" for e in saved["events"])


@pytest.mark.asyncio
async def test_produce_remote_stages_workspace_files(monkeypatch):
    """Wiring guard: the remote produce path must stage attachments +
    instructions into the workspace before the turn — removing that call
    (responses.py) has to fail here, not just leave the helper's own tests green."""
    saved = {}
    handler = _remote_handler(monkeypatch, saved)
    called = []
    monkeypatch.setattr(
        type(handler), "_stage_remote_workspace_files",
        staticmethod(lambda session, conv_id: called.append(conv_id)),
    )

    async def fake_replies(**kwargs):
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    await handler._produce_remote(
        conv_id=uuid4(), input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=_FakeBuffer(),
    )
    assert called, "produce_remote must stage workspace files before the turn"


def test_stage_remote_workspace_files_seeds_builtin_skills(monkeypatch):
    """The pod reads skills straight off the shared mount, not through a
    payload — GET /skills is not the only place a fresh org can get seeded, or
    an org that chats before it ever opens the skills menu stays empty
    forever on cloud turns (ENG-1679 review)."""
    calls = []

    class FakeSkillService:
        def __init__(self, scope):
            calls.append(scope)

        def ensure_builtin_skills(self):
            calls.append("seeded")

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return SimpleNamespace(project=SimpleNamespace(path="/tmp/proj"))

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "SkillService", FakeSkillService)
    monkeypatch.setattr(
        responses_mod, "FileService",
        lambda session: SimpleNamespace(stage_conversation_attachments=lambda *a, **k: None),
    )
    monkeypatch.setattr("cowork.services.files.stage_project_instructions", lambda *a, **k: None)

    scope = _FakeScope()
    ResponsesHandler._stage_remote_workspace_files(SimpleNamespace(scope=scope), uuid4())

    assert calls == [scope, "seeded"]


_DRAFT_MD = ("---\nname: competitive-analysis\ndescription: Compare rivals\n"
             "metadata:\n  display_name: Competitive Analysis\n---\n1. Gather\n2. Compare")


@pytest.mark.asyncio
async def test_produce_remote_surfaces_a_skill_draft_as_a_card(monkeypatch):
    """A draft the pod built comes out as the same response.skill_created card
    the desktop path emits, and lands in the persisted events log so a reload
    replays it. The skill itself is NOT saved — the card is the user's decision."""
    import json

    saved = {}
    handler = _remote_handler(monkeypatch, saved)
    handler._remote_memory = lambda session, conv_id: None

    async def fake_replies(**kwargs):
        yield "turn_delta", {"text": "Built it."}
        yield "turn_skill", {"entries": [{
            "slug": "competitive-analysis",
            "files": {"SKILL.md": _DRAFT_MD, "recipe.md": "detail"},
        }]}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    frames = []

    class RecBuffer:
        async def append(self, kind, record):
            frames.append(record["sse"])

        async def close(self, reason):
            frames.append(f"CLOSE:{reason}")

    await handler._produce_remote(
        conv_id=uuid4(), input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=RecBuffer(),
    )

    cards = [json.loads(f.split("data: ", 1)[1]) for f in frames
             if f.startswith("event: response.skill_created")]
    assert len(cards) == 1
    skill = cards[0]["skill"]
    assert skill["slug"] == "competitive-analysis"
    assert skill["name"] == "Competitive Analysis"
    assert "1. Gather" in skill["instructions"]
    assert skill["files"] == [{"name": "recipe.md", "text": "detail"}]

    # Replay on reload reads the events log, not the live stream.
    assert any(e.get("type") == "response.skill_created" for e in saved["events"])
    assert frames[-1] == "CLOSE:completed"


@pytest.mark.asyncio
async def test_a_bad_draft_does_not_break_the_turn(monkeypatch):
    """An unusable entry from the pod is dropped, not fatal: the turn still
    completes and its text is still persisted."""
    saved = {}
    handler = _remote_handler(monkeypatch, saved)
    handler._remote_memory = lambda session, conv_id: None

    async def fake_replies(**kwargs):
        yield "turn_skill", {"entries": [{"slug": "../escape", "files": {"SKILL.md": "x"}}]}
        yield "turn_delta", {"text": "Done."}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    frames = []

    class RecBuffer:
        async def append(self, kind, record):
            frames.append(record["sse"])

        async def close(self, reason):
            frames.append(f"CLOSE:{reason}")

    await handler._produce_remote(
        conv_id=uuid4(), input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=RecBuffer(),
    )

    assert not any(f.startswith("event: response.skill_created") for f in frames)
    assert frames[-1] == "CLOSE:completed"
    assert saved["assistant"] == "Done."
