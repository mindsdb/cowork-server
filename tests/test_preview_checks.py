"""Preview checks: the broker, the turn's checker, its SSE frame, and the wiring.

The broker mirrors AnswerBroker (tests/test_answer_broker.py): an id that was
never opened, has timed out or whose turn ended reads as not found, and an
open check resolves to None, never hangs, when its turn ends.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
from types import SimpleNamespace

import pytest

from cowork.harnesses.anton_harness import harness
from cowork.harnesses.anton_harness.preview_checker import (
    ARTIFACT_CHECK_EVENT,
    MAX_TIMEOUT_S,
    MIN_TIMEOUT_S,
    artifact_check_sse,
    build_preview_checker,
    preview_checker_kwargs,
)
from cowork.streaming.preview_checks import (
    CHECK_ID_PREFIX,
    CheckSubmitResult,
    PreviewCheckBroker,
    preview_checks,
)

CID = "conv-preview-check"


@pytest.fixture(autouse=True)
def _clean_broker():
    yield
    preview_checks.reset()


class _Buffer:
    """Records what the checker writes; optionally answers it like the app."""

    def __init__(self, answer=None, *, broker: PreviewCheckBroker | None = None, fail=False):
        self.records: list[tuple[str, dict]] = []
        self._answer = answer
        self._broker = broker
        self._fail = fail

    async def append(self, kind: str, data: dict) -> None:
        if self._fail:
            raise OSError("buffer gone")
        self.records.append((kind, data))
        if self._answer is not None:
            check_id = _frame(data["sse"])[1]["check_id"]
            asyncio.get_running_loop().call_soon(self._broker.submit, check_id, self._answer)

    def frames(self) -> list[tuple[str, dict]]:
        return [_frame(data["sse"]) for kind, data in self.records if kind == "sse"]


def _frame(sse: str) -> tuple[str, dict]:
    event_line, data_line, *_ = sse.split("\n")
    assert event_line.startswith("event: ") and data_line.startswith("data: ")
    return event_line[len("event: "):], json.loads(data_line[len("data: "):])


# ── broker ────────────────────────────────────────────────────────────


async def test_open_gives_a_random_prefixed_id_bound_to_its_conversation():
    broker = PreviewCheckBroker()
    first, _ = broker.open(CID)
    second, _ = broker.open(CID)
    assert first != second
    assert first.startswith(CHECK_ID_PREFIX) and len(first) == len(CHECK_ID_PREFIX) + 32
    assert broker.conversation_of(first) == CID


async def test_submit_resolves_once_then_reports_already_answered():
    broker = PreviewCheckBroker()
    check_id, future = broker.open(CID)
    assert broker.submit(check_id, {"status": "passed"}) is CheckSubmitResult.ACCEPTED
    assert future.result() == {"status": "passed"}
    assert broker.submit(check_id, {"status": "failed"}) is CheckSubmitResult.ALREADY_ANSWERED
    assert future.result() == {"status": "passed"}


async def test_unknown_and_closed_ids_are_not_found():
    broker = PreviewCheckBroker()
    assert broker.submit("acheck:nope", {"status": "passed"}) is CheckSubmitResult.NOT_FOUND
    check_id, future = broker.open(CID)
    broker.close(check_id)
    assert future.result() is None
    assert broker.conversation_of(check_id) is None
    assert broker.submit(check_id, {"status": "passed"}) is CheckSubmitResult.NOT_FOUND
    broker.close(check_id)  # idempotent


async def test_turn_end_resolves_only_that_conversations_checks_to_none():
    broker = PreviewCheckBroker()
    mine, my_future = broker.open(CID)
    other, other_future = broker.open("another-conversation")
    broker.close_conversation(CID)
    assert my_future.result() is None
    assert broker.conversation_of(mine) is None
    assert not other_future.done()
    assert broker.conversation_of(other) == "another-conversation"


# ── the SSE frame ─────────────────────────────────────────────────────


def test_the_frame_carries_exactly_the_contract_fields():
    event, data = _frame(artifact_check_sse(check_id="acheck:ab", artifact_slug="sales-dash", timeout_s=30))
    assert event == ARTIFACT_CHECK_EVENT == "response.artifact_check"
    assert data == {
        "type": "response.artifact_check",
        "check_id": "acheck:ab",
        "artifact_slug": "sales-dash",
        "timeout_s": 30,
    }


# ── the turn's checker ────────────────────────────────────────────────


def _checker(buffer, broker=None):
    broker = broker or PreviewCheckBroker()
    return build_preview_checker(CID, broker=broker, buffer_for=lambda _cid: buffer), broker


async def test_publishes_one_frame_and_returns_the_apps_answer():
    broker = PreviewCheckBroker()
    answer = {"status": "failed", "errors": [{"type": "http", "message": "GET /api → 500: boom"}], "ms": 3400}
    buffer = _Buffer(answer, broker=broker)
    checker, _ = _checker(buffer, broker)
    assert await checker("sales-dash", 30) == answer
    [(event, data)] = buffer.frames()
    assert event == "response.artifact_check"
    assert data["artifact_slug"] == "sales-dash" and data["timeout_s"] == 30
    # Answered checks are dropped: a second POST for the same id is a 404.
    assert broker.conversation_of(data["check_id"]) is None


async def test_unavailable_reads_as_no_answer():
    broker = PreviewCheckBroker()
    checker, _ = _checker(_Buffer({"status": "unavailable"}, broker=broker), broker)
    assert await checker("sales-dash", 30) is None


async def test_a_timeout_returns_none_and_drops_the_check(monkeypatch):
    monkeypatch.setattr("cowork.harnesses.anton_harness.preview_checker.MIN_TIMEOUT_S", 0)
    buffer = _Buffer()
    checker, broker = _checker(buffer)
    assert await checker("sales-dash", 0.01) is None
    check_id = buffer.frames()[0][1]["check_id"]
    assert broker.conversation_of(check_id) is None


async def test_turn_end_while_waiting_returns_none():
    buffer = _Buffer()
    checker, broker = _checker(buffer)
    task = asyncio.ensure_future(checker("sales-dash", 60))
    while not buffer.records:
        await asyncio.sleep(0)
    broker.close_conversation(CID)
    assert await asyncio.wait_for(task, 1) is None


async def test_no_live_turn_means_no_frame_and_no_answer():
    checker, broker = _checker(None)
    assert await checker("sales-dash", 30) is None
    assert broker._pending == {}


async def test_a_failing_buffer_is_unverified_not_an_exception():
    checker, broker = _checker(_Buffer(fail=True))
    assert await checker("sales-dash", 30) is None
    assert broker._pending == {}


@pytest.mark.parametrize("slug", ["", "   ", "x" * 201, None])
async def test_a_bad_slug_asks_nothing(slug):
    buffer = _Buffer()
    checker, _ = _checker(buffer)
    assert await checker(slug, 30) is None
    assert buffer.records == []


@pytest.mark.parametrize(
    "asked,sent",
    [(0, MIN_TIMEOUT_S), (-3, MIN_TIMEOUT_S), (12.2, 13), (10_000, MAX_TIMEOUT_S),
     (float("inf"), MAX_TIMEOUT_S), (float("nan"), MIN_TIMEOUT_S), ("soon", MIN_TIMEOUT_S)],
)
async def test_the_timeout_on_the_wire_is_a_bounded_int(asked, sent):
    broker = PreviewCheckBroker()
    buffer = _Buffer({"status": "passed", "errors": [], "ms": 1}, broker=broker)
    checker, _ = _checker(buffer, broker)
    await checker("sales-dash", asked)
    assert buffer.frames()[0][1]["timeout_s"] == sent


async def test_the_default_buffer_is_the_live_turns(monkeypatch):
    """No buffer_for override: the checker finds the turn in the registry, and
    a finished turn (is_answering False) gets no frame."""
    from cowork.streaming.registry import registry

    buffer = _Buffer({"status": "passed", "errors": [], "ms": 1}, broker=preview_checks)
    live = SimpleNamespace(is_answering=True, buffer=buffer)
    monkeypatch.setattr(registry, "get", lambda cid: live if cid == CID else None)
    checker = build_preview_checker(CID)
    assert (await checker("sales-dash", 30))["status"] == "passed"
    live.is_answering = False
    assert await checker("sales-dash", 30) is None
    assert len(buffer.frames()) == 1


# ── wiring into anton ─────────────────────────────────────────────────


@dataclasses.dataclass
class _ConfigWithField:
    llm_client: object = None
    preview_checker: object = None


@dataclasses.dataclass
class _ConfigWithoutField:
    llm_client: object = None


def test_the_checker_is_passed_only_to_an_anton_that_declares_it():
    async def checker(slug, timeout_s):
        return None

    assert preview_checker_kwargs(_ConfigWithField, checker) == {"preview_checker": checker}
    assert preview_checker_kwargs(_ConfigWithoutField, checker) == {}
    assert preview_checker_kwargs(_ConfigWithField, None) == {}
    # Both configs still construct: the server never passes an unknown field.
    _ConfigWithoutField(**preview_checker_kwargs(_ConfigWithoutField, checker))
    assert _ConfigWithField(**preview_checker_kwargs(_ConfigWithField, checker)).preview_checker is checker


def test_the_installed_anton_builds_with_whatever_it_gets():
    from anton.core.session import ChatSessionConfig

    kwargs = preview_checker_kwargs(ChatSessionConfig, build_preview_checker(CID))
    declared = {f.name for f in dataclasses.fields(ChatSessionConfig)}
    assert set(kwargs) == ({"preview_checker"} & declared)


def test_the_session_builder_passes_it_through_the_guard():
    source = inspect.getsource(harness.AntonHarness._build_turn_session)
    assert "**preview_checker_kwargs(ChatSessionConfig, preview_checker)" in source
    # Only a turn the app tails can be answered.
    assert "if interactive and channel_context is None" in source


def test_turn_end_closes_the_conversations_open_checks(monkeypatch):
    import cowork.services.artifact_autopublish as autopublish
    import cowork.services.task_objects as task_objects

    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_a, **_k: (set(), {}))
    monkeypatch.setattr(task_objects, "cards_for_slugs", lambda *_a, **_k: [])

    async def _no_autopublish(*_a, **_k):
        return set()

    monkeypatch.setattr(autopublish, "autopublish_project_artifacts", _no_autopublish)
    opened: list = []

    class _Session:
        async def turn_stream(self, user_input, *, turn_id=None):
            opened.append(preview_checks.open("conv-1"))
            return
            yield  # noqa: marks this an async generator

    async def _fake_build(self, conversation, **kwargs):
        return harness.BuiltChatSession(session=_Session(), turn_scratch_dir=None, seed_info=None, learns=True)

    monkeypatch.setattr(harness.AntonHarness, "_build_chat_session", _fake_build)
    conversation = SimpleNamespace(id="conv-1", project_id="proj-1", project=SimpleNamespace(path="/tmp", name="tmp"))

    async def _drain():
        async for _event in harness.AntonHarness().stream_response(
            conversation=conversation, input=[{"type": "text", "text": "hi"}],
        ):
            pass

    asyncio.run(_drain())
    [(check_id, future)] = opened
    assert future.result() is None
    assert preview_checks.conversation_of(check_id) is None
