"""Host tools for remote (pod) turns: a pod's host_tool_call runs once, off the
relay loop, and answers on the pod's stdin queue in lines that fit it."""
from __future__ import annotations

import asyncio
import json
import logging

import fakeredis.aioredis
import pytest
from redis.exceptions import RedisError

from cowork.turnqueue import host_tools
from cowork.turnqueue.answers import fits_stdin_line
from cowork.turnqueue.host_tools import (
    MAX_RESULT_BYTES,
    HostCall,
    HostToolOutcome,
    HostToolRunner,
    call_key,
    capped,
    progress_entry,
    result_entries,
)
from cowork.turnqueue.redis_client import answer_queue_key

CORR = "corr-1"
CALL = "htool:" + "ab" * 16
STEP = {"step": "host_tool_call", "call_id": CALL, "tool": "read_doc", "input": {"doc": "Plan"}}


@pytest.fixture
def redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _runner(redis, execute, timeouts=None):
    return HostToolRunner(correlation_id=CORR, timeouts=timeouts or {"read_doc": 30}, execute=execute, r=redis)


async def _queued(redis) -> list[dict]:
    return [json.loads(e) for e in await redis.lrange(answer_queue_key(CORR), 0, -1)]


async def _settle(runner):
    await asyncio.gather(*runner.pending_tasks(), return_exceptions=True)


async def test_a_call_answers_with_its_steps_then_its_result(redis):
    seen = []

    async def execute(call, emit):
        seen.append(call)
        await emit("Reading “Plan”")
        return HostToolOutcome("rev 3")

    runner = _runner(redis, execute)
    runner.start(STEP)
    await _settle(runner)
    assert seen == [HostCall(call_id=CALL, tool="read_doc", input={"doc": "Plan"})]
    assert await _queued(redis) == [
        {"kind": "host_tool_progress", "call_id": CALL, "entry_id": f"{CALL}:0", "text": "Reading “Plan”"},
        {"kind": "host_tool_result", "call_id": CALL, "entry_id": f"{CALL}:0", "seq": 0, "last": True,
         "ok": True, "text": "rev 3"},
    ]
    assert 30 < await redis.ttl(call_key(CORR, CALL)) <= 90
    assert 0 < await redis.ttl(answer_queue_key(CORR)) <= 60


def test_the_once_only_key_holds_the_turns_correlation_id():
    assert call_key(CORR, CALL) == f"cowork:htool:{CORR}:{CALL}"


async def test_the_same_call_id_runs_once_even_from_another_replica(redis):
    runs = []

    async def execute(call, emit):
        runs.append(call.call_id)
        return HostToolOutcome("done")

    first, other = _runner(redis, execute), _runner(redis, execute)
    first.start(STEP)
    first.start(STEP)
    other.start(STEP)
    await _settle(first)
    await _settle(other)
    assert runs == [CALL]
    assert len([e for e in await _queued(redis) if e["kind"] == "host_tool_result"]) == 1


async def test_a_tool_this_turn_does_not_offer_is_refused(redis):
    async def execute(call, emit):
        raise AssertionError("ran a tool the turn does not offer")

    runner = _runner(redis, execute)
    runner.start({**STEP, "tool": "delete_doc"})
    await _settle(runner)
    (result,) = await _queued(redis)
    assert result["ok"] is False and result["last"] is True and "delete_doc" in result["text"]


@pytest.mark.parametrize("bad", [
    {**STEP, "call_id": "htool:../x"},
    {**STEP, "call_id": "acheck:" + "ab" * 16},
    {**STEP, "call_id": "htool:" + "AB" * 16},
    {**STEP, "tool": 7},
])
async def test_a_malformed_call_is_dropped(redis, bad):
    async def execute(call, emit):
        raise AssertionError("ran a malformed call")

    runner = _runner(redis, execute)
    runner.start(bad)
    assert runner.pending_tasks() == []
    assert await _queued(redis) == []


async def test_a_failing_tool_answers_not_ok_and_logs_its_stack(redis, caplog):
    async def execute(call, emit):
        raise RuntimeError("boom")

    runner = _runner(redis, execute)
    with caplog.at_level(logging.ERROR, logger=host_tools.logger.name):
        runner.start(STEP)
        await _settle(runner)
    (result,) = await _queued(redis)
    assert result["ok"] is False and "boom" not in result["text"]
    assert any(record.exc_info for record in caplog.records)


async def test_nothing_runs_when_redis_cannot_hold_the_once_only_key():
    class Down:
        async def set(self, *args, **kwargs):
            raise RedisError("down")

    async def execute(call, emit):
        raise AssertionError("ran without the once-only key")

    runner = HostToolRunner(correlation_id=CORR, timeouts={"read_doc": 30}, execute=execute, r=Down())
    runner.start(STEP)
    await _settle(runner)


def test_a_long_answer_is_cut_into_lines_that_each_fit_the_pods_stdin():
    text = 'quote " backslash \\ tab \t nul \x00 emoji 😀 ελληνικά 日本語 ' * 4000
    raw = result_entries(CALL, ok=True, text=text)
    entries = [json.loads(e) for e in raw]
    assert len(raw) > 1 and all(fits_stdin_line(e) for e in raw)
    assert "".join(e["text"] for e in entries) == text
    assert [e["seq"] for e in entries] == list(range(len(entries)))
    assert [e["last"] for e in entries] == [False] * (len(entries) - 1) + [True]
    assert {e["ok"] for e in entries} == {True}
    assert all(e["entry_id"] == f"{CALL}:{e['seq']}" for e in entries)


def test_an_empty_answer_is_one_line():
    (entry,) = [json.loads(e) for e in result_entries(CALL, ok=False, text="")]
    assert entry == {"kind": "host_tool_result", "call_id": CALL, "entry_id": f"{CALL}:0", "seq": 0,
                     "last": True, "ok": False, "text": ""}


def test_an_answer_over_one_mib_becomes_a_refusal():
    at_cap = HostToolOutcome("x" * MAX_RESULT_BYTES)
    assert capped(at_cap) is at_cap
    assert capped(HostToolOutcome("é" * (MAX_RESULT_BYTES // 2 + 1)), "read it by tab") == HostToolOutcome(
        "read it by tab", ok=False)


async def test_an_answer_over_one_mib_is_refused_whatever_the_tool_checked(redis):
    async def execute(call, emit):
        return HostToolOutcome("x" * (MAX_RESULT_BYTES + 1))

    runner = HostToolRunner(correlation_id=CORR, timeouts={"read_doc": 30}, execute=execute,
                            too_big_text="read it by tab", r=redis)
    runner.start(STEP)
    await _settle(runner)
    (result,) = await _queued(redis)
    assert result["ok"] is False and result["last"] is True and result["text"] == "read it by tab"


async def test_an_answer_that_cannot_be_encoded_is_refused_and_logged(redis, caplog):
    async def execute(call, emit):
        return HostToolOutcome("broken \ud800 text")

    runner = _runner(redis, execute)
    with caplog.at_level(logging.ERROR, logger=host_tools.logger.name):
        runner.start(STEP)
        await _settle(runner)
    (result,) = await _queued(redis)
    assert result["ok"] is False and result["last"] is True and "host_error" in result["text"]
    assert any(CALL in record.getMessage() and record.exc_info for record in caplog.records)
    assert not any("broken" in record.getMessage() for record in caplog.records)


async def test_a_step_with_a_lone_surrogate_is_sent_with_it_replaced(redis):
    async def execute(call, emit):
        await emit("Reading \ud800 doc")
        return HostToolOutcome("rev 3")

    runner = _runner(redis, execute)
    runner.start(STEP)
    await _settle(runner)
    progress, result = await _queued(redis)
    assert progress["kind"] == "host_tool_progress" and progress["text"].startswith("Reading ")
    assert "\ud800" not in progress["text"]
    assert result["ok"] is True and result["text"] == "rev 3"


async def test_an_entry_redis_cannot_encode_is_dropped_and_logged_by_call_id(redis, caplog):
    runner = _runner(redis, None)
    call = HostCall(call_id=CALL, tool="read_doc", input={})
    with caplog.at_level(logging.WARNING, logger=host_tools.logger.name):
        await runner._push(redis, call, ["broken \ud800 entry"])
    assert any(CALL in record.getMessage() for record in caplog.records)
    assert not any("broken" in record.getMessage() for record in caplog.records)


def test_a_step_is_cut_to_500_characters():
    entry = json.loads(progress_entry(CALL, 3, "x" * 900))
    assert entry["entry_id"] == f"{CALL}:3" and len(entry["text"]) == 500


async def test_after_the_turn_ends_a_call_still_running_sends_nothing(redis):
    release = asyncio.Event()

    async def execute(call, emit):
        await release.wait()
        await emit("late step")
        return HostToolOutcome("late")

    runner = _runner(redis, execute)
    runner.start(STEP)
    await asyncio.sleep(0.01)
    late = await runner.finish(timeout=0.05)
    assert late == [HostCall(call_id=CALL, tool="read_doc", input={"doc": "Plan"})]
    release.set()
    await _settle(runner)
    assert await _queued(redis) == []


async def test_the_turn_end_waits_for_a_call_that_finishes_in_time(redis):
    async def execute(call, emit):
        await asyncio.sleep(0.05)
        return HostToolOutcome("done")

    runner = _runner(redis, execute)
    runner.start(STEP)
    assert await runner.finish(timeout=5) == []
    assert runner.pending_tasks() == []
    # The pod's turn is over, so the answer is not sent.
    assert await _queued(redis) == []


async def test_the_turn_end_waits_for_a_hung_call_only_once(redis):
    release = asyncio.Event()

    async def execute(call, emit):
        await release.wait()
        return HostToolOutcome("late")

    runner = _runner(redis, execute)
    runner.start(STEP)
    await asyncio.sleep(0.01)
    assert len(await runner.finish(timeout=0.05)) == 1
    loop = asyncio.get_running_loop()
    started = loop.time()
    # A second turn-end pass (a Stop after the clean finish) neither waits nor reports it again.
    assert await runner.finish(timeout=5) == []
    assert loop.time() - started < 1
    release.set()
    await _settle(runner)


async def test_the_answer_is_cut_up_off_the_event_loop(redis, monkeypatch):
    import threading

    loop_thread = threading.get_ident()
    threads: list[int] = []
    real = host_tools.result_entries

    def spy(call_id, *, ok, text):
        threads.append(threading.get_ident())
        return real(call_id, ok=ok, text=text)

    monkeypatch.setattr(host_tools, "result_entries", spy)

    async def execute(call, emit):
        return HostToolOutcome("rev 3")

    runner = _runner(redis, execute)
    runner.start(STEP)
    await _settle(runner)
    assert threads and loop_thread not in threads
