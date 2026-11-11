"""turnqueue.addons: hand a mid-task message to a pod turn, or take it back
(ENG-4194). The pod's verdicts come back as `addon` turn_steps."""

from __future__ import annotations

import asyncio
import json

import fakeredis.aioredis
import pytest
from redis.exceptions import RedisError

from cowork.turnqueue import addons as remote
from cowork.turnqueue.addons import (
    RemoteAddonResult,
    RemoteWithdrawResult,
    remote_added_texts,
    remote_progress,
    submit_remote_addon,
    withdraw_remote_addon,
)
from cowork.turnqueue.answers import MAX_ANSWER_BYTES

STREAM = "scratchpad:reply:conv-1"
KEY = "cowork:answer:corr-1"
INDEX = "cowork:addons:corr-1"


@pytest.fixture
def r():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _reply(kind, data, corr="corr-1"):
    return {"payload": json.dumps({"correlation_id": corr, "kind": kind, "data": data})}


async def _queued(r, index=0):
    for _ in range(300):
        raw = await r.lindex(KEY, index)
        if raw:
            return raw
        await asyncio.sleep(0.01)
    raise AssertionError("nothing was queued")


async def _pod_answers_addon(r, state, corr="corr-1"):
    entry = json.loads(await _queued(r))
    await r.xadd(STREAM, _reply("turn_step", {
        "step": "addon", "type": "response.addon", "addon_id": entry["addon_id"], "state": state,
        "text": entry["text"], "decided_by": entry["decided_by"],
    }, corr))


async def _pod_answers_withdraw(r, state):
    entry = json.loads(await _queued(r))
    await r.xadd(STREAM, _reply("turn_step", {
        "step": "addon", "type": "response.addon", "addon_id": entry["addon_id"], "state": state,
        "request_id": entry["request_id"],
    }))


def _submit(r, text="use the sales DB", decided_by="jev", timeout=2.0):
    return submit_remote_addon(conversation_id="conv-1", correlation_id="corr-1", text=text,
                               decided_by=decided_by, r=r, ack_timeout_s=timeout)


def _withdraw(r, addon_id, timeout=2.0):
    return withdraw_remote_addon(conversation_id="conv-1", correlation_id="corr-1",
                                 addon_id=addon_id, r=r, ack_timeout_s=timeout)


# ── submit_remote_addon ────────────────────────────────────────────────────


async def test_added_when_the_pod_takes_it(r):
    pod = asyncio.create_task(_pod_answers_addon(r, "added"))
    result, addon_id = await _submit(r, text="usá la base de ventas")
    await pod
    assert result is RemoteAddonResult.ADDED
    raw = await r.lindex(KEY, 0)
    # The wire contract, byte for byte: compact, ensure_ascii=False.
    assert raw == ('{"kind":"addon","addon_id":"%s","text":"usá la base de ventas","decided_by":"jev"}'
                   % addon_id)
    assert len(addon_id) == 32 and int(addon_id, 16) >= 0
    assert 0 < await r.ttl(KEY) <= 60
    # Kept for the next decision and for a withdraw.
    assert await remote_added_texts("corr-1", r=r) == ["usá la base de ventas"]
    assert 0 < await r.ttl(INDEX) <= remote.ADDON_INDEX_TTL_S


async def test_a_user_decision_is_sent_as_such(r):
    pod = asyncio.create_task(_pod_answers_addon(r, "added"))
    result, _ = await _submit(r, decided_by="user")
    await pod
    assert result is RemoteAddonResult.ADDED
    assert json.loads(await r.lindex(KEY, 0))["decided_by"] == "user"


async def test_refused_is_not_remembered(r):
    pod = asyncio.create_task(_pod_answers_addon(r, "refused"))
    result, _ = await _submit(r)
    await pod
    assert result is RemoteAddonResult.REFUSED
    assert await r.exists(INDEX) == 0
    assert await r.llen(KEY) == 1   # no withdraw for a refusal


async def test_no_verdict_queues_a_withdraw_of_it(r):
    result, addon_id = await _submit(r, timeout=0.3)
    assert result is RemoteAddonResult.TIMEOUT
    entries = [json.loads(raw) for raw in await r.lrange(KEY, 0, -1)]
    assert [e["kind"] for e in entries] == ["addon", "addon_withdraw"]
    withdraw = entries[1]
    assert set(withdraw) == {"kind", "addon_id", "request_id"}
    assert withdraw["addon_id"] == addon_id
    assert len(withdraw["request_id"]) == 32
    assert await r.exists(INDEX) == 0


async def test_verdicts_for_another_addon_or_turn_are_ignored(r):
    async def noise():
        entry = json.loads(await _queued(r))
        await r.xadd(STREAM, _reply("turn_step", {"step": "addon", "addon_id": "other", "state": "added"}))
        await r.xadd(STREAM, _reply("turn_step", {"step": "addon", "addon_id": entry["addon_id"],
                                                  "state": "added"}, corr="other-turn"))
    task = asyncio.create_task(noise())
    result, _ = await _submit(r, timeout=0.3)
    await task
    assert result is RemoteAddonResult.TIMEOUT


async def test_a_redis_error_while_waiting_withdraws_and_times_out(r, monkeypatch):
    async def boom(*args, **kwargs):
        raise RedisError("connection lost")

    monkeypatch.setattr(r, "xread", boom)
    result, addon_id = await _submit(r)
    assert result is RemoteAddonResult.TIMEOUT
    kinds = [json.loads(raw)["kind"] for raw in await r.lrange(KEY, 0, -1)]
    assert kinds == ["addon", "addon_withdraw"]


async def test_a_turn_that_already_ended_queues_nothing(r):
    await r.xadd(STREAM, _reply("turn_completed", {}))
    result, _ = await _submit(r)
    assert result is RemoteAddonResult.NOT_FOUND
    assert await r.llen(KEY) == 0


async def test_a_turn_ending_before_the_verdict_is_not_found(r):
    async def ends():
        await _queued(r)
        await r.xadd(STREAM, _reply("turn_completed", {}))
    task = asyncio.create_task(ends())
    result, _ = await _submit(r)
    await task
    assert result is RemoteAddonResult.NOT_FOUND


async def test_an_entry_over_the_stdin_bound_is_rejected_unqueued(r):
    # Multi-byte text counts at its UTF-8 width: under the bound in
    # characters, over it in bytes.
    text = "é" * (MAX_ANSWER_BYTES // 2)
    result, _ = await _submit(r, text=text)
    assert result is RemoteAddonResult.TOO_LARGE
    assert await r.llen(KEY) == 0


async def test_an_entry_just_under_the_bound_is_queued(r):
    overhead = len(remote.addon_entry(addon_id="a" * 32, text="", decided_by="jev").encode())
    text = "a" * (MAX_ANSWER_BYTES - remote._WRAP_HEADROOM_BYTES - overhead)
    result, _ = await _submit(r, text=text, timeout=0.1)
    assert result is RemoteAddonResult.TIMEOUT   # queued; nobody answers
    assert len((await r.lindex(KEY, 0)).encode()) + remote._WRAP_HEADROOM_BYTES == MAX_ANSWER_BYTES
    result, _ = await _submit(r, text=text + "a")
    assert result is RemoteAddonResult.TOO_LARGE


# ── withdraw_remote_addon ──────────────────────────────────────────────────


async def _added(r, text="use the sales DB"):
    pod = asyncio.create_task(_pod_answers_addon(r, "added"))
    result, addon_id = await _submit(r, text=text)
    await pod
    assert result is RemoteAddonResult.ADDED
    await r.delete(KEY)
    return addon_id


async def test_withdrawn_before_the_model_reads_it(r):
    addon_id = await _added(r)
    pod = asyncio.create_task(_pod_answers_withdraw(r, "withdrawn"))
    assert await _withdraw(r, addon_id) is RemoteWithdrawResult.WITHDRAWN
    await pod
    entry = await r.lindex(KEY, 0)
    request_id = json.loads(entry)["request_id"]
    assert entry == '{"kind":"addon_withdraw","addon_id":"%s","request_id":"%s"}' % (addon_id, request_id)
    assert await remote_added_texts("corr-1", r=r) == []


async def test_already_read_keeps_it_added(r):
    addon_id = await _added(r)
    pod = asyncio.create_task(_pod_answers_withdraw(r, "already_read"))
    assert await _withdraw(r, addon_id) is RemoteWithdrawResult.ALREADY_READ
    await pod
    assert await remote_added_texts("corr-1", r=r) == ["use the sales DB"]


async def test_no_withdraw_verdict_is_unavailable(r):
    addon_id = await _added(r)
    assert await _withdraw(r, addon_id, timeout=0.3) is RemoteWithdrawResult.UNAVAILABLE
    assert await remote_added_texts("corr-1", r=r) == ["use the sales DB"]


async def test_a_redis_error_on_withdraw_is_unavailable(r, monkeypatch):
    addon_id = await _added(r)

    async def boom(*args, **kwargs):
        raise RedisError("connection lost")

    monkeypatch.setattr(r, "xread", boom)
    assert await _withdraw(r, addon_id) is RemoteWithdrawResult.UNAVAILABLE


async def test_withdrawing_an_addon_the_turn_never_took_queues_nothing(r):
    assert await _withdraw(r, "f" * 32) is RemoteWithdrawResult.NOT_FOUND
    assert await r.llen(KEY) == 0


async def test_withdrawing_after_the_turn_ended_is_not_found(r):
    addon_id = await _added(r)
    await r.xadd(STREAM, _reply("turn_completed", {}))
    assert await _withdraw(r, addon_id) is RemoteWithdrawResult.NOT_FOUND
    assert await r.llen(KEY) == 0


# ── What Jev reads ─────────────────────────────────────────────────────────


async def test_added_texts_come_back_oldest_first(r):
    await _added(r, text="first")
    await _added(r, text="second")
    assert await remote_added_texts("corr-1", r=r) == ["first", "second"]


async def test_progress_is_this_turns_recent_tool_calls(r):
    await r.xadd(STREAM, _reply("turn_step", {"step": "tool_start", "name": "old"}, corr="previous"))
    for i in range(10):
        await r.xadd(STREAM, _reply("turn_step", {"step": "tool_start", "id": str(i), "name": f"tool{i}"}))
        await r.xadd(STREAM, _reply("turn_delta", {"text": "x"}))
    lines = await remote_progress(conversation_id="conv-1", correlation_id="corr-1", r=r)
    assert lines == [f"[ran tool: tool{i}]" for i in range(2, 10)]
