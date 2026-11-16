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
    relay_addon_step,
    remote_added_texts,
    remote_addon_totals,
    remote_addons_ready,
    remote_progress,
    remote_rate_limited,
    submit_remote_addon,
    withdraw_remote_addon,
)
from cowork.streaming.addons import TurnAddon
from cowork.turnqueue.answers import MAX_ANSWER_BYTES, WRAP_HEADROOM_BYTES

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


def _submit(r, text="use the sales DB", decided_by="jev", timeout=2.0, local_key=None):
    return submit_remote_addon(conversation_id="conv-1", correlation_id="corr-1", text=text,
                               decided_by=decided_by, local_key=local_key, r=r, ack_timeout_s=timeout)


async def _status(r, addon_id):
    raw = await r.hget(INDEX, addon_id)
    return json.loads(raw)["status"] if raw else None


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
    result, addon_id = await _submit(r)
    await pod
    assert result is RemoteAddonResult.REFUSED
    # Abandoned, not held: Jev never reads it, and no pod step is relayed for it.
    assert await _status(r, addon_id) == "abandoned"
    assert await remote_added_texts("corr-1", r=r) == []
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
    assert await _status(r, addon_id) == "abandoned"
    assert await remote_added_texts("corr-1", r=r) == []


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
    text = "a" * (MAX_ANSWER_BYTES - WRAP_HEADROOM_BYTES - overhead)
    result, _ = await _submit(r, text=text, timeout=0.1)
    assert result is RemoteAddonResult.TIMEOUT   # queued; nobody answers
    assert len((await r.lindex(KEY, 0)).encode()) + WRAP_HEADROOM_BYTES == MAX_ANSWER_BYTES
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
    # Kept as withdrawn: the pod's "withdrawn" step is relayed against it,
    # and a second withdraw finds nothing to take back.
    assert await _status(r, addon_id) == "withdrawn"
    assert await _withdraw(r, addon_id) is RemoteWithdrawResult.NOT_FOUND


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


async def test_progress_is_this_turns_recent_steps(r):
    await r.xadd(STREAM, _reply("turn_step", {"step": "tool_start", "id": "old", "name": "old"}, corr="previous"))
    for i in range(10):
        await r.xadd(STREAM, _reply("turn_step", {"step": "tool_start", "id": str(i), "name": f"tool{i}"}))
        await r.xadd(STREAM, _reply("turn_step", {"step": "tool_result", "id": str(i), "name": f"tool{i}"}))
        await r.xadd(STREAM, _reply("turn_delta", {"text": "x"}))
    lines = await remote_progress(conversation_id="conv-1", correlation_id="corr-1", r=r)
    assert lines == [f"[done] tool{i}" for i in range(2, 10)]


async def test_progress_carries_step_descriptions_and_flags_the_running_one(r):
    args = json.dumps({"one_line_description": "Read the tuition pages", "code": "x" * 50})
    await r.xadd(STREAM, _reply("turn_step", {"step": "tool_start", "id": "a", "name": "scratchpad"}))
    await r.xadd(STREAM, _reply("turn_step", {"step": "tool_end", "id": "a", "args": args}))
    await r.xadd(STREAM, _reply("turn_step", {"step": "tool_result", "id": "a", "name": "scratchpad"}))
    # The pod truncates long args: the description is still found.
    cut = '{"one_line_description": "Fill coverage gaps", "code": "' + "y" * 40
    await r.xadd(STREAM, _reply("turn_step", {"step": "tool_start", "id": "b", "name": "scratchpad"}))
    await r.xadd(STREAM, _reply("turn_step", {"step": "tool_end", "id": "b", "args": cut}))
    lines = await remote_progress(conversation_id="conv-1", correlation_id="corr-1", r=r)
    assert lines == [
        "[done] Read the tuition pages (scratchpad)",
        "[in progress] Fill coverage gaps (scratchpad)",
    ]


# ── Recorded before the push; the relay trusts only what was recorded ──────


async def test_the_addon_is_recorded_before_it_reaches_the_pod(r):
    seen = {}

    async def pod():
        entry = json.loads(await _queued(r))
        # By the time the pod can read the entry, the relay can trust it.
        seen["status"] = await _status(r, entry["addon_id"])
        seen["relayed"] = await relay_addon_step(correlation_id="corr-1", data={
            "step": "addon", "addon_id": entry["addon_id"], "state": "added",
            "text": "something else", "decided_by": "user",
        }, r=r)
        await r.xadd(STREAM, _reply("turn_step", {"step": "addon", "addon_id": entry["addon_id"], "state": "added"}))

    task = asyncio.create_task(pod())
    result, addon_id = await _submit(r, local_key="k-1")
    await task
    assert result is RemoteAddonResult.ADDED
    assert seen["status"] == "pending"
    # Text and decider from the record, not the pod; the app's local_key added.
    assert seen["relayed"] == [TurnAddon(addon_id, "added", "use the sales DB", "jev", "k-1")]
    assert await _status(r, addon_id) == "added"


async def test_a_redis_error_before_the_push_reaches_nothing(r, monkeypatch):
    async def boom(*args, **kwargs):
        raise RedisError("connection lost")

    monkeypatch.setattr(r, "xrevrange", boom)
    result, _ = await _submit(r)
    assert result is RemoteAddonResult.REDIS_ERROR
    assert await r.llen(KEY) == 0


@pytest.mark.parametrize("state", ["added", "read", "returned", "withdrawn"])
async def test_steps_for_an_addon_this_turn_never_took_are_dropped(r, state):
    assert await relay_addon_step(correlation_id="corr-1", data={
        "step": "addon", "addon_id": "f" * 32, "state": state, "text": "injected", "decided_by": "jev",
    }, r=r) == []


@pytest.mark.parametrize("state", ["added", "returned"])
async def test_a_late_add_for_an_abandoned_addon_is_dropped(r, state):
    """/addon answered queued, so the app runs it next: an "added" would show
    it twice and a "returned" would run it twice."""
    result, addon_id = await _submit(r, timeout=0.2)
    assert result is RemoteAddonResult.TIMEOUT
    assert await relay_addon_step(correlation_id="corr-1", data={
        "step": "addon", "addon_id": addon_id, "state": state}, r=r) == []


@pytest.mark.parametrize("state", ["read", "withdrawn"])
async def test_read_and_withdrawn_still_reach_the_app_for_an_abandoned_addon(r, state):
    """The app queued it: "read" tells it the model saw it anyway (it drops
    its queued copy), "withdrawn" that the follow-up withdraw landed."""
    result, addon_id = await _submit(r, text="use the sales DB", timeout=0.2, local_key="k-9")
    assert result is RemoteAddonResult.TIMEOUT
    assert await _status(r, addon_id) == "abandoned"
    assert await relay_addon_step(correlation_id="corr-1", data={
        "step": "addon", "addon_id": addon_id, "state": state, "text": "pod's words", "decided_by": "user"}, r=r) == [
        TurnAddon(addon_id, state, "use the sales DB", "jev", "k-9")]


async def test_another_turns_addon_is_not_relayed_here(r):
    addon_id = await _added(r)
    assert await relay_addon_step(correlation_id="corr-2", data={
        "step": "addon", "addon_id": addon_id, "state": "read"}, r=r) == []


@pytest.mark.parametrize("state", ["added", "read", "returned", "withdrawn"])
async def test_the_states_the_app_renders_are_relayed(r, state):
    addon_id = await _added(r)
    assert await relay_addon_step(correlation_id="corr-1", data={
        "step": "addon", "addon_id": addon_id, "state": state}, r=r) == [
        TurnAddon(addon_id, state, "use the sales DB", "jev", None)]


@pytest.mark.parametrize("state", ["refused", "already_read", "something_new"])
async def test_verdicts_on_one_request_are_not_relayed(r, state):
    addon_id = await _added(r)
    assert await relay_addon_step(correlation_id="corr-1", data={
        "step": "addon", "addon_id": addon_id, "state": state, "request_id": "r" * 32}, r=r) == []


async def test_ready_is_recorded_and_never_relayed(r):
    assert await remote_addons_ready("corr-1", r=r) is False
    assert await relay_addon_step(correlation_id="corr-1", data={
        "kind": "addon", "type": "response.addon", "step": "addon", "state": "ready"}, r=r) == []
    assert await remote_addons_ready("corr-1", r=r) is True
    assert await remote_addons_ready("corr-2", r=r) is False
    assert 0 < await r.ttl("cowork:addons-ready:corr-1") <= remote.ADDON_INDEX_TTL_S


async def test_a_redis_error_drops_the_step(r, monkeypatch):
    addon_id = await _added(r)

    async def boom(*args, **kwargs):
        raise RedisError("connection lost")

    monkeypatch.setattr(r, "hget", boom)
    assert await relay_addon_step(correlation_id="corr-1", data={
        "step": "addon", "addon_id": addon_id, "state": "read"}, r=r) == []


# ── Per-turn totals and the shared rate limit ──────────────────────────────


async def test_totals_count_withdrawn_but_not_abandoned(r):
    kept = await _added(r, text="éé")
    pod = asyncio.create_task(_pod_answers_withdraw(r, "withdrawn"))
    assert await _withdraw(r, kept) is RemoteWithdrawResult.WITHDRAWN
    await pod
    await r.delete(KEY)
    await _submit(r, text="never confirmed", timeout=0.1)
    assert await remote_addon_totals("corr-1", r=r) == (1, 4)


async def test_the_rate_limit_is_per_person_and_per_window(r):
    hits = [await remote_rate_limited("o:u", limit=3, window_s=60, r=r) for _ in range(4)]
    assert hits == [False, False, False, True]
    assert await remote_rate_limited("o:someone-else", limit=3, window_s=60, r=r) is False
