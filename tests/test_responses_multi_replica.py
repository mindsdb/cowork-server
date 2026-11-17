"""Endpoints must answer for a turn this replica did not start.

With two replicas behind a load balancer, a reconnect or a stop lands on
whichever one the balancer picks. Before this, the replica that did not start
the turn had no handle and no buffer, so it reported the turn as missing.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from uuid import uuid4

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from cowork.api.v1.endpoints import responses as responses_ep
from cowork.db.scoped import TenantScope
from cowork.server import create_app
from cowork.streaming import backend as backend_mod
from cowork.streaming import buffer as buffer_mod
from cowork.streaming import cancellation, turn_claim, turn_index
from cowork.streaming.buffer import RedisStreamBuffer
from cowork.streaming.registry import RunRegistry, TurnInProgress, TurnLifecycle, mark_question_saved, registry
from cowork.streaming.turn_claim import TurnClaim, TurnClaimUnavailable, claim_key, pod_name
from tests._logging_assertions import assert_log_fields

from _fakes import UnreachableRedis

REGISTRY_LOGGER = "cowork.streaming.registry"
REFUSAL = {
    "detail": (
        "Another question is still being answered in this conversation. "
        "Wait for it to finish, then send yours again."
    ),
    "code": "turn_in_progress",
}
COMPLETED = {"sse": "event: response.completed\ndata: {}\n\n"}


@pytest.fixture
def fake_redis(monkeypatch):
    """One Redis, no local handles: this replica is the one that did not start
    the turn."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    for mod in (turn_claim, turn_index, buffer_mod, cancellation):
        monkeypatch.setattr(mod, "get_redis", lambda: client)
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.responses.get_redis", lambda: client)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    registry.reset()
    yield client
    registry.reset()


@pytest.fixture
def client():
    return TestClient(create_app())


def _sse(text: str) -> dict:
    """The record shape producers write: readers emit data["sse"] verbatim."""
    return {"sse": f"event: response.output_text.delta\ndata: {{\"delta\": \"{text}\"}}\n\n"}


async def _running_turn(conversation_id: str, turn_id: int, correlation_id: str):
    await turn_index.record_turn(
        conversation_id, turn_id=turn_id, correlation_id=correlation_id,
        org_id=None, user_id=None,
    )
    buf = RedisStreamBuffer(conversation_id=conversation_id, turn_id=turn_id)
    await buf.append("sse", _sse("hi"))
    return buf


@pytest.mark.asyncio
async def test_in_flight_reports_a_turn_started_elsewhere(client, fake_redis):
    """Answering False would make the UI think the turn had finished."""
    await _running_turn("c1", 4, "corr-1")

    body = client.get("/api/v1/responses/in-flight?conversation_id=c1").json()

    assert body["in_flight"] is True
    assert body["has_buffer"] is True
    assert body["latest_seq"] == 1
    assert body["turn_id"] == 4


@pytest.mark.asyncio
async def test_in_flight_is_false_once_the_buffer_is_closed(client, fake_redis):
    """Liveness comes from the terminal record, not from a process."""
    buf = await _running_turn("c2", 1, "corr-2")
    await buf.close("completed")

    body = client.get("/api/v1/responses/in-flight?conversation_id=c2").json()

    assert body["in_flight"] is False
    assert body["has_buffer"] is True


@pytest.mark.asyncio
async def test_in_flight_list_includes_turns_started_elsewhere(client, fake_redis):
    await _running_turn("c3", 1, "corr-3")

    body = client.get("/api/v1/responses/in-flight-list").json()

    assert [row["conversation_id"] for row in body["in_flight"]] == ["c3"]


@pytest.mark.asyncio
async def test_in_flight_list_skips_finished_turns(client, fake_redis):
    buf = await _running_turn("c4", 1, "corr-4")
    await buf.close("completed")

    body = client.get("/api/v1/responses/in-flight-list").json()

    assert body["in_flight"] == []


@pytest.mark.asyncio
async def test_tail_streams_a_turn_started_elsewhere(client, fake_redis):
    buf = await _running_turn("c5", 1, "corr-5")
    await buf.append("sse", _sse("hello-from-elsewhere"))
    await buf.close("completed")

    with client.stream("GET", "/api/v1/responses/tail?conversation_id=c5") as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "hello-from-elsewhere" in body


@pytest.mark.asyncio
async def test_cancel_sets_the_flag_the_controller_reads(client, fake_redis):
    """The pod is where the tokens are spent, and only the flag reaches it."""
    await _running_turn("c6", 1, "corr-6")

    body = client.post(
        "/api/v1/responses/cancel", json={"conversation_id": "c6"}).json()

    assert body["cancelled"] is True
    assert await fake_redis.exists("cowork:cancel:corr-6") == 1


@pytest.mark.asyncio
async def test_cancel_404s_when_no_turn_is_recorded(client, fake_redis):
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": "ghost"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_404s_for_a_finished_turn(client, fake_redis):
    buf = await _running_turn("c7", 1, "corr-7")
    await buf.close("completed")

    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": "c7"})

    assert resp.status_code == 404
    assert await fake_redis.exists("cowork:cancel:corr-7") == 0


@pytest.mark.asyncio
async def test_a_just_enqueued_turn_reads_as_running(client, fake_redis):
    """The index entry is written when the job is enqueued; the first record
    only lands once the pod answers. An empty stream in between means "not
    started yet", not "finished"."""
    await turn_index.record_turn(
        "c8", turn_id=1, correlation_id="corr-8", org_id=None, user_id=None)

    body = client.get("/api/v1/responses/in-flight?conversation_id=c8").json()

    assert body["in_flight"] is True
    assert body["latest_seq"] == 0


@pytest.mark.asyncio
async def test_an_old_entry_with_no_stream_reads_as_finished(client, fake_redis):
    """Past the grace period an empty stream is a truncated conversation, whose
    buffers were deleted while the index entry lived on."""
    await turn_index.record_turn(
        "c9", turn_id=1, correlation_id="corr-9", org_id=None, user_id=None)
    await fake_redis.hset("cowork:turn:c9", "started_at", "1")   # 1970

    body = client.get("/api/v1/responses/in-flight?conversation_id=c9").json()

    assert body["in_flight"] is False


@pytest.mark.asyncio
async def test_cancel_works_on_a_turn_that_has_not_spoken_yet(client, fake_redis):
    """Stop must work while the pod is still starting, which is exactly when a
    user is most likely to press it."""
    await turn_index.record_turn(
        "c10", turn_id=1, correlation_id="corr-10", org_id=None, user_id=None)

    body = client.post(
        "/api/v1/responses/cancel", json={"conversation_id": "c10"}).json()

    assert body["cancelled"] is True
    assert await fake_redis.exists("cowork:cancel:corr-10") == 1


@pytest.mark.asyncio
async def test_a_stall_cancel_writes_its_cause_in_one_transaction_with_the_flag(
    client, fake_redis, monkeypatch,
):
    """The controller deletes the flag right after it reports the cancel, so
    the cause must be in Redis whenever the flag is. Written together in one
    transaction, a failed write can leave no stray cause for a later Stop to
    pick up."""
    await _running_turn("c11", 1, "corr-11")
    transactions: list[bool | None] = []
    direct_sets: list[str] = []
    real_pipeline, real_set = fake_redis.pipeline, fake_redis.set

    def recording_pipeline(*args, **kwargs):
        transactions.append(kwargs.get("transaction"))
        return real_pipeline(*args, **kwargs)

    async def recording_set(key, value, *args, **kwargs):
        direct_sets.append(key)
        return await real_set(key, value, *args, **kwargs)

    monkeypatch.setattr(fake_redis, "pipeline", recording_pipeline)
    monkeypatch.setattr(fake_redis, "set", recording_set)

    body = client.post(
        "/api/v1/responses/cancel",
        json={"conversation_id": "c11", "reason": "stalled"},
    ).json()

    assert body["cancelled"] is True
    assert transactions == [True]
    assert direct_sets == [], "the cause or the flag was written outside the transaction"
    assert await fake_redis.get("cowork:cancel_cause:corr-11") == "stalled"
    assert await fake_redis.exists("cowork:cancel:corr-11") == 1
    assert 0 < await fake_redis.ttl("cowork:cancel_cause:corr-11") <= 300


@pytest.mark.asyncio
async def test_a_stop_after_a_stall_clears_the_stall_cause(client, fake_redis):
    """A running cell swallows the stall cancel and the user then clicks Stop.
    The latest cancel decides, so the owner must find no stall cause."""
    await _running_turn("c14", 1, "corr-14")

    client.post("/api/v1/responses/cancel", json={"conversation_id": "c14", "reason": "stalled"})
    client.post("/api/v1/responses/cancel", json={"conversation_id": "c14"})

    assert await fake_redis.exists("cowork:cancel:corr-14") == 1
    assert await fake_redis.exists("cowork:cancel_cause:corr-14") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "user", "something-newer"])
async def test_only_a_stall_writes_a_cancel_cause(client, fake_redis, reason):
    """A Stop, and any reason this server does not know, cancels exactly as
    before and leaves no cause behind."""
    await _running_turn("c12", 1, "corr-12")
    payload = {"conversation_id": "c12"} if reason is None else {
        "conversation_id": "c12", "reason": reason,
    }

    resp = client.post("/api/v1/responses/cancel", json=payload)

    assert resp.status_code == 200
    assert await fake_redis.exists("cowork:cancel:corr-12") == 1
    assert await fake_redis.exists("cowork:cancel_cause:corr-12") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled", [True, False])
async def test_the_owner_saves_a_stall_another_replica_reported(monkeypatch, fake_redis, stalled):
    """The replica that owns the turn learns of the cancel only from the
    controller's turn_failed, after the controller deleted the flag. The cause
    key is what tells it a stall from a Stop."""
    import cowork.handlers.responses as responses_mod
    from test_responses_remote_backend import _RecBuffer, _remote_handler

    monkeypatch.setattr(responses_mod, "get_redis", lambda: fake_redis)
    if stalled:
        await fake_redis.set("cowork:cancel_cause:corr-13", "stalled", ex=300)
    saved: dict = {}
    handler = _remote_handler(monkeypatch, saved)

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_delta", {"text": "partial"}
        yield "turn_failed", {"error": "cancelled"}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", replies)
    buffer = _RecBuffer()

    await handler._produce_remote(
        conv_id="c13", input_text="hi", original_content="hi", model="anton",
        harness_id="anton", buffer=buffer, turn_llm={"correlation_id": "corr-13"},
    )

    assert saved["assistant"] == "partial"
    if stalled:
        assert buffer.frames[-1] == "CLOSE:interrupted"
        failed = [f for f in buffer.frames if f.startswith("event: response.failed")]
        assert len(failed) == 1 and '"code": "stalled"' in failed[0]
        assert saved["events"][-1]["code"] == "stalled"
        assert saved["events"][-1]["request_id"] == "corr-13"
    else:
        # A Stop: partial answer, no error row, closed cancelled.
        assert buffer.frames[-1] == "CLOSE:cancelled"
        assert not any("response.failed" in f for f in buffer.frames)
        assert not any(e.get("type") == "response.failed" for e in saved["events"])



@pytest.mark.asyncio
async def test_the_owner_drops_a_remote_turn_another_replica_deleted(monkeypatch, fake_redis):
    """A delete served by a replica that does not own the turn reaches it only
    through the cancel flag, which the controller answers with its cancel
    reply. The cause key says the conversation was deleted, so the owner
    saves nothing into it and writes no terminal record, which would
    recreate the stream the delete removed. Read as a Stop, the partial
    answer used to land in the deleted history."""
    import cowork.handlers.responses as responses_mod
    from cowork.streaming.registry import RunIdentity
    from test_responses_remote_backend import _RecBuffer, _remote_handler

    monkeypatch.setattr(responses_mod, "get_redis", lambda: fake_redis)
    await fake_redis.set("cowork:cancel_cause:corr-14", "discarded", ex=300)
    saved: dict = {}
    handler = _remote_handler(monkeypatch, saved)

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_delta", {"text": "partial"}
        yield "turn_failed", {"error": "cancelled"}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", replies)
    buffer = _RecBuffer()
    lifecycle = TurnLifecycle(identity=RunIdentity(correlation_id="corr-14", remote=True))

    await handler._produce_remote(
        conv_id="c14", input_text="hi", original_content="hi", model="anton",
        harness_id="anton", buffer=buffer, turn_llm={"correlation_id": "corr-14"}, lifecycle=lifecycle,
    )

    assert (lifecycle.discarded, lifecycle.discarded_elsewhere) == (True, True)
    assert "assistant" not in saved and "events" not in saved
    assert not any(frame.startswith("CLOSE:") for frame in buffer.frames)


# ── admission across replicas ────────────────────────────────────────
#
# Two RunRegistry objects on one Redis are two replicas: each knows only the
# turns it started, and the turn claim is all they share.


def _waiting(started: asyncio.Event, release: asyncio.Event):
    """A producer that writes a frame, then answers once ``release`` is set."""

    async def produce(buffer):
        await buffer.append("sse", _sse("partial"))
        started.set()
        await release.wait()
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    return produce


async def _answered(buffer):
    await buffer.append("sse", COMPLETED)
    await buffer.close("completed")


def _opens(conversation_id: str, turn_id: int, opened: list[int] | None = None):
    async def open_buffer():
        if opened is not None:
            opened.append(turn_id)
        return await backend_mod.new_buffer(conversation_id, turn_id)

    return open_buffer


def _foreign_claim(*, turn_id: int, run_id: str = "run-elsewhere") -> str:
    return TurnClaim(
        run_id=run_id, turn_id=turn_id, correlation_id="corr-elsewhere", pod="other-pod", started_at=1.0,
    ).model_dump_json()


def _refusal_records(caplog, event: str) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.name == REGISTRY_LOGGER and r.levelno == logging.WARNING and getattr(r, "event", None) == event
    ]


@pytest.mark.asyncio
async def test_a_duplicate_send_on_another_replica_is_refused(fake_redis, caplog, owned_logger):
    """Replica B used to admit it: its own memory held no turn. It then wrote
    into replica A's stream from seq 0 and replaced A's run in the turn index,
    so Stop and answers reached B's run instead of A's.

    The refusal logs one WARNING naming the run and pod that hold the claim,
    which the hosted deployment prints."""
    logged = owned_logger(REGISTRY_LOGGER, level=logging.WARNING)
    replica_a, replica_b = RunRegistry(), RunRegistry()
    started, release = asyncio.Event(), asyncio.Event()
    first = await replica_a.start(
        conversation_id="dup", turn_id=1, open_buffer=_opens("dup", 1), produce=_waiting(started, release),
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    opened: list[int] = []

    with pytest.raises(TurnInProgress) as refused:
        await replica_b.start(
            conversation_id="dup", turn_id=1, open_buffer=_opens("dup", 1, opened), produce=_answered,
        )

    assert (refused.value.conversation_id, refused.value.turn_id) == ("dup", 1)
    fields = {
        "event": "turn_claim_held", "conversation_id": "dup", "turn_id": 1,
        "holder_run_id": first.run_id, "holder_pod": pod_name(),
    }
    records = _refusal_records(caplog, "turn_claim_held")
    assert [{name: getattr(r, name) for name in fields} for r in records] == [fields]
    assert_log_fields(logged.output(), **fields)
    assert opened == []
    assert replica_b.get("dup") is None
    assert (await turn_index.get_turn("dup")).run_id == first.run_id
    assert first.is_answering
    assert await fake_redis.xlen("cowork:stream:dup:1") == 1
    release.set()
    await asyncio.wait_for(first.task, timeout=2)
    # Released with the terminal record, so the next send is admitted anywhere.
    after = await replica_b.start(
        conversation_id="dup", turn_id=3, open_buffer=_opens("dup", 3), produce=_answered,
    )
    await asyncio.wait_for(after.task, timeout=2)


@pytest.mark.asyncio
async def test_an_absent_claim_admits_and_the_winner_writes_the_index(fake_redis):
    runs = RunRegistry()
    started, release = asyncio.Event(), asyncio.Event()

    handle = await runs.start(
        conversation_id="fresh", turn_id=0, open_buffer=_opens("fresh", 0), produce=_waiting(started, release),
    )
    await asyncio.wait_for(started.wait(), timeout=2)

    stored = TurnClaim.model_validate_json(await fake_redis.get(claim_key("fresh")))
    assert (stored.run_id, stored.turn_id) == (handle.run_id, 0)
    assert stored.correlation_id == handle.lifecycle.identity.correlation_id
    assert (await turn_index.get_turn("fresh")).run_id == handle.run_id
    release.set()
    await asyncio.wait_for(handle.task, timeout=2)
    assert await fake_redis.exists(claim_key("fresh")) == 0


@pytest.mark.asyncio
async def test_a_follow_up_on_another_replica_waits_for_a_holder_writing_its_last_frame(fake_redis):
    """The web UI sends a queued question on the final client frame, before
    the terminal record lands, and the other replica must not answer it 409."""
    replica_a, replica_b = RunRegistry(), RunRegistry()
    final_written, release = asyncio.Event(), asyncio.Event()

    async def finishing(buffer):
        await buffer.append("sse", COMPLETED)
        final_written.set()
        await release.wait()
        await buffer.close("completed")

    first = await replica_a.start(
        conversation_id="fu", turn_id=0, open_buffer=_opens("fu", 0), produce=finishing,
    )
    await asyncio.wait_for(final_written.wait(), timeout=2)
    following = asyncio.create_task(replica_b.start(
        conversation_id="fu", turn_id=2, open_buffer=_opens("fu", 2), produce=_answered,
    ))
    try:
        await asyncio.sleep(0.3)
        assert not following.done(), "a queued follow-up must wait rather than receive 409"
    finally:
        release.set()
    second = await asyncio.wait_for(following, timeout=5)
    await asyncio.gather(first.task, second.task)
    assert second.claim is not None


@pytest.mark.asyncio
async def test_a_holder_that_never_writes_its_terminal_refuses_after_the_bounded_wait(fake_redis, monkeypatch):
    monkeypatch.setattr(turn_claim, "ENDING_HOLDER_WAIT_SECONDS", 0.2)
    await fake_redis.set(claim_key("stuck"), _foreign_claim(turn_id=0), px=70_000)
    await RedisStreamBuffer(conversation_id="stuck", turn_id=0).append("sse", COMPLETED)

    with pytest.raises(TurnInProgress) as refused:
        await asyncio.wait_for(RunRegistry().start(
            conversation_id="stuck", turn_id=2, open_buffer=_opens("stuck", 2), produce=_answered,
        ), timeout=5)

    assert refused.value.turn_id == 0
    assert TurnClaim.model_validate_json(await fake_redis.get(claim_key("stuck"))).run_id == "run-elsewhere"


@pytest.mark.asyncio
async def test_a_claim_left_over_an_ended_turn_is_cleared_and_the_send_admitted(fake_redis, monkeypatch):
    """A claim whose release failed lingers over a stream that already ended;
    the next send waits the bounded time for its holder's own release, then
    clears it by compare-and-delete rather than waiting out its TTL."""
    monkeypatch.setattr(turn_claim, "ENDING_HOLDER_WAIT_SECONDS", 0.2)
    await fake_redis.set(claim_key("left"), _foreign_claim(turn_id=1), px=70_000)
    ended = RedisStreamBuffer(conversation_id="left", turn_id=1)
    await ended.append("sse", COMPLETED)
    await ended.close("completed")

    handle = await RunRegistry().start(
        conversation_id="left", turn_id=3, open_buffer=_opens("left", 3), produce=_answered,
    )

    assert handle.claim is not None
    await asyncio.wait_for(handle.task, timeout=2)
    assert [f["type"] for _id, f in await fake_redis.xrange("cowork:stream:left:1")] == ["sse", "Done"]


@pytest.mark.asyncio
async def test_a_redis_error_at_admission_refuses_the_turn_before_it_opens(fake_redis, monkeypatch):
    monkeypatch.setattr(turn_claim, "get_redis", lambda: UnreachableRedis())
    opened: list[int] = []

    with pytest.raises(TurnClaimUnavailable):
        await RunRegistry().start(
            conversation_id="blind", turn_id=0, open_buffer=_opens("blind", 0, opened), produce=_answered,
        )

    assert opened == []
    assert await turn_index.get_turn("blind") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing,refusal",
    [("open_buffer", OSError), ("open_buffer_redis", TurnClaimUnavailable), ("turn_index", TurnClaimUnavailable)],
)
async def test_a_start_that_fails_after_taking_the_claim_releases_it(fake_redis, monkeypatch, failing, refusal):
    """The claim is released whatever failed. A Redis error or timeout there
    is refused as one before the claim is: TurnClaimUnavailable, which
    answers 503 with Retry-After, so the client sends again. It used to
    surface as the raw Redis error, a 500 no client retries."""
    async def failing_open():
        raise OSError("no buffer")

    async def redis_failing_open():
        raise RedisConnectionError("Redis is unreachable")

    async def failing_record(*_args, **_kwargs):
        raise RedisConnectionError("Redis is unreachable")

    opens = {"open_buffer": failing_open, "open_buffer_redis": redis_failing_open}
    if failing == "turn_index":
        monkeypatch.setattr(turn_index, "record_turn", failing_record)
    with pytest.raises(refusal):
        await RunRegistry().start(
            conversation_id="failed", turn_id=0, open_buffer=opens.get(failing, _opens("failed", 0)),
            produce=_answered,
        )

    assert await fake_redis.exists(claim_key("failed"), turn_claim.CLAIMS_KEY) == 0


def _built_handlers(monkeypatch) -> list[object]:
    built: list[object] = []

    class _Handler:
        def __init__(self, **kwargs):
            built.append(kwargs)
            raise AssertionError("the request got past the early refusal")

    monkeypatch.setattr(responses_ep, "ResponsesHandler", _Handler)
    return built


@pytest.mark.asyncio
async def test_a_send_into_a_turn_another_replica_runs_is_refused_before_the_routing_gate(
    client, fake_redis, monkeypatch, caplog, owned_logger,
):
    """Refused before any database read or model call, with the 409 body the
    web UI already handles for a turn on the same replica, and one WARNING
    naming the run and pod that hold the claim."""
    logged = owned_logger(REGISTRY_LOGGER, level=logging.WARNING)
    conversation_id = str(uuid4())
    await fake_redis.set(claim_key(conversation_id), _foreign_claim(turn_id=2), px=70_000)
    await _running_turn(conversation_id, 2, "corr-elsewhere")
    built = _built_handlers(monkeypatch)

    resp = client.post(
        "/api/v1/responses/", json={"input": "again", "stream": True, "conversation": conversation_id},
    )

    assert resp.status_code == 409
    assert resp.json() == REFUSAL
    assert built == []
    fields = {
        "event": "turn_claim_held", "conversation_id": conversation_id, "turn_id": 2,
        "holder_run_id": "run-elsewhere", "holder_pod": "other-pod",
    }
    records = _refusal_records(caplog, "turn_claim_held")
    assert [{name: getattr(r, name) for name in fields} for r in records] == [fields]
    assert_log_fields(logged.output(), **fields)


@pytest.mark.asyncio
async def test_a_send_redis_cannot_answer_for_is_a_503_with_retry_after(client, fake_redis, monkeypatch):
    monkeypatch.setattr(turn_claim, "get_redis", lambda: UnreachableRedis())
    built = _built_handlers(monkeypatch)

    resp = client.post(
        "/api/v1/responses/", json={"input": "hi", "stream": True, "conversation": str(uuid4())},
    )

    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "2"
    assert resp.json() == {"detail": "Cowork is busy. Try again in about 2 seconds.", "code": "server_busy"}
    assert built == []


@pytest.mark.asyncio
async def test_the_early_refusal_counts_another_replicas_turn_only_for_its_owner(fake_redis):
    """Where conversations are partitioned the claim alone names no owner;
    the turn index entry its run wrote does, so another user's turn is never
    revealed by a refusal."""
    conversation_id = str(uuid4())
    await fake_redis.set(claim_key(conversation_id), _foreign_claim(turn_id=1), px=70_000)
    await turn_index.record_turn(
        conversation_id, turn_id=1, correlation_id="corr-elsewhere", run_id="run-elsewhere",
        org_id="org", user_id="alice",
    )
    await RedisStreamBuffer(conversation_id=conversation_id, turn_id=1).append("sse", _sse("hi"))

    await responses_ep._refuse_while_answering(
        conversation=conversation_id, scope=TenantScope(org_mode=True, org_id="org", user_id="bob"),
    )
    with pytest.raises(TurnInProgress):
        await responses_ep._refuse_while_answering(
            conversation=conversation_id, scope=TenantScope(org_mode=True, org_id="org", user_id="alice"),
        )

    # An entry another run wrote (an earlier turn, or a holder that has not
    # published yet) says nothing about who owns the claim's run, even when
    # it names the caller: start() decides that request.
    await turn_index.record_turn(
        conversation_id, turn_id=0, correlation_id="corr-earlier", run_id="run-earlier",
        org_id="org", user_id="alice",
    )
    await responses_ep._refuse_while_answering(
        conversation=conversation_id, scope=TenantScope(org_mode=True, org_id="org", user_id="alice"),
    )


@pytest.mark.asyncio
async def test_a_stale_local_handle_after_a_truncation_elsewhere_does_not_refuse(fake_redis):
    """A start naming the number of a turn that saved its question read the
    history before that save, and is refused on every replica. Once another
    replica truncates the conversation, its delete forgets the turn index
    entry, while this replica keeps its finished handle: that handle used to
    refuse every send at its number for as long as the process lived."""
    replica_a, replica_b = RunRegistry(), RunRegistry()
    ended = await replica_a.start(
        conversation_id="trunc", turn_id=3, open_buffer=_opens("trunc", 3), produce=_answered,
        lifecycle=TurnLifecycle(saved_question=True, question_id=uuid4()),
    )
    await asyncio.wait_for(ended.task, timeout=2)
    for runs in (replica_a, replica_b):
        with pytest.raises(TurnInProgress):
            await runs.start(
                conversation_id="trunc", turn_id=3, open_buffer=_opens("trunc", 3), produce=_answered,
            )

    # Replica B deletes the turn: discard_conversation forgets the index there.
    await turn_index.forget_turn("trunc")

    admitted = await replica_a.start(
        conversation_id="trunc", turn_id=3, open_buffer=_opens("trunc", 3), produce=_answered,
    )
    assert admitted is not ended
    await asyncio.wait_for(admitted.task, timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "elsewhere_turn,elsewhere_saved,refused_at",
    [(1, False, None), (3, False, None), (5, True, 5)],
    ids=["truncated", "same-number-saved-nothing", "moved-on"],
)
async def test_a_stale_local_handle_defers_to_the_entry_another_replicas_run_wrote(
    fake_redis, elsewhere_turn, elsewhere_saved, refused_at,
):
    """Replica A keeps the finished handle of its turn 3, which saved its
    question. Once replica B's run writes the conversation's index entry,
    that entry decides on every replica, A included. After a truncation on
    B, B's next turn carries a lower number than A's handle, or the same
    number with nothing saved, and every replica admits a start at 3; A's
    handle used to refuse it there alone. A later turn on B that saved its
    question refuses a start at 3 everywhere, by its own stamp."""
    replica_a, replica_b = RunRegistry(), RunRegistry()
    ended = await replica_a.start(
        conversation_id="moved", turn_id=3, open_buffer=_opens("moved", 3), produce=_answered,
        lifecycle=TurnLifecycle(saved_question=True, question_id=uuid4()),
    )
    await asyncio.wait_for(ended.task, timeout=2)
    if elsewhere_turn <= 3:
        await turn_index.forget_turn("moved")
    elsewhere = await replica_b.start(
        conversation_id="moved", turn_id=elsewhere_turn, open_buffer=_opens("moved", elsewhere_turn),
        produce=_answered, lifecycle=TurnLifecycle(saved_question=elsewhere_saved),
    )
    await asyncio.wait_for(elsewhere.task, timeout=2)

    for runs in (replica_a, replica_b):
        start = runs.start(
            conversation_id="moved", turn_id=3, open_buffer=_opens("moved", 3), produce=_answered,
        )
        if refused_at is not None:
            with pytest.raises(TurnInProgress) as caught:
                await start
            assert caught.value.turn_id == refused_at
            continue
        admitted = await start
        await asyncio.wait_for(admitted.task, timeout=2)


@pytest.mark.asyncio
async def test_a_channel_turn_does_not_replace_a_claimed_turns_index_entry(fake_redis):
    """Cancel and answer resolve a turn through its index entry. A channel
    turn on the same conversation used to replace it, so they reached the
    channel turn instead of the web turn holding the claim."""
    runs = RunRegistry()
    started, release = asyncio.Event(), asyncio.Event()
    handle = await runs.start(
        conversation_id="chan", turn_id=1, open_buffer=_opens("chan", 1), produce=_waiting(started, release),
    )
    await asyncio.wait_for(started.wait(), timeout=2)

    wrote = await turn_index.record_channel_turn(
        conversation_id="chan", turn_id=0, correlation_id="corr-channel", org_id=None, user_id=None,
    )

    assert wrote is False
    assert (await turn_index.get_turn("chan")).run_id == handle.run_id
    release.set()
    await asyncio.wait_for(handle.task, timeout=2)
    assert await turn_index.record_channel_turn(
        conversation_id="chan", turn_id=0, correlation_id="corr-channel", org_id=None, user_id=None,
    ) is True
    assert (await turn_index.get_turn("chan")).correlation_id == "corr-channel"


# ── the stale-number rule across replicas ────────────────────────────
#
# Two questions that read the history at the same moment compute the same
# turn number. The first saves its question at that number; the second read
# the history before that save, so it is refused on every replica, whenever
# it reaches start(), and never deletes the first turn's stream.


def _saving(
    *, conversation_id: str, lifecycle: TurnLifecycle, final_written: asyncio.Event, release: asyncio.Event,
):
    """A producer that saves its question as the handler does, writes its
    final client frame, then writes its terminal record once ``release`` is
    set."""

    async def produce(buffer):
        await mark_question_saved(lifecycle=lifecycle, conversation_id=conversation_id)
        await buffer.append("sse", _sse("A-PARTIAL"))
        await buffer.append("sse", COMPLETED)
        final_written.set()
        await release.wait()
        await buffer.close("completed")

    return produce


async def _stream_records(fake_redis, *, conversation_id: str, turn_id: int) -> list[str]:
    """Each record's SSE text, or its type for a record with none."""
    records = [
        RedisStreamBuffer._record(fields)
        for _id, fields in await fake_redis.xrange(f"cowork:stream:{conversation_id}:{turn_id}")
    ]
    return [(record.data or {}).get("sse", record.type) for record in records]


@pytest.mark.asyncio
async def test_a_stale_send_that_waited_for_a_finishing_turn_on_another_replica_is_refused(fake_redis):
    """Replica B's start read the history before replica A's turn saved its
    question at turn 2, and reached start() while A wrote its final frame.
    The early stale check found no stamp yet, so B waited for A, and used to
    be admitted at turn 2 once A released: its acquire deleted A's ended
    stream and wrote its own records there. The acquire now reads the stamp
    A's release writes, in the transaction that takes the claim.

    The lifecycle starts with saved_question set, so the save leaves no stamp
    of its own: only the release's stamp can refuse B here."""
    replica_a, replica_b = RunRegistry(), RunRegistry()
    final_written, release = asyncio.Event(), asyncio.Event()

    async def finishing(buffer):
        await buffer.append("sse", _sse("A-PARTIAL"))
        await buffer.append("sse", COMPLETED)
        final_written.set()
        await release.wait()
        await buffer.close("completed")

    first = await replica_a.start(
        conversation_id="fin", turn_id=2, open_buffer=_opens("fin", 2), produce=finishing,
        lifecycle=TurnLifecycle(saved_question=True, question_id=uuid4()),
    )
    await asyncio.wait_for(final_written.wait(), timeout=2)
    opened: list[int] = []
    stale = asyncio.create_task(replica_b.start(
        conversation_id="fin", turn_id=2, open_buffer=_opens("fin", 2, opened), produce=_answered,
    ))
    await asyncio.sleep(0.3)
    assert not stale.done(), "the stale send must have reached the wait for the finishing turn"
    release.set()
    await asyncio.wait_for(first.task, timeout=2)

    with pytest.raises(TurnInProgress) as refused:
        await asyncio.wait_for(stale, timeout=5)

    assert refused.value.turn_id == 2
    assert opened == []
    assert await _stream_records(fake_redis, conversation_id="fin", turn_id=2) == [
        _sse("A-PARTIAL")["sse"], COMPLETED["sse"], "Done",
    ]
    assert await fake_redis.exists(claim_key("fin")) == 0
    # The next question reads the history with A's question in it.
    after = await replica_b.start(
        conversation_id="fin", turn_id=3, open_buffer=_opens("fin", 3), produce=_answered,
    )
    await asyncio.wait_for(after.task, timeout=2)


@pytest.mark.asyncio
async def test_a_stale_send_over_another_replicas_ended_turn_waits_for_its_release_and_is_refused(
    fake_redis, monkeypatch,
):
    """Replica A's turn at 2 has written its terminal record, and its release
    (which stamps the saved question first) has not run yet. Replica B used to
    delete A's claim at once and take it, with no stamp to refuse it. It now
    waits the bounded time for A's own release, and that release's stamp
    refuses it."""
    registry_mod = importlib.import_module("cowork.streaming.registry")
    gate = asyncio.Event()
    real_end_claim = registry_mod._end_claim

    async def held_end_claim(**kwargs):
        await gate.wait()
        await real_end_claim(**kwargs)

    monkeypatch.setattr(registry_mod, "_end_claim", held_end_claim)
    replica_a, replica_b = RunRegistry(), RunRegistry()
    first = await replica_a.start(
        conversation_id="end", turn_id=2, open_buffer=_opens("end", 2), produce=_answered,
        lifecycle=TurnLifecycle(saved_question=True, question_id=uuid4()),
    )

    async def ended() -> bool:
        return (await _stream_records(fake_redis, conversation_id="end", turn_id=2))[-1:] == ["Done"]

    while not await ended():
        await asyncio.sleep(0.01)
    assert await fake_redis.exists(claim_key("end")) == 1
    opened: list[int] = []
    stale = asyncio.create_task(replica_b.start(
        conversation_id="end", turn_id=2, open_buffer=_opens("end", 2, opened), produce=_answered,
    ))
    await asyncio.sleep(0.3)
    assert not stale.done(), "the stale send must wait for the ended turn's own release"
    gate.set()
    await asyncio.wait_for(first.task, timeout=2)

    with pytest.raises(TurnInProgress) as refused:
        await asyncio.wait_for(stale, timeout=5)

    assert refused.value.turn_id == 2
    assert opened == []
    assert await _stream_records(fake_redis, conversation_id="end", turn_id=2) == [COMPLETED["sse"], "Done"]


@pytest.mark.asyncio
async def test_a_turn_stamps_its_saved_question_when_it_saves_it(fake_redis, monkeypatch, caplog, owned_logger):
    """The stamp lands with the save, while the turn still answers, so a
    release whose own stamp fails still leaves the conversation refusing a
    stale send at that turn's number on every replica, each with one WARNING
    naming the turn whose saved question refused it."""
    logged = owned_logger(REGISTRY_LOGGER, level=logging.WARNING)
    replica_a, replica_b = RunRegistry(), RunRegistry()
    lifecycle = TurnLifecycle(question_id=uuid4())
    final_written, release = asyncio.Event(), asyncio.Event()
    first = await replica_a.start(
        conversation_id="early", turn_id=2, open_buffer=_opens("early", 2),
        produce=_saving(conversation_id="early", lifecycle=lifecycle, final_written=final_written, release=release),
        lifecycle=lifecycle,
    )
    await asyncio.wait_for(final_written.wait(), timeout=2)

    entry = await turn_index.get_turn("early")
    assert (entry.run_id, entry.saved_question, entry.released_at) == (first.run_id, True, None)

    async def failing_stamp(**_kwargs):
        raise RedisConnectionError("Redis is unreachable")

    monkeypatch.setattr(turn_index, "stamp_question", failing_stamp)
    release.set()
    await asyncio.wait_for(first.task, timeout=2)
    assert await fake_redis.exists(claim_key("early")) == 0

    for runs in (replica_a, replica_b):
        with pytest.raises(TurnInProgress) as refused:
            await runs.start(
                conversation_id="early", turn_id=2, open_buffer=_opens("early", 2), produce=_answered,
            )
        assert refused.value.turn_id == 2
    fields = {"event": "turn_stale_refused", "conversation_id": "early", "turn_id": 2, "saved_turn_id": 2}
    records = _refusal_records(caplog, "turn_stale_refused")
    assert [{name: getattr(r, name) for name in fields} for r in records] == [fields, fields]
    assert_log_fields(logged.output(), **fields)


@pytest.mark.asyncio
async def test_a_remote_turns_producer_stamps_the_turn_index_as_it_saves_the_question(fake_redis, monkeypatch):
    """The handler's own save, not a test producer's: the remote producer
    saves the question in its first write, and the stamp lands then, while
    the turn still answers. Set only in this process's memory, a release
    whose stamp failed would leave every other replica admitting a start
    that read the history before the save."""
    import cowork.handlers.responses as handlers_mod
    from test_responses_remote_backend import _FakeBuffer, _remote_handler

    handler = _remote_handler(monkeypatch, {})
    asked, release = asyncio.Event(), asyncio.Event()

    async def replies_once_released(**_kwargs):
        asked.set()
        await release.wait()
        yield "turn_completed", {}

    monkeypatch.setattr(handlers_mod, "stream_remote_replies", replies_once_released)
    conversation_id = uuid4()
    lifecycle = TurnLifecycle()
    await turn_index.record_turn(
        str(conversation_id), turn_id=2, correlation_id=lifecycle.identity.correlation_id,
        org_id=None, user_id=None, run_id=lifecycle.identity.run_id,
    )
    producing = asyncio.create_task(handler._produce_remote(
        conv_id=conversation_id, input_text="hi", original_content="hi", model="anton", harness_id="anton",
        buffer=_FakeBuffer(), turn_id=2, lifecycle=lifecycle,
    ))
    try:
        await asyncio.wait_for(asked.wait(), timeout=5)
        entry = await turn_index.get_turn(str(conversation_id))
        assert (entry.run_id, entry.saved_question, entry.released_at) == (lifecycle.identity.run_id, True, None)
    finally:
        release.set()
        await asyncio.wait_for(producing, timeout=5)


@pytest.mark.asyncio
async def test_an_unreadable_turn_index_refuses_the_start_as_unavailable(fake_redis, monkeypatch):
    """Could not read the turn index is not "no entry": read as none, a
    stale start would be admitted at the number of a turn that saved its
    question. Only the index is unreachable here; the claim reads fine."""
    monkeypatch.setattr(turn_index, "get_redis", lambda: UnreachableRedis())
    opened: list[int] = []

    with pytest.raises(TurnClaimUnavailable):
        await RunRegistry().start(
            conversation_id="index-down", turn_id=2, open_buffer=_opens("index-down", 2, opened),
            produce=_answered,
        )

    assert opened == []
    assert await fake_redis.exists(claim_key("index-down"), turn_claim.CLAIMS_KEY) == 0


@pytest.mark.asyncio
async def test_a_turn_index_entry_no_reader_can_parse_refuses_the_start_as_unavailable(fake_redis):
    """An entry missing its turn number is parsed in the same bounded step
    that reads the hash, so it refuses the start with 503 and Retry-After, as
    an unreadable index does. Parsed after that step, it raised a KeyError: a
    500."""
    await fake_redis.hset(turn_index.turn_key("torn"), mapping={"correlation_id": "corr-torn"})
    opened: list[int] = []

    with pytest.raises(TurnClaimUnavailable):
        await RunRegistry().start(
            conversation_id="torn", turn_id=2, open_buffer=_opens("torn", 2, opened), produce=_answered,
        )

    assert opened == []
