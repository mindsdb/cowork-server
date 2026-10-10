"""The turn index: which turn is current for a conversation, readable anywhere."""
import fakeredis.aioredis
import pytest

from cowork.streaming import turn_index


@pytest.fixture
def fake_redis(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(turn_index, "get_redis", lambda: client)
    return client


@pytest.mark.asyncio
async def test_records_and_reads_back_a_turn(fake_redis):
    await turn_index.record_turn(
        "c1", turn_id=4, correlation_id="corr-1", org_id="o1", user_id="u1")

    turn = await turn_index.get_turn("c1")
    assert int(turn["turn_id"]) == 4
    assert turn["correlation_id"] == "corr-1"
    assert turn["org_id"] == "o1"


@pytest.mark.asyncio
async def test_a_new_turn_replaces_the_previous_one(fake_redis):
    """A conversation has at most one current turn, and the next one wins.
    No claim and no lock: the controller serialises execution per conversation."""
    await turn_index.record_turn(
        "c1", turn_id=1, correlation_id="corr-1", org_id=None, user_id=None)
    await turn_index.record_turn(
        "c1", turn_id=2, correlation_id="corr-2", org_id=None, user_id=None)

    turn = await turn_index.get_turn("c1")
    assert int(turn["turn_id"]) == 2
    assert turn["correlation_id"] == "corr-2"


@pytest.mark.asyncio
async def test_unknown_conversation_reads_as_none(fake_redis):
    assert await turn_index.get_turn("nope") is None


@pytest.mark.asyncio
async def test_forget_removes_the_entry(fake_redis):
    await turn_index.record_turn(
        "c1", turn_id=1, correlation_id="corr-1", org_id=None, user_id=None)
    await turn_index.forget_turn("c1")

    assert await turn_index.get_turn("c1") is None
    assert await fake_redis.sismember("cowork:turns", "c1") == 0


@pytest.mark.asyncio
async def test_list_prunes_members_whose_hash_expired(fake_redis):
    """The set has no TTL of its own, so an expired hash leaves a member."""
    await turn_index.record_turn(
        "c1", turn_id=1, correlation_id="corr-1", org_id=None, user_id=None)
    await fake_redis.delete("cowork:turn:c1")   # stands in for TTL expiry

    assert await turn_index.list_turns() == []
    assert await fake_redis.sismember("cowork:turns", "c1") == 0


@pytest.mark.asyncio
async def test_list_returns_every_recorded_turn(fake_redis):
    await turn_index.record_turn(
        "c1", turn_id=1, correlation_id="corr-1", org_id="o1", user_id=None)
    await turn_index.record_turn(
        "c2", turn_id=1, correlation_id="corr-2", org_id="o2", user_id=None)

    turns = {t["conversation_id"]: t for t in await turn_index.list_turns()}
    assert set(turns) == {"c1", "c2"}
    assert turns["c2"]["org_id"] == "o2"


@pytest.mark.asyncio
async def test_list_pairs_each_turn_with_its_own_conversation_and_prunes_only_expired_ones(fake_redis):
    for n in range(6):
        await turn_index.record_turn(
            f"c{n}", turn_id=n, correlation_id=f"corr-{n}", org_id=f"o{n}", user_id=None)
    for gone in ("c1", "c4"):
        await fake_redis.delete(f"cowork:turn:{gone}")   # stands in for TTL expiry

    turns = await turn_index.list_turns()

    assert sorted((t["conversation_id"], t["turn_id"], t["correlation_id"]) for t in turns) == [
        (f"c{n}", str(n), f"corr-{n}") for n in (0, 2, 3, 5)
    ]
    assert await fake_redis.smembers("cowork:turns") == {"c0", "c2", "c3", "c5"}


@pytest.mark.asyncio
async def test_list_of_no_turns_removes_nothing(fake_redis, monkeypatch):
    async def no_srem(*_args):
        raise AssertionError("SREM with no members is a Redis error")

    monkeypatch.setattr(fake_redis, "srem", no_srem)

    assert await turn_index.list_turns() == []


@pytest.mark.asyncio
async def test_list_reads_every_turn_in_one_round_trip(fake_redis, monkeypatch):
    """/in-flight-list runs this every few seconds per open client, so it
    reads every member's hash in one pipeline, not one call each."""
    for n in range(3):
        await turn_index.record_turn(
            f"c{n}", turn_id=n, correlation_id=f"corr-{n}", org_id=None, user_id=None)

    async def per_member(*_args):
        raise AssertionError("list_turns must not read one hash per call")

    monkeypatch.setattr(fake_redis, "hgetall", per_member)

    assert {t["conversation_id"] for t in await turn_index.list_turns()} == {"c0", "c1", "c2"}


@pytest.mark.asyncio
async def test_a_turn_is_recorded_in_one_transaction(fake_redis, monkeypatch):
    """Sent command by command, the DELETE leaves the hash empty until the
    HSET lands, and a /in-flight-list read in that gap sees an expired turn."""
    pipeline = fake_redis.pipeline
    transactions = []

    def recorded_pipeline(*args, transaction=True, **kwargs):
        transactions.append(transaction)
        return pipeline(*args, transaction=transaction, **kwargs)

    async def separate_call(*_args, **_kwargs):
        raise AssertionError("record_turn must write in one MULTI, not one command at a time")

    monkeypatch.setattr(fake_redis, "pipeline", recorded_pipeline)
    for command in ("delete", "hset", "expire", "sadd"):
        monkeypatch.setattr(fake_redis, command, separate_call)

    await turn_index.record_turn(
        "c1", turn_id=3, correlation_id="corr-3", org_id=None, user_id=None)

    assert transactions == [True]
    assert (await turn_index.get_turn("c1"))["correlation_id"] == "corr-3"
    assert 0 < await fake_redis.ttl("cowork:turn:c1") <= turn_index.TURN_INDEX_TTL_SECONDS
    assert await fake_redis.sismember("cowork:turns", "c1") == 1


def _run_once_after(fake_redis, monkeypatch, *, in_transaction: bool, command: str, step):
    """Run `step` once, right after the first ``command`` awaited on a
    pipeline with ``transaction=in_transaction`` returns."""
    pipeline = fake_redis.pipeline
    pending = [step]

    def hooked_pipeline(*args, transaction=True, **kwargs):
        pipe = pipeline(*args, transaction=transaction, **kwargs)
        if transaction == in_transaction:
            original = getattr(pipe, command)

            async def then_step(*command_args, **command_kwargs):
                reply = await original(*command_args, **command_kwargs)
                if pending:
                    await pending.pop()()
                return reply

            setattr(pipe, command, then_step)
        return pipe

    monkeypatch.setattr(fake_redis, "pipeline", hooked_pipeline)


@pytest.mark.asyncio
async def test_a_turn_recorded_after_the_read_keeps_its_member(fake_redis, monkeypatch):
    """A turn starts on a conversation whose last hash expired, just after
    list_turns read that hash. Its SADD is a no-op on a member still in the
    set, so a prune that removed the member anyway would hide the running
    turn from /in-flight-list until the conversation's next turn."""
    for conversation_id in ("c1", "c2"):
        await turn_index.record_turn(
            conversation_id, turn_id=1, correlation_id=f"old-{conversation_id}", org_id=None, user_id=None)
        await fake_redis.delete(f"cowork:turn:{conversation_id}")   # stands in for TTL expiry

    async def c1_starts_a_turn():
        await turn_index.record_turn("c1", turn_id=2, correlation_id="new-c1", org_id=None, user_id=None)

    _run_once_after(fake_redis, monkeypatch, in_transaction=False, command="execute", step=c1_starts_a_turn)

    assert await turn_index.list_turns() == []
    assert await fake_redis.sismember("cowork:turns", "c1") == 1
    # The prune that skipped c1 left c2 too; the next list removes it.
    assert [t["correlation_id"] for t in await turn_index.list_turns()] == ["new-c1"]
    assert await fake_redis.smembers("cowork:turns") == {"c1"}


@pytest.mark.asyncio
async def test_a_turn_recorded_after_the_prune_checked_its_hash_keeps_its_member(fake_redis, monkeypatch):
    """The prune checks that the hash is still missing before it removes the
    member. Its WATCH aborts the removal when a turn is recorded between the
    check and the SREM."""
    await turn_index.record_turn("c1", turn_id=1, correlation_id="old-c1", org_id=None, user_id=None)
    await fake_redis.delete("cowork:turn:c1")   # stands in for TTL expiry

    async def c1_starts_a_turn():
        await turn_index.record_turn("c1", turn_id=2, correlation_id="new-c1", org_id=None, user_id=None)

    _run_once_after(fake_redis, monkeypatch, in_transaction=True, command="exists", step=c1_starts_a_turn)

    assert await turn_index.list_turns() == []
    assert await fake_redis.sismember("cowork:turns", "c1") == 1
    assert [t["correlation_id"] for t in await turn_index.list_turns()] == ["new-c1"]


def test_discard_conversation_forgets_the_turn(monkeypatch, tmp_path):
    """Truncating a conversation deletes its buffers. Leaving the index entry
    behind would have /in-flight keep naming a turn with nothing behind it."""
    import fakeredis
    from fakeredis import FakeServer

    from cowork.streaming import backend as backend_mod
    from cowork.streaming import discard_conversation

    server = FakeServer()
    sync_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(turn_index, "get_sync_redis", lambda: sync_client)
    monkeypatch.setattr(backend_mod, "get_sync_redis", lambda: sync_client)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")

    sync_client.hset("cowork:turn:c1", mapping={"turn_id": "1", "correlation_id": "corr-1"})
    sync_client.sadd("cowork:turns", "c1")

    discard_conversation("c1")

    assert sync_client.exists("cowork:turn:c1") == 0
    assert sync_client.sismember("cowork:turns", "c1") == 0


def test_forget_sync_survives_a_redis_that_is_down(monkeypatch):
    """Conversation delete is best effort and must not fail on this."""
    def boom():
        raise ConnectionError("redis down")
    monkeypatch.setattr(turn_index, "get_sync_redis", boom)

    turn_index.forget_turn_sync("c1")
