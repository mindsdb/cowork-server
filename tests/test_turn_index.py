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
