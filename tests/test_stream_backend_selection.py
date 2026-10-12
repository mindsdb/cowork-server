"""Which backend the factory builds, and what conversation delete removes."""
import fakeredis
import fakeredis.aioredis
import pytest
from fakeredis import FakeServer

from cowork.streaming import backend as backend_mod
from cowork.streaming import buffer as buffer_mod
from cowork.streaming.buffer import FileStreamBuffer, RedisStreamBuffer, stream_key


@pytest.fixture
def fake_redis(monkeypatch):
    # One server, two clients: the write path is async, the delete path is sync
    # (threadpool thread, no event loop to await on).
    server = FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(buffer_mod, "get_redis", lambda: client)
    monkeypatch.setattr(
        backend_mod, "get_sync_redis",
        lambda: fakeredis.FakeRedis(server=server, decode_responses=True),
    )
    return client


@pytest.mark.asyncio
async def test_redis_backend_selected_by_env(monkeypatch, fake_redis):
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    assert isinstance(await backend_mod.new_buffer("c1", 1), RedisStreamBuffer)


@pytest.mark.asyncio
async def test_file_backend_is_still_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_STREAM_BACKEND", raising=False)
    monkeypatch.setenv("COWORK_STREAMS_DIR", str(tmp_path))
    assert isinstance(await backend_mod.new_buffer("c1", 1), FileStreamBuffer)


@pytest.mark.asyncio
async def test_conversation_delete_removes_the_redis_keys(monkeypatch, fake_redis):
    """The file backend deletes a directory here. Without this the Redis
    backend silently keeps a deleted conversation's turns."""
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    buf = await backend_mod.new_buffer("c7", 1)
    await buf.append("Delta", {"text": "a"})
    assert await fake_redis.exists(stream_key("c7", 1)) == 1

    backend_mod.remove_conversation_buffers("c7")

    assert await fake_redis.exists(stream_key("c7", 1)) == 0


@pytest.mark.asyncio
async def test_delete_falls_through_to_the_file_backend(monkeypatch, tmp_path):
    """One entry point for both backends, so callers do not branch."""
    monkeypatch.delenv("COWORK_STREAM_BACKEND", raising=False)
    monkeypatch.setenv("COWORK_STREAMS_DIR", str(tmp_path))
    buf = await backend_mod.new_buffer("c8", 1)
    await buf.append("Delta", {"text": "a"})
    assert buf.path.is_file()

    backend_mod.remove_conversation_buffers("c8")

    assert not buf.path.exists()


@pytest.mark.asyncio
async def test_a_new_redis_turn_starts_clear_of_an_ended_turn_at_its_number(monkeypatch, fake_redis):
    """A turn that saved no message leaves its stream at the number the next
    turn takes. The next turn's readers must not replay it. A stream with no
    terminal record may still be live on another replica, so it is kept."""
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    ended = RedisStreamBuffer(conversation_id="c9", turn_id=1)
    await ended.append("sse", {"sse": "event: response.failed"})
    await ended.close("error")
    live = RedisStreamBuffer(conversation_id="c9", turn_id=2)
    await live.append("sse", {"sse": "event: response.created"})

    fresh = await backend_mod.new_buffer("c9", 1)
    await fresh.append("sse", {"sse": "event: response.created"})
    await fresh.close("completed")
    await backend_mod.new_buffer("c9", 2)

    assert [(record.seq, record.type) async for record in fresh.tail(0)] == [(0, "sse"), (1, "Done")]
    assert await fake_redis.xlen(stream_key("c9", 2)) == 1


@pytest.mark.asyncio
async def test_a_new_file_turn_starts_clear_of_an_ended_turn_at_its_number(monkeypatch, tmp_path):
    """The file backend's twin: a new turn's file starts empty, so its readers
    see only its own records."""
    monkeypatch.delenv("COWORK_STREAM_BACKEND", raising=False)
    monkeypatch.setenv("COWORK_STREAMS_DIR", str(tmp_path))
    ended = await backend_mod.new_buffer("c10", 1)
    await ended.append("sse", {"sse": "event: response.failed"})
    await ended.close("error")

    fresh = await backend_mod.new_buffer("c10", 1)
    await fresh.append("sse", {"sse": "event: response.created"})
    await fresh.close("completed")

    assert [(record.seq, record.type) async for record in fresh.tail(0)] == [(0, "sse"), (1, "Done")]
