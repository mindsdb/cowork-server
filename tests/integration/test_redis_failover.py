"""cowork-server's Redis clients against a real Redis that fails the way a failover does.

A connection the server kills mid-call, a primary demoted to a read-only
replica (REPLICAOF), a password changed under a live client, a connection
closed during its handshake, and a reply lost after Redis applied the write.
The last two need a TCP proxy between the client and Redis, which this module
starts in-process; it also delays new connections and stops accepting them.

These tests run only against the Redis ``COWORK_TEST_REDIS_URL`` names, never a
default one, since they change the whole server: REPLICAOF, ACL users, CLIENT
KILL and FLUSHDB. Each test that changes the server (REPLICAOF, an ACL user)
puts it back in a ``finally``, since the service job runs every test against
one Redis.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from urllib.parse import urlsplit

import pytest
import redis
import redis.asyncio as aioredis
from redis.exceptions import AuthenticationError
from redis.exceptions import ConnectionError as RedisConnectionError

from cowork.streaming import backend as backend_mod
from cowork.streaming.buffer import RedisStreamBuffer, stream_key
from cowork.turnqueue import redis_client
from cowork.turnqueue.redis_client import build_async_redis, build_sync_redis, send_once, xread_blocking
from tests.integration.prereq import missing_prerequisite, report_redis_version

pytestmark = [pytest.mark.integration, pytest.mark.redis_service]

ACL_USER = "cowork-failover-test"


@pytest.fixture
async def redis_url(monkeypatch, pytestconfig):
    url = os.environ.get("COWORK_TEST_REDIS_URL")
    if not url:
        missing_prerequisite(
            "COWORK_TEST_REDIS_URL is not set, and these tests change the whole Redis they run on",
            required_by="COWORK_REQUIRE_SERVICES",
        )
    admin = aioredis.from_url(url, decode_responses=True)
    try:
        await admin.ping()
    except Exception:
        missing_prerequisite(f"no Redis reachable at {url}", required_by="COWORK_REQUIRE_SERVICES")
    await report_redis_version(client=admin, config=pytestconfig)
    await admin.flushdb()
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    monkeypatch.setenv("COWORK_TURN_REDIS_URL", url)
    redis_client.reset_redis()
    yield url
    await admin.aclose()
    redis_client.reset_redis()


@pytest.fixture
async def admin(redis_url):
    """A plain redis-py client, with none of the retries under test."""
    client = aioredis.from_url(redis_url, decode_responses=True)
    yield client
    await client.aclose()


def _with_netloc(*, url: str, host: str, port: int, credentials: str | None = None) -> str:
    """``url`` aimed at ``host:port``, keeping its database."""
    netloc = f"{credentials}@{host}:{port}" if credentials else f"{host}:{port}"
    return urlsplit(url)._replace(netloc=netloc).geturl()


class _Proxy:
    """A TCP proxy in front of the test Redis.

    ``lose_next_reply`` drops the next reply Redis sends and closes that
    connection, after Redis has applied the command. ``connect_delay_s`` holds
    each new connection that long before it reaches Redis. The next command
    whose bytes contain ``drop_marker`` is held for ``drop_after_s``, then the
    proxy stops accepting connections and closes that one without sending it.
    The next command whose bytes contain ``cut_marker`` closes its connection
    without sending it, and later connections pass. ``refuse`` stops accepting
    connections, and ``listen`` starts again on the same port.
    """

    def __init__(self, *, upstream_url: str) -> None:
        parts = urlsplit(upstream_url)
        self._upstream = (parts.hostname or "localhost", parts.port or 6379)
        self.lose_next_reply = False
        self.connect_delay_s = 0.0
        self.drop_marker: bytes | None = None
        self.drop_after_s = 0.0
        self.cut_marker: bytes | None = None
        self.connections = 0
        self.port = 0
        self._server: asyncio.Server | None = None
        self._handlers: set[asyncio.Task] = set()

    async def listen(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        self.port = self._server.sockets[0].getsockname()[1]

    def refuse(self) -> None:
        self._server.close()

    async def close(self) -> None:
        self._server.close()
        for task in self._handlers:
            task.cancel()
        await asyncio.gather(*self._handlers, return_exceptions=True)

    async def _handle(self, client_reader, client_writer) -> None:
        self._handlers.add(asyncio.current_task())
        self.connections += 1
        upstream_writer = None
        try:
            if self.connect_delay_s:
                await asyncio.sleep(self.connect_delay_s)
            upstream_reader, upstream_writer = await asyncio.open_connection(*self._upstream)
            await asyncio.gather(
                self._to_redis(client_reader, upstream_writer),
                self._to_client(upstream_reader, client_writer),
            )
        except (ConnectionError, OSError):
            pass  # Either side closed, or a pump ended the connection on purpose.
        finally:
            client_writer.close()
            if upstream_writer is not None:
                upstream_writer.close()

    async def _to_redis(self, client_reader, upstream_writer) -> None:
        while data := await client_reader.read(65536):
            if self.cut_marker is not None and self.cut_marker in data:
                self.cut_marker = None
                raise ConnectionResetError("command cut")
            if self.drop_marker is not None and self.drop_marker in data:
                self.drop_marker = None
                await asyncio.sleep(self.drop_after_s)
                self.refuse()
                raise ConnectionResetError("command dropped")
            upstream_writer.write(data)
            await upstream_writer.drain()
        raise ConnectionResetError("client closed")

    async def _to_client(self, upstream_reader, client_writer) -> None:
        while data := await upstream_reader.read(65536):
            if self.lose_next_reply:
                self.lose_next_reply = False
                raise ConnectionResetError("reply lost")
            client_writer.write(data)
            await client_writer.drain()
        raise ConnectionResetError("redis closed")


@pytest.fixture
async def proxy(redis_url):
    proxy = _Proxy(upstream_url=redis_url)
    await proxy.listen()
    proxy.url = _with_netloc(url=redis_url, host="127.0.0.1", port=proxy.port)
    yield proxy
    await proxy.close()


async def _blocked_client_id(*, admin, command: str, other_than: str | None = None, user: str | None = None) -> str:
    """The id of the connection blocked in ``command``, once there is one."""
    for _ in range(250):
        for client in await admin.client_list():
            if client.get("cmd") != command or client["id"] == other_than:
                continue
            if user is not None and client.get("user") != user:
                continue
            return client["id"]
        await asyncio.sleep(0.02)
    raise AssertionError(f"no connection blocked in {command}")


async def test_a_call_whose_connection_is_killed_mid_call_is_retried_and_succeeds(redis_url, admin):
    client = build_async_redis(url=redis_url, call_budget_s=10)
    read = asyncio.create_task(client.xread({"killed:s": "0-0"}, block=redis_client.REDIS_BLOCK_MS))
    killed = await _blocked_client_id(admin=admin, command="xread")

    await admin.client_kill_filter(_id=killed)
    # The retry reconnects and blocks again, on a new connection.
    await _blocked_client_id(admin=admin, command="xread", other_than=killed)
    await admin.xadd("killed:s", {"k": "v"})

    reply = await asyncio.wait_for(read, timeout=10)
    assert reply[0][1][0][1] == {"k": "v"}
    await client.aclose()


async def test_a_write_to_a_demoted_primary_is_retried_until_it_is_promoted_again(redis_url, admin):
    """REPLICAOF makes this Redis a replica of an unreachable primary, so it
    answers every write READONLY until REPLICAOF NO ONE, as a demoted primary
    does until the Service points at the new one. A write sent once is retried
    too: READONLY means Redis refused it."""
    client = build_async_redis(url=redis_url, call_budget_s=10)
    reader = asyncio.create_task(xread_blocking(r=client, streams={"ro:s": "0-0"}))
    await _blocked_client_id(admin=admin, command="xread")

    async def write_once():
        with send_once():
            return await client.xadd("ro:jobs", {"payload": "job"})

    try:
        await admin.execute_command("REPLICAOF", "10.255.255.1", "6379")
        # The blocked reader was answered -UNBLOCKED, read as nothing read.
        assert await asyncio.wait_for(reader, timeout=5) == []

        demoted_at = time.monotonic()
        written = asyncio.gather(client.set("ro:key", "v"), write_once())
        await asyncio.sleep(1.0)
        assert not written.done()
        await admin.execute_command("REPLICAOF", "NO", "ONE")
        set_reply, job_id = await asyncio.wait_for(written, timeout=10)
    finally:
        await admin.execute_command("REPLICAOF", "NO", "ONE")

    assert set_reply is True
    assert job_id
    assert time.monotonic() - demoted_at < 10
    assert await admin.get("ro:key") == "v"
    assert await admin.xlen("ro:jobs") == 1
    await client.aclose()


async def test_an_auth_error_on_a_reconnect_is_raised_without_retrying(redis_url, admin):
    """A dropped connection is retried, and the reconnect meets a changed
    password. redis-py makes the AuthenticationError a ConnectionError, which
    the retry list names."""
    await admin.execute_command("ACL", "SETUSER", ACL_USER, "on", ">first-password", "~*", "&*", "+@all")
    try:
        parts = urlsplit(redis_url)
        user_url = _with_netloc(
            url=redis_url, host=parts.hostname, port=parts.port or 6379, credentials=f"{ACL_USER}:first-password",
        )
        client = build_async_redis(url=user_url, call_budget_s=5)
        read = asyncio.create_task(client.xread({"auth:s": "0-0"}, block=redis_client.REDIS_BLOCK_MS))
        await _blocked_client_id(admin=admin, command="xread", user=ACL_USER)
        denied_before = int((await admin.info("stats"))["acl_access_denied_auth"])

        await admin.execute_command("ACL", "SETUSER", ACL_USER, "resetpass", ">second-password")
        await admin.client_kill_filter(user=ACL_USER)
        killed_at = time.monotonic()
        with pytest.raises(AuthenticationError):
            await asyncio.wait_for(read, timeout=10)

        assert time.monotonic() - killed_at < 2
        assert int((await admin.info("stats"))["acl_access_denied_auth"]) - denied_before == 1
        await client.aclose()
    finally:
        await admin.execute_command("ACL", "DELUSER", ACL_USER)


@pytest.mark.parametrize("client_kind", ["async", "sync"])
@pytest.mark.parametrize("step", [b"AUTH", b"SELECT"], ids=["auth", "select"])
async def test_a_connection_closed_during_its_handshake_is_retried(redis_url, proxy, admin, step, client_kind):
    """redis-py retries only the TCP connect of a new connection. Its handshake
    (AUTH, SELECT) runs outside that loop and before the command's, so a server
    that accepts a connection and closes it mid-handshake, as a restarting
    Redis or a Sentinel CLIENT KILL does, would fail the call at once."""
    credentials = None
    if step == b"AUTH":
        await admin.execute_command("ACL", "SETUSER", ACL_USER, "on", ">handshake-password", "~*", "&*", "+@all")
        credentials = f"{ACL_USER}:handshake-password"
    try:
        url = _with_netloc(url=redis_url, host="127.0.0.1", port=proxy.port, credentials=credentials)
        await admin.set("handshake:key", "v")
        proxy.cut_marker = step
        if client_kind == "async":
            client = build_async_redis(url=url, call_budget_s=5)
            value = await client.get("handshake:key")
            await client.aclose()
        else:
            client = build_sync_redis(url=url, call_budget_s=5)
            # In a thread: the proxy runs on this event loop.
            value = await asyncio.to_thread(client.get, "handshake:key")
            client.close()
    finally:
        if step == b"AUTH":
            await admin.execute_command("ACL", "DELUSER", ACL_USER)

    assert value == "v"
    assert proxy.connections == 2


@pytest.mark.parametrize("client_kind", ["async", "sync"])
async def test_a_wrong_password_on_a_first_connect_is_raised_without_retrying(redis_url, admin, client_kind):
    """The pool retries a new connection's handshake, and AUTH is part of it,
    so the auth exclusion has to hold there as well as in the command's loop."""
    await admin.execute_command("ACL", "SETUSER", ACL_USER, "on", ">right-password", "~*", "&*", "+@all")
    try:
        parts = urlsplit(redis_url)
        url = _with_netloc(
            url=redis_url, host=parts.hostname, port=parts.port or 6379, credentials=f"{ACL_USER}:wrong-password",
        )
        denied_before = int((await admin.info("stats"))["acl_access_denied_auth"])
        started = time.monotonic()
        with pytest.raises(AuthenticationError):
            if client_kind == "async":
                client = build_async_redis(url=url, call_budget_s=5)
                try:
                    await client.get("auth:first")
                finally:
                    await client.aclose()
            else:
                client = build_sync_redis(url=url, call_budget_s=5)
                try:
                    await asyncio.to_thread(client.get, "auth:first")
                finally:
                    client.close()

        assert time.monotonic() - started < 2
        assert int((await admin.info("stats"))["acl_access_denied_auth"]) - denied_before == 1
    finally:
        await admin.execute_command("ACL", "DELUSER", ACL_USER)


async def test_an_append_resent_after_a_lost_reply_reaches_a_reader_once(redis_url, proxy, admin, monkeypatch):
    """The client resends a buffer append whose reply was lost, so Redis holds
    that seq twice. A reader, live or replaying, yields it once."""
    writer = RedisStreamBuffer(conversation_id="dup", turn_id=1)
    await writer.append("sse", {"sse": "event: a\n\n"})
    # The live reader holds the direct client, so the proxy loses only the writer's reply.
    live = RedisStreamBuffer(conversation_id="dup", turn_id=1)
    live_seen = []

    async def follow():
        async for rec in live.tail(from_seq=0):
            live_seen.append(rec.seq)

    following = asyncio.create_task(follow())
    await _blocked_client_id(admin=admin, command="xread")
    monkeypatch.setenv("COWORK_TURN_REDIS_URL", proxy.url)
    redis_client.reset_redis()
    await redis_client.get_redis().ping()

    proxy.lose_next_reply = True
    assert await writer.append("sse", {"sse": "event: b\n\n"}) == 1
    await writer.close("completed")
    await asyncio.wait_for(following, timeout=10)

    stored = [int(fields["seq"]) for _id, fields in await admin.xrange(stream_key("dup", 1))]
    assert stored == [0, 1, 1, 2]
    assert live_seen == [0, 1, 2]
    replayed = [rec.seq async for rec in RedisStreamBuffer(conversation_id="dup", turn_id=1).tail(from_seq=0)]
    assert replayed == [0, 1, 2]


async def test_a_write_sent_once_is_not_resent_after_a_lost_reply(proxy, admin):
    """The job XADD duplicates when resent. Without send_once the client
    resends it, which the first case shows. A transaction sent under
    send_once is not resent either."""
    client = build_async_redis(url=proxy.url, call_budget_s=5)

    await client.ping()
    proxy.lose_next_reply = True
    await client.xadd("once:resent", {"payload": "job"})
    assert await admin.xlen("once:resent") == 2

    await client.ping()
    proxy.lose_next_reply = True
    with pytest.raises(RedisConnectionError), send_once():
        await client.xadd("once:jobs", {"payload": "job"})
    assert await admin.xlen("once:jobs") == 1

    await client.ping()
    proxy.lose_next_reply = True
    with pytest.raises(RedisConnectionError), send_once():
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush("once:transaction", "entry")
            pipe.expire("once:transaction", 60)
            await pipe.execute()
    assert await admin.llen("once:transaction") == 1
    await client.aclose()


async def test_a_reconnect_inside_a_retry_spends_what_is_left_of_the_calls_deadline(proxy):
    """redis-py reconnects inside the command's retry loop, and the connect
    retries in a loop of its own. With a deadline per loop, the reconnect starts
    a new budget after the command spent part of its own."""
    budget_s = 2.0
    client = build_async_redis(url=proxy.url, call_budget_s=budget_s)
    await client.ping()
    proxy.drop_marker = b"deadline:nested"
    proxy.drop_after_s = 0.7

    started = time.monotonic()
    with pytest.raises(RedisConnectionError):
        await client.get("deadline:nested")

    assert time.monotonic() - started < budget_s + 0.4
    await client.aclose()


@pytest.mark.parametrize("client_kind", ["async", "sync"])
async def test_a_call_whose_connect_retried_spends_the_same_deadline_on_its_command(proxy, client_kind):
    """Taking a connection from the pool retries its connect before the
    command's own retry loop starts. With a deadline per loop, a call that
    connected late gets a new budget for its command. The blocking client
    opens the call's deadline around both (_SyncRedis.execute_command)."""
    budget_s = 2.0
    proxy.refuse()
    proxy.drop_marker = b"deadline:late"
    if client_kind == "async":
        client = build_async_redis(url=proxy.url, call_budget_s=budget_s)
        call_coro = client.get("deadline:late")
    else:
        client = build_sync_redis(url=proxy.url, call_budget_s=budget_s)
        # In a thread: the proxy runs on this event loop.
        call_coro = asyncio.to_thread(client.get, "deadline:late")

    started = time.monotonic()
    call = asyncio.create_task(call_coro)
    await asyncio.sleep(0.8)
    await proxy.listen()
    with pytest.raises(RedisConnectionError):
        await call

    assert proxy.connections == 1
    assert time.monotonic() - started < budget_s + 0.4
    if client_kind == "async":
        await client.aclose()
    else:
        client.close()


@pytest.mark.parametrize("client_kind", ["async", "sync"])
@pytest.mark.parametrize("send", ["execute", "watch"])
async def test_a_pipeline_whose_connect_retried_spends_the_same_deadline_on_its_commands(proxy, send, client_kind):
    """A pipeline takes its connection from the pool before its own retry
    loop starts, on execute and on an immediate command such as WATCH. With a
    deadline per loop, a pipeline that connected late gets a new budget for
    its commands. The blocking client's pipeline is the one a conversation
    delete releases a turn claim on (turn_claim.release_sync)."""
    budget_s = 2.0
    proxy.refuse()
    proxy.drop_marker = b"deadline:pipeline"

    if client_kind == "async":
        client = build_async_redis(url=proxy.url, call_budget_s=budget_s)

        async def run():
            async with client.pipeline(transaction=True) as pipe:
                if send == "watch":
                    await pipe.watch("deadline:pipeline")
                else:
                    pipe.get("deadline:pipeline")
                    await pipe.execute()
    else:
        client = build_sync_redis(url=proxy.url, call_budget_s=budget_s)

        def run_blocking():
            with client.pipeline(transaction=True) as pipe:
                if send == "watch":
                    pipe.watch("deadline:pipeline")
                else:
                    pipe.get("deadline:pipeline")
                    pipe.execute()

        async def run():
            # In a thread: the proxy runs on this event loop.
            await asyncio.to_thread(run_blocking)

    started = time.monotonic()
    call = asyncio.create_task(run())
    await asyncio.sleep(0.8)
    await proxy.listen()
    with pytest.raises(RedisConnectionError):
        await call

    assert proxy.connections == 1
    assert time.monotonic() - started < budget_s + 0.4
    if client_kind == "async":
        await client.aclose()
    else:
        client.close()


async def test_an_ack_wait_that_hears_no_verdict_ends_unconfirmed_not_on_a_redis_error(
    redis_url, caplog, owned_logger,
):
    """The client cuts each read's wait to the time left before its deadline,
    and the verdict wait's last read blocks until the wait ends. With its
    deadline at the end of its BLOCK, that read timed out just as Redis
    answered its empty read, and every unconfirmed answer logged a Redis
    error."""
    from cowork.turnqueue import answers

    owned_logger(answers.logger.name, level=logging.WARNING)
    client = build_async_redis(url=redis_url)

    result = await answers.submit_remote_answer(
        conversation_id="ack-quiet", correlation_id="corr-ack-quiet", question_id="ask:1",
        payload={"values": ["pg"]}, r=client, ack_timeout_s=1.0,
    )

    assert result is answers.RemoteAnswerResult.ACCEPTED
    warnings = [
        record.msg for record in caplog.records
        if record.name == answers.logger.name and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1 and warnings[0].startswith("ask_user answer not confirmed")
    await client.aclose()


async def test_a_turn_claim_whose_connection_drops_inside_its_watch_stops_at_its_bound(redis_url, proxy, monkeypatch):
    """The connection drops on the GET after the claim's WATCH, and Redis then
    refuses every connect, as when the primary's pod dies mid-transaction.
    redis-py resets the pipeline and its reset reconnects to send UNWATCH. The
    claim's bound cancels that reconnect, which leaves the pipeline watching,
    so the ``async with`` exit reconnects again, outside the call's deadline,
    for a whole client budget past the bound."""
    from cowork.streaming import turn_claim

    monkeypatch.setattr(turn_claim, "REDIS_CALL_SECONDS", 1)
    monkeypatch.setenv("COWORK_TURN_REDIS_URL", proxy.url)
    redis_client.reset_redis()
    await redis_client.get_redis().ping()
    claim = turn_claim.new_claim(run_id="drop-run", turn_id=1, correlation_id=None)
    proxy.drop_marker = b"$3\r\nGET\r\n"

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            turn_claim._bounded(turn_claim.acquire(claim=claim, conversation_id="claim-drop")), timeout=10,
        )

    assert proxy.drop_marker is None
    assert time.monotonic() - started < turn_claim.REDIS_CALL_SECONDS + 0.5


async def test_callers_that_need_a_connection_do_not_wait_for_each_others_connect(proxy):
    """redis-py 5's pool connects under its lock, so five callers that each
    need a new connection connect one after another."""
    proxy.connect_delay_s = 0.5
    client = build_async_redis(url=proxy.url, call_budget_s=10)

    started = time.monotonic()
    await asyncio.gather(*(client.get(f"concurrent:{i}") for i in range(5)))

    assert time.monotonic() - started < 2 * proxy.connect_delay_s
    assert proxy.connections == 5
    await client.aclose()


@pytest.mark.parametrize("reconnected_by", ["the walk", "another thread"])
async def test_a_conversation_delete_scan_restarts_after_its_connection_is_replaced(
    redis_url, monkeypatch, reconnected_by,
):
    """A SCAN cursor means something only to the server that issued it. After
    a failover the next page can reach the new primary, so the walk starts
    again from cursor 0 instead of skipping keys. The blocking pool shares its
    connections between threadpool threads, so another thread can take the
    dropped connection and reconnect it before the walk's next page goes out."""
    admin = redis.Redis.from_url(redis_url, decode_responses=True)
    client = build_sync_redis(url=redis_url, call_budget_s=5)
    monkeypatch.setattr(backend_mod, "get_sync_redis", lambda: client)
    with admin.pipeline(transaction=False) as pipe:
        for i in range(400):
            pipe.xadd(stream_key("scan-me", i), {"seq": "0"})
            pipe.set(f"other:{i}", "1")
        pipe.execute()

    cursors: list[int] = []
    real_scan = redis_client._SyncRedis.scan

    def scan(self, *args, **kwargs):
        cursors.append(kwargs["cursor"])
        reply = real_scan(self, *args, **kwargs)
        if len(cursors) == 1:
            # The server drops the walk's connection between two pages.
            admin.client_kill_filter(_id=self.client_id())
            if reconnected_by == "another thread":
                # A request on another threadpool thread takes a connection
                # from the pool and reconnects it in its own context.
                other = threading.Thread(target=client.get, args=("other:0",))
                other.start()
                other.join()
        return reply

    monkeypatch.setattr(redis_client._SyncRedis, "scan", scan)
    backend_mod._remove_redis_buffers("scan-me")

    assert cursors[0] == 0 and cursors[1] != 0 and cursors[2] == 0
    assert list(admin.scan_iter(match="cowork:stream:scan-me:*")) == []
    assert len(list(admin.scan_iter(match="other:*"))) == 400
    client.close()
    admin.close()
