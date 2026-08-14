import asyncio
import json

import pytest

from cowork.turnqueue import producer as prod


@pytest.fixture(autouse=True)
def _stub_llm_mint(monkeypatch):
    """Every produce_remote_turn call now mints a turn key before enqueuing.
    Stub it by default so the pre-existing tests (which only exercise the
    reply-loop/SSE behavior) don't need to know about the llm block; tests
    that care override this explicitly."""

    async def _fake_mint(**kw):
        return "mdb_turnkey"

    monkeypatch.setattr(prod, "mint_turn_key", _fake_mint)


class FakeRedis:
    def __init__(self, replies): self.added = []; self.registered = []; self._replies = replies
    async def sadd(self, key, member): self.registered.append((key, member)); return 1
    async def hset(self, key, mapping=None): return 1
    async def expire(self, key, seconds): return 1
    async def delete(self, *keys): return len(keys)
    async def xadd(self, stream, fields): self.added.append((stream, fields)); return "1-0"
    async def xread(self, streams, count=None, block=None):
        if self._replies:
            stream, fields = self._replies.pop(0)
            return [[stream, [["9-0", fields]]]]
        return None


class RecBuffer:
    def __init__(self): self.records = []; self.closed = None
    async def append(self, type_, data): self.records.append((type_, data)); return len(self.records)
    async def close(self, reason, extra=None): self.closed = reason


def _reply(kind, data):
    return {"payload": json.dumps({"correlation_id": "r", "kind": kind, "data": data})}


@pytest.mark.asyncio
async def test_job_goes_to_the_conversations_own_stream(monkeypatch):
    """One stream per conversation, listed in the registry set. The controller locks a
    conversation before reading its stream, so a busy pod's next job stays undelivered
    instead of blocking every other conversation on one shared stream."""
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=RecBuffer())
    assert fake.added[0][0] == "scratchpad:requests:conv-1"
    # Registered in the controller's queue registry, and in cowork's own turn
    # index so another replica can find this turn.
    assert ("scratchpad:requests:queues", "conv-1") in fake.registered
    assert ("cowork:turns", "conv-1") in fake.registered


@pytest.mark.asyncio
async def test_produce_remote_turn_streams_deltas(monkeypatch):
    replies = [("scratchpad:reply:conv-1", _reply("turn_delta", {"text": "he"})),
               ("scratchpad:reply:conv-1", _reply("turn_delta", {"text": "llo"})),
               ("scratchpad:reply:conv-1", _reply("turn_completed", {}))]
    fake = FakeRedis(replies=replies)
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    buf = RecBuffer()
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=buf,
                                   history=[{"role": "user", "content": "prev"}])
    job = json.loads(fake.added[0][1]["payload"])
    assert job["op"] == "anton_turn"
    assert job["params"]["history"] == [{"role": "user", "content": "prev"}]
    sse = [r[1]["sse"] for r in buf.records]
    assert "response.created" in sse[0]
    assert "he" in sse[1] and "response.output_text.delta" in sse[1]
    assert "llo" in sse[2]
    assert "response.completed" in sse[3]
    assert buf.closed == "completed"


@pytest.mark.asyncio
async def test_turn_failed_renders_as_response_failed(monkeypatch):
    # A failed turn must stream the same response.failed frame the in-process
    # path emits (the client renders nothing for a bare completed+error), with
    # the pod's typed error string mapped to a friendly (code, message).
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply(
        "turn_failed", {"error": "ProviderOverloadedError: Anthropic is momentarily overloaded."}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    buf = RecBuffer()
    events = []
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=buf,
                                   on_event=lambda kind, data: events.append((kind, data)))
    failed = buf.records[-1][1]["sse"]
    assert "response.failed" in failed
    assert "provider_overloaded" in failed
    assert "momentarily overloaded" in failed
    assert buf.closed == "error"
    # on_event sees the same classification the frame carried
    assert events[-1][0] == "turn_failed"
    assert events[-1][1]["code"] == "provider_overloaded"


@pytest.mark.asyncio
async def test_unmapped_turn_failure_is_redacted(monkeypatch):
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply(
        "turn_failed", {"error": "RuntimeError: secret-internal-detail"}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    buf = RecBuffer()
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=buf)
    failed = buf.records[-1][1]["sse"]
    assert "response.failed" in failed
    assert "anton_error" in failed
    assert "secret-internal-detail" not in failed  # raw text never reaches the client


@pytest.mark.asyncio
async def test_produce_remote_turn_history_defaults_empty(monkeypatch):
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=RecBuffer())
    assert json.loads(fake.added[0][1]["payload"])["params"]["history"] == []


@pytest.mark.asyncio
async def test_produce_remote_turn_forwards_skills(monkeypatch):
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    skills = {"csv-summary": {"files": {"SKILL.md": "---\nname: csv-summary\n---\nbody"}}}
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=RecBuffer(),
                                   skills=skills)
    assert json.loads(fake.added[0][1]["payload"])["params"]["skills"] == skills


@pytest.mark.asyncio
async def test_oversized_request_drops_skills_then_memory(monkeypatch):
    # A valid memory + skills combination can exceed the pod's stdin cap. Rather
    # than let the request line truncate into unparseable JSON, the producer
    # sheds skills first, then memory, so the turn still runs.
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setattr(prod, "_MAX_REQUEST_BYTES", 4096)
    monkeypatch.setattr(prod, "_REQUEST_BYTES_MARGIN", 0)

    big_skills = {"s": {"files": {"SKILL.md": "x" * 5000}}}
    big_memory = {"global": {"rules": "y" * 5000}}
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=RecBuffer(),
                                   memory=big_memory, skills=big_skills)
    params = json.loads(fake.added[0][1]["payload"])["params"]
    # skills shed first; memory then also shed because it alone still overruns
    assert "skills" not in params
    assert "memory" not in params
    assert prod._request_wire_size(params) <= 4096


@pytest.mark.asyncio
async def test_request_keeps_memory_when_dropping_skills_suffices(monkeypatch):
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setattr(prod, "_MAX_REQUEST_BYTES", 4096)
    monkeypatch.setattr(prod, "_REQUEST_BYTES_MARGIN", 0)

    big_skills = {"s": {"files": {"SKILL.md": "x" * 5000}}}
    small_memory = {"global": {"rules": "keep me"}}
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=RecBuffer(),
                                   memory=small_memory, skills=big_skills)
    params = json.loads(fake.added[0][1]["payload"])["params"]
    assert "skills" not in params
    assert params["memory"] == small_memory   # shedding skills alone was enough


@pytest.mark.asyncio
async def test_produce_remote_turn_omits_empty_skills(monkeypatch):
    # Like memory: a skill-less turn keeps the pre-existing payload shape.
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=RecBuffer(),
                                   skills={})
    assert "skills" not in json.loads(fake.added[0][1]["payload"])["params"]


@pytest.mark.asyncio
async def test_produce_remote_turn_mints_and_attaches_llm_block(monkeypatch):
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    # _reply() hardcodes correlation_id="r" on the reply envelope, so the
    # minted correlation id must match it or the reply loop never sees a
    # terminal reply and spins forever.
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")

    captured = {}

    async def _fake_mint(**kw):
        captured.update(kw)
        assert kw["correlation_id"]
        return "mdb_turnkey"

    monkeypatch.setattr(prod, "mint_turn_key", _fake_mint)

    await prod.produce_remote_turn(
        conversation_id="conv-1", org_id="o1", user_id="u1", input_text="hi",
        model="mindshub_air", buffer=RecBuffer(),
    )

    payload = json.loads(fake.added[0][1]["payload"])
    llm = payload["params"]["llm"]
    assert llm["provider"] == "minds-cloud"
    assert llm["api_key"] == "mdb_turnkey"
    assert llm["base_url"]
    assert llm["coding_model"]  # rides along so the pod's coding calls stay payable
    # Only the internal shared secret authenticates the mint call; no
    # per-tenant credential is looked up or passed.
    assert "credential" not in captured
    assert captured["correlation_id"] == "r"
    assert captured["user_id"] == "u1"
    assert captured["org_id"] == "o1"


@pytest.mark.asyncio
async def test_mint_llm_block_uses_minds_coding_default():
    # The pod runs on minds-cloud, so the coding model must be a minds alias.
    # Default = the free-bucket model (mindshub_air): a fresh org has no wallet
    # balance, so any premium alias 402s on the first turn.
    from cowork.common.settings.app_settings import TurnQueueSettings, MINDS_FREE_MODEL

    settings = TurnQueueSettings(minds_base_url="https://api.example.dev/v1")
    block = await prod._mint_llm_block(org_id="o", user_id="u", correlation_id="c", settings=settings)
    assert block["coding_model"] == MINDS_FREE_MODEL


@pytest.mark.asyncio
async def test_mint_llm_block_coding_model_override():
    from cowork.common.settings.app_settings import TurnQueueSettings

    settings = TurnQueueSettings(minds_base_url="https://x/v1", minds_coding_model="sonnet")
    block = await prod._mint_llm_block(org_id="o", user_id="u", correlation_id="c", settings=settings)
    assert block["coding_model"] == "sonnet"


@pytest.mark.asyncio
async def test_mint_llm_block_uses_base_url_override():
    """An explicit minds_base_url (per-PR / non-standard env) overrides the
    env-slug derivation, so the pod calls the right environment's inference."""
    from cowork.common.settings.app_settings import TurnQueueSettings

    settings = TurnQueueSettings(minds_base_url="https://api-pr-cowork-server-243.dev.mindshub.ai/v1")
    block = await prod._mint_llm_block(org_id="o", user_id="u", correlation_id="c", settings=settings)
    assert block["base_url"] == "https://api-pr-cowork-server-243.dev.mindshub.ai/v1"
    assert block["provider"] == "minds-cloud" and block["api_key"] == "mdb_turnkey"


@pytest.mark.asyncio
async def test_mint_llm_block_derives_base_url_when_override_empty():
    from cowork.common.settings.app_settings import TurnQueueSettings

    settings = TurnQueueSettings(minds_base_url="")
    block = await prod._mint_llm_block(org_id="o", user_id="u", correlation_id="c", settings=settings)
    assert "mindshub.ai" in block["base_url"] and block["base_url"].endswith("/v1")


class SilentRedis:
    """A worker that never answers: xread always times out empty."""

    def __init__(self, delay=0.02):
        self.added = []
        self.registered = []
        self.reads = 0
        self._delay = delay

    async def sadd(self, key, member):
        self.registered.append((key, member))
        return 1

    async def hset(self, key, mapping=None):
        return 1

    async def expire(self, key, seconds):
        return 1

    async def delete(self, *keys):
        return len(keys)

    async def xadd(self, stream, fields):
        self.added.append((stream, fields))
        return "1-0"

    async def xread(self, streams, count=None, block=None):
        self.reads += 1
        await asyncio.sleep(self._delay)
        return None


@pytest.mark.asyncio
async def test_unresponsive_worker_fails_the_turn_instead_of_spinning(monkeypatch):
    """With the worker down the reply loop used to `continue` forever: the
    buffer was never closed, so tail() never returned and the SSE response
    never ended — which the keepalive now keeps alive indefinitely. The loop
    must give up and emit the same response.failed frame turn_failed emits."""
    fake = SilentRedis()
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setenv("COWORK_TURN_REPLY_IDLE_TIMEOUT_SECONDS", "0.01")
    buf = RecBuffer()
    events = []

    await asyncio.wait_for(prod.produce_remote_turn(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi",
        model=None, buffer=buf, on_event=lambda kind, data: events.append((kind, data)),
    ), timeout=5)

    failed = buf.records[-1][1]["sse"]
    assert "response.failed" in failed
    assert "anton_error" in failed
    # Closed, so the buffer's tail() ends and the SSE response with it.
    assert buf.closed == "error"
    # And persisted as a failure, so a reload shows the error card.
    assert events[-1][0] == "turn_failed"
    assert events[-1][1]["code"] == "anton_error"


@pytest.mark.asyncio
async def test_replies_refresh_the_idle_deadline(monkeypatch):
    """The bound is on silence, not on turn duration: a turn that keeps
    streaming must never be cut off by it."""

    class TrickleRedis(SilentRedis):
        """A live turn as the real client sees it: the blocking read times out
        empty between the replies that do arrive."""

        def __init__(self):
            super().__init__(delay=0.02)
            self._deltas_left = 9
            self._quiet = False

        async def xread(self, streams, count=None, block=None):
            self.reads += 1
            await asyncio.sleep(self._delay)
            self._quiet = not self._quiet
            if self._quiet:
                return None
            if self._deltas_left:
                self._deltas_left -= 1
                return [["scratchpad:reply:conv-1", [[f"{self.reads}-0",
                                                      _reply("turn_delta", {"text": "x"})]]]]
            return [["scratchpad:reply:conv-1", [[f"{self.reads}-0",
                                                  _reply("turn_completed", {})]]]]

    fake = TrickleRedis()
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    # Shorter than the whole turn (20 reads x 20ms = ~0.4s), several times
    # longer than any single quiet gap (~0.04s) — so this can only pass if each
    # reply pushes the deadline out, with enough slack for a loaded CI box.
    monkeypatch.setenv("COWORK_TURN_REPLY_IDLE_TIMEOUT_SECONDS", "0.15")
    buf = RecBuffer()

    await asyncio.wait_for(prod.produce_remote_turn(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi",
        model=None, buffer=buf,
    ), timeout=5)

    assert buf.closed == "completed"
    assert "response.failed" not in "".join(r[1]["sse"] for r in buf.records)


@pytest.mark.asyncio
async def test_idle_bound_can_be_disabled(monkeypatch):
    """<= 0 keeps the old unbounded behavior for an operator who needs it."""
    fake = SilentRedis(delay=0.005)
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setenv("COWORK_TURN_REPLY_IDLE_TIMEOUT_SECONDS", "0")
    buf = RecBuffer()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(prod.produce_remote_turn(
            conversation_id="conv-1", org_id=None, user_id=None, input_text="hi",
            model=None, buffer=buf,
        ), timeout=0.2)

    assert buf.closed is None
