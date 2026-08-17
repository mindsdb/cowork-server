import asyncio
import json

import pytest

from cowork.turnqueue import producer as prod


@pytest.fixture(autouse=True)
def _stub_llm_mint(monkeypatch):
    """Every stream_remote_replies call now mints a turn key before enqueuing.
    Stub it by default so the pre-existing tests (which only exercise the
    reply-loop behavior) don't need to know about the llm block; tests
    that care override this explicitly."""

    async def _fake_mint(**kw):
        return "mdb_turnkey"

    monkeypatch.setattr(prod, "mint_turn_key", _fake_mint)


class FakeRedis:
    def __init__(self, replies): self.added = []; self.registered = []; self._replies = replies
    async def sadd(self, key, member): self.registered.append((key, member)); return 1
    async def xadd(self, stream, fields): self.added.append((stream, fields)); return "1-0"
    async def xread(self, streams, count=None, block=None):
        if self._replies:
            stream, fields = self._replies.pop(0)
            return [[stream, [["9-0", fields]]]]
        return None


def _reply(kind, data):
    return {"payload": json.dumps({"correlation_id": "r", "kind": kind, "data": data})}


async def _drain(gen):
    """Exhaust the reply generator, returning its (kind, data) yields.
    Nothing happens (no mint, no XADD) before the first __anext__."""
    return [item async for item in gen]


@pytest.mark.asyncio
async def test_job_goes_to_the_conversations_own_stream(monkeypatch):
    """One stream per conversation, listed in the registry set. The controller locks a
    conversation before reading its stream, so a busy pod's next job stays undelivered
    instead of blocking every other conversation on one shared stream."""
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    await _drain(prod.stream_remote_replies(conversation_id="conv-1", org_id=None,
                                            user_id=None, input_text="hi", model="m"))
    assert fake.added[0][0] == "scratchpad:requests:conv-1"
    assert fake.registered == [("scratchpad:requests:queues", "conv-1")]


@pytest.mark.asyncio
async def test_stream_remote_replies_yields_deltas_in_order(monkeypatch):
    replies = [("scratchpad:reply:conv-1", _reply("turn_delta", {"text": "he"})),
               ("scratchpad:reply:conv-1", _reply("turn_delta", {"text": "llo"})),
               ("scratchpad:reply:conv-1", _reply("turn_completed", {}))]
    fake = FakeRedis(replies=replies)
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    items = await _drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi",
        model="m", history=[{"role": "user", "content": "prev"}]))
    job = json.loads(fake.added[0][1]["payload"])
    assert job["op"] == "anton_turn"
    assert job["params"]["history"] == [{"role": "user", "content": "prev"}]
    assert items == [("turn_delta", {"text": "he"}),
                     ("turn_delta", {"text": "llo"}),
                     ("turn_completed", {})]


@pytest.mark.asyncio
async def test_turn_failed_yields_classified_code_and_message(monkeypatch):
    # A failed turn must end with a turn_failed carrying the same friendly
    # (code, message) the caller streams as response.failed and persists,
    # with the pod's typed error string mapped to it.
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply(
        "turn_failed", {"error": "ProviderOverloadedError: Anthropic is momentarily overloaded."}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    items = await _drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi", model="m"))
    kind, data = items[-1]
    assert kind == "turn_failed"
    assert data["code"] == "provider_overloaded"
    assert "momentarily overloaded" in data["message"]
    assert items == [items[-1]]  # terminal is the only yield, then the generator ends


@pytest.mark.asyncio
async def test_unmapped_turn_failure_is_redacted(monkeypatch):
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply(
        "turn_failed", {"error": "RuntimeError: secret-internal-detail"}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    items = await _drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi", model="m"))
    kind, data = items[-1]
    assert kind == "turn_failed"
    assert data["code"] == "anton_error"
    # The client-facing message never carries the raw text.
    assert "secret-internal-detail" not in data["message"]


@pytest.mark.asyncio
async def test_stream_remote_replies_history_defaults_empty(monkeypatch):
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    await _drain(prod.stream_remote_replies(conversation_id="conv-1", org_id=None,
                                            user_id=None, input_text="hi", model="m"))
    assert json.loads(fake.added[0][1]["payload"])["params"]["history"] == []


@pytest.mark.asyncio
async def test_stream_remote_replies_forwards_skills(monkeypatch):
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    skills = {"csv-summary": {"files": {"SKILL.md": "---\nname: csv-summary\n---\nbody"}}}
    await _drain(prod.stream_remote_replies(conversation_id="conv-1", org_id=None,
                                            user_id=None, input_text="hi", model="m",
                                            skills=skills))
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
    await _drain(prod.stream_remote_replies(conversation_id="conv-1", org_id=None,
                                            user_id=None, input_text="hi", model="m",
                                            memory=big_memory, skills=big_skills))
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
    await _drain(prod.stream_remote_replies(conversation_id="conv-1", org_id=None,
                                            user_id=None, input_text="hi", model="m",
                                            memory=small_memory, skills=big_skills))
    params = json.loads(fake.added[0][1]["payload"])["params"]
    assert "skills" not in params
    assert params["memory"] == small_memory   # shedding skills alone was enough


@pytest.mark.asyncio
async def test_stream_remote_replies_omits_empty_skills(monkeypatch):
    # Like memory: a skill-less turn keeps the pre-existing payload shape.
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    await _drain(prod.stream_remote_replies(conversation_id="conv-1", org_id=None,
                                            user_id=None, input_text="hi", model="m",
                                            skills={}))
    assert "skills" not in json.loads(fake.added[0][1]["payload"])["params"]


@pytest.mark.asyncio
async def test_stream_remote_replies_mints_and_attaches_llm_block(monkeypatch):
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

    await _drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id="o1", user_id="u1", input_text="hi",
        model="mindshub_air",
    ))

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

    async def xadd(self, stream, fields):
        self.added.append((stream, fields))
        return "1-0"

    async def xread(self, streams, count=None, block=None):
        self.reads += 1
        await asyncio.sleep(self._delay)
        return None


@pytest.mark.asyncio
async def test_unresponsive_worker_fails_the_turn_instead_of_spinning(monkeypatch):
    """With the worker down the reply loop used to `continue` forever: no
    terminal was ever yielded, so the caller's SSE response never ended. The
    loop must give up and yield the same classified turn_failed a pod failure
    yields."""
    fake = SilentRedis()
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setenv("COWORK_TURN_REPLY_IDLE_TIMEOUT_SECONDS", "0.01")

    items = await asyncio.wait_for(_drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi",
        model="m",
    )), timeout=5)

    # Terminal yielded, so the generator (and the SSE response with it) ends,
    # and the caller persists a failure so a reload shows the error card.
    kind, data = items[-1]
    assert kind == "turn_failed"
    assert data["code"] == "anton_error"
    assert data["error"] == prod.UNRESPONSIVE_WORKER_ERROR


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

    items = await asyncio.wait_for(_drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi",
        model="m",
    )), timeout=5)

    assert items[-1] == ("turn_completed", {})
    assert all(kind != "turn_failed" for kind, _ in items)


@pytest.mark.asyncio
async def test_idle_bound_can_be_disabled(monkeypatch):
    """<= 0 keeps the old unbounded behavior for an operator who needs it."""
    fake = SilentRedis(delay=0.005)
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setenv("COWORK_TURN_REPLY_IDLE_TIMEOUT_SECONDS", "0")

    items = []

    async def consume():
        async for item in prod.stream_remote_replies(
                conversation_id="conv-1", org_id=None, user_id=None,
                input_text="hi", model="m"):
            items.append(item)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(consume(), timeout=0.2)

    assert items == []  # no synthesized terminal — the loop just keeps waiting


def test_step_stream_events_tool_end_replays_args():
    from anton.core.llm.provider import StreamToolUseDelta, StreamToolUseEnd

    events = prod.step_stream_events(
        {"step": "tool_end", "id": "t1", "args": '{"code": "1+1"}'})
    assert isinstance(events[0], StreamToolUseDelta)
    assert events[0].json_delta == '{"code": "1+1"}'
    assert events[0].id == "t1"
    assert isinstance(events[1], StreamToolUseEnd)
    assert events[1].id == "t1"


def test_step_stream_events_tool_end_without_args_skips_delta():
    from anton.core.llm.provider import StreamToolUseEnd

    events = prod.step_stream_events({"step": "tool_end", "id": "t1"})
    assert len(events) == 1 and isinstance(events[0], StreamToolUseEnd)


def test_step_stream_events_round_end_carries_tool_call_truthiness():
    from anton.core.llm.provider import StreamComplete

    with_calls = prod.step_stream_events(
        {"step": "round_end", "had_tool_calls": True, "stop_reason": "tool_use"})
    assert len(with_calls) == 1 and isinstance(with_calls[0], StreamComplete)
    assert with_calls[0].response.tool_calls
    assert with_calls[0].response.stop_reason == "tool_use"

    without = prod.step_stream_events({"step": "round_end", "had_tool_calls": False})
    assert not without[0].response.tool_calls


def test_step_stream_events_unknown_step_yields_nothing():
    assert prod.step_stream_events({"step": "mystery"}) == []


@pytest.mark.asyncio
async def test_turn_skill_reaches_the_caller(monkeypatch):
    """The kind filter is a whitelist: a kind missing from it is dropped
    silently, so a skill draft would vanish between the pod and the card."""
    draft = {"slug": "my-skill", "files": {"SKILL.md": "body"}}
    fake = FakeRedis(replies=[
        ("scratchpad:reply:conv-1", _reply("turn_skill", {"entries": [draft]})),
        ("scratchpad:reply:conv-1", _reply("turn_completed", {})),
    ])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    items = await _drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi", model="m"))
    assert items == [("turn_skill", {"entries": [draft]}), ("turn_completed", {})]
