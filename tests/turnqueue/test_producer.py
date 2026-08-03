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
    def __init__(self, replies): self.added = []; self._replies = replies
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
    # Default = the minds-cloud coding default, NOT the tenant's resolved model
    # (which for a hosted user defaults to an Anthropic name minds 404s).
    from cowork.common.settings.app_settings import TurnQueueSettings, CODING_MODEL_DEFAULTS

    settings = TurnQueueSettings(minds_base_url="https://api.example.dev/v1")
    block = await prod._mint_llm_block(org_id="o", user_id="u", correlation_id="c", settings=settings)
    assert block["coding_model"] == CODING_MODEL_DEFAULTS["minds_cloud"]


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
