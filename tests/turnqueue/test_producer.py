import json
import pytest
from cowork.turnqueue import producer as prod


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
                                   input_text="hi", model=None, buffer=buf)
    assert json.loads(fake.added[0][1]["payload"])["op"] == "anton_turn"
    sse = [r[1]["sse"] for r in buf.records]
    assert "response.created" in sse[0]
    assert "he" in sse[1] and "response.output_text.delta" in sse[1]
    assert "llo" in sse[2]
    assert "response.completed" in sse[3]
    assert buf.closed == "completed"
