import json
import pytest
from cowork.turnqueue import producer as prod


class FakeRedis:
    def __init__(self, replies):
        self.added = []
        self._replies = replies  # list of (stream, {"payload": json})
    async def xadd(self, stream, fields):
        self.added.append((stream, fields))
        return "1-0"
    async def xread(self, streams, count=None, block=None):
        if self._replies:
            stream, fields = self._replies.pop(0)
            return [[stream, [["9-0", fields]]]]
        return None


class RecBuffer:
    def __init__(self): self.records = []; self.closed = None
    async def append(self, type_, data): self.records.append((type_, data)); return len(self.records)
    async def close(self, reason, extra=None): self.closed = reason


@pytest.mark.asyncio
async def test_produce_remote_turn_completes(monkeypatch):
    reply = {"payload": json.dumps({"correlation_id": "r", "kind": "turn_completed",
                                    "data": {"final_text": "hello"}})}
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", reply)])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")

    buf = RecBuffer()
    await prod.produce_remote_turn(conversation_id="conv-1", org_id=None, user_id=None,
                                   input_text="hi", model=None, buffer=buf)

    stream, fields = fake.added[0]
    assert stream == "scratchpad:requests"
    job = json.loads(fields["payload"])
    assert job["op"] == "anton_turn"
    assert job["params"]["input"] == "hi"
    types = [t for t, _ in buf.records]
    assert types == ["sse", "sse", "sse"]
    assert "hello" in buf.records[1][1]["sse"]
    assert buf.closed == "completed"
