import json

import pytest

from cowork.turnqueue.models import TurnJob, TurnReply


def test_turn_job_roundtrips_the_envelope():
    job = TurnJob(op="anton_turn", conversation_id="c", correlation_id="r",
                  reply_stream="scratchpad:reply:c",
                  params={"input": "hi", "history": []})
    field = job.model_dump_json()
    back = json.loads(field)
    assert back["op"] == "anton_turn"
    assert back["params"]["input"] == "hi"


def test_turn_reply_parses_and_validates():
    reply = TurnReply.model_validate_json(
        '{"correlation_id":"r","kind":"turn_delta","data":{"text":"hi"}}')
    assert reply.kind == "turn_delta"
    assert reply.data["text"] == "hi"


def test_deadline_ms_is_a_duration_not_an_epoch():
    """The controller reads this as a relative budget. An epoch value would mean a
    ~57 year deadline, i.e. no timeout at all, so it must fail loudly here."""
    from pydantic import ValidationError

    from cowork.turnqueue.models import MAX_DEADLINE_MS, TurnJob

    base = dict(op="anton_turn", conversation_id="c", correlation_id="r",
                reply_stream="scratchpad:reply:c")
    assert TurnJob(**base, deadline_ms=90_000).deadline_ms == 90_000
    assert TurnJob(**base, deadline_ms=MAX_DEADLINE_MS).deadline_ms == MAX_DEADLINE_MS
    with pytest.raises(ValidationError, match="not an epoch timestamp"):
        TurnJob(**base, deadline_ms=1_790_000_000_000)
    with pytest.raises(ValidationError, match="must be positive"):
        TurnJob(**base, deadline_ms=0)
