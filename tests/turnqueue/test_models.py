import json

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
