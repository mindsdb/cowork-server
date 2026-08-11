"""CoworkElicitor — waits on the broker's future, and nothing else.

Publication is the stream layer's job and the open/close pairing belongs to
anton's elicit(), so this class is deliberately thin.
"""

from __future__ import annotations

import asyncio

import pytest

from anton.core.interaction.elicit import AskOption, AskRequest
from cowork.harnesses.anton_harness.elicitor import CoworkElicitor
from cowork.streaming.answers import AnswerBroker, SubmitResult

CID = "conv-1"
QID = "ask:abc"


def _request(timeout_s=300, **over) -> AskRequest:
    base = dict(
        prompt="Which database?",
        timeout_s=timeout_s,
        options=(
            AskOption(value="pg", label="postgres"),
            AskOption(value="my", label="mysql"),
        ),
    )
    base.update(over)
    return AskRequest(**base)


@pytest.fixture()
def wired():
    broker = AnswerBroker()
    return broker, CoworkElicitor(CID, broker, timeout_s=300)


def test_declares_choice_only_and_a_hint(wired):
    _, elicitor = wired
    assert elicitor.supported_kinds == ("choice",)
    assert elicitor.timeout_s == 300
    assert isinstance(elicitor.answer_hint, str) and elicitor.answer_hint


async def test_ask_returns_the_submitted_answer(wired):
    broker, elicitor = wired
    request = _request()
    await elicitor.begin(QID, request)

    async def _answer_soon():
        await asyncio.sleep(0.01)
        broker.submit(CID, QID, {"values": ["my"]})

    asyncio.create_task(_answer_soon())
    answer = await elicitor.ask(QID, request)
    assert answer.status == "answered"
    assert answer.values == ("my",)
    assert answer.text == ""


async def test_text_only_answer(wired):
    broker, elicitor = wired
    request = _request()
    await elicitor.begin(QID, request)
    broker.submit(CID, QID, {"text": "clickhouse"})
    answer = await elicitor.ask(QID, request)
    assert answer.status == "answered"
    assert (answer.values, answer.text) == ((), "clickhouse")


async def test_values_and_text_together(wired):
    """Two buttons plus a typed third answer — legitimate only for a
    multi-select question, since the broker rejects two values for
    select="one". Asserted ACCEPTED first: a rejection would leave the future
    unresolved and this test would spend the full 300 s timeout before failing
    on the wrong assertion."""
    broker, elicitor = wired
    request = _request(select="many")
    await elicitor.begin(QID, request)
    assert (
        broker.submit(CID, QID, {"values": ["pg", "my"], "text": "and duckdb"})
        is SubmitResult.ACCEPTED
    )
    answer = await elicitor.ask(QID, request)
    assert answer.values == ("pg", "my")
    assert answer.text == "and duckdb"


async def test_skipped_becomes_cancelled(wired):
    broker, elicitor = wired
    request = _request()
    await elicitor.begin(QID, request)
    broker.submit(CID, QID, {"skipped": True})
    assert (await elicitor.ask(QID, request)).status == "cancelled"


async def test_timeout_status_when_nobody_answers(wired):
    broker, elicitor = wired
    request = _request(timeout_s=0.05)
    await elicitor.begin(QID, request)
    assert (await elicitor.ask(QID, request)).status == "timeout"


async def test_a_timeout_does_not_cancel_the_broker_s_future(wired):
    """ask()'s own wait_for must not mutate state the broker owns.

    Without asyncio.shield, the timeout cancels the future in place while it is
    still in the broker's _pending dict. submit() gates on future.done(), which
    a cancelled future satisfies, so the answer would be reported
    ALREADY_ANSWERED and then silently dropped. end() has not run yet at this
    point, so the entry is still the broker's to own.
    """
    broker, elicitor = wired
    request = _request(timeout_s=0.05)
    await elicitor.begin(QID, request)
    assert (await elicitor.ask(QID, request)).status == "timeout"

    # end() deliberately not called yet: this is the window the shield protects.
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.ACCEPTED


async def test_timeout_falls_back_to_instance_timeout_s(wired):
    """request.timeout_s is None -- ask() must fall back to self.timeout_s."""
    broker = AnswerBroker()
    elicitor = CoworkElicitor(CID, broker, timeout_s=0.05)
    request = _request(timeout_s=None)
    await elicitor.begin(QID, request)
    assert (await elicitor.ask(QID, request)).status == "timeout"


async def test_an_empty_payload_is_treated_as_an_empty_answered_answer(wired):
    """broker.submit(cid, qid, {}) is not rejected: it resolves the future
    with an empty answered AskAnswer, indistinguishable from a real one.

    This is intentional-by-omission here: the HTTP endpoint (a later task)
    rejects an empty request body before it ever reaches the broker, so
    production never constructs this. Documented here, not fixed here.
    """
    broker, elicitor = wired
    request = _request()
    await elicitor.begin(QID, request)
    broker.submit(CID, QID, {})
    answer = await elicitor.ask(QID, request)
    assert answer.status == "answered"
    assert (answer.values, answer.text) == ((), "")


async def test_end_drops_the_broker_entry(wired):
    broker, elicitor = wired
    request = _request()
    await elicitor.begin(QID, request)
    await elicitor.end(QID)

    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.NOT_FOUND


async def test_cancelling_ask_propagates(wired):
    """anton's elicit() runs end() in its finally, so this class does not
    need cleanup of its own -- but cancellation must not be swallowed."""
    broker, elicitor = wired
    request = _request()
    await elicitor.begin(QID, request)
    task = asyncio.create_task(elicitor.ask(QID, request))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
