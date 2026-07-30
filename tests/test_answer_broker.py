"""AnswerBroker — the in-process rendezvous between a blocked turn and the
HTTP request carrying the user's answer.

One process per user (see the design doc's deployment section), so an
in-memory Future is sufficient: the POST reaches the process holding the run
by construction.
"""

from __future__ import annotations

import asyncio

import pytest

from anton.core.interaction.elicit import AskOption, AskRequest
from cowork.streaming.answers import AnswerBroker, SubmitResult

CID = "conv-1"
QID = "ask:abc"


def _request(**over) -> AskRequest:
    base = dict(
        prompt="Which database?",
        options=(
            AskOption(value="pg", label="postgres"),
            AskOption(value="my", label="mysql"),
        ),
    )
    base.update(over)
    return AskRequest(**base)


@pytest.fixture()
def broker() -> AnswerBroker:
    return AnswerBroker()


async def test_open_then_submit_resolves_the_future(broker):
    future = broker.open(CID, QID, _request())
    assert not future.done()
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.ACCEPTED
    assert await asyncio.wait_for(future, timeout=1) == {"values": ["pg"]}


async def test_submit_for_an_unknown_question_is_not_found(broker):
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.NOT_FOUND
    broker.open(CID, QID, _request())
    assert broker.submit(CID, "ask:other", {"values": ["pg"]}) is SubmitResult.NOT_FOUND
    assert broker.submit("other-conv", QID, {"values": ["pg"]}) is SubmitResult.NOT_FOUND


async def test_second_submit_is_already_answered(broker):
    broker.open(CID, QID, _request())
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.ACCEPTED
    assert broker.submit(CID, QID, {"values": ["my"]}) is SubmitResult.ALREADY_ANSWERED


async def test_a_value_that_was_never_offered_is_rejected(broker):
    """Otherwise the model receives an option it never proposed and treats it
    as its own. Mirrors handle_select_path's 'invalid' guard."""
    future = broker.open(CID, QID, _request())
    assert broker.submit(CID, QID, {"values": ["sqlite"]}) is SubmitResult.INVALID_OPTION
    assert broker.submit(CID, QID, {"values": ["pg", "sqlite"]}) is SubmitResult.INVALID_OPTION
    assert not future.done()
    # Free-form text is a separate field, so nothing legitimate is blocked.
    assert broker.submit(CID, QID, {"text": "clickhouse"}) is SubmitResult.ACCEPTED


async def test_values_and_text_together_are_accepted(broker):
    broker.open(CID, QID, _request(select="many"))
    assert (
        broker.submit(CID, QID, {"values": ["pg", "my"], "text": "and duckdb"})
        is SubmitResult.ACCEPTED
    )


async def test_skipped_needs_no_option_validation(broker):
    broker.open(CID, QID, _request())
    assert broker.submit(CID, QID, {"skipped": True}) is SubmitResult.ACCEPTED


async def test_close_is_idempotent_and_drops_the_entry(broker):
    broker.open(CID, QID, _request())
    broker.close(CID, QID)
    broker.close(CID, QID)  # must not raise
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.NOT_FOUND


async def test_close_cancels_a_future_nobody_answered(broker):
    future = broker.open(CID, QID, _request())
    broker.close(CID, QID)
    assert future.cancelled() or future.done()


async def test_open_is_idempotent_for_the_same_key(broker):
    """elicit() calls begin() and the elicitor's ask() both reach open();
    a fresh future on the second call would orphan the first."""
    first = broker.open(CID, QID, _request())
    second = broker.open(CID, QID, _request())
    assert first is second
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.ACCEPTED
    assert await asyncio.wait_for(first, timeout=1) == {"values": ["pg"]}


async def test_open_is_idempotent_after_the_future_is_resolved(broker):
    """A resolved entry must not be replaced: that would make a duplicate
    submit look fresh and return ACCEPTED where ALREADY_ANSWERED is expected."""
    first = broker.open(CID, QID, _request())
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.ACCEPTED
    second = broker.open(CID, QID, _request())
    assert second is first
    assert broker.submit(CID, QID, {"values": ["my"]}) is SubmitResult.ALREADY_ANSWERED


def test_module_exposes_a_process_global_broker():
    from cowork.streaming.answers import broker as global_broker

    assert isinstance(global_broker, AnswerBroker)
