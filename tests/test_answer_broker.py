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


async def test_more_values_than_the_question_offered_is_rejected(broker):
    """select="one" means one. Rejected rather than truncated: anton's
    CLIElicitor truncates (`picked[:1]`) because a human typed it at a
    terminal, but this path's caller is a GUI that rendered the card, so a
    shape it never offered is a frontend bug that must stay visible. Nothing
    downstream compensates — handle_ask_user passes answer.values through."""
    future = broker.open(CID, QID, _request(select="one"))
    assert broker.submit(CID, QID, {"values": ["pg", "my"]}) is SubmitResult.INVALID_OPTION
    assert not future.done()


async def test_duplicate_values_are_rejected(broker):
    """The offered-options check is set-based, so duplicates would otherwise
    pass it unchanged. Pinned on select="many" so cardinality cannot be what
    rejects it."""
    future = broker.open(CID, QID, _request(select="many"))
    assert (
        broker.submit(CID, QID, {"values": ["pg", "pg", "pg"]})
        is SubmitResult.INVALID_OPTION
    )
    assert not future.done()


async def test_free_form_text_is_rejected_when_custom_is_not_allowed(broker):
    """allow_custom=False asked for a button press, not prose."""
    future = broker.open(CID, QID, _request(allow_custom=False))
    assert broker.submit(CID, QID, {"text": "clickhouse"}) is SubmitResult.INVALID_OPTION
    assert not future.done()
    # ...and the button press itself still lands.
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.ACCEPTED


async def test_skipped_needs_no_option_validation(broker):
    broker.open(CID, QID, _request())
    assert broker.submit(CID, QID, {"skipped": True}) is SubmitResult.ACCEPTED


async def test_skipping_bypasses_every_shape_check(broker):
    """A skip is not an answer to the question, so it is never measured
    against the offer — even one that allows neither text nor a second value."""
    future = broker.open(CID, QID, _request(allow_custom=False))
    assert broker.submit(CID, QID, {"skipped": True}) is SubmitResult.ACCEPTED
    assert future.done()


async def test_reset_drops_every_entry_and_cancels_its_future(broker):
    """The module-level `broker` is a process global, so tests need a way to
    forget state that does not reach into _pending. Cancelling rather than just
    clearing keeps close()'s invariant: an entry never leaves the map with a
    live future nobody will resolve."""
    first = broker.open(CID, QID, _request())
    second = broker.open("conv-2", "ask:other", _request())
    broker.reset()
    assert first.cancelled() and second.cancelled()
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.NOT_FOUND
    broker.reset()  # must not raise on an empty map


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
