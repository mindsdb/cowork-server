"""Billing classification survives the live stream and reopening its saved turn."""

from __future__ import annotations

import json
from uuid import UUID

import httpx
import pytest
from anton.core.llm.provider import (
    AllowanceExhaustedError,
    FreeServingPausedError,
    MindsHubBillingStop,
    ModelCallTimeoutError,
    WalletEmptyError,
    mindshub_billing_stop,
)
from openai import APIStatusError

from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import unit_session
from cowork.handlers import responses, turn_errors
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID
from tests.test_inprocess_request_id import _RecBuffer, _run
from tests.test_turn_errors import _handler_with_raising_formatter


RESET_AT = "2026-10-19T00:00:00Z"
GATEWAY_URL = "https://api.mindshub.ai/v1/chat/completions"


@pytest.fixture
def conversation_id():
    with unit_session(scope=LOCAL_SCOPE) as session:
        conversation = ConversationService(session).create_conversation(
            "billing failure", project_id=GENERAL_PROJECT_ID,
        )
        conversation_id = conversation.id
    yield conversation_id
    with unit_session(scope=LOCAL_SCOPE) as session:
        # This fixture creates database rows only, with no Redis turn or scratchpad.
        service = ConversationService(session)
        service.stage_delete_conversation_row(
            service.get_conversation(conversation_id), include_org_attachments=False,
        )


def _sdk_error(*, status: int, reason: str | None, url: str = GATEWAY_URL) -> APIStatusError:
    body = {"code": reason, "message": "Denied"} if reason else {"message": "Denied"}
    response = httpx.Response(
        status,
        request=httpx.Request("POST", url),
        headers={"X-MindsHub-Reset-At": RESET_AT},
        json={"error": body},
    )
    return APIStatusError("Denied", response=response, body=body)


def _billing_stop(*, status: int, reason: str) -> MindsHubBillingStop:
    sdk_error = _sdk_error(status=status, reason=reason)
    stop = mindshub_billing_stop(exc=sdk_error, body=sdk_error.body, status_code=status)
    assert stop is not None
    # Both Anton provider mappers raise the factory's result from this SDK error.
    stop.__cause__ = sdk_error
    return stop


def _saved_and_streamed_failure(monkeypatch, conversation_id: UUID, error: Exception):
    handler = _handler_with_raising_formatter(error)
    monkeypatch.setattr(responses, "get_harness", lambda name: handler.harness)
    buffer = _RecBuffer()
    _run(handler, buffer, conv_id=conversation_id)

    assert buffer.closed == "error"
    [frame] = [frame for frame in buffer.frames if frame.startswith("event: response.failed\n")]
    streamed = json.loads(frame.split("data: ", 1)[1])
    with unit_session(scope=LOCAL_SCOPE) as session:
        messages = ConversationService(session).get_ordered_messages(conversation_id)
        question, answer = messages
        assert not question.pending
        assert str(answer.id) == streamed["assistant_message_id"]
        [saved] = [
            event.event_data for event in answer.message_events
            if event.event_data.get("type") == "response.failed"
        ]
        assert saved == {key: value for key, value in streamed.items() if key != "assistant_message_id"}
    return streamed


@pytest.mark.parametrize(
    ("status", "reason", "error_type", "code", "message", "has_reset"),
    [
        pytest.param(
            402, "wallet_empty", WalletEmptyError, turn_errors.TOKEN_LIMIT_CODE,
            turn_errors.TOKEN_LIMIT_USER_MESSAGE, False, id="wallet",
        ),
        pytest.param(
            429, "included_allowance_exhausted", AllowanceExhaustedError,
            turn_errors.ALLOWANCE_EXHAUSTED_CODE, turn_errors.ALLOWANCE_EXHAUSTED_USER_MESSAGE,
            True, id="allowance",
        ),
        pytest.param(
            429, "free_air_daily_spend_fuse_exceeded", FreeServingPausedError,
            turn_errors.FREE_SERVING_PAUSED_CODE, turn_errors.FREE_SERVING_PAUSED_USER_MESSAGE,
            True, id="fuse",
        ),
    ],
)
def test_real_billing_stop_keeps_its_card_and_reset_on_reload(
    monkeypatch, self_hosted_minds_url, conversation_id, status, reason, error_type, code, message, has_reset,
):
    error = _billing_stop(status=status, reason=reason)
    assert type(error) is error_type
    assert "X-MindsHub-Reason" not in error.__cause__.response.headers

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["code"] == code
    assert failed["error"] == message
    if has_reset:
        assert failed["reset_at"] == RESET_AT
    else:
        assert "reset_at" not in failed


@pytest.mark.parametrize("carrier", ["no_reason", "unknown_header", "unknown_body_code"])
@pytest.mark.parametrize("minds_url", ["https://minds.invalid", "https://api.mindshub.ai"])
def test_unclassified_gateway_429_stays_generic_on_reload(
    monkeypatch, self_hosted_minds_url, conversation_id, minds_url, carrier,
):
    self_hosted_minds_url.minds_url = minds_url
    error = _sdk_error(
        status=429, reason="future_billing_limit" if carrier == "unknown_body_code" else None,
    )
    if carrier == "unknown_header":
        error.response.headers["X-MindsHub-Reason"] = "future_billing_limit"

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["code"] == turn_errors.GENERIC_TURN_ERROR_CODE
    assert failed["error"] == turn_errors.GENERIC_TURN_ERROR_MESSAGE
    assert "reset_at" not in failed


def test_third_party_billing_carriers_cannot_persist_a_billing_card(
    monkeypatch, self_hosted_minds_url, conversation_id,
):
    error = _sdk_error(
        status=429, reason="included_allowance_exhausted", url="https://openrouter.ai/api/v1/x",
    )
    error.response.headers["X-MindsHub-Reason"] = "included_allowance_exhausted"
    assert mindshub_billing_stop(exc=error, body=error.body, status_code=429) is None

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["code"] == turn_errors.GENERIC_TURN_ERROR_CODE
    assert failed["error"] == turn_errors.GENERIC_TURN_ERROR_MESSAGE
    assert "reset_at" not in failed


def test_model_deadline_over_a_billing_stop_keeps_its_own_saved_failure(
    monkeypatch, self_hosted_minds_url, conversation_id,
):
    error = ModelCallTimeoutError(idle_timeout_s=60)
    error.__context__ = _billing_stop(status=429, reason="included_allowance_exhausted")

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["code"] == turn_errors.MODEL_TIMEOUT_CODE
    assert failed["error"] == str(error)
    assert "reset_at" not in failed
