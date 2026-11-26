"""Billing classification survives the live stream and reopening its saved turn."""

from __future__ import annotations

import asyncio
import json
from uuid import UUID, uuid4

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
from fastapi import HTTPException
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


@pytest.mark.parametrize(("scope", "owner"), [("workspace", "This workspace's"), (None, "A")])
def test_a_spent_budget_names_its_scope_live_and_on_reload(
    monkeypatch, self_hosted_minds_url, conversation_id, scope, owner,
):
    error = _billing_stop(status=402, reason="budget_exceeded")
    if scope is not None:
        error.__cause__.response.headers["X-MindsHub-Budget-Scope"] = scope

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["code"] == turn_errors.BUDGET_EXCEEDED_CODE
    assert failed["error"].startswith(f"{owner} spending budget is used up.")
    assert failed.get("budget_scope") == scope
    assert "reset_at" not in failed


def test_a_spent_budget_names_its_period_live_and_on_reload(monkeypatch, self_hosted_minds_url, conversation_id):
    error = _billing_stop(status=402, reason="budget_exceeded")
    error.__cause__.response.headers["X-MindsHub-Budget-Scope"] = "workspace"
    error.__cause__.response.headers["X-MindsHub-Budget-Period"] = "day"

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["error"].startswith("This workspace's daily spending budget is used up.")
    assert failed.get("budget_period") == "day"


# ── ENG-4186: the two subscription stops ─────────────────────────────────
# anton has no typed stop for either yet, so the raw SDK error reaches the
# ladder and the gateway's reason header decides it. That needs the gateway to
# be the configured `minds_url` host: on self-hosted's nonresolving one only a
# typed anton stop could carry the verdict, as for the stops above. The gateway puts `window`
# and `required_tier` at the top level of the body, beside `error`, where the
# OpenAI SDK's `body` (the `error` envelope alone) does not keep them; they are
# read back off the response.


@pytest.fixture
def mindshub_minds_url(self_hosted_minds_url):
    self_hosted_minds_url.minds_url = "https://api.mindshub.ai"
    return self_hosted_minds_url


def _plan_stop(*, status: int, reason: str, url: str = GATEWAY_URL, **top_level: str) -> APIStatusError:
    body = {"code": reason, "message": "Denied"}
    headers = {"X-MindsHub-Reason": reason}
    if status == 429:
        headers["X-MindsHub-Reset-At"] = RESET_AT
    response = httpx.Response(
        status,
        request=httpx.Request("POST", url),
        headers=headers,
        json={"error": body, **top_level},
    )
    return APIStatusError("Denied", response=response, body=body)


@pytest.mark.parametrize("window", ["5h", "7d"])
def test_a_spent_plan_window_keeps_its_card_reset_and_window_on_reload(
    monkeypatch, mindshub_minds_url, conversation_id, window,
):
    error = _plan_stop(status=429, reason="window_allowance_exhausted", reset_at=RESET_AT, window=window)

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["code"] == turn_errors.WINDOW_ALLOWANCE_EXHAUSTED_CODE
    assert failed["error"] == turn_errors.WINDOW_ALLOWANCE_EXHAUSTED_USER_MESSAGE
    assert failed["reset_at"] == RESET_AT
    assert failed["window"] == window
    assert "required_tier" not in failed


def test_an_unknown_plan_window_is_left_off_the_frame(monkeypatch, mindshub_minds_url, conversation_id):
    error = _plan_stop(status=429, reason="window_allowance_exhausted", window="1d")

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["code"] == turn_errors.WINDOW_ALLOWANCE_EXHAUSTED_CODE
    assert failed["reset_at"] == RESET_AT
    assert "window" not in failed


@pytest.mark.parametrize(
    ("tier", "expected"), [("plus", "plus"), ("pro", "pro"), ("free", None), ("enterprise", None)],
)
def test_a_model_outside_the_tier_keeps_its_card_and_tier_on_reload(
    monkeypatch, mindshub_minds_url, conversation_id, tier, expected,
):
    error = _plan_stop(status=403, reason="model_not_in_tier", required_tier=tier)

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["code"] == turn_errors.MODEL_NOT_IN_TIER_CODE
    assert failed["error"] == turn_errors.MODEL_NOT_IN_TIER_USER_MESSAGE
    assert failed.get("required_tier") == expected
    assert "reset_at" not in failed
    assert "window" not in failed


def test_a_third_party_cannot_persist_a_plan_card(monkeypatch, mindshub_minds_url, conversation_id):
    error = _plan_stop(
        status=429, reason="window_allowance_exhausted", url="https://openrouter.ai/api/v1/x", window="5h",
    )

    failed = _saved_and_streamed_failure(monkeypatch, conversation_id, error)

    assert failed["code"] == turn_errors.GENERIC_TURN_ERROR_CODE
    assert "window" not in failed


# ── The non-streaming 400 carries the same fields ────────────────────────
# `POST /responses` with `stream=false` raises the failure frame as the 400's
# detail. These pin that the detail keeps the reset instant and the plan fields
# for their own codes, as the streaming frame above does, and drops them for
# any other code even when the inference service sent them.


def _collected_failure(error: Exception) -> dict:
    handler = _handler_with_raising_formatter(error)
    with pytest.raises(HTTPException) as raised:
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", original_content="hi"))
    assert raised.value.status_code == 400
    assert raised.value.detail["type"] == "response.failed"
    return raised.value.detail


@pytest.mark.parametrize(
    ("status", "reason", "code", "has_reset"),
    [
        pytest.param(402, "wallet_empty", turn_errors.TOKEN_LIMIT_CODE, False, id="wallet"),
        pytest.param(429, "included_allowance_exhausted", turn_errors.ALLOWANCE_EXHAUSTED_CODE, True, id="allowance"),
        pytest.param(
            429, "free_air_daily_spend_fuse_exceeded", turn_errors.FREE_SERVING_PAUSED_CODE, True, id="fuse",
        ),
    ],
)
def test_the_non_streaming_400_carries_the_reset_instant_for_its_codes(
    self_hosted_minds_url, status, reason, code, has_reset,
):
    # Every row's response from the inference service carries a reset instant.
    # Only the codes that offer waiting may pass it on.
    failed = _collected_failure(_billing_stop(status=status, reason=reason))

    assert failed["code"] == code
    assert failed.get("reset_at") == (RESET_AT if has_reset else None)
    assert "window" not in failed
    assert "required_tier" not in failed


def test_the_non_streaming_400_names_the_spent_plan_window(mindshub_minds_url):
    error = _plan_stop(status=429, reason="window_allowance_exhausted", window="7d", required_tier="pro")

    failed = _collected_failure(error)

    assert failed["code"] == turn_errors.WINDOW_ALLOWANCE_EXHAUSTED_CODE
    assert failed["reset_at"] == RESET_AT
    assert failed["window"] == "7d"
    assert "required_tier" not in failed


def test_the_non_streaming_400_names_the_tier_that_covers_the_model(mindshub_minds_url):
    error = _plan_stop(status=403, reason="model_not_in_tier", required_tier="pro", window="7d")
    error.response.headers["X-MindsHub-Reset-At"] = RESET_AT

    failed = _collected_failure(error)

    assert failed["code"] == turn_errors.MODEL_NOT_IN_TIER_CODE
    assert failed["required_tier"] == "pro"
    assert "reset_at" not in failed
    assert "window" not in failed
