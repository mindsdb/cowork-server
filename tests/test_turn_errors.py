"""User-facing turn-error handling (ported from cowork PR #156).

When a turn dies on a cryptic provider 400 — most notably an image
arriving as an OpenAI-style ``image_url`` block instead of Anthropic's
``image`` block — the handler must surface a clean ``response.failed``
event (streaming) / 400 (non-streaming) with curated copy, while any
unmapped failure stays generic so provider internals never leak.

These tests pin the detection/mapping policy and the handler emission on
both the streaming and non-streaming paths.
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest
from fastapi import HTTPException

from cowork.handlers import turn_errors as te
from cowork.handlers.responses import ResponsesHandler


# ── Detection / mapping policy ────────────────────────────────────

def test_detects_anthropic_image_url_rejection():
    exc = Exception(
        "Input tag 'image_url' found using 'type' does not match "
        "any of the expected tags: 'image'"
    )
    assert te.is_image_format_error(exc) is True


def test_detects_unsupported_image_phrasing():
    assert te.is_image_format_error(Exception("Unsupported image media type")) is True


def test_ignores_unrelated_errors():
    assert te.is_image_format_error(Exception("Internal server error")) is False
    # A tool_use 400 must NOT be misread as an image failure.
    assert te.is_image_format_error(
        Exception("tool_use ids were found without tool_result blocks")
    ) is False


def test_maps_image_error_to_curated_copy():
    result = te.friendly_turn_error(
        Exception("'image_url' does not match expected tags: 'image'")
    )
    assert result is not None
    code, message = result
    assert code == "image_format"
    assert "PNG or JPEG" in message


def test_returns_none_for_unmapped_error():
    assert te.friendly_turn_error(Exception("boom")) is None


def test_response_failed_sse_shape():
    frame = te.response_failed_sse("oops", "image_format")
    assert frame.startswith("event: response.failed\ndata: ")
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload == {"type": "response.failed", "code": "image_format", "error": "oops"}


# ── Handler emission ──────────────────────────────────────────────

def _handler_with_raising_formatter(exc: Exception) -> ResponsesHandler:
    """A ResponsesHandler whose formatter yields one frame then raises —
    built without __init__ so no DB/harness setup is needed."""
    handler = object.__new__(ResponsesHandler)
    handler.principal = None  # __init__ bypassed; local-mode producer scope

    async def _formatter(stream, model, event_sink):
        yield "event: response.created\ndata: {}\n\n"
        raise exc

    async def _stream_response(
        *, conversation, input, disabled_connections=None,
        trace_tags=None, trace_metadata=None,
    ):
        if False:
            yield

    class _Harness:
        id = "anton"
        formatter = staticmethod(_formatter)
        stream_response = staticmethod(_stream_response)

    handler.harness = _Harness()
    return handler


async def _collect_produce_sse(handler: ResponsesHandler) -> list[str]:
    """Drive the streaming (_produce) error path and collect SSE frames."""
    from unittest.mock import MagicMock, patch

    frames: list[str] = []

    class _Buffer:
        async def append(self, _kind, data):
            frames.append(data["sse"])

        async def close(self, _status):
            pass

    conv_id = uuid4()
    mock_session = MagicMock()

    with (
        patch("cowork.handlers.responses.get_open_session", return_value=mock_session),
        patch("cowork.handlers.responses.ConversationService") as conv_svc,
        patch("cowork.handlers.responses.get_harness", return_value=handler.harness),
    ):
        conv_svc.return_value.get_conversation.return_value = MagicMock()
        await handler._produce(
            conv_id=conv_id,
            harness_input=[{"type": "text", "text": "hi"}],
            original_content="hi",
            model="anton",
            disabled=None,
            harness_name="anton",
            harness_id="anton",
            buffer=_Buffer(),
        )

    return frames


async def test_stream_emits_friendly_failed_event_for_image_error():
    exc = Exception("Input tag 'image_url' ... does not match the expected tags: 'image'")
    frames = await _collect_produce_sse(_handler_with_raising_formatter(exc))
    # created frame still came through, then a clean failure — no raise.
    assert any("response.created" in f for f in frames)
    failed = [f for f in frames if "response.failed" in f]
    assert len(failed) == 1
    payload = json.loads(failed[0].split("data: ", 1)[1].strip())
    assert payload["code"] == "image_format"
    assert "PNG or JPEG" in payload["error"]


async def test_stream_redacts_generic_error():
    frames = await _collect_produce_sse(
        _handler_with_raising_formatter(Exception("psycopg2: password authentication failed for user 'admin'"))
    )
    failed = [f for f in frames if "response.failed" in f]
    assert len(failed) == 1
    payload = json.loads(failed[0].split("data: ", 1)[1].strip())
    assert payload["code"] == te.GENERIC_TURN_ERROR_CODE
    assert payload["error"] == te.GENERIC_TURN_ERROR_MESSAGE
    # The raw provider/internal detail must not leak.
    assert "password" not in failed[0]


def test_collect_raises_400_with_curated_message_for_image_error():
    handler = _handler_with_raising_formatter(
        Exception("'image_url' does not match the expected tags: 'image'")
    )
    with pytest.raises(HTTPException) as err:
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", output_item_id="msg-1"))
    assert err.value.status_code == 400
    assert "PNG or JPEG" in err.value.detail


def test_collect_raises_500_generic_for_unmapped_error():
    handler = _handler_with_raising_formatter(Exception("kaboom: secret-token-xyz"))
    with pytest.raises(HTTPException) as err:
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", output_item_id="msg-1"))
    assert err.value.status_code == 500
    assert err.value.detail == te.GENERIC_TURN_ERROR_MESSAGE
    assert "secret-token" not in err.value.detail


# ── Token-limit (quota) detection / mapping ───────────────────────
#
# When an account's included-token allowance is spent, anton raises
# TokenLimitExceeded mid-turn. Before this was mapped, the exception
# aborted the SSE generator with no terminal event — the connection just
# closed and the renderer's spinner stopped, reading as "Anton is dead".
# These tests pin that a quota failure now surfaces curated copy on both
# paths instead.

# The stable 429 message anton builds for this case. Used to exercise the
# type-independent fallback path (no anton import needed).
_TOKEN_LIMIT_MESSAGE = (
    "Server returned 429 — Monthly limit exceeded for tokens: 5000000/5000000 "
    "Visit https://console.mindshub.ai to upgrade or to top up your tokens."
)


def test_detects_token_limit_via_anton_type():
    provider = pytest.importorskip("anton.core.llm.provider")
    assert te.is_token_limit_error(provider.TokenLimitExceeded(_TOKEN_LIMIT_MESSAGE)) is True


def test_detects_token_limit_via_message_fallback():
    # Even when the anton type isn't importable, the 429 message is stable
    # enough to recognise so the quota case never falls through to generic.
    assert te.is_token_limit_error(Exception(_TOKEN_LIMIT_MESSAGE)) is True


def test_token_limit_requires_both_signals():
    # A bare 429 (rate limit, not quota) or the tokens phrase on its own
    # must NOT be misread as an exhausted allowance.
    assert te.is_token_limit_error(Exception("Server returned 429 — too many requests")) is False
    assert te.is_token_limit_error(Exception("Monthly limit exceeded for tokens")) is False


def test_maps_token_limit_to_curated_copy():
    result = te.friendly_turn_error(Exception(_TOKEN_LIMIT_MESSAGE))
    assert result is not None
    code, message = result
    assert code == te.TOKEN_LIMIT_CODE
    assert message == te.TOKEN_LIMIT_USER_MESSAGE
    # Raw provider usage figures must not leak into the user copy.
    assert "5000000" not in message


def test_token_limit_takes_precedence_over_generic():
    # A quota failure must map to curated copy, never the redacted generic.
    code, _ = te.friendly_turn_error(Exception(_TOKEN_LIMIT_MESSAGE))
    assert code != te.GENERIC_TURN_ERROR_CODE


async def test_stream_emits_friendly_failed_event_for_token_limit():
    frames = await _collect_produce_sse(_handler_with_raising_formatter(Exception(_TOKEN_LIMIT_MESSAGE)))
    # created frame still came through, then a clean quota failure — no raise.
    assert any("response.created" in f for f in frames)
    failed = [f for f in frames if "response.failed" in f]
    assert len(failed) == 1
    payload = json.loads(failed[0].split("data: ", 1)[1].strip())
    assert payload["code"] == te.TOKEN_LIMIT_CODE
    assert payload["error"] == te.TOKEN_LIMIT_USER_MESSAGE


def test_collect_raises_400_with_curated_message_for_token_limit():
    handler = _handler_with_raising_formatter(Exception(_TOKEN_LIMIT_MESSAGE))
    with pytest.raises(HTTPException) as err:
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", output_item_id="msg-1"))
    assert err.value.status_code == 400
    assert err.value.detail == te.TOKEN_LIMIT_USER_MESSAGE


# ── Provider auth (401) → provider_auth ──────────────────────────────


def test_detects_auth_error_from_openai_401_message():
    # anton's openai provider maps a gateway 401 to this ConnectionError message.
    exc = ConnectionError("Invalid API key — check your OpenAI API key configuration.")
    assert te.is_auth_error(exc) is True


def test_detects_auth_error_from_anthropic_401_message():
    exc = ConnectionError("Invalid API key — check your ANTHROPIC_API_KEY environment variable.")
    assert te.is_auth_error(exc) is True


def test_bare_401_not_flagged():
    # Tightened: a 401/"unauthorized" without anton's specific "Invalid API key"
    # copy is NOT a provider-auth error (avoids mislabeling e.g. a tool API 401).
    assert te.is_auth_error(Exception("Server returned 401 — Unauthorized")) is False
    assert te.is_auth_error(Exception("connection reset")) is False


def test_auth_error_maps_to_provider_auth_code():
    code, message = te.friendly_turn_error(
        ConnectionError("Invalid API key — check your OpenAI API key configuration.")
    )
    assert code == te.AUTH_ERROR_CODE == "provider_auth"
    assert "reconnect" in message.lower()


def test_token_limit_wins_over_auth_for_credit_case():
    # A 429 credit/quota case must stay token_limit, not be misread as auth.
    code, _ = te.friendly_turn_error(Exception(_TOKEN_LIMIT_MESSAGE))
    assert code == te.TOKEN_LIMIT_CODE


def test_auth_error_detail_is_provider_aware():
    # MindsHub → reconnect; BYOK → fix your own key in Settings (no "reconnect").
    minds = te.auth_error_detail("MindsHub", reconnectable=True)
    assert "reconnect" in minds.lower()
    byok = te.auth_error_detail("OpenAI", reconnectable=False)
    assert "reconnect" not in byok.lower()
    assert "OpenAI" in byok and "Settings" in byok


def test_response_failed_payload_carries_auth_fields():
    p = te.response_failed_payload("msg", te.AUTH_ERROR_CODE, reconnectable=True, provider_label="MindsHub")
    assert p["reconnectable"] is True and p["provider_label"] == "MindsHub"
    # Unrelated failures keep the original shape (no extra keys).
    assert "reconnectable" not in te.response_failed_payload("boom", "anton_error")


# ── Model-403 (model_access_denied / model_disabled) → plan card ──────
#
# The gateway rejecting the requested MODEL (tier gate or admin kill switch)
# used to surface as the generic "Server returned 403 — temporarily
# unavailable" prose. anton now raises ModelUnavailableError carrying the
# gateway's structured code + the model alias; these tests pin the mapping.
# Detection is typed-or-duck-typed on the code/model attributes — the venv's
# anton may predate the class (version skew), which is exactly what the duck
# path covers. NO string matching: a message merely mentioning "model_disabled"
# must never trigger the plan card.


class _FakeModelErr(ConnectionError):
    """Duck-typed stand-in for anton's ModelUnavailableError."""

    def __init__(self, message, code, model):
        super().__init__(message)
        self.code = code
        self.model = model


_PLAN_MSG = (
    "The model 'sonnet' isn't included in your current MindsHub plan. "
    "Visit https://console.mindshub.ai to upgrade, or switch models in Settings."
)


def test_model_unavailable_detected_via_duck_typing():
    info = te.model_unavailable_info(_FakeModelErr(_PLAN_MSG, "model_access_denied", "sonnet"))
    assert info == ("model_access_denied", "sonnet")
    info = te.model_unavailable_info(_FakeModelErr("x", "model_disabled", "opus"))
    assert info == ("model_disabled", "opus")


def test_model_unavailable_requires_the_structured_code():
    # Unknown code attr, non-string code, or a message that merely mentions
    # the code → not a model-403.
    assert te.model_unavailable_info(_FakeModelErr("x", "other_code", "sonnet")) is None
    assert te.model_unavailable_info(_FakeModelErr("x", 403, "sonnet")) is None
    assert te.model_unavailable_info(Exception("error code model_disabled happened")) is None
    assert te.model_unavailable_info(ConnectionError("Server returned 403")) is None


def test_model_unavailable_maps_code_and_passes_message_through():
    # anton's message is already curated user copy — surfaced verbatim.
    code, message = te.friendly_turn_error(_FakeModelErr(_PLAN_MSG, "model_access_denied", "sonnet"))
    assert code == te.MODEL_ACCESS_DENIED_CODE == "model_access_denied"
    assert message == _PLAN_MSG


def test_model_unavailable_empty_message_gets_fallback_copy():
    code, message = te.friendly_turn_error(_FakeModelErr("", "model_disabled", "sonnet"))
    assert code == te.MODEL_DISABLED_CODE
    assert message == te.MODEL_UNAVAILABLE_FALLBACK_MESSAGE


def test_token_limit_wins_over_model_403():
    # A quota failure carrying a model-ish code attr must stay token_limit.
    exc = _FakeModelErr(_TOKEN_LIMIT_MESSAGE, "model_disabled", "sonnet")
    code, _ = te.friendly_turn_error(exc)
    assert code == te.TOKEN_LIMIT_CODE


def test_auth_error_not_shadowed_by_model_mapping():
    exc = ConnectionError("Invalid API key — check your OpenAI API key configuration.")
    code, _ = te.friendly_turn_error(exc)
    assert code == te.AUTH_ERROR_CODE


def test_response_failed_payload_carries_model_field():
    p = te.response_failed_payload(
        "msg", te.MODEL_ACCESS_DENIED_CODE, model="sonnet", provider_label="MindsHub"
    )
    assert p["model"] == "sonnet" and p["provider_label"] == "MindsHub"
    # Unrelated failures keep the original shape (no extra keys).
    assert "model" not in te.response_failed_payload("boom", "anton_error")


async def test_stream_emits_model_unavailable_with_extras():
    exc = _FakeModelErr(_PLAN_MSG, "model_access_denied", "sonnet")
    frames = await _collect_produce_sse(_handler_with_raising_formatter(exc))
    failed = [f for f in frames if "response.failed" in f]
    assert len(failed) == 1
    payload = json.loads(failed[0].split("data: ", 1)[1].strip())
    assert payload["code"] == "model_access_denied"
    assert payload["error"] == _PLAN_MSG
    assert payload["model"] == "sonnet"
    # No provider_label on the model-403 path — the card doesn't render it and
    # it would name the wrong provider when the coding model was rejected.
    assert "provider_label" not in payload


def test_collect_raises_400_with_plan_message_for_model_403():
    handler = _handler_with_raising_formatter(
        _FakeModelErr(_PLAN_MSG, "model_access_denied", "sonnet")
    )
    with pytest.raises(HTTPException) as err:
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", output_item_id="msg-1"))
    assert err.value.status_code == 400
    assert err.value.detail == _PLAN_MSG
