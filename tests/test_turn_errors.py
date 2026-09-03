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
import sys
from types import ModuleType
from uuid import uuid4

import pytest
from anton.core.llm.provider import ProviderAuthError
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


# ── content-shaped rejections (ENG-1992) ──────────────────────────────

def test_detects_content_validation_via_anton_type():
    provider = pytest.importorskip("anton.core.llm.provider")
    error_cls = getattr(provider, "ContentValidationError", None)
    if error_cls is None:
        pytest.skip("installed anton predates ContentValidationError (ENG-1992)")
    assert te.is_content_validation_error(error_cls("boom")) is True


def test_detects_openai_responses_dialect():
    # The exact phrasing from the live ENG-1992 incident.
    exc = Exception(
        "Invalid value: 'image'. Supported values are: 'input_text', "
        "'input_image', 'input_audio', 'output_text', 'refusal', "
        "'input_file', 'computer_screenshot', 'summary_text', and "
        "'encrypted_content'."
    )
    assert te.is_content_validation_error(exc) is True


def test_detects_anthropic_dialect_via_content_validation():
    # This is also the earlier, narrower is_image_format_error case — a
    # content-shape error can trigger both detectors; friendly_turn_error's
    # ordering decides which code wins (see below).
    exc = Exception(
        "Input tag 'image_url' found using 'type' does not match "
        "any of the expected tags: 'image'"
    )
    assert te.is_content_validation_error(exc) is True


def test_content_validation_ignores_unrelated_errors():
    assert te.is_content_validation_error(Exception("Internal server error")) is False
    assert te.is_content_validation_error(
        Exception("tool_use ids were found without tool_result blocks")
    ) is False


def test_maps_content_validation_error_to_curated_copy():
    result = te.friendly_turn_error(
        Exception("Invalid value: 'image'. Supported values are: 'input_text', 'input_image'")
    )
    assert result is not None
    code, message = result
    assert code == te.CONTENT_RECOVERY_CODE
    assert "fixed it automatically" in message
    # And NOT the old "re-upload as PNG/JPEG" copy — that advice is wrong
    # here; the failure isn't anything wrong with the image itself.
    assert "PNG" not in message


def test_content_validation_wins_over_image_format_for_the_full_anthropic_phrasing():
    # The REAL Anthropic dialect matches both detectors; content_recovery
    # must win, because that path actually repairs the conversation —
    # image_format's docstring says it explicitly can't.
    result = te.friendly_turn_error(
        Exception(
            "Input tag 'image_url' found using 'type' does not match "
            "any of the expected tags: 'image'"
        )
    )
    assert result is not None
    code, _ = result
    assert code == te.CONTENT_RECOVERY_CODE


def test_remote_content_validation_error_maps_to_curated_copy():
    code, message = te.remote_turn_error("ContentValidationError: some provider detail")
    assert code == te.CONTENT_RECOVERY_CODE
    assert message == te.CONTENT_RECOVERY_USER_MESSAGE


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
        *, conversation, input, model=None, reasoning_effort=None, disabled_connections=None,
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


async def test_produce_pending_persist_failure_does_not_clear_all_pending():
    # ENG-1231 hardening (in-process _produce, mirror of the _produce_remote test):
    # if the pending user persist raises before its id is captured, this turn owns
    # no pending row — persist() must NOT fall back to finalize_pending(conv, None),
    # which would clear a pending row stranded by an earlier crashed turn.
    from unittest.mock import MagicMock, patch

    handler = _handler_with_raising_formatter(Exception("unused — save fails first"))

    class _Buffer:
        async def append(self, _kind, data):
            pass

        async def close(self, _status):
            pass

    with (
        patch("cowork.handlers.responses.get_open_session", return_value=MagicMock()),
        patch("cowork.handlers.responses.ConversationService") as conv_svc,
        patch("cowork.handlers.responses.get_harness", return_value=handler.harness),
    ):
        conv_svc.return_value.get_conversation.return_value = MagicMock()
        conv_svc.return_value.save_user_message.side_effect = RuntimeError("db down")
        await handler._produce(
            conv_id=uuid4(),
            harness_input=[{"type": "text", "text": "hi"}],
            original_content="hi",
            model="anton",
            disabled=None,
            harness_name="anton",
            harness_id="anton",
            buffer=_Buffer(),
        )
        # No row was persisted for this turn → finalize must not have run at all,
        # in particular never the clear-all (message_id=None) form.
        conv_svc.return_value.finalize_pending.assert_not_called()


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
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", original_content="hi"))
    assert err.value.status_code == 400
    assert "PNG or JPEG" in err.value.detail


def test_collect_raises_500_generic_for_unmapped_error():
    handler = _handler_with_raising_formatter(Exception("kaboom: secret-token-xyz"))
    with pytest.raises(HTTPException) as err:
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", original_content="hi"))
    assert err.value.status_code == 500
    assert err.value.detail == te.GENERIC_TURN_ERROR_MESSAGE
    assert "secret-token" not in err.value.detail


# ── Conversation repair on content validation error (ENG-1992) ────

def test_stream_repairs_conversation_on_content_validation_error():
    from unittest.mock import MagicMock, patch

    exc = Exception("Invalid value: 'image'. Supported values are: 'input_text', 'input_image'")
    handler = _handler_with_raising_formatter(exc)

    class _Buffer:
        async def append(self, _kind, data):
            pass

        async def close(self, _status):
            pass

    conv_id = uuid4()
    with (
        patch("cowork.handlers.responses.get_open_session", return_value=MagicMock()),
        patch("cowork.handlers.responses.ConversationService") as conv_svc,
        patch("cowork.handlers.responses.get_harness", return_value=handler.harness),
    ):
        conv_svc.return_value.get_conversation.return_value = MagicMock()
        conv_svc.return_value.repair_image_content.return_value = [uuid4()]
        asyncio.run(handler._produce(
            conv_id=conv_id,
            harness_input=[{"type": "text", "text": "hi"}],
            original_content="hi",
            model="anton",
            disabled=None,
            harness_name="anton",
            harness_id="anton",
            buffer=_Buffer(),
        ))
        conv_svc.return_value.repair_image_content.assert_called_once_with(conv_id)


def test_stream_does_not_repair_conversation_for_unrelated_errors():
    from unittest.mock import MagicMock, patch

    handler = _handler_with_raising_formatter(Exception("boom, totally unrelated"))

    class _Buffer:
        async def append(self, _kind, data):
            pass

        async def close(self, _status):
            pass

    conv_id = uuid4()
    with (
        patch("cowork.handlers.responses.get_open_session", return_value=MagicMock()),
        patch("cowork.handlers.responses.ConversationService") as conv_svc,
        patch("cowork.handlers.responses.get_harness", return_value=handler.harness),
    ):
        conv_svc.return_value.get_conversation.return_value = MagicMock()
        asyncio.run(handler._produce(
            conv_id=conv_id,
            harness_input=[{"type": "text", "text": "hi"}],
            original_content="hi",
            model="anton",
            disabled=None,
            harness_name="anton",
            harness_id="anton",
            buffer=_Buffer(),
        ))
        conv_svc.return_value.repair_image_content.assert_not_called()


def test_collect_repairs_conversation_on_content_validation_error():
    from unittest.mock import MagicMock, patch

    exc = Exception("Invalid value: 'image'. Supported values are: 'input_text', 'input_image'")
    handler = _handler_with_raising_formatter(exc)
    handler.scoped = MagicMock()  # __init__ bypassed; _collect's repair path needs this
    conv_id = uuid4()

    with patch("cowork.handlers.responses.ConversationService") as conv_svc:
        conv_svc.return_value.repair_image_content.return_value = [uuid4()]
        with pytest.raises(HTTPException) as err:
            asyncio.run(handler._collect(stream=None, conversation_id=conv_id, model="anton", original_content="hi"))
        assert err.value.status_code == 400
        conv_svc.return_value.repair_image_content.assert_called_once_with(conv_id)


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
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", original_content="hi"))
    assert err.value.status_code == 400
    assert err.value.detail == te.TOKEN_LIMIT_USER_MESSAGE


# ── Provider auth (401) → provider_auth ──────────────────────────────


def test_detects_canonical_provider_auth_error():
    exc = ProviderAuthError("provider rejected the credential")
    assert te.is_auth_error(exc) is True


def test_legacy_auth_fallback_when_anton_lacks_typed_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "anton.core.llm.provider", ModuleType("anton.core.llm.provider")
    )
    exc = ConnectionError(
        "Invalid API key — check your OpenAI API key configuration."
    )

    assert te.is_auth_error(exc) is True


@pytest.mark.parametrize(
    "message",
    [
        "Invalid API key — check your OpenAI API key configuration.",
        "Invalid API key — check your ANTHROPIC_API_KEY environment variable.",
    ],
)
def test_connection_error_auth_lookalikes_are_not_provider_auth(message):
    exc = ConnectionError(message)
    assert te.is_auth_error(exc) is False
    assert te.friendly_turn_error(exc) is None


def test_bare_401_not_flagged():
    # Tightened: a 401/"unauthorized" without anton's specific "Invalid API key"
    # copy is NOT a provider-auth error (avoids mislabeling e.g. a tool API 401).
    assert te.is_auth_error(Exception("Server returned 401 — Unauthorized")) is False
    assert te.is_auth_error(Exception("connection reset")) is False


def test_auth_error_maps_to_provider_auth_code():
    code, message = te.friendly_turn_error(
        ProviderAuthError("provider rejected the credential")
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


async def test_auth_reconnectable_keys_on_the_failing_role_not_planning():
    from unittest.mock import patch
    from cowork.common.settings.user_settings import Provider

    class _FakeSettings:
        resolved_planning_provider = Provider.MINDS_CLOUD
        resolved_coding_provider = Provider.ANTHROPIC
        resolved_router_provider = Provider.OPENAI

    exc = ProviderAuthError("provider rejected the credential")
    exc.role = "coding"
    with patch("cowork.handlers.responses.get_user_settings", return_value=_FakeSettings()):
        frames = await _collect_produce_sse(_handler_with_raising_formatter(exc))

    payload = json.loads(
        [f for f in frames if "response.failed" in f][0].split("data: ", 1)[1].strip()
    )
    assert payload["code"] == te.AUTH_ERROR_CODE
    assert payload["reconnectable"] is False
    assert payload["provider_label"] == Provider.ANTHROPIC.label


async def test_auth_reconnectable_uses_planning_role_in_a_mixed_config():
    from unittest.mock import patch

    from cowork.common.settings.user_settings import Provider

    class _MixedSettings:
        resolved_planning_provider = Provider.MINDS_CLOUD
        resolved_coding_provider = Provider.ANTHROPIC
        resolved_router_provider = Provider.OPENAI

    exc = ProviderAuthError("provider rejected the credential")
    exc.role = "planning"
    with patch(
        "cowork.handlers.responses.get_user_settings",
        return_value=_MixedSettings(),
    ):
        frames = await _collect_produce_sse(_handler_with_raising_formatter(exc))

    failed = next(f for f in frames if "response.failed" in f)
    payload = json.loads(failed.split("data: ", 1)[1].strip())
    assert payload["code"] == te.AUTH_ERROR_CODE
    assert payload["reconnectable"] is True
    assert payload["provider_label"] == Provider.MINDS_CLOUD.label


async def test_auth_without_a_role_does_not_name_a_provider_in_a_mixed_config():
    """An auth error can reach the handler without a role stamped on it.

    Defaulting to planning would show a MindsHub "Reconnect" card for a failure
    that may have been the BYOK Anthropic key, so an unattributable auth error
    keeps the generic copy and no provider fields.
    """
    from unittest.mock import patch
    from cowork.common.settings.user_settings import Provider

    class _MixedSettings:
        resolved_planning_provider = Provider.MINDS_CLOUD
        resolved_coding_provider = Provider.ANTHROPIC
        resolved_router_provider = Provider.OPENAI

    # role is never stamped: ProviderAuthError defaults it to None, and only
    # LLMClient's confirmation wrappers set it.
    exc = ProviderAuthError("provider rejected the credential")
    assert exc.role is None
    with patch("cowork.handlers.responses.get_user_settings", return_value=_MixedSettings()):
        frames = await _collect_produce_sse(_handler_with_raising_formatter(exc))

    payload = json.loads(
        [f for f in frames if "response.failed" in f][0].split("data: ", 1)[1].strip()
    )
    assert payload["code"] == te.AUTH_ERROR_CODE
    assert "reconnectable" not in payload
    assert "provider_label" not in payload


async def test_auth_without_a_role_still_names_an_unambiguous_provider():
    """Both required roles agree, so there is nothing to attribute wrongly."""
    from unittest.mock import patch
    from cowork.common.settings.user_settings import Provider

    class _MindsSettings:
        resolved_planning_provider = Provider.MINDS_CLOUD
        resolved_coding_provider = Provider.MINDS_CLOUD
        resolved_router_provider = Provider.OPENAI

    exc = ProviderAuthError("provider rejected the credential")
    with patch("cowork.handlers.responses.get_user_settings", return_value=_MindsSettings()):
        frames = await _collect_produce_sse(_handler_with_raising_formatter(exc))

    payload = json.loads(
        [f for f in frames if "response.failed" in f][0].split("data: ", 1)[1].strip()
    )
    assert payload["code"] == te.AUTH_ERROR_CODE
    assert payload["reconnectable"] is True
    assert payload["provider_label"] == Provider.MINDS_CLOUD.label


# ── Model-403 (model_access_denied / model_disabled), legacy back-compat ─
#
# Only pre-wallet gateway/anton versions emit these structured codes (a
# plan/tier exclusion or an admin kill switch); the current gateway denies a
# wallet-locked model as 402 wallet_empty instead. The branch is kept so a
# version-skewed deployment still gets curated copy rather than the generic
# "Server returned 403" prose. Detection is typed-or-duck-typed on the
# code/model attributes — the venv's anton may predate the class (version
# skew), which is exactly what the duck path covers. NO string matching: a
# message merely mentioning "model_disabled" must never trigger the card.


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
    exc = ProviderAuthError("provider rejected the credential")
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
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", original_content="hi"))
    assert err.value.status_code == 400
    assert err.value.detail == _PLAN_MSG


# ── Wallet-model gateway mapping (402/429/404/503 + X-MindsHub-Reason) ─
#
# The inference gateway now denies calls with a precise HTTP status plus an
# X-MindsHub-Reason header (wallet_empty / included_allowance_exhausted /
# policy_unavailable / unknown_model). anton wraps the provider SDK's
# APIStatusError (which carries the status + response headers + request URL)
# in a ConnectionError via `raise ... from`, so the structured detail lives on
# the chained cause. These tests pin that we prefer the header, fall back to
# the bare status ONLY when the failing request went to the MindsHub gateway
# (a BYOK provider's own 402/429/503 must stay generic), and never mislabel a
# transient 503 as out-of-credits.


class _FakeHeaders(dict):
    """Case-insensitive .get(), like httpx.Headers."""

    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


class _FakeResponse:
    def __init__(self, headers, url=None):
        self.headers = _FakeHeaders(headers or {})
        if url is not None:
            self.url = url


class _FakeAPIStatusError(Exception):
    """Stand-in for openai.APIStatusError — carries status_code + response."""

    def __init__(self, status_code, headers=None, message="upstream error", url=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeResponse(headers, url=url)


def _minds_gateway_url() -> str:
    """A request URL on the host this install treats as the MindsHub gateway."""
    return f"https://{te._configured_minds_host()}/v1/chat/completions"


def _failure(status_code, reason=None, message="Server returned an upstream error", url=None):
    """The exception anton surfaces: a ConnectionError wrapping the SDK's
    APIStatusError (chained via `raise ... from`), carrying the HTTP status,
    any X-MindsHub-Reason header, and the request URL."""
    headers = {"X-MindsHub-Reason": reason} if reason else {}
    wrapped = ConnectionError(message)
    wrapped.__cause__ = _FakeAPIStatusError(status_code, headers, url=url)
    return wrapped


def _gateway_failure(status_code, reason=None, message="Server returned an upstream error"):
    """A failure whose request went to the configured MindsHub gateway."""
    return _failure(status_code, reason=reason, message=message, url=_minds_gateway_url())


def _byok_failure(status_code, message="Server returned an upstream error"):
    """A failure from the user's own provider (BYOK), not the gateway."""
    return _failure(status_code, message=message, url="https://api.openai.com/v1/chat/completions")


@pytest.mark.parametrize(
    ("status", "reason", "expected_code"),
    [
        (402, "wallet_empty", te.TOKEN_LIMIT_CODE),
        (429, "included_allowance_exhausted", te.ALLOWANCE_EXHAUSTED_CODE),
        (429, "rate_limited", te.RATE_LIMITED_CODE),
        (503, "policy_unavailable", te.POLICY_UNAVAILABLE_CODE),
        (404, "unknown_model", te.MODEL_NOT_FOUND_CODE),
    ],
)
def test_typed_auth_narrowing_preserves_gateway_reason_mapping(
    status, reason, expected_code
):
    code, _ = te.friendly_turn_error(_gateway_failure(status, reason=reason))
    assert code == expected_code


def test_http_error_context_walks_cause_chain():
    status, reason, host = te._http_error_context(_gateway_failure(402, reason="wallet_empty"))
    assert status == 402
    assert reason == "wallet_empty"
    assert host == te._configured_minds_host()


def test_plain_exception_has_no_http_context():
    assert te._http_error_context(Exception("boom")) == (None, None, None)


def test_status_and_host_come_from_the_reason_bearing_exception():
    # A chain where an earlier exception carries a different status: the
    # status/host reported must belong to the SAME exception as the header,
    # never a mix of chain entries.
    outer = ConnectionError("wrapper")
    outer.__cause__ = mid = _FakeAPIStatusError(500, url="https://api.openai.com/v1/x")
    mid.__cause__ = _FakeAPIStatusError(
        402, {"X-MindsHub-Reason": "wallet_empty"}, url=_minds_gateway_url()
    )
    status, reason, host = te._http_error_context(outer)
    assert (status, reason, host) == (402, "wallet_empty", te._configured_minds_host())


def test_reason_header_wallet_empty_maps_to_out_of_credits():
    code, message = te.friendly_turn_error(_gateway_failure(402, reason="wallet_empty"))
    assert code == te.TOKEN_LIMIT_CODE
    assert message == te.TOKEN_LIMIT_USER_MESSAGE


def test_spent_free_allowance_is_its_own_card_not_out_of_credits():
    # ENG-1537. These used to share the credits card, but they are different
    # situations: `access.py` only issues this reason for a free-bucket model on
    # an org that has NEVER topped up, so the user has not spent money — they
    # used the monthly grant, which resets. Telling them "you're out of credits"
    # both misdescribes it and hides the free way forward.
    code, message = te.friendly_turn_error(
        _gateway_failure(429, reason="included_allowance_exhausted")
    )
    assert code == te.ALLOWANCE_EXHAUSTED_CODE
    assert code != te.TOKEN_LIMIT_CODE
    assert message == te.ALLOWANCE_EXHAUSTED_USER_MESSAGE
    # Still an actionable path to keep working — ENG-1169's requirement holds
    # even though the code changed.
    assert "add credits" in message.lower()


def test_empty_wallet_keeps_the_out_of_credits_card():
    # The other half of the split must be untouched: a drained wallet really is
    # "out of credits" and keeps its existing card.
    code, message = te.friendly_turn_error(_gateway_failure(402, reason="wallet_empty"))
    assert code == te.TOKEN_LIMIT_CODE
    assert message == te.TOKEN_LIMIT_USER_MESSAGE


def test_allowance_reset_at_is_read_off_the_chain():
    # The gate sends this on the allowance denial and NOT on a velocity one, so
    # the card can name when the grant refreshes instead of only asking for money.
    exc = _gateway_failure(429, reason="included_allowance_exhausted")
    exc.__cause__.response.headers["X-MindsHub-Reset-At"] = "2026-09-01T00:00:00Z"
    assert te.allowance_reset_at(exc) == "2026-09-01T00:00:00Z"
    # Absent → the renderer falls back to "resets next month"; never invented here.
    assert te.allowance_reset_at(_gateway_failure(429, reason="included_allowance_exhausted")) is None
    assert te.allowance_reset_at(Exception("bare")) is None


def test_reset_at_rides_the_failed_payload_only_when_present():
    with_reset = te.response_failed_payload(
        "msg", te.ALLOWANCE_EXHAUSTED_CODE, reset_at="2026-09-01T00:00:00Z"
    )
    assert with_reset["reset_at"] == "2026-09-01T00:00:00Z"
    assert "reset_at" not in te.response_failed_payload("msg", te.TOKEN_LIMIT_CODE)


# ── The two 429 flavours must never share a card (ENG-1537) ────────


def test_velocity_rate_limit_is_not_out_of_credits():
    # THE defect. `rate_limited` was the one gateway reason this module didn't
    # know, so it fell to the bare-status 429 rule and rendered as
    # "You're out of credits. Add credits to keep working." — advertising a
    # purchase that cannot lift a per-minute token ceiling.
    code, message = te.friendly_turn_error(_gateway_failure(429, reason="rate_limited"))
    assert code == te.RATE_LIMITED_CODE
    assert message == te.RATE_LIMITED_USER_MESSAGE
    assert code != te.TOKEN_LIMIT_CODE
    # The copy must not send the user to billing, in either direction.
    assert "add credits" not in message.lower()


def test_velocity_rate_limit_maps_from_the_body_code_when_the_header_is_lost():
    # ENG-1363: the Anthropic /v1/messages lane strips X-MindsHub-* headers.
    # The gateway sets `code` and `reason` to the same value, so the body is a
    # second carrier for the identical decision — without it, that lane would
    # fall to the bare-status rule and show the credits card again.
    exc = ConnectionError("Server returned 429")
    exc.__cause__ = inner = _FakeAPIStatusError(429, {}, url=_minds_gateway_url())
    inner.body = {"error": {"code": "rate_limited", "message": "Rate limit exceeded"}}
    code, _ = te.friendly_turn_error(exc)
    assert code == te.RATE_LIMITED_CODE


@pytest.mark.parametrize("code", ["wallet_empty", "rate_limited", "included_allowance_exhausted"])
def test_a_third_party_body_cannot_select_a_billing_verdict(code):
    # ENG-1537 review. The body-`code` carrier must be host-gated exactly like
    # the bare-status rule. A response body is third-party-controlled on a BYOK
    # OPENAI_COMPATIBLE provider, so without the gate any endpoint could send
    # {"code": "wallet_empty"} and put our billing CTA — and the MindsHub top-up
    # link — in front of a user with no MindsHub balance at all.
    #
    # The allowlist alone does NOT prevent this: it constrains which verdict can
    # be selected, not who may select one. That is why these are the codes we
    # recognise rather than junk.
    exc = ConnectionError("Server returned 429")
    exc.__cause__ = inner = _FakeAPIStatusError(429, {}, url="https://openrouter.ai/api/v1/x")
    inner.body = {"error": {"code": code}}
    result = te.friendly_turn_error(exc)
    assert result is None or result[0] not in (
        te.TOKEN_LIMIT_CODE, te.RATE_LIMITED_CODE, te.ALLOWANCE_EXHAUSTED_CODE,
    ), f"a third-party body selected {result!r}"


def test_the_gateways_own_body_code_still_maps():
    # The gate must not break the carrier it exists to protect: on OUR host the
    # body code is still honoured when the header didn't survive (ENG-1363).
    exc = ConnectionError("Server returned 429")
    exc.__cause__ = inner = _FakeAPIStatusError(429, {}, url=_minds_gateway_url())
    inner.body = {"error": {"code": "rate_limited"}}
    assert te.friendly_turn_error(exc)[0] == te.RATE_LIMITED_CODE


def test_exhausted_rate_limit_wait_keeps_its_code_over_the_bare_status_rule():
    # ENG-1537: when anton's wait budget runs out it re-raises with
    # code="rate_limited", but the ORIGINAL 429 is still in the cause chain —
    # so the bare-status rule would relabel the honest "waiting didn't clear
    # it" failure as out-of-credits, undoing the whole point of the wait.
    exhausted = _FakeOverloadedErr(
        "Too many requests too quickly — the rate limit didn't clear in time.",
        code="rate_limited",
        model="sonnet",
    )
    # NO reason header and NO body code: the ONLY thing that can save this from
    # the bare-status 429 rule is the hoisted code check. The earlier version of
    # this test passed the header, so it never exercised the hoist it is named
    # after — deleting the whole block left it green (ENG-1537 review).
    exhausted.__cause__ = _FakeAPIStatusError(429, {}, url=_minds_gateway_url())
    code, _ = te.friendly_turn_error(exhausted)
    assert code == te.RATE_LIMITED_CODE


def test_a_real_provider_incident_still_maps_to_provider_overloaded():
    # The rate-limit early check must not swallow the incident case it sits in
    # front of.
    incident = _FakeOverloadedErr(
        "Anthropic is experiencing an incident and didn't recover in time.",
        code="provider_overloaded",
        model="sonnet",
    )
    assert te.friendly_turn_error(incident)[0] == te.PROVIDER_OVERLOADED_CODE


@pytest.mark.parametrize("status,reason,expected", [
    (402, "wallet_empty", "token_limit"),
    (429, "included_allowance_exhausted", "included_allowance_exhausted"),
])
def test_billing_denials_still_card_immediately(status, reason, expected):
    # ENG-1169 regression guard, in the other direction. These share the 429
    # status (and 402) with the velocity limit but are permanent for the
    # identical request — the allowance resets monthly — so they must keep
    # going straight to the credits card and must never be routed to a wait.
    code, message = te.friendly_turn_error(_gateway_failure(status, reason=reason))
    assert code == expected
    # Whichever card it is, it must offer the user a way to keep working.
    assert "credits" in message.lower()


def test_retry_after_is_read_off_the_chain_for_the_card_gate():
    # ENG-1537: the renderer needs the server's own interval to time-gate its
    # Retry. Integer seconds only — a date form would gate the button for
    # centuries, so it is dropped in favour of no gate.
    exc = _gateway_failure(429, reason="rate_limited")
    exc.__cause__.response.headers["Retry-After"] = "30"
    assert te.retry_after_seconds(exc) == 30.0

    dated = _gateway_failure(429, reason="rate_limited")
    dated.__cause__.response.headers["Retry-After"] = "Wed, 21 Oct 2026 07:28:00 GMT"
    assert te.retry_after_seconds(dated) is None

    assert te.retry_after_seconds(_gateway_failure(429, reason="rate_limited")) is None
    assert te.retry_after_seconds(Exception("bare")) is None

    # Clamped at the source so the interval and the instant never disagree on
    # the wire (review: pnewsam). Unclamped, the payload carried
    # retry_after=999999999999 while retry_at was dropped as out-of-range —
    # two fields describing one wait, one absurd and one absent.
    huge = _gateway_failure(429, reason="rate_limited")
    huge.__cause__.response.headers["Retry-After"] = "999999999999"
    assert te.retry_after_seconds(huge) == te._MAX_RETRY_AFTER_S
    # And the pair it feeds is therefore consistent: both present, both bounded.
    _a = te.retry_after_seconds(huge)
    _p = te.response_failed_payload(
        "m", te.RATE_LIMITED_CODE, retry_after=_a, retry_at=te.retry_at_instant(_a),
    )
    assert _p["retry_after"] == te._MAX_RETRY_AFTER_S
    assert _p["retry_at"] is not None


def test_retry_after_rides_the_failed_payload_only_when_present():
    # Additive field: absent unless we actually have a number, so the wire shape
    # is unchanged for every other failure and older clients are unaffected.
    with_hint = te.response_failed_payload("msg", te.RATE_LIMITED_CODE, retry_after=30.0)
    assert with_hint["retry_after"] == 30.0
    assert "retry_after" not in te.response_failed_payload("msg", te.TOKEN_LIMIT_CODE)


def test_reasonless_gateway_429_still_cards_as_credits():
    # A gateway old enough to omit the header only ever meant "allowance" by a
    # 429, so the legacy assumption is preserved for a 429 carrying neither a
    # reason nor a body code. Narrowing this instead would have stripped the
    # credits card from a real allowance exhaustion.
    code, _ = te.friendly_turn_error(_gateway_failure(429))
    assert code == te.TOKEN_LIMIT_CODE


def test_reason_header_policy_unavailable_is_transient_not_out_of_credits():
    code, message = te.friendly_turn_error(_gateway_failure(503, reason="policy_unavailable"))
    assert code == te.POLICY_UNAVAILABLE_CODE
    assert code != te.TOKEN_LIMIT_CODE
    assert message == te.POLICY_UNAVAILABLE_USER_MESSAGE


def test_reason_header_unknown_model_steers_to_settings_not_credits():
    code, message = te.friendly_turn_error(_gateway_failure(404, reason="unknown_model"))
    assert code == te.MODEL_NOT_FOUND_CODE
    assert code != te.TOKEN_LIMIT_CODE
    assert message == te.MODEL_NOT_FOUND_USER_MESSAGE


def test_unknown_model_prefers_antons_model_naming_copy_over_the_header():
    """ENG-1358: the gateway 404 carries BOTH the reason header and anton's typed
    ModelUnavailableError. The header's copy is generic ("That model isn't
    available"); anton's names the offending id. The user can only act on the
    latter, so it must win — returning the header copy is what left ENG-1358's
    user with three dead turns and no idea which model was wrong.
    """
    exc = _FakeModelErr(
        "The model 'deepseek-v4-flash' isn't available: The model "
        "'deepseek-v4-flash' does not exist or you do not have access to it. "
        "Switch models in Settings.",
        "model_not_found",
        "deepseek-v4-flash",
    )
    exc.__cause__ = _gateway_failure(404, reason="unknown_model")

    code, message = te.friendly_turn_error(exc)
    assert code == te.MODEL_NOT_FOUND_CODE
    assert "deepseek-v4-flash" in message
    assert message != te.MODEL_NOT_FOUND_USER_MESSAGE


def test_model_not_found_is_a_model_unavailable_code():
    """The renderer keys one card on this set; model_not_found must be in it or
    the 404 falls through to a plain text line with no action (ENG-1358)."""
    exc = _FakeModelErr(
        "The model 'x' isn't available. Switch models in Settings.",
        "model_not_found",
        "x",
    )
    assert te.model_unavailable_info(exc) == ("model_not_found", "x")


def test_remote_model_unavailable_does_not_promise_credits_will_fix_it():
    """The remote wire loses the structured code, so a 404 and a legacy 403 look
    identical. Defaulting to model_access_denied would render a "Top up balance"
    button for a model that simply doesn't exist."""
    code, message = te.remote_turn_error(
        "ModelUnavailableError: The model 'deepseek-v4-flash' isn't available. "
        "Switch models in Settings."
    )
    assert code == te.MODEL_NOT_FOUND_CODE
    assert "deepseek-v4-flash" in message


def test_bare_402_status_maps_to_out_of_credits_without_header():
    # Older gateway with no reason header: the 402 status from the gateway's
    # host is enough.
    code, _ = te.friendly_turn_error(_gateway_failure(402))
    assert code == te.TOKEN_LIMIT_CODE


def test_bare_429_status_maps_to_out_of_credits_without_header():
    code, _ = te.friendly_turn_error(_gateway_failure(429))
    assert code == te.TOKEN_LIMIT_CODE


def test_bare_503_status_maps_to_transient_without_header():
    code, message = te.friendly_turn_error(_gateway_failure(503))
    assert code == te.POLICY_UNAVAILABLE_CODE
    assert message == te.POLICY_UNAVAILABLE_USER_MESSAGE


def test_byok_402_stays_generic():
    # A BYOK provider's own 402 is not a gateway billing decision — it must
    # NOT surface the "add credits" card (the user has no wallet to top up
    # for that key).
    assert te.friendly_turn_error(_byok_failure(402)) is None


def test_byok_429_stays_generic():
    # An OpenAI/Anthropic rate limit on the user's own key must not be
    # presented as out-of-credits.
    assert te.friendly_turn_error(_byok_failure(429)) is None


def test_byok_503_stays_generic():
    # A BYOK provider outage is not "Billing is temporarily unavailable".
    assert te.friendly_turn_error(_byok_failure(503)) is None


def test_bare_status_with_unknown_origin_stays_generic():
    # No request URL on the failure → origin can't be proven → the bare-status
    # billing fallbacks must not fire.
    assert te.friendly_turn_error(_failure(402)) is None
    assert te.friendly_turn_error(_failure(503)) is None


def test_reason_header_maps_even_without_request_url():
    # Only the gateway sets X-MindsHub-Reason, so the header path stays
    # unconditional on origin — it maps even when the response carries no URL.
    code, _ = te.friendly_turn_error(_failure(402, reason="wallet_empty"))
    assert code == te.TOKEN_LIMIT_CODE


def test_raising_url_property_never_escapes_the_error_handler():
    # httpx.Response.url is a property that RAISES (RuntimeError) when the
    # response has no request attached; friendly_turn_error runs inside except
    # handlers and must never raise, so origin extraction has to swallow it.
    import httpx

    wrapped = ConnectionError("Server returned an upstream error")
    err = _FakeAPIStatusError(402)
    err.response = httpx.Response(402)  # no request → .url raises
    wrapped.__cause__ = err
    assert te._response_url_host(err.response) is None
    # Origin unprovable → the bare-status billing fallback stays generic.
    assert te.friendly_turn_error(wrapped) is None


def test_bare_404_stays_generic_even_from_the_gateway():
    # Deliberate asymmetry vs 402/429/503: a header-less 404 is any missing
    # route/resource, not necessarily an unknown model, so it is never mapped
    # to unknown_model on status alone.
    assert te.friendly_turn_error(_gateway_failure(404)) is None


def test_reason_header_wins_over_status():
    # A 503 carrying an out-of-credits reason maps to out-of-credits — the
    # header is preferred over the status code.
    code, _ = te.friendly_turn_error(_gateway_failure(503, reason="wallet_empty"))
    assert code == te.TOKEN_LIMIT_CODE


def test_untyped_gateway_401_stays_generic():
    # A status and familiar copy cannot prove that the active LLM credential is
    # invalid. Only Anton's canonical typed exception selects provider_auth.
    exc = _gateway_failure(
        401, message="Invalid API key — check your OpenAI API key configuration."
    )
    assert te.friendly_turn_error(exc) is None


def test_typed_gateway_401_maps_to_provider_auth():
    exc = ProviderAuthError("provider rejected the credential")
    exc.__cause__ = _FakeAPIStatusError(401, url=_minds_gateway_url())
    code, _ = te.friendly_turn_error(exc)
    assert code == te.AUTH_ERROR_CODE


def test_out_of_credits_copy_is_credits_oriented_not_plan():
    # Copy must speak wallet/credits, never plans/tiers/upgrades.
    lowered = te.TOKEN_LIMIT_USER_MESSAGE.lower()
    assert "credit" in lowered
    assert "plan" not in lowered and "upgrade" not in lowered and "tier" not in lowered


async def test_stream_emits_transient_failed_event_for_policy_unavailable():
    frames = await _collect_produce_sse(
        _handler_with_raising_formatter(_gateway_failure(503, reason="policy_unavailable"))
    )
    failed = [f for f in frames if "response.failed" in f]
    assert len(failed) == 1
    payload = json.loads(failed[0].split("data: ", 1)[1].strip())
    assert payload["code"] == te.POLICY_UNAVAILABLE_CODE
    assert payload["error"] == te.POLICY_UNAVAILABLE_USER_MESSAGE
    # Flows through the generic path — no auth/model extras leak in.
    assert "reconnectable" not in payload and "model" not in payload


# ── provider_overloaded (ENG-673) ────────────────────────────────────
# A transient provider incident that outlasted anton's retry budget surfaces as
# anton's ProviderOverloadedError (code=provider_overloaded + model). Same
# typed-or-duck-typed detection as the model-403 case; NO string matching.


class _FakeOverloadedErr(ConnectionError):
    """Duck-typed stand-in for anton's ProviderOverloadedError."""

    def __init__(self, message, code="provider_overloaded", model="", provider=""):
        super().__init__(message)
        self.code = code
        self.model = model
        self.provider = provider


_OVERLOAD_MSG = "Anthropic is experiencing an incident and didn't recover in time."


def test_provider_overloaded_detected_via_duck_typing():
    info = te.provider_overloaded_info(_FakeOverloadedErr(_OVERLOAD_MSG, model="sonnet"))
    assert info == ("provider_overloaded", "sonnet")


def test_provider_overloaded_requires_the_structured_code():
    assert te.provider_overloaded_info(_FakeOverloadedErr("x", code="other")) is None
    # A message merely mentioning the words must not trigger the card.
    assert te.provider_overloaded_info(ConnectionError("provider_overloaded happened")) is None
    assert te.provider_overloaded_info(ConnectionError("Server returned 500")) is None


def test_provider_overloaded_maps_code_and_passes_message_through():
    code, message = te.friendly_turn_error(_FakeOverloadedErr(_OVERLOAD_MSG, model="sonnet"))
    assert code == te.PROVIDER_OVERLOADED_CODE == "provider_overloaded"
    assert message == _OVERLOAD_MSG


def test_provider_overloaded_empty_message_gets_fallback_copy():
    code, message = te.friendly_turn_error(_FakeOverloadedErr(""))
    assert code == te.PROVIDER_OVERLOADED_CODE
    assert message == te.PROVIDER_OVERLOADED_FALLBACK_MESSAGE


def test_token_limit_wins_over_provider_overloaded():
    # A quota failure carrying an overload-ish code must stay token_limit.
    exc = _FakeOverloadedErr(_TOKEN_LIMIT_MESSAGE)
    code, _ = te.friendly_turn_error(exc)
    assert code == te.TOKEN_LIMIT_CODE


def test_response_failed_payload_carries_overload_fields():
    p = te.response_failed_payload(
        _OVERLOAD_MSG, te.PROVIDER_OVERLOADED_CODE,
        model="sonnet", provider_label="Anthropic", reconnectable=False,
    )
    assert p["model"] == "sonnet"
    assert p["provider_label"] == "Anthropic"
    assert p["reconnectable"] is False


async def test_stream_emits_provider_overloaded_with_model():
    exc = _FakeOverloadedErr(_OVERLOAD_MSG, model="sonnet")
    frames = await _collect_produce_sse(_handler_with_raising_formatter(exc))
    failed = [f for f in frames if "response.failed" in f]
    assert len(failed) == 1
    payload = json.loads(failed[0].split("data: ", 1)[1].strip())
    assert payload["code"] == "provider_overloaded"
    assert payload["error"] == _OVERLOAD_MSG
    assert payload["model"] == "sonnet"


async def test_overloaded_reconnectable_keys_on_the_failing_model_not_planning():
    # ENG-673 (Sam's review): planning=MindsHub, coding=BYOK. When the CODING
    # model overloads, the card must reflect the BYOK provider that actually
    # failed — reconnectable=False so the MindsHub failover nudge is shown — NOT
    # reconnectable=True (which planning=MindsHub would wrongly imply, suppressing
    # the nudge that would help).
    from unittest.mock import patch
    from cowork.common.settings.user_settings import Provider

    class _FakeSettings:
        resolved_planning_model = "latest:sonnet"
        resolved_coding_model = "latest:haiku"
        resolved_planning_provider = Provider.MINDS_CLOUD
        resolved_coding_provider = Provider.ANTHROPIC

    exc = _FakeOverloadedErr(_OVERLOAD_MSG, model="latest:haiku")  # the coding model
    with patch("cowork.handlers.responses.get_user_settings", return_value=_FakeSettings()):
        frames = await _collect_produce_sse(_handler_with_raising_formatter(exc))
    payload = json.loads(
        [f for f in frames if "response.failed" in f][0].split("data: ", 1)[1].strip()
    )
    assert payload["code"] == "provider_overloaded"
    assert payload["model"] == "latest:haiku"
    assert payload["reconnectable"] is False
    assert payload["provider_label"] == Provider.ANTHROPIC.label


async def test_overloaded_reconnectable_true_when_failing_model_is_managed():
    # The mirror case: the failing (planning) model is on MindsHub Cloud → already
    # routed through failover, so no pitch — reconnectable=True (Retry-only).
    from unittest.mock import patch
    from cowork.common.settings.user_settings import Provider

    class _FakeSettings:
        resolved_planning_model = "latest:sonnet"
        resolved_coding_model = "latest:haiku"
        resolved_planning_provider = Provider.MINDS_CLOUD
        resolved_coding_provider = Provider.MINDS_CLOUD

    exc = _FakeOverloadedErr(_OVERLOAD_MSG, model="latest:sonnet")  # the planning model
    with patch("cowork.handlers.responses.get_user_settings", return_value=_FakeSettings()):
        frames = await _collect_produce_sse(_handler_with_raising_formatter(exc))
    payload = json.loads(
        [f for f in frames if "response.failed" in f][0].split("data: ", 1)[1].strip()
    )
    assert payload["reconnectable"] is True


# -- remote_turn_error: string classification for pod turn_failed errors ------

def test_remote_error_token_limit():
    from cowork.handlers.turn_errors import remote_turn_error, TOKEN_LIMIT_CODE
    code, msg = remote_turn_error("TokenLimitExceeded: Server returned 429 ...")
    assert code == TOKEN_LIMIT_CODE
    assert "credits" in msg


def test_remote_error_overloaded_passes_curated_copy():
    from cowork.handlers.turn_errors import remote_turn_error, PROVIDER_OVERLOADED_CODE
    code, msg = remote_turn_error(
        "ProviderOverloadedError: The model provider is experiencing an incident.")
    assert code == PROVIDER_OVERLOADED_CODE
    assert msg == "The model provider is experiencing an incident."


def test_remote_error_auth():
    from cowork.handlers.turn_errors import remote_turn_error, AUTH_ERROR_CODE
    code, _ = remote_turn_error("ProviderAuthError: provider rejected the credential")
    assert code == AUTH_ERROR_CODE


def test_remote_error_turn_interrupted_keeps_its_curated_copy():
    # The pod's own no-terminal-event fallback (a pod torn down mid-turn).
    # Same generic code as any unmapped failure — no dedicated card exists
    # for this — but the curated sentence must survive instead of being
    # discarded for the fully generic message.
    from cowork.handlers.turn_errors import remote_turn_error, GENERIC_TURN_ERROR_CODE
    code, msg = remote_turn_error(
        "TurnInterrupted: The turn ended unexpectedly. Please try again.")
    assert code == GENERIC_TURN_ERROR_CODE
    assert msg == "The turn ended unexpectedly. Please try again."


def test_remote_error_turn_worker_lost_keeps_its_curated_copy():
    # pel_reclaim.py's ORPHANED_ERROR — a worker died mid-turn and the entry
    # was reclaimed from Redis's PEL rather than retried (retrying would bill
    # the tenant's tokens twice). Already correctly shaped; just missing from
    # the allowlist.
    from cowork.handlers.turn_errors import remote_turn_error, GENERIC_TURN_ERROR_CODE
    code, msg = remote_turn_error(
        "TurnWorkerLost: the worker running this turn stopped before it "
        "finished; the turn was not retried")
    assert code == GENERIC_TURN_ERROR_CODE
    assert "the turn was not retried" in msg


def test_remote_error_pod_stream_ended_without_terminal_gets_curated_copy():
    # scratchpad-controller's own literal (main.py) — an OOM-killed pod or a
    # dropped exec channel. No "TypeName:" prefix at all, so the generic
    # type_name parse below would never match it; matched directly instead.
    # The optional stderr tail must never reach the user verbatim.
    from cowork.handlers.turn_errors import remote_turn_error, GENERIC_TURN_ERROR_CODE
    code, msg = remote_turn_error("pod stream ended without a terminal event")
    assert code == GENERIC_TURN_ERROR_CODE
    assert msg == "The turn ended unexpectedly. Please try again."

    code, msg = remote_turn_error(
        "pod stream ended without a terminal event; stderr tail: Traceback ...")
    assert code == GENERIC_TURN_ERROR_CODE
    assert msg == "The turn ended unexpectedly. Please try again."
    assert "Traceback" not in msg


def test_remote_error_turn_aborted_on_hard_timeout_gets_curated_copy():
    # scratchpad-controller's with_limits() hard wall-clock deadline.
    from cowork.handlers.turn_errors import remote_turn_error, GENERIC_TURN_ERROR_CODE
    code, msg = remote_turn_error("turn aborted: hard turn timeout")
    assert code == GENERIC_TURN_ERROR_CODE
    assert "too long" in msg


def test_remote_error_turn_aborted_on_stall_gets_curated_copy():
    # scratchpad-controller's with_limits() no-output stall detector.
    from cowork.handlers.turn_errors import remote_turn_error, GENERIC_TURN_ERROR_CODE
    code, msg = remote_turn_error(
        "turn aborted: no output within stall window; stderr tail: boom")
    assert code == GENERIC_TURN_ERROR_CODE
    assert "stopped producing output" in msg
    assert "boom" not in msg


def test_remote_error_missing_organization_gets_curated_copy():
    # scratchpad-controller's MissingOrganization (_scratchpad_id_for_job) —
    # a job dispatched with no organization_id, so it can't be routed to an
    # EFS access point. The message's own correlation_id prefix varies per
    # job; matched on the static suffix, which never changes.
    from cowork.handlers.turn_errors import remote_turn_error, GENERIC_TURN_ERROR_CODE
    code, msg = remote_turn_error(
        "job corr-123 has no organization_id; refusing to run a turn "
        "without an organization-scoped workspace")
    assert code == GENERIC_TURN_ERROR_CODE
    assert "workspace" in msg
    assert "support" in msg
    # The raw correlation_id in the source message must not leak — request_id
    # already carries it, separately and reliably, on the payload.
    assert "corr-123" not in msg


def test_remote_error_live_pod_terminal_before_running_gets_curated_copy():
    # scratchpad-controller's live_pod.py — a pod that reached Failed (or any
    # other terminal phase) before ever reaching Running, i.e. it never
    # scheduled/started. Raised as a bare RuntimeError with no type prefix,
    # caught by _handle_anton_turn_k8s's own generic `except Exception`. The
    # pod name and phase are both k8s-controlled (safe), but a fixed message
    # is still returned rather than echoing them to the user.
    from cowork.handlers.turn_errors import remote_turn_error, GENERIC_TURN_ERROR_CODE
    code, msg = remote_turn_error(
        "live pod sp-abc123 reached terminal phase 'Failed' before Running")
    assert code == GENERIC_TURN_ERROR_CODE
    assert "sandbox" in msg
    assert "sp-abc123" not in msg


def test_remote_legacy_connection_error_still_maps_to_provider_auth():
    """The remote worker pods still emit anton's pre-typed 401 copy.

    They run the `minds-anton-scratchpad` image, pinned in scratchpad-controller
    at anton `61ec5db6` (staging/dev) and `d4f1db2c` (prod). Neither carries
    `ProviderAuthError`, and no PR in this ENG-2116 set bumps that image, so
    keying only on the typed name would strip the Reconnect card from every
    hosted 401.
    """
    code, message = te.remote_turn_error(
        "ConnectionError: Invalid API key — check your OpenAI API key configuration."
    )
    assert code == te.AUTH_ERROR_CODE
    assert message == te.AUTH_ERROR_USER_MESSAGE


def test_remote_unanchored_invalid_key_text_stays_generic():
    """A tool's own 'invalid api key' mid-message must not select the auth card.

    The classifier anchors with ``startswith`` precisely so an arbitrary tool
    exception cannot borrow the Reconnect card. A substring check would map this
    to ``provider_auth`` and tell the user to reconnect MindsHub over a failure
    that has nothing to do with their session.
    """
    code, message = te.remote_turn_error(
        "ConnectionError: Stripe rejected the request: invalid api key for account acct_1"
    )
    assert code == te.GENERIC_TURN_ERROR_CODE
    assert message == te.GENERIC_TURN_ERROR_MESSAGE


def test_remote_untyped_auth_lookalikes_are_redacted():
    """A 401 that is not anton's anchored invalid-key copy stays generic."""
    code, message = te.remote_turn_error(
        "ConnectionError: Server returned 401 - Unauthorized"
    )
    assert code == te.GENERIC_TURN_ERROR_CODE
    assert message == te.GENERIC_TURN_ERROR_MESSAGE
    assert "api key" not in message.lower()


def test_remote_error_unknown_is_redacted():
    from cowork.handlers.turn_errors import remote_turn_error, GENERIC_TURN_ERROR_CODE
    code, msg = remote_turn_error("RuntimeError: secret internals")
    assert code == GENERIC_TURN_ERROR_CODE
    assert "secret" not in msg


def test_remote_error_none_is_redacted():
    from cowork.handlers.turn_errors import remote_turn_error, GENERIC_TURN_ERROR_CODE
    assert remote_turn_error(None)[0] == GENERIC_TURN_ERROR_CODE


# ── Wire-code inventory (ENG-1282) ────────────────────────────────

def test_wire_code_inventory_matches_the_renderer_contract():
    """Pin the set of turn-failure codes this module can emit.

    The cowork renderer (mindsdb/cowork ``ChatView.jsx``) draws a card for
    every code except the generic ``anton_error`` fallback, and its
    ``ChatView.turnFailureCards.test.jsx`` sweeps for a matching branch per
    code. The two lists must move together: adding a code here without a
    renderer branch would make that failure render with no next step, which
    is the gap ENG-1282 closed. If this test fails, add the branch (and the
    code to the renderer test's list) in the same change that adds the code.
    """
    codes = {
        value
        for name, value in vars(te).items()
        if name.endswith("_CODE") and isinstance(value, str)
    }
    assert codes == {
        "token_limit",
        "policy_unavailable",
        "model_not_found",
        "provider_auth",
        "model_access_denied",
        "model_disabled",
        "provider_overloaded",
        "image_format",
        # ENG-1537. This tripwire did its job: adding the constant failed this
        # test before the renderer branch existed, which is exactly the gap
        # ENG-1282 built it to catch. The matching branch lands in
        # mindsdb/cowork's ChatView.jsx + its turnFailureCards list.
        "rate_limited",
        # ENG-1537 — the spent free allowance, split off the credits card.
        "included_allowance_exhausted",
        # ENG-1992 — a content-shaped rejection the server already repaired;
        # distinct copy from image_format (no re-upload needed).
        "content_recovery",
        # ENG-2126 — the worker never answered, so the turn never ran. Split off
        # anton_error because the two need opposite next steps: this one is ours
        # to fix, and reads as an agent bug while it shares that code.
        "worker_unresponsive",
        "anton_error",
    }


def test_no_return_emits_a_literal_code():
    """Every ``(code, message)`` return must take its code from a constant.

    The inventory test above only sees ``*_CODE`` module constants — a code
    returned as a bare string literal (how ``image_format`` originally
    shipped, fixed in ENG-1282) would bypass it entirely. Parsing the module
    keeps that authoring path closed: together the two tests cover both ways
    a new code can reach the wire.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(te))
    offenders = [
        node.value.elts[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and node.value.elts
        and isinstance(node.value.elts[0], ast.Constant)
        and isinstance(node.value.elts[0].value, str)
    ]
    assert offenders == []


# ── Wiring coverage (ENG-1537 review finding 3) ────────────────────────────
# Four mutations survived the full suite: the reset_at and retry_after extras
# in responses.py, the never-throttle exemption, and PHASE_LABELS.

def test_retry_at_is_an_absolute_offset_bearing_instant():
    # The renderer gates its Retry on this. It cannot use the message's own
    # created_at — cowork-server serialises that naive and offset-less, so JS
    # parses it as LOCAL time: west of UTC the button gates for hours, east of
    # it the gate no-ops, and a TZ=UTC suite sees neither.
    from datetime import datetime

    from datetime import timedelta, timezone

    before = datetime.now(timezone.utc)
    got = te.retry_at_instant(30)
    after = datetime.now(timezone.utc)
    assert got is not None and got.endswith("Z"), got
    parsed = datetime.fromisoformat(got.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None  # the whole point

    # VALUE, not just shape. Format-only assertions let three arithmetic
    # mutations through the full suite: seconds=0 (the gate never fires at
    # all), a sign flip (instant in the past, same no-op), and /1000 (a 30s
    # wait becomes 30ms). Each silently disables the feature this exists for.
    assert before + timedelta(seconds=30) <= parsed <= after + timedelta(seconds=30)

    assert te.retry_at_instant(None) is None
    assert te.retry_at_instant(-5) is None
    # Bounded before the arithmetic: `timedelta` raises OverflowError past the
    # datetime range, and this runs inside the terminal error handler, where an
    # unhandled raise strands the SSE stream with no failure frame.
    assert te.retry_at_instant(999_999_999_999) is None
    assert te.retry_at_instant(86_401) is None
    assert te.retry_at_instant(86_400) is not None


def test_rate_limit_extras_carry_both_the_interval_and_the_instant():
    payload = te.response_failed_payload(
        "msg", te.RATE_LIMITED_CODE, retry_after=30.0, retry_at="2026-09-01T00:00:30Z",
    )
    assert payload["retry_after"] == 30.0
    assert payload["retry_at"] == "2026-09-01T00:00:30Z"
    # Additive: absent for every other failure, so the wire shape is unchanged.
    plain = te.response_failed_payload("msg", te.TOKEN_LIMIT_CODE)
    assert "retry_after" not in plain and "retry_at" not in plain


def test_the_rate_limit_notice_is_exempt_from_progress_throttling():
    # It fires once per wait. Throttled away, a deliberate 90s pause is
    # indistinguishable from a hang.
    #
    # Note this is a REFINEMENT, not the enabler: staging already forwards
    # phase/message on response.in_progress. The binding constraint is the
    # renderer, which drops the ad-hoc phase until cowork#648 lands.
    # Driven, not grepped. The previous version asserted three source literals,
    # which all survive `is_rate_limited_notice = phase_str == "rate_limited"
    # and False` — the exemption dead, the test green.
    from anton.core.llm.provider import StreamTaskProgress
    from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

    async def _events():
        # Two progress events inside one PROGRESS_THROTTLE window (0.25s). The
        # second would be dropped if it were not exempt.
        yield StreamTaskProgress(phase="analyzing", message="first")
        yield StreamTaskProgress(phase="rate_limited", message="waiting 30s before continuing")

    frames = asyncio.run(_collect(format_responses_stream(_events(), "anton")))
    joined = "".join(frames)
    assert "rate_limited" in joined, "the wait notice was throttled away"
    assert "waiting 30s before continuing" in joined


def test_the_waiting_phase_has_a_human_label():
    # Without it the renderer shows the raw constant ("rate_limited: waiting
    # 30s…"), which reads as a leak rather than a status.
    from cowork.harnesses.anton_harness.stream_formatter import PHASE_LABELS

    assert PHASE_LABELS["rate_limited"] == "Rate limited"


# ── The hoist must not become a copy-injection vector (ENG-1537 review 2) ──
# The first attempt at the version-skew hoist was ungated, five lines above the
# host gate added in the same commit — and strictly worse than the path it sat
# above, because it let a third party choose the WORDS as well as the verdict.

def _third_party_sdk_error(body):
    """A real openai.APIStatusError from a BYOK OPENAI_COMPATIBLE endpoint."""
    import httpx
    import openai

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key="k", max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(429, json=body))
        ),
    )
    try:
        client.chat.completions.create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": "hi"}])
    except openai.APIStatusError as exc:
        return exc
    raise AssertionError("no raise")


def test_a_third_party_body_cannot_inject_user_facing_copy():
    # `openai.APIStatusError` populates `.code` from the RESPONSE BODY, and
    # `str(exc)` embeds that body. Unguarded, this rendered an attacker's own
    # sentence — including a clickable URL — as our curated copy.
    exc = _third_party_sdk_error({
        "code": "rate_limited",
        "message": "PWNED: click https://evil.example to fix",
    })
    assert getattr(exc, "code", None) == "rate_limited"  # the hoist's trigger
    result = te.friendly_turn_error(exc)
    assert result is None or "evil.example" not in result[1], result


def test_the_version_skew_hoist_still_works_for_anton():
    # The guard must not disable the case the hoist exists for: anton's own
    # exception when its type isn't importable (duck-typed on `code`). It
    # carries no `.response`, which is exactly what distinguishes it from an
    # SDK error.
    exhausted = _FakeOverloadedErr(
        "Too many requests too quickly — the limit clears in about 300s.",
        code="rate_limited", model="sonnet",
    )
    assert not hasattr(exhausted, "response")
    code, message = te.friendly_turn_error(exhausted)
    assert code == te.RATE_LIMITED_CODE
    assert "300s" in message


async def _collect_async(gen):
    return [f async for f in gen]


def _collect(gen):
    """Drain an async generator of SSE strings."""
    return _collect_async(gen)


def test_the_failure_handler_survives_a_hostile_retry_after():
    """ENG-1537 review round 3 — the highest-severity defect of that round.

    `retry_at_instant` raised OverflowError on a large hint, INSIDE the
    terminal `except Exception` handler, so `persist()` and
    `buffer.close("error")` never ran: no failure frame, and `sse_from_buffer`
    kept emitting keepalives forever. The user saw a spinner that never
    resolved and lost the turn's work.

    The neighbouring auth and provider_overloaded branches already stated the
    rule ("Never break the handler"); this branch didn't follow it.
    """
    # The value that used to raise.
    assert te.retry_at_instant(999_999_999_999) is None
    # And the extras assembly must tolerate anything the helper does.
    payload = te.response_failed_payload(
        "msg", te.RATE_LIMITED_CODE,
        retry_after=999_999_999_999, retry_at=te.retry_at_instant(999_999_999_999),
    )
    assert payload["code"] == te.RATE_LIMITED_CODE
    assert "retry_at" not in payload      # dropped, not a crash
# ── The wire `model` for model_not_found (ENG-1358 re-review) ────────


def test_model_not_found_is_in_the_set_responses_uses_to_emit_model():
    """responses.py attaches `model` to the failure frame for exactly these
    codes. Naming the id IS the fix — if model_not_found drops out of this set
    the card silently falls back to its unnamed copy, which is the defect the
    ticket exists to close, and nothing else in the suite notices.

    Shared as a set rather than re-listed inline in responses.py so a merge
    conflict in that elif-chain has no tuple members to drop."""
    assert te.MODEL_NOT_FOUND_CODE in te.MODEL_UNAVAILABLE_CODES
    assert te.MODEL_ACCESS_DENIED_CODE in te.MODEL_UNAVAILABLE_CODES
    assert te.MODEL_DISABLED_CODE in te.MODEL_UNAVAILABLE_CODES


def test_responses_emits_model_for_every_model_unavailable_code():
    """Guards the branch itself: the handler must reach the model-emitting arm
    via the shared set, not a hand-maintained tuple."""
    import inspect

    from cowork.handlers import responses as rp

    src = inspect.getsource(rp)
    assert "elif code in MODEL_UNAVAILABLE_CODES:" in src, (
        "responses.py must branch on the shared set — an inline tuple here is "
        "what let a rebase silently drop model_not_found"
    )
    for code in te.MODEL_UNAVAILABLE_CODES:
        payload = te.response_failed_payload("msg", code, model="deepseek-v4-flash")
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["code"] == code


# ── The header carrier is origin-checked too (ENG-1686) ────────────────────
# ENG-1537 gated the body-`code` carrier and left its header twin unconditional,
# on the assumption that "only the gateway sets X-MindsHub-Reason". That is true
# of every honest provider and not enforceable against a hostile one: on a BYOK
# OPENAI_COMPATIBLE endpoint the whole response is third-party controlled.

@pytest.mark.parametrize("reason,forbidden_code", [
    ("wallet_empty", "token_limit"),
    ("included_allowance_exhausted", "included_allowance_exhausted"),
    ("rate_limited", "rate_limited"),
    ("policy_unavailable", "policy_unavailable"),
    # The fifth. Not a billing verdict, but a third party should not get to
    # pick our model card either, and the gate already refuses it — this pins
    # the behaviour so the "every reason" claim below is literally true
    # (review: pnewsam).
    ("unknown_model", "model_not_found"),
])
def test_a_third_party_header_cannot_select_a_billing_verdict(reason, forbidden_code):
    # Mirror of test_a_third_party_body_cannot_select_a_billing_verdict, over
    # the carrier that was left open. Parametrised across every reason the
    # gateway defines, so adding a sixth cannot quietly reopen one lane.
    exc = _failure(402, reason=reason, url="https://openrouter.ai/api/v1/chat/completions")
    result = te.friendly_turn_error(exc)
    assert result is None or result[0] != forbidden_code, (
        f"a third-party header selected {result!r}"
    )


def test_the_gateways_own_header_still_maps():
    # The gate must not break the carrier it exists to protect.
    code, _ = te.friendly_turn_error(_gateway_failure(402, reason="wallet_empty"))
    assert code == te.TOKEN_LIMIT_CODE


def test_the_unknown_origin_residual_is_not_remote_reachable():
    # The deliberate residual. `_origin_is_known_third_party` is three-valued so
    # an unknown origin stays trusted; that the unknown origin still MAPS is
    # asserted by test_reason_header_maps_even_without_request_url, not here —
    # this test only shows the residual cannot be reached by a remote (review:
    # pnewsam noted the old name claimed the mapping too).
    #
    # Safe because a remote server cannot produce it: a real SDK error always
    # carries its request, so `host` resolves for every genuine HTTP response.
    # The only route to host=None is a response with no request attached, whose
    # `.url` raises RuntimeError — our own plumbing, never the peer's choice.
    # Asserted here rather than left as prose, since it is the entire argument
    # for the narrow gate.
    import httpx
    import openai

    def _handler(request):
        return httpx.Response(402, json={}, headers={"X-MindsHub-Reason": "wallet_empty"})

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key="k", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    try:
        client.chat.completions.create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": "hi"}])
    except openai.APIStatusError as exc:
        # A REAL third-party error resolves its host, so it is gated — it can
        # never fall into the trusted unknown-origin residual.
        assert te._http_error_context(exc)[2] == "openrouter.ai"
        assert te._origin_is_known_third_party("openrouter.ai") is True
    else:  # pragma: no cover
        raise AssertionError("the SDK did not raise")

    # And the residual itself is only constructible locally.
    detached = httpx.Response(402, json={}, headers={"X-MindsHub-Reason": "wallet_empty"})
    with pytest.raises(RuntimeError):
        _ = detached.url
    assert te._origin_is_known_third_party(None) is False


# -- remote_turn_error: an unresponsive worker is not an agent error ----------
#
# These pin the distinction the 2026-08-31 outage needed and did not have. Every
# scratchpad pod failed to start, so no turn ever reached anton, and the failure
# still surfaced as the generic anton_error. Nothing in the user-facing string or
# the wire code said "infrastructure", which is why the cause took hours to find.

def test_remote_error_unresponsive_worker_is_not_the_generic_code():
    code, msg = te.remote_turn_error(
        "TurnWorkerUnresponsive: the turn worker stopped responding"
    )
    assert code == te.WORKER_UNRESPONSIVE_CODE
    assert code != te.GENERIC_TURN_ERROR_CODE
    assert msg == te.WORKER_UNRESPONSIVE_MESSAGE
    assert msg != te.GENERIC_TURN_ERROR_MESSAGE


def test_producer_error_string_is_built_from_the_type_name_we_branch_on():
    """The producer's literal and this module's branch must not drift.

    They lived in two files as two literals. A rename in either one would have
    silently returned the turn to the generic code.
    """
    from cowork.turnqueue.producer import UNRESPONSIVE_WORKER_ERROR

    assert UNRESPONSIVE_WORKER_ERROR.startswith(te.WORKER_UNRESPONSIVE_TYPE_NAME + ":")
    code, _ = te.remote_turn_error(UNRESPONSIVE_WORKER_ERROR)
    assert code == te.WORKER_UNRESPONSIVE_CODE


def test_remote_error_unmapped_type_still_falls_through_to_generic():
    """The negative case: anton raising something we don't know stays generic.

    Redacting an unrecognised provider error is the point of the fallback, so
    the new branch must not widen it.
    """
    code, msg = te.remote_turn_error("ModuleNotFoundError: No module named 'httpx'")
    assert code == te.GENERIC_TURN_ERROR_CODE
    assert msg == te.GENERIC_TURN_ERROR_MESSAGE
    assert "httpx" not in msg
