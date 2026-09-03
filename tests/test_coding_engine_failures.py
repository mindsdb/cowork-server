from __future__ import annotations

import pytest

from cowork.coding.context import EngineFailure, classify_engine_failure
from cowork.coding.engines.base import EngineCredentials

CREDS = EngineCredentials(minds_url="https://api.mindshub.ai", minds_api_key="mdb_secret_key_value")

# The exact text Codex surfaces for a terminal 400 from the loopback proxy: the body, verbatim.
REWRITTEN_402 = (
    '{"error": {"message": "Your wallet has no balance to cover the model \'gpt\'.", '
    '"type": "invalid_request_error", "code": "insufficient_credits", "upstream_status": 402}}'
)


def test_a_rewritten_credit_rejection_is_classified_from_its_error_code() -> None:
    failure = classify_engine_failure(REWRITTEN_402, CREDS, model="gpt")

    assert failure == EngineFailure(
        message="This model needs credits. Add credits or choose another model.",
        code="insufficient_credits",
        detail=REWRITTEN_402,
        model="gpt",
    )
    assert failure.event_data() == {"code": "insufficient_credits", "detail": REWRITTEN_402, "model": "gpt"}


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("unexpected status 402 Payment Required: Your wallet has no balance to cover the model 'gpt'., url: http://x", "insufficient_credits"),
        ("unexpected status 401 Unauthorized: Invalid API key, url: http://x", "model_authentication_failed"),
        ("unexpected status 403 Forbidden: denied, url: http://x", "model_authentication_failed"),
        ("unexpected status 404 Not Found: The model 'gpt-5.6-sol' does not exist or you do not have access to it., url: http://x", "model_unavailable"),
        ('unexpected status 400 Bad Request: {"error": {"code": "model_authentication_failed", "message": "Invalid API key"}}', "model_authentication_failed"),
    ],
)
def test_status_line_and_embedded_json_shapes_are_classified(message: str, code: str) -> None:
    failure = classify_engine_failure(message, CREDS)

    assert failure.code == code
    assert failure.detail == message
    assert failure.message != message


def test_unrelated_failures_keep_their_redacted_text_and_no_code() -> None:
    failure = classify_engine_failure("adapter stream disconnected mdb_secret_key_value", CREDS, model="gpt")

    assert failure == EngineFailure(message="adapter stream disconnected [redacted]")
    assert failure.event_data() == {}


def test_unknown_embedded_codes_are_not_promoted_to_a_contract() -> None:
    failure = classify_engine_failure('{"error": {"code": "something_new", "message": "x"}}', None)

    assert failure.code == ""
    assert failure.message == '{"error": {"code": "something_new", "message": "x"}}'


@pytest.mark.parametrize(
    "message",
    [
        "unexpected status 503 Service Unavailable: fake, url: http://127.0.0.1:27968/api/v1/coding/inference/responses",
        "unexpected status 502 Bad Gateway: <html>upstream error</html>, url: http://127.0.0.1:27966/api/v1/coding/inference/responses",
        "unexpected status 504 Gateway Timeout, url: http://x",
    ],
)
def test_transient_upstream_failures_get_plain_copy_without_the_proxy_url(message: str) -> None:
    failure = classify_engine_failure(message, CREDS, model="gpt")

    assert failure.code == "model_upstream_unavailable"
    assert failure.message == "The model service is temporarily unavailable. Try again in a moment."
    assert "url:" not in failure.detail
    assert "127.0.0.1" not in failure.detail
    assert failure.detail.startswith("unexpected status 50")
    assert failure.model == "gpt"
