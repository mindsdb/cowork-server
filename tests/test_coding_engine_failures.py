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
