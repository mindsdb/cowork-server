from __future__ import annotations

from cowork.coding.redaction import redact_text


def test_redact_text_covers_json_quoted_keys_in_flat_strings() -> None:
    payload = '{"model": "fable", "api_key": "sk-live-123", "Client_Secret": "s3cret", "count": 2}'

    assert redact_text(payload) == (
        '{"model": "fable", "api_key": "[redacted]", "Client_Secret": "[redacted]", "count": 2}'
    )
    assert redact_text('token=abc api_key: "sk-1"') == "token=[redacted] api_key: [redacted]"
