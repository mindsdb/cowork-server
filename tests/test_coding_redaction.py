from __future__ import annotations

from cowork.coding.redaction import redact_text


def test_redact_text_covers_json_quoted_keys_in_flat_strings() -> None:
    payload = '{"model": "fable", "api_key": "sk-live-123", "Client_Secret": "s3cret", "count": 2}'

    assert redact_text(payload) == (
        '{"model": "fable", "api_key": "[redacted]", "Client_Secret": "[redacted]", "count": 2}'
    )
    assert redact_text('token=abc api_key: "sk-1"') == "token=[redacted] api_key: [redacted]"


def test_usage_token_counts_survive_sanitizing() -> None:
    from cowork.coding.redaction import sanitize

    usage = {"total": {"inputTokens": 1200, "cachedInputTokens": 800, "outputTokens": 90,
                       "reasoningOutputTokens": 40, "totalTokens": 1290},
             "max_tokens": 4096, "token_count": 7, "modelContextWindow": 258400}

    assert sanitize(usage) == usage


def test_credentials_named_like_token_counts_are_still_redacted() -> None:
    from cowork.coding.redaction import sanitize

    payload = {"accessTokens": "ghp_live_secret", "refresh_tokens": ["r1"], "token": 12345,
               "apiKeyTokens": True}

    assert sanitize(payload) == {"accessTokens": "[redacted]", "refresh_tokens": "[redacted]",
                                 "token": "[redacted]", "apiKeyTokens": "[redacted]"}
