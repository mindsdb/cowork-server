"""User-facing turn-failure mapping.

Ported from the bundled server (mindsdb/cowork PR #156), which is being
retired in favour of this package.

A turn can die on a cryptic provider 400 — most notably an image that
reaches an Anthropic-backed model as an OpenAI-style ``image_url``
content block instead of Anthropic's ``image`` block. The raw provider
JSON is useless (and unsafe) to show a user, so we recognise the failure
and trade it for a clean, actionable message plus a stable ``code``.

Everything we haven't explicitly mapped stays generic — provider
internals must never leak into the chat, so unmapped failures surface as
``GENERIC_TURN_ERROR_MESSAGE`` under the ``anton_error`` code.
"""

from __future__ import annotations

import json
import re

# Curated copy for the unsupported-image case. Surfaced verbatim.
IMAGE_FORMAT_USER_MESSAGE = (
    "Sorry, I couldn't process that image. Try uploading it as a PNG or JPEG."
)

# Curated copy for a spent token/credit allowance. Without this the turn
# would die on a 429/402 mid-stream with no completion event and no error
# frame — the SSE connection just closes and the renderer's spinner stops,
# which reads as "Anton is dead" rather than a quota message. The desktop
# renders a richer card for the `token_limit` code (Add credits / Bring
# your own keys); this text is the fallback copy.
TOKEN_LIMIT_USER_MESSAGE = (
    "You've reached your monthly token limit. To keep going, upgrade your plan "
    "or add your own LLM provider key in Settings — or wait until your allowance "
    "resets."
)

# Wire-level code for the quota case. Distinct from the image/generic
# codes so a client can branch on it if it ever wants a richer affordance.
TOKEN_LIMIT_CODE = "token_limit"

# Redacted stand-in for any failure we haven't mapped — never the raw
# provider text.
GENERIC_TURN_ERROR_MESSAGE = "An unexpected error occurred."

# Wire-level code for an unmapped failure. Kept stable so existing
# clients (which may branch on it) keep working after the migration.
GENERIC_TURN_ERROR_CODE = "anton_error"


def is_image_format_error(exc: Exception) -> bool:
    """Detect the Anthropic 400 raised when an image reaches the model as
    the OpenAI-style ``image_url`` content block instead of Anthropic's
    ``image`` block. Surfaces as e.g.::

        Input tag 'image_url' found using 'type' does not match any of
        the expected tags: 'image'

    The block format is built upstream (anton-core / the provider
    adapter), so we can't repair it here — but we can recognise the
    failure and trade the raw provider JSON for a clean message.
    """
    s = str(exc).lower()
    if "image_url" in s and ("expected tag" in s or "does not match" in s):
        return True
    # Other phrasings of "this image content block was rejected".
    return "image" in s and ("unsupported image" in s or "could not process image" in s)


def is_token_limit_error(exc: Exception) -> bool:
    """Detect a spent allowance — anton's ``TokenLimitExceeded`` (429 token
    limit) OR an exhausted credit balance (the gateway may instead report a
    402 / "insufficient credits"). Both mean "out of credits", so we map them
    to the same ``token_limit`` code and let the client show the curated
    out-of-credits card instead of a generic crash.
    """
    try:
        from anton.core.llm.provider import TokenLimitExceeded

        if isinstance(exc, TokenLimitExceeded):
            return True
    except Exception:
        # anton not importable / the type moved — fall back to matching the
        # stable messages the upstream constructs for these cases.
        pass
    s = str(exc).lower()
    # 429 token-allowance exhausted (the original case).
    if "429" in s and "limit exceeded for tokens" in s:
        return True
    # Spent credit balance — a 402 paired with credit/quota keywords.
    # "402 + token" alone is intentionally excluded: a JWT-expiry or session
    # error may produce "402 … token … expired" which is not a quota failure.
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    is_402 = status == 402 or re.search(r"\b402\b", s) is not None
    if is_402 and (
        "credit" in s or "quota" in s or re.search(r"token\s+(limit|quota|allowance)", s)
    ):
        return True
    if "insufficient" in s and ("credit" in s or "quota" in s):
        return True
    # "out of credit/credits" — fine as-is.
    if "out of credit" in s or "out of quota" in s:
        return True
    # "no credits" (plural) is unambiguous quota exhaustion.
    # "no credit" (singular) is excluded to avoid matching "no credit card on
    # file" — a payment-method setup message that must not show the quota card.
    if "no credits" in s:
        return True
    return False


def friendly_turn_error(exc: Exception) -> tuple[str, str] | None:
    """Map a known, cryptic turn failure to ``(code, user_message)``.

    Returns ``None`` when the exception isn't one we have curated copy
    for — the caller then falls back to the generic redacted message.
    """
    if is_token_limit_error(exc):
        return TOKEN_LIMIT_CODE, TOKEN_LIMIT_USER_MESSAGE
    if is_image_format_error(exc):
        return "image_format", IMAGE_FORMAT_USER_MESSAGE
    return None


def response_failed_sse(error: str, code: str) -> str:
    """Build a ``response.failed`` SSE frame.

    Same wire shape the legacy server emitted, so the renderer's existing
    parser handles it unchanged: ``{type, code, error}``.
    """
    payload = {"type": "response.failed", "code": code, "error": error}
    return f"event: response.failed\ndata: {json.dumps(payload)}\n\n"
