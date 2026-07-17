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

# Curated copy for a provider auth failure — the credential the model gateway
# sees is invalid (revoked / rotated / never provisioned / org drift), so calls
# come back 401 mid-conversation. The desktop renders a richer card for the
# `provider_auth` code (Reconnect MindsHub / Open Settings); this is the fallback
# text. Distinct from token_limit (out of credits) and from the config-absence
# case (no provider configured at all).
AUTH_ERROR_USER_MESSAGE = (
    "Your MindsHub session is no longer valid — reconnect to keep going, or "
    "update your provider key in Settings."
)

# Wire-level code for the auth case. The renderer branches on it to offer a
# "Reconnect" action (re-provision the key in place) instead of "Subscribe".
AUTH_ERROR_CODE = "provider_auth"

# Wire-level codes for the model-403 case — the gateway rejected the requested
# MODEL (the credential itself is fine). Two distinct codes because the
# remedies differ: access_denied is a plan gate (an upgrade fixes it),
# disabled is an admin kill switch (an upgrade does not). They mirror the
# gateway's own ``error.code`` values so nothing is lost in translation, and
# the renderer keys its card (Upgrade vs Switch-model affordances) on them.
MODEL_ACCESS_DENIED_CODE = "model_access_denied"
MODEL_DISABLED_CODE = "model_disabled"
_MODEL_UNAVAILABLE_CODES = frozenset({MODEL_ACCESS_DENIED_CODE, MODEL_DISABLED_CODE})

# Fallback copy if the exception somehow carries no usable message — anton
# normally supplies curated, user-facing copy which we pass through verbatim.
MODEL_UNAVAILABLE_FALLBACK_MESSAGE = (
    "The selected model isn't available on your MindsHub plan. Switch models "
    "in Settings, or upgrade your plan."
)

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


def model_unavailable_info(exc: Exception) -> tuple[str, str] | None:
    """``(code, model)`` when the turn died on the gateway's structured
    model-403 — anton's ``ModelUnavailableError`` carrying
    ``code ∈ {model_access_denied, model_disabled}`` and the model alias.

    Prefers the typed check; falls back to duck-typing on the ``code``/
    ``model`` attributes so a version-skewed anton (type not importable /
    moved) still maps correctly. Deliberately NO string matching: a message
    mentioning "model" or "403" must never trigger the plan card — only the
    structured code the gateway emitted can.
    """
    try:
        from anton.core.llm.provider import ModelUnavailableError

        if isinstance(exc, ModelUnavailableError):
            return exc.code, exc.model
    except Exception:
        pass
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code in _MODEL_UNAVAILABLE_CODES:
        return code, str(getattr(exc, "model", "") or "")
    return None


def is_auth_error(exc: Exception) -> bool:
    """Detect an **LLM-provider** auth failure — a 401 from the model gateway
    because the credential it sees is invalid (revoked / rotated / never
    provisioned / wrong org).

    Matched narrowly on anton's specific 401 copy — both providers raise a
    ``ConnectionError`` whose message starts ``Invalid API key — …``
    (``openai.py`` / ``anthropic.py``). Deliberately does NOT match a bare
    "401"/"unauthorized" anywhere in the text: that would mislabel an unrelated
    failure (e.g. a connector/tool API 401 that bubbles up) as a provider-auth
    error and pop the wrong "Reconnect" card. Credit/quota exhaustion (402/429)
    is handled by ``is_token_limit_error`` (checked first).
    """
    return "invalid api key" in str(exc).lower()


def auth_error_detail(provider_label: str, reconnectable: bool) -> str:
    """Provider-aware copy for an auth failure.

    MindsHub (managed) → the fix is to re-provision the key in place
    ("reconnect"); a BYOK provider → the user must fix their own key in Settings,
    so do NOT tell them to reconnect MindsHub.
    """
    if reconnectable:
        return "Your MindsHub session is no longer valid — reconnect to keep going."
    return f"Your {provider_label} API key is no longer valid — update it in Settings."


_UNSET: object = object()


def friendly_turn_error(
    exc: Exception, model_info: tuple[str, str] | None | object = _UNSET
) -> tuple[str, str] | None:
    """Map a known, cryptic turn failure to ``(code, user_message)``.

    Returns ``None`` when the exception isn't one we have curated copy
    for — the caller then falls back to the generic redacted message.

    ``model_info`` lets a caller that already resolved ``model_unavailable_info``
    (the streaming handler needs the rejected model for the card) pass it in so
    it isn't computed twice; omit it and it's resolved on demand.
    """
    # token_limit first: a 402/429 credit/quota case must not be misread as
    # auth or as a model gate.
    if is_token_limit_error(exc):
        return TOKEN_LIMIT_CODE, TOKEN_LIMIT_USER_MESSAGE
    if model_info is _UNSET:
        model_info = model_unavailable_info(exc)
    if model_info is not None:
        # anton's ModelUnavailableError message is already curated user copy
        # (plan guidance / hedged kill-switch wording) — pass it through.
        return model_info[0], str(exc) or MODEL_UNAVAILABLE_FALLBACK_MESSAGE
    if is_auth_error(exc):
        return AUTH_ERROR_CODE, AUTH_ERROR_USER_MESSAGE
    if is_image_format_error(exc):
        return "image_format", IMAGE_FORMAT_USER_MESSAGE
    return None


def response_failed_payload(
    error: str,
    code: str,
    *,
    reconnectable: bool | None = None,
    provider_label: str | None = None,
    model: str | None = None,
) -> dict:
    """Wire payload for a ``response.failed`` event (SSE + DB sidecar).

    ``reconnectable`` / ``provider_label`` are included only for the
    ``provider_auth`` case so the renderer can offer "Reconnect" (MindsHub) vs
    "Open Settings" (BYOK); ``model`` only for the model-403 case so the card
    can name the locked model ("Sonnet isn't included in your plan") — omitted
    otherwise to keep the shape unchanged for every other failure.
    """
    payload = {"type": "response.failed", "code": code, "error": error}
    if reconnectable is not None:
        payload["reconnectable"] = reconnectable
    if provider_label is not None:
        payload["provider_label"] = provider_label
    if model is not None:
        payload["model"] = model
    return payload


def response_failed_sse(
    error: str,
    code: str,
    *,
    reconnectable: bool | None = None,
    provider_label: str | None = None,
    model: str | None = None,
) -> str:
    """Build a ``response.failed`` SSE frame (same wire shape the renderer's
    parser already handles, plus the optional auth/model fields)."""
    payload = response_failed_payload(
        error,
        code,
        reconnectable=reconnectable,
        provider_label=provider_label,
        model=model,
    )
    return f"event: response.failed\ndata: {json.dumps(payload)}\n\n"
