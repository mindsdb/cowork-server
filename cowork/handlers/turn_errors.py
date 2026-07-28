"""User-facing turn-failure mapping.

Ported from the bundled server (mindsdb/cowork PR #156), which is being
retired in favour of this package.

A turn can die on a cryptic provider 400 — most notably an image that
reaches an Anthropic-backed model as an OpenAI-style ``image_url``
content block instead of Anthropic's ``image`` block. The raw provider
JSON is useless (and unsafe) to show a user, so we recognise the failure
and trade it for a clean, actionable message plus a stable ``code``.

A turn can also die on a billing decision from the wallet-model inference
gateway: 402 (wallet empty), 429 (free monthly allowance spent), 404
(unknown model), or 503 (billing/auth policy service down). The gateway
names each with an ``X-MindsHub-Reason`` header, which we prefer over
status/message heuristics to route to the right, actionable copy.

Everything we haven't explicitly mapped stays generic — provider
internals must never leak into the chat, so unmapped failures surface as
``GENERIC_TURN_ERROR_MESSAGE`` under the ``anton_error`` code.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

from cowork.common.settings.app_settings import default_minds_url

# Curated copy for the unsupported-image case. Surfaced verbatim.
IMAGE_FORMAT_USER_MESSAGE = (
    "Sorry, I couldn't process that image. Try uploading it as a PNG or JPEG."
)

# Curated copy for the out-of-credits case. In the wallet billing model this
# fires when either the org's wallet is empty (gateway 402 `wallet_empty`) or
# the free monthly included-token allowance is spent (gateway 429
# `included_allowance_exhausted`). Without this the turn would die mid-stream
# with no completion event and no error frame — the SSE connection just closes
# and the renderer's spinner stops, which reads as "Anton is dead" rather than
# an out-of-credits message. The desktop renders a richer card for the
# `token_limit` code (Add credits / Bring your own keys); this text is the
# fallback copy.
TOKEN_LIMIT_USER_MESSAGE = (
    "You're out of credits. Add credits to keep going, or bring your own LLM "
    "provider key in Settings."
)

# Wire-level code for the out-of-credits case. Named `token_limit` for wire
# back-compat with clients already branching on it; it now covers both the
# empty-wallet and spent-allowance reasons above.
TOKEN_LIMIT_CODE = "token_limit"

# Curated copy + wire code for a transient billing/auth policy outage — the
# gateway couldn't reach the service that decides whether a call is paid for
# (gateway 503 `policy_unavailable`). This is retryable and must NOT be shown
# as out-of-credits: the user has done nothing wrong and just needs to retry.
POLICY_UNAVAILABLE_CODE = "policy_unavailable"
POLICY_UNAVAILABLE_USER_MESSAGE = (
    "Billing is temporarily unavailable. Please retry in a moment."
)

# Curated copy + wire code for an unknown/removed model (gateway 404
# `unknown_model`). Adding credits can't fix it, so the copy steers to Settings
# rather than to the out-of-credits card.
UNKNOWN_MODEL_CODE = "unknown_model"
UNKNOWN_MODEL_USER_MESSAGE = (
    "That model isn't available. Switch to another model in Settings."
)

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
# MODEL (the credential itself is fine). Only older pre-wallet gateway/anton
# versions emit these: access_denied meant a plan/tier exclusion and disabled an
# admin kill switch. The current gateway never sends them — it denies a model
# the wallet can't pay for as a 402 ``wallet_empty`` (mapped to ``token_limit``
# above) — so this branch exists purely as back-compat for version-skewed
# deployments. The codes mirror the gateway's own ``error.code`` values so
# nothing is lost in translation, and the renderer keys its card on them.
MODEL_ACCESS_DENIED_CODE = "model_access_denied"
MODEL_DISABLED_CODE = "model_disabled"
_MODEL_UNAVAILABLE_CODES = frozenset({MODEL_ACCESS_DENIED_CODE, MODEL_DISABLED_CODE})

# Fallback copy if the exception somehow carries no usable message — anton
# normally supplies curated, user-facing copy which we pass through verbatim.
# Deliberately neutral: this legacy denial isn't necessarily fixable with
# credits, so the copy steers to picking another model rather than to billing.
MODEL_UNAVAILABLE_FALLBACK_MESSAGE = (
    "That model isn't available right now. Switch to another model in Settings."
)

# The X-MindsHub-Reason header values the inference gateway sets to name the
# billing decision precisely. Preferred over status/message heuristics.
_REASON_WALLET_EMPTY = "wallet_empty"
_REASON_ALLOWANCE_EXHAUSTED = "included_allowance_exhausted"
_REASON_POLICY_UNAVAILABLE = "policy_unavailable"
_REASON_UNKNOWN_MODEL = "unknown_model"

# Wire-level code for a transient provider incident that didn't clear within
# anton's retry budget (ENG-673) — the model provider (or an upstream it depends
# on) was overloaded/erroring mid-stream and backoff-retry ran out of time. The
# renderer keys a card on it (retry, and — for BYOK/direct users — a MindsHub
# cross-provider-failover nudge), so it's distinct from the model-gate codes.
PROVIDER_OVERLOADED_CODE = "provider_overloaded"

# Fallback copy if the exception carries no usable message — anton normally
# supplies curated, user-facing copy which we pass through verbatim.
PROVIDER_OVERLOADED_FALLBACK_MESSAGE = (
    "The model provider is having a temporary incident and didn't recover in "
    "time. Try again in a moment."
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
    # Spent credit balance — a 402, or any "insufficient/no credits|quota"
    # phrasing. Scoped to credit/quota/token context so unrelated 402s or
    # "insufficient permissions" don't get mislabelled.
    if "402" in s and ("credit" in s or "quota" in s or "token" in s):
        return True
    if "insufficient" in s and ("credit" in s or "quota" in s):
        return True
    if "out of credit" in s or "no credit" in s or "out of quota" in s:
        return True
    return False


def model_unavailable_info(exc: Exception) -> tuple[str, str] | None:
    """``(code, model)`` when the turn died on the legacy gateway's structured
    model-403 — anton's ``ModelUnavailableError`` carrying
    ``code ∈ {model_access_denied, model_disabled}`` and the model alias.
    Only pre-wallet gateway/anton versions raise it; kept as back-compat.

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


def provider_overloaded_info(exc: Exception) -> tuple[str, str] | None:
    """``(code, model)`` when the turn died on a transient provider incident
    that outlasted anton's retry budget — anton's ``ProviderOverloadedError``
    carrying ``code == provider_overloaded`` and the model alias (ENG-673).

    Prefers the typed check; falls back to duck-typing on ``code``/``model`` so a
    version-skewed anton (type not importable / moved) still maps. Deliberately
    NO string matching — only the structured code triggers the card.
    """
    try:
        from anton.core.llm.provider import ProviderOverloadedError

        if isinstance(exc, ProviderOverloadedError):
            return exc.code, str(getattr(exc, "model", "") or "")
    except Exception:
        pass
    code = getattr(exc, "code", None)
    if code == PROVIDER_OVERLOADED_CODE:
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


def _response_url_host(resp: object) -> str | None:
    """Hostname the request that produced ``resp`` was sent to, lowercased.

    httpx carries the URL on the response itself (``resp.url``) and on its
    ``request``; the SDKs' status errors expose that response. Returns ``None``
    when no URL is available (a synthetic error, or headers attached directly
    to the exception with no response object).
    """
    # Everything is guarded: this runs inside turn-failure handling, which
    # must never raise. Notably httpx.Response.url is a property that RAISES
    # (RuntimeError) when the response has no request attached — getattr does
    # not swallow that.
    try:
        url = getattr(resp, "url", None)
        if url is None:
            url = getattr(getattr(resp, "request", None), "url", None)
        if url is None:
            return None
        # httpx.URL exposes .host directly; anything else is parsed as a string.
        host = getattr(url, "host", None)
        if not host:
            host = urlparse(str(url)).hostname
    except Exception:
        return None
    return str(host).lower() if host else None


def _http_error_context(
    exc: BaseException,
) -> tuple[int | None, str | None, str | None]:
    """Extract ``(status, reason, host)`` from a turn failure — the upstream
    HTTP status, the gateway's ``X-MindsHub-Reason`` header, and the hostname
    the failing request was sent to.

    anton wraps the provider SDK's ``APIStatusError`` in a ``ConnectionError`` /
    ``TokenLimitExceeded`` (``raise ... from exc``), so the structured status and
    the response headers live on the chained cause, not the exception we're
    handed. We walk the ``__cause__`` / ``__context__`` chain looking for a
    response carrying ``X-MindsHub-Reason`` (wallet_empty /
    included_allowance_exhausted / policy_unavailable / unknown_model), which
    names the billing decision exactly and lets us skip brittle status/message
    matching. When the header is found, the status and host are taken from that
    SAME exception so the trio always describes one response; otherwise they
    come from the first exception in the chain with a ``status_code``. The host
    lets callers tell a gateway billing status from a BYOK provider's own
    402/429/503. Returns ``(None, None, None)`` for a plain exception with no
    HTTP context.
    """
    status: int | None = None
    host: str | None = None
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        code = getattr(cur, "status_code", None)
        # httpx.Headers (case-insensitive) on the SDK error's `.response`,
        # or a headers mapping some clients attach directly to the error.
        resp = getattr(cur, "response", None)
        headers = getattr(resp, "headers", None)
        if headers is None:
            headers = getattr(cur, "headers", None)
        reason = None
        if headers is not None:
            try:
                reason = headers.get("x-mindshub-reason") or headers.get(
                    "X-MindsHub-Reason"
                )
            except Exception:
                reason = None
        if reason:
            # The header names the billing decision; report the status and
            # origin of the exception that carries it, never a mix of chain
            # entries.
            if not isinstance(code, int):
                code = getattr(resp, "status_code", None)
            return (
                code if isinstance(code, int) else None,
                str(reason).strip().lower(),
                _response_url_host(resp),
            )
        if status is None and isinstance(code, int):
            status = code
            host = _response_url_host(resp)
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return status, None, host


def _configured_minds_host() -> str | None:
    """Hostname of the MindsHub API URL this install is configured to call.

    Read from user settings (``minds_url``); when settings can't be loaded
    (e.g. no DB in a bare context) falls back to the environment-aware default
    URL, which is what the settings field itself defaults to.
    """
    url: str | None
    try:
        from cowork.common.settings.user_settings import get_user_settings

        url = get_user_settings().minds_url
    except Exception:
        url = None
    if not url:
        url = default_minds_url()
    if "://" not in url:
        url = f"https://{url}"
    try:
        host = urlparse(url).hostname
    except Exception:
        return None
    return host.lower() if host else None


def _from_minds_gateway(host: str | None) -> bool:
    """Whether the failing request went to the configured MindsHub gateway.

    Gates the bare-status billing fallbacks: only the gateway's 402/429/503
    are billing decisions. The same statuses from a BYOK provider mean
    something else entirely (an OpenAI rate limit, an Anthropic overload) and
    must not surface billing copy. An unknown origin (no URL on the failure)
    is treated as not-the-gateway, so ambiguous failures stay generic.
    """
    if not host:
        return False
    expected = _configured_minds_host()
    return expected is not None and host == expected


def _map_gateway_reason(reason: str) -> tuple[str, str] | None:
    """Map an ``X-MindsHub-Reason`` header value to ``(code, user_message)``.

    Empty wallet and spent free-allowance both mean "out of credits" (the fix is
    to add credits), so they share the ``token_limit`` out-of-credits card. A
    policy outage is transient. An unknown model can't be fixed with credits.
    """
    if reason in (_REASON_WALLET_EMPTY, _REASON_ALLOWANCE_EXHAUSTED):
        return TOKEN_LIMIT_CODE, TOKEN_LIMIT_USER_MESSAGE
    if reason == _REASON_POLICY_UNAVAILABLE:
        return POLICY_UNAVAILABLE_CODE, POLICY_UNAVAILABLE_USER_MESSAGE
    if reason == _REASON_UNKNOWN_MODEL:
        return UNKNOWN_MODEL_CODE, UNKNOWN_MODEL_USER_MESSAGE
    return None


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
    status, reason, host = _http_error_context(exc)

    # The gateway's explicit reason header wins — it names the billing decision
    # exactly, so we never have to guess from a status code or message text.
    # Unconditional on origin: only the gateway sets X-MindsHub-Reason.
    if reason is not None:
        mapped = _map_gateway_reason(reason)
        if mapped is not None:
            return mapped

    # Precedence per ENG-673: token_limit / the billing-status fallback /
    # provider_auth / the ENG-598 model gate all WIN over provider_overloaded
    # (ranked below, after auth). These exception types are disjoint in practice
    # (a ProviderOverloadedError is never a 401 / quota / model-403), so the
    # order is behavior-preserving — made explicit so the stated contract and the
    # code can't silently drift apart (Sam's review).
    #
    # Out-of-credits first: a credit/quota failure must not be misread as auth
    # or a model gate. Covers anton's typed TokenLimitExceeded and the stable
    # message heuristics.
    if is_token_limit_error(exc):
        return TOKEN_LIMIT_CODE, TOKEN_LIMIT_USER_MESSAGE

    # Bare-status fallback for a gateway that omits the reason header (older
    # versions). Gated on the failing request having gone to the configured
    # MindsHub gateway: a BYOK provider's own 402/429/503 (an OpenAI rate
    # limit, an Anthropic overload) is not a billing decision, so it falls
    # through — usually to the generic redacted message. 402/429 both mean
    # out-of-credits; 503 is a transient policy outage, retryable and never
    # the out-of-credits card.
    if status in (402, 429, 503) and _from_minds_gateway(host):
        if status == 503:
            return POLICY_UNAVAILABLE_CODE, POLICY_UNAVAILABLE_USER_MESSAGE
        return TOKEN_LIMIT_CODE, TOKEN_LIMIT_USER_MESSAGE
    if model_info is _UNSET:
        model_info = model_unavailable_info(exc)
    if model_info is not None:
        # Legacy pre-wallet gateway/anton denial — its ModelUnavailableError
        # message is already curated user copy, so pass it through verbatim.
        return model_info[0], str(exc) or MODEL_UNAVAILABLE_FALLBACK_MESSAGE
    if is_auth_error(exc):
        return AUTH_ERROR_CODE, AUTH_ERROR_USER_MESSAGE
    # A transient-incident timeout (ENG-673) — anton's message is already curated
    # ("<provider> is experiencing an incident…"); pass it through. Ranked after
    # auth/quota/model-gate per the precedence note above.
    overloaded = provider_overloaded_info(exc)
    if overloaded is not None:
        return overloaded[0], str(exc) or PROVIDER_OVERLOADED_FALLBACK_MESSAGE
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
    can name the locked model ("Sonnet needs credits") — omitted otherwise to
    keep the shape unchanged for every other failure.
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
