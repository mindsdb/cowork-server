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
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from cowork.common.settings.app_settings import default_minds_url

# Curated copy for the unsupported-image case. Surfaced verbatim.
IMAGE_FORMAT_USER_MESSAGE = (
    "Sorry, I couldn't process that image. Try uploading it as a PNG or JPEG."
)

# Wire-level code for the unsupported-image case.
IMAGE_FORMAT_CODE = "image_format"

# ENG-1992: distinct from IMAGE_FORMAT_CODE on purpose. That copy tells the
# user to re-upload as PNG/JPEG — correct for an actually-malformed image
# file, wrong here: the failure is an internal serialization mismatch, not
# anything wrong with the image itself, and by the time this code reaches
# the client the conversation has ALREADY been repaired server-side. Telling
# the user to re-upload would be both inaccurate and unnecessary.
CONTENT_RECOVERY_CODE = "content_recovery"
CONTENT_RECOVERY_USER_MESSAGE = (
    "An image earlier in this conversation couldn't be sent to the model "
    "due to an internal formatting issue. I've fixed it automatically — "
    "you can keep going."
)

# Curated copy for the out-of-credits case. In the wallet billing model this
# fires when either the org's wallet is empty (gateway 402 `wallet_empty`) or
# the free monthly included-token allowance is spent (gateway 429
# `included_allowance_exhausted`). Without this the turn would die mid-stream
# with no completion event and no error frame — the SSE connection just closes
# and the renderer's spinner stops, which reads as "Anton is dead" rather than
# an out-of-credits message. The desktop renders a card for the `token_limit`
# code with a single Add-credits button (ENG-1169) — so this copy must not
# instruct actions the card doesn't offer (the old text advertised a "bring your
# own key" button that no longer exists).
#
# NOTE, corrected 2026-08-13: this comment used to claim the server text
# "always outranks the client's fallback copy". That is no longer true —
# ENG-1304 made the desktop card use FIXED client copy for `token_limit` and
# ignore this string (`ChatView.jsx`), because the gateway's wording predated
# pay-as-you-go. So editing this message alone changes nothing a desktop user
# sees; the client must change too. It still reaches non-desktop consumers and
# is the fallback for a client that doesn't hardcode.
# This copy is now the EMPTY-WALLET half only. The spent-free-allowance case was
# split off into ALLOWANCE_EXHAUSTED_CODE below (ENG-1537 step 8, in scope from
# the start) — an earlier revision of this comment argued for deferring it and
# is superseded. Because the split introduces a code an older renderer has no
# branch for, the delivery order matters: the cowork card must reach users
# BEFORE this server change, or a user who has just spent their free grant
# falls through to a buttonless alert at the paid-conversion moment. That is a
# release-ordering constraint, not a merge-ordering one.
TOKEN_LIMIT_USER_MESSAGE = (
    "You're out of credits. Add credits to keep working."
)

# Wire-level code for the out-of-credits case. Named `token_limit` for wire
# back-compat with clients already branching on it; it now covers both the
# empty-wallet and spent-allowance reasons above.
TOKEN_LIMIT_CODE = "token_limit"

# Curated copy + wire code for a spent FREE monthly allowance (gateway 429
# `included_allowance_exhausted`). Split from `token_limit` in ENG-1537: both
# denials used to share the out-of-credits card, but they are different
# situations. `access.py` in auth decides it, and the logic is exact — this
# reason fires ONLY for a free-bucket model on an org that has NEVER topped up
# ("a non-free model always needs the wallet"). So the person seeing this card
# has not spent money; they have used the monthly grant. Two consequences the
# copy leans on:
#   * There is a FREE way forward — the allowance resets, and the gate tells us
#     when on `X-MindsHub-Reset-At`. Hiding that while asking for money is the
#     defect.
#   * Credits genuinely unlock the rest of the catalogue for this user, because
#     non-free models need a wallet they do not have.
# The date is interpolated client-side from `reset_at`; this string is the
# fallback for consumers that don't render the card.
ALLOWANCE_EXHAUSTED_CODE = "included_allowance_exhausted"
ALLOWANCE_EXHAUSTED_USER_MESSAGE = (
    "You've used this month's free tokens. Add credits to keep working now and "
    "unlock Claude, GPT, Gemini, Kimi, DeepSeek and more."
)

# Curated copy + wire code for a VELOCITY rate-limit (gateway 429
# `rate_limited`) — the org exceeded requests/tokens per minute, NOT its credit
# balance. ENG-1537: this was the fifth and only unmapped gateway reason, so it
# fell through to the bare-status 429 rule below and rendered as out-of-credits
# — telling the user to buy something that cannot raise a per-minute ceiling.
# The gateway is explicit that the two are different (`rate_limited()` in
# minds/inference/errors.py: "the caller should slow down and retry after
# retry_after seconds, not wait for an allowance reset or add credits").
# Copy names the remedy (wait) and rules out the one the user would otherwise
# assume, because they have just been shown a credits card for this.
RATE_LIMITED_CODE = "rate_limited"
RATE_LIMITED_USER_MESSAGE = (
    "Too many requests too quickly. Wait a moment and continue — "
    "this isn't a credits problem."
)

# Curated copy + wire code for a transient billing/auth policy outage — the
# gateway couldn't reach the service that decides whether a call is paid for
# (gateway 503 `policy_unavailable`). This is retryable and must NOT be shown
# as out-of-credits: the user has done nothing wrong and just needs to retry.
POLICY_UNAVAILABLE_CODE = "policy_unavailable"
POLICY_UNAVAILABLE_USER_MESSAGE = (
    "Billing is temporarily unavailable. Please retry in a moment."
)

# Wire code + fallback copy for a model the provider can't serve — the gateway's
# 404 (`X-MindsHub-Reason: unknown_model`, body `code: model_not_found`), a BYOK
# OpenAI 404 carrying the same body code, or the equivalent from Gemini/Anthropic.
# Adding credits can't fix it, so this steers to Settings rather than to the
# out-of-credits card.
#
# The code mirrors the OpenAI-dialect `error.code` that every one of those
# providers emits (and that anton's `classify_404` keys on), so one name travels
# the whole path — gateway → anton → here → the renderer's card. It renames the
# server-invented `unknown_model` wire code (ENG-1282 gave that one a card, so
# the rename moves in lockstep with ChatView.jsx and the inventory test below).
#
# The rename is not cosmetic: under the old name this code could only ever be
# produced by the reason-header branch, whose copy is generic. anton's typed
# ModelUnavailableError — the one that NAMES the model — carries code
# `model_not_found`, so it could never match `MODEL_UNAVAILABLE_CODES` and the
# model id never reached the card. One shared name is what closes ENG-1358.
#
# The copy is only a fallback. anton's ModelUnavailableError carries curated text
# that NAMES the rejected model, and `friendly_turn_error` prefers it — this is
# for a version-skewed anton that sends the reason header without the typed error.
MODEL_NOT_FOUND_CODE = "model_not_found"
MODEL_NOT_FOUND_USER_MESSAGE = (
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

# Canonical Anton exception name after its remote scrubber converts an
# exception to ``"TypeName: message"``. Remote errors no longer carry Python
# type identity, so an exact name is the only typed discriminator left.
PROVIDER_AUTH_ERROR_TYPE_NAME = "ProviderAuthError"

# Anton's pre-typed 401 copy, still emitted by the remote worker pods. Those run
# the `minds-anton-scratchpad` image, whose anton is pinned in
# scratchpad-controller (`values-staging.yaml`, `values-prod.yaml`) and bumped
# independently of this server's vendored dep — `turnqueue/producer.py` says so
# and relies on it to let the repos deploy in any order. Until both pins carry
# anton's ProviderAuthError, dropping this prefix would silently downgrade every
# hosted 401 to the generic code and take the Reconnect card with it.
#
# Safe here in a way the in-process `is_auth_error` is not: `remote_turn_error`
# only ever reads anton's own `_scrub` output, never an arbitrary tool
# exception, and the match is anchored to the start of the message.
LEGACY_AUTH_ERROR_MESSAGE_PREFIX = "invalid api key"

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
# Every code that means "the turn died on the MODEL, and picking another one is
# the remedy" — the two legacy 403s plus the live 404. They share a renderer
# card; only its copy differs. model_not_found is the one that still occurs.
# Public: responses.py branches on this to decide whether the failure frame
# carries `model`. Shared rather than re-listed there, so the two can't drift —
# and so a merge conflict in that elif-chain has no tuple members to silently
# drop (the ENG-1358 re-review's rebase hazard).
MODEL_UNAVAILABLE_CODES = frozenset(
    {MODEL_ACCESS_DENIED_CODE, MODEL_DISABLED_CODE, MODEL_NOT_FOUND_CODE}
)

# Fallback copy if the exception somehow carries no usable message — anton
# normally supplies curated, user-facing copy which we pass through verbatim.
# Deliberately neutral: this legacy denial isn't necessarily fixable with
# credits, so the copy steers to picking another model rather than to billing.
MODEL_UNAVAILABLE_FALLBACK_MESSAGE = (
    "That model isn't available right now. Switch to another model in Settings."
)

# The X-MindsHub-Reason header values the inference gateway sets to name the
# billing decision precisely. Preferred over status/message heuristics.
#
# This is the COMPLETE set the gateway can emit: `denial_error()`
# (minds/inference/errors.py) maps four gate reasons explicitly and fails closed
# to `policy_unavailable` for anything else, so no sixth value reaches a client.
# Verified 2026-08-13 — `rate_limited` was the one missing here (ENG-1537), and
# its absence is why a velocity limit read as out-of-credits.
_REASON_WALLET_EMPTY = "wallet_empty"
_REASON_ALLOWANCE_EXHAUSTED = "included_allowance_exhausted"
_REASON_POLICY_UNAVAILABLE = "policy_unavailable"
_REASON_UNKNOWN_MODEL = "unknown_model"
_REASON_RATE_LIMITED = "rate_limited"

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


def is_content_validation_error(exc: Exception) -> bool:
    """Detect a permanent, content-SHAPED provider rejection — a content block
    in conversation history reached the model in a shape it doesn't parse
    (ENG-1992), not a provider-availability issue. Retrying the identical
    request fails identically every time, since the same translation runs
    fresh from stored history on every call — the request never changes
    between attempts.

    A strict superset of the older, narrower `is_image_format_error`: this
    recognizes BOTH known dialects (OpenAI Responses' "Invalid value: 'x'.
    Supported values are: ..." and Anthropic's "Input tag 'x' found using
    'type' does not match any of the expected tags"), and the caller that
    detects this repairs the conversation's stored history (unlike
    `is_image_format_error`, whose own docstring says it can't).
    """
    try:
        from anton.core.llm.provider import ContentValidationError

        if isinstance(exc, ContentValidationError):
            return True
    except Exception:
        # anton not importable / the type moved — fall back to the stable
        # provider-message phrasings below.
        pass
    s = str(exc).lower()
    if "supported values are" in s:
        return True
    if "does not match any of the expected tags" in s:
        return True
    return False


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
    if isinstance(code, str) and code in MODEL_UNAVAILABLE_CODES:
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
    """Whether ``exc`` is Anton's canonical LLM-provider auth failure.

    Provider and tool errors can contain arbitrary 401 or invalid-key text. Only
    Anton's typed exception proves the failed credential belongs to the active
    LLM provider and may select the reconnect/update-key card.

    Imported lazily like every other anton type in this module
    (``ContentValidationError``, ``TokenLimitExceeded``, ``ModelUnavailableError``,
    ``ProviderOverloadedError``). A module-scope import would turn an anton
    without this symbol — staging and main today, and the ``branch = "main"``
    pin this repo's pyproject documents — into a failed app import rather than
    one missing error card.
    """
    try:
        from anton.core.llm.provider import ProviderAuthError

        return isinstance(exc, ProviderAuthError)
    except Exception:
        # A version-skewed anton predates the typed error, so its 401 still
        # arrives as the bare ConnectionError copy the pods emit.
        return isinstance(exc, ConnectionError) and str(exc).lower().startswith(
            LEGACY_AUTH_ERROR_MESSAGE_PREFIX
        )


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


def _gateway_denial_code(exc: BaseException) -> str | None:
    """The gateway's body ``code`` from anywhere in the cause chain, if any.

    Fallback discriminator for a lane that delivers the body but loses the
    ``X-MindsHub-Reason`` header — the failure mode ENG-1363 reports on the
    Anthropic ``/v1/messages`` door. The gateway sets ``code`` and ``reason`` to
    the same value, so one substitutes for the other.

    Reads both dialects: the OpenAI SDK peels the ``error`` envelope so ``code``
    sits at top level, while a proxy may deliver it nested. Returns only values
    this module knows, so an unrelated provider ``code`` can never be mistaken
    for a gateway denial (ENG-1537).

    **The caller must host-gate this**, exactly like the bare-status rule below.
    A response body is third-party-controlled on a BYOK
    ``OPENAI_COMPATIBLE`` provider, so without the gate any endpoint could send
    ``{"code": "wallet_empty"}`` and put our billing CTA — and the MindsHub
    top-up link — in front of a user who has no MindsHub balance at all. The
    allowlist alone does not prevent that: it constrains WHICH verdict can be
    selected, not WHO can select one.
    """
    known = (
        _REASON_WALLET_EMPTY,
        _REASON_ALLOWANCE_EXHAUSTED,
        _REASON_POLICY_UNAVAILABLE,
        _REASON_UNKNOWN_MODEL,
        _REASON_RATE_LIMITED,
    )
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        body = getattr(cur, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], dict):
            body = body[0]  # Gemini's single-element-array dialect
        if isinstance(body, dict):
            env = body.get("error") if isinstance(body.get("error"), dict) else {}
            code = body.get("code") or env.get("code")
            if isinstance(code, str) and code in known:
                return code
        cur = cur.__cause__ or cur.__context__
    return None


def retry_after_seconds(exc: BaseException) -> float | None:
    """The ``Retry-After`` hint from anywhere in the cause chain, in seconds.

    The gateway sends it on every velocity 429 as integer seconds. The renderer
    needs it to time-gate its Retry button: an ungated retry re-sends a large
    context into the limiter that just refused it, which is the same
    amplification loop the fix removed — only user-initiated (ENG-1537).

    Integer-seconds form only. The HTTP-date form is legal but nothing in use
    emits it, and misreading a date as a number would gate the button for
    centuries; unparseable, negative and non-finite values are dropped so the
    caller falls back to an ungated (but honest) card.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        resp = getattr(cur, "response", None)
        headers = getattr(resp, "headers", None) or getattr(cur, "headers", None)
        if headers is not None:
            try:
                raw = headers.get("retry-after") or headers.get("Retry-After")
            except Exception:
                raw = None
            if raw is not None:
                try:
                    secs = float(str(raw).strip())
                except (TypeError, ValueError):
                    secs = None
                if (
                    secs is not None
                    and secs == secs  # not NaN
                    and secs not in (float("inf"), float("-inf"))
                    and secs >= 0
                ):
                    # Clamped HERE, at the source, so `retry_after` and
                    # `retry_at` can never disagree on the wire. Unclamped, a
                    # hostile or unit-confused header put 999999999999 on the
                    # payload while `retry_at` was dropped for being out of
                    # datetime range — two fields describing one wait, one
                    # absurd and one absent (review: pnewsam). Desktop reads
                    # only `retry_at`, but the interval exists for non-desktop
                    # consumers and is reachable through the reason header,
                    # which is not host-gated.
                    return min(secs, _MAX_RETRY_AFTER_S)
        cur = cur.__cause__ or cur.__context__
    return None


# Largest `Retry-After` we will honour, in seconds — applied by BOTH
# `retry_after_seconds` and `retry_at_instant` so the interval and the instant
# always describe the same wait. Anything beyond a day is either hostile or a
# unit error (an endpoint emitting epoch-millis), and every consumer clamps far
# below it anyway: the desktop card gates at 10 minutes, and anton cards
# immediately above its own 60s cap.
_MAX_RETRY_AFTER_S = 86_400.0


def retry_at_instant(retry_after: float | None) -> str | None:
    """``retry_after`` seconds as an absolute, offset-bearing UTC instant.

    The renderer time-gates its Retry button, and it needs an anchor. It cannot
    use the message's own ``created_at``: cowork-server serialises that naive
    and offset-less, so JavaScript parses it as LOCAL time — west of UTC the
    button gates for hours, east of it the gate silently no-ops. A test suite
    pinned to ``TZ=UTC`` cannot see either.

    So the anchor is computed here, where the clock and the interval are both
    known, and sent as an explicit instant. Offset-bearing on purpose: an
    unqualified ISO string would reintroduce exactly the parsing ambiguity this
    exists to remove (ENG-1537).
    """
    # Bounded before the arithmetic. `timedelta` raises OverflowError past the
    # datetime range, and this runs INSIDE the terminal error handler — an
    # unhandled raise there skips `persist()` and `buffer.close("error")`, so
    # the SSE buffer never terminates and the client spins on keepalives
    # forever with no failure frame. A hint beyond a day is meaningless anyway:
    # the renderer clamps its gate to 10 minutes, and anton cards immediately
    # above its own 60s cap.
    if retry_after is None or not (0 <= retry_after <= _MAX_RETRY_AFTER_S):
        return None
    return (
        datetime.now(timezone.utc) + timedelta(seconds=float(retry_after))
    ).isoformat().replace("+00:00", "Z")


def allowance_reset_at(exc: BaseException) -> str | None:
    """The ``X-MindsHub-Reset-At`` instant from anywhere in the cause chain.

    The auth gate sets it from the billing window's end on an allowance denial
    (`inference_authorize.py`, ``reset_at=window.end``) and deliberately leaves
    it unset on a velocity denial — so its presence is itself a signal.

    Passed through as the opaque ISO string the gate sent; the renderer owns
    formatting and the "resets next month" fallback, because only it knows the
    viewer's locale and timezone. Returned as-is rather than parsed here: a
    server-side parse would have to pick a timezone, and picking the wrong one
    shifts the date the user reads by a day (ENG-1537).
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        resp = getattr(cur, "response", None)
        headers = getattr(resp, "headers", None) or getattr(cur, "headers", None)
        if headers is not None:
            try:
                raw = headers.get("x-mindshub-reset-at") or headers.get("X-MindsHub-Reset-At")
            except Exception:
                raw = None
            if raw:
                return str(raw)
        cur = cur.__cause__ or cur.__context__
    return None


def _origin_is_known_third_party(host: str | None) -> bool:
    """Whether the failing request provably went somewhere that is NOT our gateway.

    Deliberately three-valued, unlike ``_from_minds_gateway``: this answers
    "do we KNOW it was someone else", so an unknown origin (``host is None``)
    is not treated as third-party.

    That distinction is the whole reason the header carrier could be gated at
    all. A flat ``not _from_minds_gateway(host)`` also rejects the
    unknown-origin case, which breaks
    ``test_reason_header_maps_even_without_request_url`` — a deliberate test
    asserting the header still maps when the response carries no URL.

    Leaving unknown-origin trusted is safe, and checked (ENG-1686). A real SDK
    error always carries its request, so ``host`` resolves for every genuine
    HTTP response. There are exactly two routes to ``host is None``, and a
    remote server can choose neither:

    1. a response with no request attached — its ``.url`` raises
       ``RuntimeError``, which is our own client plumbing, not the peer's;
    2. an exception carrying ``headers`` with **no response object at all**
       (``_http_error_context`` falls back to ``getattr(cur, "headers", None)``,
       and ``_response_url_host`` names this case too). Nothing in anton or
       cowork-server raises such an exception today, so the lane is unreachable
       rather than merely unlikely — but a future client that attaches headers
       directly to an error would reopen it silently, which is why it is named
       here rather than left to be rediscovered.
    """
    return host is not None and not _from_minds_gateway(host)


def _map_gateway_reason(reason: str) -> tuple[str, str] | None:
    """Map an ``X-MindsHub-Reason`` header value to ``(code, user_message)``.

    An empty wallet is "out of credits". A spent free allowance is NOT — that
    user has never paid, and the grant resets — so it has its own code and copy
    (ENG-1537). A policy outage is transient. An unknown model can't be fixed
    with credits.

    A velocity ``rate_limited`` is NOT out of credits and must never share that
    card: the ceiling is per-minute tokens, so buying credits cannot lift it and
    the remedy is to wait (ENG-1537). It is listed here rather than left to the
    bare-status 429 rule below precisely because that rule cannot tell the two
    429s apart — and the gateway already did.
    """
    if reason == _REASON_RATE_LIMITED:
        return RATE_LIMITED_CODE, RATE_LIMITED_USER_MESSAGE
    if reason == _REASON_ALLOWANCE_EXHAUSTED:
        return ALLOWANCE_EXHAUSTED_CODE, ALLOWANCE_EXHAUSTED_USER_MESSAGE
    if reason == _REASON_WALLET_EMPTY:
        return TOKEN_LIMIT_CODE, TOKEN_LIMIT_USER_MESSAGE
    if reason == _REASON_POLICY_UNAVAILABLE:
        return POLICY_UNAVAILABLE_CODE, POLICY_UNAVAILABLE_USER_MESSAGE
    if reason == _REASON_UNKNOWN_MODEL:
        return MODEL_NOT_FOUND_CODE, MODEL_NOT_FOUND_USER_MESSAGE
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
    #
    # Origin-checked (ENG-1686). This used to read "unconditional on origin:
    # only the gateway sets X-MindsHub-Reason", which is an assumption about
    # well-behaved upstreams rather than something enforced: on a BYOK
    # OPENAI_COMPATIBLE provider the response is entirely third-party
    # controlled, so any endpoint could set the header and choose which billing
    # card the user sees — including the out-of-credits CTA for a wallet that
    # is fine. The body-`code` twin below is gated for exactly this reason and
    # the argument is carrier-agnostic.
    #
    # Skipped rather than returned-None on purpose: a spoofed header should
    # lose its authority, not suppress the rest of the ladder.
    if reason is not None and not _origin_is_known_third_party(host):
        mapped = _map_gateway_reason(reason)
        if mapped is not None:
            # ...except for unknown_model, where anton's typed error is strictly
            # better than the header: `classify_404` already resolved this to a
            # ModelUnavailableError whose copy NAMES the rejected model ("The
            # model 'deepseek-v4-flash' isn't available: …"), while the header
            # only says *that* a model was rejected. Returning the header copy
            # here threw the model id away and left the user with nothing to act
            # on — ENG-1358. The billing reasons have no such typed counterpart,
            # so they still short-circuit.
            if mapped[0] == MODEL_NOT_FOUND_CODE:
                if model_info is _UNSET:
                    model_info = model_unavailable_info(exc)
                if model_info is not None:
                    return MODEL_NOT_FOUND_CODE, str(exc) or mapped[1]
            return mapped

    # Same decision from the body's `code` when the header didn't survive the
    # trip (ENG-1363's Anthropic lane strips it). The gateway sets both fields
    # to the same value, so this is the identical decision from a second
    # carrier — NOT a heuristic. It must stay above the bare-status rule below:
    # that rule cannot tell the velocity 429 from the allowance 429, and
    # guessing "out of credits" for a rate limit is the ENG-1537 defect.
    denial_code = _gateway_denial_code(exc)
    # NOTE: strict here (provably-gateway), unlike the header above. The header
    # needs the looser three-valued check because an existing test requires an
    # unknown origin to still map; the body carrier has no such constraint, so
    # it stays as tight as it can be. Don't "unify" these without re-reading
    # test_reason_header_maps_even_without_request_url.
    if denial_code is not None and _from_minds_gateway(host):
        mapped = _map_gateway_reason(denial_code)
        if mapped is not None:
            return mapped

    # anton exhausted its rate-limit wait budget and re-raised with the code
    # (ENG-1537). Checked here, above the bare-status rule, because that
    # exception still carries the original 429 in its cause chain — so without
    # this the honest "waiting didn't clear it" failure is relabelled
    # out-of-credits, which is exactly what the wait was added to stop.
    # Read the code attribute DIRECTLY rather than via provider_overloaded_info:
    # that helper's version-skew duck-type only accepts
    # ``code == provider_overloaded``, so under the exact skew its own docstring
    # exists for (anton's type not importable), an exhausted rate-limit wait
    # would fall through to the bare-status rule and be relabelled
    # out-of-credits — the outcome this hoist exists to prevent (ENG-1537
    # review).
    # `not hasattr(exc, "response")` is load-bearing, not defensive noise.
    # `openai.APIStatusError` populates `.code` FROM THE RESPONSE BODY, and on a
    # BYOK OPENAI_COMPATIBLE provider that body is third-party-controlled — so
    # without this an endpoint could both select this verdict and, because
    # `str(exc)` embeds its body, have its own text rendered as our user-facing
    # copy (a clickable "click https://evil.example to fix" was executed against
    # the unguarded version). anton's ProviderOverloadedError carries no
    # `.response`; every SDK error does. That keeps the version-skew case this
    # hoist exists for — the duck-type still works when anton's type isn't
    # importable — while excluding every SDK exception.
    #
    # Host-gating instead would ALSO disable the skew case, because anton's
    # re-raise carries no URL and `_from_minds_gateway(None)` is False.
    if getattr(exc, "code", None) == RATE_LIMITED_CODE and not hasattr(exc, "response"):
        return RATE_LIMITED_CODE, str(exc) or RATE_LIMITED_USER_MESSAGE

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
    # Checked before is_image_format_error: a content-SHAPE mismatch (this
    # detector) and a genuinely corrupt/unsupported image FILE (that one)
    # are different failures with different correct copy — this one has
    # already been auto-repaired server-side, that one needs the user to
    # re-upload. The two detectors' phrasings don't overlap.
    if is_content_validation_error(exc):
        return CONTENT_RECOVERY_CODE, CONTENT_RECOVERY_USER_MESSAGE
    if is_image_format_error(exc):
        return IMAGE_FORMAT_CODE, IMAGE_FORMAT_USER_MESSAGE
    return None


def remote_turn_error(error: str | None) -> tuple[str, str]:
    """Map a remote pod's ``turn_failed`` error STRING to ``(code, message)``.

    The in-process path classifies exceptions (`friendly_turn_error`); remote
    turns arrive as scrubbed strings shaped ``"ExceptionType: message"`` (see
    anton.cloud_turn._scrub), so this keys on the type-name prefix. Curated
    anton copy (overloaded / model-gate) passes through; everything unmapped
    gets the generic redacted message — never the raw provider text.
    """
    text = (error or "").strip()
    type_name, _, message = text.partition(":")
    message = message.strip()
    if type_name == "TokenLimitExceeded":
        return TOKEN_LIMIT_CODE, TOKEN_LIMIT_USER_MESSAGE
    if type_name == "ProviderOverloadedError":
        return PROVIDER_OVERLOADED_CODE, message or PROVIDER_OVERLOADED_FALLBACK_MESSAGE
    if type_name == "ModelUnavailableError":
        # _scrub sends "Type: message" — the structured `code` doesn't survive,
        # so 403-gate and 404-not-found are indistinguishable here. Default to
        # the CONSERVATIVE one: model_not_found steers to Settings and promises
        # nothing, while model_access_denied renders a "Top up balance" button
        # that is simply wrong for a model that doesn't exist — and since the
        # current gateway no longer emits the 403 codes at all, not-found is
        # also the likelier case.
        #
        # The message is returned for the SSE `error` field and the DB sidecar,
        # not for the card: both model cards render their own literal copy and
        # never read `m.content` (ChatView.jsx). So the choice of code decides
        # everything the user sees, which is why it errs conservative.
        #
        # Known gap, not fixed here: this path also can't supply `model` —
        # producer.py emits response_failed_sse without it, so a remote/hosted
        # turn still shows the UNNAMED copy. Naming it needs anton to carry the
        # code+model through _scrub's wire format (tracked separately).
        return MODEL_NOT_FOUND_CODE, message or MODEL_UNAVAILABLE_FALLBACK_MESSAGE
    if type_name == PROVIDER_AUTH_ERROR_TYPE_NAME or (
        type_name == "ConnectionError"
        and message.lower().startswith(LEGACY_AUTH_ERROR_MESSAGE_PREFIX)
    ):
        return AUTH_ERROR_CODE, AUTH_ERROR_USER_MESSAGE
    if type_name == "ContentValidationError":
        # ENG-1992: the repair itself (stripping the offending image blocks
        # from stored history) is triggered by the caller, keyed on this
        # same code — see producer.py. The curated message anton constructs
        # is already safe to show verbatim, but the code is what the client
        # keys its (different, "already fixed") copy on, so return the
        # stable curated constant rather than passing `message` through.
        return CONTENT_RECOVERY_CODE, CONTENT_RECOVERY_USER_MESSAGE
    return GENERIC_TURN_ERROR_CODE, GENERIC_TURN_ERROR_MESSAGE


def response_failed_payload(
    error: str,
    code: str,
    *,
    reconnectable: bool | None = None,
    provider_label: str | None = None,
    model: str | None = None,
    retry_after: float | None = None,
    retry_at: str | None = None,
    reset_at: str | None = None,
) -> dict:
    """Wire payload for a ``response.failed`` event (SSE + DB sidecar).

    ``reconnectable`` / ``provider_label`` are included only for the
    ``provider_auth`` case so the renderer can offer "Reconnect" (MindsHub) vs
    "Open Settings" (BYOK); ``model`` only for the model-403 case so the card
    can name the locked model ("Sonnet needs credits"); ``retry_after`` only for
    ``rate_limited`` so the card can time-gate its Retry — omitted otherwise to
    keep the shape unchanged for every other failure. All additive: an older
    client ignores fields it doesn't read.
    """
    payload = {"type": "response.failed", "code": code, "error": error}
    if reconnectable is not None:
        payload["reconnectable"] = reconnectable
    if provider_label is not None:
        payload["provider_label"] = provider_label
    if model is not None:
        payload["model"] = model
    if retry_after is not None:
        payload["retry_after"] = retry_after
    if retry_at is not None:
        payload["retry_at"] = retry_at
    if reset_at is not None:
        payload["reset_at"] = reset_at
    return payload


def response_failed_sse(
    error: str,
    code: str,
    *,
    reconnectable: bool | None = None,
    provider_label: str | None = None,
    model: str | None = None,
    retry_after: float | None = None,
    retry_at: str | None = None,
    reset_at: str | None = None,
) -> str:
    """Build a ``response.failed`` SSE frame (same wire shape the renderer's
    parser already handles, plus the optional auth/model/retry-after fields)."""
    payload = response_failed_payload(
        error,
        code,
        reconnectable=reconnectable,
        provider_label=provider_label,
        model=model,
        retry_after=retry_after,
        retry_at=retry_at,
        reset_at=reset_at,
    )
    return f"event: response.failed\ndata: {json.dumps(payload)}\n\n"
