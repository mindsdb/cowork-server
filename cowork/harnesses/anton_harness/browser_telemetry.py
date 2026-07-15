"""Content-free telemetry for the browser-control tool (WS5-T3).

The `browser_control` handler assembles a content-free span for each action
and emits it through whatever `event_sink` the responses stream installed for
the current turn. The span carries ONLY typed/structural fields — the action
class, a host-only `domain`, timing, the typed `result_code`, and the shared
ID set — never page text, full URLs, titles, hrefs, cookies, or form values.

The seam is a `ContextVar` set by `format_responses_stream` at the top of a
turn (and reset in its `finally`). Because the tool handler runs inside the
same asyncio task that drives the stream, the contextvar is visible to it.
When no sink is installed (e.g. unit tests, CLI), `emit_browser_span` is a
no-op — the caller never has to care.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable, Optional

# The span funnel tag ties every event/span to the M1 browser-control funnel
# so it is countable at installation/session level without person-merge.
BROWSER_FUNNEL = "browser_control_m1"

# The exact set of keys a persisted/emitted browser span may carry. Anything
# else (page text, urls-with-content, titles, hrefs, cookies, form values, …)
# is forbidden — this is the content-free guarantee (mirrors AC8 for spans).
ALLOWED_SPAN_KEYS: frozenset[str] = frozenset(
    {
        "command_type",
        "domain",
        "duration_ms",
        "result_code",
        "funnel",
        "installation_id",
        "session_id",
        "task_id",
        "action_id",
    }
)


class DisallowedSpanKeyError(ValueError):
    """Raised when a browser span carries a non-allowlisted key."""


# (event_type, payload) sink for the current turn; None outside a turn.
_span_sink: ContextVar[Optional[Callable[[str, dict], None]]] = ContextVar(
    "browser_span_sink", default=None
)

# Shared cross-workstream IDs for the current turn (from trace_metadata):
# installation_id / task_id. session_id / action_id come from the action
# itself. Set by the harness at turn start, read by the tool handler.
_browser_ids: ContextVar[dict[str, str]] = ContextVar(
    "browser_ids", default={}
)


def set_browser_ids(ids: dict[str, str] | None):
    """Install the per-turn shared browser IDs; returns the reset token."""
    return _browser_ids.set(dict(ids or {}))


def reset_browser_ids(token) -> None:
    _browser_ids.reset(token)


def get_browser_ids() -> dict[str, str]:
    """Return the per-turn shared browser IDs (empty dict outside a turn)."""
    return dict(_browser_ids.get())


def set_span_sink(sink: Optional[Callable[[str, dict], None]]):
    """Install the per-turn span sink; returns the reset token."""
    return _span_sink.set(sink)


def reset_span_sink(token) -> None:
    """Restore the previous span sink (call in a `finally`)."""
    _span_sink.reset(token)


def build_browser_span(
    *,
    command_type: str,
    result_code: str,
    duration_ms: int,
    domain: str | None = None,
    installation_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    action_id: str | None = None,
) -> dict[str, Any]:
    """Assemble a content-free browser span.

    Only allowlisted keys are ever included; `domain` is stored host-only.
    The result is guaranteed to pass `assert_content_free_span`.
    """
    from cowork.schemas.browser import host_only

    span: dict[str, Any] = {
        "command_type": command_type,
        "result_code": result_code,
        "duration_ms": int(duration_ms),
        "funnel": BROWSER_FUNNEL,
    }
    if domain:
        span["domain"] = host_only(str(domain))
    if installation_id:
        span["installation_id"] = str(installation_id)
    if session_id:
        span["session_id"] = str(session_id)
    if task_id:
        span["task_id"] = str(task_id)
    if action_id:
        span["action_id"] = str(action_id)
    assert_content_free_span(span)
    return span


def assert_content_free_span(span: dict[str, Any]) -> None:
    """Raise `DisallowedSpanKeyError` if `span` has a disallowed key OR a
    disallowed VALUE.

    Key-only validation is insufficient: a full URL smuggled through the
    allowed `domain` key (e.g. ``https://x.com/path?token=...``) would still
    leak content into the trace. So `domain` MUST be host-only (no scheme/
    path/query/fragment/port/userinfo), and `command_type`/`result_code`
    MUST be members of their typed vocabularies.
    """
    from cowork.schemas.browser import (
        BrowserActionType,
        BrowserErrorKind,
        host_only,
    )

    if not isinstance(span, dict):
        raise DisallowedSpanKeyError(
            f"browser span must be a dict, got {type(span).__name__}"
        )
    extra = set(span.keys()) - ALLOWED_SPAN_KEYS
    if extra:
        raise DisallowedSpanKeyError(
            "browser span contains disallowed key(s): "
            + ", ".join(sorted(extra))
            + f". Allowed keys: {', '.join(sorted(ALLOWED_SPAN_KEYS))}."
        )

    # ── value guards (content-free) ──────────────────────────────────
    domain = span.get("domain")
    if domain is not None:
        if not isinstance(domain, str) or host_only(domain) != domain:
            raise DisallowedSpanKeyError(
                "browser span `domain` must be a bare host-only domain "
                f"(no scheme/path/query/port/userinfo), got {domain!r}."
            )
    ct = span.get("command_type")
    if ct is not None and ct not in {a.value for a in BrowserActionType}:
        raise DisallowedSpanKeyError(
            f"browser span `command_type` must be a known action type, got {ct!r}."
        )
    rc = span.get("result_code")
    # `result_code` may be either a WS4-internal code or a canonical external
    # kind (the tool emits the effective canonical kind).
    _valid_codes = {k.value for k in BrowserErrorKind} | {
        "timeout",
        "target_lost",
        "unapproved_tab",
        "error",
    }
    if rc is not None and rc not in _valid_codes:
        raise DisallowedSpanKeyError(
            f"browser span `result_code` must be a known code, got {rc!r}."
        )


def emit_browser_span(span: dict[str, Any]) -> None:
    """Emit a content-free browser span through the current turn's sink.

    No-op when no sink is installed. Validates the span is content-free
    before emitting (defence in depth — a malformed span is dropped rather
    than leaking content).
    """
    sink = _span_sink.get()
    if sink is None:
        return
    try:
        assert_content_free_span(span)
    except DisallowedSpanKeyError:
        return
    try:
        sink("response.browser_tool_span", dict(span))
    except Exception:
        # Telemetry is best-effort — never break the turn.
        pass
