"""Browser Control (Milestone 1, read-only) — DTOs, enums, and the
canonical status/result-code contract.

Single source of truth for the names/mappings shared across the browser
workstreams (see `/code/.plans/v1-browser-control-m1.md` "Shared
contracts"). Everything the server persists or exposes is **content-free**:
host-only `domain`, action type/class, timing, and typed codes only — never
page text, full URLs, paths/queries, titles, hrefs, cookies, or form values.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

_E = TypeVar("_E", bound=Enum)


def coerce_enum(enum_cls: type[_E], value: _E | str) -> _E:
    """Return `value` as `enum_cls`, accepting either the enum or its value."""
    return value if isinstance(value, enum_cls) else enum_cls(value)


def coerce_uuid(value: UUID | str) -> UUID:
    """Return `value` as a `UUID`, accepting either a `UUID` or a string."""
    return value if isinstance(value, UUID) else UUID(str(value))


# ── Action names (LLM verb → stored action_type) ─────────────────────
# The LLM sees `follow_link`; everything below the tool uses `navigate`.
class BrowserActionType(str, Enum):
    """The stored `action_type` on a persisted BrowserAction row."""

    inspect = "inspect"
    navigate = "navigate"
    scroll = "scroll"
    wait = "wait"


class BrowserActionClass(str, Enum):
    """Coarse capability class a grant is checked against.

    M1 is read-only: `inspect`/`scroll`/`wait` are `read`; `follow_link`
    (stored as `navigate`) is `navigate`. No `interact` class exists in M1
    (click/type/submit are out of scope).
    """

    read = "read"
    navigate = "navigate"


# LLM `action` verb → stored `action_type`. `follow_link` translates to
# `navigate`; the other three map to themselves.
LLM_ACTION_TO_TYPE: dict[str, BrowserActionType] = {
    "inspect": BrowserActionType.inspect,
    "follow_link": BrowserActionType.navigate,
    "scroll": BrowserActionType.scroll,
    "wait": BrowserActionType.wait,
}

# Stored `action_type` → the capability class a grant is checked against.
ACTION_TYPE_TO_CLASS: dict[BrowserActionType, BrowserActionClass] = {
    BrowserActionType.inspect: BrowserActionClass.read,
    BrowserActionType.navigate: BrowserActionClass.navigate,
    BrowserActionType.scroll: BrowserActionClass.read,
    BrowserActionType.wait: BrowserActionClass.read,
}


# ── Result codes vs external error kinds ─────────────────────────────
class ResultCode(str, Enum):
    """WS4-internal result codes — richer than the external vocabulary.

    These are what a BrowserAction row records. They MUST be mapped to a
    `BrowserErrorKind` before crossing into the agent tool / UI.
    """

    ok = "ok"
    timeout = "timeout"
    target_lost = "target_lost"
    unapproved_tab = "unapproved_tab"
    permission_denied = "permission_denied"
    error = "error"


class BrowserErrorKind(str, Enum):
    """The canonical external vocabulary: `ok` plus the five error kinds.

    This is the only status set the agent tool and UI speak.
    """

    ok = "ok"
    permission_denied = "permission_denied"
    bridge_disconnected = "bridge_disconnected"
    tab_closed = "tab_closed"
    navigation_failed = "navigation_failed"
    unsupported_action = "unsupported_action"


def result_code_to_error_kind(
    code: "ResultCode | str",
    action_type: "BrowserActionType | str | None" = None,
) -> BrowserErrorKind:
    """Map a WS4-internal `result_code` to the canonical external kind.

    Per the Shared-contracts table:
        ok               → ok
        timeout          → bridge_disconnected
        target_lost      → tab_closed
        unapproved_tab   → permission_denied
        permission_denied→ permission_denied
        error            → navigation_failed (for `navigate`) else bridge_disconnected

    `timeout`, `target_lost`, and `unapproved_tab` are NEVER surfaced raw.
    """
    code = coerce_enum(ResultCode, code)
    if code == ResultCode.ok:
        return BrowserErrorKind.ok
    if code == ResultCode.timeout:
        return BrowserErrorKind.bridge_disconnected
    if code == ResultCode.target_lost:
        return BrowserErrorKind.tab_closed
    if code in (ResultCode.unapproved_tab, ResultCode.permission_denied):
        return BrowserErrorKind.permission_denied
    # code == ResultCode.error
    at = coerce_enum(BrowserActionType, action_type) if action_type is not None else None
    if at == BrowserActionType.navigate:
        return BrowserErrorKind.navigation_failed
    return BrowserErrorKind.bridge_disconnected


# ── Control / bridge / lifecycle states ──────────────────────────────
class ControlState(str, Enum):
    """Per-session control gate.

    `stopped`/`taken_over` are control terminal states — NOT error kinds.
    They gate command dispatch and are surfaced by the UI as distinct
    (non-error) terminal states.
    """

    active = "active"
    stopped = "stopped"
    taken_over = "taken_over"


class BridgeState(str, Enum):
    """Electron-main CDP bridge lifecycle, mirrored server-side."""

    disconnected = "disconnected"
    awaiting_approval = "awaiting_approval"
    connected = "connected"
    lost = "lost"


class PermissionDecision(str, Enum):
    granted = "granted"
    denied = "denied"
    expired = "expired"
    revoked = "revoked"


class ActionStatus(str, Enum):
    """Persisted lifecycle of a single BrowserAction row."""

    pending = "pending"
    in_flight = "in_flight"
    observed = "observed"
    failed = "failed"


# ── Content-free digest allowlist ────────────────────────────────────
# The ONLY keys permitted in a persisted `BrowserAction.observed_result`.
# Anything else (text, url, path, query, title, href, cookies, form
# values, selectors, …) must be rejected by the store.
ALLOWED_DIGEST_KEYS: frozenset[str] = frozenset(
    {"http_status", "final_domain", "link_count", "settled"}
)


class DisallowedDigestKeyError(ValueError):
    """Raised when a persisted observed digest carries a non-allowlisted key.

    The guardrail that keeps page content out of the database (AC8).
    """


def assert_content_free_digest(digest: dict[str, Any]) -> None:
    """Raise `DisallowedDigestKeyError` if `digest` has a disallowed key OR
    a disallowed VALUE.

    Validating keys alone is not enough: a URL smuggled through an allowed
    key (e.g. ``{"final_domain": "https://x.com/path?token=..."}``) would
    still leak content. So every value is type/shape-checked too — a
    ``final_domain`` MUST be host-only (no scheme/path/query/fragment/port/
    userinfo), and the numeric/boolean fields MUST be the right type.
    """
    if not isinstance(digest, dict):
        raise DisallowedDigestKeyError(
            f"observed digest must be a dict, got {type(digest).__name__}"
        )
    extra = set(digest.keys()) - ALLOWED_DIGEST_KEYS
    if extra:
        raise DisallowedDigestKeyError(
            "observed digest contains disallowed key(s): "
            + ", ".join(sorted(extra))
            + f". Allowed keys: {', '.join(sorted(ALLOWED_DIGEST_KEYS))}."
        )

    # ── value guards (content-free) ──────────────────────────────────
    fd = digest.get("final_domain")
    if fd is not None:
        if not isinstance(fd, str) or host_only(fd) != fd:
            raise DisallowedDigestKeyError(
                "observed digest `final_domain` must be a bare host-only "
                f"domain (no scheme/path/query/port/userinfo), got {fd!r}."
            )
    hs = digest.get("http_status")
    if hs is not None and (not isinstance(hs, int) or isinstance(hs, bool)):
        raise DisallowedDigestKeyError(
            f"observed digest `http_status` must be an int, got {hs!r}."
        )
    lc = digest.get("link_count")
    if lc is not None and (not isinstance(lc, int) or isinstance(lc, bool)):
        raise DisallowedDigestKeyError(
            f"observed digest `link_count` must be an int, got {lc!r}."
        )
    st = digest.get("settled")
    if st is not None and not isinstance(st, bool):
        raise DisallowedDigestKeyError(
            f"observed digest `settled` must be a bool, got {st!r}."
        )


def build_observed_digest(transient: dict[str, Any] | None) -> dict[str, Any]:
    """Build the content-free persisted digest from a transient observed blob.

    Picks ONLY the allowlisted keys, dropping everything else (text,
    headings, links, urls, titles, …). The result is guaranteed to pass
    `assert_content_free_digest`.
    """
    if not transient:
        return {}
    digest: dict[str, Any] = {}
    if "http_status" in transient and transient["http_status"] is not None:
        digest["http_status"] = int(transient["http_status"])
    # Accept an already host-only `final_domain`; otherwise derive from
    # links count only. We never store a full url — only a bare host.
    fd = transient.get("final_domain")
    if fd:
        digest["final_domain"] = host_only(str(fd))
    links = transient.get("links")
    if isinstance(links, list):
        digest["link_count"] = len(links)
    elif transient.get("link_count") is not None:
        digest["link_count"] = int(transient["link_count"])
    if "settled" in transient and transient["settled"] is not None:
        digest["settled"] = bool(transient["settled"])
    return digest


def host_only(value: str) -> str:
    """Reduce any URL/host string to its bare hostname.

    Strips scheme, path, query, fragment, port, and userinfo, leaving only
    the hostname — the only URL-derived value we ever persist or trace.

    This is the FULL hostname, NOT a PSL registrable domain:
    `sub.example.co.uk` stays `sub.example.co.uk`. The approved grant
    domain is the PSL-registrable-or-exact host Electron computes with its
    PSL library; grant MATCHING against it goes through
    `host_matches_grant` (registrable-host equality via
    `registrable_host`), never equality on this function's output.

    `host_only` is strictly a normalizer and is idempotent —
    `host_only(host_only(x)) == host_only(x)` — because digest validation
    and telemetry use `host_only(v) != v` as an "already normalized" check.
    IPv6 literals are handled bracket-aware: `[::1]:8080` → `::1`, and a
    bare `::1` passes through unchanged.
    """
    if not value:
        return ""
    v = value.strip()
    # Drop scheme.
    if "://" in v:
        v = v.split("://", 1)[1]
    # Drop path/query/fragment.
    for sep in ("/", "?", "#"):
        if sep in v:
            v = v.split(sep, 1)[0]
    # Drop userinfo.
    if "@" in v:
        v = v.rsplit("@", 1)[1]
    # Drop port — bracket-aware so IPv6 literals survive.
    if v.startswith("["):
        # `[::1]:8080` / `[::1]` — the host is the bracket literal; strip
        # the brackets (anything after `]` is the port).
        end = v.find("]")
        v = v[1:end] if end != -1 else v[1:]
    elif v.count(":") == 1:
        # Exactly one colon and no brackets → `host:port`. Two or more
        # colons is a bare IPv6 literal (e.g. an already-normalized `::1`)
        # — leave it intact so the function stays idempotent.
        v = v.split(":", 1)[0]
    return v.lower()


@lru_cache(maxsize=1)
def _psl_extract():
    """The process-wide PSL extractor, configured strictly OFFLINE.

    `suffix_list_urls=()` + `cache_dir=None` pin tldextract to its bundled
    Public Suffix List snapshot — no network fetch and no cache write at
    import or runtime (tests and airgapped installs must never hit the
    network). `include_psl_private_domains=True` mirrors Electron's tldts
    `allowPrivateDomains: true`, so `foo.github.io` is registrable on both
    sides.
    """
    import tldextract

    return tldextract.TLDExtract(
        suffix_list_urls=(),
        cache_dir=None,
        include_psl_private_domains=True,
    )


def registrable_host(value: str) -> str:
    """The PSL registrable host of a URL/host — Electron's grant function.

    Mirrors the TS side exactly (tldts `getDomain` with
    `allowPrivateDomains: true`, falling back to the exact host): normalize
    to the bare hostname via `host_only`, then reduce to the PSL
    registrable domain (`sub.example.co.uk` → `example.co.uk`,
    `foo.github.io` → `foo.github.io`). When the PSL yields nothing —
    `localhost`, IP literals, IPv6, or a host that IS a public/private
    suffix (`github.io`, `co.uk`) — the bare host is returned unchanged.

    This is a MATCHING-only helper: persisted/traced values always stay
    `host_only` output.
    """
    host = host_only(value)
    if not host:
        return ""
    reg = _psl_extract()(host).top_domain_under_public_suffix
    return reg or host


def host_matches_grant(host_or_url: str, grant: str) -> bool:
    """True iff `host_or_url`'s registrable host equals the grant domain.

    The grant is the PSL-registrable-or-exact host Electron computed when
    the tab was approved (`example.co.uk`, `foo.github.io`, `::1`,
    `localhost`, …). Reducing the candidate through the SAME PSL function
    (`registrable_host`) and comparing for equality accepts exactly the set
    Electron accepts: `shop.example.com` under an `example.com` grant, but
    NOT `foo.github.io` under a `github.io` grant (a grant that is itself a
    public/private suffix only ever matches that exact host — Electron's
    getDomain returns null there and it refuses subdomains too). An empty
    host or empty grant NEVER matches — an empty grant must not match
    everything.
    """
    host = registrable_host(host_or_url)
    grant_host = host_only(grant)
    if not host or not grant_host:
        return False
    return host == grant_host


# ── DTOs ─────────────────────────────────────────────────────────────
class BrowserToolVerdict(BaseModel):
    """The typed verdict `BridgeClient` returns to the agent tool.

    `result_code` is the WS4-internal code; the tool maps it to a
    `BrowserErrorKind`. `observed` is the TRANSIENT blob (may carry visible
    extraction for the answer path) — it is NEVER persisted; the store keeps
    only the content-free digest built from it.
    """

    result_code: ResultCode
    action_type: BrowserActionType
    observed: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    domain: str | None = None
    action_id: str | None = None
    detail: str | None = None
    # A control terminal state (`stopped` / `taken_over`) carried SEPARATELY
    # from `result_code`. These are NOT `BrowserErrorKind`s — the agent tool
    # surfaces them as distinct (non-error) terminal states so the UI can
    # render a stopped / taken-over turn differently from a permission
    # denial. `None` when the gate is active.
    control_state: ControlState | None = None


class BridgeCommand(BaseModel):
    """A command the server enqueues for the Electron-main poller to pull."""

    command_id: str
    action_type: BrowserActionType
    session_id: str
    conversation_id: str | None = None
    domain: str | None = None
    href: str | None = None
    direction: str | None = None


class BridgeCommandResult(BaseModel):
    """The result the poller POSTs back for a pulled command."""

    command_id: str
    result_code: ResultCode
    observed: dict[str, Any] | None = None
    detail: str | None = None


class ResumeState(BaseModel):
    """Reconnect/resume snapshot for a session (content-free)."""

    session_id: str
    available: bool = False
    control_state: ControlState = ControlState.active
    bridge_state: BridgeState = BridgeState.disconnected
    domain: str | None = None
    requires_reapproval: bool = False
    last_result_code: ResultCode | None = None
    last_action_type: BrowserActionType | None = None
    action_count: int = 0
