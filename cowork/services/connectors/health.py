"""Provider-agnostic connection health + test-result metadata.

This module generalizes the Google-specific token-expiry logic (see
:mod:`cowork.services.connectors.oauth.google`, which refreshes an OAuth
connection's ``access_token`` when its ``expires_at`` is within 10 minutes)
into a single health verdict that applies to *every* saved connection,
OAuth or not:

* ``healthy``       — nothing wrong that we can detect.
* ``expiring_soon`` — an OAuth token will expire within
  :data:`EXPIRING_WINDOW` (the connection still works *now* but should be
  refreshed before it lapses).
* ``broken``        — an OAuth token has already expired with no refresh
  token to recover it, or the last connection test failed.
* ``unknown``       — there isn't enough signal to judge (e.g. a database
  connection that has never been tested and carries no expiry).

The verdict is derived purely from data already on the vault record — the
credential ``fields`` (``auth_type``, ``expires_at``, ``refresh_token``) and
the test-result metadata stamped by :func:`record_test_result`
(``last_tested_at``, ``last_test_result``). It performs **no** network I/O, so
it is cheap to compute for the whole connection list on every page load.

Health is intentionally conservative: we only ever flag ``broken`` /
``expiring_soon`` on a positive signal (an expiry we can read, or a recorded
failure). Absence of signal is ``unknown``, never ``broken`` — we don't want
to scare the user about a perfectly good Postgres connection just because we
can't introspect it without a live probe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# An OAuth token is "expiring_soon" once it's within this window of its
# expiry. Wider than the google refresher's 10-minute auto-refresh threshold
# so the badge warns the user *before* the background refresher would even
# act — by the time it's <10m the refresher should already be handling it.
EXPIRING_WINDOW = timedelta(hours=24)

# Health verdicts. Kept as plain strings (not an Enum) so they serialize
# straight to JSON and match the literals the front-end switches on.
HEALTHY = "healthy"
EXPIRING_SOON = "expiring_soon"
BROKEN = "broken"
UNKNOWN = "unknown"

# Test-result values stamped onto the record by record_test_result(). These
# mirror ProbeOutcome.status minus the transient "unresolved".
TEST_PASS = "pass"
TEST_FAIL = "fail"


@dataclass
class HealthStatus:
    """Computed health of a single connection.

    ``status``       — one of HEALTHY / EXPIRING_SOON / BROKEN / UNKNOWN.
    ``detail``       — short human-readable reason, safe to show in a tooltip.
    ``reconnectable``— True when the UI should offer a "Reconnect" affordance
                        (OAuth connections, or anything we've seen fail). A
                        healthy non-OAuth connection has nothing to reconnect.
    ``expires_at``   — the OAuth expiry we read (echoed back for the UI), or
                        None for connections without one.
    """

    status: str
    detail: str = ""
    reconnectable: bool = False
    expires_at: str | None = None


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating naive values and junk.

    Naive datetimes are assumed UTC (that's how every writer in this codebase
    stamps them — ``datetime.now(timezone.utc).isoformat()``). Returns None on
    anything unparseable so callers can treat "no usable timestamp" uniformly.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_oauth(fields: dict[str, Any] | None) -> bool:
    """True when a credential record is an OAuth connection.

    Matches the marker the OAuth save path writes (``auth_type == "oauth"``)
    and also treats the presence of an access/refresh token as OAuth, so a
    record saved by a path that forgot the marker still gets token-aware
    health.
    """
    fields = fields or {}
    if fields.get("auth_type") == "oauth":
        return True
    return bool(fields.get("access_token") or fields.get("refresh_token"))


def compute_health(
    fields: dict[str, Any] | None,
    *,
    last_test_result: str | None = None,
    now: datetime | None = None,
) -> HealthStatus:
    """Derive a connection's health from its fields + last test result.

    Precedence:
      1. A recorded test *failure* is the strongest negative signal — if the
         last live probe failed, the connection is BROKEN regardless of token
         math (the credentials themselves don't work).
      2. For OAuth connections, token expiry decides:
           * expired with no refresh token        → BROKEN
           * expired but a refresh token exists    → EXPIRING_SOON (the
             background refresher / a reconnect can recover it)
           * expiring within EXPIRING_WINDOW       → EXPIRING_SOON
           * otherwise                             → HEALTHY
      3. A recorded test *pass* with nothing else wrong → HEALTHY.
      4. No signal at all → UNKNOWN.

    ``now`` is injectable for deterministic tests.
    """
    fields = fields or {}
    now = now or datetime.now(timezone.utc)
    oauth = is_oauth(fields)

    # (1) A failed probe trumps everything — the credentials don't work.
    if last_test_result == TEST_FAIL:
        return HealthStatus(
            status=BROKEN,
            detail="The last connection test failed.",
            reconnectable=True,
            expires_at=(fields.get("expires_at") or None) if oauth else None,
        )

    # (2) OAuth token math.
    if oauth:
        expires_at_raw = fields.get("expires_at") or None
        expires_dt = _parse_iso(expires_at_raw)
        has_refresh = bool(str(fields.get("refresh_token") or "").strip())

        if expires_dt is not None:
            if expires_dt <= now:
                if has_refresh:
                    return HealthStatus(
                        status=EXPIRING_SOON,
                        detail="The access token has expired; reconnect to refresh it.",
                        reconnectable=True,
                        expires_at=expires_at_raw,
                    )
                return HealthStatus(
                    status=BROKEN,
                    detail="The access token has expired and cannot be refreshed automatically.",
                    reconnectable=True,
                    expires_at=expires_at_raw,
                )
            if expires_dt - now <= EXPIRING_WINDOW:
                return HealthStatus(
                    status=EXPIRING_SOON,
                    detail="The access token expires soon.",
                    reconnectable=True,
                    expires_at=expires_at_raw,
                )
            return HealthStatus(
                status=HEALTHY,
                detail="Connected.",
                reconnectable=True,  # OAuth is always reconnectable
                expires_at=expires_at_raw,
            )

        # OAuth connection with no readable expiry — treat a prior pass as
        # healthy, otherwise unknown. Still reconnectable (it's OAuth).
        if last_test_result == TEST_PASS:
            return HealthStatus(status=HEALTHY, detail="Last test passed.", reconnectable=True)
        return HealthStatus(status=UNKNOWN, detail="Connection status not yet verified.", reconnectable=True)

    # (3) Non-OAuth with a recorded pass.
    if last_test_result == TEST_PASS:
        return HealthStatus(status=HEALTHY, detail="Last test passed.", reconnectable=False)

    # (4) No signal.
    return HealthStatus(
        status=UNKNOWN,
        detail="Connection status not yet verified — run a test.",
        reconnectable=False,
    )


def safe_runtime_load(vault: Any, engine: str, name: str) -> dict[str, Any]:
    """Load a connection's credentials for agent runtime use, defensively.

    The agent runs credentials inside arbitrary scratchpad code, so a *live*
    auth/timeout failure surfaces there, not here — but the credential *load*
    itself (decrypt of a corrupt record, a missing file, a vault that raises)
    is a chokepoint we own. When that load fails we (a) log it, (b) stamp the
    connection ``broken`` with an actionable message via ``record_test_result``
    (so the next connections-list render shows a Broken badge + Reconnect
    instead of the user silently getting an agent that "can't find" the data
    source), and (c) return ``{}`` so the caller degrades gracefully rather
    than crashing the whole turn with a raw stack trace.

    Returns the credential fields dict (possibly empty). Never raises.
    """
    try:
        fields = vault.load(engine, name)
        if fields is None:
            return {}
        return fields
    except Exception:
        logger.warning(
            "Could not load credentials for %s/%s at runtime; marking the "
            "connection broken so the user is prompted to reconnect.",
            engine, name, exc_info=True,
        )
        try:
            if hasattr(vault, "record_test_result"):
                vault.record_test_result(
                    engine, name,
                    result=TEST_FAIL,
                    error="The agent could not read this connection's saved credentials.",
                )
        except Exception:
            logger.debug("Could not stamp broken health for %s/%s", engine, name, exc_info=True)
        return {}
