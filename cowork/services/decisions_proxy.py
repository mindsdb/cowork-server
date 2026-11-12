"""The organization's own decision model, for a web turn's browser, through
this server (ENG-4138).

A web turn runs in a cloud pod beside the agent's code, so the organization's
decision model key must never reach it. Instead the turn's browser block
carries this server's route (``decisions_proxy``: its URL and the model name,
no key), and anton posts each browser decision there with the credential the
pod already holds: the turn key minted for that turn. The route
(``POST /api/v1/browse/decisions``) checks the key and forwards the decision
to the organization's stored ``decision_model_url`` with its stored key, out
through the route gate's vetted egress (``jev_route._post_decision``).

The turn key is checked against what this server recorded when it queued the
turn (``admit``): the organization and person it was minted for, under a
digest of the key, so the key itself is never stored. Only a turn whose
browser was handed the proxy is recorded, the record lives as long as the key
(extended with it, ``extend``), and it goes when the turn ends (``forget``),
when the key is revoked too. Any other credential, a person's API key or a
turn key of a turn without the proxy included, is refused. Behind the MindsHub
gateway the request also carries the identity the gateway verified, which
must name the same organization and person.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

from cowork.common.settings.app_settings import get_app_settings

logger = logging.getLogger(__name__)

#: The route's path under this server's public base URL.
PROXY_PATH = "/api/v1/browse/decisions"

#: The largest decision body the route reads. anton's browser decisions are a
#: few kilobytes (a shortlist of elements, 1500 characters of page text).
MAX_BODY_BYTES = 256 * 1024

#: How long the route waits on the organization's endpoint: under anton's own
#: wait for a browser decision (``browser_jev_timeout_s``, 4 s), so the pod
#: hears a timeout from here rather than giving up on the hop.
UPSTREAM_TIMEOUT_S = 3.5

_PREFIX = "cowork:decisions-proxy"


def _key_record(digest: str) -> str:
    return f"{_PREFIX}:key:{digest}"


def _turn_record(correlation_id: str) -> str:
    return f"{_PREFIX}:turn:{correlation_id}"


def key_digest(turn_key: str) -> str:
    return hashlib.sha256(turn_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TurnHolder:
    """Whom a recorded turn key was minted for."""

    org_id: str
    user_id: str
    correlation_id: str


def proxy_url() -> str:
    """The route's full URL as a cloud pod reaches it, or "" when this
    deployment has none: ``COWORK_BROWSER_DECISIONS_URL``, else the route under
    ``COWORK_PUBLIC_BASE_URL``. Without one a web turn's browser gets no
    proxy, and asks Jev on MindsHub as before."""
    settings = get_app_settings()
    if settings.browser_decisions_url:
        return settings.browser_decisions_url
    base = (settings.public_base_url or "").strip().rstrip("/")
    if not base.startswith("https://"):
        return ""
    return f"{base}{PROXY_PATH}"


async def admit(r, *, turn_key: str, org_id: str, user_id: str, correlation_id: str, ttl_seconds: int) -> None:
    """Record a turn key whose turn's browser asks through the proxy, for as
    long as the key lives. Raises on a Redis failure: the producer then sends
    the turn without the proxy."""
    digest = key_digest(turn_key)
    holder = json.dumps({"org_id": org_id, "user_id": user_id, "correlation_id": correlation_id})
    async with r.pipeline(transaction=True) as pipe:
        pipe.set(_key_record(digest), holder, ex=ttl_seconds)
        pipe.set(_turn_record(correlation_id), digest, ex=ttl_seconds)
        await pipe.execute()


async def extend(r, *, correlation_id: str, ttl_seconds: int) -> None:
    """Keep a recorded turn key's record as long as the key, once auth
    extended it ("Keep going"). Best effort: a lapsed record only means the
    browser's later steps go back to the model."""
    try:
        digest = await r.get(_turn_record(correlation_id))
        if digest:
            await r.expire(_key_record(digest), ttl_seconds)
            await r.expire(_turn_record(correlation_id), ttl_seconds)
    except Exception as exc:
        logger.warning(
            "decisions proxy: could not extend a turn key's record",
            extra={"event": "decisions_proxy_extend_failed", "correlation_id": correlation_id,
                   "error_type": type(exc).__name__},
        )


async def forget(r, *, correlation_id: str) -> None:
    """Drop a turn's record when the turn ends, as its key is revoked. Best
    effort: a record left behind lapses with the key's lifetime."""
    try:
        digest = await r.get(_turn_record(correlation_id))
        keys = [_turn_record(correlation_id)] + ([_key_record(digest)] if digest else [])
        await r.delete(*keys)
    except Exception as exc:
        logger.warning(
            "decisions proxy: could not drop a turn key's record; it lapses with the key",
            extra={"event": "decisions_proxy_forget_failed", "correlation_id": correlation_id,
                   "error_type": type(exc).__name__},
        )


async def holder(r, turn_key: str) -> TurnHolder | None:
    """Whom ``turn_key`` was minted for, when it is a recorded turn key that
    still lives; None for any other credential. Raises on a Redis failure."""
    if not turn_key:
        return None
    raw = await r.get(_key_record(key_digest(turn_key)))
    if not raw:
        return None
    try:
        record = json.loads(raw)
        return TurnHolder(
            org_id=str(record["org_id"]), user_id=str(record["user_id"]),
            correlation_id=str(record["correlation_id"]),
        )
    except (ValueError, KeyError, TypeError):
        return None
