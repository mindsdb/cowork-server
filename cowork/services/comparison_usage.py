"""What each side of a comparison cost, from the MindsHub gateway's usage records.

A side is one conversation, and every model call its turns make reaches the
gateway tagged with that conversation's id (``Langfuse-Session-Id``), verifier
and delegate calls included. The gateway answers with those calls, their tokens
and a list-price estimate (``GET /v1/usage/sessions/{id}``); this module files
each call under the side's turn it belongs to.

Same rules as ``hub_usage``: the bearer is the caller's own, and the host is the
operator's (``default_turn_minds_api_host``), never the tenant-settable
``minds_url``, so the credential only ever goes to this deployment's gateway.
Every failure reads as "no usage for this side", so a gateway without the route,
a BYOK model that never touched the gateway, or an outage hides the figure
rather than failing the comparison.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

import httpx

from cowork.schemas.comparisons import ComparisonUsageResponse, SideUsage, TurnUsage

logger = logging.getLogger(__name__)

# Total budget per gateway read; the two sides are read concurrently.
_TIMEOUT_S = 5.0

_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens")


def _usage_url(conversation_id: str) -> str:
    from cowork.common.settings.app_settings import default_turn_minds_api_host

    return f"{default_turn_minds_api_host().rstrip('/')}/v1/usage/sessions/{quote(conversation_id, safe='')}"


async def _gateway_usage(conversation_id: str, bearer_token: str) -> Optional[dict]:
    """The gateway's usage payload for one conversation, or None on any failure."""

    async def _fetch() -> httpx.Response:
        # No redirects: the bearer must not follow a redirect to another host.
        async with httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT_S), follow_redirects=False) as client:
            return await client.get(_usage_url(conversation_id), headers={"Authorization": f"Bearer {bearer_token}"})

    try:
        response = await asyncio.wait_for(_fetch(), _TIMEOUT_S)
    except Exception as exc:
        logger.debug("comparison usage read failed: %s", exc)
        return None
    if response.status_code != 200:
        logger.debug("comparison usage read returned HTTP %s", response.status_code)
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("requests"), list) else None


def _as_utc(value: Any) -> Optional[datetime]:
    """An aware UTC datetime. SQLite hands timestamps back naive, in UTC."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def side_usage(turn_starts: list[datetime], payload: Optional[dict], *, turn_limit: Optional[int] = None) -> SideUsage:
    """File the gateway's calls under the side's turns.

    ``turn_starts`` are the side's user messages in order; a call belongs to the
    latest turn that started at or before it. Messages keep second precision, so
    "at" matters: a turn's first call usually lands in the same second.
    ``turn_limit`` is how many turns the comparison owns (a continued side keeps
    going as a normal task); calls from the turns after it are left out.
    """
    starts = sorted(t for t in (_as_utc(s) for s in turn_starts) if t is not None)
    if payload is None or not starts:
        return SideUsage(available=False)
    owned = starts if turn_limit is None else starts[:turn_limit]
    if not owned:
        return SideUsage(available=False)
    cutoff = starts[len(owned)] if len(owned) < len(starts) else None

    turns = [TurnUsage(turn=index + 1) for index in range(len(owned))]
    unpriced = 0
    for call in payload["requests"]:
        if not isinstance(call, dict):
            continue
        at = _as_utc(call.get("timestamp"))
        if at is None or (cutoff is not None and at >= cutoff):
            continue
        turn = turns[max(0, bisect.bisect_right(owned, at) - 1)]
        for field in _TOKEN_FIELDS:
            setattr(turn, field, getattr(turn, field) + _count(call.get(field)))
        cost = call.get("estimated_cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            turn.estimated_cost_usd = (turn.estimated_cost_usd or 0.0) + float(cost)
        else:
            turn.unpriced_calls += 1
            unpriced += 1

    priced = [t.estimated_cost_usd for t in turns if t.estimated_cost_usd is not None]
    return SideUsage(
        available=True,
        turns=turns,
        input_tokens=sum(t.input_tokens for t in turns),
        output_tokens=sum(t.output_tokens for t in turns),
        cached_input_tokens=sum(t.cached_input_tokens for t in turns),
        cache_write_tokens=sum(t.cache_write_tokens for t in turns),
        estimated_cost_usd=sum(priced) if priced else None,
        unpriced_calls=unpriced,
        truncated=bool(payload.get("truncated")),
    )


async def comparison_usage(
    sides: list[tuple[str, str, list[datetime], Optional[int]]], *, bearer_token: str
) -> ComparisonUsageResponse:
    """Usage for each side: ``(label, conversation_id, turn_starts, turn_limit)``."""
    if not bearer_token:
        return ComparisonUsageResponse(sides={label: SideUsage(available=False) for label, *_ in sides})
    payloads = await asyncio.gather(*(_gateway_usage(conversation_id, bearer_token) for _, conversation_id, _, _ in sides))
    return ComparisonUsageResponse(
        sides={
            label: side_usage(starts, payload, turn_limit=limit)
            for (label, _, starts, limit), payload in zip(sides, payloads)
        }
    )
