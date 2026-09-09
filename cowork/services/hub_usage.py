"""MindsHub usage, read on behalf of the caller.

Three auth reads behind one route: ``GET /v1/entitlements/me/`` for the free
monthly allowance, ``GET /v1/wallet/`` for the paid balance and auto top up, and
``GET /v1/usage/summary/`` for what the period's credit spend adds up to (the
same figure the console's "Credit spend this period" shows).
The desktop shows them above the composer and in Settings so a person sees
"running low" before a turn fails on it.

Same rules as ``hub_workspaces``: the sidecar makes the call because auth's
ingress does not allow Cowork origins, the bearer is the caller's own, and the
host is the operator's (``default_minds_auth_host``), never the tenant-settable
``minds_url``. Transport, cache key and sweep come from ``hub_workspaces``.

The cache is per credential, not per org or per user id: the allowance and
``is_billing_owner`` are the caller's own, in org mode one process serves
everyone in the org, and on a desktop install the scope carries no user at all
(``LOCAL_SCOPE``), so only the bearer tells one signed-in account from the next.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from cowork.schemas.hub_usage import (
    HubAutoTopUp,
    HubBalance,
    HubCreditSpend,
    HubFreeTokens,
    HubUsageView,
)
from cowork.services.hub_workspaces import cache_key, get_auth_json, sweep_cache

logger = logging.getLogger(__name__)

ENTITLEMENTS_PATH = "/entitlements/me/"
WALLET_PATH = "/wallet/"
# No ``group_by``: auth collapses the breakdown into one totals-only bucket, and
# ``totals`` is summed over every row before paging either way. Only ``totals``,
# ``range`` and ``meta`` are read here, so the console's per-model query would
# fetch up to 200 rows to discard.
USAGE_SUMMARY_PATH = "/usage/summary/"

# Short on purpose. A top up made in the console should show up in the desktop
# within a poll or two, and a failed read should be retried soon.
_TTL_OK = 30.0
_TTL_FAIL = 15.0
# Derived, not restated, so raising a TTL cannot start evicting live entries.
_MAX_TTL_S = max(_TTL_OK, _TTL_FAIL)

# Keyed by (auth host, org, user, credential digest): the workspace caches' key,
# for the same reason. `user_id` is empty outside org mode, so without the bearer
# in the key every caller on a desktop install would share one entry, and a
# sign-out and sign-in as another account would be served the first account's
# balance and owner flag for the rest of the TTL.
_cache: dict[tuple[str, str, str, str], tuple[float, HubUsageView]] = {}


def _usd(value: Any) -> Optional[float]:
    """Auth serialises money as decimal strings ("12.40"). None when unreadable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_free_tokens(payload: Any) -> Optional[HubFreeTokens]:
    if not isinstance(payload, dict):
        return None
    included = payload.get("included_tokens")
    if not isinstance(included, dict):
        return None
    # Auth sends ``limit: null`` (and ``remaining: null``) for an unlimited
    # allowance; the desktop reads -1 as unlimited and 0 as no grant.
    limit = -1 if included.get("limit") is None else _int(included["limit"])
    used = _int(included.get("used"))
    if limit < 0:
        remaining = -1  # uncapped: nothing to count down
    elif included.get("remaining") is not None:
        remaining = _int(included["remaining"])
    else:
        remaining = max(0, limit - used)
    return HubFreeTokens(
        limit=limit,
        used=used,
        remaining=remaining,
        resets_at=payload.get("next_refresh_at") if isinstance(payload.get("next_refresh_at"), str) else None,
    )


def _parse_balance(payload: Any) -> Optional[HubBalance]:
    if not isinstance(payload, dict):
        return None
    usd = _usd(payload.get("balance_usd"))
    if usd is None:
        return None
    return HubBalance(
        usd=usd,
        can_consume=bool(payload.get("can_consume", True)),
        has_topped_up=bool(payload.get("has_topped_up", False)),
        alert=str(payload.get("low_balance_alert") or ""),
    )


def _parse_auto_top_up(payload: Any) -> Optional[HubAutoTopUp]:
    if not isinstance(payload, dict):
        return None
    auto = payload.get("auto_recharge")
    if not isinstance(auto, dict):
        return None
    # Same worst-first fallback the console applies when auth predates `status`.
    status = auto.get("status") or (
        "payment_failed" if auto.get("last_charge_failed")
        else "pending_action" if auto.get("pending_action")
        else "cap_reached" if auto.get("cap_reached")
        else "ok"
    )
    return HubAutoTopUp(
        enabled=bool(auto.get("enabled", False)),
        threshold_usd=_usd(auto.get("threshold_usd")),
        recharge_to_usd=_usd(auto.get("recharge_to_usd")),
        status=str(status),
    )


def _parse_credit_spend(summary: Any, wallet: Any) -> Optional[HubCreditSpend]:
    """The period's paid spend.

    Prefer the usage summary, the console's own source, but only when its
    ``meta.cost_source`` is ``metronome_invoice`` (or absent, on an auth that
    predates the field). Anything else means the cost is not known yet, and
    showing $0.00 would be wrong. Fall back to the wallet's ``credit_spend``
    block, which older auth versions serve, and give up rather than guess.
    """
    if isinstance(summary, dict):
        meta = summary.get("meta") if isinstance(summary.get("meta"), dict) else {}
        totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
        cost = totals.get("cost") if isinstance(totals.get("cost"), dict) else {}
        rng = summary.get("range") if isinstance(summary.get("range"), dict) else {}
        usd = _usd(cost.get("total_usd"))
        source = meta.get("cost_source")
        if usd is not None and source in (None, "metronome_invoice"):
            return HubCreditSpend(
                usd=usd,
                period_start=rng.get("start") if isinstance(rng.get("start"), str) else None,
                period_end=rng.get("end") if isinstance(rng.get("end"), str) else None,
            )
    if isinstance(wallet, dict) and isinstance(wallet.get("credit_spend"), dict):
        spend = wallet["credit_spend"]
        usd = _usd(spend.get("amount_usd"))
        if usd is not None:
            return HubCreditSpend(
                usd=usd,
                period_start=spend.get("period_start") if isinstance(spend.get("period_start"), str) else None,
                period_end=spend.get("period_end") if isinstance(spend.get("period_end"), str) else None,
            )
    return None


def parse_usage(entitlements: Any, wallet: Any, summary: Any = None) -> HubUsageView:
    """Turn the auth bodies into one view. Any body may be None."""
    if entitlements is None and wallet is None and summary is None:
        return HubUsageView()
    return HubUsageView(
        reachable=True,
        is_billing_owner=bool(entitlements.get("is_billing_owner")) if isinstance(entitlements, dict) else False,
        free_tokens=_parse_free_tokens(entitlements),
        balance=_parse_balance(wallet),
        auto_top_up=_parse_auto_top_up(wallet),
        credit_spend=_parse_credit_spend(summary, wallet),
    )


async def fetch_hub_usage(*, bearer_token: str, org_id: str, user_id: str) -> HubUsageView:
    """The caller's free allowance and their organization's wallet.

    Unreachable returns ``reachable=False`` rather than raising: the surfaces
    render nothing and keep whatever they last had.
    """
    if not bearer_token:
        return HubUsageView()

    key = cache_key(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    cached = _cache.get(key)
    if cached:
        stamped, value = cached
        if (time.monotonic() - stamped) < (_TTL_OK if value.reachable else _TTL_FAIL):
            return value

    # get_auth_json never raises by contract; return_exceptions keeps one
    # surprise from taking the other two reads down with it.
    results = await asyncio.gather(
        get_auth_json(ENTITLEMENTS_PATH, bearer_token),
        get_auth_json(WALLET_PATH, bearer_token),
        get_auth_json(USAGE_SUMMARY_PATH, bearer_token),
        return_exceptions=True,
    )
    entitlements, wallet, summary = (None if isinstance(r, BaseException) else r for r in results)
    view = parse_usage(entitlements, wallet, summary)
    sweep_cache(_cache, max_ttl_s=_MAX_TTL_S)
    _cache[key] = (time.monotonic(), view)
    return view


def reset_cache_for_tests() -> None:
    _cache.clear()
