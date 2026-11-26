"""MindsHub usage, read on behalf of the caller.

Three MindsHub reads behind one route: ``GET /v1/entitlements/me/`` for the free
Air allowance and the subscription plan, ``GET /v1/wallet/`` for the paid balance and auto top up, and
``GET /v1/usage/summary/`` for what the period's credit spend adds up to (the
same figure the console's "Credit spend this period" shows).
The desktop shows them above the composer and in Settings so a person sees
"running low" before a turn fails on it.

Same rules as ``mindshub_workspaces``: the sidecar makes the call because
MindsHub's ingress does not allow Cowork origins, the bearer is the caller's own, and the
host is the operator's (``default_minds_auth_host``), never the tenant-settable
``minds_url``. Transport and cache come from ``mindshub_client``.

The cache is per credential, not per org or per user id: the allowance and
``is_billing_owner`` are the caller's own, in MindsHub mode one process serves
everyone in the org, and on a desktop install the scope carries no user at all
(``LOCAL_SCOPE``), so only the bearer tells one signed-in account from the next.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Optional

from cowork.schemas.mindshub_usage import (
    PLAN_STATUSES,
    PLAN_TIERS,
    PLAN_WINDOWS,
    MindsHubAutoTopUp,
    MindsHubBalance,
    MindsHubCreditSpend,
    MindsHubFreeTokens,
    MindsHubPlan,
    MindsHubPlanWindow,
    MindsHubUsageView,
    is_one_of,
)
from cowork.services.mindshub_client import (
    ENTITLEMENTS_PATH,
    CallerCacheKey,
    CallerTtlCache,
    get_mindshub_json,
)

logger = logging.getLogger(__name__)

WALLET_PATH = "/wallet/"
# No ``group_by``: auth collapses the breakdown into one totals-only bucket, and
# ``totals`` is summed over every row before paging either way. Only ``totals``,
# ``range`` and ``meta`` are read here, so the console's per-model query would
# fetch up to 200 rows to discard.
USAGE_SUMMARY_PATH = "/usage/summary/"

# The allowance is reported out of this, because auth publishes a proportion
# and not a size. Desktop builds that predate ``percentRemaining`` divide by it.
_PROPORTION_LIMIT = 100.0

# Short on purpose. A top up made in the console should show up in the desktop
# within a poll or two, and a failed read should be retried soon. Per caller
# (CallerCacheKey): the allowance and ``is_billing_owner`` are the caller's own.
_TTL_OK = 30.0
_TTL_FAIL = 15.0
_cache: CallerTtlCache[MindsHubUsageView] = CallerTtlCache(ttl_ok_s=_TTL_OK, ttl_fail_s=_TTL_FAIL)


def _usd(value: Any) -> Optional[float]:
    """Auth serialises money as decimal strings ("12.40"). None when unreadable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_free_tokens(payload: Any) -> Optional[MindsHubFreeTokens]:
    """Read the allowance as a proportion.

    ``included_percent_remaining`` is the canonical field and the only one read.
    The deprecated ``included_tokens`` object is deliberately ignored: it used
    to carry the allowance size, auth has withdrawn that, and dividing by a
    withdrawn limit reads as 0% used and draws a full bar for an account that
    has nothing left.

    The three states older desktop builds branch on are preserved in ``limit``:
    -1 uncapped, 0 no grant, 100 a live allowance.
    """
    if not isinstance(payload, dict):
        return None

    resets_at = payload.get("next_refresh_at") if isinstance(payload.get("next_refresh_at"), str) else None

    # A server that never sends the canonical field predates it, and there is no
    # honest allowance to report: the absolute it would have sent instead has
    # been withdrawn. Reporting nothing beats inventing a proportion, and beats
    # guessing "uncapped", which would tell a caller with an empty wallet that
    # Air can carry the task while every turn fails.
    if "included_percent_remaining" not in payload:
        return None

    # An explicit denial is the only thing that means "no grant"; a server that
    # does not send the flag has not said so. No refill time either: auth still
    # sends `next_refresh_at` for this organization, but it has no allowance to
    # refill, so relaying the instant would promise tokens that never arrive.
    if payload.get("free_grant_eligible") is False:
        return MindsHubFreeTokens(percent_remaining=0.0, limit=0, used=0, remaining=0, resets_at=None)

    percent = payload.get("included_percent_remaining")
    if percent is None:
        # Uncapped: nothing to count down, and nothing for a bar to show.
        return MindsHubFreeTokens(percent_remaining=None, limit=-1, used=0, remaining=-1, resets_at=resets_at)
    # Fail closed on anything that is not a real number. ``NaN`` would survive
    # the clamp below as 100.0, because every comparison against it is false,
    # and hand the caller a full allowance on a malformed body. A JSON integer
    # too large for a float raises rather than clamping, which would take the
    # whole usage route down with it.
    if not isinstance(percent, (int, float)) or isinstance(percent, bool):
        return None
    try:
        percent = float(percent)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(percent):
        return None

    percent = max(0.0, min(100.0, percent))
    return MindsHubFreeTokens(
        percent_remaining=percent,
        limit=_PROPORTION_LIMIT,
        used=_PROPORTION_LIMIT - percent,
        remaining=percent,
        resets_at=resets_at,
    )


def _optional_str(value: Any) -> tuple[bool, Optional[str]]:
    """``(ok, value)`` for a field that must be a string or null."""
    if value is None or isinstance(value, str):
        return True, value
    return False, None


def _percent_used(value: Any) -> Optional[int]:
    """A window's ``percent_used`` as an int clamped to 0-100, or None when it is not a real number.

    Same rules as the free allowance's percentage: a bool, NaN, an infinity or
    an integer too large for a float would otherwise slip through as a full or
    an empty window.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(round(max(0.0, min(100.0, number))))


def _parse_plan(payload: Any) -> Optional[MindsHubPlan]:
    """Read entitlements/me's ``plan`` block (ENG-4186), failing closed.

    Absent means an auth that predates subscriptions, or one whose
    subscriptions gate is off, and the clients then render no plan surface.
    Anything malformed reports no plan rather than a guessed one, as
    ``_parse_free_tokens`` does: a percentage misread as 0 would draw an empty
    meter for a window that is spent, and an unknown tier or status would make
    the client claim a plan the person does not have.

    A window name this server does not know is skipped rather than failing the
    whole block, so a newer auth adding a window does not hide the two known
    ones. A known window that does not parse fails the block.
    """
    if not isinstance(payload, dict):
        return None
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        return None

    tier = plan.get("tier")
    status = plan.get("status")
    if not is_one_of(tier, PLAN_TIERS) or not is_one_of(status, PLAN_STATUSES):
        return None
    ok, renews_at = _optional_str(plan.get("renews_at"))
    if not ok:
        return None
    cancel_at_period_end = plan.get("cancel_at_period_end", False)
    if not isinstance(cancel_at_period_end, bool):
        return None

    raw_windows = plan.get("windows")
    if raw_windows is None:
        raw_windows = []
    if not isinstance(raw_windows, list):
        return None
    windows: list[MindsHubPlanWindow] = []
    seen: set[str] = set()
    for raw in raw_windows:
        if not isinstance(raw, dict):
            return None
        name = raw.get("window")
        if not is_one_of(name, PLAN_WINDOWS):
            continue
        if name in seen:
            return None
        seen.add(name)
        percent = _percent_used(raw.get("percent_used"))
        if percent is None:
            return None
        ok, resets_at = _optional_str(raw.get("resets_at"))
        if not ok:
            return None
        windows.append(MindsHubPlanWindow(window=name, percent_used=percent, resets_at=resets_at))

    return MindsHubPlan(
        tier=tier,
        status=status,
        renews_at=renews_at,
        cancel_at_period_end=cancel_at_period_end,
        windows=windows,
    )


def _parse_balance(payload: Any) -> Optional[MindsHubBalance]:
    if not isinstance(payload, dict):
        return None
    usd = _usd(payload.get("balance_usd"))
    if usd is None:
        return None
    return MindsHubBalance(
        usd=usd,
        can_consume=bool(payload.get("can_consume", True)),
        has_topped_up=bool(payload.get("has_topped_up", False)),
        alert=str(payload.get("low_balance_alert") or ""),
    )


def _parse_auto_top_up(payload: Any) -> Optional[MindsHubAutoTopUp]:
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
    return MindsHubAutoTopUp(
        enabled=bool(auto.get("enabled", False)),
        threshold_usd=_usd(auto.get("threshold_usd")),
        recharge_to_usd=_usd(auto.get("recharge_to_usd")),
        status=str(status),
    )


def _parse_credit_spend(summary: Any, wallet: Any) -> Optional[MindsHubCreditSpend]:
    """The period's paid spend.

    Prefer the usage summary, the console's own source, but only when its
    ``meta.cost_source`` is ``metronome_invoice`` (or absent, on an auth that
    predates the field). Anything else means the cost is not known yet, and
    showing $0.00 would be wrong. Fall back to the wallet's ``credit_spend``
    block, which older auth versions serve, and give up rather than guess.

    ``totals.billed_usd`` is the invoice, set only for the organization's own
    view of a cycle, and heads the console there; elsewhere it is null and the
    stamped ``cost.total_usd`` is the figure.
    """
    if isinstance(summary, dict):
        meta = summary.get("meta") if isinstance(summary.get("meta"), dict) else {}
        totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
        cost = totals.get("cost") if isinstance(totals.get("cost"), dict) else {}
        rng = summary.get("range") if isinstance(summary.get("range"), dict) else {}
        usd = _usd(totals.get("billed_usd"))
        if usd is None:
            usd = _usd(cost.get("total_usd"))
        source = meta.get("cost_source")
        if usd is not None and source in (None, "metronome_invoice"):
            return MindsHubCreditSpend(
                usd=usd,
                period_start=rng.get("start") if isinstance(rng.get("start"), str) else None,
                period_end=rng.get("end") if isinstance(rng.get("end"), str) else None,
            )
    if isinstance(wallet, dict) and isinstance(wallet.get("credit_spend"), dict):
        spend = wallet["credit_spend"]
        usd = _usd(spend.get("amount_usd"))
        if usd is not None:
            return MindsHubCreditSpend(
                usd=usd,
                period_start=spend.get("period_start") if isinstance(spend.get("period_start"), str) else None,
                period_end=spend.get("period_end") if isinstance(spend.get("period_end"), str) else None,
            )
    return None


def _wallet_can_consume(entitlements: Any) -> Optional[bool]:
    """``wallet.can_consume`` from entitlements/me, or None when it has no wallet block."""
    wallet = entitlements.get("wallet") if isinstance(entitlements, dict) else None
    if not isinstance(wallet, dict) or not isinstance(wallet.get("can_consume"), bool):
        return None
    return wallet["can_consume"]


def parse_usage(entitlements: Any, wallet: Any, summary: Any = None) -> MindsHubUsageView:
    """Turn the auth bodies into one view. Any body may be None."""
    if entitlements is None and wallet is None and summary is None:
        return MindsHubUsageView()
    return MindsHubUsageView(
        reachable=True,
        is_billing_owner=bool(entitlements.get("is_billing_owner")) if isinstance(entitlements, dict) else False,
        free_tokens=_parse_free_tokens(entitlements),
        plan=_parse_plan(entitlements),
        balance=_parse_balance(wallet),
        auto_top_up=_parse_auto_top_up(wallet),
        credit_spend=_parse_credit_spend(summary, wallet),
        wallet_can_consume=_wallet_can_consume(entitlements),
    )


async def fetch_mindshub_usage(*, bearer_token: str, org_id: str, user_id: str) -> MindsHubUsageView:
    """The caller's free allowance and their organization's wallet.

    Unreachable returns ``reachable=False`` rather than raising: the surfaces
    render nothing and keep whatever they last had.
    """
    if not bearer_token:
        return MindsHubUsageView()

    key = CallerCacheKey.for_caller(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    cached = _cache.fresh(key)
    if cached is not None:
        return cached

    # get_mindshub_json never raises by contract; return_exceptions keeps one
    # surprise from taking the other two reads down with it.
    results = await asyncio.gather(
        get_mindshub_json(ENTITLEMENTS_PATH, bearer_token),
        get_mindshub_json(WALLET_PATH, bearer_token),
        get_mindshub_json(USAGE_SUMMARY_PATH, bearer_token),
        return_exceptions=True,
    )
    entitlements, wallet, summary = (None if isinstance(r, BaseException) else r for r in results)
    view = parse_usage(entitlements, wallet, summary)
    _cache.put(key, view)
    return view


def reset_cache_for_tests() -> None:
    _cache.clear()
