"""Wire shape for the desktop's usage read: free tokens, plan, balance, auto top up."""

from typing import Literal, Optional, TypeGuard, get_args

from cowork.schemas.base import CamelResponse

# The plan values, spelled once. Every other module that checks one imports the
# set derived from it below, so a new tier or window is one edit here. Keep this
# module free of heavy imports: the services and the turn-failure mapping all
# import it, and a heavy one would make a cycle.
PaidPlanTier = Literal["plus", "pro"]
PlanTier = Literal["free", PaidPlanTier]
PlanStatus = Literal["active", "past_due", "canceled"]
PlanWindowName = Literal["5h", "7d"]

PLAN_TIERS: frozenset[str] = frozenset(get_args(PlanTier))
# The tiers a model can require: the ones above Free.
PAID_PLAN_TIERS: frozenset[str] = frozenset(get_args(PaidPlanTier))
PLAN_STATUSES: frozenset[str] = frozenset(get_args(PlanStatus))
PLAN_WINDOWS: frozenset[str] = frozenset(get_args(PlanWindowName))


def is_one_of(value: object, allowed: frozenset[str]) -> TypeGuard[str]:
    """True when ``value`` is a string in ``allowed``.

    The type is checked first. These values come from untrusted JSON, and a
    list or an object raises ``TypeError`` (unhashable) in a bare ``in`` test.
    Every check against the sets above goes through here, so no caller can
    leave the type check out.
    """
    return isinstance(value, str) and value in allowed


class MindsHubFreeTokens(CamelResponse):
    """The free MindsHub Air allowance, as a proportion.

    Auth does not publish the size of the allowance, so there is no token count
    to relay. ``percentRemaining`` is the figure to read.

    ``limit``/``used``/``remaining`` stay for desktop builds that predate
    ``percentRemaining``. They carry the same proportion out of a ``limit`` of
    100, so a client dividing ``remaining`` by ``limit`` still gets the right
    ratio. ``limit`` -1 still means uncapped and 0 still means no grant, so the
    branches those builds already take are unchanged. They are floats because a
    caller with 0.4% left has not run out, and rounding that to zero would read
    as exhausted.
    """

    # None when the allowance is uncapped, so there is nothing to count down.
    percent_remaining: Optional[float] = None
    limit: float = 0
    used: float = 0
    remaining: float = 0
    # When the allowance refreshes, as auth's opaque ISO string. Formatted on
    # the client, which is the only side that knows the viewer's timezone.
    # None when auth sent none, and when there is no allowance to refresh
    # (``limit`` 0, the organization has no free grant).
    resets_at: Optional[str] = None


class MindsHubBalance(CamelResponse):
    """The paid wallet."""

    usd: float = 0.0
    can_consume: bool = True
    has_topped_up: bool = False
    # Auth decides what "low" means; this is its verdict: '' | 'low' | 'depleted'.
    alert: str = ""


class MindsHubAutoTopUp(CamelResponse):
    enabled: bool = False
    threshold_usd: Optional[float] = None
    # Auth tops the balance up TO this amount, it does not add this amount.
    recharge_to_usd: Optional[float] = None
    # 'ok' | 'cap_reached' | 'pending_action' | 'payment_failed'
    status: str = "ok"


class MindsHubCreditSpend(CamelResponse):
    """Paid credit spent in the current usage period, as the console shows it."""

    usd: float = 0.0
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class MindsHubPlanWindow(CamelResponse):
    """One rolling allowance window of a paid plan, as a proportion (ENG-4186).

    Never dollars: auth publishes how much of the window is used, not its size.
    """

    # "5h" or "7d".
    window: PlanWindowName
    # An integer 0-100, as auth sends it.
    percent_used: int
    # When the window frees up, as auth's opaque ISO string. None when nothing
    # in the window has been counted yet.
    resets_at: Optional[str] = None


class MindsHubPlan(CamelResponse):
    """The organization's subscription tier and its allowance windows (ENG-4186).

    Free orgs get ``tier`` "free" and no windows; their allowance stays in
    ``freeTokens``.
    """

    tier: PlanTier
    status: PlanStatus
    renews_at: Optional[str] = None
    cancel_at_period_end: bool = False
    windows: list[MindsHubPlanWindow] = []


class MindsHubUsageView(CamelResponse):
    """Everything the usage surfaces need, in one round trip.

    ``reachable`` false means auth could not be asked at all. Each block is
    None when its own read failed, so a wallet the caller may not see (a
    starter-tier org) still leaves the free-token block usable.
    """

    reachable: bool = False
    is_billing_owner: bool = False
    free_tokens: Optional[MindsHubFreeTokens] = None
    # None when auth sent no plan block (an auth that predates subscriptions,
    # or the subscriptions gate is off), or sent one that does not parse.
    plan: Optional[MindsHubPlan] = None
    balance: Optional[MindsHubBalance] = None
    auto_top_up: Optional[MindsHubAutoTopUp] = None
    credit_spend: Optional[MindsHubCreditSpend] = None
    # entitlements/me's ``wallet.can_consume``, which every member gets. Only the
    # client's availability re-check reads it; it never drives UI.
    wallet_can_consume: Optional[bool] = None
