"""Wire shape for the desktop's usage read: free tokens, balance, auto top up."""

from typing import Optional

from cowork.schemas.base import CamelResponse


class HubFreeTokens(CamelResponse):
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
    resets_at: Optional[str] = None


class HubBalance(CamelResponse):
    """The paid wallet."""

    usd: float = 0.0
    can_consume: bool = True
    has_topped_up: bool = False
    # Auth decides what "low" means; this is its verdict: '' | 'low' | 'depleted'.
    alert: str = ""


class HubAutoTopUp(CamelResponse):
    enabled: bool = False
    threshold_usd: Optional[float] = None
    # Auth tops the balance up TO this amount, it does not add this amount.
    recharge_to_usd: Optional[float] = None
    # 'ok' | 'cap_reached' | 'pending_action' | 'payment_failed'
    status: str = "ok"


class HubCreditSpend(CamelResponse):
    """Paid credit spent in the current usage period, as the console shows it."""

    usd: float = 0.0
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class HubUsageView(CamelResponse):
    """Everything the usage surfaces need, in one round trip.

    ``reachable`` false means auth could not be asked at all. Each block is
    None when its own read failed, so a wallet the caller may not see (a
    starter-tier org) still leaves the free-token block usable.
    """

    reachable: bool = False
    is_billing_owner: bool = False
    free_tokens: Optional[HubFreeTokens] = None
    balance: Optional[HubBalance] = None
    auto_top_up: Optional[HubAutoTopUp] = None
    credit_spend: Optional[HubCreditSpend] = None
