"""Whether an inbound message came from a shared channel, not a DM.

Only a conversation from a shared channel is ever marked, and so readable to
org admins; a direct message to the bot, one-to-one or a group DM, is readable
by nobody in the app. Each adapter places its own events, from its platform's
own fields, when it builds them. This is separate from `is_group`, which
drives mention handling and differs from it for Slack group DMs.

The flag rides on the event like the adapters' `_dedupe_key`, since
`InboundMessage` lives in anton. An event no adapter placed counts as not
shared, so a new or changed adapter fails closed.
"""
from __future__ import annotations

from typing import Any

_ATTRIBUTE = "_shared"


def mark_shared(event: Any, shared: bool) -> Any:
    """Record on `event` whether it came from a shared channel; returns `event`."""
    setattr(event, _ATTRIBUTE, bool(shared))
    return event


def is_shared(event: Any) -> bool:
    """True only when an adapter placed `event` as coming from a shared channel."""
    return getattr(event, _ATTRIBUTE, False) is True
