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


def mark_shared(event: Any, shared: bool | None) -> Any:
    """Record on `event` whether it came from a shared channel: True, False,
    or None when the platform did not say (a Slack mention in a G channel,
    which may be a private channel or a group DM). Returns `event`."""
    setattr(event, _ATTRIBUTE, shared if shared is None else bool(shared))
    return event


def shared_state(event: Any) -> bool | None:
    """True or False as the adapter placed `event`, None when it could not
    tell. An event no adapter placed reads False, never None."""
    value = getattr(event, _ATTRIBUTE, False)
    if value is None:
        return None
    return value is True


def is_shared(event: Any) -> bool:
    """True only when an adapter placed `event` as coming from a shared channel."""
    return shared_state(event) is True
