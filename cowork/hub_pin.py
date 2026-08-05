"""Tell the hub when this instance has messaging channels connected (ENG-1003).

Hosted instances are stopped by an hourly hibernation sweep once
``last_activity`` is older than the idle timeout (48h). Nothing bumps that
timestamp for inbound webhook traffic — the Cloudflare worker proxies the
webhook straight to origin and never touches the hub's registry — so a
WhatsApp/Slack/Discord instance looks idle no matter how many messages it
handles, and the sweep also deletes its DNS record, so the URL the provider has
on file stops resolving. Telegram is no different: its outbound long-poll needs
the process alive just the same.

So the instance reports its own channel state, and the hibernation lambda skips
an instance whose report says "channels connected" and is recent.

**Level-triggered on purpose.** Each report carries current truth rather than
announcing a change, so this module is the single writer of that state. The
alternative — set-on-connect / clear-on-disconnect — has five-plus call sites in
the channels API, and missing any *clear* pins the instance always-on forever
(the ENG-597 lesson: a state-gated fix is only as good as its writer
inventory). A missed report here just corrects itself on the next tick, and the
hub expires a stale report rather than trusting it indefinitely.

Desktop and local dev never call out: with no hub environment the loop returns
immediately, so this is purely additive for every non-hosted platform.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# How often the instance re-asserts its channel state. Short relative to the
# hub's expiry window (so several misses are survivable) and to the 48h idle
# timeout it protects against.
REPORT_INTERVAL_SECONDS = float(os.environ.get("COWORK_HUB_PIN_INTERVAL", "900"))

# The instance identifies itself as this hub agent. Hardcoded rather than
# derived: this server *is* the cowork agent when it runs hosted.
AGENT = "cowork"

_HTTP_TIMEOUT_SECONDS = 10

_task: asyncio.Task | None = None
# Set when a channel changes so the next report goes out immediately instead of
# waiting out the interval. Purely an accelerator: the loop always recomputes
# the truth, so a missed nudge costs at most one interval and can never write a
# wrong value.
_nudge: asyncio.Event | None = None


def hub_endpoint() -> str | None:
    """The hub's channel-state URL, or None when not running on the hub.

    ``COWORK_HUB_API_URL`` is set by the snapshot's ``docker run`` (see
    ``anton_services/snapshots/cowork/setup.sh``). Its absence is the signal
    that this is a desktop or local process, which must never phone home.
    """
    base = (os.environ.get("COWORK_HUB_API_URL") or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/instance/channel-state"


def _instance_key() -> str | None:
    """The per-instance API key minted at provision time.

    Read from the environment rather than from user settings on purpose: this is
    the instance's own identity, and it must keep working after a user changes
    their model credentials in Settings. The hub derives the row to write from
    this key, so a report can only ever affect this instance.
    """
    return (os.environ.get("ANTON_MINDS_API_KEY") or "").strip() or None


def channels_connected() -> bool:
    """Whether any messaging channel currently has its credentials configured.

    "Configured" — not "adapter running" — is the right signal: it is what the
    UI calls connected, it is what makes a provider's webhook URL live, and it
    does not flap while the process restarts (which would hand the hub a
    momentary "no channels" during exactly the window that matters).

    Telegram counts. It ingests by outbound long-poll so it never needed the
    edge to be reachable, but it does need this process alive.
    """
    from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.services.channels import ChannelConfigService

    session = get_open_session()
    try:
        status = ChannelConfigService(ScopedSession(session, SYSTEM_SCOPE)).status()
        return any(item.configured for item in status.channels)
    finally:
        session.close()


def _post(endpoint: str, key: str, channel_active: bool) -> None:
    payload = json.dumps({"agent": AGENT, "channel_active": channel_active}).encode()
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "cowork-server/hub-pin",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
        response.read()


async def report_once() -> bool | None:
    """Send one report. Returns the reported value, or None if it didn't send.

    Never raises: a hub that is unreachable, slow or unhappy must not take the
    instance down with it. The consequence of a failed report is bounded — the
    hub expires the previous one, so the instance eventually hibernates as it
    does today rather than being pinned on by a stale flag.
    """
    endpoint = hub_endpoint()
    key = _instance_key()
    if not endpoint or not key:
        return None
    try:
        active = await asyncio.to_thread(channels_connected)
        await asyncio.to_thread(_post, endpoint, key, active)
    except urllib.error.HTTPError as exc:
        logger.warning("hub-pin: hub rejected the channel report (%s)", exc.code)
        return None
    except Exception:
        logger.warning("hub-pin: could not report channel state", exc_info=True)
        return None
    logger.info("hub-pin: reported channel_active=%s", active)
    return active


async def _loop() -> None:
    while True:
        await report_once()
        try:
            # Wake early on a channel change; otherwise report on the interval.
            await asyncio.wait_for(_nudge.wait(), timeout=REPORT_INTERVAL_SECONDS)
        except TimeoutError:
            pass
        else:
            _nudge.clear()


def nudge() -> None:
    """Ask for an out-of-band report because a channel just changed.

    Safe to call from anywhere, including when the loop isn't running (desktop)
    — it is a hint, never the source of truth.
    """
    if _nudge is not None and not _nudge.is_set():
        _nudge.set()


def start() -> None:
    """Start the reporting loop. No-op off the hub, and idempotent."""
    global _task, _nudge
    if hub_endpoint() is None or _instance_key() is None:
        logger.debug("hub-pin: no hub environment; channel reporting disabled")
        return
    if _task is not None and not _task.done():
        return
    _nudge = asyncio.Event()
    _task = asyncio.create_task(_loop())
    logger.info(
        "hub-pin: reporting channel state every %ss", int(REPORT_INTERVAL_SECONDS)
    )


async def stop() -> None:
    """Cancel the reporting loop (process shutdown)."""
    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await _task
    except (asyncio.CancelledError, Exception):
        pass
    _task = None
