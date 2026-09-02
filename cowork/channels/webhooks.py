from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import APIRouter, Request, Response

from cowork.channels.plugin import ChannelPlugin
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession, TenantScope, scope_for_org
from cowork.db.session import get_open_session
from cowork.services.channel_events import ChannelEventService

log = logging.getLogger(__name__)


@dataclass
class WebhookHandshake:
    """Outcome of :meth:`WebhookBridge.try_handshake`. When ``handled`` is true
    the route returns ``response_body`` immediately and skips verify/parse."""

    handled: bool
    response_body: str = ""
    content_type: str = "text/plain"
    status_code: int = 200


@dataclass
class WebhookAck:
    """Optional custom ACK response from a bridge's ``ack_response(events)`` hook."""
    body: str = ""
    content_type: str = "text/plain"
    status_code: int = 200


class SignatureError(Exception):
    """Raised by :meth:`WebhookBridge.verify_signature` on a bad signature."""


class _DummyBridge:
    """Minimal bridge wrapper for platform handshakes that require no live adapter."""

    def __init__(self, plugin: ChannelPlugin) -> None:
        self.plugin = plugin

    def try_handshake(
        self,
        *,
        method: str,
        body: bytes,
        headers: Mapping[str, str],
        query: Mapping[str, str],
    ) -> WebhookHandshake:
        """Delegate to plugin's own handshake method if it exists."""
        if hasattr(self.plugin, "try_handshake"):
            return self.plugin.try_handshake(body=body, headers=headers)
        return WebhookHandshake(handled=False)


class WebhookBridge(Protocol):
    """The minimum a live channel adapter exposes to the webhook route layer."""

    def try_handshake(
        self,
        *,
        method: str,
        body: bytes,
        headers: Mapping[str, str],
        query: Mapping[str, str],
    ) -> WebhookHandshake: ...

    def verify_signature(self, *, body: bytes, headers: Mapping[str, str]) -> None: ...

    async def parse_inbound(
        self,
        *,
        body: bytes,
        headers: Mapping[str, str],
        route_name: str | None,
    ) -> list[Any]: ...

    def dedupe_key(self, event: Any) -> str | None: ...


BridgeResolver = Callable[[str], "WebhookBridge | None"]
# (channel_type, routing_key) -> (bridge, org_id), or None if nothing claims
# that routing key. org_id is the org the bridge belongs to.
OrgBridgeResolver = Callable[[str, str], Awaitable["tuple[WebhookBridge, str | None] | None"]]
InboundSink = Callable[[str, Any, "str | None"], Awaitable[None]]
Scheduler = Callable[[Coroutine[Any, Any, None]], None]


async def resolve_bridge(
    *,
    channel_type: str,
    plugin: ChannelPlugin,
    body: bytes,
    headers: Mapping[str, str],
    resolver: BridgeResolver,
    org_resolver: OrgBridgeResolver | None,
) -> tuple["WebhookBridge | None", str | None]:
    """Bridge to use for this inbound request, and the org it belongs to (None
    for the plain/local resolver path). Falls back to `resolver(channel_type)`
    whenever the plugin has no extractor, the extractor finds nothing, or the
    key doesn't resolve — so non-participating plugins are untouched."""
    if plugin.extract_routing_key is not None and org_resolver is not None:
        routing_key = plugin.extract_routing_key(body, headers)
        if routing_key is not None:
            resolved = await org_resolver(channel_type, routing_key)
            if resolved is not None:
                return resolved
    return resolver(channel_type), None


_background_tasks: set[asyncio.Task[Any]] = set()


def _default_scheduler(coro: Coroutine[Any, Any, None]) -> None:
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def drain_background_tasks(*, timeout: float = 3.0) -> None:
    """Await in-flight inbound processing. Call from shutdown so a task spawned
    just before teardown isn't abandoned mid-run. Best-effort."""
    tasks = list(_background_tasks)
    if not tasks:
        return
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("%d channel inbound task(s) did not finish within %.1fs", len(tasks), timeout)


def build_channel_webhook_router(
    plugin: ChannelPlugin,
    *,
    resolver: BridgeResolver,
    sink: InboundSink,
    scheduler: Scheduler = _default_scheduler,
    org_resolver: OrgBridgeResolver | None = None,
) -> APIRouter:
    """Build an APIRouter exposing every webhook a plugin declares.
    """
    router = APIRouter()
    for webhook in plugin.webhooks:
        _add_webhook_route(
            router, plugin, webhook.path, webhook.name, list(webhook.methods),
            resolver=resolver, sink=sink, scheduler=scheduler, org_resolver=org_resolver,
        )
    return router


def _add_webhook_route(
    router: APIRouter,
    plugin: ChannelPlugin,
    path: str,
    route_name: str | None,
    methods: list[str],
    *,
    resolver: BridgeResolver,
    sink: InboundSink,
    scheduler: Scheduler,
    org_resolver: OrgBridgeResolver | None,
) -> None:
    channel_type = plugin.channel_type

    async def handler(request: Request) -> Response:
        # Read the body before resolving: a plugin with extract_routing_key
        # needs it to pick which org's bridge this inbound belongs to.
        body = await request.body()
        headers = {k.lower(): v for k, v in request.headers.items()}
        query = dict(request.query_params)

        # Handle platform handshakes (e.g. Slack url_verification) first,
        # before routing by org. Handshakes carry no org routing key and must
        # succeed in every deployment mode.
        plugin_bridge = _DummyBridge(plugin)
        handshake = plugin_bridge.try_handshake(
            method=request.method, body=body, headers=headers, query=query,
        )
        if handshake.handled:
            return Response(
                content=handshake.response_body,
                media_type=handshake.content_type,
                status_code=handshake.status_code,
            )

        bridge, org_id = await resolve_bridge(
            channel_type=channel_type, plugin=plugin, body=body, headers=headers,
            resolver=resolver, org_resolver=org_resolver,
        )
        if bridge is None:
            # Inbound arrived but no live adapter is registered: the channel
            # isn't active (setup/reload not run, or credentials incomplete).
            # Warn because this silently drops every message and is a common
            # misconfig — the 204 keeps the platform from retrying.
            log.warning(
                "channel %s: webhook hit but no live adapter registered; "
                "dropping inbound (is the channel active?)", channel_type,
            )
            return Response(status_code=204)

        log.info("channel %s: webhook received (%d bytes)", channel_type, len(body))

        try:
            bridge.verify_signature(body=body, headers=headers)
        except SignatureError as exc:
            # Surface the specific reason (e.g. "signing_secret not configured")
            # so a misconfig — like a Socket-Mode-only Slack app still receiving
            # webhook posts — isn't an opaque 401. Response stays generic.
            log.warning("channel %s webhook signature verification failed: %s", channel_type, exc)
            return Response("invalid signature", status_code=401)

        try:
            events = await bridge.parse_inbound(body=body, headers=headers, route_name=route_name)
        except Exception:
            log.exception("channel %s parse_inbound failed", channel_type)
            return Response("could not parse webhook payload", status_code=400)

        log.info("channel %s: parsed %d inbound event(s)", channel_type, len(events))
        intake_events(channel_type, bridge, events, sink=sink, scheduler=scheduler, org_id=org_id)
        return _success_ack(bridge, events)

    router.add_api_route(
        f"/{channel_type}{path}",
        handler,
        methods=methods,
        name=f"channel_{channel_type}_webhook_{route_name or 'default'}",
        include_in_schema=False,
    )


def _success_ack(bridge: WebhookBridge, events: list[Any]) -> Response:
    hook = getattr(bridge, "ack_response", None)
    if hook is not None:
        ack = hook(events)
        if ack is not None:
            return Response(
                content=ack.body, media_type=ack.content_type, status_code=ack.status_code
            )
    return Response(status_code=200)


def intake_events(
    channel_type: str,
    bridge: WebhookBridge,
    events: list[Any],
    *,
    sink: InboundSink,
    scheduler: Scheduler | None = None,
    org_id: str | None = None,
) -> None:
    """De-dup, record, and schedule each parsed event for background sink
    processing. Shared by both ingress paths — webhook routes and server-side
    polling — so dedupe and event logging behave identically regardless of how
    the event arrived. For webhooks this runs in request scope, so the ACK is
    sent only after duplicates are filtered and recorded; the sink call itself
    always happens in the background."""
    sched = scheduler or _default_scheduler
    session = get_open_session()
    try:
        # Dedupe is per installation: LOCAL_SCOPE when no org was resolved
        # (local/desktop's one installation), the resolved org's scope otherwise.
        scope = scope_for_org(org_id)
        channel_log = ChannelEventService(ScopedSession(session, scope))
        for event in events:
            key = bridge.dedupe_key(event)
            if channel_log.is_duplicate_inbound(channel_type, key):
                log.info("channel %s dropping duplicate inbound key=%s", channel_type, key)
                continue
            event_id = channel_log.record_inbound(channel_type, dedupe_key=key, external_message_id=key)
            if event_id is None:
                log.info("channel %s dropping duplicate inbound (insert race) key=%s", channel_type, key)
                continue
            log.info("channel %s: accepted inbound key=%s; dispatching to runtime", channel_type, key)
            sched(_process_event(channel_type, event, event_id, sink, org_id=org_id))
    finally:
        session.close()


async def _process_event(
    channel_type: str, event: Any, event_id: Any, sink: InboundSink, *, org_id: str | None = None
) -> None:
    """Route one event to the sink and record the outcome. Opens its own session
    since it runs after the request's session is closed."""
    session = get_open_session()
    try:
        scope = scope_for_org(org_id)
        channel_log = ChannelEventService(ScopedSession(session, scope))
        try:
            await sink(channel_type, event, org_id)
            channel_log.set_status(event_id, "routed")
        except Exception as exc:
            channel_log.set_status(event_id, "failed", error=type(exc).__name__)
            log.exception("channel %s inbound sink failed", channel_type)
    finally:
        session.close()
