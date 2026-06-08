from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import APIRouter, Request, Response

from cowork.channels.plugin import ChannelPlugin
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
InboundSink = Callable[[str, Any], Awaitable[None]]
Scheduler = Callable[[Coroutine[Any, Any, None]], None]


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
) -> APIRouter:
    """Build an APIRouter exposing every webhook a plugin declares.
    """
    router = APIRouter()
    for webhook in plugin.webhooks:
        _add_webhook_route(
            router, plugin.channel_type, webhook.path, webhook.name, list(webhook.methods),
            resolver=resolver, sink=sink, scheduler=scheduler,
        )
    return router


def _add_webhook_route(
    router: APIRouter,
    channel_type: str,
    path: str,
    route_name: str | None,
    methods: list[str],
    *,
    resolver: BridgeResolver,
    sink: InboundSink,
    scheduler: Scheduler,
) -> None:
    async def handler(request: Request) -> Response:
        bridge = resolver(channel_type)
        if bridge is None:
            return Response(status_code=204)

        body = await request.body()
        headers = {k.lower(): v for k, v in request.headers.items()}
        query = dict(request.query_params)

        handshake = bridge.try_handshake(
            method=request.method, body=body, headers=headers, query=query,
        )
        if handshake.handled:
            return Response(
                content=handshake.response_body,
                media_type=handshake.content_type,
                status_code=handshake.status_code,
            )

        try:
            bridge.verify_signature(body=body, headers=headers)
        except SignatureError:
            log.warning("channel %s webhook signature verification failed", channel_type)
            return Response("invalid signature", status_code=401)

        try:
            events = await bridge.parse_inbound(body=body, headers=headers, route_name=route_name)
        except Exception:
            log.exception("channel %s parse_inbound failed", channel_type)
            return Response("could not parse webhook payload", status_code=400)

        _intake_events(channel_type, bridge, events, sink=sink, scheduler=scheduler)
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


def _intake_events(
    channel_type: str,
    bridge: WebhookBridge,
    events: list[Any],
    *,
    sink: InboundSink,
    scheduler: Scheduler,
) -> None:
    """De-dup, record, and schedule each parsed event. Runs in request scope so
    the ACK is sent only after duplicates are filtered and events recorded; the
    actual sink call happens in the background."""
    session = get_open_session()
    try:
        channel_log = ChannelEventService(session)
        for event in events:
            key = bridge.dedupe_key(event)
            if channel_log.is_duplicate_inbound(channel_type, key):
                log.info("channel %s dropping duplicate inbound key=%s", channel_type, key)
                continue
            event_id = channel_log.record_inbound(channel_type, dedupe_key=key, external_message_id=key)
            if event_id is None:
                log.info("channel %s dropping duplicate inbound (insert race) key=%s", channel_type, key)
                continue
            scheduler(_process_event(channel_type, event, event_id, sink))
    finally:
        session.close()


async def _process_event(channel_type: str, event: Any, event_id: Any, sink: InboundSink) -> None:
    """Route one event to the sink and record the outcome. Opens its own session
    since it runs after the request's session is closed."""
    session = get_open_session()
    try:
        channel_log = ChannelEventService(session)
        try:
            await sink(channel_type, event)
            channel_log.set_status(event_id, "routed")
        except Exception as exc:
            channel_log.set_status(event_id, "failed", error=type(exc).__name__)
            log.exception("channel %s inbound sink failed", channel_type)
    finally:
        session.close()
