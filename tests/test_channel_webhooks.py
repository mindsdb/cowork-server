"""Bridge resolution for inbound webhooks: falls back to the plain
channel_type-only resolver whenever a plugin doesn't declare a routing-key
extractor, the extractor finds nothing, or the key doesn't resolve to a known
installation — so plugins that don't participate (today: everyone but Slack)
are completely unaffected."""
from __future__ import annotations

import asyncio
import gc
import hashlib
import hmac
import json
import logging
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import select

from cowork.channels import webhooks
from cowork.channels.plugin import ChannelPlugin, CredentialSchema
from cowork.channels.registry import PluginRegistry
from cowork.channels.runtime import LiveAdapterRegistry
from cowork.channels.webhooks import build_channel_webhook_router, intake_events, resolve_bridge
from cowork.common import chat_session
from cowork.common.chat_session import close_session_scratchpads, drain_scratchpad_closes
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.db.units import run_efs, run_to_completion
from cowork.services.channel_events import ChannelEventService
from cowork.services.channels import ChannelConfigService
from tests._logging_assertions import assert_log_fields

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


class _FakeBridge:
    def dedupe_key(self, event):
        return event["key"]


async def _fake_factory(creds):
    return None


def _plugin(*, extract_routing_key=None) -> ChannelPlugin:
    return ChannelPlugin(
        channel_type="slack",
        display_name="Slack",
        factory=_fake_factory,
        credentials=CredentialSchema(),
        extract_routing_key=extract_routing_key,
    )


async def test_falls_back_when_plugin_has_no_extractor():
    plugin = _plugin()
    bridge, org_id = await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b"{}", headers={},
        resolver=lambda ct: "local-bridge", org_resolver=None,
    )
    assert bridge == "local-bridge"
    assert org_id is None


async def test_falls_back_when_extractor_finds_no_routing_key():
    plugin = _plugin(extract_routing_key=lambda body, headers: None)
    calls = []

    async def org_resolver(ct, key):
        calls.append((ct, key))
        return ("org-bridge", "org-x")

    bridge, org_id = await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b"{}", headers={},
        resolver=lambda ct: "local-bridge", org_resolver=org_resolver,
    )
    assert bridge == "local-bridge"
    assert org_id is None
    assert calls == []  # never asked — nothing to resolve


async def test_resolves_org_bridge_when_routing_key_found():
    plugin = _plugin(extract_routing_key=lambda body, headers: "T-A")

    async def org_resolver(ct, key):
        assert (ct, key) == ("slack", "T-A")
        return ("org-a-bridge", "org-a")

    bridge, org_id = await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b"{}", headers={},
        resolver=lambda ct: "local-bridge", org_resolver=org_resolver,
    )
    assert bridge == "org-a-bridge"
    assert org_id == "org-a"


async def test_falls_back_when_routing_key_does_not_resolve():
    plugin = _plugin(extract_routing_key=lambda body, headers: "T-unknown")

    async def org_resolver(ct, key):
        return None  # nobody claims this account

    bridge, org_id = await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b"{}", headers={},
        resolver=lambda ct: "local-bridge", org_resolver=org_resolver,
    )
    assert bridge == "local-bridge"
    assert org_id is None


async def test_extractor_receives_the_actual_body_and_headers():
    seen = {}

    def extract(body, headers):
        seen["body"] = body
        seen["headers"] = headers
        return None

    plugin = _plugin(extract_routing_key=extract)
    await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b'{"team_id":"T-A"}',
        headers={"x-slack-signature": "v0=abc"},
        resolver=lambda ct: "local-bridge", org_resolver=lambda ct, key: None,
    )
    assert seen["body"] == b'{"team_id":"T-A"}'
    assert seen["headers"] == {"x-slack-signature": "v0=abc"}


# --- intake_events: dedupe/event-log land in the resolved org's scope ---

def _capturing_scheduler(store):
    def sched(coro):
        store.append(coro)
    return sched


def _delete_channel_events(*dedupe_keys: str) -> None:
    # These tests hit the shared session-scoped test DB (get_open_session),
    # not an isolated engine — clean up so other tests' own counts stay valid.
    from cowork.models.channel import ChannelEvent

    session = get_open_session()
    try:
        for row in session.exec(
            select(ChannelEvent).where(ChannelEvent.dedupe_key.in_(dedupe_keys))
        ).all():
            session.delete(row)
        session.commit()
    finally:
        session.close()


async def test_intake_events_stamps_the_resolved_org(monkeypatch, org_deployment):
    captured = []
    try:
        await intake_events(
            "slack", _FakeBridge(), [{"key": "evt-a"}],
            sink=lambda ct, evt, org_id=None: None, scheduler=_capturing_scheduler(captured), org_id=ORG_A,
        )
        for coro in captured:
            await coro

        session = get_open_session()
        try:
            svc = ChannelEventService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A)))
            assert svc.is_duplicate_inbound("slack", "evt-a") is True
        finally:
            session.close()
    finally:
        _delete_channel_events("evt-a")


async def test_intake_events_isolates_dedupe_across_orgs(org_deployment):
    # The real signal isn't "is_duplicate_inbound returns True" — org B seeing
    # its OWN row and org B incorrectly seeing org A's leaked row both look
    # like that. The real proof is that org B's intake actually scheduled
    # processing instead of silently treating a same-key event as a dup.
    captured_a: list = []
    captured_b: list = []
    try:
        await intake_events(
            "slack", _FakeBridge(), [{"key": "evt-shared"}],
            sink=lambda ct, evt, org_id=None: None, scheduler=_capturing_scheduler(captured_a), org_id=ORG_A,
        )
        for coro in captured_a:
            await coro

        await intake_events(
            "slack", _FakeBridge(), [{"key": "evt-shared"}],
            sink=lambda ct, evt, org_id=None: None, scheduler=_capturing_scheduler(captured_b), org_id=ORG_B,
        )

        assert len(captured_a) == 1
        assert len(captured_b) == 1  # would be 0 if org B's write got deduped against org A's
        for coro in captured_b:
            await coro
    finally:
        _delete_channel_events("evt-shared")


async def test_a_cancel_while_an_event_is_recorded_still_schedules_it(monkeypatch, org_deployment):
    """A cancel of the intake (a credentials change, a lost lease, shutdown)
    lands while an event's record unit runs. The unit commits the row, and
    the cancel waits for the event's schedule too: a recorded event left
    unscheduled would have its redelivery dropped as a duplicate, and the
    message would never get a reply."""
    import threading

    inside = threading.Event()
    release = threading.Event()
    real_record = webhooks._record_inbound

    def slow_record(session, **kwargs):
        inside.set()
        release.wait(timeout=5)
        return real_record(session, **kwargs)

    monkeypatch.setattr(webhooks, "_record_inbound", slow_record)
    captured: list = []
    try:
        intake = asyncio.create_task(intake_events(
            "slack", _FakeBridge(), [{"key": "evt-cancelled"}],
            sink=lambda ct, evt, org_id=None: None, scheduler=_capturing_scheduler(captured), org_id=ORG_A,
        ))
        deadline = time.monotonic() + 5
        while not inside.is_set():
            assert time.monotonic() < deadline, "the record unit never ran"
            await asyncio.sleep(0.01)
        intake.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await intake

        session = get_open_session()
        try:
            svc = ChannelEventService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A)))
            recorded = svc.is_duplicate_inbound("slack", "evt-cancelled")
        finally:
            session.close()
        assert (recorded, len(captured)) == (True, 1)
    finally:
        release.set()
        for coro in captured:
            coro.close()
        _delete_channel_events("evt-cancelled")


# --- Real plugins' extractors: pulled unverified, used only as a lookup key ---

def test_slack_extract_team_id():
    from cowork.channels.plugins.slack import extract_team_id

    body = b'{"type":"event_callback","team_id":"T0001","event":{}}'
    assert extract_team_id(body, {}) == "T0001"


def test_slack_extract_team_id_missing():
    from cowork.channels.plugins.slack import extract_team_id

    assert extract_team_id(b'{"type":"event_callback","event":{}}', {}) is None


def test_slack_extract_team_id_malformed_body():
    from cowork.channels.plugins.slack import extract_team_id

    assert extract_team_id(b"not json", {}) is None


def test_slack_plugin_declares_the_extractor():
    from cowork.channels.plugins.slack import extract_team_id, plugin

    assert plugin.extract_routing_key is extract_team_id


def test_discord_extract_application_id():
    from cowork.channels.plugins.discord import extract_application_id

    body = b'{"type":2,"application_id":"A0001","guild_id":"G0001"}'
    assert extract_application_id(body, {}) == "A0001"


def test_discord_extract_application_id_present_without_guild():
    # DM-context interactions have no guild_id — application_id is what's
    # actually reliable, which is exactly why it's the routing key, not guild_id.
    from cowork.channels.plugins.discord import extract_application_id

    body = b'{"type":2,"application_id":"A0001"}'
    assert extract_application_id(body, {}) == "A0001"


def test_discord_extract_application_id_missing():
    from cowork.channels.plugins.discord import extract_application_id

    assert extract_application_id(b'{"type":2}', {}) is None


def test_discord_plugin_declares_the_extractor():
    from cowork.channels.plugins.discord import extract_application_id, plugin

    assert plugin.extract_routing_key is extract_application_id


# --- End-to-end: local mode's real wiring, exactly as _install_channels builds it ---

def _signed_slack_headers(body: bytes, signing_secret: str) -> dict:
    ts = str(int(time.time()))
    base = f"v0:{ts}:".encode("utf-8") + body
    sig = "v0=" + hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return {"x-slack-request-timestamp": ts, "x-slack-signature": sig}


async def test_local_mode_webhook_falls_through_to_local_bridge_end_to_end():
    """The claim from the design conversation, proven rather than reasoned
    through: in local mode, a real inbound webhook whose team_id matches
    nothing (true today — nothing sets external_account_id on a local
    install) still resolves and verifies correctly, via the same plain-
    resolver path as before this feature existed. Uses the real Slack
    plugin, the real registry, and the real resolve_org_bridge — the actual
    production wiring _install_channels builds, not a stand-in for it."""
    from cowork.channels.plugins.slack import plugin as slack_plugin

    registry = PluginRegistry()
    registry.register(slack_plugin)

    signing_secret = "test-signing-secret"
    session = get_open_session()
    try:
        ChannelConfigService(ScopedSession(session, LOCAL_SCOPE), registry=registry).set_config(
            "slack", {"bot_token": "xoxb-test", "signing_secret": signing_secret},
        )
    finally:
        session.close()

    try:
        adapters = LiveAdapterRegistry(registry)
        assert await adapters.refresh_all() == ["slack"]  # mirrors the real lifespan bootstrap

        delivered = []

        async def sink(channel_type, event, org_id=None):
            delivered.append((channel_type, event))

        app = FastAPI()
        app.include_router(
            build_channel_webhook_router(
                slack_plugin, resolver=adapters.get, sink=sink,
                org_resolver=adapters.resolve_org_bridge,
            )
        )
        client = TestClient(app)

        body = json.dumps({
            "type": "event_callback",
            "team_id": "T-UNMATCHED",  # nothing sets external_account_id on a local install
            "event_id": "Ev0001",
            "event": {
                "type": "message", "channel": "C1", "user": "U1", "text": "hello",
                "ts": "1700000000.000000",
            },
        }).encode()

        response = client.post("/slack/events", content=body, headers=_signed_slack_headers(body, signing_secret))

        assert response.status_code == 200
        assert len(delivered) == 1
        assert delivered[0][0] == "slack"
    finally:
        session = get_open_session()
        try:
            ChannelConfigService(ScopedSession(session, LOCAL_SCOPE), registry=registry).delete_config("slack")
        finally:
            session.close()
        _delete_channel_events("slack:event:Ev0001")


def test_the_shutdown_drain_lets_a_cancelled_turn_unwind_before_the_event_loop_closes(
    monkeypatch, caplog,
):
    """A channel turn cancelled at the drain's timeout still writes the
    controller's cancel flag, records its artifacts on the file store and
    queues its scratchpad close. The teardown after the drain can end at once,
    and the event loop's close then cancels anything still running a second
    time, so the drain waits for that unwind rather than returning at the
    cancel."""
    monkeypatch.setattr(webhooks, "_background_tasks", set())
    monkeypatch.setattr(chat_session, "_scratchpad_closes", set())
    happened: list[str] = []

    class _Scratchpads:
        async def close_all(self):
            await asyncio.sleep(0.01)
            happened.append("scratchpads closed")

    class _ChatSession:
        _scratchpads = _Scratchpads()

    async def channel_turn():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            happened.append("cancel flag written")
            raise
        finally:
            close_session_scratchpads(_ChatSession(), owner="a channel turn")
            await run_to_completion(run_efs(
                lambda: happened.append("artifacts recorded"), scope=LOCAL_SCOPE, deadline=False,
            ))

    async def lifespan_teardown():
        webhooks._default_scheduler(channel_turn())
        await asyncio.sleep(0)
        await webhooks.drain_background_tasks(timeout=0.05)
        await drain_scratchpad_closes()

    with caplog.at_level(logging.WARNING):
        asyncio.run(lifespan_teardown())

    assert sorted(happened) == ["artifacts recorded", "cancel flag written", "scratchpads closed"]
    assert [
        (r.event, r.count) for r in caplog.records if r.name == webhooks.log.name
    ] == [("shutdown_channel_tasks_cancelled", 1)]
    assert not [r for r in caplog.records if r.name == chat_session.logger.name]


async def test_the_shutdown_drain_gives_up_on_a_task_that_waits_out_its_cancel(
    monkeypatch, caplog, owned_logger,
):
    """A channel turn's end-of-turn bookkeeping runs under run_to_completion,
    so a cancel waits until that work ends, which a file store stall can hold
    off with no bound. The drain cancels the turn at its timeout, waits
    cancel_timeout for it to end, then returns without it, so the teardown
    steps after the drain still run: the drain takes timeout plus
    cancel_timeout, never more."""
    logged = owned_logger(webhooks.log.name, level=logging.WARNING)
    monkeypatch.setattr(webhooks, "_background_tasks", set())
    bookkeeping_may_end = asyncio.Event()

    async def channel_turn():
        await run_to_completion(bookkeeping_may_end.wait())

    webhooks._default_scheduler(channel_turn())
    (turn,) = webhooks._background_tasks
    timeout, cancel_timeout = 0.2, 0.3
    loop = asyncio.get_running_loop()
    started = loop.time()
    drain = asyncio.create_task(
        webhooks.drain_background_tasks(timeout=timeout, cancel_timeout=cancel_timeout)
    )
    try:
        await asyncio.wait([drain], timeout=2)
        assert drain.done(), "the drain waited for a task that waits out its cancel"
        waited = loop.time() - started
        assert timeout + cancel_timeout - 0.02 <= waited < timeout + cancel_timeout + 0.5, (
            f"the drain returned after {waited:.2f}s"
        )
        assert turn.cancelling(), "the drain left the task running without cancelling it"
        assert not turn.done()
    finally:
        bookkeeping_may_end.set()
        await asyncio.wait([drain, turn], timeout=5)

    assert turn.cancelled(), "the task did not end cancelled once its bookkeeping ended"
    records = [
        r for r in caplog.records
        if r.name == webhooks.log.name and r.levelno == logging.WARNING
    ]
    assert [(r.event, r.count) for r in records] == [
        ("shutdown_channel_tasks_cancelled", 1),
        ("shutdown_channel_tasks_unfinished", 1),
    ]
    assert_log_fields(logged.output(), event="shutdown_channel_tasks_cancelled", count=1)
    assert_log_fields(logged.output(), event="shutdown_channel_tasks_unfinished", count=1)


async def test_the_shutdown_drain_waits_for_tasks_that_finish_and_raises_none_of_their_failures(
    monkeypatch, caplog,
):
    """A task that ends within the timeout is waited for. One that raised
    neither fails the drain nor logs a cancel, and the drain retrieves its
    failure, so asyncio does not log it as never retrieved once the task is
    collected."""
    monkeypatch.setattr(webhooks, "_background_tasks", set())
    finished: list[str] = []

    async def quick_turn():
        await asyncio.sleep(0.05)
        finished.append("quick")

    async def failing_turn():
        await asyncio.sleep(0.05)
        raise RuntimeError("adapter send failed")

    webhooks._default_scheduler(quick_turn())
    webhooks._default_scheduler(failing_turn())
    tasks = list(webhooks._background_tasks)
    with caplog.at_level(logging.WARNING, logger=webhooks.log.name):
        await webhooks.drain_background_tasks(timeout=2)

    assert finished == ["quick"]
    assert all(task.done() for task in tasks)
    assert not [r for r in caplog.records if r.name == webhooks.log.name]
    del tasks
    gc.collect()
    assert not [
        r for r in caplog.records
        if r.name == "asyncio" and "Task exception was never retrieved" in r.getMessage()
    ]
