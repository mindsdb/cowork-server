"""An org-mode channel turn has no member behind it, so the product permission
check refuses it before any job reaches the turn queue.

The binding's capability switches reach in-process turns only. Remote turns
do not carry them yet, so this refusal is what keeps a remote channel turn
from running with every tool. A change that gives channel turns a principal
must also send the switches; this test fails first, as a reminder.
"""
from __future__ import annotations

import asyncio
import sys

import httpx
from sqlmodel import delete

from cowork.channels import webhooks as webhooks_mod
from cowork.channels.webhooks import drain_background_tasks
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.models.channel import SENDER_ALLOWED, ChannelSender
from cowork.services.channels import ChannelConfigService
from cowork.services.product_permissions import ProductPermissionDenied
from cowork.turnqueue import producer
from tests.test_channels_remote_turn_e2e import (  # noqa: F401, F811  (org_mode_remote_backend is a fixture, used by name)
    ORG_A,
    SIGNING_SECRET,
    _delete_binding_and_its_rows,
    _delete_channel_events,
    _ReplyOnlyRedis,
    _signed_slack_headers,
    _slack_event_body,
    org_mode_remote_backend,
)


class _RecordingRedis(_ReplyOnlyRedis):
    """The fake Redis, recording every job written to a stream. A queued job
    ends the turn at once: no pod answers here, so the turn would otherwise
    wait for replies instead of letting the test fail."""

    def __init__(self) -> None:
        super().__init__([])
        self.jobs: list[str] = []

    async def xadd(self, stream, fields):
        self.jobs.append(stream)
        raise RuntimeError("a channel turn with no principal reached the turn queue")


class _RecordingLog:
    """Stands in for the webhooks module logger; keeps each logged exception."""

    def __init__(self) -> None:
        self.exceptions: list[BaseException] = []

    def exception(self, *_args, **_kwargs) -> None:
        self.exceptions.append(sys.exc_info()[1])

    def __getattr__(self, _name):
        return lambda *_a, **_k: None


def test_an_org_mode_channel_turn_with_no_principal_queues_no_job(monkeypatch, org_mode_remote_backend):
    """Today the sender is told nothing: the refusal ends as a logged sink
    failure. That silence is the current state, not the wanted one; the ticket
    that gives channel turns a principal decides what the sender hears."""
    # Every step after the permission check is stubbed as the allowance test
    # does, so only the permission check can stop this turn.
    for helper, value in (
        ("_stage_remote_workspace_files", None),
        ("_remote_artifacts_context", None),
        ("_remote_seed_history", ([], None)),
        ("_remote_workspace", None),
    ):
        monkeypatch.setattr(
            f"cowork.handlers.responses.ResponsesHandler.{helper}",
            staticmethod(lambda session, conv_id, value=value: value),
        )

    async def fake_llm_block(**kwargs):
        return {"api_key": "mdb_turnkey"}

    async def no_oauth_block(**kwargs):
        return None

    fake_redis = _RecordingRedis()
    monkeypatch.setattr(producer, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(producer, "mint_llm_block", fake_llm_block)
    monkeypatch.setattr(producer, "_mint_oauth_block", no_oauth_block)
    recording_log = _RecordingLog()
    monkeypatch.setattr(webhooks_mod, "log", recording_log)

    posted: list[str] = []
    real_post = httpx.AsyncClient.post

    async def fake_post(self, url, json=None, **kw):
        if not str(url).startswith("https://slack.com/api"):
            return await real_post(self, url, json=json, **kw)
        posted.append(str(url).rsplit("/", 1)[-1])

        class R:
            def json(self):
                return {"ok": True, "ts": "1700000009.000000"}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    from cowork.server import create_app
    app = create_app()
    adapters = app.state.channel_adapters

    try:
        session = get_open_session()
        try:
            svc = ChannelConfigService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A)))
            svc.set_config("slack", {"bot_token": "xoxb-org-a-refusal", "signing_secret": SIGNING_SECRET})
            svc.set_external_account_id("slack", "T-ORG-A-REFUSAL")
            # A sender the owner approved, so the sender gate lets the turn start.
            session.add(ChannelSender(
                channel_type="slack", external_group_id="D1", external_sender_id="U1",
                status=SENDER_ALLOWED, org_id=ORG_A,
            ))
            session.commit()
        finally:
            session.close()

        async def flow():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert await adapters.get_or_refresh("slack", ORG_A) is not None
                body = _slack_event_body("T-ORG-A-REFUSAL", "Ev-remote-refusal-1")
                r = await client.post(
                    "/api/v1/channels/slack/events", content=body,
                    headers=_signed_slack_headers(body, SIGNING_SECRET),
                )
                assert r.status_code == 200
                await drain_background_tasks()

        asyncio.run(flow())

        assert fake_redis.jobs == []
        assert [type(exc) for exc in recording_log.exceptions] == [ProductPermissionDenied]
        assert "chat.postMessage" not in posted
    finally:
        session = get_open_session()
        try:
            ChannelConfigService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A))).delete_config("slack")
            session.exec(delete(ChannelSender).where(
                ChannelSender.org_id == ORG_A,
                ChannelSender.channel_type == "slack",
                ChannelSender.external_group_id == "D1",
                ChannelSender.external_sender_id == "U1",
            ))
            session.commit()
        finally:
            session.close()
        _delete_channel_events("slack:event:Ev-remote-refusal-1")
        _delete_binding_and_its_rows("slack", "D1", ORG_A)
