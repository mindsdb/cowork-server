"""WS4-T2/T3/T4: command broker (timeout never ok), control/stop/takeover,
BridgeClient dispatch, reconnect/resume, and the /browse endpoints.
"""
from __future__ import annotations

import asyncio

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.browser import BrowserSession, BrowserTabGrant
from cowork.models.conversation import Conversation
from cowork.schemas.browser import (
    BridgeCommandResult,
    BridgeState,
    BrowserActionClass,
    BrowserActionType,
    ControlState,
    PermissionDecision,
    ResultCode,
)
from cowork.server import app
from cowork.services.browser.bridge import BridgeCommandService
from cowork.services.browser.client import BridgeClient
from cowork.services.browser.control import BrowserControlService
from cowork.services.projects import GENERAL_PROJECT_ID

client = TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _make_session(session: Session, domain="example.com", grant=True) -> BrowserSession:
    conv = Conversation(topic="t", project_id=GENERAL_PROJECT_ID)
    session.add(conv)
    session.commit()
    session.refresh(conv)
    bs = BrowserSession(
        conversation_id=conv.id, project_id=GENERAL_PROJECT_ID,
        active_domain=domain, available=True,
    )
    session.add(bs)
    session.commit()
    session.refresh(bs)
    if grant:
        session.add(
            BrowserTabGrant(
                session_id=bs.id, domain=domain,
                action_class=BrowserActionClass.navigate.value,
                decision=PermissionDecision.granted.value,
                granted_at=datetime.now(timezone.utc),
            )
        )
        session.commit()
    return bs


# ── broker: hung command times out, never ok ─────────────────────────
def test_broker_execute_times_out_without_poller():
    async def go():
        broker = BridgeCommandService(default_timeout_s=0.1)
        return await broker.execute(
            session_id="s1", action_type=BrowserActionType.inspect, timeout_s=0.1
        )
    result = asyncio.run(go())
    assert result.result_code == ResultCode.timeout
    assert result.result_code != ResultCode.ok


def test_broker_next_then_resolve_roundtrip():
    async def go():
        broker = BridgeCommandService(default_timeout_s=2.0)

        async def poller():
            cmd = await broker.next("s2", wait_s=2.0)
            assert cmd is not None
            await broker.resolve(
                cmd.command_id,
                BridgeCommandResult(
                    command_id=cmd.command_id, result_code=ResultCode.ok,
                    observed={"http_status": 200, "settled": True},
                ),
            )
            return cmd

        exec_task = asyncio.create_task(
            broker.execute(session_id="s2", action_type=BrowserActionType.inspect, timeout_s=2.0)
        )
        await poller()
        return await exec_task
    result = asyncio.run(go())
    assert result.result_code == ResultCode.ok
    assert result.observed["http_status"] == 200


def test_broker_next_returns_none_on_wait_elapsed():
    async def go():
        broker = BridgeCommandService()
        return await broker.next("empty", wait_s=0.05)
    assert asyncio.run(go()) is None


def test_broker_execute_cancelled_discards_command():
    # A cancelled producer must not leave a queued command or a dead future
    # behind — a subsequent poll finds nothing.
    async def go():
        broker = BridgeCommandService(default_timeout_s=5.0)
        task = asyncio.create_task(
            broker.execute(
                session_id="s-cancel",
                action_type=BrowserActionType.inspect,
                timeout_s=5.0,
            )
        )
        await asyncio.sleep(0.02)  # let it enqueue + await
        assert broker.pending_count("s-cancel") == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Command dropped from the queue; nothing left to pull.
        assert broker.pending_count("s-cancel") == 0
        return await broker.next("s-cancel", wait_s=0.05)

    assert asyncio.run(go()) is None


def test_broker_drain_session_resolves_awaiting_producers():
    # Draining a gated session resolves every awaiting producer with a
    # terminal (non-ok) code so it stops waiting instead of hanging.
    async def go():
        broker = BridgeCommandService(default_timeout_s=5.0)
        task = asyncio.create_task(
            broker.execute(
                session_id="s-drain",
                action_type=BrowserActionType.inspect,
                timeout_s=5.0,
            )
        )
        await asyncio.sleep(0.02)
        drained = await broker.drain_session("s-drain", ResultCode.error, detail="stopped")
        result = await task
        return drained, result

    drained, result = asyncio.run(go())
    assert drained == 1
    assert result.result_code == ResultCode.error
    assert result.result_code != ResultCode.ok


# ── control/stop/takeover/reconnect ──────────────────────────────────
def test_control_stop_blocks_and_survives_reconnect(session):
    bs = _make_session(session)
    control = BrowserControlService(session)
    assert not control.is_blocked(bs.id)
    control.stop(bs.id)
    assert control.is_blocked(bs.id)
    # reconnect must NOT clear stopped
    control.reconnect(bs.id)
    session.refresh(bs)
    assert bs.control_state == ControlState.stopped.value
    assert bs.available is False
    assert control.is_blocked(bs.id)


def test_control_takeover_marks_and_pauses(session):
    bs = _make_session(session)
    control = BrowserControlService(session)
    control.takeover(bs.id)
    session.refresh(bs)
    assert bs.control_state == ControlState.taken_over.value
    assert bs.available is False
    assert control.is_blocked(bs.id)


def test_on_bridge_state_chrome_restart_marks_lost_requires_reapprove(session):
    bs = _make_session(session)
    control = BrowserControlService(session)
    control.on_bridge_state(bs.id, BridgeState.connected, target_changed=True)
    session.refresh(bs)
    assert bs.bridge_state == BridgeState.lost.value
    assert bs.requires_reapproval is True
    assert bs.available is False


def test_on_bridge_state_connected_respects_stopped(session):
    bs = _make_session(session)
    control = BrowserControlService(session)
    control.stop(bs.id)
    control.on_bridge_state(bs.id, BridgeState.connected)
    session.refresh(bs)
    # stopped stays stopped, never becomes available
    assert bs.control_state == ControlState.stopped.value
    assert bs.available is False


def test_stop_by_conversation_creates_session_and_gate_persists(session):
    # A Stop arriving BEFORE any browser session exists must persist: the
    # session row is created stopped, and a later approve does not flip it
    # back to active/available (P2 review finding).
    conv = Conversation(topic="pre-stop", project_id=GENERAL_PROJECT_ID)
    session.add(conv)
    session.commit()
    session.refresh(conv)

    control = BrowserControlService(session)
    assert control.get_by_conversation(conv.id) is None
    sess = control.stop_by_conversation(conv.id)
    assert sess is not None
    assert sess.control_state == ControlState.stopped.value
    assert sess.available is False

    # A subsequent approval keeps the gate: stopped stays stopped,
    # available stays False.
    from cowork.services.browser.approval import BrowserApprovalService
    approved = BrowserApprovalService(session).approve(conv.id, "example.com")
    assert approved.control_state == ControlState.stopped.value
    assert approved.available is False


def test_takeover_by_conversation_creates_session(session):
    conv = Conversation(topic="pre-takeover", project_id=GENERAL_PROJECT_ID)
    session.add(conv)
    session.commit()
    session.refresh(conv)
    sess = BrowserControlService(session).takeover_by_conversation(conv.id)
    assert sess is not None
    assert sess.control_state == ControlState.taken_over.value


def test_explicit_reapproval_ends_a_takeover(session):
    # Per BrowserControlService.resume_on_new_turn()'s docstring, a
    # `taken_over` gate is NOT cleared by a new turn -- "only an explicit
    # re-approval ends a takeover". Verify approve() actually does this
    # (unlike `stopped`, which approve() must NOT clear -- see
    # test_stop_by_conversation_creates_session_and_gate_persists above).
    conv = Conversation(topic="reapprove-after-takeover", project_id=GENERAL_PROJECT_ID)
    session.add(conv)
    session.commit()
    session.refresh(conv)

    control = BrowserControlService(session)
    control.takeover_by_conversation(conv.id)
    sess = control.get_by_conversation(conv.id)
    assert sess.control_state == ControlState.taken_over.value
    assert sess.available is False

    from cowork.services.browser.approval import BrowserApprovalService
    approved = BrowserApprovalService(session).approve(conv.id, "example.com")
    assert approved.control_state == ControlState.active.value
    assert approved.available is True

    # And a stopped gate remains unaffected by this change: approve() must
    # still leave `stopped` alone (regression guard for the fix above).
    control.stop(sess.id)
    session.refresh(sess)
    assert sess.control_state == ControlState.stopped.value
    approved_again = BrowserApprovalService(session).approve(conv.id, "example.com")
    assert approved_again.control_state == ControlState.stopped.value
    assert approved_again.available is False


def test_stop_by_conversation_nonexistent_conversation_noop(session):
    from uuid import uuid4
    assert BrowserControlService(session).stop_by_conversation(uuid4()) is None


# ── BridgeClient dispatch ────────────────────────────────────────────
def test_bridge_client_blocked_when_stopped(session):
    bs = _make_session(session)
    BrowserControlService(session).stop(bs.id)
    bc = BridgeClient(session)
    verdict = asyncio.run(bc.inspect(bs.id))
    assert verdict.result_code == ResultCode.permission_denied


def test_bridge_client_unapproved_domain(session):
    bs = _make_session(session, domain="example.com")
    bc = BridgeClient(session)
    verdict = asyncio.run(bc.navigate(bs.id, href="https://evil.com/x"))
    assert verdict.result_code == ResultCode.unapproved_tab


def test_bridge_client_dispatch_ok_with_poller(session):
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=2.0)
    bc = BridgeClient(session, broker=broker)

    async def go():
        async def poller():
            cmd = await broker.next(str(bs.id), wait_s=2.0)
            await broker.resolve(
                cmd.command_id,
                BridgeCommandResult(
                    command_id=cmd.command_id, result_code=ResultCode.ok,
                    observed={"http_status": 200, "links": [1, 2], "settled": True},
                ),
            )
        task = asyncio.create_task(bc.inspect(bs.id))
        await poller()
        return await task
    verdict = asyncio.run(go())
    assert verdict.result_code == ResultCode.ok
    assert verdict.observed["http_status"] == 200
    # persisted row is a content-free digest, not the transient observed
    from cowork.services.browser.actions import BrowserActionStore
    last = BrowserActionStore(session).last_observed(bs.id)
    assert last is not None
    assert set(last.observed_result.keys()) <= {"http_status", "final_domain", "link_count", "settled"}


def test_bridge_client_ok_without_observed_downgraded_and_not_ok_in_db(session):
    # A bridge that returns `ok` with an EMPTY observed must NOT be persisted
    # as observed=ok — WS4 downgrades it to a failed row before persistence.
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=2.0)
    bc = BridgeClient(session, broker=broker)

    async def go():
        async def poller():
            cmd = await broker.next(str(bs.id), wait_s=2.0)
            await broker.resolve(
                cmd.command_id,
                BridgeCommandResult(
                    command_id=cmd.command_id, result_code=ResultCode.ok,
                    observed=None,
                ),
            )
        task = asyncio.create_task(bc.inspect(bs.id))
        await poller()
        return await task

    verdict = asyncio.run(go())
    # Verdict is no longer `ok` — downgraded to an internal error.
    assert verdict.result_code != ResultCode.ok
    assert verdict.observed is None
    # The DB row is `failed` with NO digest.
    from cowork.models.browser import BrowserAction
    from sqlmodel import select
    row = session.exec(
        select(BrowserAction).where(BrowserAction.session_id == bs.id)
    ).first()
    assert row.status == "failed"
    assert row.observed_result is None
    assert row.result_code != ResultCode.ok.value


def test_bridge_client_refuses_while_reapproval_required(session):
    # After a target change the old grants are stale — nothing dispatches
    # until an explicit re-approval clears the flag (P1 review finding).
    bs = _make_session(session, domain="example.com")
    control = BrowserControlService(session)
    control.on_bridge_state(bs.id, BridgeState.connected, target_changed=True)
    session.refresh(bs)
    assert bs.requires_reapproval is True

    broker = BridgeCommandService(default_timeout_s=2.0)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(bc.inspect(bs.id))
    assert verdict.result_code == ResultCode.unapproved_tab
    # Refused pre-dispatch: nothing enqueued, no action row.
    assert broker.pending_count(str(bs.id)) == 0
    from cowork.services.browser.actions import BrowserActionStore
    assert BrowserActionStore(session).action_count(bs.id) == 0

    # An explicit (user-driven) re-approval clears the flag; dispatch then
    # reaches the broker again (times out here — no poller — but is no
    # longer refused as unapproved).
    from cowork.services.browser.approval import BrowserApprovalService
    BrowserApprovalService(session).approve(bs.conversation_id, "example.com")
    session.refresh(bs)
    assert bs.requires_reapproval is False
    fast = BridgeCommandService(default_timeout_s=0.1)
    verdict2 = asyncio.run(BridgeClient(session, broker=fast).inspect(bs.id))
    assert verdict2.result_code == ResultCode.timeout


def test_bridge_client_cancel_mid_execute_marks_row_failed(session):
    # A cancelled producer must not strand the action row `in_flight`
    # (P2 review finding): the row goes terminal `failed`, never observed=ok.
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=5.0)
    bc = BridgeClient(session, broker=broker)

    async def go():
        task = asyncio.create_task(bc.inspect(bs.id))
        await asyncio.sleep(0.05)  # let it enqueue + await the future
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    from cowork.models.browser import BrowserAction
    from sqlmodel import select
    row = session.exec(
        select(BrowserAction).where(BrowserAction.session_id == bs.id)
    ).first()
    assert row is not None
    assert row.status == "failed"
    assert row.result_code == ResultCode.error.value
    assert row.observed_result is None


def test_bridge_client_no_poller_times_out_never_ok(session):
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(bc.inspect(bs.id))
    assert verdict.result_code == ResultCode.timeout
    assert verdict.result_code != ResultCode.ok


def test_bridge_client_send_by_conversation_ok(session):
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=2.0)
    bc = BridgeClient(session, broker=broker)

    async def go():
        async def poller():
            cmd = await broker.next(str(bs.id), wait_s=2.0)
            await broker.resolve(
                cmd.command_id,
                BridgeCommandResult(
                    command_id=cmd.command_id, result_code=ResultCode.ok,
                    observed={"http_status": 200},
                ),
            )
        # `send` resolves the session from the conversation id (the LLM verb
        # surface), translating `follow_link` → navigate.
        task = asyncio.create_task(
            bc.send(bs.conversation_id, "inspect")
        )
        await poller()
        return await task
    verdict = asyncio.run(go())
    assert verdict.result_code == ResultCode.ok


def test_bridge_client_send_unknown_action(session):
    bs = _make_session(session, domain="example.com")
    bc = BridgeClient(session)
    verdict = asyncio.run(bc.send(bs.conversation_id, "click"))
    assert verdict.result_code == ResultCode.error
    assert "unsupported" in (verdict.detail or "")


def test_bridge_client_send_no_session(session):
    from uuid import uuid4
    bc = BridgeClient(session)
    verdict = asyncio.run(bc.send(uuid4(), "inspect"))
    assert verdict.result_code == ResultCode.error
    assert "no browser session" in (verdict.detail or "")


# ── endpoints ────────────────────────────────────────────────────────
def _create_conv() -> str:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        conv = Conversation(topic="ep", project_id=GENERAL_PROJECT_ID)
        s.add(conv)
        s.commit()
        s.refresh(conv)
        bs = BrowserSession(
            conversation_id=conv.id, project_id=GENERAL_PROJECT_ID,
            active_domain="example.com", available=True,
        )
        s.add(bs)
        s.commit()
        return str(conv.id)


def test_status_legacy_shape_without_conversation():
    r = client.get("/api/v1/browse/status")
    assert r.status_code == 200
    assert r.json() == {"available": False}


def test_control_stop_endpoint_sets_gate():
    conv_id = _create_conv()
    r = client.post("/api/v1/browse/control/stop", json={"conversation_id": conv_id})
    assert r.status_code == 200
    assert r.json()["control_state"] == "stopped"
    # status reflects stopped
    r2 = client.get("/api/v1/browse/status", params={"conversation_id": conv_id})
    assert r2.json()["control_state"] == "stopped"


def test_control_takeover_endpoint():
    conv_id = _create_conv()
    r = client.post("/api/v1/browse/control/takeover", json={"conversation_id": conv_id})
    assert r.status_code == 200
    assert r.json()["control_state"] == "taken_over"


def test_control_stop_no_session_is_ok():
    import uuid
    r = client.post(
        "/api/v1/browse/control/stop", json={"conversation_id": str(uuid.uuid4())}
    )
    assert r.status_code == 200
    assert r.json()["stopped"] is True


def test_bridge_hello_and_resume():
    conv_id = _create_conv()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        sess = BrowserControlService(s).get_by_conversation(conv_id)
        sid = str(sess.id)
    r = client.post("/api/v1/browse/bridge/hello", json={"session_id": sid})
    assert r.status_code == 200
    # The poller learns/echoes the session_id from hello (legacy path too).
    assert r.json()["session_id"] == sid
    r2 = client.get("/api/v1/browse/resume", params={"session_id": sid})
    assert r2.status_code == 200
    body = r2.json()
    assert body.get("session_id") == sid


def test_bridge_state_accepts_all_enum_values_and_422s_hyphenated():
    conv_id = _create_conv()
    sid = _session_id_for_conv(conv_id)
    for bs in ("disconnected", "awaiting_approval", "connected", "lost"):
        r = client.post(
            "/api/v1/browse/bridge/state",
            json={"session_id": sid, "bridge_state": bs},
        )
        assert r.status_code == 200, (bs, r.text)
        assert r.json()["session_id"] == sid
    # The client maps its hyphenated form before sending; a raw hyphenated
    # value must 422 cleanly, not be silently coerced.
    r = client.post(
        "/api/v1/browse/bridge/state",
        json={"session_id": sid, "bridge_state": "awaiting-approval"},
    )
    assert r.status_code == 422


def test_poller_handshake_hello_next_result_end_to_end(monkeypatch):
    # The full shipping-path handshake the Electron poller runs:
    #   hello(conversation_id, domain) → session_id
    #   → commands/next(session_id) pulls the queued command
    #   → commands/{id}/result resolves the awaiting producer.
    #
    # The producer and the two poller endpoint handlers must share ONE event
    # loop (the broker's asyncio primitives are loop-bound), so hello goes
    # over HTTP and next/result exercise the endpoint handlers directly on
    # the test loop, against a fresh (monkeypatched) global broker.
    import cowork.services.browser.bridge as bridge_mod
    from cowork.api.v1.endpoints.compat.stubs import (
        _CommandsNextRequest,
        browse_commands_next,
        browse_commands_result,
    )

    fresh_broker = BridgeCommandService(default_timeout_s=5.0)
    monkeypatch.setattr(bridge_mod, "bridge_command_service", fresh_broker)

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        conv = Conversation(topic="handshake", project_id=GENERAL_PROJECT_ID)
        s.add(conv)
        s.commit()
        s.refresh(conv)
        conv_id = str(conv.id)

    # 1. hello upserts the session + grant and returns the session_id.
    r = client.post(
        "/api/v1/browse/bridge/hello",
        json={"conversation_id": conv_id, "domain": "example.com"},
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid

    # 2/3. Producer dispatches through BridgeClient while the "poller" pulls
    # the command via the /commands/next handler with the session_id learned
    # from hello and posts its result via the /commands/{id}/result handler.
    async def go():
        with Session(engine) as producer_db, Session(engine) as poller_db:
            bc = BridgeClient(producer_db, broker=fresh_broker)
            task = asyncio.create_task(bc.send(conv_id, "inspect"))
            await asyncio.sleep(0.05)  # let the command enqueue

            next_resp = await browse_commands_next(
                _CommandsNextRequest(session_id=sid, wait_s=2.0),
                session=poller_db,
            )
            cmd = next_resp["command"]
            assert cmd is not None
            assert cmd["session_id"] == sid
            assert cmd["action_type"] == "inspect"

            result_resp = await browse_commands_result(
                cmd["command_id"],
                BridgeCommandResult(
                    command_id=cmd["command_id"],
                    result_code=ResultCode.ok,
                    observed={"http_status": 200, "settled": True},
                ),
            )
            assert result_resp["resolved"] is True
            return await task

    verdict = asyncio.run(go())
    assert verdict.result_code == ResultCode.ok
    assert verdict.observed["http_status"] == 200


def test_responses_cancel_marks_session_stopped():
    conv_id = _create_conv()
    r = client.post("/api/v1/responses/cancel", json={"conversation_id": conv_id})
    assert r.status_code == 200
    r2 = client.get("/api/v1/browse/status", params={"conversation_id": conv_id})
    assert r2.json()["control_state"] == "stopped"


def _session_id_for_conv(conv_id: str) -> str:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        sess = BrowserControlService(s).get_by_conversation(conv_id)
        return str(sess.id)


def test_commands_next_refuses_when_stopped():
    # Stop after a session exists; /commands/next must never hand out a
    # command for a stopped session (drains + reports blocked).
    conv_id = _create_conv()
    sid = _session_id_for_conv(conv_id)
    client.post("/api/v1/browse/control/stop", json={"conversation_id": conv_id})
    r = client.post("/api/v1/browse/commands/next", json={"session_id": sid, "wait_s": 0.1})
    assert r.status_code == 200
    body = r.json()
    assert body["command"] is None
    assert body.get("blocked") == "stopped"


def test_control_approve_creates_session_and_grant():
    # The production approval path: no session exists for a fresh conversation
    # until approve() upserts it + grants the host, after which send() works.
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        conv = Conversation(topic="approve", project_id=GENERAL_PROJECT_ID)
        s.add(conv)
        s.commit()
        s.refresh(conv)
        conv_id = str(conv.id)
        # No browser session yet.
        assert BrowserControlService(s).get_by_conversation(conv_id) is None

    r = client.post(
        "/api/v1/browse/control/approve",
        json={"conversation_id": conv_id, "domain": "https://Example.com/some/path"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["active_domain"] == "example.com"  # host-only
    assert body["control_state"] == "active"

    # A session now exists and the tool's send() lookup + permission check pass.
    with Session(engine) as s:
        bc = BridgeClient(s, broker=BridgeCommandService(default_timeout_s=0.1))
        verdict = asyncio.run(bc.send(conv_id, "inspect"))
    # Not a session/permission failure — it reaches the broker and times out.
    assert verdict.result_code == ResultCode.timeout


def test_bridge_hello_upserts_session_from_conversation():
    # A conversation-scoped hello with a domain creates the session + grant
    # rather than 404-ing on first connect.
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        conv = Conversation(topic="hello-upsert", project_id=GENERAL_PROJECT_ID)
        s.add(conv)
        s.commit()
        s.refresh(conv)
        conv_id = str(conv.id)

    r = client.post(
        "/api/v1/browse/bridge/hello",
        json={"conversation_id": conv_id, "domain": "example.com"},
    )
    assert r.status_code == 200
    with Session(engine) as s:
        sess = BrowserControlService(s).get_by_conversation(conv_id)
        assert sess is not None
        assert sess.active_domain == "example.com"


def test_hello_never_self_approves_pending_reapproval():
    # Chrome restart → target_changed sets requires_reapproval. The poller's
    # automatic re-hello (which carries a domain) must NOT clear the flag /
    # re-grant — only the explicit user-driven /control/approve may.
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        conv = Conversation(topic="no-self-approve", project_id=GENERAL_PROJECT_ID)
        s.add(conv)
        s.commit()
        s.refresh(conv)
        conv_id = str(conv.id)

    # Initial approved attach.
    r = client.post(
        "/api/v1/browse/bridge/hello",
        json={"conversation_id": conv_id, "domain": "example.com"},
    )
    sid = r.json()["session_id"]

    # Chrome restart: target ids changed.
    r = client.post(
        "/api/v1/browse/bridge/hello",
        json={"conversation_id": conv_id, "domain": "example.com", "target_changed": True},
    )
    assert r.json()["requires_reapproval"] is True

    # Automatic re-hello with a domain — must NOT self-approve.
    r = client.post(
        "/api/v1/browse/bridge/hello",
        json={"conversation_id": conv_id, "domain": "example.com"},
    )
    assert r.status_code == 200
    assert r.json()["requires_reapproval"] is True

    # Dispatch is still refused while re-approval is pending.
    with Session(engine) as s:
        bc = BridgeClient(s, broker=BridgeCommandService(default_timeout_s=0.1))
        verdict = asyncio.run(bc.send(conv_id, "inspect"))
    assert verdict.result_code == ResultCode.unapproved_tab

    # Explicit user-driven approve clears it; dispatch reaches the broker.
    r = client.post(
        "/api/v1/browse/control/approve",
        json={"conversation_id": conv_id, "domain": "example.com"},
    )
    assert r.status_code == 200
    with Session(engine) as s:
        sess = BrowserControlService(s).get_session(sid)
        assert sess.requires_reapproval is False
        bc = BridgeClient(s, broker=BridgeCommandService(default_timeout_s=0.1))
        verdict = asyncio.run(bc.send(conv_id, "inspect"))
    assert verdict.result_code == ResultCode.timeout


# ── review fixes: href-host permission, post-wakeup gate, fresh turn ──
def test_send_follow_link_cross_host_refused(session):
    # send() must check the permission against the href's registrable host,
    # not fall back to the session's approved active_domain: with an
    # approved example.com tab, follow_link to evil.com must be refused.
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(bs.conversation_id, "follow_link", href="https://evil.com/x")
    )
    assert verdict.result_code == ResultCode.unapproved_tab
    assert verdict.domain == "evil.com"
    # Nothing was enqueued for the poller.
    assert broker.pending_count(str(bs.id)) == 0


def test_send_follow_link_same_host_reaches_broker(session):
    # Same-host follow_link still passes the grant (reaches the broker and
    # times out with no poller — not a permission failure).
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(bs.conversation_id, "follow_link", href="https://example.com/a/b")
    )
    assert verdict.result_code == ResultCode.timeout


def test_commands_next_rechecks_gate_after_wakeup(session):
    # A Stop that lands while /commands/next is awaiting (draining an empty
    # queue), followed by an enqueue from a producer that had already passed
    # its pre-dispatch check, must NOT hand the command to the extension:
    # the endpoint re-reads the session after the wakeup and blocks.
    from cowork.api.v1.endpoints.compat.stubs import (
        _CommandsNextRequest,
        browse_commands_next,
    )
    from cowork.schemas.browser import BridgeCommand
    from cowork.services.browser.bridge import bridge_command_service

    bs = _make_session(session, domain="example.com")
    sid = str(bs.id)

    async def go():
        poll = asyncio.create_task(
            browse_commands_next(
                _CommandsNextRequest(session_id=sid, wait_s=2.0), session
            )
        )
        await asyncio.sleep(0.05)  # poller is now awaiting next()

        # Stop lands in a DIFFERENT db session (as a real request would) and
        # drains the (empty) queue.
        engine = get_engine(get_app_settings().database.uri)
        with Session(engine) as other:
            BrowserControlService(other).stop(sid)
        await bridge_command_service.drain_session(sid, ResultCode.error)

        # A producer that had already passed its pre-dispatch check enqueues,
        # waking the long-poll.
        cmd = BridgeCommand(
            command_id=bridge_command_service.new_command_id(),
            action_type=BrowserActionType.inspect,
            session_id=sid,
        )
        future = await bridge_command_service.enqueue(cmd)
        body = await poll
        return body, future

    body, future = asyncio.run(go())
    assert body["command"] is None
    assert body.get("blocked") == "stopped"
    # The pulled command's producer was resolved terminally (never ok).
    assert future.done()
    assert future.result().result_code != ResultCode.ok


def test_resume_on_new_turn_clears_stopped_gate(session):
    bs = _make_session(session, domain="example.com")
    control = BrowserControlService(session)
    control.on_bridge_state(bs.id, BridgeState.connected)
    control.stop(bs.id)
    assert control.is_blocked(bs.id)

    sess = control.resume_on_new_turn(bs.conversation_id)
    assert sess.control_state == ControlState.active.value
    assert sess.available is True
    assert not control.is_blocked(bs.id)

    # A stopped-but-disconnected session resumes the gate without becoming
    # available.
    control.on_bridge_state(bs.id, BridgeState.disconnected)
    control.stop(bs.id)
    sess = control.resume_on_new_turn(bs.conversation_id)
    assert sess.control_state == ControlState.active.value
    assert sess.available is False


def test_resume_on_new_turn_preserves_takeover(session):
    # Only Stop is turn-scoped; a takeover is user-driven and must survive a
    # fresh turn until an explicit re-approval ends it.
    bs = _make_session(session, domain="example.com")
    control = BrowserControlService(session)
    control.takeover(bs.id)
    sess = control.resume_on_new_turn(bs.conversation_id)
    assert sess.control_state == ControlState.taken_over.value


def test_responses_post_resumes_stopped_browser_session(monkeypatch):
    # POST /responses (a fresh user turn) resets a stopped gate to active
    # before the turn runs — without it, a cancelled browser-enabled turn
    # would block every later browser action for the conversation forever.
    import cowork.api.v1.endpoints.responses as responses_ep

    conv_id = _create_conv()
    client.post("/api/v1/responses/cancel", json={"conversation_id": conv_id})
    r = client.get("/api/v1/browse/status", params={"conversation_id": conv_id})
    assert r.json()["control_state"] == "stopped"

    class _StubHandler:
        def __init__(self, session):
            pass

        async def handle(self, request):
            return {"ok": True}

    monkeypatch.setattr(responses_ep, "ResponsesHandler", _StubHandler)
    r = client.post(
        "/api/v1/responses/",
        json={"conversation": conv_id, "input": "hi", "stream": False},
    )
    assert r.status_code == 200

    r = client.get("/api/v1/browse/status", params={"conversation_id": conv_id})
    assert r.json()["control_state"] == "active"


# ── registrable-host grant matching (server/Electron same-host contract) ──
def test_permission_check_subdomain_covered_by_grant(session):
    # The grant domain is Electron's PSL-registrable host; a same-site
    # SUBDOMAIN target must be granted (its registrable host equals the
    # grant), while a lookalike suffix host must not.
    bs = _make_session(session, domain="example.com")
    from cowork.services.browser.permissions import BrowserPermissionService

    svc = BrowserPermissionService(session)
    assert svc.check(bs.id, "shop.example.com", BrowserActionClass.read).granted
    assert svc.check(bs.id, "shop.example.com", BrowserActionClass.navigate).granted
    assert not svc.check(bs.id, "notexample.com", BrowserActionClass.read).granted
    assert not svc.check(
        bs.id, "example.com.evil.com", BrowserActionClass.navigate
    ).granted


def test_send_follow_link_subdomain_reaches_broker(session):
    # A same-site subdomain href rides the registrable-host grant — it must
    # reach the broker (timeout with no poller), NOT be refused as an
    # unapproved tab. This is the divergence Finding 2 fixed: Electron
    # would allow it, so the server must too.
    bs = _make_session(session, domain="example.com")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(bs.conversation_id, "follow_link", href="https://shop.example.com/a")
    )
    assert verdict.result_code == ResultCode.timeout


def test_send_follow_link_lookalike_suffix_refused(session):
    # bank.co.uk.evil.com must NOT ride a bank.co.uk grant.
    bs = _make_session(session, domain="bank.co.uk")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(
            bs.conversation_id, "follow_link", href="https://bank.co.uk.evil.com/x"
        )
    )
    assert verdict.result_code == ResultCode.unapproved_tab
    assert broker.pending_count(str(bs.id)) == 0


def test_send_ipv6_href_matches_ipv6_grant(session):
    # host_only must not mangle the IPv6 literal; [::1]:8080 rides a ::1
    # grant.
    bs = _make_session(session, domain="::1")
    broker = BridgeCommandService(default_timeout_s=0.1)
    bc = BridgeClient(session, broker=broker)
    verdict = asyncio.run(
        bc.send(bs.conversation_id, "follow_link", href="http://[::1]:8080/x")
    )
    assert verdict.result_code == ResultCode.timeout


def test_permission_check_suffix_grant_covers_only_exact_host(session):
    # A grant that is itself a public/private suffix (approved tab was
    # literally at that host; Electron fell back to the exact host) covers
    # ONLY that exact host — foo.github.io's registrable host is
    # foo.github.io, not github.io, so Electron refuses it and so must we.
    bs = _make_session(session, domain="github.io")
    from cowork.services.browser.permissions import BrowserPermissionService

    svc = BrowserPermissionService(session)
    assert svc.check(bs.id, "github.io", BrowserActionClass.read).granted
    assert not svc.check(bs.id, "foo.github.io", BrowserActionClass.read).granted
    assert not svc.check(
        bs.id, "foo.github.io", BrowserActionClass.navigate
    ).granted


# ── idempotent stop tokens (/control/stop stop_id) ───────────────────
def _resume(conv_id: str) -> str:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        sess = BrowserControlService(s).resume_on_new_turn(conv_id)
        return sess.control_state


def test_control_stop_new_stop_id_stops(session):
    conv_id = _create_conv()
    r = client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "stop-1"},
    )
    assert r.status_code == 200
    assert r.json()["stopped"] is True
    assert r.json()["control_state"] == "stopped"


def test_control_stop_same_stop_id_after_resume_is_pure_ack(session):
    # stop → fresh user turn resumes → the poller's ack (same stop_id) must
    # NOT re-stop the freshly-resumed session.
    conv_id = _create_conv()
    client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "stop-ack"},
    )
    assert _resume(conv_id) == "active"
    r = client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "stop-ack"},
    )
    assert r.status_code == 200
    assert r.json()["stopped"] is False
    assert r.json()["control_state"] == "active"
    r2 = client.get("/api/v1/browse/status", params={"conversation_id": conv_id})
    assert r2.json()["control_state"] == "active"


def test_control_stop_fresh_stop_id_after_resume_stops_again(session):
    # A genuinely NEW user stop (different stop_id) after a resume must
    # apply — only the SAME token is an ack.
    conv_id = _create_conv()
    client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "stop-a"},
    )
    assert _resume(conv_id) == "active"
    r = client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "stop-b"},
    )
    assert r.json()["stopped"] is True
    assert r.json()["control_state"] == "stopped"


def test_control_stop_without_stop_id_keeps_legacy_behavior(session):
    # Legacy/curl callers without a stop_id always stop, even repeatedly
    # and even right after a resume.
    conv_id = _create_conv()
    client.post("/api/v1/browse/control/stop", json={"conversation_id": conv_id})
    assert _resume(conv_id) == "active"
    r = client.post("/api/v1/browse/control/stop", json={"conversation_id": conv_id})
    assert r.json()["stopped"] is True
    assert r.json()["control_state"] == "stopped"


def test_control_stop_pure_ack_does_not_drain(monkeypatch):
    # A pure ack must not drain the broker queue; a genuinely new stop does.
    import cowork.api.v1.endpoints.compat.stubs as stubs

    calls = []

    async def _fake_drain(sess):
        calls.append(sess)

    monkeypatch.setattr(stubs, "_drain_gated_session", _fake_drain)
    conv_id = _create_conv()
    client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "stop-drain"},
    )
    assert len(calls) == 1
    _resume(conv_id)
    client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "stop-drain"},
    )
    assert len(calls) == 1  # ack did NOT drain
    client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "stop-drain-2"},
    )
    assert len(calls) == 2  # new stop drains again


def test_control_stop_delayed_ack_for_older_token_is_pure_ack(monkeypatch):
    # Stop A (t1) → resume → Stop B (t2) → resume → DELAYED ack for t1: t1
    # is still in the recent-token history, so it must be a pure ack — no
    # re-stop, no drain.
    import cowork.api.v1.endpoints.compat.stubs as stubs

    calls = []

    async def _fake_drain(sess):
        calls.append(sess)

    monkeypatch.setattr(stubs, "_drain_gated_session", _fake_drain)
    conv_id = _create_conv()
    client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "t1"},
    )
    assert _resume(conv_id) == "active"
    client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "t2"},
    )
    assert _resume(conv_id) == "active"
    assert len(calls) == 2  # both real stops drained
    r = client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "t1"},
    )
    assert r.json()["stopped"] is False
    assert r.json()["control_state"] == "active"
    assert len(calls) == 2  # delayed ack did NOT drain


def test_control_stop_token_history_is_bounded(session):
    # The per-conversation history is capped (_STOP_ID_HISTORY = 16, FIFO):
    # after 17 distinct stops the 1st token is evicted, so its late ack
    # redundantly re-stops (acceptable); a still-remembered token stays a
    # pure ack. Distinct new tokens always stop.
    from cowork.services.browser.control import _STOP_ID_HISTORY, _applied_stop_ids

    conv_id = _create_conv()
    n = _STOP_ID_HISTORY + 1  # 17
    for i in range(n):
        r = client.post(
            "/api/v1/browse/control/stop",
            json={"conversation_id": conv_id, "stop_id": f"tok-{i}"},
        )
        assert r.json()["stopped"] is True  # distinct new tokens still stop
        assert _resume(conv_id) == "active"
    recent = _applied_stop_ids[conv_id]
    assert len(recent) == _STOP_ID_HISTORY
    assert "tok-0" not in recent  # 17th evicted the 1st
    assert f"tok-{n - 1}" in recent
    # Evicted token: treated as a new stop (redundant re-stop, acceptable).
    r = client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": "tok-0"},
    )
    assert r.json()["stopped"] is True
    assert _resume(conv_id) == "active"
    # Remembered token: still a pure ack.
    r = client.post(
        "/api/v1/browse/control/stop",
        json={"conversation_id": conv_id, "stop_id": f"tok-{n - 1}"},
    )
    assert r.json()["stopped"] is False
    assert r.json()["control_state"] == "active"
