from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

import cowork.harnesses.anton_harness.browser_tools as bt
from cowork.models.approval import Approval
from cowork.models.approval_token import ApprovalToken
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services.approvals import ApprovalService, register_executor
from cowork.services.projects import GENERAL_PROJECT_ID

BRIDGE_CALLS: list[tuple[str, str, dict | None]] = []


async def _fake_bridge(method: str, path: str, *, params=None, body=None, timeout=10.0):
    BRIDGE_CALLS.append((method, path, body))
    if path == "/snapshot":
        return {
            "elements": [
                {"index": 3, "text": "[!] Send", "consequential": True},
                {"index": 4, "text": "Search"},
            ]
        }
    if path in ("/inspect-point", "/inspect-active"):
        return {"found": True, "consequential": True, "text": "[!] Send"}
    return {"ok": True}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    BRIDGE_CALLS.clear()
    monkeypatch.setattr(bt, "_bridge_call", _fake_bridge)
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    session.commit()
    conv = Conversation(project_id=GENERAL_PROJECT_ID, topic="gate")
    session.add(conv)
    session.commit()
    session.refresh(conv)

    from cowork.db import session as db_session_module

    monkeypatch.setattr(db_session_module, "get_open_session", lambda: Session(engine))
    yield session, conv
    session.close()


def _chat(conv) -> SimpleNamespace:
    return SimpleNamespace(_session_id=str(conv.id))


def _bridge_hit(path: str) -> bool:
    return any(call[1] == path for call in BRIDGE_CALLS)


async def test_consequential_click_parks_and_never_reaches_the_bridge(_env):
    session, conv = _env
    result = await bt._browser_click(_chat(conv), {"index": 3, "snapshot_v": 9})

    assert result.startswith("PAUSED for approval")
    assert "END YOUR TURN" in result
    assert not _bridge_hit("/click")

    approval = session.exec(select(Approval)).one()
    assert approval.status == "pending"
    assert approval.action_descriptor["tool"] == "browser_click"
    assert approval.action_descriptor["args"] == {"index": 3, "v": 9}


async def test_safe_click_proceeds(_env):
    session, conv = _env
    result = await bt._browser_click(_chat(conv), {"index": 4})
    assert result.startswith("Clicked element [4]")
    assert _bridge_hit("/click")
    assert session.exec(select(Approval)).all() == []


async def test_gate_disabled_bypasses(monkeypatch, _env):
    session, conv = _env
    monkeypatch.setattr(bt, "_gate_enabled", lambda: False)
    result = await bt._browser_click(_chat(conv), {"index": 3})
    assert result.startswith("Clicked element [3]")
    assert session.exec(select(Approval)).all() == []


async def test_valid_token_passes_once_and_consumes(_env):
    session, conv = _env
    service = ApprovalService(session)
    approval = service.create(
        conversation_id=conv.id,
        descriptor=__import__("cowork.schemas.approvals", fromlist=["ActionDescriptorV1"]).ActionDescriptorV1(
            tool="browser_click", args={"index": 3, "v": 9}, summary="Send"
        ),
    )
    raw = service._issue_token(approval, tool="browser_click", args={"index": 3, "v": 9})

    result = await bt._browser_click(_chat(conv), {"index": 3, "snapshot_v": 9, "approval_token": raw})
    assert result.startswith("Clicked element [3]")
    assert _bridge_hit("/click")

    # Same token again → rejected, no second bridge call.
    BRIDGE_CALLS.clear()
    result2 = await bt._browser_click(_chat(conv), {"index": 3, "snapshot_v": 9, "approval_token": raw})
    assert "approval token rejected" in result2
    assert not _bridge_hit("/click")


async def test_forged_token_is_rejected(_env):
    session, conv = _env
    result = await bt._browser_click(_chat(conv), {"index": 3, "approval_token": "forged"})
    assert "approval token rejected" in result
    assert not _bridge_hit("/click")


async def test_press_key_enter_gated_but_not_shift_enter_or_plain_keys(_env):
    session, conv = _env
    # Bare Enter in a compose → gated.
    r1 = await bt._browser_press_key(_chat(conv), {"key": "enter"})
    assert r1.startswith("PAUSED for approval")
    assert not _bridge_hit("/press")
    # Shift+Enter is a newline — never gate.
    r2 = await bt._browser_press_key(_chat(conv), {"key": "enter", "modifiers": ["shift"]})
    assert r2.startswith("Pressed enter")
    # A plain key → never gate.
    r3 = await bt._browser_press_key(_chat(conv), {"key": "a"})
    assert r3.startswith("Pressed a")
    # Ctrl+Enter (the Gmail send chord) → gated.
    BRIDGE_CALLS.clear()
    r4 = await bt._browser_press_key(_chat(conv), {"key": "enter", "modifiers": ["control"]})
    assert r4.startswith("PAUSED for approval")
    assert not _bridge_hit("/press")


async def test_click_at_gated_via_inspect_point(_env):
    session, conv = _env
    result = await bt._browser_click_at(_chat(conv), {"x": 100, "y": 200})
    assert result.startswith("PAUSED for approval")
    assert not _bridge_hit("/click-at")
    approval = session.exec(select(Approval)).one()
    assert approval.action_descriptor["args"] == {"x": 100, "y": 200}


async def test_executor_replays_exact_approved_call(_env):
    session, conv = _env
    bt._register_gate_executors()
    service = ApprovalService(session)
    descriptor = __import__("cowork.schemas.approvals", fromlist=["ActionDescriptorV1"]).ActionDescriptorV1(
        tool="browser_click", args={"index": 3, "v": 9}, summary="Send"
    )
    approval = service.create(conversation_id=conv.id, descriptor=descriptor)

    resolved, executed = service.resolve(approval.id, resolution="approved")
    assert executed is True
    assert resolved.receipt["executed"] is True
    # The replay hit the bridge with EXACTLY the approved payload.
    assert ("POST", "/click", {"index": 3, "v": 9}) in BRIDGE_CALLS
    # Token spent — a second resolve never re-executes.
    BRIDGE_CALLS.clear()
    again, executed_again = service.resolve(approval.id, resolution="approved")
    assert executed_again is False
    assert not _bridge_hit("/click")


async def test_label_lookup_failure_defers_to_normal_path(monkeypatch, _env):
    session, conv = _env

    async def _failing_probe(method, path, *, params=None, body=None, timeout=10.0):
        if path == "/snapshot":
            raise bt._BridgeUnavailable("down")
        return await _fake_bridge(method, path, params=params, body=body, timeout=timeout)

    monkeypatch.setattr(bt, "_bridge_call", _failing_probe)
    result = await bt._browser_click(_chat(conv), {"index": 3})
    # Gate couldn't see → the action proceeds (and the bridge answers normally).
    assert result.startswith("Clicked element [3]")
    assert _bridge_hit("/click")
