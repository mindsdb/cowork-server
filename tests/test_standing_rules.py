from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

import cowork.harnesses.anton_harness.browser_tools as bt
from cowork.models.approval import Approval
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.standing_rule import StandingRule
from cowork.schemas.approvals import ActionDescriptorV1
from cowork.services.approvals import ApprovalService, register_executor
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.rules import RuleService, normalize_label, scope_of

STATE = {
    "activeTabId": "t1",
    "tabs": [{"id": "t1", "title": "Inbox", "url": "https://mail.google.com/mail/u/0/#inbox"}],
}


async def _fake_bridge(method: str, path: str, *, params=None, body=None, timeout=10.0):
    if path == "/state":
        return STATE
    if path == "/snapshot":
        return {"v": 4, "url": "https://mail.google.com/mail/u/0/#inbox",
                "elements": [{"index": 3, "text": "[!] Send", "consequential": True}]}
    return {"ok": True}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(bt, "_bridge_call", _fake_bridge)
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    session.commit()
    conv = Conversation(project_id=GENERAL_PROJECT_ID, topic="rules")
    session.add(conv)
    session.commit()
    session.refresh(conv)
    from cowork.db import session as db_session_module

    monkeypatch.setattr(db_session_module, "get_open_session", lambda: Session(engine))
    bt._register_gate_executors()
    yield session, conv
    session.close()


def _chat(conv):
    return SimpleNamespace(_session_id=str(conv.id))


def test_scope_and_label_normalization():
    assert normalize_label("[!] Send") == "send"
    assert normalize_label("  Send  NOW ") == "send now"
    assert scope_of(origin="MAIL.google.com ", tool="browser_click", label="[!] Send") == (
        "mail.google.com",
        "browser_click:send",
    )


def test_grant_dedupe_revoke_and_matching(_env):
    session, _ = _env
    rules = RuleService(session)
    r1 = rules.grant(origin="mail.google.com", action_kind="browser_click:send", source_approval_id=GENERAL_PROJECT_ID)
    r2 = rules.grant(origin="mail.google.com", action_kind="browser_click:send", source_approval_id=GENERAL_PROJECT_ID)
    assert r1.id == r2.id  # one active rule per scope

    assert rules.matching(origin="mail.google.com", action_kind="browser_click:send") is not None
    assert rules.matching(origin="mail.google.com", action_kind="browser_click:delete") is None

    rules.record_hit(r1)
    rules.record_hit(r1)
    assert session.get(StandingRule, r1.id).hit_count == 2

    rules.revoke(r1.id)
    assert rules.matching(origin="mail.google.com", action_kind="browser_click:send") is None  # checked at act time
    assert len(rules.list()) == 0
    assert len(rules.list(include_revoked=True)) == 1


def _scoped_approval(session, conv, *, status="approved") -> Approval:
    ap = Approval(
        conversation_id=conv.id,
        kind="action",
        status=status,
        action_descriptor=ActionDescriptorV1(
            tool="browser_click",
            args={"index": 3, "origin": "mail.google.com"},
            summary="Send",
        ).model_dump(),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=72),
    )
    session.add(ap)
    session.commit()
    return ap


def test_evidence_gate_counts_only_unmodified_approvals(_env):
    session, conv = _env
    rules = RuleService(session)
    for _ in range(2):
        _scoped_approval(session, conv, status="approved")
    assert not rules.eligible_for_always(origin="mail.google.com", action_kind="browser_click:send")
    _scoped_approval(session, conv, status="edited")  # edited proves nothing
    assert not rules.eligible_for_always(origin="mail.google.com", action_kind="browser_click:send")
    _scoped_approval(session, conv, status="approved")
    assert rules.eligible_for_always(origin="mail.google.com", action_kind="browser_click:send")


async def test_gate_bypasses_on_matching_rule_and_counts_the_hit(_env):
    session, conv = _env
    RuleService(session).grant(
        origin="mail.google.com", action_kind="browser_click:send", source_approval_id=GENERAL_PROJECT_ID
    )
    result = await bt._browser_click(_chat(conv), {"index": 3, "tab_id": "t1"})
    assert result.startswith("Clicked element [3]")  # straight through, no proposal
    assert session.exec(select(Approval)).all() == []
    rule = session.exec(select(StandingRule)).one()
    assert rule.hit_count == 1


async def test_gate_parks_with_origin_stamped(_env):
    session, conv = _env
    result = await bt._browser_click(_chat(conv), {"index": 3, "tab_id": "t1"})
    assert result.startswith("PAUSED for approval")
    approval = session.exec(select(Approval)).one()
    assert approval.action_descriptor["args"]["origin"] == "mail.google.com"


async def test_revoked_rule_re_gates(_env):
    session, conv = _env
    rules = RuleService(session)
    rule = rules.grant(origin="mail.google.com", action_kind="browser_click:send", source_approval_id=GENERAL_PROJECT_ID)
    rules.revoke(rule.id)
    result = await bt._browser_click(_chat(conv), {"index": 3, "tab_id": "t1"})
    assert result.startswith("PAUSED for approval")


def test_resolve_always_grants_and_executes_once_evidence_exists(_env):
    session, conv = _env
    def _click_exec(s, a, t):
        ApprovalService(s).consume_token(t, tool="browser_click", args=a)
        return {"clicked": True}

    register_executor("browser_click", _click_exec)
    for _ in range(3):
        _scoped_approval(session, conv, status="approved")

    service = ApprovalService(session)
    target = _scoped_approval(session, conv, status="pending")
    resolved, executed = service.resolve(target.id, resolution="always")
    assert executed is True
    assert resolved.status == "approved"
    assert session.exec(select(StandingRule)).one().action_kind == "browser_click:send"


def test_resolve_always_under_evidence_400s(_env):
    session, conv = _env
    service = ApprovalService(session)
    target = _scoped_approval(session, conv, status="pending")
    import pytest as _pytest

    with _pytest.raises(ValueError, match="3 identical"):
        service.resolve(target.id, resolution="always")
