from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from cowork.models.approval import Approval
from cowork.models.approval_token import ApprovalToken
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.schemas.approvals import ActionDescriptorV1, AuthDescriptorV1
from cowork.services.approvals import (
    ApprovalService,
    register_executor,
)
from cowork.services.projects import GENERAL_PROJECT_ID

TOOL = "test_send"
EXECUTOR_CALLS: list[tuple[dict, str]] = []


def _executor(session: Session, args: dict, raw_token: str) -> dict:
    EXECUTOR_CALLS.append((dict(args), raw_token))
    # Every real executor must spend its token through the service — mirror that.
    ApprovalService(session).consume_token(raw_token, tool=TOOL, args=args)
    return {"sent": True, "via": "stub"}


@pytest.fixture(autouse=True)
def _stub_executor():
    EXECUTOR_CALLS.clear()
    register_executor(TOOL, _executor)
    yield
    EXECUTOR_CALLS.clear()


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    s = Session(engine)
    s.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    s.commit()
    yield s
    s.close()


@pytest.fixture()
def conversation(session: Session) -> Conversation:
    conv = Conversation(project_id=GENERAL_PROJECT_ID, topic="approvals")
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


def _descriptor(**overrides) -> ActionDescriptorV1:
    base = dict(tool=TOOL, args={"text": "original draft", "snapshot_v": 7}, summary="Send the reply")
    base.update(overrides)
    return ActionDescriptorV1(**base)


def test_create_list_get(session, conversation):
    service = ApprovalService(session)
    approval = service.create(conversation_id=conversation.id, descriptor=_descriptor(), draft="hello")

    assert approval.status == "pending"
    assert approval.ttl_seconds == 72 * 3600
    assert service.get(approval.id).id == approval.id

    pending = service.list(status="pending")
    assert [a.id for a in pending] == [approval.id]
    by_conv = service.list(conversation_id=conversation.id)
    assert [a.id for a in by_conv] == [approval.id]
    assert service.list(status="approved") == []
    with pytest.raises(ValueError, match="Approval not found"):
        service.get(GENERAL_PROJECT_ID)  # any non-approval UUID


def test_approve_executes_once_with_exact_payload_and_consumes_token(session, conversation):
    service = ApprovalService(session)
    approval = service.create(conversation_id=conversation.id, descriptor=_descriptor(), draft="hello")

    resolved, executed = service.resolve(approval.id, resolution="approved")
    assert executed is True
    assert resolved.status == "approved"
    assert resolved.resolved_at is not None
    assert resolved.receipt["sent"] is True
    assert resolved.receipt["executed"] is True

    # Payload fidelity: the executor saw EXACTLY the approved args, once.
    assert EXECUTOR_CALLS == [({"text": "original draft", "snapshot_v": 7}, EXECUTOR_CALLS[0][1])]
    # Token single-use: the executor already spent it; spending again fails.
    with pytest.raises(ValueError, match="already consumed"):
        service.consume_token(EXECUTOR_CALLS[0][1], tool=TOOL, args={"text": "original draft", "snapshot_v": 7})

    # Idempotent double-resolve: no re-execution, same state back.
    again, executed_again = service.resolve(approval.id, resolution="approved")
    assert executed_again is False
    assert again.status == "approved"
    assert len(EXECUTOR_CALLS) == 1


def test_edited_resolution_substitutes_draft_and_marks_edited(session, conversation):
    service = ApprovalService(session)
    approval = service.create(conversation_id=conversation.id, descriptor=_descriptor())

    resolved, executed = service.resolve(approval.id, resolution="edited", edited_draft="rewritten by user")
    assert executed is True
    assert resolved.status == "edited"
    assert EXECUTOR_CALLS[0][0]["text"] == "rewritten by user"
    assert EXECUTOR_CALLS[0][0]["snapshot_v"] == 7  # untouched


def test_edited_resolution_requires_a_draft(session, conversation):
    service = ApprovalService(session)
    approval = service.create(conversation_id=conversation.id, descriptor=_descriptor())
    with pytest.raises(ValueError, match="edited_draft"):
        service.resolve(approval.id, resolution="edited")


def test_skip_never_executes(session, conversation):
    service = ApprovalService(session)
    approval = service.create(conversation_id=conversation.id, descriptor=_descriptor())

    resolved, executed = service.resolve(approval.id, resolution="skipped")
    assert executed is True  # a real resolution happened (status transition)
    assert resolved.status == "skipped"
    assert EXECUTOR_CALLS == []


def test_auth_cards_park_without_execution(session, conversation):
    service = ApprovalService(session)
    approval = service.create(
        conversation_id=conversation.id,
        descriptor=AuthDescriptorV1(app_name="Gmail", tab_id="tab-9"),
    )

    resolved, executed = service.resolve(approval.id, resolution="approved")
    assert executed is True
    assert resolved.status == "approved"
    assert resolved.receipt == {
        "executed": False,
        "handed_to_user": "Gmail",
        "resolved_at": resolved.receipt["resolved_at"],
    }
    assert EXECUTOR_CALLS == []


def test_expired_pending_cannot_resolve_and_sweep_marks_expired(session, conversation):
    service = ApprovalService(session)
    approval = service.create(
        conversation_id=conversation.id,
        descriptor=_descriptor(),
        ttl_seconds=1,
    )
    # Age it past the TTL by hand.
    approval.expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    session.add(approval)
    session.commit()

    resolved, executed = service.resolve(approval.id, resolution="approved")
    assert executed is False
    assert resolved.status == "expired"
    assert EXECUTOR_CALLS == []

    # And the sweep settles overdue pendings in bulk.
    another = service.create(conversation_id=conversation.id, descriptor=_descriptor(), ttl_seconds=1)
    another.expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    session.add(another)
    session.commit()
    assert service.sweep_expired() == 1
    assert service.get(another.id).status == "expired"
    assert service.sweep_expired() == 0  # nothing left overdue


def test_token_payload_mismatch_and_expiry(session, conversation):
    service = ApprovalService(session)
    approval = service.create(conversation_id=conversation.id, descriptor=_descriptor())
    resolved, _ = service.resolve(approval.id, resolution="approved")
    raw = EXECUTOR_CALLS[0][1]  # consumed already by the stub — use fresh ones below
    assert raw

    # Fresh token via a second approval, left unconsumed by reaching into issue.
    approval2 = service.create(conversation_id=conversation.id, descriptor=_descriptor())
    raw2 = service._issue_token(approval2, tool=TOOL, args={"text": "x", "snapshot_v": 1})

    with pytest.raises(ValueError, match="payload mismatch"):
        service.consume_token(raw2, tool=TOOL, args={"text": "DIFFERENT", "snapshot_v": 1})
    with pytest.raises(ValueError, match="payload mismatch"):
        service.consume_token(raw2, tool="other_tool", args={"text": "x", "snapshot_v": 1})
    with pytest.raises(ValueError, match="invalid approval token"):
        service.consume_token("nope", tool=TOOL, args={"text": "x", "snapshot_v": 1})

    token_row = session.exec(select(ApprovalToken)).all()[-1]
    token_row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    session.add(token_row)
    session.commit()
    with pytest.raises(ValueError, match="expired"):
        service.consume_token(raw2, tool=TOOL, args={"text": "x", "snapshot_v": 1})


def test_transcript_events_document_the_lifecycle(session, conversation):
    from cowork.models.message_event import MessageEvent
    from sqlmodel import select

    service = ApprovalService(session)
    approval = service.create(conversation_id=conversation.id, descriptor=_descriptor())
    service.resolve(approval.id, resolution="approved")

    events = session.exec(select(MessageEvent)).all()
    types = [e.event_data.get("type") for e in events]
    assert "response.approval_requested" in types
    assert "response.approval_resolved" in types
    resolved_event = next(e for e in events if e.event_data.get("type") == "response.approval_resolved")
    assert resolved_event.event_data["approval"]["status"] == "approved"
