from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from cowork.models.approval import Approval
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.schemas.approvals import (
    ActionDescriptorV1,
    ApprovalCreateRequest,
    AuthDescriptorV1,
    parse_descriptor,
)
from cowork.services.projects import GENERAL_PROJECT_ID


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    session.commit()
    return session


def _conversation(session: Session) -> Conversation:
    conv = Conversation(project_id=GENERAL_PROJECT_ID, topic="approvals test")
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


def _action_approval(conversation_id, **overrides) -> Approval:
    descriptor = ActionDescriptorV1(
        tool="browser_click",
        args={"index": 42, "snapshot_v": 3},
        summary="Send the drafted reply",
    )
    return Approval(
        conversation_id=conversation_id,
        kind="action",
        action_descriptor=descriptor.model_dump(),
        draft="Thanks Abi — scope looks right.",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=72),
        **overrides,
    )


def test_action_approval_round_trips_with_defaults():
    session = _session()
    conv = _conversation(session)

    approval = _action_approval(conv.id)
    session.add(approval)
    session.commit()
    session.refresh(approval)

    assert approval.status == "pending"
    assert approval.ttl_seconds == 259200  # 72h
    assert approval.receipt is None
    assert approval.resolved_at is None
    assert approval.draft == "Thanks Abi — scope looks right."
    assert approval.created_at is not None

    # The descriptor parses back through the versioned union, discriminator intact.
    parsed = parse_descriptor(approval.action_descriptor)
    assert isinstance(parsed, ActionDescriptorV1)
    assert parsed.version == 1
    assert parsed.tool == "browser_click"
    assert parsed.args == {"index": 42, "snapshot_v": 3}
    session.close()


def test_auth_approval_round_trips():
    session = _session()
    conv = _conversation(session)

    descriptor = AuthDescriptorV1(app_name="Gmail", tab_id="tab-9", reason="needsAuth on inbox read")
    approval = Approval(
        conversation_id=conv.id,
        kind="auth",
        action_descriptor=descriptor.model_dump(),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )
    session.add(approval)
    session.commit()
    session.refresh(approval)

    parsed = parse_descriptor(approval.action_descriptor)
    assert isinstance(parsed, AuthDescriptorV1)
    assert parsed.app_name == "Gmail"
    assert parsed.tab_id == "tab-9"
    session.close()


def test_create_request_accepts_camel_and_snake():
    conv_id = GENERAL_PROJECT_ID  # any UUID parses; FK isn't exercised by the schema
    descriptor = {
        "version": 1,
        "kind": "action",
        "tool": "browser_click",
        "args": {"index": 1},
        "summary": "Click send",
    }
    snake = ApprovalCreateRequest(
        conversation_id=conv_id, kind="action", descriptor=descriptor, ttl_seconds=60
    )
    camel = ApprovalCreateRequest(
        conversationId=conv_id, kind="action", descriptor=descriptor, ttlSeconds=60
    )
    assert snake.ttl_seconds == camel.ttl_seconds == 60
    assert isinstance(camel.descriptor, ActionDescriptorV1)


def test_descriptor_rejects_unknown_version_and_kind():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ApprovalCreateRequest(
            conversation_id=GENERAL_PROJECT_ID,
            kind="action",
            descriptor={"version": 2, "kind": "action", "tool": "x"},
        )
    with pytest.raises(ValidationError):
        ApprovalCreateRequest(
            conversation_id=GENERAL_PROJECT_ID,
            kind="action",
            descriptor={"version": 1, "kind": "teleport", "tool": "x"},
        )


def test_response_serves_camelcase_descriptor():
    from cowork.schemas.approvals import ApprovalResponse

    session = _session()
    conv = _conversation(session)
    approval = _action_approval(conv.id)
    session.add(approval)
    session.commit()
    session.refresh(approval)

    data = ApprovalResponse.serialize(approval)
    assert data["actionDescriptor"]["tool"] == "browser_click"
    assert data["actionDescriptor"]["summary"] == "Send the drafted reply"
    # args is an opaque payload — passed through untouched
    assert data["actionDescriptor"]["args"] == {"index": 42, "snapshot_v": 3}

    auth = Approval(
        conversation_id=conv.id,
        kind="auth",
        action_descriptor=AuthDescriptorV1(app_name="Gmail", tab_id="tab-9").model_dump(),
        expires_at=approval.expires_at,
    )
    session.add(auth)
    session.commit()
    session.refresh(auth)
    auth_data = ApprovalResponse.serialize(auth)
    assert auth_data["actionDescriptor"]["appName"] == "Gmail"
    assert auth_data["actionDescriptor"]["tabId"] == "tab-9"
    assert "app_name" not in auth_data["actionDescriptor"]
    session.close()
