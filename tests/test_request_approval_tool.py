from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from cowork.harnesses.anton_harness.tools import (
    _REQUEST_APPROVAL_PROMPT,
    _cowork_request_approval,
    build_cowork_request_approval_tool,
)
from cowork.models.approval import Approval
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.schemas.approvals import ActionDescriptorV1, parse_descriptor
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture()
def db_session(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    session.commit()
    conv = Conversation(project_id=GENERAL_PROJECT_ID, topic="approval tool")
    session.add(conv)
    session.commit()
    session.refresh(conv)

    # The handler opens its own session via get_open_session — point it at
    # this engine instead of the app database.
    from cowork.db import session as db_session_module

    monkeypatch.setattr(
        db_session_module,
        "get_open_session",
        lambda: Session(engine),
    )
    # create() validates executability — the test tool must have an executor.
    from cowork.services.approvals import register_executor

    register_executor("t", lambda s, a, tok: {"ok": True})
    yield session, conv
    session.close()


def _chat_session(conv) -> SimpleNamespace:
    return SimpleNamespace(_session_id=str(conv.id))


async def test_creates_pending_approval_and_ends_turn(db_session):
    session, conv = db_session
    result = await _cowork_request_approval(
        _chat_session(conv),
        {
            "title": "Send reply to Abi Tedder",
            "summary": "Send the drafted reply",
            "draft": "Thanks Abi — scope looks right.",
            "action": {"tool": "browser_click", "args": {"index": 42, "snapshot_v": 3, "text": "Thanks Abi — scope looks right."}},
        },
    )

    assert "END YOUR TURN" in result
    assert "Send reply to Abi Tedder" in result

    approvals = session.exec(select(Approval)).all()
    assert len(approvals) == 1
    approval = approvals[0]
    assert approval.conversation_id == conv.id
    assert approval.kind == "action"
    assert approval.status == "pending"
    assert approval.draft == "Thanks Abi — scope looks right."
    assert approval.ttl_seconds == 259200

    descriptor = parse_descriptor(approval.action_descriptor)
    assert isinstance(descriptor, ActionDescriptorV1)
    assert descriptor.tool == "browser_click"
    assert descriptor.args["index"] == 42
    assert descriptor.summary == "Send the drafted reply"


async def test_validation_errors_return_strings_never_raise(db_session):
    session, conv = db_session
    assert "`title` is required" in await _cowork_request_approval(_chat_session(conv), {"summary": "x", "action": {"tool": "t", "args": {}}})
    assert "`summary` is required" in await _cowork_request_approval(_chat_session(conv), {"title": "x", "action": {"tool": "t", "args": {}}})
    assert "`action.tool` is required" in await _cowork_request_approval(_chat_session(conv), {"title": "x", "summary": "y", "action": {"args": {}}})
    assert "`action.args` must be an object" in await _cowork_request_approval(_chat_session(conv), {"title": "x", "summary": "y", "action": {"tool": "t"}})
    assert session.exec(select(Approval)).all() == []


async def test_missing_conversation_context_degrades_gracefully(db_session):
    result = await _cowork_request_approval(
        SimpleNamespace(),  # no _session_id
        {"title": "x", "summary": "y", "action": {"tool": "t", "args": {}}},
    )
    assert "couldn't be queued" in result


async def test_custom_ttl(db_session):
    session, conv = db_session
    await _cowork_request_approval(
        _chat_session(conv),
        {"title": "x", "summary": "y", "action": {"tool": "t", "args": {}}, "ttl_seconds": 600},
    )
    approval = session.exec(select(Approval)).one()
    assert approval.ttl_seconds == 600


def test_tooldef_shape_and_prompt_contract():
    tool = build_cowork_request_approval_tool()
    assert tool.name == "request_approval"
    assert tool.input_schema["required"] == ["title", "summary", "action"]
    assert set(tool.input_schema["properties"]) == {"title", "summary", "draft", "action", "ttl_seconds"}

    # The prompt contract is the behavior half of the unit — pin its load-bearing rules.
    for phrase in (
        "request_approval",
        "send, submit, delete, pay",
        "END YOUR TURN",
        "Never click a send/submit/delete/pay control yourself",
        "never park those",
    ):
        assert phrase in _REQUEST_APPROVAL_PROMPT
