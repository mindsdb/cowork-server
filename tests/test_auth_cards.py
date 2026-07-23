from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

import cowork.harnesses.anton_harness.browser_tools as bt
from cowork.models.approval import Approval
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services.projects import GENERAL_PROJECT_ID

READ_CALLED = []


async def _fake_bridge(method: str, path: str, *, params=None, body=None, timeout=10.0):
    if path == "/state":
        return {
            "activeTabId": "tab-gmail",
            "tabs": [
                {"id": "tab-gmail", "title": "Gmail", "url": "https://accounts.google.com/signin", "needsAuth": True},
                {"id": "tab-linear", "title": "Linear", "url": "https://linear.app", "needsAuth": False},
            ],
        }
    if path == "/read":
        READ_CALLED.append(1)
        return {"url": "https://linear.app", "title": "Linear", "text": "work work work"}
    return {"ok": True}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    READ_CALLED.clear()
    monkeypatch.setattr(bt, "_bridge_call", _fake_bridge)
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    session.commit()
    conv = Conversation(project_id=GENERAL_PROJECT_ID, topic="auth cards")
    session.add(conv)
    session.commit()
    session.refresh(conv)

    from cowork.db import session as db_session_module

    monkeypatch.setattr(db_session_module, "get_open_session", lambda: Session(engine))
    yield session, conv
    session.close()


def _chat(conv) -> SimpleNamespace:
    return SimpleNamespace(_session_id=str(conv.id))


async def test_needs_auth_tab_parks_an_auth_card(_env):
    session, conv = _env
    result = await bt._browser_read(_chat(conv), {})
    assert result.startswith("PAUSED — sign-in needed")
    assert "Gmail" in result
    assert READ_CALLED == []  # the read never ran

    approval = session.exec(select(Approval)).one()
    assert approval.kind == "auth"
    assert approval.status == "pending"
    assert approval.action_descriptor["app_name"] == "Gmail"
    assert approval.action_descriptor["tab_id"] == "tab-gmail"


async def test_auth_cards_dedupe_per_tab(_env):
    session, conv = _env
    await bt._browser_read(_chat(conv), {})
    await bt._browser_snapshot(_chat(conv), {})
    await bt._browser_click(_chat(conv), {"index": 1})
    approvals = session.exec(select(Approval)).all()
    assert len(approvals) == 1  # three tools, one card


async def test_clean_tab_proceeds(_env):
    session, conv = _env
    result = await bt._browser_read(_chat(conv), {"tab_id": "tab-linear"})
    assert result.startswith('<untrusted-page-content')
    assert READ_CALLED == [1]
    assert session.exec(select(Approval)).all() == []


async def test_navigate_is_not_walled(_env):
    session, conv = _env
    # The sign-in path itself must stay open — navigate is how the user (and
    # the agent pointing at the SSO page) gets there.
    result = await bt._browser_navigate(_chat(conv), {"url": "https://linear.app"})
    assert "PAUSED" not in result
    assert session.exec(select(Approval)).all() == []
