from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from cowork.models.approval import Approval
from cowork.models.project import Project
from cowork.models.schedule import Schedule
from cowork.services.digest import DIGEST_TITLE
from cowork.services.onboarding import (
    DIGEST_CARD_TOOL,
    PIN_CARD_TOOL,
    ensure_onboarding_cards,
)
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    s = Session(engine)
    s.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    s.commit()
    yield s
    s.close()


def _cards(session, tool):
    return [a for a in session.exec(select(Approval)).all()
            if (a.action_descriptor or {}).get("tool") == tool]


def test_empty_world_seeds_card1_only(session):
    result = ensure_onboarding_cards(session, pinned_apps=[])
    cards = _cards(session, PIN_CARD_TOOL)
    assert len(cards) == 1
    assert cards[0].status == "pending"
    assert cards[0].action_descriptor["summary"] == "Pin the tools you work in"
    assert result["pin_card"] == str(cards[0].id)
    # Card 3 waits for card 1's resolution.
    assert result["digest_card"] is None
    assert _cards(session, DIGEST_CARD_TOOL) == []


def test_established_user_gets_nothing(session):
    result = ensure_onboarding_cards(session, pinned_apps=[{"id": "app-gmail", "name": "Gmail"}])
    assert result == {"pin_card": None, "digest_card": None}
    assert session.exec(select(Approval)).all() == []


def test_idempotent_second_call(session):
    first = ensure_onboarding_cards(session, pinned_apps=[])
    second = ensure_onboarding_cards(session, pinned_apps=[])
    assert first == second
    assert len(_cards(session, PIN_CARD_TOOL)) == 1


def test_card3_follows_card1_resolution(session):
    ensure_onboarding_cards(session, pinned_apps=[])
    card1 = _cards(session, PIN_CARD_TOOL)[0]
    card1.status = "approved"
    session.add(card1)
    session.commit()

    result = ensure_onboarding_cards(session, pinned_apps=[{"id": "app-gmail"}])
    cards3 = _cards(session, DIGEST_CARD_TOOL)
    assert len(cards3) == 1
    assert cards3[0].action_descriptor["args"] == {"hour": 9, "timezone": "UTC"}
    assert result["digest_card"] == str(cards3[0].id)


def test_card3_not_seeded_when_digest_schedule_exists(session):
    ensure_onboarding_cards(session, pinned_apps=[])
    card1 = _cards(session, PIN_CARD_TOOL)[0]
    card1.status = "skipped"
    session.add(card1)
    session.commit()
    session.add(Schedule(
        title=DIGEST_TITLE, prompt="x", cadence="daily", timezone="UTC",
        next_run_at=card1.expires_at, enabled=True, project_id=GENERAL_PROJECT_ID, model="default",
    ))
    session.commit()

    result = ensure_onboarding_cards(session, pinned_apps=[])
    assert result["digest_card"] is None
    assert _cards(session, DIGEST_CARD_TOOL) == []


def test_digest_executor_writes_the_real_schedule(session):
    from cowork.services.approvals import ApprovalService

    ensure_onboarding_cards(session, pinned_apps=[])
    card1 = _cards(session, PIN_CARD_TOOL)[0]
    card1.status = "approved"
    session.add(card1)
    session.commit()
    ensure_onboarding_cards(session, pinned_apps=[])
    card3 = _cards(session, DIGEST_CARD_TOOL)[0]

    resolved, executed = ApprovalService(session).resolve(card3.id, resolution="approved")
    assert executed is True
    assert resolved.status == "approved"
    assert resolved.receipt["schedule_id"]
    assert resolved.receipt["summary"] == "See you at 9:00."
    schedule = session.exec(select(Schedule).where(Schedule.title == DIGEST_TITLE)).one()
    assert schedule.requires_browser is True
