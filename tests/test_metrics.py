from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from cowork.models.approval import Approval
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services.metrics import approval_metrics
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    s = Session(engine)
    s.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    s.commit()
    conv = Conversation(project_id=GENERAL_PROJECT_ID, topic="metrics")
    s.add(conv)
    s.commit()
    s.refresh(conv)
    yield s, conv
    s.close()


def _row(conv, status, *, created_hours_ago=2, resolved=True):
    now = datetime.now(timezone.utc)
    return Approval(
        conversation_id=conv.id,
        kind="action",
        status=status,
        action_descriptor={},
        created_at=now - timedelta(hours=created_hours_ago),
        resolved_at=(now - timedelta(hours=created_hours_ago - 1)) if resolved else None,
        expires_at=now + timedelta(hours=72),
    )


def test_metrics_composition(session):
    s, conv = session
    s.add(_row(conv, "approved"))
    s.add(_row(conv, "approved"))
    s.add(_row(conv, "edited"))
    s.add(_row(conv, "skipped"))
    s.add(_row(conv, "pending", resolved=False))
    s.commit()

    m = approval_metrics(s)
    assert m["shipped"] == 3
    assert m["needsYou"] == 2
    assert m["autonomyRatio"] == 1.5
    assert m["editRate"] == round(1 / 4, 3)
    assert m["skipRate"] == round(1 / 4, 3)
    # All four resolved rows sat ~1 hour → median 3600s.
    assert m["medianTimeToResolveSeconds"] == 3600.0
    assert isinstance(m["injectionTripwireHits"], dict)
    assert isinstance(m["gateQuality"], dict)


def test_empty_world_is_nulls_and_zeroes_not_crashes(session):
    s, _ = session
    m = approval_metrics(s)
    assert m["shipped"] == 0
    assert m["needsYou"] == 0
    assert m["autonomyRatio"] is None
    assert m["medianTimeToResolveSeconds"] is None


def test_gate_hits_counter_flows_through(session, monkeypatch):
    import cowork.harnesses.anton_harness.browser_tools as bt

    bt.GATE_HITS.clear()
    bt._gate_hit("browser_click", "parked")
    bt._gate_hit("browser_click", "token_rejected")
    s, _ = session
    m = approval_metrics(s)
    assert m["gateQuality"]["browser_click"] == {"parked": 1, "token_rejected": 1}
    bt.GATE_HITS.clear()
