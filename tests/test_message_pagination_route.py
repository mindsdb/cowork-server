"""ENG-2768: GET /conversations/{id}/items route-level dispatch — omitting
both `limit`/`before` must stay byte-identical to today's bare list (the
existing consumer this preserves is asserted directly in
test_conversation_model_persistence.py/test_org_isolation_e2e.py); passing
either opts into the new envelope. Service-level pagination correctness is
covered in test_message_pagination.py; this file only checks the route's
own wiring and HTTP status mapping.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.schemas.responses import Role
from cowork.server import create_app


def _make_conversation_with_messages(count):
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = Project(name=f"route-test-{count}", path=f"/tmp/route-test-{count}")
        session.add(project)
        session.commit()
        session.refresh(project)
        conv = Conversation(project_id=project.id, topic="t")
        session.add(conv)
        session.commit()
        session.refresh(conv)
        for i in range(count):
            role = Role.user if i % 2 == 0 else Role.assistant
            session.add(Message(conversation_id=conv.id, role=role, content=f"m{i}", seq=i))
        session.commit()
        return conv.id


client = TestClient(create_app())


def test_no_params_returns_the_bare_unbounded_list():
    conv_id = _make_conversation_with_messages(3)
    r = client.get(f"/api/v1/conversations/{conv_id}/items")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 3


def test_limit_param_returns_the_paginated_envelope():
    conv_id = _make_conversation_with_messages(5)
    r = client.get(f"/api/v1/conversations/{conv_id}/items", params={"limit": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, dict)
    assert set(body.keys()) == {"items", "hasMore", "nextBefore"}
    assert len(body["items"]) == 2
    assert body["hasMore"] is True
    assert body["nextBefore"] is not None
    # Item dicts keep their existing (non-camelCased) field names.
    assert "created_at" in body["items"][0]
    assert "createdAt" not in body["items"][0]


def test_before_param_alone_also_opts_into_the_envelope():
    conv_id = _make_conversation_with_messages(2)
    first = client.get(f"/api/v1/conversations/{conv_id}/items", params={"limit": 1}).json()
    r = client.get(
        f"/api/v1/conversations/{conv_id}/items", params={"before": first["nextBefore"]}
    )
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), dict)


def test_malformed_cursor_is_a_400_not_a_500():
    conv_id = _make_conversation_with_messages(1)
    r = client.get(
        f"/api/v1/conversations/{conv_id}/items", params={"before": "not-a-real-cursor"}
    )
    assert r.status_code == 400, r.text


def test_oversized_limit_is_a_400_not_a_500():
    conv_id = _make_conversation_with_messages(1)
    r = client.get(f"/api/v1/conversations/{conv_id}/items", params={"limit": 100_000})
    assert r.status_code == 400, r.text


def test_unknown_conversation_is_a_404_on_the_paginated_branch():
    r = client.get(
        "/api/v1/conversations/00000000-0000-0000-0000-000000000000/items",
        params={"limit": 5},
    )
    assert r.status_code == 404, r.text
