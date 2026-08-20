"""Cross-tenant behaviour of the swept ConversationService.

Covers the request path (org isolation, 404-shaped answers), the detached
producer contract (child writes re-anchored to a parent loaded through the
writer's own scope), and the background-caller rule (local mode = today's
behavior; org mode fails loudly until a service principal exists).
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlmodel import Session, select

import cowork.models.message_event  # noqa: F401 — resolve mappers
from cowork.db.scoped import (
    LOCAL_SCOPE,
    MissingTenantScopeError,
    ScopedSession,
    TenantScope,
    scope_for_background_context,
)
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.services.conversations import ConversationService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _scope(org: str, user: str = "user-1") -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id=user)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    from sqlalchemy.pool import StaticPool
    from sqlmodel import SQLModel, create_engine

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as seed:
        seed.add(Project(name="a-proj", path="/tmp/a", org_id=ORG_A))
        seed.add(Project(name="b-proj", path="/tmp/b", org_id=ORG_B))
        seed.commit()
    yield engine
    get_app_settings.cache_clear()


def _svc(engine, scope: TenantScope) -> ConversationService:
    return ConversationService(ScopedSession(Session(engine), scope))


def _project_id(engine, name: str):
    with Session(engine) as s:
        return s.exec(select(Project).where(Project.name == name)).one().id


def test_creation_stamps_org_and_author(db):
    svc = _svc(db, _scope(ORG_A, "alice"))
    conv = svc.create_conversation(topic="hello", project_id=_project_id(db, "a-proj"))
    assert conv.org_id == ORG_A
    assert conv.created_by == "alice"


def test_creation_persists_harness_and_model(db):
    svc = _svc(db, _scope(ORG_A, "alice"))
    conv = svc.create_conversation(
        topic="hello",
        project_id=_project_id(db, "a-proj"),
        harness="claude-code",
        model="kimi",
    )
    assert conv.harness == "claude-code"
    assert conv.model == "kimi"


def test_creation_defaults_harness_and_model_to_none(db):
    svc = _svc(db, _scope(ORG_A, "alice"))
    conv = svc.create_conversation(topic="hello", project_id=_project_id(db, "a-proj"))
    assert conv.harness is None
    assert conv.model is None


def test_creation_rejects_another_orgs_project(db):
    # Linking to a foreign project would leak its name/path via serialization.
    b = _svc(db, _scope(ORG_B, "bob"))
    with pytest.raises(ValueError, match="not found"):
        b.create_conversation(topic="spy", project_id=_project_id(db, "a-proj"))


def test_move_rejects_another_orgs_project(db):
    a = _svc(db, _scope(ORG_A, "alice"))
    conv = a.create_conversation(topic="t", project_id=_project_id(db, "a-proj"))
    with pytest.raises(ValueError, match="not found"):
        a.update_conversation(conv.id, project_id=_project_id(db, "b-proj"))


def test_other_org_cannot_see_or_touch(db):
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))
    conv = a.create_conversation(topic="secret", project_id=_project_id(db, "a-proj"))

    assert conv.id not in {c.id for c in b.list_conversations(all_projects=True)}
    with pytest.raises(ValueError, match="not found"):
        b.get_conversation(conv.id)
    with pytest.raises(ValueError, match="not found"):
        b.get_messages(conv.id)
    with pytest.raises(ValueError, match="not found"):
        b.update_conversation(conv.id, topic="stolen")
    with pytest.raises(ValueError, match="not found"):
        b.delete_turn(conv.id, 0)
    assert b.delete_conversation(conv.id) is False  # same answer as nonexistent
    assert a.get_conversation(conv.id).topic == "secret"


def test_messages_are_transitively_tenant_safe(db):
    # Messages carry no org_id; their safety comes from the conversation gate.
    a = _svc(db, _scope(ORG_A))
    conv = a.create_conversation(topic="t", project_id=_project_id(db, "a-proj"))
    a.save_assistant_turn(conv.id, "answer", [])

    b = _svc(db, _scope(ORG_B))
    with pytest.raises(ValueError, match="not found"):
        b.get_messages(conv.id)


def test_ordered_messages_anchor_to_scoped_parent(db):
    # get_ordered_messages is the harness replay path — it takes a bare id
    # (the remote-turn path passes one from the wire), so a foreign id must
    # answer exactly like a nonexistent one, not return another org's history.
    a = _svc(db, _scope(ORG_A))
    conv = a.create_conversation(topic="t", project_id=_project_id(db, "a-proj"))
    a.save_assistant_turn(conv.id, "answer", [])

    b = _svc(db, _scope(ORG_B))
    with pytest.raises(ValueError, match="not found"):
        b.get_ordered_messages(conv.id)
    with pytest.raises(ValueError, match="not found"):
        b.get_ordered_messages(uuid4())  # same shape as a nonexistent id

    # the owning org still reads its history
    assert [m.content for m in a.get_ordered_messages(conv.id)] == ["answer"]


# ── detached producer contract ──────────────────────────────────────────────

def test_detached_writer_cannot_append_to_another_orgs_conversation(db):
    a = _svc(db, _scope(ORG_A))
    conv = a.create_conversation(topic="t", project_id=_project_id(db, "a-proj"))

    # a producer whose principal resolves to org B, on its own fresh session
    producer_b = _svc(db, _scope(ORG_B))
    with pytest.raises(ValueError, match="not found"):
        producer_b.save_assistant_turn(conv.id, "injected", [])
    with Session(db) as s:
        assert s.exec(select(Message)).all() == []


def test_detached_writer_fails_when_conversation_deleted_before_persist(db):
    a = _svc(db, _scope(ORG_A))
    conv = a.create_conversation(topic="t", project_id=_project_id(db, "a-proj"))
    assert a.delete_conversation(conv.id) is True

    producer = _svc(db, _scope(ORG_A))  # same org, fresh session (like _produce)
    with pytest.raises(ValueError, match="not found"):
        producer.save_assistant_turn(conv.id, "late", [])
    with Session(db) as s:
        assert s.exec(select(Message)).all() == []


# ── background callers ──────────────────────────────────────────────────────

def test_background_scope_is_local_in_local_mode(db, monkeypatch):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    assert scope_for_background_context() == LOCAL_SCOPE
    svc = _svc(db, scope_for_background_context())
    conv = svc.create_conversation(topic="cron", project_id=_project_id(db, "a-proj"))
    assert conv.org_id is None and conv.created_by is None  # today's behavior


def test_background_scope_fails_closed_in_org_mode(db, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()
    try:
        with pytest.raises(MissingTenantScopeError):
            scope_for_background_context()
        # nothing was written anywhere
        with Session(db) as s:
            assert s.exec(select(Conversation)).all() == []
    finally:
        get_app_settings.cache_clear()


def test_same_org_users_cannot_see_each_others_conversations(db):
    """The staging leak (Aug-20): conversations are personal, but the list was
    scoped by org only, so every member saw every member's tasks. A member must
    see only their own — list, and get-by-id."""
    pid = _project_id(db, "a-proj")
    alice = _svc(db, _scope(ORG_A, "alice"))
    bob = _svc(db, _scope(ORG_A, "bob"))

    a_conv = alice.create_conversation(topic="alice secret", project_id=pid)
    b_conv = bob.create_conversation(topic="bob secret", project_id=pid)

    # List: each sees only their own, though they share org + project.
    alice_ids = {c.id for c in alice.list_conversations(all_projects=True)}
    bob_ids = {c.id for c in bob.list_conversations(all_projects=True)}
    assert alice_ids == {a_conv.id}
    assert bob_ids == {b_conv.id}

    # Get-by-id: bob cannot open alice's conversation by guessing its id.
    with pytest.raises(ValueError):
        bob.get_conversation(a_conv.id)

    # Delete-by-id: bob cannot delete alice's conversation.
    assert bob.delete_conversation(a_conv.id) is False
    assert alice.get_conversation(a_conv.id).id == a_conv.id  # still there
