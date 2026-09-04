"""Channel bindings are org-scoped but not owner-scoped: any org member could
bind a channel into another member's private conversation and read the
traffic that flows through it. These tests pin down the fix — a binding
pointed at someone else's conversation is invisible and unwritable to anyone
but its owner.
"""
from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.channels.plugin import ChannelPlugin, CredentialField, CredentialSchema
from cowork.channels.registry import PluginRegistry
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.channel import ChannelBinding
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.schemas.channels import BindingCreateRequest, BindingUpdateRequest
from cowork.services.channel_bindings import (
    BindingNotFoundError,
    ChannelBindingService,
)
from cowork.services.conversations import ConversationService

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
USER_A = "11111111-1111-4111-8111-111111111111"
USER_B = "22222222-2222-4222-8222-222222222222"


def _fake_registry() -> PluginRegistry:
    async def _factory(creds):
        return None

    registry = PluginRegistry()
    registry.register(
        ChannelPlugin(
            channel_type="slack",
            display_name="Slack",
            factory=_factory,
            credentials=CredentialSchema(fields=(CredentialField(name="signing_secret", label="Signing secret"),)),
        )
    )
    return registry


@pytest.fixture()
def engine():
    import cowork.models.project, cowork.models.conversation  # noqa: F401

    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    return eng


def _scope(user_id: str) -> TenantScope:
    return TenantScope(org_mode=True, org_id=ORG, user_id=user_id)


def _scoped(engine, user_id: str) -> ScopedSession:
    return ScopedSession(Session(engine), _scope(user_id))


def _svc(engine, user_id: str) -> ChannelBindingService:
    return ChannelBindingService(_scoped(engine, user_id), registry=_fake_registry())


def _project(engine, user_id: str) -> Project:
    scoped = _scoped(engine, user_id)
    project = Project(name="proj", path="/tmp/proj")
    scoped.add(project)
    scoped.commit()
    scoped.refresh(project)
    return project


def _conversation(engine, user_id: str, project_id) -> Conversation:
    scoped = _scoped(engine, user_id)
    conv = ConversationService(scoped).create_conversation(topic="private chat", project_id=project_id)
    return conv


# --- create: binding a channel into someone else's conversation ------------

def test_create_denies_a_conversation_owned_by_another_org_member(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)

    with pytest.raises(ValueError, match="conversation not found"):
        _svc(engine, USER_B).create(
            BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
        )


def test_create_allows_the_conversations_own_owner(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)

    binding = _svc(engine, USER_A).create(
        BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
    )
    assert binding.anton_conversation_id == conv.id


# --- list: bindings pinned to another member's conversation stay hidden ----

def test_list_excludes_a_binding_pinned_to_another_members_conversation(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    _svc(engine, USER_A).create(
        BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
    )

    assert _svc(engine, USER_B).list() == []
    assert len(_svc(engine, USER_A).list()) == 1


def test_list_still_includes_a_binding_not_pinned_to_any_conversation(engine):
    # Unpinned bindings aren't anyone's private data yet — org membership is
    # still the right (and only) gate for those.
    _svc(engine, USER_A).create(BindingCreateRequest(channel_type="slack", external_group_id="g2"))

    assert len(_svc(engine, USER_B).list()) == 1


# --- update: can't touch a binding tied to someone else's conversation -----

def test_update_denies_caller_who_is_not_the_conversations_owner(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    binding = _svc(engine, USER_A).create(
        BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
    )

    with pytest.raises(BindingNotFoundError):
        _svc(engine, USER_B).update(binding.id, BindingUpdateRequest(display_name="renamed"))


def test_update_allows_the_conversations_own_owner(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    binding = _svc(engine, USER_A).create(
        BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
    )

    updated = _svc(engine, USER_A).update(binding.id, BindingUpdateRequest(display_name="renamed"))
    assert updated.display_name == "renamed"


# --- delete: same gate, reported the same way as a missing binding ---------

def test_delete_denies_caller_who_is_not_the_conversations_owner(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    binding = _svc(engine, USER_A).create(
        BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
    )

    assert _svc(engine, USER_B).delete(binding.id) is False
    assert _svc(engine, USER_A).list() != []


def test_delete_allows_the_conversations_own_owner(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    binding = _svc(engine, USER_A).create(
        BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
    )

    assert _svc(engine, USER_A).delete(binding.id) is True


# --- edges: legacy rows and local mode are unaffected -----------------------

def test_unstamped_legacy_conversation_stays_visible(engine):
    # A conversation from before org attribution landed (created_by NULL)
    # isn't "owned" by anyone in particular — same lenient treatment
    # ScopedSession already gives other NULL-owner rows elsewhere. Written
    # through a raw session: ScopedSession.add() would stamp created_by itself,
    # which is exactly the pre-attribution row this test needs to not have.
    raw = Session(engine)
    project = Project(name="legacy-proj", path="/tmp/legacy-proj", org_id=ORG)
    raw.add(project)
    raw.commit()
    raw.refresh(project)
    conv = Conversation(topic="legacy", project_id=project.id, org_id=ORG, created_by=None)
    raw.add(conv)
    raw.commit()
    raw.refresh(conv)

    binding = _svc(engine, USER_B).create(
        BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
    )
    assert binding.anton_conversation_id == conv.id


def test_local_mode_is_unaffected_by_the_owner_gate(engine):
    session = Session(engine)
    scoped = ScopedSession(session, LOCAL_SCOPE)
    project = Project(name="local-proj", path="/tmp/local-proj")
    scoped.add(project)
    scoped.commit()
    scoped.refresh(project)
    conv = ConversationService(scoped).create_conversation(topic="local chat", project_id=project.id)

    svc = ChannelBindingService(scoped, registry=_fake_registry())
    binding = svc.create(
        BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
    )
    assert len(svc.list()) == 1
    assert svc.update(binding.id, BindingUpdateRequest(display_name="x")).display_name == "x"
    assert svc.delete(binding.id) is True
