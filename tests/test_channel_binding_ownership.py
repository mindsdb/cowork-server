"""Channel bindings are org-scoped but not owner-scoped. In org mode a route
can no longer be pinned to a conversation at all: the runtime keeps each
route's conversation, so nobody can steer a chat (a DM in particular) into a
conversation they or an admin can read. Routes pinned before that rule still
exist; for those, a binding pointed at someone else's conversation stays
invisible and unwritable to anyone but its owner.
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
NOT_IN_ORG_MODE = "cannot be pinned to a conversation in org mode"


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


def _ownerless_conversation(engine) -> Conversation:
    """An org conversation with no owner, as the channel runtime creates them.
    Written raw: ScopedSession.add() would stamp an owner."""
    raw = Session(engine)
    project = Project(name="legacy-proj", path="/tmp/legacy-proj", org_id=ORG)
    raw.add(project)
    raw.commit()
    raw.refresh(project)
    conv = Conversation(topic="slack: g", project_id=project.id, org_id=ORG, created_by=None)
    raw.add(conv)
    raw.commit()
    raw.refresh(conv)
    return conv


def _legacy_pinned_binding(engine, conversation_id, created_by: str = USER_A) -> ChannelBinding:
    """A route pinned to a conversation before pinning was refused in org
    mode; such rows exist and the read and write gates still apply to them."""
    raw = Session(engine)
    binding = ChannelBinding(
        channel_type="slack", external_group_id="g1", external_thread_key="__default__",
        org_id=ORG, created_by=created_by, anton_conversation_id=conversation_id,
    )
    raw.add(binding)
    raw.commit()
    raw.refresh(binding)
    return binding


# --- create and update never pin in org mode --------------------------------

def test_create_refuses_a_pin_even_to_the_callers_own_conversation(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)

    with pytest.raises(ValueError, match=NOT_IN_ORG_MODE):
        _svc(engine, USER_A).create(
            BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
        )


def test_create_refuses_a_pin_to_another_members_conversation(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)

    with pytest.raises(ValueError, match=NOT_IN_ORG_MODE):
        _svc(engine, USER_B).create(
            BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
        )


def test_create_refuses_a_pin_to_a_conversation_with_no_owner(engine):
    conv = _ownerless_conversation(engine)

    with pytest.raises(ValueError, match=NOT_IN_ORG_MODE):
        _svc(engine, USER_B).create(
            BindingCreateRequest(channel_type="slack", external_group_id="g1", anton_conversation_id=conv.id)
        )


def test_update_refuses_a_new_pin(engine):
    conv = _ownerless_conversation(engine)
    binding = _svc(engine, USER_A).create(BindingCreateRequest(channel_type="slack", external_group_id="g2"))

    with pytest.raises(ValueError, match=NOT_IN_ORG_MODE):
        _svc(engine, USER_A).update(binding.id, BindingUpdateRequest(anton_conversation_id=conv.id))


def test_update_may_still_unpin(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    binding = _legacy_pinned_binding(engine, conv.id)

    updated = _svc(engine, USER_A).update(binding.id, BindingUpdateRequest(anton_conversation_id=None))

    assert updated.anton_conversation_id is None


# --- list: legacy bindings pinned to another member's conversation stay hidden

def test_list_excludes_a_binding_pinned_to_another_members_conversation(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    _legacy_pinned_binding(engine, conv.id)

    assert _svc(engine, USER_B).list() == []
    assert len(_svc(engine, USER_A).list()) == 1


def test_list_still_includes_a_binding_not_pinned_to_any_conversation(engine):
    # Unpinned bindings aren't anyone's private data yet — org membership is
    # still the right (and only) gate for those.
    _svc(engine, USER_A).create(BindingCreateRequest(channel_type="slack", external_group_id="g2"))

    assert len(_svc(engine, USER_B).list()) == 1


# --- update: can't touch a legacy binding tied to someone else's conversation

def test_update_denies_caller_who_is_not_the_conversations_owner(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    binding = _legacy_pinned_binding(engine, conv.id)

    with pytest.raises(BindingNotFoundError):
        _svc(engine, USER_B).update(binding.id, BindingUpdateRequest(display_name="renamed"))


def test_update_allows_the_conversations_own_owner_to_rename_it(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    binding = _legacy_pinned_binding(engine, conv.id)

    updated = _svc(engine, USER_A).update(binding.id, BindingUpdateRequest(display_name="renamed"))
    assert updated.display_name == "renamed"


# --- delete: same gate, reported the same way as a missing binding ---------

def test_delete_denies_caller_who_is_not_the_conversations_owner(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    binding = _legacy_pinned_binding(engine, conv.id)

    assert _svc(engine, USER_B).delete(binding.id) is False
    assert _svc(engine, USER_A).list() != []


def test_delete_allows_the_conversations_own_owner(engine):
    project = _project(engine, USER_A)
    conv = _conversation(engine, USER_A, project.id)
    binding = _legacy_pinned_binding(engine, conv.id)

    assert _svc(engine, USER_A).delete(binding.id) is True


# --- edges: ownerless conversations and local mode --------------------------

def test_a_binding_pinned_to_a_conversation_with_no_owner_stays_visible(engine):
    # The runtime's own conversations have no owner, so the route pointing at
    # one is org-wide: the owner removes it in the reported flow.
    conv = _ownerless_conversation(engine)
    _legacy_pinned_binding(engine, conv.id)

    assert len(_svc(engine, USER_B).list()) == 1


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
