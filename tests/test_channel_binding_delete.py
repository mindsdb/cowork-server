"""Removing a channel route that has already served a message.

The first inbound message writes a ChannelSession row that points at its
route. Postgres enforces that foreign key, so these run on SQLite with
`PRAGMA foreign_keys=ON`; the default test SQLite would let a wrong delete
order through.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.channels.plugins.discord import DiscordBridge
from cowork.channels.registry import PluginRegistry, load_first_party_plugins
from cowork.channels.runtime import AntonChannelRuntime, LiveAdapterRegistry
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope, scope_for_org
from cowork.models.channel import ChannelBinding, ChannelSession
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.schemas.channels import BindingCreateRequest
from cowork.services.channel_bindings import ChannelBindingService

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
MEMBER = "11111111-1111-4111-8111-111111111111"
OWNER = "22222222-2222-4222-8222-222222222222"


def _fk_enforced_engine() -> Engine:
    """In-memory SQLite that enforces foreign keys the way Postgres does."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def org_mode(monkeypatch: pytest.MonkeyPatch):
    """Org tenancy, so `scope_for_org` gives the runtime its org scope."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def engine() -> Engine:
    """A fresh FK-enforced database per test."""
    return _fk_enforced_engine()


@pytest.fixture()
def registry() -> PluginRegistry:
    """The first-party plugins, so `discord` is a known channel type."""
    plugins = PluginRegistry()
    load_first_party_plugins(plugins)
    return plugins


@pytest.fixture()
def project_id(engine: Engine) -> UUID:
    """A project in ORG for routes to point at."""
    with Session(engine) as session:
        project = Project(name="routes", path="/routes", org_id=ORG)
        session.add(project)
        session.commit()
        return project.id


def _user(user_id: str) -> TenantScope:
    return TenantScope(org_mode=True, org_id=ORG, user_id=user_id)


def _discord_message(channel_id: str) -> object:
    """An inbound event as the Discord Gateway normalizer produces it."""
    event = DiscordBridge({})._normalize_message(
        {"id": f"m-{channel_id}", "channel_id": channel_id, "content": "hello", "author": {"id": "42"}}
    )
    assert event is not None
    return event


def _serve_first_message(engine: Engine, project_id: UUID, binding_id: UUID | None, channel_id: str) -> UUID:
    """Run the runtime's first-message bookkeeping for a route under its org
    scope, as `AntonChannelRuntime.handle` does. Returns the route id."""
    runtime = AntonChannelRuntime(LiveAdapterRegistry(), default_project_id=project_id)
    message = _discord_message(channel_id)
    with Session(engine) as session:
        scoped = ScopedSession(session, scope_for_org(ORG))
        if binding_id is None:
            binding = runtime._resolve_or_create_binding(scoped, "discord", message)
        else:
            binding = scoped.get(ChannelBinding, binding_id)
        conversation = runtime._ensure_conversation(scoped, binding)
        runtime._touch_channel_session(scoped, binding, conversation, message)
        return binding.id


def _service(engine: Engine, registry: PluginRegistry, user_id: str) -> ChannelBindingService:
    return ChannelBindingService(ScopedSession(Session(engine), _user(user_id)), registry=registry)


def _rows(engine: Engine, binding_id: UUID) -> tuple[ChannelBinding | None, list[ChannelSession]]:
    with Session(engine) as session:
        binding = session.get(ChannelBinding, binding_id)
        sessions = session.exec(select(ChannelSession).where(ChannelSession.binding_id == binding_id)).all()
        return binding, list(sessions)


def test_owner_removes_a_members_route_after_its_first_message(engine, registry, project_id):
    binding_id = _service(engine, registry, MEMBER).create(
        BindingCreateRequest(channel_type="discord", external_group_id="123", anton_project_id=project_id)
    ).id
    _serve_first_message(engine, project_id, binding_id, "123")
    assert _rows(engine, binding_id)[1], "the first message must have written a session row"

    owner = _service(engine, registry, OWNER)
    assert owner.delete(binding_id) is True

    assert _rows(engine, binding_id) == (None, [])
    assert [b.id for b in owner.list()] == []


def test_route_created_by_the_first_message_can_be_removed(engine, registry, project_id):
    binding_id = _serve_first_message(engine, project_id, None, "456")
    binding, sessions = _rows(engine, binding_id)
    assert binding is not None and binding.created_by is None and sessions

    assert _service(engine, registry, OWNER).delete(binding_id) is True

    assert _rows(engine, binding_id) == (None, [])


def test_removing_a_route_keeps_its_conversation(engine, registry, project_id):
    binding_id = _serve_first_message(engine, project_id, None, "789")
    conversation_id = _rows(engine, binding_id)[0].anton_conversation_id

    assert _service(engine, registry, OWNER).delete(binding_id) is True

    with Session(engine) as session:
        assert session.get(Conversation, conversation_id) is not None


def test_unused_route_still_removes(engine, registry, project_id):
    binding_id = _service(engine, registry, MEMBER).create(
        BindingCreateRequest(channel_type="discord", external_group_id="321", anton_project_id=project_id)
    ).id

    assert _service(engine, registry, OWNER).delete(binding_id) is True

    assert _rows(engine, binding_id) == (None, [])
