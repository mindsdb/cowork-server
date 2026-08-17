"""Per-org channel installations, credentials, and inbound dedupe.

Installations are org-wide, not per-member (matches the existing provider-
credential pattern: org-shared, not per-user). Local/desktop mode keeps
today's behavior verbatim — exactly one installation per channel_type,
credentials in global (scope=NULL) rows, dedupe with no org dimension.
"""
from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.channels.plugin import ChannelPlugin, CredentialField, CredentialSchema
from cowork.channels.registry import PluginRegistry
from cowork.db.scoped import LOCAL_SCOPE, MissingTenantScopeError, ScopedSession, TenantScope
from cowork.models.channel import ChannelEvent, ChannelInstallation
from cowork.services.channel_events import ChannelEventService
from cowork.services.channels import ChannelConfigService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


async def _fake_factory(creds):
    return None


def _fake_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(
        ChannelPlugin(
            channel_type="slack",
            display_name="Slack",
            factory=_fake_factory,
            credentials=CredentialSchema(
                fields=(CredentialField(name="signing_secret", label="Signing secret"),)
            ),
        )
    )
    return registry


@pytest.fixture()
def engine():
    import cowork.models.project, cowork.models.conversation  # noqa: F401
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    return eng


def _org(org: str) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org)


def _scoped(engine, scope: TenantScope = LOCAL_SCOPE) -> ScopedSession:
    return ScopedSession(Session(engine), scope)


def _config_svc(engine, scope: TenantScope) -> ChannelConfigService:
    return ChannelConfigService(_scoped(engine, scope), registry=_fake_registry())


# --- ChannelInstallation: one per (channel_type, org_id), org_id NULL included ---

def test_installation_unique_per_channel_type_in_local_mode(engine):
    session = Session(engine)
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack"))
    session.commit()
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack again"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_installation_allowed_per_org(engine):
    session = Session(engine)
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack", org_id=ORG_A))
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack", org_id=ORG_B))
    session.commit()  # must not raise: two orgs, same channel_type


def test_installation_still_unique_per_org(engine):
    session = Session(engine)
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack", org_id=ORG_A))
    session.commit()
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack again", org_id=ORG_A))
    with pytest.raises(IntegrityError):
        session.commit()


def test_installation_org_row_does_not_collide_with_local_row(engine):
    session = Session(engine)
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack"))
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack", org_id=ORG_A))
    session.commit()  # must not raise: a local row and an org row are different tiers


# --- ChannelConfigService: org-scoped credential storage ---

def test_credentials_isolated_across_orgs(engine):
    svc_a = _config_svc(engine, _org(ORG_A))
    svc_b = _config_svc(engine, _org(ORG_B))

    svc_a.set_config("slack", {"signing_secret": "org-a-secret"})

    assert svc_a.load_credentials("slack") == {"signing_secret": "org-a-secret"}
    assert svc_b.load_credentials("slack") == {}


def test_org_write_does_not_clobber_another_orgs_value(engine):
    svc_a = _config_svc(engine, _org(ORG_A))
    svc_b = _config_svc(engine, _org(ORG_B))

    svc_a.set_config("slack", {"signing_secret": "org-a-secret"})
    svc_b.set_config("slack", {"signing_secret": "org-b-secret"})

    assert svc_a.load_credentials("slack") == {"signing_secret": "org-a-secret"}
    assert svc_b.load_credentials("slack") == {"signing_secret": "org-b-secret"}


def test_local_mode_credentials_are_global_rows(engine):
    svc = _config_svc(engine, LOCAL_SCOPE)
    svc.set_config("slack", {"signing_secret": "desktop-secret"})
    assert svc.load_credentials("slack") == {"signing_secret": "desktop-secret"}


def test_org_mode_without_org_fails_closed_on_write(engine):
    svc = _config_svc(engine, TenantScope(org_mode=True))
    with pytest.raises(MissingTenantScopeError):
        svc.set_config("slack", {"signing_secret": "x"})


def test_org_mode_without_org_fails_closed_on_read(engine):
    svc = _config_svc(engine, TenantScope(org_mode=True))
    with pytest.raises(MissingTenantScopeError):
        svc.load_credentials("slack")


def test_installation_row_is_scoped_to_its_org(engine):
    svc_a = _config_svc(engine, _org(ORG_A))
    svc_b = _config_svc(engine, _org(ORG_B))

    svc_a.set_config("slack", {"signing_secret": "org-a-secret"})

    assert [i.channel_type for i in svc_a.list_installations()] == ["slack"]
    assert svc_b.list_installations() == []


# --- ChannelEventService: org-scoped inbound dedupe ---

def test_dedupe_isolated_across_orgs(engine):
    svc_a = ChannelEventService(_scoped(engine, _org(ORG_A)))
    svc_b = ChannelEventService(_scoped(engine, _org(ORG_B)))

    assert svc_a.record_inbound("slack", dedupe_key="evt-1") is not None
    # Same channel_type + dedupe_key, different org: not a duplicate.
    assert svc_b.is_duplicate_inbound("slack", "evt-1") is False
    assert svc_b.record_inbound("slack", dedupe_key="evt-1") is not None


def test_dedupe_still_works_within_the_same_org(engine):
    svc_a = ChannelEventService(_scoped(engine, _org(ORG_A)))
    assert svc_a.record_inbound("slack", dedupe_key="evt-1") is not None
    assert svc_a.is_duplicate_inbound("slack", "evt-1") is True


def test_local_mode_dedupe_is_unaffected(engine):
    svc = ChannelEventService(_scoped(engine, LOCAL_SCOPE))
    assert svc.record_inbound("slack", dedupe_key="evt-1") is not None
    assert svc.is_duplicate_inbound("slack", "evt-1") is True
