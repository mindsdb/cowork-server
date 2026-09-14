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
from cowork.channels.runtime import LiveAdapterRegistry
from cowork.db.scoped import LOCAL_SCOPE, MissingTenantScopeError, ScopedSession, TenantScope
from cowork.models.channel import ChannelEvent, ChannelInstallation
from cowork.services.channel_events import ChannelEventService
from cowork.services.channels import ChannelConfigService, resolve_installation_by_external_account

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


class _FakeAdapter:
    """Echoes back the creds it was built from, so a test can tell which
    org's credentials produced this particular cached adapter."""

    def __init__(self, creds):
        self.creds = creds

    async def shutdown(self) -> None:
        pass


def _live_fake_registry() -> PluginRegistry:
    async def _factory(creds):
        return _FakeAdapter(creds)

    registry = PluginRegistry()
    registry.register(
        ChannelPlugin(
            channel_type="slack",
            display_name="Slack",
            factory=_factory,
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


# --- ChannelInstallation.external_account_id: the pre-scope webhook-routing key ---

def test_external_account_id_unique_per_channel_type(engine):
    session = Session(engine)
    session.add(ChannelInstallation(
        channel_type="slack", display_name="Slack", org_id=ORG_A, external_account_id="T-A",
    ))
    session.commit()
    session.add(ChannelInstallation(
        channel_type="slack", display_name="Slack again", org_id=ORG_B, external_account_id="T-A",
    ))
    with pytest.raises(IntegrityError):
        session.commit()


def test_external_account_id_can_be_null_on_multiple_rows(engine):
    # Not yet discovered (setup incomplete) — must not collide with each other.
    session = Session(engine)
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack", org_id=ORG_A))
    session.add(ChannelInstallation(channel_type="slack", display_name="Slack", org_id=ORG_B))
    session.commit()  # must not raise: both external_account_id are NULL


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


def test_set_external_account_id_stamps_the_installation(engine):
    svc = _config_svc(engine, _org(ORG_A))
    svc.set_config("slack", {"signing_secret": "org-a-secret"})

    svc.set_external_account_id("slack", "T-A")

    installations = svc.list_installations()
    assert len(installations) == 1
    assert installations[0].external_account_id == "T-A"


def test_set_external_account_id_creates_installation_if_missing(engine):
    svc = _config_svc(engine, _org(ORG_A))
    svc.set_external_account_id("slack", "T-A")

    installations = svc.list_installations()
    assert len(installations) == 1
    assert installations[0].external_account_id == "T-A"


def test_set_external_account_id_fails_closed_without_org(engine):
    svc = _config_svc(engine, TenantScope(org_mode=True))
    with pytest.raises(MissingTenantScopeError):
        svc.set_external_account_id("slack", "T-A")


def test_set_external_account_id_rejects_taken_id_across_orgs(engine):
    svc_a = _config_svc(engine, _org(ORG_A))
    svc_b = _config_svc(engine, _org(ORG_B))
    svc_a.set_external_account_id("slack", "T-A")

    with pytest.raises(IntegrityError):
        svc_b.set_external_account_id("slack", "T-A")


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


# --- resolve_installation_by_external_account: the pre-scope webhook lookup ---

def test_resolve_installation_finds_the_right_org(engine):
    svc_a = _config_svc(engine, _org(ORG_A))
    svc_b = _config_svc(engine, _org(ORG_B))
    svc_a.set_external_account_id("slack", "T-A")
    svc_b.set_external_account_id("slack", "T-B")

    session = Session(engine)
    install = resolve_installation_by_external_account(session, "slack", "T-B")
    assert install is not None
    assert install.org_id == ORG_B


def test_resolve_installation_returns_none_for_unknown_account(engine):
    session = Session(engine)
    assert resolve_installation_by_external_account(session, "slack", "T-nope") is None


def test_resolve_installation_needs_no_scope_at_all(engine):
    # The whole point: this runs BEFORE any org context exists. A raw Session
    # works; nothing here requires a ScopedSession or a TenantScope.
    svc = _config_svc(engine, _org(ORG_A))
    svc.set_external_account_id("slack", "T-A")

    session = Session(engine)
    install = resolve_installation_by_external_account(session, "slack", "T-A")
    assert install is not None and install.org_id == ORG_A


# --- LiveAdapterRegistry: cached per (channel_type, org_id) ---

async def test_refresh_caches_independently_per_org(engine):
    registry = LiveAdapterRegistry(_live_fake_registry())
    _config_svc(engine, _org(ORG_A)).set_config("slack", {"signing_secret": "org-a-secret"})
    _config_svc(engine, _org(ORG_B)).set_config("slack", {"signing_secret": "org-b-secret"})

    assert await registry.refresh("slack", ORG_A, session=_scoped(engine, _org(ORG_A)))
    assert await registry.refresh("slack", ORG_B, session=_scoped(engine, _org(ORG_B)))

    a = registry.get("slack", ORG_A)
    b = registry.get("slack", ORG_B)
    assert a is not b
    assert a.creds == {"signing_secret": "org-a-secret"}
    assert b.creds == {"signing_secret": "org-b-secret"}


async def test_local_mode_registry_is_unaffected(engine):
    registry = LiveAdapterRegistry(_live_fake_registry())
    _config_svc(engine, LOCAL_SCOPE).set_config("slack", {"signing_secret": "desktop-secret"})

    assert await registry.refresh("slack", session=_scoped(engine, LOCAL_SCOPE))

    adapter = registry.get("slack")
    assert adapter is not None
    assert adapter.creds == {"signing_secret": "desktop-secret"}
    # An org lookup must never see the local/desktop adapter.
    assert registry.get("slack", ORG_A) is None


async def test_remove_only_drops_the_named_org(engine):
    registry = LiveAdapterRegistry(_live_fake_registry())
    _config_svc(engine, _org(ORG_A)).set_config("slack", {"signing_secret": "org-a-secret"})
    _config_svc(engine, _org(ORG_B)).set_config("slack", {"signing_secret": "org-b-secret"})
    await registry.refresh("slack", ORG_A, session=_scoped(engine, _org(ORG_A)))
    await registry.refresh("slack", ORG_B, session=_scoped(engine, _org(ORG_B)))

    await registry.remove("slack", ORG_A)

    assert registry.get("slack", ORG_A) is None
    assert registry.get("slack", ORG_B) is not None


async def test_get_or_refresh_returns_cached_without_touching_session(engine):
    registry = LiveAdapterRegistry(_live_fake_registry())
    _config_svc(engine, _org(ORG_A)).set_config("slack", {"signing_secret": "org-a-secret"})
    await registry.refresh("slack", ORG_A, session=_scoped(engine, _org(ORG_A)))
    cached = registry.get("slack", ORG_A)

    # session=None here would try get_open_session() if this weren't a cache hit.
    adapter = await registry.get_or_refresh("slack", ORG_A, session=_scoped(engine, _org(ORG_A)))
    assert adapter is cached


async def test_get_or_refresh_builds_on_first_use(engine):
    registry = LiveAdapterRegistry(_live_fake_registry())
    _config_svc(engine, _org(ORG_A)).set_config("slack", {"signing_secret": "org-a-secret"})
    assert registry.get("slack", ORG_A) is None

    adapter = await registry.get_or_refresh("slack", ORG_A, session=_scoped(engine, _org(ORG_A)))

    assert adapter is not None
    assert adapter.creds == {"signing_secret": "org-a-secret"}
    assert registry.get("slack", ORG_A) is adapter


async def test_get_or_refresh_returns_none_for_unregistered_channel_type(engine):
    # Real and fake plugin factories both build a bridge unconditionally —
    # incomplete credentials surface later at verify_signature, not here.
    registry = LiveAdapterRegistry(_live_fake_registry())
    adapter = await registry.get_or_refresh("discord", ORG_B, session=_scoped(engine, _org(ORG_B)))
    assert adapter is None


# --- LiveAdapterRegistry.resolve_org_bridge: the real org_resolver server.py wires in ---

async def test_resolve_org_bridge_finds_the_right_orgs_bridge(engine):
    registry = LiveAdapterRegistry(_live_fake_registry())
    _config_svc(engine, _org(ORG_A)).set_config("slack", {"signing_secret": "org-a-secret"})
    _config_svc(engine, _org(ORG_A)).set_external_account_id("slack", "T-A")

    resolved = await registry.resolve_org_bridge("slack", "T-A", session=Session(engine))

    assert resolved is not None
    bridge, org_id = resolved
    assert org_id == ORG_A
    assert bridge.creds == {"signing_secret": "org-a-secret"}
    assert registry.get("slack", ORG_A) is bridge  # cached for next time


async def test_resolve_org_bridge_returns_none_for_unknown_routing_key(engine):
    registry = LiveAdapterRegistry(_live_fake_registry())
    resolved = await registry.resolve_org_bridge("slack", "T-nope", session=Session(engine))
    assert resolved is None


async def test_resolve_org_bridge_resolves_the_local_installation_too(engine):
    # A local/desktop install can have external_account_id set too — nothing
    # about resolution is org-mode-specific, and it must not crash on org_id=None.
    registry = LiveAdapterRegistry(_live_fake_registry())
    _config_svc(engine, LOCAL_SCOPE).set_config("slack", {"signing_secret": "desktop-secret"})
    _config_svc(engine, LOCAL_SCOPE).set_external_account_id("slack", "T-local")

    resolved = await registry.resolve_org_bridge("slack", "T-local", session=Session(engine))

    assert resolved is not None
    bridge, org_id = resolved
    assert org_id is None
    assert bridge.creds == {"signing_secret": "desktop-secret"}
    assert registry.get("slack") is bridge  # same cache slot as the local bootstrap


# --- ChannelLifecycleService: setup/teardown act on the caller's own org slot ---

def _lifecycle_registry() -> PluginRegistry:
    from cowork.channels.lifecycle import ChannelLifecycle, LifecycleResult

    async def _factory(creds):
        return _FakeAdapter(creds)

    async def _setup(ctx):
        await ctx.refresh_adapter()
        return LifecycleResult(active=True, detail="ok")

    async def _teardown(ctx):
        await ctx.remove_adapter()
        return LifecycleResult(active=False, detail="ok")

    registry = PluginRegistry()
    registry.register(
        ChannelPlugin(
            channel_type="slack",
            display_name="Slack",
            factory=_factory,
            credentials=CredentialSchema(
                fields=(CredentialField(name="signing_secret", label="Signing secret"),)
            ),
            lifecycle=ChannelLifecycle(setup=_setup, teardown=_teardown),
        )
    )
    return registry


async def test_lifecycle_setup_and_teardown_hit_the_callers_own_org_slot(engine):
    from cowork.services.channel_lifecycle import ChannelLifecycleService

    plugins = _lifecycle_registry()
    scoped = _scoped(engine, _org(ORG_A))
    ChannelConfigService(scoped, registry=plugins).set_config("slack", {"signing_secret": "org-a-secret"})
    adapters = LiveAdapterRegistry(plugins)
    svc = ChannelLifecycleService(scoped, adapters, plugins)

    await svc.setup("slack")
    assert adapters.get("slack", ORG_A) is not None
    assert adapters.get("slack") is None  # never the local slot

    await svc.teardown("slack")
    assert adapters.get("slack", ORG_A) is None
