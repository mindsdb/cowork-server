"""One org member's model rules never change another member's model resolution.

Auth resolves each ``/v1/models`` row's ``enabled`` and ``disabled_reason`` from
the model rules for the caller's org, workspace and team, so two members of one
org get different rows. The availability map resolution reads
(``minds_model_enabled``) is an org row. Written from whichever member opened the
picker last, a team-A restriction swapped member B off the model, and B's next
load lifted it for A, whose turn then hit the restricted card.

In org mode ``recommended_models`` now writes the member-invariant facts to the
org map and the caller's ``model_restricted`` ids to the caller's own
``minds_model_restricted`` row, which ``UserSettings._minds_enabled_map`` lays
over the map. A desktop install has one member and keeps restrictions in the
map, unchanged.

These drive the real endpoint and the real settings service on a private
in-memory database, so each member's resolution reads exactly the rows the
endpoint wrote.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import settings as ep
from cowork.common.settings import user_settings as us
from cowork.common.settings.app_settings import MINDS_FREE_MODEL
from cowork.common.settings.user_settings import setting_is_org_scoped
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services.providers import MindsModelListing
from cowork.services.settings import SettingService
from _fakes import FakeRequest

ORG = "org-member-rules"
MEMBER_A = "u-team-a"
MEMBER_B = "u-team-b"
PINNED = "sonnet"


def _listing(*, restricted: bool, flags: bool = True) -> MindsModelListing:
    """The catalog one caller gets: ``PINNED`` restricted for them or not."""
    enabled = {MINDS_FREE_MODEL: True, PINNED: not restricted} if flags else {}
    reasons = {PINNED: "model_restricted"} if restricted and flags else {}
    return MindsModelListing([MINDS_FREE_MODEL, PINNED], {}, enabled, {}, {}, {}, {}, reasons)


def _mode(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("COWORK_TENANCY_MODE", mode)
    us.get_app_settings.cache_clear()


@pytest.fixture()
def session():
    import cowork.models.setting  # noqa: F401

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture()
def org(monkeypatch):
    """Org tenancy, with each member's catalog answered from ``restricted_for``."""
    _mode(monkeypatch, "org")
    restricted_for: set[str] = set()

    async def fake_org_catalog(*, org_id, user_id, bearer_token, refresh=False):
        return _listing(restricted=user_id in restricted_for)

    monkeypatch.setattr(ep, "fetch_org_model_catalog", fake_org_catalog)
    yield restricted_for
    us.get_app_settings.cache_clear()


def _scope(user_id: str) -> TenantScope:
    return TenantScope(org_mode=True, org_id=ORG, user_id=user_id)


def _open_picker(session, user_id: str) -> None:
    request = FakeRequest({"Authorization": f"Bearer jwt-{user_id}"})
    asyncio.run(ep.recommended_models(request, session, _scope(user_id)))


def _pin(session, user_id: str) -> None:
    SettingService(session, _scope(user_id)).upsert_setting("planning_model", PINNED)


def _resolved(session, user_id: str) -> str | None:
    return SettingService(session, _scope(user_id)).load().resolved_planning_model


def test_a_members_restriction_does_not_swap_another_member(session, org):
    org.add(MEMBER_A)
    _pin(session, MEMBER_B)

    _open_picker(session, MEMBER_A)

    assert _resolved(session, MEMBER_B) == PINNED
    # A's lock went to A's own row, not to the map every member reads.
    shared = json.loads(SettingService(session, _scope(MEMBER_B)).load().minds_model_enabled)
    assert shared == {MINDS_FREE_MODEL: True, PINNED: True}
    assert json.loads(SettingService(session, _scope(MEMBER_A)).load().minds_model_restricted) == [PINNED]


def test_another_members_refresh_does_not_lift_a_restriction(session, org):
    org.add(MEMBER_A)
    _pin(session, MEMBER_A)

    _open_picker(session, MEMBER_A)
    _open_picker(session, MEMBER_B)

    assert _resolved(session, MEMBER_A) == MINDS_FREE_MODEL


def test_a_members_own_restricted_pin_still_resolves_as_unavailable(session, org):
    org.add(MEMBER_A)
    _pin(session, MEMBER_A)

    _open_picker(session, MEMBER_A)

    settings = SettingService(session, _scope(MEMBER_A)).load()
    assert settings._minds_enabled_map() == {MINDS_FREE_MODEL: True, PINNED: False}
    assert settings.resolved_planning_model == MINDS_FREE_MODEL


def test_a_lifted_rule_lifts_the_members_lock(session, org):
    org.add(MEMBER_A)
    _pin(session, MEMBER_A)
    _open_picker(session, MEMBER_A)

    org.discard(MEMBER_A)
    _open_picker(session, MEMBER_A)

    assert _resolved(session, MEMBER_A) == PINNED
    assert json.loads(SettingService(session, _scope(MEMBER_A)).load().minds_model_restricted) == []


def test_a_listing_without_flags_leaves_the_members_lock_alone(session, org, monkeypatch):
    # No `enabled` flags is no evidence about locks (gateway version skew), the
    # same rule that keeps the shared map from being wiped.
    org.add(MEMBER_A)
    _pin(session, MEMBER_A)
    _open_picker(session, MEMBER_A)

    async def flagless_catalog(*, org_id, user_id, bearer_token, refresh=False):
        return _listing(restricted=False, flags=False)

    monkeypatch.setattr(ep, "fetch_org_model_catalog", flagless_catalog)
    _open_picker(session, MEMBER_A)

    assert _resolved(session, MEMBER_A) == MINDS_FREE_MODEL


def test_the_restriction_list_is_per_member_and_the_map_per_org():
    assert setting_is_org_scoped("minds_model_restricted") is False
    assert setting_is_org_scoped("minds_model_enabled") is True


def test_desktop_keeps_restrictions_in_the_map(session, monkeypatch):
    _mode(monkeypatch, "local")

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        return _listing(restricted=True)

    monkeypatch.setattr(ep, "fetch_minds_models", fake_fetch)
    service = SettingService(session, LOCAL_SCOPE)
    service.upsert_setting("minds_api_key", "mdb_desktop")
    service.upsert_setting("planning_provider", "minds_cloud")
    service.upsert_setting("planning_model", PINNED)
    try:
        asyncio.run(ep.recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        settings = SettingService(session, LOCAL_SCOPE).load()
        assert json.loads(settings.minds_model_enabled) == {MINDS_FREE_MODEL: True, PINNED: False}
        assert SettingService(session, LOCAL_SCOPE)._fetch_row("minds_model_restricted") is None
        assert settings.resolved_planning_model == MINDS_FREE_MODEL
    finally:
        us.get_app_settings.cache_clear()
