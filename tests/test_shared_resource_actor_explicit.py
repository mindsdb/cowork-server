"""Server-owned attribution writes that have no request Principal (ENG-2961).

An artifact is claimed at the end of the turn that wrote it and by the startup
backfill; neither path carries a Principal, so these methods take the actor
explicitly while still stamping and filtering rows by organization.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.shared_resource import (
    SharedResourceAttribution,
    SharedResourceMutation,
)
from cowork.services.shared_resources import SharedResourceAccess

KIND = "artifact"


def _engine():
    return get_engine(get_app_settings().database.uri)


def _access(raw: Session, org_id: str | None) -> SharedResourceAccess:
    return SharedResourceAccess(
        ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
    )


def _events(org_id: str, key: str) -> list[tuple[str, str]]:
    with Session(_engine()) as raw:
        rows = raw.exec(
            select(SharedResourceMutation).where(
                SharedResourceMutation.org_id == org_id,
                SharedResourceMutation.resource_key == key,
            )
        ).all()
        return sorted((row.action, row.actor_id) for row in rows)


def _row(org_id: str, key: str) -> SharedResourceAttribution | None:
    with Session(_engine()) as raw:
        return raw.exec(
            select(SharedResourceAttribution).where(
                SharedResourceAttribution.org_id == org_id,
                SharedResourceAttribution.resource_key == key,
            )
        ).first()


@pytest.fixture
def org_id() -> str:
    return str(uuid4())


def test_claim_as_records_the_creator_without_a_principal(org_id):
    creator = str(uuid4())
    with Session(_engine()) as raw:
        row, created = _access(raw, org_id).claim_as(
            KIND, "p/a", creator_id=creator, action="create"
        )
        assert created is True
        assert row.created_by_id == creator
        assert row.updated_by_id == creator
    stored = _row(org_id, "p/a")
    assert stored.org_id == org_id
    assert stored.resource_kind == KIND
    assert _events(org_id, "p/a") == [("create", creator)]


def test_claim_as_keeps_the_first_writer(org_id):
    first, second = str(uuid4()), str(uuid4())
    with Session(_engine()) as raw:
        _access(raw, org_id).claim_as(KIND, "p/a", creator_id=first, action="create")
    with Session(_engine()) as raw:
        row, created = _access(raw, org_id).claim_as(
            KIND, "p/a", creator_id=second, action="backfill"
        )
        assert created is False
        assert row.created_by_id == first
    assert _events(org_id, "p/a") == [("create", first)]


def test_claim_as_loser_rereads_the_winner(org_id, monkeypatch):
    """The race arbiter is the unique key: a writer that saw no row, then lost
    the insert, returns the winner instead of raising or overwriting. Forced
    deterministically by hiding the winner from the first lookup."""
    winner, loser = str(uuid4()), str(uuid4())
    with Session(_engine()) as raw:
        _access(raw, org_id).claim_as(KIND, "p/race", creator_id=winner, action="create")
    with Session(_engine()) as raw:
        access = _access(raw, org_id)
        real_find = access._find
        calls = {"n": 0}

        def stale_first_find(kind, key):
            calls["n"] += 1
            return None if calls["n"] == 1 else real_find(kind, key)

        monkeypatch.setattr(access, "_find", stale_first_find)
        row, created = access.claim_as(KIND, "p/race", creator_id=loser, action="backfill")
        assert created is False
        assert row.created_by_id == winner
    assert _events(org_id, "p/race") == [("create", winner)]


def test_claim_as_is_a_noop_in_local_mode():
    with Session(_engine()) as raw:
        access = SharedResourceAccess(ScopedSession(raw, LOCAL_SCOPE))
        assert access.claim_as(KIND, "p/a", creator_id="u", action="create") == (None, False)


def test_actor_explicit_writes_require_an_org_id():
    with Session(_engine()) as raw:
        access = _access(raw, None)
        with pytest.raises(RuntimeError, match="organization scope"):
            access.claim_as(KIND, "p/a", creator_id="u", action="create")
        with pytest.raises(RuntimeError, match="organization scope"):
            access.rekey_as(KIND, "p/a", "p/b", actor_id="u")
        with pytest.raises(RuntimeError, match="organization scope"):
            access.delete_as(KIND, "p/a", actor_id="u")


def test_attribution_is_invisible_to_another_org(org_id):
    creator = str(uuid4())
    with Session(_engine()) as raw:
        _access(raw, org_id).claim_as(KIND, "p/a", creator_id=creator, action="create")
    with Session(_engine()) as raw:
        assert _access(raw, str(uuid4())).creator_id(KIND, "p/a") is None


def test_rekey_as_moves_the_row_and_replaces_a_stale_destination(org_id):
    owner, stale_owner, actor = str(uuid4()), str(uuid4()), str(uuid4())
    with Session(_engine()) as raw:
        access = _access(raw, org_id)
        access.claim_as(KIND, "p/a", creator_id=owner, action="create")
        access.claim_as(KIND, "q/a", creator_id=stale_owner, action="create")
    with Session(_engine()) as raw:
        row = _access(raw, org_id).rekey_as(KIND, "p/a", "q/a", actor_id=actor)
        assert row.resource_key == "q/a"
        assert row.created_by_id == owner
        assert row.updated_by_id == actor
    assert _row(org_id, "p/a") is None
    assert _row(org_id, "q/a").created_by_id == owner
    assert _events(org_id, "q/a") == sorted(
        [("create", stale_owner), ("delete", actor), ("move", actor)]
    )


def test_rekey_as_without_a_row_returns_none(org_id):
    with Session(_engine()) as raw:
        assert _access(raw, org_id).rekey_as(KIND, "p/x", "q/x", actor_id="u") is None


def test_delete_as_removes_the_row_and_audits(org_id):
    owner, actor = str(uuid4()), str(uuid4())
    with Session(_engine()) as raw:
        _access(raw, org_id).claim_as(KIND, "p/a", creator_id=owner, action="create")
    with Session(_engine()) as raw:
        assert _access(raw, org_id).delete_as(KIND, "p/a", actor_id=actor) is True
        assert _access(raw, org_id).delete_as(KIND, "p/a", actor_id=actor) is False
    assert _row(org_id, "p/a") is None
    assert _events(org_id, "p/a") == sorted([("create", owner), ("delete", actor)])
