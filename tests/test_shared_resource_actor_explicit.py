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


# ── attribution outside org mode ───────────────────────────────────────────

PARTITION_A = "44e92829-e701-599b-88f9-69f5bf1165eb"
SUB_A = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def test_a_personal_scope_reports_no_author_for_its_own_items():
    """A person's own project carries their sub as created_by and a skill's
    creator_id answers the caller outside org mode, so a fallback would name
    the caller with no email, which the web UI renders as "Another member"."""
    from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
    from cowork.principal import Principal

    scope = TenantScope(
        personal=True, org_id=PARTITION_A, user_id=SUB_A, personal_resources=ALL_PERSONAL_RESOURCES
    )
    with Session(_engine()) as raw:
        access = SharedResourceAccess(
            ScopedSession(raw, scope), Principal(user_id=SUB_A, org_id=PARTITION_A)
        )
        creator = access.creator_id("skill", "documents")
        assert creator == SUB_A
        attribution = access.attribution("skill", "documents", fallback_creator_id=creator)
        assert attribution.created_by is None
        assert attribution.last_modified_by is None
        # The operator view of the same row names nobody either.
        operator = SharedResourceAccess(ScopedSession(raw, LOCAL_SCOPE))
        assert operator.attribution("project", "p", fallback_creator_id=SUB_A).created_by is None


def test_a_personal_scope_with_a_live_apps_email_still_names_nobody():
    """With the live-apps overlay the principal carries the person's verified
    email (X-User-Email), which only the live-apps service reads: attribution
    outside org mode reports the same nobody as without it."""
    from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
    from cowork.principal import Principal

    scope = TenantScope(
        personal=True, org_id=PARTITION_A, user_id=SUB_A, personal_resources=ALL_PERSONAL_RESOURCES
    )
    with Session(_engine()) as raw:
        session = ScopedSession(raw, scope)
        without = SharedResourceAccess(session, Principal(user_id=SUB_A, org_id=PARTITION_A))
        with_email = SharedResourceAccess(
            session, Principal(user_id=SUB_A, org_id=PARTITION_A, email="a@example.com")
        )
        for access in (without, with_email):
            attribution = access.attribution("skill", "documents", fallback_creator_id=SUB_A)
            assert (attribution.created_by, attribution.last_modified_by) == (None, None)
        assert with_email.attribution("connection", "postgres/demo", fallback_creator_id="someone-else") == (
            without.attribution("connection", "postgres/demo", fallback_creator_id="someone-else")
        )


def test_org_mode_still_reports_the_fallback_author(org_id):
    """Hosted regression: an unattributed resource still names its creator."""
    from cowork.principal import Principal

    with Session(_engine()) as raw:
        scope = TenantScope(org_mode=True, org_id=org_id, user_id="viewer")
        access = SharedResourceAccess(
            ScopedSession(raw, scope), Principal(user_id="viewer", org_id=org_id, email="v@example.com")
        )
        other = access.attribution("project", str(uuid4()), fallback_creator_id="someone-else")
        assert other.created_by is not None and other.created_by.user_id == "someone-else"
        mine = access.attribution("project", str(uuid4()), fallback_creator_id="viewer")
        assert mine.created_by is not None and mine.created_by.email == "v@example.com"
