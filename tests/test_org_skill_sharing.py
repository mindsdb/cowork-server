"""Hosted org skill sharing (ENG-3787): who sees, uses and changes a skill.

In hosted org tenancy every member reads one skill store, and who each skill
is shared with is its row in ``skill_access`` (named as the auth service's ArtifactAccess:
``org_allowed``, ``allowed_team_ids``, ``allowed_user_ids``, plus
``always_on``): only its owner (a new skill's row), people by Keycloak id,
teams, or the whole organization. A skill from before sharing gets an
org-wide row once, owned by its attribution's creator, the first time its
org's rows are read; a folder with no row after that is its recorded
creator's alone, or nobody's with none. Org-wide and team skills are
recalled on demand like any other unless marked always on, which puts them
in every turn's prompt of the people they reach.

Handlers are called directly with an explicit SkillViewer, the suite's
convention for org routes (cowork.server wires the identity middleware only
when the process starts in org mode). The auth service is stubbed at
get_auth_json.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import httpx
import pytest
from anton.core.tools.skill_format import dump_skill
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import hub_people as hub_people_ep
from cowork.api.v1.endpoints import hub_teams as hub_teams_ep
from cowork.api.v1.endpoints import skills as skills_ep
from cowork.api.v1.endpoints.artifact_workspace import ARTIFACT_ORG_REFUSAL
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope, unsafe_unscoped_session
from cowork.models.skill import META_CREATED_BY, Skill
from cowork.models.skill_access import SkillAccess
from cowork.principal import ORG_MANAGE_ROLE, Principal
from cowork.models.skill_user_setting import SkillUserSetting
from cowork.schemas.skills import (
    SkillCreateRequest,
    SkillEnabledForMeRequest,
    SkillSharingRequest,
    SkillUpdateRequest,
)
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.services import hub_people, hub_teams
from cowork.services.shared_resources import SKILL, SharedResourceAccess
from cowork.services.skill_access import (
    access_row,
    access_rows,
    stage_access,
    stage_delete,
    stage_only_owner,
    stage_rename,
)
from cowork.services.skill_sharing import (
    HUB_UNAVAILABLE,
    MAX_SHARED_PEOPLE,
    SkillAccessRule,
    SkillViewer,
    access_rule_for,
    get_org_member_lookup,
    normalize_user_ids,
    resolve_skill_viewer,
)
from cowork.services.skills import (
    BUILTIN_SKILL_SLUGS,
    SkillService,
    build_turn_skills,
    link_turn_skills,
)
from cowork.services.skill_user_settings import off_for_me as user_off_for_me
from cowork.services.skill_user_settings import stage_enabled_for_me
from cowork.turnqueue.producer import _fit_request
from tests._org_sharing import (
    ADMIN,
    ALICE,
    BOB,
    CAROL,
    DAVE,
    EMAILS,
    EVE,
    MAL,
    MEMBER_BEARER,
    TEAM_ENG,
    TEAM_OPS,
    access_service_answers,
    artifact_rule,
    member_request,
)

ORG_ID = "33333333-3333-4333-8333-333333333333"

NOTHING_SHARED = {"orgAllowed": False, "allowedTeamIds": [], "allowedUserIds": [], "alwaysOn": False}


def _viewer(
    user: str, *, teams=(), admin=False, share_org=False, write_org=False, known=True, perms_known=True,
) -> SkillViewer:
    return SkillViewer(
        user_id=user,
        email=EMAILS[user],
        team_ids=frozenset(teams),
        teams_known=known,
        is_admin=admin,
        can_share_org=admin or share_org,
        can_write_org=admin or write_org,
        permissions_known=admin or perms_known,
    )


def _principal(user: str, *, admin: bool = False) -> Principal:
    roles = frozenset({ORG_MANAGE_ROLE}) if admin else frozenset()
    return Principal(user_id=user, org_id=ORG_ID, email=EMAILS[user], roles=roles)


@pytest.fixture
def org(org_deployment, monkeypatch, tmp_path):
    """One org store and database; ``as_(user)`` is that member's request session."""
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    sessions: list[ScopedSession] = []

    def as_(user: str, org_id: str = ORG_ID) -> ScopedSession:
        session = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id=org_id, user_id=user))
        sessions.append(session)
        return session

    try:
        yield SimpleNamespace(as_=as_)
    finally:
        for session in sessions:
            session.close()
        get_app_settings.cache_clear()


def _create(org, user: str, label: str, **viewer_kwargs) -> dict:
    return skills_ep.create_skill(
        SkillCreateRequest(label=label, instructions=f"Body of {label}."),
        org.as_(user),
        _principal(user),
        _viewer(user, **viewer_kwargs),
    )


def _members(*user_ids: str, calls: list | None = None):
    """An org member lookup that knows ``user_ids`` (every test user by default)."""
    known = frozenset(user_ids or EMAILS)

    async def lookup(asked):
        asked = list(asked)
        if calls is not None:
            calls.append(asked)
        return frozenset(u for u in asked if u in known)

    return lookup


def _share(
    ctx, user: str, slug: str, viewer: SkillViewer | None = None, *, admin=False, members=None, **body,
) -> dict:
    """PUT /skills/{slug}/sharing as ``user``, running the org membership
    check (verify_people_in_org) first, as the route's dependency does."""
    request = SkillSharingRequest(**body)
    scoped = ctx.as_(user)
    viewer = viewer or _viewer(user, teams=(TEAM_ENG,), admin=admin)
    stored = asyncio.run(skills_ep.stored_sharing_rule(slug, scoped, viewer))
    verified = asyncio.run(
        skills_ep.verify_people_in_org(request, viewer, stored, members or _members())
    )
    return skills_ep.set_skill_sharing(slug, request, scoped, _principal(user, admin=admin), viewer, verified)


def _listed(org, user: str, viewer: SkillViewer) -> dict[str, dict]:
    listing = skills_ep.list_skills(org.as_(user), _principal(user), viewer)
    return {s["id"]: s for s in listing["skills"] if s["id"] not in BUILTIN_SKILL_SLUGS}


def _refused(call, status: int, detail: str | None = None) -> None:
    with pytest.raises(HTTPException) as caught:
        call()
    assert caught.value.status_code == status
    if detail is not None:
        assert detail in caught.value.detail


def _legacy_skill(scope: TenantScope, slug: str, body: str = "Legacy body.", **metadata) -> None:
    """A skill written straight to the store, with no skill_access row: one
    from before sharing, or one an agent wrote."""
    root = SkillService(scope).root / slug
    root.mkdir(parents=True)
    skill = Skill(name=slug, description=slug, instructions=body, metadata=metadata)
    (root / "SKILL.md").write_text(dump_skill(skill))


def _row(org, slug: str) -> SkillAccess | None:
    return access_row(org.as_(ALICE), slug)


def _store_rule(org, slug: str, rule: SkillAccessRule) -> None:
    session = org.as_(ALICE)
    stage_access(session, slug, rule)
    session.commit()


# ── the table ────────────────────────────────────────────────────────────────


def test_the_table_keeps_one_row_per_skill_per_org(org):
    session = org.as_(ALICE)
    stage_only_owner(session, "notes", ALICE)
    session.commit()
    row = access_row(session, "notes")
    assert (row.org_id, row.owner_user_id, row.org_allowed, row.allowed_team_ids, row.allowed_user_ids,
            row.always_on) == (ORG_ID, ALICE, False, [], [], False)
    assert row.created_at is not None and row.modified_at is not None

    # A rule saved again updates the same row; always_on only sticks on a
    # skill shared org-wide or with a team.
    stage_access(session, "notes", SkillAccessRule(
        owner_user_id=ALICE, allowed_team_ids=(TEAM_ENG,), allowed_user_ids=(BOB,), always_on=True,
    ))
    session.commit()
    stage_access(session, "solo", SkillAccessRule(owner_user_id=ALICE, allowed_user_ids=(BOB,), always_on=True))
    session.commit()
    rows = access_rows(session)
    assert set(rows) == {"notes", "solo"}
    assert (rows["notes"].allowed_team_ids, rows["notes"].allowed_user_ids, rows["notes"].always_on) == (
        [TEAM_ENG], [BOB], True,
    )
    assert rows["solo"].always_on is False

    # Another org neither sees nor collides with these rows.
    other = org.as_(BOB, org_id="44444444-4444-4444-8444-444444444444")
    assert access_rows(other) == {}
    stage_only_owner(other, "notes", BOB)
    other.commit()
    assert access_row(org.as_(ALICE), "notes").owner_user_id == ALICE

    stage_rename(session, "notes", "solo")  # drops the stale row the new name had
    session.commit()
    assert set(access_rows(session)) == {"solo"}
    assert access_rows(session)["solo"].allowed_team_ids == [TEAM_ENG]
    stage_delete(session, "solo")
    session.commit()
    assert access_rows(session) == {}
    assert set(access_rows(other)) == {"notes"}


def test_a_skill_with_no_row_is_its_recorded_creators_or_nobodys():
    """Its SKILL.md's created_by decides nothing: the row does, or with no
    row the creator its attribution records."""
    assert access_rule_for(None) == SkillAccessRule()
    assert access_rule_for(None, ALICE) == SkillAccessRule(owner_user_id=ALICE)
    row = SkillAccess(org_id=ORG_ID, skill_slug="s", owner_user_id=BOB, org_allowed=True, always_on=True)
    assert access_rule_for(row) == SkillAccessRule(org_allowed=True, always_on=True, owner_user_id=BOB)
    assert normalize_user_ids([f" {ALICE.upper()} ", ALICE, ""]) == (ALICE,)
    with pytest.raises(ValueError):
        normalize_user_ids(["a,b"])


# ── creating and listing ─────────────────────────────────────────────────────


def test_a_new_skill_is_only_its_creators(org):
    created = _create(org, ALICE, "alice-notes", teams=(TEAM_ENG,))

    assert created["section"] == "mine"
    assert created["access"] == NOTHING_SHARED
    assert "sharing" not in created
    assert created["canEdit"] is True
    assert created["owner"] == {"id": ALICE, "email": "alice@x.com"}
    row = _row(org, "alice-notes")
    assert (row.owner_user_id, row.org_allowed, row.allowed_team_ids, row.allowed_user_ids) == (ALICE, False, [], [])
    # Nothing about access or ownership is in SKILL.md: the owner is the
    # row's, and the creator the attribution's.
    metadata = SkillService(org.as_(ALICE).scope).get_skill("alice-notes").metadata
    assert not {META_CREATED_BY, "share_org", "share_teams", "share_emails", "always_on", "created_by_email"} & set(
        metadata
    )
    assert "alice-notes" not in _listed(org, BOB, _viewer(BOB, teams=(TEAM_ENG,)))
    _refused(lambda: skills_ep.get_skill("alice-notes", org.as_(BOB), _principal(BOB), _viewer(BOB)), 404)


def test_an_upload_gets_a_row_shared_with_nobody_whatever_its_file_claims(org):
    skill = Skill(
        name="uploaded", description="u", instructions="x",
        metadata={"share_org": "true", META_CREATED_BY: DAVE, "share_teams": TEAM_OPS, "always_on": "true"},
    )
    uploaded = skills_ep._upload_skill_bytes(
        dump_skill(skill).encode(), "SKILL.md", org.as_(ALICE), _principal(ALICE), viewer=_viewer(ALICE),
    )
    assert (uploaded["section"], uploaded["access"]) == ("mine", NOTHING_SHARED)
    # The file's claimed creator is dropped, not kept as a second owner.
    assert SkillService(org.as_(ALICE).scope).get_skill("uploaded").created_by is None
    assert _row(org, "uploaded").owner_user_id == ALICE
    assert "uploaded" not in _listed(org, DAVE, _viewer(DAVE, teams=(TEAM_OPS,)))


# ── skills from before sharing ───────────────────────────────────────────────


def _attributed(org, slug: str, creator: str | None, *, pending: bool = False, expired: bool = False) -> None:
    """The SharedResourceAttribution a skill created through the API has:
    its creator, or a claim still pending for a create in flight, or, with
    ``expired``, for one a crash cut short."""
    session = org.as_(ALICE)
    session.add(SharedResourceAttribution(
        resource_kind=SKILL, resource_key=slug, created_by_id=creator, updated_by_id=creator,
        pending_claim_token=str(uuid4()) if pending else None,
        pending_claim_expires_at=(
            datetime.now(timezone.utc) + timedelta(minutes=-1 if expired else 5) if pending else None
        ),
    ))
    session.commit()


def test_a_skill_from_before_sharing_gets_an_org_wide_row_once(org):
    """The org's first read of its rows gives each skill of the store with no
    row an org-wide one (what every member saw before sharing), owned by its
    attribution's creator; built-ins get none. Once: a folder written around
    the API afterwards is nobody's."""
    alice = org.as_(ALICE).scope
    _legacy_skill(alice, "house-style")
    _legacy_skill(alice, "alice-guide")
    _attributed(org, "alice-guide", ALICE)

    seen = _listed(org, BOB, _viewer(BOB))["house-style"]
    assert (seen["section"], seen["access"]["orgAllowed"], seen["access"]["alwaysOn"], seen["canEdit"]) == (
        "org", True, False, False,
    )
    # Changing an org-wide skill someone else owns takes skills.write_org.
    assert _listed(org, BOB, _viewer(BOB, write_org=True))["house-style"]["canEdit"] is True
    rows = access_rows(org.as_(ALICE))
    assert {slug: (row.org_allowed, row.owner_user_id, row.always_on) for slug, row in rows.items()} == {
        "house-style": (True, None, False), "alice-guide": (True, ALICE, False),
    }
    assert "house-style" in build_turn_skills(
        org.as_(BOB).scope, project_name="general", viewer=_viewer(BOB), session=org.as_(BOB),
    )

    _legacy_skill(alice, "written-later")
    assert "written-later" not in _listed(org, BOB, _viewer(BOB, write_org=True))
    assert _row(org, "written-later") is None
    assert _listed(org, ADMIN, _viewer(ADMIN, admin=True))["written-later"]["owner"]["id"] is None


def test_an_uploaded_skill_from_before_sharing_stays_org_wide_and_its_creator_keeps_it(org):
    """Its SKILL.md names another creator, which decides nothing: the
    attribution's creator owns it. They edit it, take it back from everyone
    and delete it with neither skills.write_org nor skills.share_org; a
    member who isn't its owner can't edit it."""
    alice_scope = org.as_(ALICE).scope
    _legacy_skill(alice_scope, "uploaded-guide", created_by=EVE)
    _attributed(org, "uploaded-guide", ALICE)
    _legacy_skill(alice_scope, "uploaded-notes", created_by=EVE)
    _attributed(org, "uploaded-notes", ALICE)
    alice = _viewer(ALICE)

    carol = _listed(org, CAROL, _viewer(CAROL))["uploaded-guide"]
    assert (carol["section"], carol["access"]["orgAllowed"], carol["owner"]["id"], carol["canEdit"]) == (
        "org", True, ALICE, False,
    )
    _refused(
        lambda: skills_ep.update_skill(
            "uploaded-guide", SkillUpdateRequest(instructions="Bob's edit."), org.as_(BOB), _principal(BOB),
            _viewer(BOB),
        ),
        403,
        skills_ep.ORG_WIDE_EDIT_REFUSAL,
    )
    assert _listed(org, ALICE, alice)["uploaded-guide"]["canEdit"] is True
    edited = skills_ep.update_skill(
        "uploaded-guide", SkillUpdateRequest(instructions="Alice's edit."), org.as_(ALICE), _principal(ALICE), alice,
    )
    assert edited["declarative"] == "Alice's edit."

    out = _share(org, ALICE, "uploaded-guide", alice, org_allowed=False)
    assert (out["access"]["orgAllowed"], out["section"]) == (False, "mine")
    assert "uploaded-guide" not in _listed(org, CAROL, _viewer(CAROL))
    # Making it org-wide again still takes skills.share_org.
    _refused(lambda: _share(org, ALICE, "uploaded-guide", alice, org_allowed=True), 403, "skills.share_org")

    skills_ep.delete_skill("uploaded-notes", org.as_(ALICE), _principal(ALICE), alice)
    assert _row(org, "uploaded-notes") is None
    assert "uploaded-notes" not in _listed(org, CAROL, _viewer(CAROL))


def test_two_backfills_of_one_org_at_once_write_each_row_once(org, monkeypatch):
    """Two replicas reading the org for the first time at once both find no
    settings row and both backfill: the unique keys decide, each skill gets
    one row, the org one settings row, and neither read fails."""
    import cowork.services.skill_access as skill_access
    from cowork.models.setting import Setting

    _legacy_skill(org.as_(ALICE).scope, "house-style")
    _attributed(org, "house-style", ALICE)
    # Each read the org's rows and its settings before either wrote.
    monkeypatch.setattr(skill_access, "_backfilled", lambda session: False)
    real_rows = skill_access.access_rows
    monkeypatch.setattr(skill_access, "access_rows", lambda session: {})

    skill_access.backfill_legacy_access(org.as_(ALICE), lambda: ["house-style"])
    skill_access.backfill_legacy_access(org.as_(BOB), lambda: ["house-style"])
    monkeypatch.setattr(skill_access, "access_rows", real_rows)

    session = org.as_(ALICE)
    rows = session.exec(session.select(SkillAccess)).all()
    assert [(row.skill_slug, row.org_allowed, row.owner_user_id) for row in rows] == [("house-style", True, ALICE)]
    marks = session.exec(session.select(Setting).where(Setting.key == skill_access.BACKFILL_SETTING_KEY)).all()
    assert [(mark.scope, mark.org_id) for mark in marks] == [("org", ORG_ID)]


def test_a_creator_id_too_long_for_the_row_leaves_the_skill_ownerless(org):
    """An attribution id longer than skill_access.owner_user_id holds (one
    the database has but the model would refuse) leaves the skill org-wide
    with no owner, rather than failing the org's first read every time."""
    _legacy_skill(org.as_(ALICE).scope, "old-skill")
    session = org.as_(ALICE)
    unsafe_unscoped_session(session).execute(
        text(
            "INSERT INTO shared_resource_attributions (id, org_id, resource_kind, resource_key, created_by_id)"
            " VALUES (:id, :org, :kind, :key, :creator)"
        ),
        {"id": uuid4().hex, "org": ORG_ID, "kind": SKILL, "key": "old-skill", "creator": "x" * 40},
    )
    session.commit()

    assert _listed(org, BOB, _viewer(BOB))["old-skill"]["section"] == "org"
    assert _row(org, "old-skill").owner_user_id is None


def test_the_backfill_leaves_a_create_in_flight_to_its_create(org):
    _legacy_skill(org.as_(ALICE).scope, "being-made")
    _attributed(org, "being-made", ALICE, pending=True)

    assert "being-made" not in _listed(org, BOB, _viewer(BOB, write_org=True))
    assert _row(org, "being-made") is None


def test_a_create_a_crash_cut_short_is_its_creators_and_recovery_gives_it_its_row(org):
    """A create that wrote the skill and crashed before its commit leaves
    the claim pending and no row, and the backfill skips it. The skill is
    still its creator's alone, so their listing reaches it, and once the
    claim's lease runs out the recovery that finalizes it commits the row
    the create would have: shared with nobody but its creator."""
    _legacy_skill(org.as_(ALICE).scope, "crashed")
    _attributed(org, "crashed", ALICE, pending=True, expired=True)

    assert "crashed" not in _listed(org, BOB, _viewer(BOB, write_org=True))
    assert _row(org, "crashed") is None

    seen = _listed(org, ALICE, _viewer(ALICE))["crashed"]
    assert (seen["section"], seen["access"], seen["canEdit"]) == ("mine", NOTHING_SHARED, True)
    row = _row(org, "crashed")
    assert (row.owner_user_id, row.org_allowed, row.allowed_team_ids, row.allowed_user_ids) == (ALICE, False, [], [])
    assert "crashed" not in _listed(org, BOB, _viewer(BOB, write_org=True))


def test_a_skill_made_after_the_backfill_with_no_row_is_its_creators(org):
    """A server from before sharing, still running during the rollout,
    creates a skill after the org's backfill: it records the creator but
    writes no row. The skill is its creator's alone, not nobody's: they see
    and change it, share it, and nobody else sees it until they do."""
    import cowork.services.skill_access as skill_access

    _create(org, BOB, "bob-notes")  # the org's first read: its backfill runs
    assert skill_access._backfilled(org.as_(BOB))
    _legacy_skill(org.as_(ALICE).scope, "made-during-rollout")
    _attributed(org, "made-during-rollout", ALICE)

    seen = _listed(org, ALICE, _viewer(ALICE))["made-during-rollout"]
    assert (seen["section"], seen["access"], seen["canEdit"], seen["owner"]["id"]) == (
        "mine", NOTHING_SHARED, True, ALICE,
    )
    assert "made-during-rollout" not in _listed(org, BOB, _viewer(BOB, write_org=True))
    assert _listed(org, ADMIN, _viewer(ADMIN, admin=True))["made-during-rollout"]["owner"]["id"] == ALICE

    _share(org, ALICE, "made-during-rollout", _viewer(ALICE), allowed_user_ids=[BOB])
    assert _row(org, "made-during-rollout").owner_user_id == ALICE
    assert "made-during-rollout" in _listed(org, BOB, _viewer(BOB))


def test_a_new_skill_is_never_org_wide_while_it_is_being_created(org, monkeypatch):
    """The org's first read can land right after a create finalizes its
    claim. Its row commits with the claim, so the backfill never takes a new
    skill for one from before sharing and shares it with everyone."""
    real = SharedResourceAccess.finalize_claim
    seen: list[dict] = []

    def finalize_then_read(self, *args, **kwargs):
        finalized = real(self, *args, **kwargs)
        seen.append(_listed(org, CAROL, _viewer(CAROL)))
        return finalized

    monkeypatch.setattr(SharedResourceAccess, "finalize_claim", finalize_then_read)
    _create(org, ALICE, "brand-new")

    assert "brand-new" not in seen[0]
    row = _row(org, "brand-new")
    assert (row.org_allowed, row.owner_user_id) == (False, ALICE)


def test_the_listing_reads_the_orgs_rows_in_one_query(org, monkeypatch):
    for label in ("one", "two", "three"):
        _create(org, ALICE, label)
    import cowork.services.skill_access as skill_access

    calls: list[int] = []
    real = skill_access.access_rows

    def counted(session):
        calls.append(1)
        return real(session)

    monkeypatch.setattr(skill_access, "access_rows", counted)
    monkeypatch.setattr(skill_access, "access_row", lambda *a: pytest.fail("one query per skill"))
    assert {"one", "two", "three"} <= set(_listed(org, ALICE, _viewer(ALICE)))
    assert calls == [1]


def test_each_member_sees_a_skill_in_their_own_section(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], allowed_user_ids=[CAROL])

    # Its owner lists it by its audience too: shared with a team.
    assert _listed(org, ALICE, _viewer(ALICE, teams=(TEAM_ENG,)))["team-skill"]["section"] == "team"
    bob = _listed(org, BOB, _viewer(BOB, teams=(TEAM_ENG,)))["team-skill"]
    assert (bob["section"], bob["canEdit"], bob["owner"]["id"]) == ("team", True, ALICE)
    # Recipients see who owns it, by id (named through /hub/people/?ids=),
    # but only the owner or an admin sees the people.
    assert (bob["owner"]["email"], bob["access"]["allowedUserIds"]) == (None, [])
    carol = _listed(org, CAROL, _viewer(CAROL))["team-skill"]
    assert (carol["section"], carol["canEdit"]) == ("shared_with_me", False)
    assert "team-skill" not in _listed(org, DAVE, _viewer(DAVE, teams=(TEAM_OPS,)))
    # An admin sees and changes every skill, but their turns don't use
    # another member's private one.
    _create(org, DAVE, "dave-private")
    admin = _viewer(ADMIN, admin=True)
    seen = _listed(org, ADMIN, admin)["dave-private"]
    assert (seen["section"], seen["canEdit"]) == ("shared_with_me", True)
    assert "dave-private" not in build_turn_skills(
        org.as_(ADMIN).scope, project_name="general", viewer=admin, session=org.as_(ADMIN),
    )


def test_a_person_is_matched_by_user_id_not_email(org):
    _create(org, ALICE, "for-bob")
    _share(org, ALICE, "for-bob", allowed_user_ids=[BOB])

    assert "for-bob" in _listed(org, BOB, _viewer(BOB))
    # Bob's address on someone else's viewer reaches nothing.
    assert "for-bob" not in _listed(org, CAROL, replace(_viewer(CAROL), email="bob@x.com"))


def test_an_unreachable_team_listing_hides_team_only_skills(org):
    _create(org, ALICE, "team-only")
    _share(org, ALICE, "team-only", allowed_team_ids=[TEAM_ENG])
    _create(org, ALICE, "for-bob")
    _share(org, ALICE, "for-bob", allowed_user_ids=[BOB])

    blind = _viewer(BOB, known=False)
    assert set(_listed(org, BOB, blind)) == {"for-bob"}
    assert "team-only" not in build_turn_skills(
        org.as_(BOB).scope, project_name="general", viewer=blind, session=org.as_(BOB),
    )


def test_the_viewer_resolves_teams_and_permissions_from_the_access_service_and_fails_closed(monkeypatch):
    hub_teams.reset_caches_for_tests()
    answers = {
        hub_teams.TEAMS_PATH: {"results": [{"id": TEAM_ENG, "display_name": "Eng", "is_default": False}]},
        hub_teams.ENTITLEMENTS_PATH: {"permissions": ["skills.share_org"]},
    }

    async def get_auth_json(path, bearer):
        return answers.get(path)

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=BOB)
    viewer = asyncio.run(resolve_skill_viewer(scope=scope, principal=_principal(BOB), bearer_token="t"))
    assert (viewer.team_ids, viewer.teams_known, viewer.can_share_org, viewer.can_write_org) == (
        frozenset({TEAM_ENG}), True, True, False,
    )
    assert viewer.email == "bob@x.com"

    hub_teams.reset_caches_for_tests()
    answers.clear()
    blind = asyncio.run(resolve_skill_viewer(scope=scope, principal=_principal(BOB), bearer_token="t"))
    assert (blind.team_ids, blind.teams_known, blind.can_share_org) == (frozenset(), False, False)
    admin = asyncio.run(resolve_skill_viewer(
        scope=TenantScope(org_mode=True, org_id=ORG_ID, user_id=ADMIN),
        principal=_principal(ADMIN, admin=True), bearer_token="t",
    ))
    assert (admin.can_share_org, admin.can_write_org) == (True, True)
    assert asyncio.run(resolve_skill_viewer(scope=LOCAL_SCOPE, principal=None, bearer_token="t")) is None
    hub_teams.reset_caches_for_tests()


# ── PUT /skills/{slug}/sharing ───────────────────────────────────────────────


def test_sharing_normalizes_people_and_keeps_teams(org):
    _create(org, ALICE, "shared")
    out = _share(
        org, ALICE, "shared",
        allowed_team_ids=[TEAM_ENG, TEAM_ENG], allowed_user_ids=[f" {CAROL.upper()}", CAROL, BOB],
    )

    assert out["access"] == {
        "orgAllowed": False, "allowedTeamIds": [TEAM_ENG], "allowedUserIds": [CAROL, BOB], "alwaysOn": False,
    }
    row = _row(org, "shared")
    assert (row.allowed_team_ids, row.allowed_user_ids) == ([TEAM_ENG], [CAROL, BOB])
    _refused(lambda: _share(org, ALICE, "shared", allowed_user_ids=["not an id"]), 400, "not a valid person id")


def test_only_the_owner_or_an_admin_changes_teams_and_people(org):
    _create(org, ALICE, "shared")
    _share(org, ALICE, "shared", allowed_team_ids=[TEAM_ENG])

    _refused(lambda: _share(org, BOB, "shared", allowed_user_ids=[DAVE]), 403, "owner or an organization admin")
    admin = _viewer(ADMIN, admin=True)
    out = _share(org, ADMIN, "shared", admin, admin=True, allowed_team_ids=[TEAM_ENG, TEAM_OPS])
    # An admin may name a team they aren't in, and the owner stays the owner.
    assert out["access"]["allowedTeamIds"] == [TEAM_ENG, TEAM_OPS]
    assert _row(org, "shared").owner_user_id == ALICE


def test_a_team_must_be_one_the_sharer_belongs_to(org):
    _create(org, ALICE, "shared")

    _refused(lambda: _share(org, ALICE, "shared", allowed_team_ids=[TEAM_OPS]), 400, "teams you belong to")
    blind = _viewer(ALICE, known=False)
    _refused(lambda: _share(org, ALICE, "shared", blind, allowed_team_ids=[TEAM_ENG]), 503)
    # Keeping a team already listed needs no listing.
    _share(org, ALICE, "shared", allowed_team_ids=[TEAM_ENG])
    _share(org, ALICE, "shared", blind, allowed_team_ids=[TEAM_ENG], allowed_user_ids=[BOB])


def test_org_wide_takes_the_share_org_capability(org):
    _create(org, ALICE, "shared")

    _refused(lambda: _share(org, ALICE, "shared", org_allowed=True), 403, "skills.share_org")
    out = _share(org, ALICE, "shared", _viewer(ALICE, share_org=True), org_allowed=True)
    # Its owner's own skill moves to the org section once org-wide.
    assert (out["access"]["orgAllowed"], out["section"]) == (True, "org")
    assert _listed(org, ALICE, _viewer(ALICE, share_org=True))["shared"]["section"] == "org"
    assert _row(org, "shared").org_allowed is True
    # Its owner takes it back without the capability.
    out = _share(org, ALICE, "shared", _viewer(ALICE), org_allowed=False)
    assert (out["access"]["orgAllowed"], _row(org, "shared").org_allowed) == (False, False)


def test_org_wide_also_takes_owning_the_skill_or_admin(org):
    """skills.share_org alone doesn't let someone a skill was shared with
    make it org-wide, or take that back: the owner or an admin decides."""
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], allowed_user_ids=[BOB])
    bob = _viewer(BOB, teams=(TEAM_ENG,), share_org=True)

    _refused(
        lambda: _share(org, BOB, "team-skill", bob, org_allowed=True), 403, skills_ep.SHARING_OWNER_REFUSAL,
    )
    assert _row(org, "team-skill").org_allowed is False
    out = _share(org, ADMIN, "team-skill", _viewer(ADMIN, admin=True, share_org=True), admin=True, org_allowed=True)
    assert out["access"]["orgAllowed"] is True
    _refused(
        lambda: _share(org, BOB, "team-skill", bob, org_allowed=False), 403, skills_ep.SHARING_OWNER_REFUSAL,
    )
    out = _share(org, ALICE, "team-skill", _viewer(ALICE, teams=(TEAM_ENG,), share_org=True), org_allowed=False)
    assert out["access"]["orgAllowed"] is False


def test_an_owners_skill_is_listed_by_its_audience(org):
    """The owner's section follows who the skill reaches: only them or
    people is "mine", a team is "team", org-wide is "org"."""
    _create(org, ALICE, "mine")
    alice = _viewer(ALICE, teams=(TEAM_ENG,), share_org=True)
    assert _listed(org, ALICE, alice)["mine"]["section"] == "mine"
    out = _share(org, ALICE, "mine", alice, allowed_user_ids=[BOB])
    assert out["section"] == "mine"
    out = _share(org, ALICE, "mine", alice, allowed_team_ids=[TEAM_ENG], allowed_user_ids=[BOB])
    assert out["section"] == "team"
    assert _listed(org, ALICE, alice)["mine"]["section"] == "team"
    out = _share(org, ALICE, "mine", alice, org_allowed=True, allowed_team_ids=[TEAM_ENG])
    assert out["section"] == "org"
    assert _listed(org, ALICE, alice)["mine"]["section"] == "org"
    out = _share(org, ALICE, "mine", alice, org_allowed=False, allowed_team_ids=[])
    assert out["section"] == "mine"


TEAM_ALL = "team-everyone"


def _with_default(viewer: SkillViewer) -> SkillViewer:
    """``viewer`` in an org whose default team (everyone) is TEAM_ALL."""
    return replace(viewer, team_ids=viewer.team_ids | {TEAM_ALL}, default_team_id=TEAM_ALL)


def test_sharing_with_the_default_team_is_sharing_org_wide(org):
    _create(org, ALICE, "shared")

    member = _with_default(_viewer(ALICE, teams=(TEAM_ENG,)))
    _refused(lambda: _share(org, ALICE, "shared", member, allowed_team_ids=[TEAM_ALL]), 403, "skills.share_org")
    _refused(
        lambda: _share(org, ALICE, "shared", member, allowed_team_ids=[TEAM_ENG, TEAM_ALL]), 403, "skills.share_org",
    )
    assert _row(org, "shared").org_allowed is False

    sharer = _with_default(_viewer(ALICE, teams=(TEAM_ENG,), share_org=True))
    out = _share(org, ALICE, "shared", sharer, allowed_team_ids=[TEAM_ALL, TEAM_ENG])
    assert (out["access"]["orgAllowed"], out["access"]["allowedTeamIds"]) == (True, [TEAM_ENG])
    row = _row(org, "shared")
    assert (row.org_allowed, row.allowed_team_ids) == (True, [TEAM_ENG])


def test_a_stored_default_team_share_reads_as_org_wide(org):
    _create(org, ALICE, "everyone")
    _store_rule(org, "everyone", SkillAccessRule(owner_user_id=ALICE, allowed_team_ids=(TEAM_ALL,), always_on=True))
    bob = _with_default(_viewer(BOB))
    seen = _listed(org, BOB, bob)["everyone"]
    assert (seen["section"], seen["access"]["orgAllowed"], seen["access"]["allowedTeamIds"]) == ("org", True, [])
    # Org-wide's edit rule: a member of the default team alone doesn't change it.
    assert seen["canEdit"] is False
    assert _listed(org, BOB, replace(bob, can_write_org=True))["everyone"]["canEdit"] is True
    assert build_turn_skills(org.as_(BOB).scope, project_name="general", viewer=bob, session=org.as_(BOB))[
        "everyone"
    ]["always_on"] is True
    # Rewritten as org-wide the next time its access is saved.
    _share(org, ALICE, "everyone", _with_default(_viewer(ALICE)), allowed_user_ids=[CAROL])
    row = _row(org, "everyone")
    assert (row.org_allowed, row.allowed_team_ids, row.allowed_user_ids) == (True, [], [CAROL])


def test_the_viewer_learns_the_default_team_from_the_access_service(monkeypatch):
    hub_teams.reset_caches_for_tests()

    async def get_auth_json(path, bearer):
        if path == hub_teams.TEAMS_PATH:
            return {"results": [{"id": TEAM_ENG}, {"id": TEAM_ALL, "is_default": True}]}
        return {"permissions": []}

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    viewer = asyncio.run(resolve_skill_viewer(
        scope=TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE),
        principal=_principal(ALICE), bearer_token="t",
    ))
    assert viewer.default_team_id == TEAM_ALL
    hub_teams.reset_caches_for_tests()


def test_an_org_mode_shared_only_body_changes_nothing(org):
    """The personal-partition body, {"shared": bool}, reaches the org route as
    an audience with nothing set, which keeps what the skill has."""
    _create(org, ALICE, "shared")
    out = _share(org, ALICE, "shared", shared=True)
    assert out["access"] == NOTHING_SHARED


def test_a_skill_the_caller_cannot_see_or_a_builtin_is_refused(org):
    _create(org, DAVE, "dave-private")
    _refused(lambda: _share(org, ALICE, "dave-private", org_allowed=False), 404)
    SkillService(org.as_(ALICE).scope).ensure_builtin_skills()
    builtin = sorted(BUILTIN_SKILL_SLUGS)[0]
    _refused(lambda: _share(org, ADMIN, builtin, _viewer(ADMIN, admin=True), admin=True, org_allowed=False), 403)


# ── changing a shared skill ──────────────────────────────────────────────────


def test_team_members_change_a_team_skill_and_recipients_only_read(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], allowed_user_ids=[CAROL])

    edited = skills_ep.update_skill(
        "team-skill", SkillUpdateRequest(instructions="Bob's edit."), org.as_(BOB), _principal(BOB),
        _viewer(BOB, teams=(TEAM_ENG,)),
    )
    assert edited["declarative"] == "Bob's edit."
    _refused(
        lambda: skills_ep.update_skill(
            "team-skill", SkillUpdateRequest(enabled=False), org.as_(CAROL), _principal(CAROL), _viewer(CAROL),
        ),
        403,
        skills_ep.SKILL_EDIT_REFUSAL,
    )
    _refused(lambda: skills_ep.delete_skill("team-skill", org.as_(CAROL), _principal(CAROL), _viewer(CAROL)), 403)
    _refused(lambda: skills_ep.delete_skill("team-skill", org.as_(DAVE), _principal(DAVE), _viewer(DAVE)), 404)


def test_a_rename_moves_the_row_and_a_delete_removes_it(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], allowed_user_ids=[CAROL])
    bob = _viewer(BOB, teams=(TEAM_ENG,))

    renamed = skills_ep.update_skill(
        "team-skill", SkillUpdateRequest(label="eng-skill"), org.as_(BOB), _principal(BOB), bob,
    )
    assert renamed["access"]["allowedTeamIds"] == [TEAM_ENG]
    assert _row(org, "team-skill") is None
    row = _row(org, "eng-skill")
    assert (row.owner_user_id, row.allowed_team_ids, row.allowed_user_ids) == (ALICE, [TEAM_ENG], [CAROL])
    assert "eng-skill" in _listed(org, CAROL, _viewer(CAROL))

    skills_ep.delete_skill("eng-skill", org.as_(BOB), _principal(BOB), bob)
    assert _row(org, "eng-skill") is None
    # A new skill of the old name starts over as its creator's alone.
    _create(org, DAVE, "eng-skill")
    assert "eng-skill" not in _listed(org, CAROL, _viewer(CAROL))
    assert _row(org, "eng-skill").owner_user_id == DAVE


def test_an_org_wide_skill_takes_write_org_to_change_unless_you_own_it(org):
    _create(org, ALICE, "policy")
    _share(org, ALICE, "policy", _viewer(ALICE, share_org=True), org_allowed=True)

    def edit(user, viewer):
        return skills_ep.update_skill(
            "policy", SkillUpdateRequest(instructions=f"{user} edit"), org.as_(user), _principal(user), viewer,
        )

    # Its owner keeps changing it, org-wide or not.
    assert edit(ALICE, _viewer(ALICE))["declarative"] == f"{ALICE} edit"
    _refused(lambda: edit(CAROL, _viewer(CAROL)), 403, "skills.write_org")
    _refused(
        lambda: skills_ep.update_skill(
            "policy", SkillUpdateRequest(enabled=False), org.as_(BOB), _principal(BOB), _viewer(BOB),
        ),
        403,
        "skills.write_org",
    )
    assert edit(BOB, _viewer(BOB, write_org=True))["declarative"] == f"{BOB} edit"
    assert edit(ADMIN, _viewer(ADMIN, admin=True))["declarative"] == f"{ADMIN} edit"


# ── concurrent changes ───────────────────────────────────────────────────────


def _race_on(monkeypatch, lock_name: str, slug: str, other_change) -> None:
    """Run ``other_change`` to its commit while the first request to take the
    skill's ``lock_name`` lock (coordination_lock or mutation_lock) waits on
    it, after that request read the skill's row."""
    real = getattr(SharedResourceAccess, lock_name)
    ran: list[bool] = []

    @contextmanager
    def lock(self, namespace, key, *args, **kwargs):
        if namespace == SKILL and key == slug and not ran:
            ran.append(True)
            other_change()
        with real(self, namespace, key, *args, **kwargs) as held:
            yield held

    monkeypatch.setattr(SharedResourceAccess, lock_name, lock)


def test_the_rows_are_read_from_the_database_not_the_sessions_copy(org):
    """A session that still holds a skill's row reads another session's
    committed change, not the copy it loaded before: what a re-check under a
    skill's lock needs, whoever else holds the row."""
    _create(org, ALICE, "shared")
    one, many = org.as_(ALICE), org.as_(ALICE)
    held = (access_row(one, "shared"), access_rows(many)["shared"])
    other = org.as_(ALICE)
    stage_access(other, "shared", SkillAccessRule(owner_user_id=ALICE, allowed_team_ids=(TEAM_ENG,)))
    other.commit()

    assert access_row(one, "shared").allowed_team_ids == [TEAM_ENG]
    assert access_rows(many)["shared"].allowed_team_ids == [TEAM_ENG]
    assert [row.allowed_team_ids for row in held] == [[TEAM_ENG], [TEAM_ENG]]


def test_two_concurrent_sharing_changes_both_land(org, monkeypatch):
    """Alice saves the team again from one tab while another tab's save, which
    took the team off and added Carol, commits first. The later save reads
    the row again under the skill's lock: read from before it waited, the
    team already looked shared, nothing was written, and the reply claimed a
    team the row no longer had."""
    _create(org, ALICE, "shared")
    _share(org, ALICE, "shared", allowed_team_ids=[TEAM_ENG])
    _race_on(
        monkeypatch, "coordination_lock", "shared",
        lambda: _share(org, ALICE, "shared", allowed_team_ids=[], allowed_user_ids=[CAROL]),
    )

    out = _share(org, ALICE, "shared", allowed_team_ids=[TEAM_ENG])

    row = _row(org, "shared")
    assert (row.allowed_team_ids, row.allowed_user_ids) == ([TEAM_ENG], [CAROL])
    assert (out["access"]["allowedTeamIds"], out["access"]["allowedUserIds"]) == ([TEAM_ENG], [CAROL])


def test_an_edit_rechecks_a_sharing_change_made_while_it_waited(org, monkeypatch):
    """Bob edits a skill shared with his team; while he waits on its lock,
    its owner takes it back from the team. The re-check under the lock reads
    the row again and refuses him."""
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG])
    _race_on(
        monkeypatch, "mutation_lock", "team-skill",
        lambda: _share(org, ALICE, "team-skill", allowed_team_ids=[]),
    )

    _refused(
        lambda: skills_ep.update_skill(
            "team-skill", SkillUpdateRequest(instructions="Bob's edit."), org.as_(BOB), _principal(BOB),
            _viewer(BOB, teams=(TEAM_ENG,)),
        ),
        403,
        skills_ep.SKILL_EDIT_REFUSAL,
    )
    assert SkillService(org.as_(ALICE).scope).get_skill("team-skill").instructions == "Body of team-skill."


# ── the credential and the auth service being unreachable ────────────────────


def test_a_credential_from_another_org_grants_nothing(org, monkeypatch):
    """A Member of this org presents, in X-MindsHub-Authorization, a
    credential from another org of theirs whose entitlements list
    skills.share_org. The viewer reads teams and permissions with the
    Authorization the gateway validated, so making a skill org-wide is still
    a 403, and every read to the auth service carried that bearer."""
    hub_teams.reset_caches_for_tests()
    asked: list[tuple[str, str]] = []

    async def get_auth_json(path, bearer):
        asked.append((path, bearer))
        if bearer == "other-org-credential":
            return {"permissions": ["skills.share_org", "skills.write_org"], "results": []}
        if path == hub_teams.ENTITLEMENTS_PATH:
            return {"permissions": []}
        return {"results": [{"id": TEAM_ENG}]}

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    _create(org, ALICE, "shared")
    scope = org.as_(ALICE).scope
    viewer = asyncio.run(skills_ep.get_skill_viewer(member_request(), scope, _principal(ALICE)))

    assert (viewer.can_share_org, viewer.can_write_org, viewer.permissions_known) == (False, False, True)
    _refused(lambda: _share(org, ALICE, "shared", viewer, org_allowed=True), 403, skills_ep.SHARING_ORG_REFUSAL)
    assert _row(org, "shared").org_allowed is False
    assert {bearer for _path, bearer in asked} == {MEMBER_BEARER}
    hub_teams.reset_caches_for_tests()


def test_the_turn_reads_the_callers_teams_with_the_gateway_credential(monkeypatch):
    from cowork.api.v1.endpoints import responses as responses_ep

    built: list[str] = []

    class _Handler:
        def __init__(self, *, principal, hub_bearer):
            built.append(hub_bearer)
            raise RuntimeError("stop here")

    monkeypatch.setattr(responses_ep, "ResponsesHandler", _Handler)
    monkeypatch.setattr(responses_ep, "_refuse_while_answering", lambda **kwargs: None)
    with pytest.raises(RuntimeError):
        asyncio.run(responses_ep.responses(SimpleNamespace(conversation=None), member_request(), None))
    assert built == [MEMBER_BEARER]


def test_an_unreachable_permissions_read_is_a_503_not_a_403(org):
    """While the auth service can't list the caller's permissions, a change
    that rests on them answers 503 hub_unavailable, never the 403 that says
    they lack a permission they may hold."""
    _create(org, ALICE, "shared")
    blind = _viewer(ALICE, teams=(TEAM_ENG,), perms_known=False)

    for change in ({"org_allowed": True}, {"allowed_team_ids": [TEAM_ALL]}):
        with pytest.raises(HTTPException) as caught:
            _share(org, ALICE, "shared", replace(blind, team_ids=blind.team_ids | {TEAM_ALL}, default_team_id=TEAM_ALL),
                   **change)
        assert (caught.value.status_code, caught.value.detail["code"]) == (503, HUB_UNAVAILABLE)
    assert _row(org, "shared").org_allowed is False
    # A change that needs no permission still goes through.
    assert _share(org, ALICE, "shared", blind, allowed_user_ids=[BOB])["access"]["allowedUserIds"] == [BOB]

    _share(org, ADMIN, "shared", _viewer(ADMIN, admin=True), admin=True, org_allowed=True)
    with pytest.raises(HTTPException) as caught:
        _share(org, BOB, "shared", _viewer(BOB, perms_known=False), always_on=True)
    assert (caught.value.status_code, caught.value.detail["code"]) == (503, HUB_UNAVAILABLE)
    with pytest.raises(HTTPException) as caught:
        skills_ep.update_skill(
            "shared", SkillUpdateRequest(instructions="Bob's edit."), org.as_(BOB), _principal(BOB),
            _viewer(BOB, perms_known=False),
        )
    assert (caught.value.status_code, caught.value.detail["code"]) == (503, HUB_UNAVAILABLE)


def test_an_unreachable_team_listing_is_a_503_for_a_team_change(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG])
    _share(org, ALICE, "team-skill", allowed_user_ids=[CAROL])

    with pytest.raises(HTTPException) as caught:
        _share(org, ALICE, "team-skill", _viewer(ALICE, known=False), allowed_team_ids=[TEAM_ENG, TEAM_OPS])
    assert (caught.value.status_code, caught.value.detail["code"]) == (503, HUB_UNAVAILABLE)
    # Carol, named on it, would be a 403 if her teams were known; unknown, a 503.
    with pytest.raises(HTTPException) as caught:
        _share(org, CAROL, "team-skill", _viewer(CAROL, known=False), always_on=True)
    assert (caught.value.status_code, caught.value.detail["code"]) == (503, HUB_UNAVAILABLE)
    _refused(lambda: _share(org, CAROL, "team-skill", _viewer(CAROL), always_on=True), 403)


def test_the_viewer_knows_when_permissions_could_not_be_read(monkeypatch):
    hub_teams.reset_caches_for_tests()

    async def get_auth_json(path, bearer):
        return {"results": [{"id": TEAM_ENG}]} if path == hub_teams.TEAMS_PATH else None

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=BOB)
    viewer = asyncio.run(resolve_skill_viewer(scope=scope, principal=_principal(BOB), bearer_token="t"))
    assert (viewer.teams_known, viewer.permissions_known, viewer.can_share_org) == (True, False, False)
    admin = asyncio.run(resolve_skill_viewer(
        scope=TenantScope(org_mode=True, org_id=ORG_ID, user_id=ADMIN),
        principal=_principal(ADMIN, admin=True), bearer_token="t",
    ))
    assert (admin.permissions_known, admin.can_share_org) == (True, True)
    hub_teams.reset_caches_for_tests()


def test_adding_a_team_while_the_teams_are_unreadable_is_a_503_even_for_an_admin(org):
    """The org's default team is never stored: named among the teams it is
    org-wide. While the caller's teams can't be read the default team can't
    be told apart, so a change that adds a team is a 503, an admin's
    included, and nothing is written. Keeping the teams the skill has still
    saves."""
    _create(org, ALICE, "shared")
    _share(org, ALICE, "shared", allowed_team_ids=[TEAM_ENG])
    admin = _viewer(ADMIN, admin=True, known=False)
    with pytest.raises(HTTPException) as caught:
        _share(org, ADMIN, "shared", admin, admin=True, allowed_team_ids=[TEAM_ENG, TEAM_ALL])
    assert (caught.value.status_code, caught.value.detail["code"]) == (503, HUB_UNAVAILABLE)
    row = _row(org, "shared")
    assert (row.org_allowed, row.allowed_team_ids) == (False, [TEAM_ENG])
    out = _share(org, ADMIN, "shared", admin, admin=True, allowed_team_ids=[TEAM_ENG], allowed_user_ids=[BOB])
    assert (out["access"]["allowedTeamIds"], out["access"]["allowedUserIds"]) == ([TEAM_ENG], [BOB])


def test_a_kept_team_the_org_deleted_stays_and_grants_nothing(org):
    """A team deleted after the skill was shared with it is kept on the
    owner's next save, which asks the auth service for no list of the org's
    teams: its id matches no one's teams, so it grants nothing."""
    _create(org, ALICE, "shared")
    _store_rule(org, "shared", SkillAccessRule(owner_user_id=ALICE, allowed_team_ids=(TEAM_ENG, "team-gone")))
    out = _share(org, ALICE, "shared", _viewer(ALICE, teams=(TEAM_ENG,)), allowed_user_ids=[BOB])
    assert out["access"]["allowedTeamIds"] == [TEAM_ENG, "team-gone"]
    assert "shared" not in _listed(org, CAROL, _viewer(CAROL, teams=("team-other",)))


def test_a_skill_is_shared_with_at_most_the_access_services_number_of_people(org):
    _create(org, ALICE, "shared")
    people = [str(uuid4()) for _ in range(MAX_SHARED_PEOPLE + 1)]
    _refused(
        lambda: _share(org, ALICE, "shared", allowed_user_ids=people),
        400,
        f"A skill can be shared with at most {MAX_SHARED_PEOPLE} people.",
    )
    assert MAX_SHARED_PEOPLE == 100


# ── always on ────────────────────────────────────────────────────────────────


ALWAYS_ON_UNSHARED = "Only skills shared with the organization or a team can be always on."
ALWAYS_ON_ORG = skills_ep.ALWAYS_ON_ORG_REFUSAL


def _stored_always_on(org, slug: str) -> bool:
    return _row(org, slug).always_on


def test_org_and_team_skills_are_on_demand_by_default(org):
    _legacy_skill(org.as_(ALICE).scope, "house-style")
    _create(org, ALICE, "team-skill")
    out = _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG])

    assert out["access"]["alwaysOn"] is False
    listed = _listed(org, BOB, _viewer(BOB, teams=(TEAM_ENG,)))
    assert listed["house-style"]["access"]["alwaysOn"] is False
    assert listed["team-skill"]["access"]["alwaysOn"] is False


def test_always_on_needs_an_org_or_team_share(org):
    _create(org, ALICE, "private")
    _refused(lambda: _share(org, ALICE, "private", always_on=True), 400, ALWAYS_ON_UNSHARED)
    _refused(lambda: _share(org, ALICE, "private", allowed_user_ids=[BOB], always_on=True), 400, ALWAYS_ON_UNSHARED)
    # Shared with a team in the same request is enough.
    out = _share(org, ALICE, "private", allowed_team_ids=[TEAM_ENG], always_on=True)
    assert (out["access"]["alwaysOn"], _stored_always_on(org, "private")) == (True, True)


def test_an_org_wide_skill_takes_share_org_to_be_always_on(org):
    _legacy_skill(org.as_(ALICE).scope, "house-style")
    _create(org, ALICE, "rules")
    _share(org, ALICE, "rules", _viewer(ALICE, share_org=True), org_allowed=True)

    _refused(
        lambda: _share(org, ALICE, "rules", _viewer(ALICE, write_org=True), always_on=True), 403, ALWAYS_ON_ORG,
    )
    out = _share(org, BOB, "rules", _viewer(BOB, share_org=True), always_on=True)
    assert out["access"]["alwaysOn"] is True
    assert _listed(org, CAROL, _viewer(CAROL))["rules"]["access"]["alwaysOn"] is True
    _refused(lambda: _share(org, ALICE, "rules", always_on=False), 403, ALWAYS_ON_ORG)
    out = _share(org, ADMIN, "rules", _viewer(ADMIN, admin=True), admin=True, always_on=False)
    assert out["access"]["alwaysOn"] is False
    assert _stored_always_on(org, "rules") is False
    # A skill from before sharing is org-wide, so it can be made always on too.
    out = _share(org, BOB, "house-style", _viewer(BOB, share_org=True), always_on=True)
    assert (out["access"]["orgAllowed"], out["access"]["alwaysOn"]) == (True, True)
    assert (_row(org, "house-style").org_allowed, _row(org, "house-style").always_on) == (True, True)


def test_a_team_skill_takes_edit_rights_to_be_always_on(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], allowed_user_ids=[CAROL])

    # Carol only reads it (shared with her by id).
    _refused(lambda: _share(org, CAROL, "team-skill", _viewer(CAROL), always_on=True), 403)
    out = _share(org, BOB, "team-skill", _viewer(BOB, teams=(TEAM_ENG,)), always_on=True)
    assert out["access"]["alwaysOn"] is True
    _refused(lambda: _share(org, CAROL, "team-skill", _viewer(CAROL), always_on=False), 403)
    out = _share(org, ADMIN, "team-skill", _viewer(ADMIN, admin=True), admin=True, always_on=False)
    assert out["access"]["alwaysOn"] is False
    out = _share(org, ALICE, "team-skill", _viewer(ALICE), always_on=True)
    assert out["access"]["alwaysOn"] is True


def test_unsharing_from_the_org_and_every_team_clears_always_on(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], always_on=True)

    out = _share(org, ALICE, "team-skill", allowed_team_ids=[], allowed_user_ids=[BOB])
    assert out["access"]["alwaysOn"] is False
    assert _stored_always_on(org, "team-skill") is False
    out = _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG])
    assert out["access"]["alwaysOn"] is False


def test_the_sharing_body_accepts_the_camel_case_spelling(org):
    _create(org, ALICE, "team-skill")
    body = SkillSharingRequest.model_validate({"allowedTeamIds": [TEAM_ENG], "alwaysOn": True, "orgAllowed": None})
    assert (body.allowed_team_ids, body.always_on) == ([TEAM_ENG], True)
    out = skills_ep.set_skill_sharing(
        "team-skill", body, org.as_(ALICE), _principal(ALICE), _viewer(ALICE, teams=(TEAM_ENG,)),
    )
    assert out["access"]["alwaysOn"] is True


def test_always_on_survives_edits_renames_and_toggles(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], always_on=True)
    alice = _viewer(ALICE, teams=(TEAM_ENG,))

    skills_ep.update_skill(
        "team-skill", SkillUpdateRequest(label="renamed", instructions="New body."),
        org.as_(ALICE), _principal(ALICE), alice,
    )
    skills_ep.update_skill("renamed", SkillUpdateRequest(enabled=False), org.as_(ALICE), _principal(ALICE), alice)
    out = skills_ep.update_skill(
        "renamed", SkillUpdateRequest(enabled=True), org.as_(ALICE), _principal(ALICE), alice,
    )
    assert out["access"]["alwaysOn"] is True
    assert _stored_always_on(org, "renamed") is True
    out = _share(org, ALICE, "renamed", allowed_user_ids=[CAROL])
    assert out["access"]["alwaysOn"] is True


# ── turns ────────────────────────────────────────────────────────────────────


def _turn_fixture(org):
    """Skills of every kind, as Bob (team eng) meets them."""
    alice = org.as_(ALICE).scope
    _legacy_skill(alice, "zeta-legacy", "Zeta body.")
    # An always_on key an agent wrote into SKILL.md means nothing.
    _legacy_skill(alice, "omega-legacy-claims", "Omega body.", always_on="true")
    _create(org, ALICE, "alpha-org")
    _share(org, ALICE, "alpha-org", _viewer(ALICE, share_org=True), org_allowed=True, always_on=True)
    _create(org, ALICE, "beta-team")
    _share(org, ALICE, "beta-team", allowed_team_ids=[TEAM_ENG], always_on=True)
    _create(org, ALICE, "delta-org")
    _share(org, ALICE, "delta-org", _viewer(ALICE, share_org=True), org_allowed=True)
    _create(org, ALICE, "eta-team")
    _share(org, ALICE, "eta-team", allowed_team_ids=[TEAM_ENG])
    _create(org, ALICE, "gamma-person")
    _share(org, ALICE, "gamma-person", allowed_user_ids=[BOB])
    _create(org, BOB, "bob-own")
    _create(org, DAVE, "dave-private")
    _create(org, ALICE, "off-org")
    _share(org, ALICE, "off-org", _viewer(ALICE, share_org=True), org_allowed=True)
    skills_ep.update_skill(
        "off-org", SkillUpdateRequest(enabled=False), org.as_(ALICE), _principal(ALICE),
        _viewer(ALICE, write_org=True),
    )
    _create(org, ALICE, "elsewhere-team")
    _share(org, ALICE, "elsewhere-team", allowed_team_ids=[TEAM_ENG])
    SkillService(alice).update_skill("elsewhere-team", projects=["other"])


def test_the_pod_gets_only_visible_skills_with_always_on_first(org):
    _turn_fixture(org)

    block = build_turn_skills(
        org.as_(BOB).scope, project_name="general", viewer=_viewer(BOB, teams=(TEAM_ENG,)), session=org.as_(BOB),
    )

    always_on = [slug for slug, entry in block.items() if entry.get("always_on")]
    assert always_on == ["alpha-org", "beta-team"]
    assert list(block)[:2] == always_on
    assert {"gamma-person", "bob-own", "zeta-legacy", "omega-legacy-claims", "delta-org", "eta-team"} <= set(block)
    assert not any(block[slug].get("always_on") for slug in ("zeta-legacy", "omega-legacy-claims", "delta-org"))
    assert not {"dave-private", "off-org", "elsewhere-team"} & set(block)
    assert not any(block[slug].get("always_on") for slug in BUILTIN_SKILL_SLUGS if slug in block)
    assert block["alpha-org"]["files"]["SKILL.md"].startswith("---\n")
    # Without the rows (no session) every skill is nobody's but a built-in.
    blind = build_turn_skills(org.as_(BOB).scope, project_name="general", viewer=_viewer(BOB, teams=(TEAM_ENG,)))
    assert set(blind) <= set(BUILTIN_SKILL_SLUGS)


def test_a_turn_without_a_viewer_gets_only_org_wide_skills_always_on(org):
    """No caller credential (a scheduled or channel turn): the fail-closed
    viewer's set, so only org-wide skills apply always on, and no team-only
    skill is there to recall."""
    _turn_fixture(org)

    block = build_turn_skills(org.as_(BOB).scope, project_name="general", viewer=None, session=org.as_(BOB))
    assert [slug for slug, entry in block.items() if entry.get("always_on")] == ["alpha-org"]
    assert not {"beta-team", "eta-team", "dave-private"} & set(block)
    assert {"alpha-org", "delta-org", "gamma-person", "bob-own"} <= set(block)


def test_the_request_sheds_on_demand_skills_before_always_on_ones(monkeypatch):
    from cowork.turnqueue import producer

    monkeypatch.setattr(producer, "_MAX_REQUEST_BYTES", 2000)
    monkeypatch.setattr(producer, "_REQUEST_BYTES_MARGIN", 0)
    params = {
        "input": "hi",
        "skills": {
            "rule": {"files": {"SKILL.md": "---\nname: rule\n---\nshort"}, "always_on": True},
            "big": {"files": {"SKILL.md": "b" * 3000}},
        },
    }
    fitted = _fit_request(params, "conv")
    assert list(fitted["skills"]) == ["rule"]

    # Not even the always-on ones fit: the block stays, empty, so the pod
    # recalls nothing rather than its whole skills volume.
    params = {"input": "hi", "skills": {"rule": {"files": {"SKILL.md": "r" * 3000}, "always_on": True}}}
    assert _fit_request(params, "conv")["skills"] == {}


def test_the_request_sheds_on_demand_skills_one_at_a_time_before_project_memory(monkeypatch):
    """The lowest ranked on-demand skill (the block's last) goes first, then
    the next, and only once none is left does project memory go; the
    always-on skills go last. The caller's block is left as it was."""
    from cowork.turnqueue import producer
    from cowork.turnqueue.producer import _request_wire_size

    monkeypatch.setattr(producer, "_REQUEST_BYTES_MARGIN", 0)
    memory = {"project": {"rules": "m" * 200}}
    skills = {
        "rule": {"files": {"SKILL.md": "r" * 100}, "always_on": True},
        "alpha": {"files": {"SKILL.md": "a" * 300}},
        "beta": {"files": {"SKILL.md": "b" * 300}},
        "gamma": {"files": {"SKILL.md": "c" * 300}},
    }

    def fit(budget: int) -> dict:
        monkeypatch.setattr(producer, "_MAX_REQUEST_BYTES", budget)
        return _fit_request({"input": "hi", "memory": memory, "skills": skills}, "conv")

    def size(*kept: str, with_memory: bool = True) -> int:
        params = {"input": "hi", "skills": {slug: skills[slug] for slug in kept}}
        return _request_wire_size({**params, "memory": memory} if with_memory else params)

    # Exactly room for all but gamma: gamma alone goes, and the memory stays.
    fitted = fit(size("rule", "alpha", "beta"))
    assert (list(fitted["skills"]), fitted["memory"]) == (["rule", "alpha", "beta"], memory)
    assert list(skills) == ["rule", "alpha", "beta", "gamma"]
    fitted = fit(size("rule", "alpha"))
    assert (list(fitted["skills"]), fitted["memory"]) == (["rule", "alpha"], memory)
    # Room for no on-demand skill: they all go, and the memory still stays.
    fitted = fit(size("rule"))
    assert (list(fitted["skills"]), fitted["memory"]) == (["rule"], memory)
    # One byte less: the memory goes too, and the always-on skill stays.
    fitted = fit(size("rule") - 1)
    assert ("memory" in fitted, list(fitted["skills"])) == (False, ["rule"])
    assert _request_wire_size(fitted) == size("rule", with_memory=False)


def test_the_producer_sends_the_skills_block(monkeypatch):
    """stream_remote_replies always puts ``skills`` on the job, ``{}`` when
    the caller sees nothing or no block was built."""
    from cowork.turnqueue import producer

    captured: list[dict] = []

    def capture(params, conversation_id):
        captured.append(params)
        raise RuntimeError("stop here")

    monkeypatch.setattr(producer, "_fit_request", capture)

    async def run(skills):
        async def noop(*a, **k):
            return None

        monkeypatch.setattr(producer, "require_product_permission", noop)

        async def has(*a, **k):
            return True

        monkeypatch.setattr(producer.product_permissions, "has_product_permission", has)

        class _Redis:
            async def delete(self, *a):
                return 0

        monkeypatch.setattr(producer, "get_redis", lambda: _Redis())

        async def oauth(**kwargs):
            return None

        async def datasource(**kwargs):
            return None

        monkeypatch.setattr(producer, "_mint_oauth_block", oauth)
        monkeypatch.setattr(producer, "_mint_datasource_block", datasource)
        gen = producer.stream_remote_replies(
            conversation_id="c", org_id=ORG_ID, user_id=BOB, input_text="hi", model="m",
            llm={"api_key": "k"}, skills=skills,
        )
        with pytest.raises(RuntimeError):
            await gen.__anext__()

    block = {"rule": {"files": {"SKILL.md": "---\nname: rule\n---\nx"}, "always_on": True}}
    asyncio.run(run(block))
    asyncio.run(run({}))
    asyncio.run(run(None))
    assert captured[0]["skills"] == block
    assert captured[1]["skills"] == {}
    assert captured[2]["skills"] == {}


# ── desktop and local deployments ────────────────────────────────────────────


def test_local_deployments_are_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    try:
        service = SkillService(LOCAL_SCOPE)
        service.create_skill("desk", "Desk body.")
        assert SkillService(LOCAL_SCOPE).get_skill("desk").created_by is None
        assert build_turn_skills(LOCAL_SCOPE, project_name="general") == {
            "desk": {"files": {"SKILL.md": (service.root / "desk" / "SKILL.md").read_text()}}
        }
        assert skills_ep.skill_sharing_capabilities(None) == {
            "enabled": False, "canShareOrg": False, "canWriteOrg": False, "orgName": None,
        }
    finally:
        get_app_settings.cache_clear()


def test_sharing_capabilities_report_the_viewer():
    assert skills_ep.skill_sharing_capabilities(_viewer(BOB, share_org=True)) == {
        "enabled": True, "canShareOrg": True, "canWriteOrg": False, "orgName": None,
    }
    assert skills_ep.skill_sharing_capabilities(_viewer(ADMIN, admin=True))["canWriteOrg"] is True


# ── people must be in the org ────────────────────────────────────────────────


def test_sharing_with_people_all_in_the_org_asks_the_access_service_once(org):
    _create(org, ALICE, "shared")
    calls: list = []
    out = _share(org, ALICE, "shared", members=_members(calls=calls), allowed_user_ids=[BOB, CAROL.upper()])
    assert out["access"]["allowedUserIds"] == [BOB, CAROL]
    assert calls == [[BOB, CAROL]]


def test_sharing_with_someone_outside_the_org_is_refused_naming_them(org):
    _create(org, ALICE, "shared")
    _refused(
        lambda: _share(org, ALICE, "shared", allowed_user_ids=[BOB, EVE, MAL]),
        400,
        f"Skills can only be shared with people in your organization: {EVE}, {MAL}",
    )
    assert _row(org, "shared").allowed_user_ids == []


def test_sharing_with_people_while_the_access_service_is_unreachable_is_a_503(org):
    _create(org, ALICE, "shared")

    async def unreachable(asked):
        return None

    _refused(lambda: _share(org, ALICE, "shared", members=unreachable, allowed_user_ids=[BOB]), 503)


def test_people_already_shared_with_are_not_asked_about_again(org):
    _create(org, ALICE, "shared")
    _share(org, ALICE, "shared", allowed_user_ids=[BOB])

    calls: list = []

    async def unreachable(asked):
        calls.append(list(asked))
        return None

    # Keeping bob, adding a team, and the sharer's own id: nothing to ask.
    out = _share(
        org, ALICE, "shared", members=unreachable, allowed_team_ids=[TEAM_ENG], allowed_user_ids=[BOB, ALICE],
    )
    assert out["access"]["allowedTeamIds"] == [TEAM_ENG]
    assert calls == []
    asked: list = []
    _share(org, ALICE, "shared", members=_members(calls=asked), allowed_user_ids=[BOB, CAROL])
    assert asked == [[CAROL]]


def test_an_id_added_without_the_check_is_a_conflict(org):
    """The route re-checks under the skill's lock: an id the dependency didn't
    confirm (the skill changed in between) isn't written."""
    _create(org, ALICE, "shared")
    _refused(
        lambda: skills_ep.set_skill_sharing(
            "shared", SkillSharingRequest(allowed_user_ids=[BOB]), org.as_(ALICE), _principal(ALICE),
            _viewer(ALICE), frozenset(),
        ),
        409,
    )


def test_the_route_lookup_asks_the_access_service_by_id(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def get_auth_json(path, bearer):
        calls.append((path, bearer))
        return {"results": [{"id": BOB.upper(), "name": "Bob", "email": "bob@x.com"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    lookup = get_org_member_lookup(member_request())
    assert asyncio.run(lookup([BOB, EVE])) == frozenset({BOB})
    # With the gateway's credential, never the client-set one.
    assert calls == [(f"/me/org-members/?ids={BOB}%2C{EVE}", MEMBER_BEARER)]


def test_the_org_membership_lookup_batches_and_fails_closed(monkeypatch):
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        calls.append(path)
        return {"results": [{"id": "u1", "name": "Bob", "email": "bob@x.com"}, {"id": "stranger"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    found = asyncio.run(hub_people.org_members_by_ids(bearer_token="t", ids=["U1", "u2"]))
    # Only what was asked; an id the auth service didn't return is not a member.
    assert [(m.id, m.name) for m in found] == [("u1", "Bob")]
    assert calls == ["/me/org-members/?ids=u1%2Cu2"]

    calls.clear()
    asyncio.run(hub_people.org_members_by_ids(bearer_token="t", ids=[f"p{i}" for i in range(60)]))
    assert len(calls) == 2

    async def down(path, bearer):
        return None

    monkeypatch.setattr(hub_people, "get_auth_json", down)
    assert asyncio.run(hub_people.org_members_by_ids(bearer_token="t", ids=["u1"])) is None
    assert asyncio.run(hub_people.org_members_by_ids(bearer_token="", ids=["u1"])) is None
    assert asyncio.run(hub_people.org_members_by_ids(bearer_token="t", ids=[])) == []


# ── hub people ───────────────────────────────────────────────────────────────


def test_hub_people_names_stored_ids(monkeypatch):
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        assert bearer == MEMBER_BEARER
        calls.append(path)
        return {"results": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), scope, ids=f"{BOB}, {EVE},"))
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "people": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}], "truncated": False,
    }
    # Uncached.
    asyncio.run(hub_people_ep.get_hub_people(member_request(), scope, ids=BOB))
    assert calls == [f"/me/org-members/?ids={BOB}%2C{EVE}", f"/me/org-members/?ids={BOB}"]

    async def down(path, bearer):
        return None

    monkeypatch.setattr(hub_people, "get_auth_json", down)
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), scope, ids=BOB))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": [], "truncated": False}
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), LOCAL_SCOPE, ids=BOB))
    assert view.model_dump(by_alias=True) == {"reachable": True, "people": [], "truncated": False}


def test_hub_people_asks_the_access_service_for_the_directory_without_params(monkeypatch):
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        calls.append(path)
        return {"results": [], "truncated": False}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    asyncio.run(hub_people_ep.get_hub_people(member_request(), scope))
    assert calls == ["/me/org-members/"]


def test_hub_people_reads_the_org_directory_fresh_every_time(monkeypatch):
    """The app is the directory's only reader and keeps its own copy, so this
    server keeps none: every read asks the auth service, so a member who
    joined a minute ago is listed."""
    calls: list[tuple[str, str]] = []

    async def get_auth_json(path, bearer):
        calls.append((path, bearer))
        return {
            "results": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}, {"name": "no id"}],
            "truncated": True,
        }

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), scope))
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "people": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}], "truncated": True,
    }
    assert calls == [("/me/org-members/", MEMBER_BEARER)]
    asyncio.run(hub_people_ep.get_hub_people(member_request(), scope))
    assert calls == [("/me/org-members/", MEMBER_BEARER)] * 2
    asyncio.run(hub_people_ep.get_hub_people(member_request(), scope, ids=BOB))
    assert calls[-1][0] == f"/me/org-members/?ids={BOB}"
    # Another caller asks with their own credential.
    other = TenantScope(org_mode=True, org_id=ORG_ID, user_id=BOB)
    asyncio.run(hub_people_ep.get_hub_people(member_request(bearer="other"), other))
    assert calls[-1] == ("/me/org-members/", "other")


def test_hub_people_failure_is_unreachable_and_the_next_read_asks_again(monkeypatch):
    calls: list[str] = []
    answers = [None, {"results": [{"id": BOB}]}]

    async def get_auth_json(path, bearer):
        calls.append(path)
        return answers[min(len(calls), len(answers)) - 1]

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), scope))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": [], "truncated": False}
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), scope))
    assert len(calls) == 2
    # A plain list or no flag from the auth service: not truncated.
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "people": [{"id": BOB, "name": "", "email": ""}], "truncated": False,
    }


def test_hub_people_the_access_service_refusing_the_caller_is_unreachable(monkeypatch):
    """The auth service answers 403 to someone who is not a current member of
    the active org: no directory and no lookup, reported rather than raised."""
    from cowork.services import hub_workspaces

    seen: list[str] = []
    real_client = httpx.AsyncClient

    def client(**kwargs):
        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(403, json={"detail": "Not a member of this organization."})

        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(hub_workspaces.httpx, "AsyncClient", client)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), scope))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": [], "truncated": False}
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), scope, ids=BOB))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": [], "truncated": False}
    assert [path.endswith("/me/org-members/") for path in seen] == [True, True]


def test_hub_people_outside_org_mode_is_empty_and_reachable(monkeypatch):
    async def get_auth_json(path, bearer):
        raise AssertionError("the auth service was asked outside org mode")

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), LOCAL_SCOPE))
    assert view.model_dump(by_alias=True) == {"reachable": True, "people": [], "truncated": False}
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), LOCAL_SCOPE, emails="bob@x.com"))
    assert view.model_dump(by_alias=True) == {"reachable": True, "people": [], "truncated": False}


def test_hub_people_finds_a_member_past_the_directory_cap_by_address(monkeypatch):
    """The directory the app lists is capped, so a member past the cap isn't
    on it. The address someone types still names them, asked of the access
    service, so the app shares with that person by id rather than inviting
    the address as someone outside the org."""
    calls: list[tuple[str, str]] = []

    async def get_auth_json(path, bearer):
        calls.append((path, bearer))
        if path == "/me/org-members/":
            return {"results": [{"id": EVE, "name": "Eve", "email": "eve@x.com"}], "truncated": True}
        return {"results": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    listed = asyncio.run(hub_people_ep.get_hub_people(member_request(), scope))
    assert listed.truncated and "bob@x.com" not in {p.email for p in listed.people}

    view = asyncio.run(
        hub_people_ep.get_hub_people(member_request(), scope, emails=" Bob@X.com, nobody@y.com,")
    )
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "people": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}], "truncated": False,
    }
    # With the gateway's credential, lowercased, and uncached.
    asyncio.run(hub_people_ep.get_hub_people(member_request(), scope, emails="bob@x.com"))
    assert calls[1:] == [
        ("/me/org-members/?emails=bob%40x.com%2Cnobody%40y.com", MEMBER_BEARER),
        ("/me/org-members/?emails=bob%40x.com", MEMBER_BEARER),
    ]

    async def down(path, bearer):
        return None

    monkeypatch.setattr(hub_people, "get_auth_json", down)
    view = asyncio.run(hub_people_ep.get_hub_people(member_request(), scope, emails="bob@x.com"))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": [], "truncated": False}


def test_hub_people_refuses_ids_and_emails_together(monkeypatch):
    async def get_auth_json(path, bearer):
        raise AssertionError("the access service was asked")

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(hub_people_ep.get_hub_people(member_request(), scope, ids=BOB, emails="bob@x.com"))
    assert refused.value.status_code == 400


def test_the_org_member_lookup_by_address_batches_and_fails_closed(monkeypatch):
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        calls.append(path)
        return {"results": [{"id": BOB, "name": "Bob", "email": "Bob@x.com"}, {"id": EVE, "email": "eve@x.com"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    found = asyncio.run(hub_people.org_members_by_emails(bearer_token="t", emails=["nobody@x.com", "BOB@x.com"]))
    # Only what was asked; an address the access service didn't return is no member's.
    assert [(m.id, m.name) for m in found] == [(BOB, "Bob")]
    assert calls == ["/me/org-members/?emails=nobody%40x.com%2Cbob%40x.com"]

    calls.clear()
    asyncio.run(hub_people.org_members_by_emails(bearer_token="t", emails=[f"p{i}@x.com" for i in range(60)]))
    assert len(calls) == 2

    async def down(path, bearer):
        return None

    monkeypatch.setattr(hub_people, "get_auth_json", down)
    assert asyncio.run(hub_people.org_members_by_emails(bearer_token="t", emails=["bob@x.com"])) is None
    assert asyncio.run(hub_people.org_members_by_emails(bearer_token="", emails=["bob@x.com"])) is None
    assert asyncio.run(hub_people.org_members_by_emails(bearer_token="t", emails=[])) == []


# ── hub teams ────────────────────────────────────────────────────────────────


def test_hub_teams_lists_the_callers_teams_and_caches_them(monkeypatch):
    hub_teams.reset_caches_for_tests()
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        # With the gateway's credential, never the client-set one.
        assert bearer == MEMBER_BEARER
        calls.append(path)
        return {"results": [{"id": TEAM_ENG, "display_name": "Eng", "is_default": True}, {"display_name": "no id"}]}

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=BOB)
    view = asyncio.run(hub_teams_ep.get_hub_teams(member_request(), scope))
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "teams": [{"id": TEAM_ENG, "displayName": "Eng", "isDefault": True}],
    }
    asyncio.run(hub_teams_ep.get_hub_teams(member_request(), scope))
    assert calls == [hub_teams.TEAMS_PATH]
    hub_teams.reset_caches_for_tests()


def test_hub_teams_outside_org_mode_is_empty_and_reachable(monkeypatch):
    """No organization, so no teams to share with, and no credential sent:
    on the desktop the Authorization is this server's own."""

    async def get_auth_json(path, bearer):
        raise AssertionError("the auth service was asked outside org mode")

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    view = asyncio.run(hub_teams_ep.get_hub_teams(member_request(), LOCAL_SCOPE))
    assert view.model_dump(by_alias=True) == {"reachable": True, "teams": []}


def test_hub_teams_unreachable_lists_none(monkeypatch):
    hub_teams.reset_caches_for_tests()

    async def get_auth_json(path, bearer):
        return None

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=BOB)
    view = asyncio.run(hub_teams_ep.get_hub_teams(member_request(), scope))
    assert view.model_dump(by_alias=True) == {"reachable": False, "teams": []}
    hub_teams.reset_caches_for_tests()


# ── artifacts ────────────────────────────────────────────────────────────────


# What an org-mode entry of .published.json keeps for a restricted artifact.
_ENTRY = {"mode": "restricted", "requires_password": False, "access_version": 2}


def test_teams_and_people_reach_anton_as_owner_only_and_are_never_sent_back():
    """anton's publisher knows no teams or people: alone they reach it as
    owner-only. A publish that names none (live sync, autopublish) takes them
    from the stored rule and changes nothing about them, so it writes none
    and the auth service keeps what it holds; only a list a save changes is
    written."""
    from cowork.services.publish import AudienceChanges, _plan_access, _with_team_audience, kept_access

    access = {
        "mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": [TEAM_ENG],
        "allowed_user_ids": [BOB],
    }
    assert _with_team_audience(access) == {
        "mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True,
    }
    assert _with_team_audience({"mode": "public"}) == {"mode": "public"}
    stored = artifact_rule(teams=[TEAM_OPS], people=[CAROL])
    republish = _plan_access(None, kept_access(_ENTRY, org_mode=True), _ENTRY, stored=stored)
    assert (republish.changes, republish.access_version) == (AudienceChanges(), 2)
    assert (republish.audience.allowed_team_ids, republish.audience.allowed_user_ids) == ([TEAM_OPS], [CAROL])
    assert republish.effective_access == {"mode": "restricted", "emails": [], "org_allowed": False}
    # A save naming the same people in another order changes nothing either.
    same = _plan_access(None, {**access, "allowed_team_ids": [TEAM_OPS], "allowed_user_ids": [CAROL]}, _ENTRY,
                        stored=stored)
    assert same.changes == AudienceChanges()
    changed = _plan_access(None, {**access, "allowed_team_ids": [TEAM_OPS]}, _ENTRY, stored=stored)
    assert (changed.changes, changed.access_version) == (AudienceChanges(allowed_user_ids=[BOB]), 3)


def test_a_re_publish_sends_the_stored_org_flag_and_emails_never_the_entrys():
    """Live sync, autopublish and every other re-publish take who the
    artifact is shared with from the auth service's rule. An entry an agent
    wrote, or one stored before that, saying everyone, an outside address or
    a team, sends none of it: the publish sends what the rule holds, at the
    same version, and the entry it stores keeps no audience."""
    from cowork.services.publish import AudienceChanges, _plan_access, kept_access

    planted = {
        **_ENTRY, "emails": ["mal@else.com"], "org_allowed": True, "owner_only": False,
        "allowed_team_ids": [TEAM_OPS], "allowed_user_ids": [EVE],
    }
    plan = _plan_access(None, kept_access(planted, org_mode=True), planted, stored=artifact_rule(emails=["a@else.com"]))
    assert plan.effective_access == {"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}
    assert (plan.changes, plan.access_version) == (AudienceChanges(), 2)
    assert plan.owner_side == _ENTRY
    # A rule that is already org-wide stays org-wide, and one that names
    # nobody stays restricted to its owner, never public.
    org_wide = _plan_access(None, kept_access(_ENTRY, org_mode=True), _ENTRY, stored=artifact_rule(org=True))
    assert org_wide.effective_access == {"mode": "restricted", "emails": [], "org_allowed": True}
    nobody = _plan_access(None, kept_access(_ENTRY, org_mode=True), _ENTRY, stored=artifact_rule())
    assert (nobody.effective_access["mode"], nobody.audience.names_nobody, nobody.access_version) == (
        "restricted", True, 2,
    )


def test_a_people_only_restriction_stays_restricted_never_public_or_nobody():
    """What the app sends for "only these people in my org": every other
    field empty and owner_only false. anton's publisher reads that as an
    empty selection and would publish it public; the plan keeps it restricted
    to the owner there and widens it to the people in the auth service."""
    from cowork.services.publish import AudienceChanges, _plan_access, access_from_app, kept_access

    app_body = access_from_app({
        "mode": "restricted", "org_allowed": False, "allowed_team_ids": [], "allowed_user_ids": [BOB],
        "allowed_emails": [], "owner_only": False,
    })
    plan = _plan_access(None, app_body, None, stored=artifact_rule())
    assert plan.effective_access == {"mode": "restricted", "emails": [], "org_allowed": False}
    assert plan.owner_side == {"mode": "restricted", "requires_password": False, "access_version": 1}
    # Written to the auth service with the people: not public, and not the
    # owner alone.
    assert plan.changes == AudienceChanges(allowed_user_ids=[BOB])
    # Republished once the rule holds Bob: still restricted, nothing written.
    holds_bob = artifact_rule(people=[BOB])
    again = _plan_access(None, kept_access(plan.owner_side, org_mode=True), plan.owner_side, stored=holds_bob)
    assert (again.effective_access["mode"], again.changes) == ("restricted", AudienceChanges())
    # Taking the people away (nothing else picked) is the owner alone, written
    # to the auth service so their access is revoked there too.
    cleared = _plan_access(None, {**app_body, "allowed_user_ids": [], "owner_only": True}, plan.owner_side,
                           stored=holds_bob)
    assert (cleared.owner_side["mode"], cleared.changes) == ("restricted", AudienceChanges(allowed_user_ids=[]))
    assert cleared.access_version > plan.access_version
    # Outside org mode there are no org members, and still no public rule.
    local = _plan_access(None, app_body, None, stored=None)
    assert (local.effective_access["mode"], local.owner_side["owner_only"], local.changes) == (
        "restricted", True, AudienceChanges(),
    )


def test_the_app_sends_the_access_services_four_audience_names_and_nothing_else():
    """org_allowed, allowed_team_ids, allowed_user_ids and allowed_emails;
    allowed_emails is anton's emails. A retired name is refused, naming the
    one to send, rather than dropped."""
    from cowork.services.publish import AudienceChanges, _plan_access, access_from_app

    assert access_from_app({
        "mode": "restricted", "org_allowed": True, "allowed_team_ids": [TEAM_ENG],
        "allowed_user_ids": [BOB], "allowed_emails": ["e@else.com"],
    }) == {
        "mode": "restricted", "org_allowed": True, "allowed_team_ids": [TEAM_ENG],
        "allowed_user_ids": [BOB], "emails": ["e@else.com"],
    }
    for retired, name in (("emails", "allowed_emails"), ("team_ids", "allowed_team_ids"), ("org_wide", "org_allowed")):
        with pytest.raises(ValueError, match=name):
            access_from_app({"mode": "restricted", retired: []})
    plan = _plan_access(
        None, {"mode": "restricted", "allowed_team_ids": [TEAM_ENG], "org_allowed": True}, None, stored=artifact_rule(),
    )
    assert (plan.changes, plan.audience.org_allowed, plan.audience.names_nobody) == (
        AudienceChanges(allowed_team_ids=[TEAM_ENG]), True, False,
    )


def _echo(seen: list):
    """The auth service's upsert, as far as these tests need it: it records
    the body and echoes the lists it was sent."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"url": str(request.url), "body": body})
        return httpx.Response(200, json={
            "ok": True, "allowed_team_ids": body.get("allowed_team_ids", []),
            "allowed_user_ids": body.get("allowed_user_ids", []), "org_allowed": False,
        })

    return handler


_INTERNAL_SETTINGS = SimpleNamespace(auth_internal_base_url="https://auth.test", auth_internal_secret="s")
_ORG_SCOPE = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _write(handler, **kwargs):
    from cowork.services.artifact_access import sync_artifact_team_access

    return sync_artifact_team_access(
        artifact_id=kwargs.pop("artifact_id", "artifact/1"),
        scope=_ORG_SCOPE, owner_user_id=ALICE, workspace_id=None,
        access_version=kwargs.pop("access_version", 3), settings=_INTERNAL_SETTINGS,
        client=_client(handler), **kwargs,
    )


def _read(handler):
    from cowork.services.artifact_access import read_artifact_access

    return read_artifact_access(
        artifact_id="artifact/1", scope=_ORG_SCOPE, owner_user_id=ALICE, settings=_INTERNAL_SETTINGS,
        client=_client(handler),
    )


def test_the_team_and_people_audience_is_written_to_the_access_service():
    """Only the lists a save changed, under the auth service's names; the
    emails and the org flag are the lambda's to write. The answer is what the
    rule holds."""
    seen: list[dict] = []
    held = _write(_echo(seen), allowed_team_ids=[TEAM_ENG], allowed_user_ids=[BOB])
    assert seen == [{
        "url": "https://auth.test/v1/internal/artifact-access/",
        "body": {
            "artifact_id": "artifact/1", "owner_keycloak_id": ALICE, "organization_id": ORG_ID,
            "workspace_id": None, "allowed_team_ids": [TEAM_ENG], "allowed_user_ids": [BOB], "access_version": 3,
        },
    }]
    assert (held.allowed_team_ids, held.allowed_user_ids, held.org_allowed) == ([TEAM_ENG], [BOB], False)
    # A list left out is left out of the write, for the auth service to keep.
    _write(_echo(seen), allowed_user_ids=[CAROL])
    assert "allowed_team_ids" not in seen[-1]["body"]


def test_the_rule_is_read_as_stored_and_no_rule_is_none():
    """The read the stored rule comes from: the delete's identity fields, the
    four sharing fields back. No rule is None, which a publish reads as the
    owner alone and a re-publish that isn't restricted reads as a mode that
    still matches the owner's last save. Anything else that isn't the rule
    is a failure, never a guess."""
    from cowork.services.artifact_access import ArtifactAccessUnavailable

    seen: list[dict] = []
    stored = {
        "artifact_id": "artifact/1", "org_allowed": True, "allowed_team_ids": [TEAM_ENG],
        "allowed_user_ids": [BOB], "allowed_emails": ["e@else.com"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content),
                     "secret": request.headers["X-Internal-Auth"]})
        return httpx.Response(200, json=stored)

    assert _read(handler) == artifact_rule(org=True, teams=[TEAM_ENG], people=[BOB], emails=["e@else.com"])
    assert seen == [{
        "url": "https://auth.test/v1/internal/artifact-access/read/",
        "body": {"artifact_id": "artifact/1", "owner_keycloak_id": ALICE, "organization_id": ORG_ID},
        "secret": "s",
    }]
    assert _read(lambda r: httpx.Response(404, json={"code": "no_rule", "detail": "x"})) is None

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    for handler in (
        lambda r: httpx.Response(404, text="Not Found"),
        lambda r: httpx.Response(503),
        lambda r: httpx.Response(200, json={k: v for k, v in stored.items() if k != "allowed_emails"}),
        unreachable,
    ):
        with pytest.raises(ArtifactAccessUnavailable):
            _read(handler)


def test_the_access_service_refusing_a_non_member_is_a_400_naming_them():
    from cowork.services.artifact_access import ArtifactAudienceRejected

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": "not_a_member", "detail": "x", "allowed_user_ids": [EVE]})

    with pytest.raises(ArtifactAudienceRejected) as caught:
        _write(handler, allowed_user_ids=[EVE], access_version=1)
    # A ValueError, which the access route answers as 400.
    assert isinstance(caught.value, ValueError) and EVE in str(caught.value)


def test_the_access_service_refusing_a_foreign_team_is_a_400_naming_it():
    """The auth service's unknown_team body names the teams in allowed_team_ids."""
    from cowork.services.artifact_access import ArtifactAudienceRejected

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": "unknown_team", "detail": "x", "allowed_team_ids": [TEAM_OPS]})

    with pytest.raises(ArtifactAudienceRejected) as caught:
        _write(handler, allowed_team_ids=[TEAM_OPS], access_version=1)
    assert TEAM_OPS in str(caught.value)


def test_any_400_from_the_access_service_is_a_400_naming_why_never_a_retryable_503():
    """A field the auth service won't take (too many people, a malformed id)
    is a 400 with no code: retrying can't help, so it is never
    ArtifactAccessUnavailable."""
    from cowork.services.artifact_access import ArtifactAccessUnavailable, ArtifactAudienceRejected

    answers = [
        ({"allowed_user_ids": ["Ensure this field has no more than 100 elements."]},
         "allowed_user_ids: Ensure this field has no more than 100 elements."),
        ({"detail": "Unknown owner"}, "Unknown owner"),
        ("not json", "The sharing change was refused."),
    ]
    for body, reason in answers:
        def handler(request: httpx.Request, body=body) -> httpx.Response:
            if isinstance(body, str):
                return httpx.Response(400, text=body)
            return httpx.Response(400, json=body)

        with pytest.raises(ArtifactAudienceRejected) as caught:
            _write(handler, allowed_user_ids=[BOB])
        assert reason in str(caught.value)
        assert not isinstance(caught.value, ArtifactAccessUnavailable)

    def outage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(ArtifactAccessUnavailable):
        _write(outage, allowed_user_ids=[BOB])


def test_a_409_or_403_from_the_access_service_is_a_refusal_never_a_retryable_503():
    """A recorded owner or organization that doesn't match (409), or an
    owner whose role no longer allows artifact changes (403), gets the same
    answer however often it is asked: ArtifactAccessRefused carrying the
    auth service's detail, on the write and on the read alike, saying which
    of the two it refused."""
    from cowork.services.artifact_access import ArtifactAccessRefused, ArtifactAccessUnavailable

    for status, body, detail in (
        (409, {"code": "artifact_ownership_conflict", "detail": "This artifact identity belongs to a different owner."},
         "This artifact identity belongs to a different owner."),
        (403, {"detail": "Your organization role does not allow artifact changes."},
         "Your organization role does not allow artifact changes."),
    ):
        def handler(request: httpx.Request, status=status, body=body) -> httpx.Response:
            return httpx.Response(status, json=body)

        for call, refused in ((lambda h: _write(h, allowed_user_ids=[BOB]), "the change"), (_read, "the read")):
            with pytest.raises(ArtifactAccessRefused) as caught:
                call(handler)
            assert str(caught.value) == f"The sharing service refused {refused}: {detail}"
            assert not isinstance(caught.value, ArtifactAccessUnavailable)


def test_a_200_that_does_not_echo_the_teams_and_people_is_a_failure():
    """An auth service that drops the new fields answers 200 and stores
    nothing: the share must not read as saved."""
    from cowork.services.artifact_access import ArtifactAccessUnavailable

    for body in ({"ok": True}, {"ok": True, "allowed_team_ids": [TEAM_ENG]}, {"allowed_team_ids": None,
                                                                              "allowed_user_ids": []}):
        def handler(request: httpx.Request, body=body) -> httpx.Response:
            return httpx.Response(200, json=body)

        with pytest.raises(ArtifactAccessUnavailable):
            _write(handler, allowed_team_ids=[TEAM_ENG])


def test_the_access_write_carries_notify_only_when_asked():
    seen: list[dict] = []
    _write(_echo(seen), allowed_user_ids=[BOB])
    _write(
        _echo(seen), allowed_user_ids=[BOB], notify_user_ids=[BOB], notify_emails=["e@else.com"],
        artifact_title="Q3", artifact_url="https://view.test/1",
    )
    # Empty lists are nobody to notify.
    _write(_echo(seen), allowed_user_ids=[BOB], notify_user_ids=[], notify_emails=[], artifact_title="Q3")
    notify_keys = {"notify_recipients", "notify_user_ids", "notify_emails", "artifact_title", "artifact_url"}
    assert not notify_keys & set(seen[0]["body"])
    assert {k: seen[1]["body"][k] for k in notify_keys} == {
        "notify_recipients": True, "notify_user_ids": [BOB], "notify_emails": ["e@else.com"],
        "artifact_title": "Q3", "artifact_url": "https://view.test/1",
    }
    assert not notify_keys & set(seen[2]["body"])


def test_the_post_publish_write_sends_the_changes_and_returns_the_access_services_answer(monkeypatch):
    from cowork.services import artifact_access, artifact_autopublish
    from cowork.services.artifact_access import ArtifactAudience
    from cowork.services.publish import AudienceChanges, _sync_team_audience

    sent: list[dict] = []
    answer = ArtifactAudience(org_allowed=False, allowed_team_ids=[TEAM_ENG], allowed_user_ids=[])

    def sync(**kwargs):
        sent.append(kwargs)
        return answer

    monkeypatch.setattr(artifact_access, "sync_artifact_team_access", sync)
    monkeypatch.setattr(artifact_autopublish, "_active_workspace_id", lambda scope: None)
    held = _sync_team_audience(
        artifact_id="a", scope=_ORG_SCOPE, access_version=4, changes=AudienceChanges(allowed_user_ids=[BOB]),
    )
    assert (sent[0]["allowed_team_ids"], sent[0]["allowed_user_ids"], sent[0]["access_version"]) == (None, [BOB], 4)
    # What the publish reports is the auth service's answer, not the request.
    assert held == answer


def test_an_org_entry_says_nothing_of_who_an_artifact_is_shared_with(tmp_path, monkeypatch):
    """In hosted org mode the card and the access route read no audience from
    .published.json, even from an entry stored before it stopped keeping
    one, and a re-publish keeps the mode alone (GET /access reads the rule).
    On the desktop the entry is still the record."""
    from cowork.services import artifacts
    from cowork.services.artifacts import _published_access_for
    from cowork.services.publish import published_artifact_access

    folder = tmp_path / "artifacts" / "rep"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}))
    (folder / ".published.json").write_text(json.dumps({"index.html": {
        "report_id": "r1", "url": "u", "mode": "restricted", "emails": ["e@else.com"], "org_allowed": True,
        "owner_only": False, "allowed_team_ids": [TEAM_ENG], "allowed_user_ids": [BOB], "access_version": 2,
    }}))
    monkeypatch.setattr(artifacts, "_org_mode", lambda: True)
    read = _published_access_for(folder, folder / "index.html")
    assert read["accessMode"] == "restricted"
    assert (read["accessEmails"], read["orgAllowed"], read["ownerOnly"], read["allowedTeamIds"],
            read["allowedUserIds"]) == ([], False, False, [], [])
    assert published_artifact_access(folder, artifacts_base=folder.parent, org_mode=True) == {
        "mode": "restricted", "owner_only": True,
    }

    monkeypatch.setattr(artifacts, "_org_mode", lambda: False)
    read = _published_access_for(folder, folder / "index.html")
    assert (read["accessEmails"], read["orgAllowed"], read["allowedTeamIds"]) == (["e@else.com"], True, [])
    assert published_artifact_access(folder, artifacts_base=folder.parent, org_mode=False) == {
        "mode": "restricted", "emails": ["e@else.com"], "org_allowed": True, "owner_only": False,
    }


def _check(access, *, principal=None, stored=None, scope=None, reads=None):
    """PUT /access's check of ``access``, with ``stored`` as the artifact's
    rule in the auth service (nobody but the owner by default); each read of
    it is counted in ``reads``. The access the check hands the publish."""
    from cowork.api.v1.endpoints import artifact_workspace as aw

    rule = stored or artifact_rule()

    async def stored_rule():
        if reads is not None:
            reads.append(rule)
        return rule

    session = SimpleNamespace(scope=scope or _ORG_SCOPE)
    return asyncio.run(aw._checked_team_audience(
        access, request=member_request(principal or _principal(ALICE)), session=session, stored_rule=stored_rule,
    )).access


_AS_ADMIN = {"principal": _principal(ADMIN, admin=True), "scope": TenantScope(org_mode=True, org_id=ORG_ID, user_id=ADMIN)}


def test_artifact_people_are_normalized_and_checked_only_when_they_change(monkeypatch):
    asked: list = []
    access_service_answers(monkeypatch, asked=asked)
    out = _check({"mode": "restricted", "allowed_user_ids": [f" {BOB.upper()}", BOB], "emails": ["e@else.com"]})
    assert out == {"mode": "restricted", "allowed_user_ids": [BOB], "emails": ["e@else.com"]}
    assert len(asked) == 1
    # The people the rule already holds: nothing is written, so none is asked about.
    reads: list = []
    assert _check({"mode": "restricted", "allowed_user_ids": [BOB]}, stored=artifact_rule(people=[BOB]), reads=reads) == {
        "mode": "restricted", "allowed_user_ids": [BOB],
    }
    assert (len(asked), len(reads)) == (1, 1)
    with pytest.raises(HTTPException) as caught:
        _check({"mode": "restricted", "allowed_user_ids": "nope"})
    assert caught.value.status_code == 400
    # Outside org mode, or for a save that isn't restricted, the access is
    # left as it is: the plan keeps people picked alone private
    # (test_a_people_only_restriction_stays_restricted...).
    access = {"mode": "restricted", "allowed_user_ids": [BOB], "owner_only": False}
    assert _check(access, scope=LOCAL_SCOPE) == access
    assert _check({"mode": "public", "allowed_user_ids": "nope"}) == {"mode": "public", "allowed_user_ids": "nope"}


def test_an_artifact_save_refuses_a_non_member_it_adds_before_anything_is_published(monkeypatch):
    """Every person a save names is asked about in one batched lookup before
    the publish, so the auth service's not_a_member fires only in a race. One
    the save adds who isn't a member is a 400 naming them. One the rule
    holds who has left the org is dropped: the auth service keeps a departed
    person's id on the rule and refuses it on the next write. The caller's
    own id is never asked about. While the lookup can't be asked, a 503."""
    from cowork.services.artifact_access import NOT_A_MEMBER_REFUSAL

    asked: list = []
    access_service_answers(monkeypatch, members=(BOB, DAVE), asked=asked)
    stored = artifact_rule(people=[BOB, CAROL])
    with pytest.raises(HTTPException) as caught:
        _check({"mode": "restricted", "allowed_user_ids": [BOB, CAROL, EVE, DAVE]}, stored=stored)
    assert (caught.value.status_code, caught.value.detail) == (400, f"{NOT_A_MEMBER_REFUSAL}: {EVE}")
    assert len(asked) == 1 and asked[0][1] == MEMBER_BEARER
    assert set(parse_qs(urlsplit(asked[0][0]).query)["ids"][0].split(",")) == {BOB, CAROL, EVE, DAVE}

    out = _check({"mode": "restricted", "allowed_user_ids": [BOB, CAROL, ALICE, DAVE]}, stored=stored)
    assert out["allowed_user_ids"] == [BOB, ALICE, DAVE]
    assert set(parse_qs(urlsplit(asked[-1][0]).query)["ids"][0].split(",")) == {BOB, CAROL, DAVE}

    access_service_answers(monkeypatch, members_down=True)
    with pytest.raises(HTTPException) as caught:
        _check({"mode": "restricted", "allowed_user_ids": [BOB, DAVE]}, stored=stored)
    assert (caught.value.status_code, caught.value.detail["code"]) == (503, HUB_UNAVAILABLE)


def test_an_artifact_save_refuses_the_callers_own_address(monkeypatch):
    """The owner always opens their own artifact, so their address among the
    outside emails is a 400, before anything is read or published."""
    from cowork.api.v1.endpoints.artifact_workspace import OWN_EMAIL_REFUSAL

    reads: list = []
    with pytest.raises(HTTPException) as caught:
        _check({"mode": "restricted", "emails": ["e@else.com", " Alice@X.com"]}, reads=reads)
    assert (caught.value.status_code, caught.value.detail) == (400, OWN_EMAIL_REFUSAL)
    assert reads == []
    # The caller's own address is compared lowercased too.
    mixed_case = Principal(user_id=ALICE, org_id=ORG_ID, email=" Alice@X.com ")
    with pytest.raises(HTTPException) as caught:
        _check({"mode": "restricted", "emails": ["alice@x.com"]}, principal=mixed_case)
    assert (caught.value.status_code, caught.value.detail) == (400, OWN_EMAIL_REFUSAL)
    assert _check({"mode": "restricted", "emails": ["e@else.com"]}) == {"mode": "restricted", "emails": ["e@else.com"]}


def test_an_artifact_is_shared_with_at_most_the_access_services_number_of_people(monkeypatch):
    people = [str(uuid4()) for _ in range(MAX_SHARED_PEOPLE + 1)]
    with pytest.raises(HTTPException) as caught:
        _check({"mode": "restricted", "allowed_user_ids": people})
    assert caught.value.status_code == 400
    assert caught.value.detail == f"An artifact can be shared with at most {MAX_SHARED_PEOPLE} people."


def test_setting_artifact_access_checks_the_teams_it_adds(monkeypatch):
    """A team the stored rule doesn't have must be one the caller is in; an
    admin may name any. A team the rule has is kept unchecked: the owner may
    have left it, and the auth service drops a deleted team from every rule
    itself. What the entry shows decides nothing."""
    asked: list = []
    access_service_answers(monkeypatch, asked=asked)
    ok = _check({"mode": "restricted", "allowed_team_ids": [f" {TEAM_ENG} "]})
    assert ok == {"mode": "restricted", "allowed_team_ids": [TEAM_ENG]}
    # Read with the gateway's credential, never the client-set one.
    assert {bearer for _path, bearer in asked} == {MEMBER_BEARER}
    with pytest.raises(HTTPException) as caught:
        _check({"mode": "restricted", "allowed_team_ids": [TEAM_OPS, TEAM_ENG]})
    assert (caught.value.status_code, caught.value.detail) == (
        400, f"You can only share with teams you belong to: {TEAM_OPS}.",
    )
    assert _check({"mode": "restricted", "allowed_team_ids": [TEAM_OPS]}, **_AS_ADMIN)["allowed_team_ids"] == [TEAM_OPS]
    # The owner left OPS, which the rule has: kept, beside a team they add.
    kept = _check({"mode": "restricted", "allowed_team_ids": [TEAM_OPS, TEAM_ENG]}, stored=artifact_rule(teams=[TEAM_OPS]))
    assert kept["allowed_team_ids"] == [TEAM_OPS, TEAM_ENG]
    hub_teams.reset_caches_for_tests()


def test_an_artifact_save_adding_a_team_while_they_are_unreadable_is_a_503_even_for_an_admin(monkeypatch):
    """The org's default team is never sent to the auth service: naming it is
    org_allowed. While the caller's teams can't be read it can't be told
    apart, so a save that adds a team is a 503 for everyone, an admin
    included. A save that keeps the rule's teams writes none, so it goes
    through."""
    access_service_answers(monkeypatch, teams=None)
    stored = artifact_rule(teams=[TEAM_ENG])
    for who in ({}, _AS_ADMIN):
        with pytest.raises(HTTPException) as caught:
            _check({"mode": "restricted", "allowed_team_ids": [TEAM_ENG, TEAM_ALL]}, stored=stored, **who)
        assert (caught.value.status_code, caught.value.detail["code"]) == (503, HUB_UNAVAILABLE)
    assert _check(
        {"mode": "restricted", "allowed_team_ids": [TEAM_ENG], "emails": ["e@else.com"]}, stored=stored, **_AS_ADMIN,
    ) == {"mode": "restricted", "allowed_team_ids": [TEAM_ENG], "emails": ["e@else.com"]}
    hub_teams.reset_caches_for_tests()


def test_sharing_an_artifact_with_everyone_takes_share_org_only_to_turn_it_on(monkeypatch):
    """As for a skill: making an artifact org-wide, by everyone at the org or
    its default team, takes skills.share_org (or an admin); a 503 while the
    permissions can't be read. The stored rule decides whether a save makes
    it org-wide: an owner without share_org can still save an artifact that
    is already org-wide, and an entry saying so decides nothing."""
    access_service_answers(monkeypatch, default=TEAM_ALL)
    for access in (
        {"mode": "restricted", "org_allowed": True},
        {"mode": "restricted", "allowed_team_ids": [TEAM_ALL, TEAM_ENG]},
    ):
        with pytest.raises(HTTPException) as caught:
            _check(access)
        assert (caught.value.status_code, caught.value.detail) == (403, ARTIFACT_ORG_REFUSAL)
    already = _check({"mode": "restricted", "org_allowed": True, "emails": ["e@else.com"]}, stored=artifact_rule(org=True))
    assert already == {"mode": "restricted", "org_allowed": True, "emails": ["e@else.com"]}

    access_service_answers(monkeypatch, default=TEAM_ALL, permissions=("skills.share_org",))
    assert _check({"mode": "restricted", "allowed_team_ids": [TEAM_ALL, TEAM_ENG]}) == {
        "mode": "restricted", "allowed_team_ids": [TEAM_ENG], "org_allowed": True,
    }
    access_service_answers(monkeypatch, default=TEAM_ALL)
    assert _check(
        {"mode": "restricted", "allowed_team_ids": [TEAM_ALL], "org_allowed": False}, **_AS_ADMIN,
    ) == {"mode": "restricted", "allowed_team_ids": [], "org_allowed": True}

    access_service_answers(monkeypatch, permissions=None)
    with pytest.raises(HTTPException) as caught:
        _check({"mode": "restricted", "org_allowed": True})
    assert (caught.value.status_code, caught.value.detail["code"]) == (503, HUB_UNAVAILABLE)
    hub_teams.reset_caches_for_tests()


# ── off for me (a person's own on/off) ───────────────────────────────────────


def _me(org, user: str, slug: str, enabled: bool, viewer: SkillViewer | None = None) -> dict:
    """PUT /skills/{slug}/me as ``user``."""
    return skills_ep.set_skill_enabled_for_me(
        slug, SkillEnabledForMeRequest(enabled=enabled), org.as_(user), _principal(user),
        viewer or _viewer(user, teams=(TEAM_ENG,)),
    )


def _off_rows(org, user: str) -> set[str]:
    return set(user_off_for_me(org.as_(user)))


def _turn(org, user: str, viewer: SkillViewer | None = None) -> dict:
    viewer = viewer or _viewer(user, teams=(TEAM_ENG,))
    return build_turn_skills(org.as_(user).scope, project_name="general", viewer=viewer, session=org.as_(user))


def test_off_for_me_is_one_persons_alone(org):
    _create(org, ALICE, "policy")
    _share(org, ALICE, "policy", _viewer(ALICE, share_org=True), org_allowed=True)

    out = _me(org, BOB, "policy", False)
    assert (out["enabled"], out["enabledForMe"]) == (True, False)
    assert _off_rows(org, BOB) == {"policy"}
    assert _off_rows(org, CAROL) == set()
    assert _listed(org, BOB, _viewer(BOB))["policy"]["enabledForMe"] is False
    carol = _listed(org, CAROL, _viewer(CAROL))["policy"]
    assert (carol["enabled"], carol["enabledForMe"]) == (True, True)
    assert "policy" not in _turn(org, BOB)
    assert "policy" in _turn(org, CAROL, _viewer(CAROL))

    # On again is the default: no row is kept.
    assert _me(org, BOB, "policy", True)["enabledForMe"] is True
    assert _off_rows(org, BOB) == set()
    assert "policy" in _turn(org, BOB)
    session = org.as_(BOB)
    assert session.exec(session.select(SkillUserSetting)).all() == []


def test_a_builtin_turns_off_for_one_person(org):
    SkillService(org.as_(ALICE).scope).ensure_builtin_skills()
    builtin = sorted(BUILTIN_SKILL_SLUGS)[0]
    assert builtin in _turn(org, BOB)

    out = _me(org, BOB, builtin, False)
    assert (out["isBuiltin"], out["enabled"], out["enabledForMe"]) == (True, True, False)
    assert (out["canDisableForEveryone"], out["visibleToOthers"]) == (False, True)
    assert builtin not in _turn(org, BOB)
    assert builtin in _turn(org, CAROL, _viewer(CAROL))
    # It is off for Bob in an in-process turn too.
    assert builtin not in {
        p.name for p in link_turn_skills(
            scope=org.as_(BOB).scope, project_name="general",
            skills_dir=SkillService(org.as_(BOB).scope).root.parent / "bob-turn",
            off_for_me=frozenset({builtin}),
        )
    }


def test_an_always_on_skill_off_for_me_leaves_my_turns_only(org):
    _turn_fixture(org)
    bob = _viewer(BOB, teams=(TEAM_ENG,))
    _me(org, BOB, "alpha-org", False, bob)

    block = _turn(org, BOB, bob)
    assert "alpha-org" not in block
    assert [slug for slug, entry in block.items() if entry.get("always_on")] == ["beta-team"]
    carol = _viewer(CAROL, teams=(TEAM_ENG,))
    assert [slug for slug, entry in _turn(org, CAROL, carol).items() if entry.get("always_on")] == [
        "alpha-org", "beta-team",
    ]


def test_disabling_for_everyone_still_takes_edit_rights(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], allowed_user_ids=[CAROL])
    _create(org, ALICE, "policy")
    _share(org, ALICE, "policy", _viewer(ALICE, share_org=True), org_allowed=True)

    # Carol only reads the team skill and Bob the org-wide one: each turns it
    # off for themselves, never for everyone.
    carol = _listed(org, CAROL, _viewer(CAROL))["team-skill"]
    assert (carol["canDisableForEveryone"], carol["visibleToOthers"]) == (False, True)
    _refused(
        lambda: skills_ep.update_skill(
            "team-skill", SkillUpdateRequest(enabled=False), org.as_(CAROL), _principal(CAROL), _viewer(CAROL),
        ),
        403,
        skills_ep.SKILL_EDIT_REFUSAL,
    )
    assert _me(org, CAROL, "team-skill", False, _viewer(CAROL))["enabledForMe"] is False
    assert _listed(org, BOB, _viewer(BOB))["policy"]["canDisableForEveryone"] is False
    _refused(
        lambda: skills_ep.update_skill(
            "policy", SkillUpdateRequest(enabled=False), org.as_(BOB), _principal(BOB), _viewer(BOB),
        ),
        403,
        "skills.write_org",
    )

    # A team member edits the team skill, so may turn it off for everyone.
    bob = _viewer(BOB, teams=(TEAM_ENG,))
    assert _listed(org, BOB, bob)["team-skill"]["canDisableForEveryone"] is True
    out = skills_ep.update_skill("team-skill", SkillUpdateRequest(enabled=False), org.as_(BOB), _principal(BOB), bob)
    assert (out["enabled"], out["enabledForMe"]) == (False, True)
    assert "team-skill" not in _turn(org, ALICE, _viewer(ALICE, teams=(TEAM_ENG,)))
    # Turning it on for everyone turns it on for its editor too.
    _me(org, BOB, "team-skill", False, bob)
    out = skills_ep.update_skill("team-skill", SkillUpdateRequest(enabled=True), org.as_(BOB), _principal(BOB), bob)
    assert (out["enabled"], out["enabledForMe"]) == (True, True)
    # Carol's own off is hers and stays.
    assert _off_rows(org, CAROL) == {"team-skill"}


def test_a_private_skill_is_visible_to_nobody_else(org):
    _create(org, ALICE, "mine")
    listed = _listed(org, ALICE, _viewer(ALICE))["mine"]
    assert (listed["visibleToOthers"], listed["canDisableForEveryone"]) == (False, True)
    _share(org, ALICE, "mine", allowed_user_ids=[CAROL])
    assert _listed(org, ALICE, _viewer(ALICE))["mine"]["visibleToOthers"] is True


def test_off_for_me_follows_a_rename_and_goes_with_a_delete(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], allowed_user_ids=[CAROL])
    bob = _viewer(BOB, teams=(TEAM_ENG,))
    _me(org, CAROL, "team-skill", False, _viewer(CAROL))
    _me(org, BOB, "team-skill", False, bob)
    # A stale row the new name still had is dropped, not merged.
    stale = org.as_(DAVE)
    stage_enabled_for_me(stale, "eng-skill", False)
    stale.commit()

    skills_ep.update_skill("team-skill", SkillUpdateRequest(label="eng-skill"), org.as_(BOB), _principal(BOB), bob)
    assert (_off_rows(org, CAROL), _off_rows(org, BOB), _off_rows(org, DAVE)) == ({"eng-skill"}, {"eng-skill"}, set())
    assert _listed(org, CAROL, _viewer(CAROL))["eng-skill"]["enabledForMe"] is False

    skills_ep.delete_skill("eng-skill", org.as_(BOB), _principal(BOB), bob)
    assert (_off_rows(org, CAROL), _off_rows(org, BOB)) == (set(), set())
    # A new skill of the old name starts on for everyone.
    _create(org, CAROL, "eng-skill")
    assert _listed(org, CAROL, _viewer(CAROL))["eng-skill"]["enabledForMe"] is True


def test_off_for_me_needs_a_skill_the_caller_sees(org):
    _create(org, DAVE, "dave-private")
    _refused(lambda: _me(org, ALICE, "dave-private", False), 404)
    _refused(lambda: _me(org, ALICE, "no-such-skill", False), 404)
    assert _off_rows(org, ALICE) == set()


def test_settings_are_per_org(org):
    _legacy_skill(org.as_(ALICE).scope, "house-style")
    _me(org, ALICE, "house-style", False)
    assert set(user_off_for_me(org.as_(ALICE, org_id="other-org"))) == set()


def test_off_for_me_on_the_desktop(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    try:
        SkillService(LOCAL_SCOPE).create_skill("desk", "Desk body.")
        with Session(engine) as raw:
            scoped = ScopedSession(raw, LOCAL_SCOPE)
            listed = {s["id"]: s for s in skills_ep.list_skills(scoped, None, None)["skills"]}
            assert (listed["desk"]["visibleToOthers"], listed["desk"]["enabledForMe"]) == (False, True)
            out = skills_ep.set_skill_enabled_for_me("desk", SkillEnabledForMeRequest(enabled=False), scoped, None, None)
            assert out["enabledForMe"] is False
            assert user_off_for_me(scoped) == frozenset({"desk"})
            assert "desk" not in build_turn_skills(LOCAL_SCOPE, project_name="general", session=scoped)
            assert "desk" in build_turn_skills(LOCAL_SCOPE, project_name="general")
            # A delete takes the setting with it.
            skills_ep.delete_skill("desk", scoped, None, None)
            assert user_off_for_me(scoped) == frozenset()
    finally:
        get_app_settings.cache_clear()


def test_off_for_me_on_the_desktop_keeps_one_row_and_on_clears_every_one(monkeypatch, tmp_path):
    """The desktop's rows carry no org, where the org-keyed unique constraint
    doesn't hold (NULLs are distinct): the no-org partial index refuses a
    second row, and turning a skill back on deletes every matching row, so
    a twin can never keep it off."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    try:
        with Session(engine) as raw:
            scoped = ScopedSession(raw, LOCAL_SCOPE)
            stage_enabled_for_me(scoped, "desk", False)
            scoped.commit()
            raw.add(SkillUserSetting(org_id=None, user_id="local", skill_slug="desk", enabled=False))
            with pytest.raises(IntegrityError):
                raw.flush()
            raw.rollback()

            # A twin from a database without the index.
            raw.exec(text("DROP INDEX uq_skill_user_setting_user_skill_no_org"))
            raw.add(SkillUserSetting(org_id=None, user_id="local", skill_slug="desk", enabled=False))
            raw.commit()
            assert stage_enabled_for_me(scoped, "desk", True) is True
            scoped.commit()
            assert user_off_for_me(scoped) == frozenset()
            assert scoped.exec(scoped.select(SkillUserSetting)).all() == []
    finally:
        get_app_settings.cache_clear()
