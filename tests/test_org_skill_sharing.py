"""Hosted org skill sharing (ENG-3787): who sees, uses and changes a skill.

In hosted org tenancy every member reads one skill store, and who each skill
is shared with is its row in ``skill_access`` (named as auth's ArtifactAccess:
``org_allowed``, ``allowed_team_ids``, ``allowed_user_ids``, plus
``always_on``): only its owner (a new skill's row), people by Keycloak id,
teams, or the whole organization. A skill with no row and no creator in its
SKILL.md predates sharing and reads as org-wide. Org-wide and team skills are
recalled on demand like any other unless marked always on, which puts them in
every turn's prompt of the people they reach.

Handlers are called directly with an explicit SkillViewer, the suite's
convention for org routes (cowork.server wires the identity middleware only
when the process starts in org mode). Auth is stubbed at get_auth_json.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

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
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
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
from cowork.services import hub_people, hub_teams
from cowork.services.skill_access import (
    access_row,
    access_rows,
    stage_access,
    stage_delete,
    stage_only_owner,
    stage_rename,
)
from cowork.services.skill_sharing import (
    LEGACY_ORG_WIDE,
    SkillAccessRule,
    SkillViewer,
    access_rule_for,
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

ORG_ID = "33333333-3333-4333-8333-333333333333"
ALICE = "a1111111-1111-4111-8111-111111111111"
BOB = "b2222222-2222-4222-8222-222222222222"
CAROL = "c3333333-3333-4333-8333-333333333333"
DAVE = "d4444444-4444-4444-8444-444444444444"
ADMIN = "e5555555-5555-4555-8555-555555555555"
EVE = "f6666666-6666-4666-8666-666666666666"  # not in the org
MAL = "f7777777-7777-4777-8777-777777777777"  # not in the org
EMAILS = {ALICE: "alice@x.com", BOB: "bob@x.com", CAROL: "carol@x.com", DAVE: "dave@x.com", ADMIN: "admin@x.com"}
TEAM_ENG = "team-eng"
TEAM_OPS = "team-ops"

NOTHING_SHARED = {"orgAllowed": False, "allowedTeamIds": [], "allowedUserIds": [], "alwaysOn": False}


def _viewer(user: str, *, teams=(), admin=False, share_org=False, write_org=False, known=True) -> SkillViewer:
    return SkillViewer(
        user_id=user,
        email=EMAILS[user],
        team_ids=frozenset(teams),
        teams_known=known,
        is_admin=admin,
        can_share_org=admin or share_org,
        can_write_org=admin or write_org,
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
    ctx, user: str, slug: str, viewer: SkillViewer | None = None, *, admin=False, members=None, **body
) -> dict:
    """PUT /skills/{slug}/sharing as ``user``, running the org membership
    check (verify_people_in_org) first, as the route's dependency does."""
    request = SkillSharingRequest(**body)
    scoped = ctx.as_(user)
    viewer = viewer or _viewer(user, teams=(TEAM_ENG,), admin=admin)
    verified = asyncio.run(
        skills_ep.verify_people_in_org(slug, request, scoped, viewer, members or _members())
    )
    return skills_ep.set_skill_sharing(
        slug, request, scoped, _principal(user, admin=admin), viewer, verified
    )


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


def test_a_skill_with_no_row_reads_by_its_creator():
    assert access_rule_for(None, Skill(name="s", instructions="x", metadata={})) == LEGACY_ORG_WIDE
    mine = Skill(name="s", instructions="x", metadata={META_CREATED_BY: ALICE})
    assert access_rule_for(None, mine) == SkillAccessRule(owner_user_id=ALICE)
    row = SkillAccess(org_id=ORG_ID, skill_slug="s", owner_user_id=BOB, org_allowed=True, always_on=True)
    assert access_rule_for(row, mine) == SkillAccessRule(org_allowed=True, always_on=True, owner_user_id=BOB)
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
    # The creator stays skill metadata; nothing about access is in SKILL.md.
    metadata = SkillService(org.as_(ALICE).scope).get_skill("alice-notes").metadata
    assert metadata[META_CREATED_BY] == ALICE
    assert not {"share_org", "share_teams", "share_emails", "always_on", "created_by_email"} & set(metadata)
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
    assert SkillService(org.as_(ALICE).scope).get_skill("uploaded").created_by == ALICE
    assert _row(org, "uploaded").owner_user_id == ALICE
    assert "uploaded" not in _listed(org, DAVE, _viewer(DAVE, teams=(TEAM_OPS,)))


def test_a_skill_with_no_row_and_no_creator_is_org_wide_on_demand(org):
    _legacy_skill(org.as_(ALICE).scope, "house-style")

    seen = _listed(org, BOB, _viewer(BOB))["house-style"]
    assert (seen["section"], seen["access"]["orgAllowed"], seen["access"]["alwaysOn"], seen["canEdit"]) == (
        "org", True, False, False,
    )
    # Changing an org-wide skill takes skills.write_org.
    assert _listed(org, BOB, _viewer(BOB, write_org=True))["house-style"]["canEdit"] is True
    assert _row(org, "house-style") is None
    assert "house-style" in build_turn_skills(
        org.as_(BOB).scope, project_name="general", viewer=_viewer(BOB), session=org.as_(BOB),
    )


def test_a_skill_with_no_row_but_a_creator_is_theirs_alone(org):
    _legacy_skill(org.as_(ALICE).scope, "alice-draft", created_by=ALICE)

    assert "alice-draft" not in _listed(org, BOB, _viewer(BOB))
    mine = _listed(org, ALICE, _viewer(ALICE))["alice-draft"]
    assert (mine["section"], mine["canEdit"], mine["owner"]["id"]) == ("mine", True, ALICE)
    # Its first share writes its row, owned by the creator.
    _share(org, ALICE, "alice-draft", allowed_user_ids=[BOB])
    assert _row(org, "alice-draft").owner_user_id == ALICE
    assert "alice-draft" in _listed(org, BOB, _viewer(BOB))


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


def test_the_viewer_resolves_teams_and_permissions_from_auth_and_fails_closed(monkeypatch):
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
    # Taking it back takes the capability too.
    _refused(lambda: _share(org, ALICE, "shared", _viewer(ALICE, write_org=True), org_allowed=False), 403)


def test_org_wide_also_takes_owning_the_skill_or_admin(org):
    """skills.share_org alone doesn't let someone a skill was shared with
    make it org-wide, or take that back: the owner or an admin decides."""
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", allowed_team_ids=[TEAM_ENG], allowed_user_ids=[BOB])
    bob = _viewer(BOB, teams=(TEAM_ENG,), share_org=True)

    _refused(
        lambda: _share(org, BOB, "team-skill", bob, org_allowed=True), 403, skills_ep.SHARING_ORG_REFUSAL,
    )
    assert _row(org, "team-skill").org_allowed is False
    out = _share(org, ADMIN, "team-skill", _viewer(ADMIN, admin=True, share_org=True), admin=True, org_allowed=True)
    assert out["access"]["orgAllowed"] is True
    _refused(
        lambda: _share(org, BOB, "team-skill", bob, org_allowed=False), 403, skills_ep.SHARING_ORG_REFUSAL,
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


def test_the_viewer_learns_the_default_team_from_auth(monkeypatch):
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


def test_an_org_wide_skill_takes_write_org_to_change(org):
    _create(org, ALICE, "policy")
    _share(org, ALICE, "policy", _viewer(ALICE, share_org=True), org_allowed=True)

    def edit(user, viewer):
        return skills_ep.update_skill(
            "policy", SkillUpdateRequest(instructions=f"{user} edit"), org.as_(user), _principal(user), viewer,
        )

    _refused(lambda: edit(ALICE, _viewer(ALICE)), 403, "skills.write_org")
    _refused(
        lambda: skills_ep.update_skill(
            "policy", SkillUpdateRequest(enabled=False), org.as_(BOB), _principal(BOB), _viewer(BOB),
        ),
        403,
        "skills.write_org",
    )
    assert edit(BOB, _viewer(BOB, write_org=True))["declarative"] == f"{BOB} edit"
    assert edit(ADMIN, _viewer(ADMIN, admin=True))["declarative"] == f"{ADMIN} edit"


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
    # A skill from before sharing reads as org-wide, so it can be made always
    # on too: that writes its first row.
    _legacy_skill(org.as_(ALICE).scope, "house-style")
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
    # Without the rows (no session) a skill made since sharing is its
    # creator's alone, never wider.
    blind = build_turn_skills(org.as_(BOB).scope, project_name="general", viewer=_viewer(BOB, teams=(TEAM_ENG,)))
    assert "bob-own" in blind and not {"alpha-org", "beta-team", "gamma-person"} & set(blind)


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


def test_sharing_with_people_all_in_the_org_asks_auth_once(org):
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


def test_sharing_with_people_while_auth_is_unreachable_is_a_503(org):
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


def test_the_route_lookup_asks_auth_by_id(monkeypatch):
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        calls.append(path)
        return {"results": [{"id": BOB.upper(), "name": "Bob", "email": "bob@x.com"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    lookup = skills_ep.get_org_member_lookup(_Request())
    assert asyncio.run(lookup([BOB, EVE])) == frozenset({BOB})
    assert calls == [f"/me/org-members/?ids={BOB}%2C{EVE}"]


def test_the_org_membership_lookup_batches_and_fails_closed(monkeypatch):
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        calls.append(path)
        return {"results": [{"id": "u1", "name": "Bob", "email": "bob@x.com"}, {"id": "stranger"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    found = asyncio.run(hub_people.org_members_by_ids(bearer_token="t", ids=["U1", "u2"]))
    # Only what was asked; an id auth didn't return is not a member.
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
        calls.append(path)
        return {"results": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, ids=f"{BOB}, {EVE},"))
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "people": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}], "truncated": False,
    }
    # Uncached.
    asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, ids=BOB))
    assert calls == [f"/me/org-members/?ids={BOB}%2C{EVE}", f"/me/org-members/?ids={BOB}"]

    async def down(path, bearer):
        return None

    monkeypatch.setattr(hub_people, "get_auth_json", down)
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, ids=BOB))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": [], "truncated": False}
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), LOCAL_SCOPE, ids=BOB))
    assert view.model_dump(by_alias=True) == {"reachable": True, "people": [], "truncated": False}


def test_hub_people_asks_auth_for_the_directory_without_params(monkeypatch):
    hub_people.reset_caches_for_tests()
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        calls.append(path)
        return {"results": [], "truncated": False}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    asyncio.run(hub_people_ep.get_hub_people(_Request(), scope))
    assert calls == ["/me/org-members/"]
    hub_people.reset_caches_for_tests()


def test_hub_people_lists_the_org_directory_and_caches_it_per_caller(monkeypatch):
    hub_people.reset_caches_for_tests()
    calls: list[tuple[str, str]] = []

    async def get_auth_json(path, bearer):
        calls.append((path, bearer))
        return {
            "results": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}, {"name": "no id"}],
            "truncated": True,
        }

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope))
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "people": [{"id": BOB, "name": "Bob B", "email": "bob@x.com"}], "truncated": True,
    }
    assert calls == [("/me/org-members/", "tok")]
    # Cached for the caller; ids still go to auth.
    asyncio.run(hub_people_ep.get_hub_people(_Request(), scope))
    assert len(calls) == 1
    asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, ids=BOB))
    assert calls[-1][0] == f"/me/org-members/?ids={BOB}"
    # Another caller asks for their own.
    other = TenantScope(org_mode=True, org_id=ORG_ID, user_id=BOB)
    asyncio.run(hub_people_ep.get_hub_people(_Request("other"), other))
    assert calls[-1] == ("/me/org-members/", "other")
    # Five minutes, then asked again.
    clock = [hub_people.time.monotonic() + 301]
    monkeypatch.setattr(hub_people.time, "monotonic", lambda: clock[0])
    asyncio.run(hub_people_ep.get_hub_people(_Request(), scope))
    assert calls[-1] == ("/me/org-members/", "tok") and len(calls) == 4
    hub_people.reset_caches_for_tests()


def test_hub_people_failure_is_unreachable_and_retried_after_15s(monkeypatch):
    hub_people.reset_caches_for_tests()
    calls: list[str] = []
    answers = [None, {"results": [{"id": BOB}]}]

    async def get_auth_json(path, bearer):
        calls.append(path)
        return answers[min(len(calls), len(answers)) - 1]

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": [], "truncated": False}
    asyncio.run(hub_people_ep.get_hub_people(_Request(), scope))
    assert len(calls) == 1
    now = hub_people.time.monotonic()
    monkeypatch.setattr(hub_people.time, "monotonic", lambda: now + 16)
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope))
    # A plain list or no flag from auth: not truncated.
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "people": [{"id": BOB, "name": "", "email": ""}], "truncated": False,
    }
    hub_people.reset_caches_for_tests()


def test_hub_people_auth_refusing_the_caller_is_unreachable(monkeypatch):
    """Auth answers 403 to someone who is not a current member of the active
    org: no directory and no lookup, reported rather than raised."""
    hub_people.reset_caches_for_tests()
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
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": [], "truncated": False}
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, ids=BOB))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": [], "truncated": False}
    assert [path.endswith("/me/org-members/") for path in seen] == [True, True]
    hub_people.reset_caches_for_tests()


def test_hub_people_outside_org_mode_is_empty_and_reachable(monkeypatch):
    async def get_auth_json(path, bearer):
        raise AssertionError("auth asked outside org mode")

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), LOCAL_SCOPE))
    assert view.model_dump(by_alias=True) == {"reachable": True, "people": [], "truncated": False}


# ── hub teams ────────────────────────────────────────────────────────────────


class _Request:
    def __init__(self, bearer: str = "tok"):
        self.headers = {"X-MindsHub-Authorization": f"Bearer {bearer}"}


def test_hub_teams_lists_the_callers_teams_and_caches_them(monkeypatch):
    hub_teams.reset_caches_for_tests()
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        calls.append(path)
        return {"results": [{"id": TEAM_ENG, "display_name": "Eng", "is_default": True}, {"display_name": "no id"}]}

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=BOB)
    view = asyncio.run(hub_teams_ep.get_hub_teams(_Request(), scope))
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "teams": [{"id": TEAM_ENG, "displayName": "Eng", "isDefault": True}],
    }
    asyncio.run(hub_teams_ep.get_hub_teams(_Request(), scope))
    assert calls == [hub_teams.TEAMS_PATH]
    hub_teams.reset_caches_for_tests()


def test_hub_teams_unreachable_lists_none(monkeypatch):
    hub_teams.reset_caches_for_tests()

    async def get_auth_json(path, bearer):
        return None

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=BOB)
    view = asyncio.run(hub_teams_ep.get_hub_teams(_Request(), scope))
    assert view.model_dump(by_alias=True) == {"reachable": False, "teams": []}
    hub_teams.reset_caches_for_tests()


# ── artifacts ────────────────────────────────────────────────────────────────


def test_teams_and_people_reach_anton_as_owner_only_and_survive_republish():
    from cowork.services.publish import _org_audience, _with_team_audience

    access = {
        "mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": [TEAM_ENG],
        "allowed_user_ids": [BOB],
    }
    audience = _org_audience(access, None)
    assert audience == {"allowed_team_ids": [TEAM_ENG], "allowed_user_ids": [BOB]}
    assert _with_team_audience(access, audience) == {
        "mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True,
    }
    previous = {"mode": "restricted", "allowed_team_ids": [TEAM_OPS], "allowed_user_ids": [CAROL]}
    assert _org_audience({"mode": "restricted", "emails": ["a@x.com"]}, previous) == {
        "allowed_team_ids": [TEAM_OPS], "allowed_user_ids": [CAROL],
    }
    assert _org_audience({"mode": "restricted", "allowed_user_ids": []}, previous)["allowed_user_ids"] == []
    assert _with_team_audience({"mode": "public"}, audience) == {"mode": "public"}


def test_a_people_only_restriction_stays_restricted_never_public_or_nobody():
    """What the app sends for "only these people in my org": every other
    field empty and owner_only false. anton's publisher reads that as an
    empty selection and would publish it public; the plan keeps it restricted
    to the owner there and widens it to the people at auth."""
    from cowork.services.publish import _plan_access, access_from_app

    app_body = access_from_app({
        "mode": "restricted", "org_allowed": False, "allowed_team_ids": [], "allowed_user_ids": [BOB],
        "allowed_emails": [], "owner_only": False,
    })
    effective, _pwd, version, owner_side, team_audience = _plan_access(None, app_body, None, org_mode=True)
    assert effective["mode"] == "restricted"
    assert (owner_side["mode"], owner_side["owner_only"], owner_side["allowed_user_ids"]) == (
        "restricted", True, [BOB],
    )
    # Written to auth with the people: not public, and not the owner alone.
    assert team_audience is True
    # Republished with an access that leaves people out: they stay.
    again = _plan_access(None, {"mode": "restricted", "emails": []}, owner_side, org_mode=True)
    assert (again[0]["mode"], again[3]["allowed_user_ids"], again[4]) == ("restricted", [BOB], True)
    # Taking the people away (nothing else picked) is the owner alone, written
    # to auth so their access is revoked there too.
    cleared = _plan_access(None, {**app_body, "allowed_user_ids": [], "owner_only": True}, owner_side, org_mode=True)
    assert (cleared[3]["mode"], cleared[3]["allowed_user_ids"], cleared[4]) == ("restricted", [], True)
    assert cleared[2] > version
    # Outside org mode there are no org members, and still no public rule.
    local = _plan_access(None, app_body, None, org_mode=False)
    assert (local[0]["mode"], local[3]["owner_only"], local[4]) == ("restricted", True, False)


def test_the_app_sends_auths_four_audience_names_and_nothing_else():
    """org_allowed, allowed_team_ids, allowed_user_ids and allowed_emails;
    allowed_emails is anton's emails. A retired name is refused, naming the
    one to send, rather than dropped."""
    from cowork.services.publish import _plan_access, access_from_app

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
    _e, _p, _v, owner_side, _t = _plan_access(
        None, {"mode": "restricted", "allowed_team_ids": [TEAM_ENG], "org_allowed": True}, None, org_mode=True,
    )
    assert (owner_side["allowed_team_ids"], owner_side["org_allowed"], owner_side["owner_only"]) == (
        [TEAM_ENG], True, False,
    )


def test_the_team_and_people_audience_is_written_to_auth():
    from cowork.services.artifact_access import sync_artifact_team_access

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(200, json={"ok": True})

    settings = SimpleNamespace(auth_internal_base_url="https://auth.test", auth_internal_secret="s")
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    sync_artifact_team_access(
        artifact_id="artifact/1", scope=scope, owner_user_id=ALICE, workspace_id=None,
        allowed_emails=["e@else.com"], org_allowed=False, allowed_team_ids=[TEAM_ENG], allowed_user_ids=[BOB],
        access_version=3, settings=settings, client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert seen == [{
        "url": "https://auth.test/v1/internal/artifact-access/",
        "body": {
            "artifact_id": "artifact/1", "owner_keycloak_id": ALICE, "organization_id": ORG_ID,
            "workspace_id": None, "allowed_emails": ["e@else.com"], "org_allowed": False,
            "allowed_team_ids": [TEAM_ENG], "allowed_user_ids": [BOB], "access_version": 3,
        },
    }]


def test_auth_refusing_a_non_member_is_a_400_naming_them():
    from cowork.services.artifact_access import ArtifactAudienceRejected, sync_artifact_team_access

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": "not_a_member", "detail": "x", "allowed_user_ids": [EVE]})

    settings = SimpleNamespace(auth_internal_base_url="https://auth.test", auth_internal_secret="s")
    with pytest.raises(ArtifactAudienceRejected) as caught:
        sync_artifact_team_access(
            artifact_id="a", scope=TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE),
            owner_user_id=ALICE, workspace_id=None, allowed_emails=[], org_allowed=False, allowed_team_ids=[],
            allowed_user_ids=[EVE], access_version=1, settings=settings,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    # A ValueError, which the access route answers as 400.
    assert isinstance(caught.value, ValueError) and EVE in str(caught.value)


def test_auth_refusing_a_foreign_team_is_a_400_naming_it():
    """auth's unknown_team body names the teams in allowed_team_ids."""
    from cowork.services.artifact_access import ArtifactAudienceRejected, sync_artifact_team_access

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": "unknown_team", "detail": "x", "allowed_team_ids": [TEAM_OPS]})

    settings = SimpleNamespace(auth_internal_base_url="https://auth.test", auth_internal_secret="s")
    with pytest.raises(ArtifactAudienceRejected) as caught:
        sync_artifact_team_access(
            artifact_id="a", scope=TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE),
            owner_user_id=ALICE, workspace_id=None, allowed_emails=[], org_allowed=False,
            allowed_team_ids=[TEAM_OPS], access_version=1, settings=settings,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    assert TEAM_OPS in str(caught.value)


def test_the_access_write_carries_notify_only_when_asked():
    from cowork.services.artifact_access import sync_artifact_team_access

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    settings = SimpleNamespace(auth_internal_base_url="https://auth.test", auth_internal_secret="s")
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    common = dict(
        artifact_id="artifact/1", scope=scope, owner_user_id=ALICE, workspace_id=None,
        allowed_emails=["e@else.com"], org_allowed=False, allowed_team_ids=[], allowed_user_ids=[BOB],
        access_version=3, settings=settings,
    )
    sync_artifact_team_access(**common, client=httpx.Client(transport=httpx.MockTransport(handler)))
    sync_artifact_team_access(
        **common, notify_user_ids=[BOB], notify_emails=["e@else.com"], artifact_title="Q3",
        artifact_url="https://view.test/1", client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    # Empty lists are nobody to notify.
    sync_artifact_team_access(
        **common, notify_user_ids=[], notify_emails=[], artifact_title="Q3",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    notify_keys = {"notify_recipients", "notify_user_ids", "notify_emails", "artifact_title", "artifact_url"}
    assert not notify_keys & set(seen[0])
    assert {k: seen[1][k] for k in notify_keys} == {
        "notify_recipients": True, "notify_user_ids": [BOB], "notify_emails": ["e@else.com"],
        "artifact_title": "Q3", "artifact_url": "https://view.test/1",
    }
    assert not notify_keys & set(seen[2])


def test_the_post_publish_rewrite_sends_the_people(monkeypatch):
    from cowork.services import artifact_access, artifact_autopublish
    from cowork.services.publish import _sync_team_audience

    sent: list[dict] = []
    monkeypatch.setattr(artifact_access, "sync_artifact_team_access", lambda **kw: sent.append(kw))
    monkeypatch.setattr(artifact_autopublish, "_active_workspace_id", lambda scope: None)
    _sync_team_audience(
        artifact_id="a", scope=TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE),
        owner_side={"mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": [TEAM_ENG],
                    "allowed_user_ids": [BOB], "access_version": 4},
    )
    assert (sent[0]["allowed_team_ids"], sent[0]["allowed_user_ids"], sent[0]["access_version"]) == (
        [TEAM_ENG], [BOB], 4,
    )


def test_the_access_read_returns_teams_people_and_org_allowed(tmp_path):
    from cowork.services.artifacts import _published_access_for
    from cowork.services.publish import published_artifact_access

    folder = tmp_path / "artifacts" / "rep"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}))
    (folder / ".published.json").write_text(json.dumps({"index.html": {
        "report_id": "r1", "url": "u", "mode": "restricted", "emails": [], "org_allowed": True,
        "owner_only": False, "allowed_team_ids": [TEAM_ENG], "allowed_user_ids": [BOB], "access_version": 2,
    }}))
    read = _published_access_for(folder, folder / "index.html")
    assert (read["allowedTeamIds"], read["allowedUserIds"], read["orgAllowed"]) == ([TEAM_ENG], [BOB], True)
    assert "orgWide" not in read
    again = published_artifact_access(folder, artifacts_base=folder.parent)
    assert (again["allowed_team_ids"], again["allowed_user_ids"], again["org_allowed"]) == ([TEAM_ENG], [BOB], True)
    # People alone are stored owner-only for anton, and read as people.
    (folder / ".published.json").write_text(json.dumps({"index.html": {
        "report_id": "r1", "url": "u", "mode": "restricted", "emails": [], "org_allowed": False,
        "owner_only": True, "allowed_user_ids": [BOB], "access_version": 2,
    }}))
    read = _published_access_for(folder, folder / "index.html")
    assert (read["ownerOnly"], read["allowedUserIds"]) == (False, [BOB])


def _request(principal=None):
    return SimpleNamespace(
        headers={"X-MindsHub-Authorization": "Bearer t"}, state=SimpleNamespace(principal=principal),
    )


def test_setting_artifact_access_passes_people_through(monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace as aw

    async def no_auth(path, bearer):
        raise AssertionError("people are auth's to check, on the write")

    monkeypatch.setattr(hub_teams, "get_auth_json", no_auth)
    session = SimpleNamespace(scope=TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE))
    request = _request(_principal(ALICE))
    out = asyncio.run(aw._checked_team_audience(
        {"mode": "restricted", "allowed_user_ids": [f" {BOB.upper()}", BOB], "emails": ["e@else.com"]},
        request=request, session=session,
    ))
    assert out == {"mode": "restricted", "allowed_user_ids": [BOB], "emails": ["e@else.com"]}
    with pytest.raises(HTTPException) as caught:
        asyncio.run(aw._checked_team_audience(
            {"mode": "restricted", "allowed_user_ids": "nope"}, request=request, session=session,
        ))
    assert caught.value.status_code == 400
    # Outside org mode the access is left as it is: the plan keeps people
    # picked alone private (test_a_people_only_restriction_stays_restricted...).
    local = SimpleNamespace(scope=LOCAL_SCOPE)
    access = {"mode": "restricted", "allowed_user_ids": [BOB], "owner_only": False}
    assert asyncio.run(aw._checked_team_audience(access, request=_request(), session=local)) == access


def test_setting_artifact_access_checks_the_callers_teams(monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace as aw

    hub_teams.reset_caches_for_tests()

    async def get_auth_json(path, bearer):
        return {"results": [{"id": TEAM_ENG}]}

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    session = SimpleNamespace(scope=TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE))
    request = _request(_principal(ALICE))
    ok = asyncio.run(aw._checked_team_audience(
        {"mode": "restricted", "allowed_team_ids": [f" {TEAM_ENG} "], "org_allowed": True},
        request=request, session=session,
    ))
    assert ok == {"mode": "restricted", "allowed_team_ids": [TEAM_ENG], "org_allowed": True}
    with pytest.raises(HTTPException) as caught:
        asyncio.run(aw._checked_team_audience(
            {"mode": "restricted", "allowed_team_ids": [TEAM_OPS]}, request=request, session=session,
        ))
    assert caught.value.status_code == 400
    admin_request = _request(_principal(ADMIN, admin=True))
    assert asyncio.run(aw._checked_team_audience(
        {"mode": "restricted", "allowed_team_ids": [TEAM_OPS]}, request=admin_request, session=session,
    ))["allowed_team_ids"] == [TEAM_OPS]
    # The org's default team is everyone: dropped from the teams, org-wide.
    hub_teams.reset_caches_for_tests()

    async def with_default(path, bearer):
        return {"results": [{"id": TEAM_ENG}, {"id": "team-all", "is_default": True}]}

    monkeypatch.setattr(hub_teams, "get_auth_json", with_default)
    for caller in (request, admin_request):
        assert asyncio.run(aw._checked_team_audience(
            {"mode": "restricted", "allowed_team_ids": ["team-all", TEAM_ENG]}, request=caller, session=session,
        )) == {"mode": "restricted", "allowed_team_ids": [TEAM_ENG], "org_allowed": True}
    assert asyncio.run(aw._checked_team_audience(
        {"mode": "restricted", "allowed_team_ids": ["team-all"], "org_allowed": False},
        request=request, session=session,
    )) == {"mode": "restricted", "allowed_team_ids": [], "org_allowed": True}
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
