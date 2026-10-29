"""Hosted org skill sharing (ENG-3787): who sees, uses and changes a skill.

In hosted org tenancy every member reads one skill store, and each skill
records who it is shared with in its SKILL.md metadata: only its owner (the
default for a new skill), people by email, teams, or the whole organization.
A skill with no sharing metadata predates sharing and reads as org-wide.
Org-wide and team skills are recalled on demand like any other unless marked
always on, which puts them in every turn's prompt of the people they reach.

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
from anton.core.tools.skill_format import dump_skill, parse_skill_text
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import hub_people as hub_people_ep
from cowork.api.v1.endpoints import hub_teams as hub_teams_ep
from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.skill import (
    META_ALWAYS_ON,
    META_CREATED_BY,
    META_CREATED_BY_EMAIL,
    META_SHARE_EMAILS,
    META_SHARE_ORG,
    META_SHARE_TEAMS,
    Skill,
)
from cowork.principal import ORG_MANAGE_ROLE, Principal
from cowork.schemas.skills import SkillCreateRequest, SkillSharingRequest, SkillUpdateRequest
from cowork.services import hub_people, hub_teams
from cowork.services.always_on_skills import (
    ALWAYS_ON_SKILL_MAX_CHARS,
    ALWAYS_ON_TOTAL_MAX_CHARS,
    render_always_on_skills,
)
from cowork.services.skill_sharing import (
    SkillSharing,
    SkillViewer,
    normalize_emails,
    resolve_skill_viewer,
)
from cowork.services.skills import (
    BUILTIN_SKILL_SLUGS,
    SkillService,
    always_on_prompt_skills,
    build_turn_skills,
)
from cowork.turnqueue.producer import _fit_request

ORG_ID = "33333333-3333-4333-8333-333333333333"
ALICE = "a1111111-1111-4111-8111-111111111111"
BOB = "b2222222-2222-4222-8222-222222222222"
CAROL = "c3333333-3333-4333-8333-333333333333"
DAVE = "d4444444-4444-4444-8444-444444444444"
ADMIN = "e5555555-5555-4555-8555-555555555555"
EMAILS = {ALICE: "alice@x.com", BOB: "bob@x.com", CAROL: "carol@x.com", DAVE: "dave@x.com", ADMIN: "admin@x.com"}
TEAM_ENG = "team-eng"
TEAM_OPS = "team-ops"

HEADER = (
    "\n\n## Organization and team skills (always apply)\n"
    "These are how this organization works. Follow them on every task where they apply; "
    "they override general defaults but never the user's explicit instructions in this conversation."
)


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

    def as_(user: str) -> ScopedSession:
        session = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id=ORG_ID, user_id=user))
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


def _members(*emails: str, calls: list | None = None):
    """An org member lookup that knows ``emails`` (every test user by default)."""
    known = frozenset(emails or EMAILS.values())

    async def lookup(asked):
        asked = list(asked)
        if calls is not None:
            calls.append(asked)
        return frozenset(e for e in asked if e in known)

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
    """A skill written before sharing: no share_org in its metadata."""
    root = SkillService(scope).root / slug
    root.mkdir(parents=True)
    skill = Skill(name=slug, description=slug, instructions=body, metadata=metadata)
    (root / "SKILL.md").write_text(dump_skill(skill))


# ── creating and listing ─────────────────────────────────────────────────────


def test_a_new_skill_is_only_its_creators(org):
    created = _create(org, ALICE, "alice-notes", teams=(TEAM_ENG,))

    assert created["section"] == "mine"
    assert created["sharing"] == {"org": False, "teamIds": [], "emails": [], "alwaysOn": False}
    assert created["canEdit"] is True
    assert created["owner"] == {"id": ALICE, "email": "alice@x.com"}
    metadata = SkillService(org.as_(ALICE).scope).get_skill("alice-notes").metadata
    assert (metadata[META_SHARE_ORG], metadata[META_CREATED_BY], metadata[META_CREATED_BY_EMAIL]) == (
        "false", ALICE, "alice@x.com",
    )
    assert "alice-notes" not in _listed(org, BOB, _viewer(BOB, teams=(TEAM_ENG,)))
    _refused(lambda: skills_ep.get_skill("alice-notes", org.as_(BOB), _principal(BOB), _viewer(BOB)), 404)


def test_an_upload_drops_the_sharing_and_creator_its_file_claims(org):
    skill = Skill(
        name="uploaded", description="u", instructions="x",
        metadata={
            META_SHARE_ORG: "true", META_CREATED_BY: DAVE, META_SHARE_TEAMS: TEAM_OPS, META_ALWAYS_ON: "true",
        },
    )
    uploaded = skills_ep._upload_skill_bytes(
        dump_skill(skill).encode(), "SKILL.md", org.as_(ALICE), _principal(ALICE), viewer=_viewer(ALICE),
    )
    assert (uploaded["section"], uploaded["sharing"]["org"], uploaded["sharing"]["teamIds"]) == ("mine", False, [])
    assert uploaded["sharing"]["alwaysOn"] is False
    stored = SkillService(org.as_(ALICE).scope).get_skill("uploaded")
    assert stored.created_by == ALICE
    assert META_ALWAYS_ON not in stored.metadata


def test_a_skill_with_no_sharing_metadata_is_org_wide(org):
    _legacy_skill(org.as_(ALICE).scope, "house-style", created_by=ALICE)

    seen = _listed(org, BOB, _viewer(BOB))["house-style"]
    assert (seen["section"], seen["sharing"]["org"], seen["canEdit"]) == ("org", True, False)
    # Changing an org-wide skill takes skills.write_org, even for its creator.
    assert _listed(org, ALICE, _viewer(ALICE))["house-style"]["canEdit"] is False
    assert _listed(org, ALICE, _viewer(ALICE))["house-style"]["section"] == "mine"
    assert _listed(org, BOB, _viewer(BOB, write_org=True))["house-style"]["canEdit"] is True


def test_each_member_sees_a_skill_in_their_own_section(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", team_ids=[TEAM_ENG], emails=["carol@x.com"])

    assert _listed(org, ALICE, _viewer(ALICE, teams=(TEAM_ENG,)))["team-skill"]["section"] == "mine"
    bob = _listed(org, BOB, _viewer(BOB, teams=(TEAM_ENG,)))["team-skill"]
    assert (bob["section"], bob["canEdit"], bob["owner"]["id"]) == ("team", True, ALICE)
    # Recipients see who owns it, but only the owner or an admin sees the list.
    assert (bob["owner"]["email"], bob["sharing"]["emails"]) == ("alice@x.com", [])
    carol = _listed(org, CAROL, _viewer(CAROL))["team-skill"]
    assert (carol["section"], carol["canEdit"]) == ("shared_with_me", False)
    assert "team-skill" not in _listed(org, DAVE, _viewer(DAVE, teams=(TEAM_OPS,)))
    # An admin sees and changes every skill, but their turns don't use
    # another member's private one.
    _create(org, DAVE, "dave-private")
    admin = _viewer(ADMIN, admin=True)
    seen = _listed(org, ADMIN, admin)["dave-private"]
    assert (seen["section"], seen["canEdit"]) == ("shared_with_me", True)
    assert "dave-private" not in build_turn_skills(org.as_(ADMIN).scope, project_name="general", viewer=admin)


def test_an_unreachable_team_listing_hides_team_only_skills(org):
    _create(org, ALICE, "team-only")
    _share(org, ALICE, "team-only", team_ids=[TEAM_ENG])
    _create(org, ALICE, "for-bob")
    _share(org, ALICE, "for-bob", emails=["bob@x.com"])

    blind = _viewer(BOB, known=False)
    assert set(_listed(org, BOB, blind)) == {"for-bob"}
    assert "team-only" not in build_turn_skills(org.as_(BOB).scope, project_name="general", viewer=blind)


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


def test_sharing_normalizes_emails_and_keeps_teams(org):
    _create(org, ALICE, "shared")
    out = _share(org, ALICE, "shared", team_ids=[TEAM_ENG, TEAM_ENG], emails=[" Carol@X.com", "carol@x.com", "BOB@x.com"])

    assert out["sharing"] == {
        "org": False, "teamIds": [TEAM_ENG], "emails": ["carol@x.com", "bob@x.com"], "alwaysOn": False,
    }
    metadata = SkillService(org.as_(ALICE).scope).get_skill("shared").metadata
    assert (metadata[META_SHARE_TEAMS], metadata[META_SHARE_EMAILS]) == (TEAM_ENG, "carol@x.com,bob@x.com")
    _refused(lambda: _share(org, ALICE, "shared", emails=["not-an-email"]), 400, "not a valid email")


def test_only_the_owner_or_an_admin_changes_teams_and_people(org):
    _create(org, ALICE, "shared")
    _share(org, ALICE, "shared", team_ids=[TEAM_ENG])

    _refused(lambda: _share(org, BOB, "shared", emails=["dave@x.com"]), 403, "owner or an organization admin")
    admin = _viewer(ADMIN, admin=True)
    out = _share(org, ADMIN, "shared", admin, admin=True, team_ids=[TEAM_ENG, TEAM_OPS])
    # An admin may name a team they aren't in.
    assert out["sharing"]["teamIds"] == [TEAM_ENG, TEAM_OPS]


def test_a_team_must_be_one_the_sharer_belongs_to(org):
    _create(org, ALICE, "shared")

    _refused(lambda: _share(org, ALICE, "shared", team_ids=[TEAM_OPS]), 400, "teams you belong to")
    blind = _viewer(ALICE, known=False)
    _refused(lambda: _share(org, ALICE, "shared", blind, team_ids=[TEAM_ENG]), 503)
    # Keeping a team already listed needs no listing.
    _share(org, ALICE, "shared", team_ids=[TEAM_ENG])
    _share(org, ALICE, "shared", blind, team_ids=[TEAM_ENG], emails=["bob@x.com"])


def test_org_wide_takes_the_share_org_capability(org):
    _create(org, ALICE, "shared")

    _refused(lambda: _share(org, ALICE, "shared", org=True), 403, "skills.share_org")
    out = _share(org, ALICE, "shared", _viewer(ALICE, share_org=True), org=True)
    assert (out["sharing"]["org"], out["section"]) == (True, "mine")
    # Taking it back takes the capability too.
    _refused(lambda: _share(org, ALICE, "shared", _viewer(ALICE, write_org=True), org=False), 403)


TEAM_ALL = "team-everyone"


def _with_default(viewer: SkillViewer) -> SkillViewer:
    """``viewer`` in an org whose default team (everyone) is TEAM_ALL."""
    return replace(viewer, team_ids=viewer.team_ids | {TEAM_ALL}, default_team_id=TEAM_ALL)


def test_sharing_with_the_default_team_is_sharing_org_wide(org):
    _create(org, ALICE, "shared")

    member = _with_default(_viewer(ALICE, teams=(TEAM_ENG,)))
    _refused(lambda: _share(org, ALICE, "shared", member, team_ids=[TEAM_ALL]), 403, "skills.share_org")
    _refused(lambda: _share(org, ALICE, "shared", member, team_ids=[TEAM_ENG, TEAM_ALL]), 403, "skills.share_org")
    assert SkillService(org.as_(ALICE).scope).get_skill("shared").metadata[META_SHARE_ORG] == "false"

    sharer = _with_default(_viewer(ALICE, teams=(TEAM_ENG,), share_org=True))
    out = _share(org, ALICE, "shared", sharer, team_ids=[TEAM_ALL, TEAM_ENG])
    assert (out["sharing"]["org"], out["sharing"]["teamIds"]) == (True, [TEAM_ENG])
    metadata = SkillService(org.as_(ALICE).scope).get_skill("shared").metadata
    assert (metadata[META_SHARE_ORG], metadata[META_SHARE_TEAMS]) == ("true", TEAM_ENG)


def test_a_stored_default_team_share_reads_as_org_wide(org):
    _legacy_skill(
        org.as_(ALICE).scope, "everyone", created_by=ALICE,
        **{META_SHARE_ORG: "false", META_SHARE_TEAMS: TEAM_ALL, META_ALWAYS_ON: "true"},
    )
    bob = _with_default(_viewer(BOB))
    seen = _listed(org, BOB, bob)["everyone"]
    assert (seen["section"], seen["sharing"]["org"], seen["sharing"]["teamIds"]) == ("org", True, [])
    # Org-wide's edit rule: a member of the default team alone doesn't change it.
    assert seen["canEdit"] is False
    assert _listed(org, BOB, replace(bob, can_write_org=True))["everyone"]["canEdit"] is True
    assert "everyone" in build_turn_skills(org.as_(BOB).scope, project_name="general", viewer=bob)
    # Rewritten as org-wide the next time its sharing is saved.
    _share(org, ALICE, "everyone", _with_default(_viewer(ALICE)), emails=["carol@x.com"])
    metadata = SkillService(org.as_(ALICE).scope).get_skill("everyone").metadata
    assert (metadata[META_SHARE_ORG], META_SHARE_TEAMS in metadata) == ("true", False)


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
    assert out["sharing"] == {"org": False, "teamIds": [], "emails": [], "alwaysOn": False}


def test_a_skill_the_caller_cannot_see_or_a_builtin_is_refused(org):
    _create(org, DAVE, "dave-private")
    _refused(lambda: _share(org, ALICE, "dave-private", org=False), 404)
    SkillService(org.as_(ALICE).scope).ensure_builtin_skills()
    builtin = sorted(BUILTIN_SKILL_SLUGS)[0]
    _refused(lambda: _share(org, ADMIN, builtin, _viewer(ADMIN, admin=True), admin=True, org=False), 403)


# ── changing a shared skill ──────────────────────────────────────────────────


def test_team_members_change_a_team_skill_and_recipients_only_read(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", team_ids=[TEAM_ENG], emails=["carol@x.com"])

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
    # A rename keeps the sharing: the frontmatter round trip is exact.
    renamed = skills_ep.update_skill(
        "team-skill", SkillUpdateRequest(label="eng-skill"), org.as_(BOB), _principal(BOB),
        _viewer(BOB, teams=(TEAM_ENG,)),
    )
    assert renamed["sharing"]["teamIds"] == [TEAM_ENG]
    assert SkillService(org.as_(ALICE).scope).get_skill("eng-skill").metadata[META_SHARE_EMAILS] == "carol@x.com"
    skills_ep.delete_skill("eng-skill", org.as_(BOB), _principal(BOB), _viewer(BOB, teams=(TEAM_ENG,)))


def test_an_org_wide_skill_takes_write_org_to_change(org):
    _create(org, ALICE, "policy")
    _share(org, ALICE, "policy", _viewer(ALICE, share_org=True), org=True)

    def edit(user, viewer):
        return skills_ep.update_skill(
            "policy", SkillUpdateRequest(instructions=f"{user} edit"), org.as_(user), _principal(user), viewer,
        )

    _refused(lambda: edit(ALICE, _viewer(ALICE)), 403, "skills.write_org")
    # Always-on can't be switched off by someone who can't change the skill.
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
ALWAYS_ON_ORG = "Only admins can make an organization-wide skill always on."


def _stored_always_on(org, slug: str) -> str | None:
    return SkillService(org.as_(ALICE).scope).get_skill(slug).metadata.get(META_ALWAYS_ON)


def test_org_and_team_skills_are_on_demand_by_default(org):
    _legacy_skill(org.as_(ALICE).scope, "house-style", created_by=ALICE)
    _create(org, ALICE, "team-skill")
    out = _share(org, ALICE, "team-skill", team_ids=[TEAM_ENG])

    assert out["sharing"]["alwaysOn"] is False
    listed = _listed(org, BOB, _viewer(BOB, teams=(TEAM_ENG,)))
    assert listed["house-style"]["sharing"]["alwaysOn"] is False
    assert listed["team-skill"]["sharing"]["alwaysOn"] is False


def test_always_on_needs_an_org_or_team_share(org):
    _create(org, ALICE, "private")
    _refused(lambda: _share(org, ALICE, "private", always_on=True), 400, ALWAYS_ON_UNSHARED)
    _refused(
        lambda: _share(org, ALICE, "private", emails=["bob@x.com"], always_on=True), 400, ALWAYS_ON_UNSHARED,
    )
    # Shared with a team in the same request is enough.
    out = _share(org, ALICE, "private", team_ids=[TEAM_ENG], always_on=True)
    assert (out["sharing"]["alwaysOn"], _stored_always_on(org, "private")) == (True, "true")


def test_an_org_wide_skill_takes_share_org_to_be_always_on(org):
    _create(org, ALICE, "rules")
    _share(org, ALICE, "rules", _viewer(ALICE, share_org=True), org=True)

    _refused(
        lambda: _share(org, ALICE, "rules", _viewer(ALICE, write_org=True), always_on=True), 403, ALWAYS_ON_ORG,
    )
    out = _share(org, BOB, "rules", _viewer(BOB, share_org=True), always_on=True)
    assert out["sharing"]["alwaysOn"] is True
    assert _listed(org, CAROL, _viewer(CAROL))["rules"]["sharing"]["alwaysOn"] is True
    # Turning it off takes the same.
    _refused(lambda: _share(org, ALICE, "rules", always_on=False), 403, ALWAYS_ON_ORG)
    out = _share(org, ADMIN, "rules", _viewer(ADMIN, admin=True), admin=True, always_on=False)
    assert out["sharing"]["alwaysOn"] is False
    assert _stored_always_on(org, "rules") is None
    # A legacy skill reads as org-wide, so it can be made always on too.
    _legacy_skill(org.as_(ALICE).scope, "house-style", created_by=ALICE)
    out = _share(org, BOB, "house-style", _viewer(BOB, share_org=True), always_on=True)
    assert (out["sharing"]["org"], out["sharing"]["alwaysOn"]) == (True, True)


def test_a_team_skill_takes_edit_rights_to_be_always_on(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", team_ids=[TEAM_ENG], emails=["carol@x.com"])

    # Carol only reads it (shared by email).
    _refused(lambda: _share(org, CAROL, "team-skill", _viewer(CAROL), always_on=True), 403)
    # A member of the team edits it, so may make it always on.
    out = _share(org, BOB, "team-skill", _viewer(BOB, teams=(TEAM_ENG,)), always_on=True)
    assert out["sharing"]["alwaysOn"] is True
    _refused(lambda: _share(org, CAROL, "team-skill", _viewer(CAROL), always_on=False), 403)
    out = _share(org, ADMIN, "team-skill", _viewer(ADMIN, admin=True), admin=True, always_on=False)
    assert out["sharing"]["alwaysOn"] is False
    out = _share(org, ALICE, "team-skill", _viewer(ALICE), always_on=True)
    assert out["sharing"]["alwaysOn"] is True


def test_unsharing_from_the_org_and_every_team_clears_always_on(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", team_ids=[TEAM_ENG], always_on=True)

    out = _share(org, ALICE, "team-skill", team_ids=[], emails=["bob@x.com"])
    assert out["sharing"]["alwaysOn"] is False
    assert _stored_always_on(org, "team-skill") is None
    # Sharing it with a team again doesn't bring it back.
    out = _share(org, ALICE, "team-skill", team_ids=[TEAM_ENG])
    assert out["sharing"]["alwaysOn"] is False


def test_the_sharing_body_accepts_the_camel_case_spelling(org):
    _create(org, ALICE, "team-skill")
    body = SkillSharingRequest.model_validate({"teamIds": [TEAM_ENG], "alwaysOn": True})
    assert body.always_on is True
    out = skills_ep.set_skill_sharing(
        "team-skill", body, org.as_(ALICE), _principal(ALICE), _viewer(ALICE, teams=(TEAM_ENG,)),
    )
    assert out["sharing"]["alwaysOn"] is True


def test_always_on_survives_edits_renames_and_toggles(org):
    _create(org, ALICE, "team-skill")
    _share(org, ALICE, "team-skill", team_ids=[TEAM_ENG], always_on=True)
    alice = _viewer(ALICE, teams=(TEAM_ENG,))

    skills_ep.update_skill(
        "team-skill", SkillUpdateRequest(label="renamed", instructions="New body."),
        org.as_(ALICE), _principal(ALICE), alice,
    )
    skills_ep.update_skill("renamed", SkillUpdateRequest(enabled=False), org.as_(ALICE), _principal(ALICE), alice)
    out = skills_ep.update_skill(
        "renamed", SkillUpdateRequest(enabled=True), org.as_(ALICE), _principal(ALICE), alice,
    )
    assert out["sharing"]["alwaysOn"] is True
    assert _stored_always_on(org, "renamed") == "true"
    # A sharing change that leaves it on a team keeps it.
    out = _share(org, ALICE, "renamed", emails=["carol@x.com"])
    assert out["sharing"]["alwaysOn"] is True


# ── turns ────────────────────────────────────────────────────────────────────


def _turn_fixture(org):
    """Skills of every kind, as Bob (team eng, bob@x.com) meets them."""
    alice = org.as_(ALICE).scope
    _legacy_skill(alice, "zeta-legacy", "Zeta body.")
    _legacy_skill(alice, "omega-legacy-on", "Omega body.", always_on="true")
    _create(org, ALICE, "alpha-org")
    _share(org, ALICE, "alpha-org", _viewer(ALICE, share_org=True), org=True, always_on=True)
    _create(org, ALICE, "beta-team")
    _share(org, ALICE, "beta-team", team_ids=[TEAM_ENG], always_on=True)
    _create(org, ALICE, "delta-org")
    _share(org, ALICE, "delta-org", _viewer(ALICE, share_org=True), org=True)
    _create(org, ALICE, "eta-team")
    _share(org, ALICE, "eta-team", team_ids=[TEAM_ENG])
    _create(org, ALICE, "gamma-email")
    _share(org, ALICE, "gamma-email", emails=["bob@x.com"])
    _create(org, BOB, "bob-own")
    _create(org, DAVE, "dave-private")
    _create(org, ALICE, "off-org")
    _share(org, ALICE, "off-org", _viewer(ALICE, share_org=True), org=True)
    skills_ep.update_skill(
        "off-org", SkillUpdateRequest(enabled=False), org.as_(ALICE), _principal(ALICE),
        _viewer(ALICE, write_org=True),
    )
    _create(org, ALICE, "elsewhere-team")
    _share(org, ALICE, "elsewhere-team", team_ids=[TEAM_ENG])
    SkillService(alice).update_skill("elsewhere-team", projects=["other"])


def test_the_pod_gets_only_visible_skills_with_always_on_first(org):
    _turn_fixture(org)

    block = build_turn_skills(org.as_(BOB).scope, project_name="general", viewer=_viewer(BOB, teams=(TEAM_ENG,)))

    always_on = [slug for slug, entry in block.items() if entry.get("always_on")]
    assert always_on == ["alpha-org", "omega-legacy-on", "beta-team"]
    assert list(block)[:3] == always_on
    # Org-wide and team skills not marked always on, legacy ones included,
    # go as on demand.
    assert {"gamma-email", "bob-own", "zeta-legacy", "delta-org", "eta-team"} <= set(block)
    assert not any(block[slug].get("always_on") for slug in ("zeta-legacy", "delta-org", "eta-team"))
    assert not {"dave-private", "off-org", "elsewhere-team"} & set(block)
    assert not any(block[slug].get("always_on") for slug in BUILTIN_SKILL_SLUGS if slug in block)
    assert block["alpha-org"]["files"]["SKILL.md"].startswith("---\n")
    assert "always_on" not in block["gamma-email"]


def test_the_in_process_suffix_renders_the_same_always_on_skills(org):
    _turn_fixture(org)

    skills = always_on_prompt_skills(
        scope=org.as_(BOB).scope, project_name="general", viewer=_viewer(BOB, teams=(TEAM_ENG,)),
    )
    assert skills == [
        {"name": "alpha-org", "body": "Body of alpha-org."},
        {"name": "omega-legacy-on", "body": "Omega body."},
        {"name": "beta-team", "body": "Body of beta-team."},
    ]
    assert render_always_on_skills(skills) == (
        HEADER
        + "\n\n### alpha-org\nBody of alpha-org."
        + "\n\n### omega-legacy-on\nOmega body."
        + "\n\n### beta-team\nBody of beta-team."
    )
    # Without a viewer (no caller credential) only org-wide skills apply.
    assert [s["name"] for s in always_on_prompt_skills(
        scope=org.as_(BOB).scope, project_name="general", viewer=None,
    )] == ["alpha-org", "omega-legacy-on"]


def test_the_render_respects_both_budgets():
    long_line = "x" * 50
    body = "\n".join([long_line] * 200)  # 10199 chars
    rendered = render_always_on_skills([{"name": "big", "body": body}])
    cut = body[:ALWAYS_ON_SKILL_MAX_CHARS]
    cut = cut[: cut.rfind("\n")]
    assert rendered == HEADER + f"\n\n### big\n{cut}\n[…truncated; the full skill is available on demand]"
    # One line longer than the cap is cut at the cap.
    one_line = render_always_on_skills([{"name": "line", "body": "y" * 7000}])
    assert one_line.endswith("y" * ALWAYS_ON_SKILL_MAX_CHARS + "\n[…truncated; the full skill is available on demand]")
    # Past the total, the first skill that doesn't fit and every later one,
    # however small, are named instead.
    full = "z" * (ALWAYS_ON_SKILL_MAX_CHARS - 1)
    skills = [{"name": f"s{i}", "body": full} for i in range(4)] + [{"name": "tiny", "body": "t"}]
    rendered = render_always_on_skills([{"name": "first", "body": "a" * 100}, *skills])
    assert 100 + 3 * len(full) <= ALWAYS_ON_TOTAL_MAX_CHARS < 100 + 4 * len(full)
    assert "### s2" in rendered and "### s3" not in rendered and "### tiny" not in rendered
    assert rendered.endswith("\n\n(Also available on demand: s3, tiny.)")
    assert render_always_on_skills([]) == ""


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


def test_the_producer_sends_the_skills_block(monkeypatch):
    """stream_remote_replies puts ``skills`` on the job, absent when empty."""
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
    assert captured[0]["skills"] == block
    assert "skills" not in captured[1]


# ── desktop and local deployments ────────────────────────────────────────────


def test_local_deployments_are_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    try:
        service = SkillService(LOCAL_SCOPE)
        service.create_skill("desk", "Desk body.")
        assert SkillService(LOCAL_SCOPE).get_skill("desk").metadata.get(META_SHARE_ORG) is None
        assert build_turn_skills(LOCAL_SCOPE, project_name="general") == {
            "desk": {"files": {"SKILL.md": (service.root / "desk" / "SKILL.md").read_text()}}
        }
        assert always_on_prompt_skills(scope=LOCAL_SCOPE, project_name="general", viewer=None) == []
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
    out = _share(org, ALICE, "shared", members=_members(calls=calls), emails=["bob@x.com", "Carol@x.com"])
    assert out["sharing"]["emails"] == ["bob@x.com", "carol@x.com"]
    assert calls == [["bob@x.com", "carol@x.com"]]


def test_sharing_with_someone_outside_the_org_is_refused_naming_them(org):
    _create(org, ALICE, "shared")
    _refused(
        lambda: _share(org, ALICE, "shared", emails=["bob@x.com", "eve@else.com", "mal@else.com"]),
        400,
        "Skills can only be shared with people in your organization: eve@else.com, mal@else.com",
    )
    stored = SkillService(org.as_(ALICE).scope).get_skill("shared").metadata
    assert META_SHARE_EMAILS not in stored


def test_sharing_with_people_while_auth_is_unreachable_is_a_503(org):
    _create(org, ALICE, "shared")

    async def unreachable(asked):
        return None

    _refused(lambda: _share(org, ALICE, "shared", members=unreachable, emails=["bob@x.com"]), 503)


def test_people_already_shared_with_are_not_asked_about_again(org):
    _create(org, ALICE, "shared")
    _share(org, ALICE, "shared", emails=["bob@x.com"])

    calls: list = []

    async def unreachable(asked):
        calls.append(list(asked))
        return None

    # Keeping bob, adding a team, and the sharer's own address: nothing to ask.
    out = _share(
        org, ALICE, "shared", members=unreachable, team_ids=[TEAM_ENG], emails=["BOB@x.com", "alice@x.com"]
    )
    assert out["sharing"]["teamIds"] == [TEAM_ENG]
    assert calls == []
    # Adding carol beside bob asks about carol alone.
    asked: list = []
    _share(org, ALICE, "shared", members=_members(calls=asked), emails=["bob@x.com", "carol@x.com"])
    assert asked == [["carol@x.com"]]


def test_an_address_added_without_the_check_is_a_conflict(org):
    """The route re-checks under the skill's lock: an address the dependency
    didn't confirm (the skill changed in between) isn't written."""
    _create(org, ALICE, "shared")
    _refused(
        lambda: skills_ep.set_skill_sharing(
            "shared", SkillSharingRequest(emails=["bob@x.com"]), org.as_(ALICE), _principal(ALICE),
            _viewer(ALICE), frozenset(),
        ),
        409,
    )


def test_the_org_membership_lookup_batches_and_fails_closed(monkeypatch):
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        calls.append(path)
        return {"results": [{"id": "u1", "name": "Bob", "email": "Bob@x.com"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    found = asyncio.run(hub_people.org_members_among(bearer_token="t", emails=["bob@x.com", "eve@else.com"]))
    assert found == frozenset({"bob@x.com"})
    assert calls == ["/me/org-members/?emails=bob%40x.com%2Ceve%40else.com"]

    many = [f"p{i}@x.com" for i in range(60)]
    calls.clear()
    asyncio.run(hub_people.org_members_among(bearer_token="t", emails=many))
    assert len(calls) == 2

    async def down(path, bearer):
        return None

    monkeypatch.setattr(hub_people, "get_auth_json", down)
    assert asyncio.run(hub_people.org_members_among(bearer_token="t", emails=["bob@x.com"])) is None
    assert asyncio.run(hub_people.org_members_among(bearer_token="", emails=["bob@x.com"])) is None
    assert asyncio.run(hub_people.org_members_among(bearer_token="t", emails=[])) == frozenset()


# ── hub people ───────────────────────────────────────────────────────────────


def test_hub_people_searches_the_org_and_caches_per_query(monkeypatch):
    hub_people.reset_caches_for_tests()
    calls: list[str] = []

    async def get_auth_json(path, bearer):
        calls.append(path)
        return {"results": [{"id": "u1", "name": "Bob B", "email": "bob@x.com"}, {"name": "no id"}]}

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, q="bo", limit=50))
    assert view.model_dump(by_alias=True) == {
        "reachable": True, "people": [{"id": "u1", "name": "Bob B", "email": "bob@x.com"}],
    }
    asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, q=" BO ", limit=50))
    assert calls == ["/me/org-members/?q=bo&limit=20"]
    asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, q="bob", limit=5))
    assert calls[-1] == "/me/org-members/?q=bob&limit=5"
    # Another caller doesn't get alice's cached answer.
    other = TenantScope(org_mode=True, org_id=ORG_ID, user_id=BOB)
    asyncio.run(hub_people_ep.get_hub_people(_Request("other"), other, q="bo", limit=50))
    assert len(calls) == 3
    hub_people.reset_caches_for_tests()


def test_hub_people_short_query_answers_empty_without_asking(monkeypatch):
    hub_people.reset_caches_for_tests()

    async def get_auth_json(path, bearer):
        raise AssertionError("auth asked for a one-character query")

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    for q in ("", "b", " b "):
        view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, q=q, limit=10))
        assert view.model_dump(by_alias=True) == {"reachable": True, "people": []}


def test_hub_people_unreachable_is_reported_not_raised(monkeypatch):
    hub_people.reset_caches_for_tests()

    async def get_auth_json(path, bearer):
        return None

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), scope, q="bob", limit=10))
    assert view.model_dump(by_alias=True) == {"reachable": False, "people": []}
    hub_people.reset_caches_for_tests()


def test_hub_people_outside_org_mode_is_empty_and_reachable(monkeypatch):
    async def get_auth_json(path, bearer):
        raise AssertionError("auth asked outside org mode")

    monkeypatch.setattr(hub_people, "get_auth_json", get_auth_json)
    view = asyncio.run(hub_people_ep.get_hub_people(_Request(), LOCAL_SCOPE, q="bob", limit=10))
    assert view.model_dump(by_alias=True) == {"reachable": True, "people": []}


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


# ── small pieces ─────────────────────────────────────────────────────────────


def test_sharing_metadata_reads_and_writes_strings():
    legacy = Skill(name="s", instructions="x", metadata={})
    assert SkillSharing.of(legacy) == SkillSharing(org=True)
    metadata: dict[str, str] = {META_SHARE_TEAMS: "old"}
    SkillSharing(org=False, team_ids=("a", "b"), emails=()).apply(metadata)
    assert metadata == {META_SHARE_ORG: "false", META_SHARE_TEAMS: "a,b"}
    SkillSharing(org=False, team_ids=("a",), always_on=True).apply(metadata)
    assert metadata == {META_SHARE_ORG: "false", META_SHARE_TEAMS: "a", META_ALWAYS_ON: "true"}
    on = Skill(name="s", instructions="x", metadata=dict(metadata))
    assert SkillSharing.of(on).always_on is True
    # The frontmatter round trip keeps it.
    assert parse_skill_text(dump_skill(on), "s").metadata[META_ALWAYS_ON] == "true"
    # Only an org-wide or team skill keeps it.
    SkillSharing(org=False, emails=("a@b.co",), always_on=True).apply(metadata)
    assert META_ALWAYS_ON not in metadata
    assert SkillSharing.of(Skill(name="s", instructions="x", metadata={META_ALWAYS_ON: "false"})).always_on is False
    assert normalize_emails(["A@B.co", "a@b.co", " "]) == ("a@b.co",)


# ── artifacts ────────────────────────────────────────────────────────────────


def test_team_audience_reaches_anton_as_owner_only_and_survives_republish():
    from cowork.services.publish import _team_audience, _with_team_audience

    access = {"mode": "restricted", "emails": [], "org_allowed": False, "team_ids": [TEAM_ENG], "org_wide": False}
    assert _team_audience(access, None) == [TEAM_ENG]
    assert _with_team_audience(access, [TEAM_ENG]) == {
        "mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True,
    }
    # org_wide is org_allowed under auth's new name: either one sets it.
    assert _with_team_audience({"mode": "restricted", "emails": [], "org_wide": True}, [])["org_allowed"] is True
    previous = {"mode": "restricted", "team_ids": [TEAM_OPS]}
    assert _team_audience({"mode": "restricted", "emails": ["a@x.com"]}, previous) == [TEAM_OPS]
    assert _with_team_audience({"mode": "public"}, [TEAM_ENG]) == {"mode": "public"}


def test_the_team_audience_is_written_to_auth():
    from cowork.services.artifact_access import sync_artifact_team_access

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(200, json={"ok": True})

    settings = SimpleNamespace(auth_internal_base_url="https://auth.test", auth_internal_secret="s")
    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    sync_artifact_team_access(
        artifact_id="artifact/1", scope=scope, owner_user_id=ALICE, workspace_id=None,
        allowed_emails=["bob@x.com"], org_allowed=False, team_ids=[TEAM_ENG],
        access_version=3, settings=settings, client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert seen == [{
        "url": "https://auth.test/v1/internal/artifact-access/",
        "body": {
            "artifact_id": "artifact/1", "owner_keycloak_id": ALICE, "organization_id": ORG_ID,
            "workspace_id": None, "allowed_emails": ["bob@x.com"], "org_allowed": False,
            "team_ids": [TEAM_ENG], "org_wide": False, "access_version": 3,
        },
    }]


def test_the_access_read_returns_teams_and_org_wide(tmp_path):
    from cowork.services.artifacts import _published_access_for
    from cowork.services.publish import published_artifact_access

    folder = tmp_path / "artifacts" / "rep"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}))
    (folder / ".published.json").write_text(json.dumps({"index.html": {
        "report_id": "r1", "url": "u", "mode": "restricted", "emails": [], "org_allowed": True,
        "owner_only": False, "team_ids": [TEAM_ENG], "access_version": 2,
    }}))
    read = _published_access_for(folder, folder / "index.html")
    assert (read["allowedTeamIds"], read["orgWide"]) == ([TEAM_ENG], True)
    again = published_artifact_access(folder, artifacts_base=folder.parent)
    assert (again["team_ids"], again["org_allowed"]) == ([TEAM_ENG], True)


def test_setting_artifact_access_checks_the_callers_teams(monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace as aw

    hub_teams.reset_caches_for_tests()

    async def get_auth_json(path, bearer):
        return {"results": [{"id": TEAM_ENG}]}

    monkeypatch.setattr(hub_teams, "get_auth_json", get_auth_json)
    session = SimpleNamespace(scope=TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE))
    request = SimpleNamespace(headers={"X-MindsHub-Authorization": "Bearer t"}, state=SimpleNamespace(principal=_principal(ALICE)))
    ok = asyncio.run(aw._checked_team_audience(
        {"mode": "restricted", "team_ids": [f" {TEAM_ENG} "], "org_wide": True}, request=request, session=session,
    ))
    assert ok == {"mode": "restricted", "team_ids": [TEAM_ENG], "org_wide": True}
    with pytest.raises(HTTPException) as caught:
        asyncio.run(aw._checked_team_audience({"mode": "restricted", "team_ids": [TEAM_OPS]}, request=request, session=session))
    assert caught.value.status_code == 400
    admin_request = SimpleNamespace(headers=request.headers, state=SimpleNamespace(principal=_principal(ADMIN, admin=True)))
    assert asyncio.run(aw._checked_team_audience(
        {"mode": "restricted", "team_ids": [TEAM_OPS]}, request=admin_request, session=session,
    ))["team_ids"] == [TEAM_OPS]
    # The org's default team is everyone: dropped from the teams, org-wide.
    hub_teams.reset_caches_for_tests()

    async def with_default(path, bearer):
        return {"results": [{"id": TEAM_ENG}, {"id": "team-all", "is_default": True}]}

    monkeypatch.setattr(hub_teams, "get_auth_json", with_default)
    for caller in (request, admin_request):
        assert asyncio.run(aw._checked_team_audience(
            {"mode": "restricted", "team_ids": ["team-all", TEAM_ENG]}, request=caller, session=session,
        )) == {"mode": "restricted", "team_ids": [TEAM_ENG], "org_wide": True}
    assert asyncio.run(aw._checked_team_audience(
        {"mode": "restricted", "team_ids": ["team-all"], "org_wide": False}, request=request, session=session,
    )) == {"mode": "restricted", "team_ids": [], "org_wide": True}
    local = SimpleNamespace(scope=LOCAL_SCOPE)
    assert asyncio.run(aw._checked_team_audience(
        {"mode": "restricted", "team_ids": [TEAM_OPS], "org_wide": True}, request=None, session=local,
    )) == {"mode": "restricted"}
    hub_teams.reset_caches_for_tests()


# ── the in-process harness ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_harness_puts_the_always_on_section_in_the_suffix(monkeypatch, tmp_path):
    """The in-process turn appends render_always_on_skills's section to its
    suffix, ahead of the project context; a local turn has none."""
    from cowork.harnesses.anton_harness import harness
    from tests._turns import built_turn

    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=LOCAL_SCOPE) as turn:
        assert "Organization and team skills" not in turn.config.system_prompt_context.suffix

    seen: list[dict] = []

    def always_on(**kwargs):
        seen.append(kwargs)
        return [{"name": "house-style", "body": "Use the house style."}]

    monkeypatch.setattr(harness, "always_on_prompt_skills", always_on)
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=LOCAL_SCOPE) as turn:
        suffix = turn.config.system_prompt_context.suffix
    section = HEADER + "\n\n### house-style\nUse the house style."
    assert section in suffix
    assert seen[0]["viewer"] is None and seen[0]["project_name"] == turn.project.name
