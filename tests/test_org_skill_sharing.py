"""Hosted org skill sharing (ENG-3787): who sees, uses and changes a skill.

In hosted org tenancy every member reads one skill store, and each skill
records who it is shared with in its SKILL.md metadata: only its owner (the
default for a new skill), people by email, teams, or the whole organization.
A skill with no sharing metadata predates sharing and reads as org-wide.
Org-wide and team skills are always on in a turn of the people they reach.

Handlers are called directly with an explicit SkillViewer, the suite's
convention for org routes (cowork.server wires the identity middleware only
when the process starts in org mode). Auth is stubbed at get_auth_json.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from anton.core.tools.skill_format import dump_skill
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import hub_teams as hub_teams_ep
from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.skill import (
    META_CREATED_BY,
    META_CREATED_BY_EMAIL,
    META_SHARE_EMAILS,
    META_SHARE_ORG,
    META_SHARE_TEAMS,
    Skill,
)
from cowork.principal import ORG_MANAGE_ROLE, Principal
from cowork.schemas.skills import SkillCreateRequest, SkillSharingRequest, SkillUpdateRequest
from cowork.services import hub_teams
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


def _share(ctx, user: str, slug: str, viewer: SkillViewer | None = None, *, admin=False, **body) -> dict:
    return skills_ep.set_skill_sharing(
        slug, SkillSharingRequest(**body), ctx.as_(user), _principal(user, admin=admin),
        viewer or _viewer(user, teams=(TEAM_ENG,), admin=admin),
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
    assert created["sharing"] == {"org": False, "teamIds": [], "emails": []}
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
        metadata={META_SHARE_ORG: "true", META_CREATED_BY: DAVE, META_SHARE_TEAMS: TEAM_OPS},
    )
    uploaded = skills_ep._upload_skill_bytes(
        dump_skill(skill).encode(), "SKILL.md", org.as_(ALICE), _principal(ALICE), viewer=_viewer(ALICE),
    )
    assert (uploaded["section"], uploaded["sharing"]["org"], uploaded["sharing"]["teamIds"]) == ("mine", False, [])
    assert SkillService(org.as_(ALICE).scope).get_skill("uploaded").created_by == ALICE


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

    assert out["sharing"] == {"org": False, "teamIds": [TEAM_ENG], "emails": ["carol@x.com", "bob@x.com"]}
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


def test_an_org_mode_shared_only_body_changes_nothing(org):
    """The personal-partition body, {"shared": bool}, reaches the org route as
    an audience with nothing set, which keeps what the skill has."""
    _create(org, ALICE, "shared")
    out = _share(org, ALICE, "shared", shared=True)
    assert out["sharing"] == {"org": False, "teamIds": [], "emails": []}


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


# ── turns ────────────────────────────────────────────────────────────────────


def _turn_fixture(org):
    """Skills of every kind, as Bob (team eng, bob@x.com) meets them."""
    alice = org.as_(ALICE).scope
    _legacy_skill(alice, "zeta-legacy", "Zeta body.")
    _create(org, ALICE, "alpha-org")
    _share(org, ALICE, "alpha-org", _viewer(ALICE, share_org=True), org=True)
    _create(org, ALICE, "beta-team")
    _share(org, ALICE, "beta-team", team_ids=[TEAM_ENG])
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
    assert always_on == ["alpha-org", "zeta-legacy", "beta-team"]
    assert list(block)[:3] == always_on
    assert {"gamma-email", "bob-own"} <= set(block)
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
        {"name": "zeta-legacy", "body": "Zeta body."},
        {"name": "beta-team", "body": "Body of beta-team."},
    ]
    assert render_always_on_skills(skills) == (
        HEADER
        + "\n\n### alpha-org\nBody of alpha-org."
        + "\n\n### zeta-legacy\nZeta body."
        + "\n\n### beta-team\nBody of beta-team."
    )
    # Without a viewer (no caller credential) only org-wide skills apply.
    assert [s["name"] for s in always_on_prompt_skills(
        scope=org.as_(BOB).scope, project_name="general", viewer=None,
    )] == ["alpha-org", "zeta-legacy"]


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
    assert view.model_dump(by_alias=True) == {"reachable": True, "teams": [{"id": TEAM_ENG, "displayName": "Eng"}]}
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
