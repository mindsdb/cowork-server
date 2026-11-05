"""Organization branding (ENG-4020): the org's look, its images from auth,
and its Brand guidelines skill.

Handlers are called directly, the suite's convention for org routes. Auth is
stubbed at get_mindshub_reply.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import org_branding as ep
from cowork.api.v1.endpoints import skill_sharing as sharing_ep
from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.principal import ORG_MANAGE_ROLE, Principal
from cowork.schemas.org_branding import GUIDELINES_MAX_CHARS, OrgAppearanceUpdate
from cowork.schemas.skills import (
    SkillCreateRequest,
    SkillEnabledForMeRequest,
    SkillSharingRequest,
    SkillUpdateRequest,
)
from cowork.services import org_branding
from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply
from cowork.services.skill_access import access_row
from cowork.services.skill_sharing import SkillViewer
from cowork.services.skills import (
    BRAND_GUIDELINES_REFUSAL,
    BRAND_GUIDELINES_SLUG,
    SkillService,
    build_turn_skills,
)

ORG_ID = "33333333-3333-4333-8333-333333333333"
ALICE = "a1111111-1111-4111-8111-111111111111"
ADMIN = "e5555555-5555-4555-8555-555555555555"
AUTH = "https://auth.acme.com/v1/organizations/" + ORG_ID + "/branding"
BRANDING = {
    "icon_url": f"{AUTH}/icon?v=abc",
    "logo_url": f"{AUTH}/logo?v=def",
    "logo_dark_url": None,
    "updated_at": "2026-11-04T00:00:00Z",
}
CURRENT_ORG = {"id": ORG_ID, "name": "acme", "display_name": "Acme"}
NO_IMAGES = {"icon_url": None, "logo_url": None, "logo_dark_url": None, "updated_at": None}


def _principal(user: str, *, admin: bool = False) -> Principal:
    roles = frozenset({ORG_MANAGE_ROLE}) if admin else frozenset()
    return Principal(user_id=user, org_id=ORG_ID, email=f"{user[:4]}@acme.com", roles=roles)


def _viewer(user: str, *, admin: bool = False) -> SkillViewer:
    return SkillViewer(
        user_id=user,
        email=f"{user[:4]}@acme.com",
        team_ids=frozenset(),
        teams_known=True,
        is_admin=admin,
        can_share_skills_org=admin,
        can_write_skills_org=admin,
        can_share_artifacts_org=admin,
        permissions_known=True,
    )


def _request(bearer: str = "member-session") -> SimpleNamespace:
    return SimpleNamespace(headers={"Authorization": f"Bearer {bearer}"}, state=SimpleNamespace())


@pytest.fixture
def org(org_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    org_branding._branding_cache.clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    sessions: list[ScopedSession] = []
    answers = {"branding": BRANDING, "org": CURRENT_ORG}

    async def reply(path, bearer):
        payload = answers["branding"] if path == org_branding.AUTH_BRANDING_PATH else answers["org"]
        if payload is None:
            return MindsHubReply(MindsHubAnswer.UNREACHABLE)
        return MindsHubReply(MindsHubAnswer.OK, payload=payload, status_code=200)

    monkeypatch.setattr(org_branding, "get_mindshub_reply", reply)

    def as_(user: str) -> ScopedSession:
        session = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id=ORG_ID, user_id=user))
        sessions.append(session)
        return session

    def set_auth(branding=BRANDING, current=CURRENT_ORG):
        answers["branding"], answers["org"] = branding, current
        org_branding._branding_cache.clear()

    try:
        yield SimpleNamespace(as_=as_, set_auth=set_auth)
    finally:
        for session in sessions:
            session.close()
        org_branding._branding_cache.clear()
        get_app_settings.cache_clear()


def _get(org, user: str, *, admin: bool = False) -> dict:
    return asyncio.run(ep.get_org_branding(_request(), org.as_(user), _principal(user, admin=admin)))


def _put(org, user: str, *, admin: bool = True, **body) -> dict:
    return asyncio.run(
        ep.put_org_appearance(
            OrgAppearanceUpdate(**body), _request(), org.as_(user), _principal(user, admin=admin)
        )
    )


def _skill_md(org) -> str:
    return (SkillService(org.as_(ALICE).scope).root / BRAND_GUIDELINES_SLUG / "SKILL.md").read_text()


def _refused(call, status: int, detail: str | None = None) -> None:
    with pytest.raises(HTTPException) as caught:
        call()
    assert caught.value.status_code == status
    if detail is not None:
        assert detail in caught.value.detail


# ── the read ─────────────────────────────────────────────────────────────────


def test_outside_org_mode_branding_is_off(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    try:
        scoped = SimpleNamespace(scope=LOCAL_SCOPE)
        out = asyncio.run(ep.get_org_branding(_request(), scoped, None))
    finally:
        get_app_settings.cache_clear()
    assert out == {
        "enabled": False, "iconUrl": None, "logoUrl": None, "logoDarkUrl": None,
        "appearance": None, "canEdit": False,
    }


def test_a_member_reads_the_images_from_auth_and_the_stock_look(org):
    out = _get(org, ALICE)
    assert out == {
        "enabled": True,
        "iconUrl": BRANDING["icon_url"],
        "logoUrl": BRANDING["logo_url"],
        "logoDarkUrl": None,
        "appearance": {
            "accent": None, "background": None, "font": None, "radius": None, "skin": None,
            "defaultTheme": None, "greeting": None, "membersCanChoose": True, "guidelines": None,
        },
        "canEdit": False,
    }
    assert _get(org, ADMIN, admin=True)["canEdit"] is True


def test_auth_unreachable_is_null_urls_and_still_answers(org):
    org.set_auth(branding=None)
    out = _get(org, ALICE)
    assert (out["enabled"], out["iconUrl"], out["logoUrl"]) == (True, None, None)


def test_the_auth_read_is_cached_per_caller(org, monkeypatch):
    calls = []
    original = org_branding.get_mindshub_reply

    async def counted(path, bearer):
        calls.append((path, bearer))
        return await original(path, bearer)

    monkeypatch.setattr(org_branding, "get_mindshub_reply", counted)
    _get(org, ALICE)
    _get(org, ALICE)
    assert len(calls) == 2  # branding and the org's name, once
    asyncio.run(ep.get_org_branding(_request("other-session"), org.as_(ADMIN), _principal(ADMIN)))
    assert len(calls) == 4
    assert {bearer for _path, bearer in calls} == {"member-session", "other-session"}


# ── the write ────────────────────────────────────────────────────────────────


def test_an_admin_sets_the_look_and_a_partial_change_keeps_the_rest(org):
    out = _put(
        org, ADMIN, accent="#1A2B3C", background="#ffffff", font="mono", radius=10, skin="custom",
        default_theme="light", greeting="Welcome to Acme", members_can_choose=False,
        guidelines="Write in plain English.",
    )
    assert out["appearance"] == {
        "accent": "#1a2b3c", "background": "#ffffff", "font": "mono", "radius": 10, "skin": "custom",
        "defaultTheme": "light", "greeting": "Welcome to Acme", "membersCanChoose": False,
        "guidelines": "Write in plain English.",
    }
    out = _put(org, ADMIN, greeting=None, members_can_choose=None)
    assert out["appearance"]["accent"] == "#1a2b3c"
    assert out["appearance"]["greeting"] is None
    assert out["appearance"]["membersCanChoose"] is True
    assert _get(org, ALICE)["appearance"]["font"] == "mono"


def test_a_member_cannot_change_the_look(org):
    _refused(lambda: _put(org, ALICE, admin=False, accent="#123456"), 403)
    assert _get(org, ALICE)["appearance"]["accent"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"accent": "red"},
        {"accent": "#12345"},
        {"background": "#gggggg"},
        {"font": "comic-sans"},
        {"skin": "neon"},
        {"radius": -1},
        {"radius": 17},
        {"default_theme": "sepia"},
        {"greeting": "x" * 201},
        {"guidelines": "x" * (GUIDELINES_MAX_CHARS + 1)},
    ],
)
def test_values_the_app_cannot_apply_are_refused(body):
    with pytest.raises(ValidationError):
        OrgAppearanceUpdate(**body)


def test_the_body_accepts_camel_case_too():
    update = OrgAppearanceUpdate(defaultTheme="dark", membersCanChoose=False)
    assert (update.default_theme, update.members_can_choose) == ("dark", False)


# ── the Brand guidelines skill ───────────────────────────────────────────────


def test_no_branding_means_no_skill_and_its_name_stays_reserved(org):
    org.set_auth(branding=NO_IMAGES)
    _get(org, ALICE)
    assert not (SkillService(org.as_(ALICE).scope).root / BRAND_GUIDELINES_SLUG).exists()
    assert access_row(org.as_(ALICE), BRAND_GUIDELINES_SLUG) is None
    _refused(
        lambda: skills_ep.create_skill(
            SkillCreateRequest(label=BRAND_GUIDELINES_SLUG, instructions="Mine."),
            org.as_(ALICE), _principal(ALICE), _viewer(ALICE),
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )
    assert not (SkillService(org.as_(ALICE).scope).root / BRAND_GUIDELINES_SLUG).exists()


def test_images_make_an_always_on_skill_shared_with_everyone(org):
    _get(org, ALICE)
    row = access_row(org.as_(ALICE), BRAND_GUIDELINES_SLUG)
    assert (row.org_allowed, row.always_on, row.owner_user_id) == (True, True, None)
    text = _skill_md(org)
    assert text.startswith("---\n")
    assert "name: brand-guidelines" in text
    assert "Acme brand guidelines" in text
    # The fixed public URLs, without ?v, so a new logo needs no regeneration.
    assert f"- Logo (for light backgrounds): {AUTH}/logo\n" in text
    assert f"{AUTH}/icon" in text
    assert "?v=" not in text
    turn = build_turn_skills(
        org.as_(ALICE).scope, project_name="general", viewer=_viewer(ALICE), session=org.as_(ALICE)
    )
    assert turn[BRAND_GUIDELINES_SLUG].always_on is True
    assert list(turn)[0] == BRAND_GUIDELINES_SLUG


def test_the_skill_follows_the_look_and_stays_compact(org):
    _put(
        org, ADMIN, accent="#0055ff", background="#0b0b10", font="standard", radius=8,
        default_theme="dark", guidelines="Say 'customers', never 'users'.",
    )
    text = _skill_md(org)
    for expected in ("#0055ff", "#0b0b10", "8px", "dark look", "Say 'customers', never 'users'."):
        assert expected in text
    assert len(text) < 5500
    _put(org, ADMIN, accent="#ff5500")
    text = _skill_md(org)
    assert "#ff5500" in text and "#0055ff" not in text


def test_long_guidelines_spill_into_a_file_beside_the_skill(org):
    guidelines = "\n".join(f"Rule {n}: keep it short and clear, always." for n in range(180))
    assert len(guidelines) <= GUIDELINES_MAX_CHARS
    _put(org, ADMIN, accent="#0055ff", guidelines=guidelines)
    skill_dir = SkillService(org.as_(ALICE).scope).root / BRAND_GUIDELINES_SLUG
    body = _skill_md(org).split("---\n", 2)[2]
    assert len(body) <= org_branding.BRAND_SKILL_BODY_BUDGET + 100
    assert "guidelines.md" in body
    assert (skill_dir / "guidelines.md").read_text() == guidelines
    _put(org, ADMIN, guidelines="Short now.")
    assert not (skill_dir / "guidelines.md").exists()


def test_removing_all_branding_removes_the_skill(org):
    _put(org, ADMIN, accent="#0055ff")
    assert access_row(org.as_(ALICE), BRAND_GUIDELINES_SLUG) is not None
    org.set_auth(branding=NO_IMAGES)
    _put(org, ADMIN, accent=None)
    assert not (SkillService(org.as_(ALICE).scope).root / BRAND_GUIDELINES_SLUG).exists()
    assert access_row(org.as_(ALICE), BRAND_GUIDELINES_SLUG) is None


def test_unknown_images_leave_the_skill_alone(org):
    _get(org, ALICE)
    before = _skill_md(org)
    org.set_auth(branding=None)
    _get(org, ALICE)
    assert _skill_md(org) == before
    org.set_auth(current=None)
    _put(org, ADMIN, accent="#0055ff")
    assert _skill_md(org) == before


def test_an_unchanged_skill_is_not_rewritten(org):
    _get(org, ALICE)
    path = SkillService(org.as_(ALICE).scope).root / BRAND_GUIDELINES_SLUG / "SKILL.md"
    stamp = path.stat().st_mtime_ns
    org_branding._branding_cache.clear()
    _get(org, ALICE)
    assert path.stat().st_mtime_ns == stamp


def test_nobody_edits_renames_deletes_or_shares_the_skill_as_a_skill(org):
    _get(org, ALICE)
    admin = _viewer(ADMIN, admin=True)
    _refused(
        lambda: skills_ep.update_skill(
            BRAND_GUIDELINES_SLUG, SkillUpdateRequest(instructions="Hijack."),
            org.as_(ADMIN), _principal(ADMIN, admin=True), admin,
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )
    _refused(
        lambda: skills_ep.update_skill(
            BRAND_GUIDELINES_SLUG, SkillUpdateRequest(label="mine"),
            org.as_(ADMIN), _principal(ADMIN, admin=True), admin,
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )
    _refused(
        lambda: skills_ep.delete_skill(
            BRAND_GUIDELINES_SLUG, org.as_(ADMIN), _principal(ADMIN, admin=True), admin
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )
    _refused(
        lambda: sharing_ep.set_skill_sharing(
            BRAND_GUIDELINES_SLUG, SkillSharingRequest(always_on=False),
            org.as_(ADMIN), _principal(ADMIN, admin=True), admin, frozenset(),
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )
    # A create under its name is a 409 while it exists.
    _refused(
        lambda: skills_ep.create_skill(
            SkillCreateRequest(label=BRAND_GUIDELINES_SLUG, instructions="Mine."),
            org.as_(ALICE), _principal(ALICE), _viewer(ALICE),
        ),
        409,
    )
    assert "Acme brand guidelines" in _skill_md(org)
    listed = {
        s["id"]: s
        for s in skills_ep.list_skills(org.as_(ADMIN), _principal(ADMIN, admin=True), admin)["skills"]
    }
    skill = listed[BRAND_GUIDELINES_SLUG]
    assert skill["canEdit"] is False
    assert skill["capabilities"] == {
        "canEdit": False, "canDelete": False, "canDisable": False, "canShare": False,
    }
    assert skill["access"]["alwaysOn"] is True
    assert skill["name"] == "Brand guidelines"


def test_a_member_turns_the_skill_off_for_themselves_only(org):
    _get(org, ALICE)
    out = sharing_ep.set_skill_enabled_for_me(
        BRAND_GUIDELINES_SLUG, SkillEnabledForMeRequest(enabled=False),
        org.as_(ALICE), _principal(ALICE), _viewer(ALICE),
    )
    assert out["enabledForMe"] is False
    alice_turn = build_turn_skills(
        org.as_(ALICE).scope, project_name="general", viewer=_viewer(ALICE), session=org.as_(ALICE)
    )
    admin_turn = build_turn_skills(
        org.as_(ADMIN).scope, project_name="general", viewer=_viewer(ADMIN, admin=True),
        session=org.as_(ADMIN),
    )
    assert BRAND_GUIDELINES_SLUG not in alice_turn
    assert admin_turn[BRAND_GUIDELINES_SLUG].always_on is True


def test_a_skill_of_that_name_outside_org_mode_is_an_ordinary_skill():
    service = SkillService(LOCAL_SCOPE)
    assert service.is_brand_guidelines(BRAND_GUIDELINES_SLUG) is False


def test_an_org_with_no_name_still_reads_well():
    images = org_branding.AuthBranding(logo_url=f"{AUTH}/logo?v=1", reachable=True, org_known=True)
    description, body, extra = org_branding.render_brand_skill(images, None)
    assert description.startswith("The organization's brand")
    assert body.startswith("# Brand guidelines\n")
    assert "Use the organization's brand" in body
    assert extra is None


def test_auths_json_route_ends_with_a_slash():
    assert org_branding.AUTH_BRANDING_PATH == "/organizations/current/branding/"


def test_fixed_image_url_drops_the_version():
    assert org_branding.fixed_image_url(f"{AUTH}/logo?v=abc") == f"{AUTH}/logo"
    assert org_branding.fixed_image_url(None) is None
