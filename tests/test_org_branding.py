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
from tests._org_sharing import MEMBER_BEARER
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


# A second caller's credential, so the per-caller cache is told apart.
OTHER_BEARER = f"{MEMBER_BEARER}-2"


def _request(bearer: str = MEMBER_BEARER) -> SimpleNamespace:
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


@pytest.fixture
def migrated_org(org, monkeypatch, tmp_path):
    """``org``, but over a SQLite file the real Alembic migrations built,
    as a deployed database is, rather than over SQLModel's create_all.
    ``build(drop_sidebar_title=True)`` gives the schema a deployed
    environment had after running d4b7e2a9c1f3 before it gained the column;
    ``migrate()`` then runs every pending revision."""
    import sqlite3

    from alembic import command
    from sqlalchemy import create_engine as create_sa_engine

    from cowork.db.migrations import _alembic_config, run_schema_migrations

    path = tmp_path / "migrated.db"
    uri = f"sqlite:///{path}"
    engine = create_sa_engine(uri)
    sessions: list[ScopedSession] = []

    def build(*, drop_sidebar_title: bool = False, migrate: bool = True) -> None:
        config = _alembic_config(uri)
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "d4b7e2a9c1f3")
        if drop_sidebar_title:
            with sqlite3.connect(path) as connection:
                connection.execute("alter table org_appearance drop column sidebar_title")
                connection.commit()
        if migrate:
            run_schema_migrations(engine, uri)

    def as_(user: str) -> ScopedSession:
        session = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id=ORG_ID, user_id=user))
        sessions.append(session)
        return session

    try:
        yield SimpleNamespace(as_=as_, build=build, set_auth=org.set_auth)
    finally:
        for session in sessions:
            session.close()
        engine.dispose()


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
        "enabled": False, "orgName": None, "sidebarTitle": None,
        "iconUrl": None, "logoUrl": None, "logoDarkUrl": None,
        "appearance": None, "canEdit": False,
    }


def test_a_member_reads_the_images_from_auth_and_the_stock_look(org):
    out = _get(org, ALICE)
    assert out == {
        "enabled": True,
        "orgName": "Acme",
        "sidebarTitle": "Acme",
        "iconUrl": BRANDING["icon_url"],
        "logoUrl": BRANDING["logo_url"],
        "logoDarkUrl": None,
        "appearance": {
            "accent": None, "background": None, "backgroundLight": None, "backgroundDark": None,
            "font": None, "radius": None, "skin": None,
            "defaultTheme": None, "greeting": None, "sidebarTitle": None,
            "sidebarTitleMode": "company", "membersCanChoose": True, "guidelines": None,
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
    asyncio.run(ep.get_org_branding(_request(OTHER_BEARER), org.as_(ADMIN), _principal(ADMIN)))
    assert len(calls) == 4
    assert {bearer for _path, bearer in calls} == {MEMBER_BEARER, OTHER_BEARER}


# ── the write ────────────────────────────────────────────────────────────────


def test_an_admin_sets_the_look_and_a_partial_change_keeps_the_rest(org):
    out = _put(
        org, ADMIN, accent="#1A2B3C", background="#ffffff", font="mono", radius=10, skin="custom",
        default_theme="light", greeting="Welcome to Acme", sidebar_title="Acme Assist",
        members_can_choose=False, guidelines="Write in plain English.",
    )
    assert out["appearance"] == {
        "accent": "#1a2b3c", "background": "#ffffff", "backgroundLight": "#ffffff",
        "backgroundDark": "#ffffff", "font": "mono", "radius": 10, "skin": "custom",
        "defaultTheme": "light", "greeting": "Welcome to Acme", "sidebarTitle": "Acme Assist",
        "sidebarTitleMode": "custom", "membersCanChoose": False, "guidelines": "Write in plain English.",
    }
    assert out["sidebarTitle"] == "Acme Assist"
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
        {"background_light": "white"},
        {"background_dark": "#12345z"},
        {"font": "comic-sans"},
        {"skin": "neon"},
        {"radius": -1},
        {"radius": 17},
        {"default_theme": "sepia"},
        {"greeting": "x" * 201},
        {"sidebar_title": "x" * 41},
        {"guidelines": "x" * (GUIDELINES_MAX_CHARS + 1)},
    ],
)
def test_values_the_app_cannot_apply_are_refused(body):
    with pytest.raises(ValidationError):
        OrgAppearanceUpdate(**body)


# ── light and dark backgrounds ───────────────────────────────────────────────


def test_light_and_dark_backgrounds_are_set_apart(org):
    out = _put(org, ADMIN, background_light="#FAFAFA", background_dark="#0B0B10")
    appearance = out["appearance"]
    assert (appearance["backgroundLight"], appearance["backgroundDark"], appearance["background"]) == (
        "#fafafa", "#0b0b10", "#fafafa",
    )
    # One side alone keeps the other.
    out = _put(org, ADMIN, background_dark="#101820")
    assert (out["appearance"]["backgroundLight"], out["appearance"]["backgroundDark"]) == ("#fafafa", "#101820")
    out = _put(org, ADMIN, background_light=None)
    appearance = out["appearance"]
    assert (appearance["backgroundLight"], appearance["backgroundDark"], appearance["background"]) == (
        None, "#101820", "#101820",
    )
    assert _get(org, ALICE)["appearance"]["backgroundDark"] == "#101820"


def test_an_older_clients_single_background_sets_both(org):
    out = _put(org, ADMIN, background="#123456")
    appearance = out["appearance"]
    assert (appearance["backgroundLight"], appearance["backgroundDark"], appearance["background"]) == (
        "#123456", "#123456", "#123456",
    )


def test_the_legacy_column_is_mirrored_and_read_when_the_new_ones_are_empty(org):
    _put(org, ADMIN, background_light="#fafafa", background_dark="#0b0b10")
    row = org_branding.appearance_row(org.as_(ADMIN))
    assert row.background == "#fafafa"
    # A row only a server from before the split wrote.
    session = org.as_(ADMIN)
    row = org_branding.appearance_row(session)
    row.background_light = row.background_dark = None
    row.background = "#222222"
    session.add(row)
    session.commit()
    appearance = _get(org, ALICE)["appearance"]
    assert (appearance["backgroundLight"], appearance["backgroundDark"], appearance["background"]) == (
        "#222222", "#222222", "#222222",
    )


def test_the_backgrounds_take_camel_case():
    update = OrgAppearanceUpdate(backgroundLight="#FFFFFF", backgroundDark="#000000")
    assert (update.background_light, update.background_dark) == ("#ffffff", "#000000")


def test_the_skill_lists_both_backgrounds_with_readable_text(org):
    _put(org, ADMIN, accent="#0055ff", background_light="#fafafa", background_dark="#0b0b10")
    text = _skill_md(org)
    assert "Light background `#fafafa`" in text and "Text on it: `#111111`" in text
    assert "Dark background `#0b0b10`" in text and "Text on it: `#ffffff`" in text
    _put(org, ADMIN, background_light=None)
    text = _skill_md(org)
    assert "Light background" not in text and "Dark background `#0b0b10`" in text
    # One color for both reads as one background.
    _put(org, ADMIN, background="#fafafa")
    text = _skill_md(org)
    assert "- Background `#fafafa`" in text and "Dark background" not in text


def test_a_low_contrast_accent_is_flagged_for_the_background_it_fails_on(org):
    _put(org, ADMIN, accent="#fbfbfb", background_light="#ffffff", background_dark="#000000")
    assert "low-contrast on the light background" in _skill_md(org)


def test_the_body_accepts_camel_case_too():
    update = OrgAppearanceUpdate(defaultTheme="dark", membersCanChoose=False, sidebarTitle="Acme AI")
    assert (update.default_theme, update.members_can_choose, update.sidebar_title) == (
        "dark", False, "Acme AI",
    )


# ── the sidebar title ────────────────────────────────────────────────────────


def test_the_sidebar_title_is_the_admins_then_a_short_org_name_then_mindshub(org):
    assert _get(org, ALICE)["sidebarTitle"] == "Acme"
    org.set_auth(current={**CURRENT_ORG, "display_name": "Acme Widgets International Holdings"})
    out = _get(org, ALICE)
    assert (out["orgName"], out["sidebarTitle"]) == ("Acme Widgets International Holdings", "MindsHub")
    org.set_auth(current={**CURRENT_ORG, "display_name": "x" * 24})
    assert _get(org, ALICE)["sidebarTitle"] == "x" * 24
    _put(org, ADMIN, sidebar_title="Acme AI")
    out = _get(org, ALICE)
    assert (out["sidebarTitle"], out["appearance"]["sidebarTitle"]) == ("Acme AI", "Acme AI")
    _put(org, ADMIN, sidebar_title="  ")
    assert _get(org, ALICE)["appearance"]["sidebarTitle"] is None


def test_the_sidebar_title_has_three_modes(org):
    out = _put(org, ADMIN, sidebar_title_mode="custom", sidebar_title="Acme AI")
    assert (out["sidebarTitle"], out["appearance"]["sidebarTitleMode"], out["appearance"]["sidebarTitle"]) == (
        "Acme AI", "custom", "Acme AI",
    )
    # No title, only the logo: the resolved title is null and the text is cleared.
    out = _put(org, ADMIN, sidebar_title_mode="none")
    assert (out["sidebarTitle"], out["appearance"]["sidebarTitleMode"], out["appearance"]["sidebarTitle"]) == (
        None, "none", None,
    )
    assert _get(org, ALICE)["sidebarTitle"] is None
    out = _put(org, ADMIN, sidebar_title_mode="company", sidebar_title="Ignored")
    assert (out["sidebarTitle"], out["appearance"]["sidebarTitleMode"], out["appearance"]["sidebarTitle"]) == (
        "Acme", "company", None,
    )
    org.set_auth(current={**CURRENT_ORG, "display_name": "Acme Widgets International Holdings"})
    assert _get(org, ALICE)["sidebarTitle"] == "MindsHub"


def test_a_change_without_the_mode_keeps_the_old_meaning(org):
    out = _put(org, ADMIN, sidebar_title="Acme AI")
    assert (out["sidebarTitle"], out["appearance"]["sidebarTitleMode"]) == ("Acme AI", "custom")
    out = _put(org, ADMIN, sidebar_title=None)
    assert (out["sidebarTitle"], out["appearance"]["sidebarTitleMode"]) == ("Acme", "company")
    _put(org, ADMIN, sidebar_title_mode="none")
    # A change to another field keeps the mode.
    out = _put(org, ADMIN, accent="#0055ff")
    assert (out["sidebarTitle"], out["appearance"]["sidebarTitleMode"]) == (None, "none")


@pytest.mark.parametrize("title", [None, "", "   "])
def test_a_custom_title_needs_text(title):
    body = {"sidebar_title_mode": "custom"}
    if title is not None:
        body["sidebar_title"] = title
    with pytest.raises(ValidationError) as caught:
        OrgAppearanceUpdate(**body)
    assert "custom sidebar title needs sidebar_title" in str(caught.value)


@pytest.mark.parametrize("body", [{"sidebar_title_mode": "logo"}, {"sidebar_title_mode": "custom", "sidebar_title": "x" * 41}])
def test_a_bad_mode_or_a_long_custom_title_is_refused(body):
    with pytest.raises(ValidationError):
        OrgAppearanceUpdate(**body)


def test_the_mode_accepts_camel_case():
    update = OrgAppearanceUpdate(sidebarTitleMode="custom", sidebarTitle="Acme AI")
    assert (update.sidebar_title_mode, update.sidebar_title) == ("custom", "Acme AI")


def test_the_skill_names_a_custom_title_only_in_custom_mode(org):
    _put(org, ADMIN, accent="#0055ff", sidebar_title_mode="custom", sidebar_title="Acme AI")
    assert "appears as **Acme AI**" in _skill_md(org)
    _put(org, ADMIN, sidebar_title_mode="none")
    text = _skill_md(org)
    assert "appears as" not in text and "Acme brand guidelines" in text


def test_an_unknown_org_name_falls_back_to_mindshub(org):
    org.set_auth(current=None)
    out = _get(org, ALICE)
    assert (out["orgName"], out["sidebarTitle"], out["logoUrl"]) == (None, "MindsHub", BRANDING["logo_url"])


def test_the_org_name_still_names_the_sidebar_when_the_images_are_unknown(org):
    org.set_auth(branding=None)
    out = _get(org, ALICE)
    assert (out["orgName"], out["sidebarTitle"], out["logoUrl"]) == ("Acme", "Acme", None)


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


def test_the_skill_names_the_admins_sidebar_title(org):
    _put(org, ADMIN, accent="#0055ff", sidebar_title="Acme AI")
    assert "appears as **Acme AI**" in _skill_md(org)
    _put(org, ADMIN, sidebar_title=None)
    assert "appears as" not in _skill_md(org)


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


# ── against the migrated schema ──────────────────────────────────────────────


@pytest.mark.parametrize("drop_sidebar_title", [False, True], ids=["fresh", "deployed-before-sidebar-title"])
def test_a_put_round_trips_on_the_migrated_schema(migrated_org, drop_sidebar_title):
    migrated_org.build(drop_sidebar_title=drop_sidebar_title)
    out = _put(migrated_org, ADMIN, accent="#0055ff", sidebar_title="Acme AI", members_can_choose=False)
    assert (out["sidebarTitle"], out["appearance"]["accent"]) == ("Acme AI", "#0055ff")
    read = _get(migrated_org, ALICE)
    assert read["appearance"] == {
        **out["appearance"], "accent": "#0055ff", "sidebarTitle": "Acme AI", "membersCanChoose": False,
    }
    assert read["sidebarTitle"] == "Acme AI"


def test_a_database_error_on_put_is_a_generic_500_and_a_named_log(migrated_org, caplog):
    # The deployed schema before 3e78d31dc845 ran: no sidebar_title column.
    migrated_org.build(drop_sidebar_title=True, migrate=False)
    with caplog.at_level("ERROR", logger=ep.logger.name):
        _refused(lambda: _put(migrated_org, ADMIN, accent="#0055ff"), 500, ep.SAVE_FAILED)
    with pytest.raises(HTTPException) as caught:
        _put(migrated_org, ADMIN, accent="#0055ff")
    assert caught.value.detail == ep.SAVE_FAILED
    assert "sidebar_title" not in caught.value.detail
    events = [getattr(record, "event", None) for record in caplog.records]
    assert "org_branding_appearance_save_failed" in events


def test_a_database_error_on_get_is_a_generic_500(migrated_org, caplog):
    migrated_org.build(drop_sidebar_title=True, migrate=False)
    with caplog.at_level("ERROR", logger=ep.logger.name):
        _refused(lambda: _get(migrated_org, ALICE), 500, ep.READ_FAILED)
    events = [getattr(record, "event", None) for record in caplog.records]
    assert "org_branding_appearance_read_failed" in events
