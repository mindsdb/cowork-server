"""Organization branding (ENG-4020): a pass-through to auth, which owns the
org look, and the Brand guidelines skill generated from auth's answer.

Handlers are called directly, the suite's convention for org routes. auth is
stubbed at the two transports org_branding uses (get_mindshub_reply and
send_mindshub), which record the credential and body each call carried.
"""

from __future__ import annotations

import asyncio
import copy
from functools import partial
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import org_branding as ep
from cowork.api.v1.endpoints import settings as settings_ep
from cowork.api.v1.endpoints import skill_sharing as sharing_ep
from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.units import run_efs
from cowork.models.skill import Skill
from cowork.principal import ORG_MANAGE_ROLE, Principal
from cowork.schemas.org_branding import APPEARANCE_FIELDS, OrgAppearanceUpdate
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

def _session_jwt(subject: str) -> str:
    """A credential shaped like a Keycloak session JWT (cowork.principal.
    is_session_jwt): only auth verifies it, and these tests stub auth."""
    import base64
    import json

    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'RS256', 'typ': 'JWT'})}.{part({'sub': subject})}.c2lnbmF0dXJl"


ORG_ID = "33333333-3333-4333-8333-333333333333"
ALICE = "a1111111-1111-4111-8111-111111111111"
ADMIN = "e5555555-5555-4555-8555-555555555555"
MEMBER_BEARER = _session_jwt("member")
# A second caller's credential, so the per-caller cache is told apart.
OTHER_BEARER = _session_jwt("other")
# The credential the desktop app sends with its hub calls.
HUB_BEARER = _session_jwt("hub")
# Credentials that aren't a console session: auth takes no write from them.
API_KEY = "mdb_" + "0" * 32
TURN_KEY = "turnkey-" + "0" * 32
AUTH = "https://auth.acme.com/v1/organizations/" + ORG_ID + "/branding"
STOCK_APPEARANCE = {
    "accent": None, "background_light": None, "background_dark": None, "font": None, "radius": None,
    "skin": None, "default_theme": None, "greeting": None, "sidebar_title_mode": "company",
    "sidebar_title": None, "members_can_choose": True, "guidelines": None,
}
BRANDING = {
    "icon_url": f"{AUTH}/icon?v=abc",
    "logo_url": f"{AUTH}/logo?v=def",
    "logo_dark_url": None,
    "updated_at": "2026-11-04T00:00:00Z",
    "org_name": "Acme",
    "sidebar_title": "Acme",
    "can_edit": False,
    "appearance": dict(STOCK_APPEARANCE),
}
NO_BRANDING = {
    **BRANDING, "icon_url": None, "logo_url": None, "updated_at": None, "appearance": dict(STOCK_APPEARANCE),
}


class FakeAuth:
    """auth's branding routes: GET answers ``branding`` (or ``get_answer``),
    PATCH merges its body into the appearance as auth would, or answers
    ``patch_reply``. Every call is recorded with its credential."""

    def __init__(self) -> None:
        self.branding = copy.deepcopy(BRANDING)
        self.get_answer: MindsHubReply | None = None
        self.patch_reply: MindsHubReply | None = None
        self.calls: list[tuple[str, str, str, object]] = []

    async def get(self, path, bearer):
        self.calls.append(("GET", path, bearer, None))
        if self.get_answer is not None:
            return self.get_answer
        return MindsHubReply(MindsHubAnswer.OK, payload=copy.deepcopy(self.branding), status_code=200)

    async def send(self, method, path, bearer, *, json_body=None):
        self.calls.append((method, path, bearer, json_body))
        if self.patch_reply is not None:
            return self.patch_reply
        appearance = self.branding["appearance"]
        appearance.update(json_body or {})
        if appearance.get("sidebar_title_mode") == "custom":
            self.branding["sidebar_title"] = appearance["sidebar_title"]
        elif appearance.get("sidebar_title_mode") == "none":
            self.branding["sidebar_title"] = None
        return MindsHubReply(MindsHubAnswer.OK, payload=copy.deepcopy(self.branding), status_code=200)

    def gets(self) -> list[tuple[str, str, str, object]]:
        return [call for call in self.calls if call[0] == "GET"]


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


def _request(bearer: str | None = MEMBER_BEARER, *, hub: str | None = None) -> SimpleNamespace:
    headers = {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if hub:
        headers["X-MindsHub-Authorization"] = f"Bearer {hub}"
    return SimpleNamespace(headers=headers, state=SimpleNamespace())


def _stores(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)


@pytest.fixture
def auth(monkeypatch):
    fake = FakeAuth()
    monkeypatch.setattr(org_branding, "get_mindshub_reply", fake.get)
    monkeypatch.setattr(org_branding, "send_mindshub", fake.send)
    org_branding._branding_cache.clear()
    yield fake
    org_branding._branding_cache.clear()


def _engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def org(org_deployment, monkeypatch, tmp_path, auth):
    """Hosted org mode: ``as_(user)`` is that member's request session."""
    _stores(monkeypatch, tmp_path)
    get_app_settings.cache_clear()
    engine = _engine()
    sessions: list[ScopedSession] = []

    def as_(user: str) -> ScopedSession:
        session = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id=ORG_ID, user_id=user))
        sessions.append(session)
        return session

    try:
        yield SimpleNamespace(as_=as_, auth=auth)
    finally:
        for session in sessions:
            session.close()
        get_app_settings.cache_clear()


@pytest.fixture
def desktop(monkeypatch, tmp_path, auth):
    """The desktop's local server: one unpartitioned store, no principal."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    _stores(monkeypatch, tmp_path)
    get_app_settings.cache_clear()
    engine = _engine()
    sessions: list[ScopedSession | Session] = []

    def session() -> ScopedSession:
        scoped = ScopedSession(Session(engine), LOCAL_SCOPE)
        sessions.append(scoped)
        return scoped

    def raw() -> Session:
        plain = Session(engine)
        sessions.append(plain)
        return plain

    try:
        yield SimpleNamespace(session=session, raw=raw, auth=auth)
    finally:
        for scoped in sessions:
            scoped.close()
        get_app_settings.cache_clear()


def _as_unit(route, scoped: ScopedSession):
    """Await ``route(scope)`` with its file store unit (run_efs_unit) run on
    the file store's threads against this test's session."""

    async def unit(fn, *, scope, deadline=True):
        return await run_efs(partial(fn, scoped), scope=scope, deadline=deadline)

    with mock.patch.object(ep, "run_efs_unit", unit):
        return asyncio.run(route(scoped.scope))


def _get(org, user: str, request=None) -> dict:
    return _as_unit(partial(ep.get_org_branding, request or _request()), org.as_(user))


def _put(org, user: str, request=None, **body) -> dict:
    return _as_unit(
        partial(ep.put_org_appearance, OrgAppearanceUpdate(**body), request or _request()), org.as_(user)
    )


def _desktop_get(desktop, request) -> dict:
    return _as_unit(partial(ep.get_org_branding, request), desktop.session())


def _desktop_put(desktop, request, **body) -> dict:
    return _as_unit(partial(ep.put_org_appearance, OrgAppearanceUpdate(**body), request), desktop.session())


def _skill_dir(scope):
    return SkillService(scope).root / BRAND_GUIDELINES_SLUG


def _skill_md(scope) -> str:
    return (_skill_dir(scope) / "SKILL.md").read_text()


def _org_scope(org):
    return org.as_(ALICE).scope


def _refused(call, status: int, detail: str | None = None) -> None:
    with pytest.raises(HTTPException) as caught:
        call()
    assert caught.value.status_code == status
    if detail is not None:
        assert detail in str(caught.value.detail)


def _set_look(auth: FakeAuth, **appearance) -> None:
    auth.branding["appearance"].update(appearance)
    org_branding._branding_cache.clear()


# ── the read, hosted ─────────────────────────────────────────────────────────


def test_a_member_reads_auths_branding_in_the_apps_keys(org):
    org.auth.branding["appearance"].update(
        accent="#0055ff", background_light="#fafafa", background_dark="#0b0b10", font="mono", radius=8,
        skin="custom", default_theme="dark", greeting="Hi", members_can_choose=False, guidelines="Plain words.",
    )
    org.auth.branding["can_edit"] = True
    assert _get(org, ALICE) == {
        "enabled": True,
        "orgName": "Acme",
        "sidebarTitle": "Acme",
        "iconUrl": BRANDING["icon_url"],
        "logoUrl": BRANDING["logo_url"],
        "logoDarkUrl": None,
        "appearance": {
            "accent": "#0055ff", "background": "#fafafa", "backgroundLight": "#fafafa",
            "backgroundDark": "#0b0b10", "font": "mono", "radius": 8, "skin": "custom",
            "defaultTheme": "dark", "greeting": "Hi", "sidebarTitle": None, "sidebarTitleMode": "company",
            "membersCanChoose": False, "guidelines": "Plain words.",
        },
        # auth's can_edit, not the manage-organization role.
        "canEdit": True,
    }


def test_can_edit_is_auths_answer_whatever_the_role(org):
    org.auth.branding["can_edit"] = False
    out = _get(org, ADMIN)
    assert out["canEdit"] is False


def test_hosted_reads_send_the_gateway_credential_never_the_hub_header(org):
    _get(org, ALICE, _request(MEMBER_BEARER, hub=HUB_BEARER))
    assert {bearer for _m, _p, bearer, _b in org.auth.calls} == {MEMBER_BEARER}
    assert org.auth.calls[0][1] == "/organizations/current/branding/"


def test_no_org_for_the_caller_is_enabled_false(org):
    org.auth.get_answer = MindsHubReply(MindsHubAnswer.REFUSED, status_code=403)
    out = _get(org, ALICE)
    assert out == {
        "enabled": False, "orgName": None, "sidebarTitle": None, "iconUrl": None, "logoUrl": None,
        "logoDarkUrl": None, "appearance": None, "canEdit": False,
    }


def test_auth_unreachable_is_the_stock_look_and_still_answers(org):
    org.auth.get_answer = MindsHubReply(MindsHubAnswer.UNREACHABLE)
    out = _get(org, ALICE)
    assert (out["enabled"], out["iconUrl"], out["sidebarTitle"], out["canEdit"]) == (True, None, "MindsHub", False)
    assert out["appearance"]["sidebarTitleMode"] == "company"


def test_the_read_is_cached_per_caller(org):
    _get(org, ALICE)
    _get(org, ALICE)
    assert len(org.auth.gets()) == 1
    _get(org, ADMIN, _request(OTHER_BEARER))
    assert [bearer for _m, _p, bearer, _b in org.auth.gets()] == [MEMBER_BEARER, OTHER_BEARER]


# ── the write ────────────────────────────────────────────────────────────────


def test_a_put_is_passed_to_auths_patch_as_sent(org):
    out = _put(org, ADMIN, accent="#0055ff", sidebar_title_mode="custom", sidebar_title="Acme AI", radius=8)
    method, path, bearer, body = org.auth.calls[-1]
    assert (method, path, bearer) == ("PATCH", "/organizations/current/branding/appearance/", MEMBER_BEARER)
    assert body == {"accent": "#0055ff", "sidebar_title_mode": "custom", "sidebar_title": "Acme AI", "radius": 8}
    assert (out["sidebarTitle"], out["appearance"]["accent"]) == ("Acme AI", "#0055ff")


def test_the_put_takes_camel_case_and_an_explicit_null(org):
    _put(org, ADMIN, request=None, **{"backgroundDark": "#0b0b10", "greeting": None})
    assert org.auth.calls[-1][3] == {"background_dark": "#0b0b10", "greeting": None}


def test_an_older_clients_single_background_sets_both(org):
    _put(org, ADMIN, background="#123456")
    assert org.auth.calls[-1][3] == {"background_light": "#123456", "background_dark": "#123456"}
    _put(org, ADMIN, background="#123456", background_dark="#000000")
    assert org.auth.calls[-1][3] == {"background_dark": "#000000"}


def test_the_callers_own_next_read_sees_their_change_without_waiting(org):
    _get(org, ADMIN)
    _put(org, ADMIN, accent="#ff5500")
    gets_before = len(org.auth.gets())
    assert _get(org, ADMIN)["appearance"]["accent"] == "#ff5500"
    assert len(org.auth.gets()) == gets_before


@pytest.mark.parametrize(
    ("auth_status", "route_status"), [(403, 403), (422, 422), (400, 422)],
)
def test_auths_refusals_keep_their_status_and_message(org, auth_status, route_status):
    message = {403: "You don't have organization.branding.manage.", 422: "accent: not a hex color.",
               400: "radius: must be 0 to 16."}[auth_status]
    org.auth.patch_reply = MindsHubReply(
        MindsHubAnswer.REFUSED, payload={"detail": message}, status_code=auth_status
    )
    _refused(lambda: _put(org, ALICE, accent="#0055ff"), route_status, message)


def test_only_auths_message_is_passed_on_from_a_refusal(org):
    org.auth.patch_reply = MindsHubReply(
        MindsHubAnswer.REFUSED,
        payload={"accent": ["Not a brand color."], "internal_trace": ["db host 10.0.0.5"], "stack": "boom"},
        status_code=400,
    )
    with pytest.raises(HTTPException) as caught:
        _put(org, ADMIN, accent="#0055ff")
    assert (caught.value.status_code, caught.value.detail) == (422, "accent: Not a brand color.")


def test_auth_unreachable_on_put_is_a_502(org, caplog):
    org.auth.patch_reply = MindsHubReply(MindsHubAnswer.UNREACHABLE, status_code=503)
    with caplog.at_level("WARNING", logger=ep.logger.name):
        _refused(lambda: _put(org, ADMIN, accent="#0055ff"), 502, "Couldn't reach MindsHub")
    assert "org_branding_appearance_save_failed" in [getattr(r, "event", None) for r in caplog.records]


def test_a_patch_that_answers_less_than_the_whole_body_is_read_again(org):
    org.auth.patch_reply = MindsHubReply(MindsHubAnswer.OK, payload={"accent": "#0055ff"}, status_code=200)
    org.auth.branding["appearance"]["accent"] = "#0055ff"
    out = _put(org, ADMIN, accent="#0055ff")
    assert out["appearance"]["accent"] == "#0055ff"
    assert org.auth.calls[-1][0] == "GET"


@pytest.mark.parametrize(
    "body",
    [
        {"evil": 1},
        {"accent": "#0055ff", "org_id": "44444444-4444-4444-8444-444444444444"},
        {"accent": "red"},
        {"background_light": "#12345"},
        {"font": "comic-sans"},
        {"skin": "neon"},
        {"radius": 17},
        {"radius": -1},
        {"radius": "8"},
        {"radius": True},
        {"default_theme": "sepia"},
        {"greeting": "x" * 201},
        {"sidebar_title": "x" * 41},
        {"sidebar_title_mode": "logo"},
        {"sidebar_title_mode": "custom"},
        {"sidebar_title_mode": "custom", "sidebar_title": "   "},
        {"members_can_choose": "yes"},
        {"guidelines": "x" * 8001},
    ],
)
def test_a_body_the_app_cant_apply_is_a_422_and_reaches_no_one(body):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    seen = []
    app = FastAPI()

    @app.put("/appearance")
    def appearance(update: OrgAppearanceUpdate):
        seen.append(update)
        return {}

    response = TestClient(app).put("/appearance", json=body)
    assert response.status_code == 422, body
    assert seen == []


def test_the_outbound_body_is_a_new_dict_of_allowed_fields_only(org):
    update = OrgAppearanceUpdate(
        accent="#0055FF", background="#123456", font="mono", radius=8, skin="custom", default_theme="dark",
        greeting="Hi", sidebarTitleMode="custom", sidebarTitle="Acme AI", members_can_choose=False,
        guidelines="Plain words.",
    )
    _put(org, ADMIN, **update.model_dump(exclude_unset=True))
    body = org.auth.calls[-1][3]
    assert set(body) <= set(APPEARANCE_FIELDS)
    assert "background" not in body
    assert body == {
        "accent": "#0055ff", "background_light": "#123456", "background_dark": "#123456", "font": "mono",
        "radius": 8, "skin": "custom", "default_theme": "dark", "greeting": "Hi",
        "sidebar_title_mode": "custom", "sidebar_title": "Acme AI", "members_can_choose": False,
        "guidelines": "Plain words.",
    }
    assert update.auth_changes() is not update.auth_changes()


def test_the_patch_goes_to_the_configured_auth_host_and_a_constant_path(org, monkeypatch):
    from cowork.services import mindshub_client

    sent = []

    async def capture(method, path, bearer, *, json_body=None):
        sent.append((method, f"{mindshub_client.mindshub_v1()}{path}", bearer, json_body))
        return MindsHubReply(MindsHubAnswer.OK, payload=copy.deepcopy(org.auth.branding), status_code=200)

    monkeypatch.setattr(org_branding, "send_mindshub", capture)
    monkeypatch.setattr(mindshub_client, "mindshub_v1", lambda: "https://auth.acme.com/v1")
    request = _request(MEMBER_BEARER, hub=HUB_BEARER)
    request.url = SimpleNamespace(path="/api/v1/org-branding/appearance?next=https://evil.example")
    _put(org, ADMIN, request=request, accent="#0055ff")
    assert sent == [(
        "PATCH", "https://auth.acme.com/v1/organizations/current/branding/appearance/", MEMBER_BEARER,
        {"accent": "#0055ff"},
    )]


# ── the Brand guidelines skill, hosted ───────────────────────────────────────


def test_hosted_the_skill_is_org_wide_and_always_on(org):
    _get(org, ALICE)
    row = access_row(org.as_(ALICE), BRAND_GUIDELINES_SLUG)
    assert (row.org_allowed, row.always_on, row.owner_user_id) == (True, True, None)
    text = _skill_md(_org_scope(org))
    assert "name: brand-guidelines" in text and "Acme brand guidelines" in text
    assert "managed_by: org-branding" in text
    # The fixed public URLs, without ?v, so a new logo needs no regeneration.
    assert f"- Logo (for light backgrounds): {AUTH}/logo\n" in text and "?v=" not in text
    turn = build_turn_skills(
        org.as_(ALICE).scope, project_name="general", viewer=_viewer(ALICE), session=org.as_(ALICE)
    )
    assert turn[BRAND_GUIDELINES_SLUG].always_on is True


def test_hosted_the_skill_follows_auths_look_and_stays_compact(org):
    _set_look(
        org.auth, accent="#0055ff", background_light="#fafafa", background_dark="#0b0b10", font="standard",
        radius=8, default_theme="dark", guidelines="Say 'customers', never 'users'.",
    )
    _get(org, ALICE)
    text = _skill_md(_org_scope(org))
    for expected in (
        "#0055ff", "Light background `#fafafa`", "Dark background `#0b0b10`", "Text on it: `#111111`",
        "Text on it: `#ffffff`", "8px", "dark look", "Say 'customers', never 'users'.",
    ):
        assert expected in text
    assert len(text) < 5500
    _put(org, ADMIN, accent="#ff5500")
    text = _skill_md(_org_scope(org))
    assert "#ff5500" in text and "#0055ff" not in text


def test_hosted_the_skill_names_a_custom_title_only_in_custom_mode(org):
    _put(org, ADMIN, accent="#0055ff", sidebar_title_mode="custom", sidebar_title="Acme AI")
    assert "appears as **Acme AI**" in _skill_md(_org_scope(org))
    _put(org, ADMIN, sidebar_title_mode="none")
    text = _skill_md(_org_scope(org))
    assert "appears as" not in text and "Acme brand guidelines" in text


def test_hosted_long_guidelines_spill_into_a_file_beside_the_skill(org):
    guidelines = "\n".join(f"Rule {n}: keep it short and clear, always." for n in range(180))
    _set_look(org.auth, accent="#0055ff", guidelines=guidelines)
    _get(org, ALICE)
    skill_dir = _skill_dir(_org_scope(org))
    body = _skill_md(_org_scope(org)).split("---\n", 2)[2]
    assert len(body) <= org_branding.BRAND_SKILL_BODY_BUDGET + 100
    assert (skill_dir / "guidelines.md").read_text() == guidelines
    _set_look(org.auth, guidelines="Short now.")
    _get(org, ALICE)
    assert not (skill_dir / "guidelines.md").exists()


def test_hosted_no_branding_removes_the_skill(org):
    _get(org, ALICE)
    org.auth.branding = copy.deepcopy(NO_BRANDING)
    org_branding._branding_cache.clear()
    _get(org, ALICE)
    assert not _skill_dir(_org_scope(org)).exists()
    assert access_row(org.as_(ALICE), BRAND_GUIDELINES_SLUG) is None


def test_hosted_a_members_refused_or_unknown_read_leaves_the_org_skill(org):
    _get(org, ALICE)
    before = _skill_md(_org_scope(org))
    for answer in (MindsHubReply(MindsHubAnswer.REFUSED, status_code=403), MindsHubReply(MindsHubAnswer.UNREACHABLE)):
        org.auth.get_answer = answer
        org_branding._branding_cache.clear()
        _get(org, ALICE)
        assert _skill_md(_org_scope(org)) == before
    assert access_row(org.as_(ALICE), BRAND_GUIDELINES_SLUG) is not None


def test_hosted_an_unchanged_skill_is_not_rewritten(org):
    _get(org, ALICE)
    path = _skill_dir(_org_scope(org)) / "SKILL.md"
    stamp = path.stat().st_mtime_ns
    org_branding._branding_cache.clear()
    _get(org, ALICE)
    assert path.stat().st_mtime_ns == stamp


def test_hosted_nobody_edits_renames_deletes_or_shares_the_skill_as_a_skill(org):
    _get(org, ALICE)
    admin = _viewer(ADMIN, admin=True)
    _refused(
        lambda: skills_ep._update_skill(
            org.as_(ADMIN), skill_id=BRAND_GUIDELINES_SLUG, body=SkillUpdateRequest(instructions="Hijack."),
            principal=_principal(ADMIN, admin=True), viewer=admin,
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )
    _refused(
        lambda: skills_ep._delete_skill(
            org.as_(ADMIN), skill_id=BRAND_GUIDELINES_SLUG, principal=_principal(ADMIN, admin=True), viewer=admin
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )
    _refused(
        lambda: sharing_ep._set_org_skill_sharing(
            BRAND_GUIDELINES_SLUG, SkillSharingRequest(always_on=False),
            org.as_(ADMIN), _principal(ADMIN, admin=True), admin, frozenset(),
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )
    listed = {
        s["id"]: s for s in skills_ep._list_skills(
            org.as_(ADMIN), principal=_principal(ADMIN, admin=True), viewer=admin
        )["skills"]
    }
    skill = listed[BRAND_GUIDELINES_SLUG]
    assert skill["capabilities"] == {"canEdit": False, "canDelete": False, "canDisable": False, "canShare": False}
    assert (skill["access"]["alwaysOn"], skill["name"]) == (True, "Brand guidelines")


def test_hosted_the_name_stays_reserved_with_no_branding(org):
    org.auth.branding = copy.deepcopy(NO_BRANDING)
    _get(org, ALICE)
    _refused(
        lambda: skills_ep._create_skill(
            org.as_(ALICE), body=SkillCreateRequest(label=BRAND_GUIDELINES_SLUG, instructions="Mine."),
            principal=_principal(ALICE), viewer=_viewer(ALICE),
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )


def test_hosted_a_member_turns_the_skill_off_for_themselves_only(org):
    _get(org, ALICE)
    out = sharing_ep._set_skill_enabled_for_me(
        org.as_(ALICE), skill_id=BRAND_GUIDELINES_SLUG, body=SkillEnabledForMeRequest(enabled=False),
        principal=_principal(ALICE), viewer=_viewer(ALICE),
    )
    assert out["enabledForMe"] is False
    assert BRAND_GUIDELINES_SLUG not in build_turn_skills(
        org.as_(ALICE).scope, project_name="general", viewer=_viewer(ALICE), session=org.as_(ALICE)
    )
    assert build_turn_skills(
        org.as_(ADMIN).scope, project_name="general", viewer=_viewer(ADMIN, admin=True), session=org.as_(ADMIN)
    )[BRAND_GUIDELINES_SLUG].always_on is True


# ── the desktop's local server ───────────────────────────────────────────────


def test_desktop_reads_with_the_hub_credential_and_ignores_authorization(desktop):
    out = _desktop_get(desktop, _request("this-servers-own-token", hub=HUB_BEARER))
    assert (out["enabled"], out["orgName"], out["logoUrl"]) == (True, "Acme", BRANDING["logo_url"])
    assert {bearer for _m, _p, bearer, _b in desktop.auth.calls} == {HUB_BEARER}


def test_desktop_without_a_hub_credential_is_enabled_false_and_asks_nothing(desktop):
    out = _desktop_get(desktop, _request("this-servers-own-token"))
    assert out["enabled"] is False
    assert desktop.auth.calls == []


def test_desktop_put_passes_through_with_the_hub_credential(desktop):
    out = _desktop_put(desktop, _request(hub=HUB_BEARER), accent="#0055ff")
    assert out["appearance"]["accent"] == "#0055ff"
    assert desktop.auth.calls[-1][:3] == ("PATCH", "/organizations/current/branding/appearance/", HUB_BEARER)


def test_desktop_put_without_a_hub_credential_asks_to_sign_in(desktop):
    _refused(
        lambda: _desktop_put(desktop, _request(), accent="#0055ff"),
        403, ep.SIGN_IN_TO_EDIT,
    )
    assert desktop.auth.calls == []


@pytest.mark.parametrize("credential", [API_KEY, TURN_KEY])
def test_hosted_a_put_with_an_api_or_turn_key_asks_to_sign_in_and_reaches_no_one(org, credential):
    _refused(lambda: _put(org, ADMIN, request=_request(credential), accent="#0055ff"), 403, ep.SIGN_IN_TO_EDIT)
    assert [call for call in org.auth.calls if call[0] == "PATCH"] == []


@pytest.mark.parametrize("credential", [API_KEY, TURN_KEY])
def test_desktop_a_put_with_an_api_or_turn_key_asks_to_sign_in_and_reaches_no_one(desktop, credential):
    _refused(
        lambda: _desktop_put(desktop, _request(hub=credential), accent="#0055ff"),
        403, ep.SIGN_IN_TO_EDIT,
    )
    assert desktop.auth.calls == []


def test_auths_401_is_a_403_asking_to_sign_in_never_a_401(org):
    org.auth.patch_reply = MindsHubReply(
        MindsHubAnswer.REFUSED, payload={"detail": "API keys can't change branding."}, status_code=401
    )
    _refused(lambda: _put(org, ADMIN, accent="#0055ff"), 403, ep.SIGN_IN_TO_EDIT)


def test_desktop_the_skill_is_a_local_on_demand_skill(desktop):
    _desktop_get(desktop, _request(hub=HUB_BEARER))
    text = _skill_md(LOCAL_SCOPE)
    assert "Acme brand guidelines" in text and "managed_by: org-branding" in text
    # No org rows on the desktop: nothing makes it always on.
    assert access_row(desktop.session(), BRAND_GUIDELINES_SLUG) is None
    turn = build_turn_skills(LOCAL_SCOPE, project_name="general", session=desktop.session())
    assert turn[BRAND_GUIDELINES_SLUG].always_on is False


def test_desktop_the_generated_skill_cant_be_edited_or_deleted(desktop):
    _desktop_get(desktop, _request(hub=HUB_BEARER))
    _refused(
        lambda: skills_ep._update_skill(
            desktop.session(), skill_id=BRAND_GUIDELINES_SLUG, body=SkillUpdateRequest(instructions="Hijack."),
            principal=None, viewer=None,
        ),
        403, BRAND_GUIDELINES_REFUSAL,
    )
    _refused(lambda: skills_ep._delete_skill(desktop.session(), skill_id=BRAND_GUIDELINES_SLUG, principal=None, viewer=None), 403,
             BRAND_GUIDELINES_REFUSAL)
    assert "Acme brand guidelines" in _skill_md(LOCAL_SCOPE)


def test_desktop_the_skill_is_regenerated_when_auths_look_changes(desktop):
    _desktop_get(desktop, _request(hub=HUB_BEARER))
    _set_look(desktop.auth, accent="#0055ff")
    _desktop_get(desktop, _request(hub=HUB_BEARER))
    assert "#0055ff" in _skill_md(LOCAL_SCOPE)


@pytest.mark.parametrize("how", ["signed out", "personal account", "no branding"])
def test_desktop_the_skill_is_removed_without_an_org_or_branding(desktop, how):
    _desktop_get(desktop, _request(hub=HUB_BEARER))
    assert _skill_dir(LOCAL_SCOPE).exists()
    org_branding._branding_cache.clear()
    if how == "signed out":
        _desktop_get(desktop, _request())
    elif how == "personal account":
        desktop.auth.get_answer = MindsHubReply(MindsHubAnswer.REFUSED, status_code=403)
        _desktop_get(desktop, _request(hub=HUB_BEARER))
    else:
        desktop.auth.branding = copy.deepcopy(NO_BRANDING)
        _desktop_get(desktop, _request(hub=HUB_BEARER))
    assert not _skill_dir(LOCAL_SCOPE).exists()


def test_desktop_sign_out_removes_the_skill(desktop):
    _desktop_get(desktop, _request(hub=HUB_BEARER))
    settings_ep.logout_clear_credentials(desktop.raw(), LOCAL_SCOPE)
    assert not _skill_dir(LOCAL_SCOPE).exists()


def test_desktop_a_persons_own_skill_of_that_name_is_left_alone(desktop):
    from anton.core.tools.skill_format import dump_skill

    skill_dir = _skill_dir(LOCAL_SCOPE)
    skill_dir.mkdir(parents=True)
    mine = Skill(name=BRAND_GUIDELINES_SLUG, description="Mine", instructions="My own rules.")
    (skill_dir / "SKILL.md").write_text(dump_skill(mine))
    _desktop_get(desktop, _request(hub=HUB_BEARER))
    assert "My own rules." in _skill_md(LOCAL_SCOPE)
    _desktop_get(desktop, _request())
    assert "My own rules." in _skill_md(LOCAL_SCOPE)
    # Not the generated skill, so it stays the person's to change.
    assert SkillService(LOCAL_SCOPE).is_brand_guidelines(BRAND_GUIDELINES_SLUG) is False


# ── helpers ──────────────────────────────────────────────────────────────────


def test_fixed_image_url_drops_the_version():
    assert org_branding.fixed_image_url(f"{AUTH}/logo?v=abc") == f"{AUTH}/logo"
    assert org_branding.fixed_image_url(None) is None


def test_an_org_with_no_name_still_reads_well():
    branding = org_branding.parse_auth_branding({**BRANDING, "org_name": None})
    description, body, extra = org_branding.render_brand_skill(branding)
    assert description.startswith("The organization's brand")
    assert body.startswith("# Brand guidelines\n")
    assert extra is None


def test_auths_json_routes_end_with_a_slash():
    assert org_branding.AUTH_BRANDING_PATH == "/organizations/current/branding/"
    assert org_branding.AUTH_APPEARANCE_PATH == "/organizations/current/branding/appearance/"


def test_send_mindshub_keeps_a_refusals_body_and_sends_the_json(monkeypatch):
    import httpx

    from cowork.services import mindshub_client

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers["Authorization"], request.content))
        if b"bad" in request.content:
            return httpx.Response(422, json={"detail": "accent: not a hex color."})
        return httpx.Response(200, json={"ok": True})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        mindshub_client.httpx, "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(mindshub_client, "mindshub_v1", lambda: "https://auth.acme.com/v1")

    refused = asyncio.run(
        mindshub_client.send_mindshub("PATCH", "/x/", MEMBER_BEARER, json_body={"accent": "bad"})
    )
    assert (refused.answer, refused.status_code, refused.payload) == (
        MindsHubAnswer.REFUSED, 422, {"detail": "accent: not a hex color."},
    )
    ok = asyncio.run(mindshub_client.send_mindshub("PATCH", "/x/", MEMBER_BEARER, json_body={"accent": "#000000"}))
    assert (ok.answer, ok.payload) == (MindsHubAnswer.OK, {"ok": True})
    assert seen[0][:3] == ("PATCH", "/v1/x/", f"Bearer {MEMBER_BEARER}")
