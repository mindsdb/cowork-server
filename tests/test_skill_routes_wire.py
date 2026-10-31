"""The JSON the skills routes answer, name for name, value for value and null
for null, pinned to tests/fixtures/skill_routes_wire.json: every route that
answers one skill (create, upload, read, edit, the caller's own switch and a
sharing change) and the list, in single-user, single-tenant and MindsHub mode.

The fixture holds what those routes answer through the whole HTTP stack, so
a change to how a response is built (the response model, its aliases, the
route's serialization) must leave every byte of it as it is. Each run stamps
new times, so an ISO timestamp reads as "<time>"; a packaged built-in's text
changes with the built-in, so its description and instructions read as
"<built-in text>". Everything else, nulls included, is compared exactly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.skill_sharing import SkillViewer

FIXTURE = Path(__file__).parent / "fixtures" / "skill_routes_wire.json"

ORG_ID = "7c1d5e9a-2b4f-4a6e-8d3c-9f0e1a2b3c4d"
OWNER = "a1b2c3d4-0000-4000-8000-00000000a001"
PEER = "a1b2c3d4-0000-4000-8000-00000000a002"
TEAM = "a1b2c3d4-0000-4000-8000-0000000000e1"
OWNER_EMAIL = "owner@example.com"
DEPLOYMENT_TOKEN = "skill-wire-test-token"

SKILL_MD = "---\nname: wire-upload\ndescription: Uploaded for the wire test.\n---\n\nUploaded body.\n"
_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_BUILTIN_TEXT = ("description", "declarative")


def _normalized(value):
    """``value`` with every ISO timestamp as "<time>"; nothing else changes."""
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, str) and _TIME.match(value):
        return "<time>"
    return value


def _listing(body: dict) -> dict:
    """The list with each packaged built-in's text as "<built-in text>", and
    the built-ins in id order. Built-ins have no creation time, so the list
    keeps them in the order the filesystem lists their folders, which differs
    between machines; every other skill keeps the place the route gives it."""
    builtins = iter(sorted((skill for skill in body["skills"] if skill.get("isBuiltin")), key=lambda skill: skill["id"]))
    skills = []
    for skill in body["skills"]:
        if skill.get("isBuiltin"):
            skill = {key: ("<built-in text>" if key in _BUILTIN_TEXT else item) for key, item in next(builtins).items()}
        skills.append(skill)
    return {**body, "skills": skills}


def _answer(response) -> dict:
    assert response.headers["content-type"] == "application/json", response.text
    return {"status": response.status_code, "body": _normalized(response.json())}


def _drive(client: TestClient, headers: dict, *, sharing: list[dict]) -> dict:
    """Each route that answers a skill, in order, and the list last."""
    answers = {
        "create": client.post(
            "/api/v1/skills/", json={"label": "wire-notes", "instructions": "Notes body."}, headers=headers
        ),
        "upload": client.post(
            "/api/v1/skills/upload", files={"file": ("SKILL.md", SKILL_MD.encode(), "text/markdown")},
            headers=headers,
        ),
        "read": client.get("/api/v1/skills/wire-notes", headers=headers),
        "edit": client.put(
            "/api/v1/skills/wire-notes",
            json={"instructions": "Changed body.", "description": "Notes for the wire test."},
            headers=headers,
        ),
        "off_for_me": client.put("/api/v1/skills/wire-notes/me", json={"enabled": False}, headers=headers),
    }
    for number, body in enumerate(sharing, start=1):
        answers[f"sharing_{number}"] = client.put("/api/v1/skills/wire-notes/sharing", json=body, headers=headers)
    out = {name: _answer(response) for name, response in answers.items()}
    listed = client.get("/api/v1/skills/", headers=headers)
    out["list"] = {**_answer(listed), "body": _listing(_normalized(listed.json()))}
    return out


def _app_client() -> TestClient:
    from cowork.server import create_app

    # No `with`: skipping the lifespan skips boot migrations and background workers.
    return TestClient(create_app())


@pytest.fixture
def stores(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _single_user() -> dict:
    return _drive(_app_client(), {}, sharing=[{"shared": True}])


def _single_tenant() -> dict:
    headers = {"Authorization": f"Bearer {DEPLOYMENT_TOKEN}", "X-User-Id": OWNER, "X-Organization-Id": ORG_ID}
    return _drive(_app_client(), headers, sharing=[{"shared": True}, {"shared": False}])


def _mindshub_client() -> TestClient:
    """The app in MindsHub mode, with the caller's MindsHub answers fixed:
    an owner in one team who may share with everyone."""
    from cowork.api.v1.endpoints.skills import get_skill_viewer
    from cowork.services.skill_sharing import get_org_member_lookup

    viewer = SkillViewer(
        user_id=OWNER,
        email=OWNER_EMAIL,
        team_ids=frozenset({TEAM}),
        teams_known=True,
        is_admin=False,
        can_share_skills_org=True,
        can_write_skills_org=False,
        can_share_artifacts_org=False,
        permissions_known=True,
    )

    async def members(asked):
        return frozenset(asked)

    client = _app_client()
    client.app.dependency_overrides[get_skill_viewer] = lambda: viewer
    client.app.dependency_overrides[get_org_member_lookup] = lambda: members
    return client


MINDSHUB_HEADERS = {"X-User-Id": OWNER, "X-User-Email": OWNER_EMAIL, "X-Organization-Id": ORG_ID}


def _mindshub() -> dict:
    return _drive(
        _mindshub_client(),
        MINDSHUB_HEADERS,
        sharing=[
            {"allowedTeamIds": [TEAM], "alwaysOn": True},
            {"allowedUserIds": [PEER]},
            {"orgAllowed": True},
        ],
    )


def _assert_recorded(mode: str, answers: dict) -> None:
    """``answers`` are the fixture's for ``mode``, keys in the same order:
    the response renders its keys in order, so the bytes match too."""
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))[mode]
    assert answers == recorded
    assert json.dumps(answers) == json.dumps(recorded)


def test_single_user_mode_answers_the_recorded_json(stores, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    get_app_settings.cache_clear()
    _assert_recorded("single_user", _single_user())


def test_single_tenant_mode_answers_the_recorded_json(stores, personal_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", DEPLOYMENT_TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    get_app_settings.cache_clear()
    _assert_recorded("single_tenant", _single_tenant())


def test_mindshub_mode_answers_the_recorded_json(stores, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    get_app_settings.cache_clear()
    _assert_recorded("mindshub", _mindshub())


def test_the_sharing_capabilities_name_no_organization(stores, monkeypatch):
    """GET /skills/sharing-capabilities answers what the caller may do and
    whether MindsHub could be asked, and no organization name: the app
    names the organization from its own list of the caller's."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    get_app_settings.cache_clear()
    answer = _mindshub_client().get("/api/v1/skills/sharing-capabilities", headers=MINDSHUB_HEADERS)
    assert (answer.status_code, answer.json()) == (200, {
        "enabled": True, "canShareSkillsOrg": True, "canWriteSkillsOrg": False, "canShareArtifactsOrg": False,
        "permissionsKnown": True, "teamsKnown": True,
    })

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    answer = _app_client().get("/api/v1/skills/sharing-capabilities")
    assert (answer.status_code, answer.json()) == (200, {
        "enabled": False, "canShareSkillsOrg": False, "canWriteSkillsOrg": False, "canShareArtifactsOrg": False,
        "permissionsKnown": True, "teamsKnown": True,
    })
