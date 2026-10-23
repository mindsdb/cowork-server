"""POST /api/v1/artifacts/drafts/{project}/{artifact}/link and /health's draft_links."""
from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.api.v1.route_walker import undeclared_routes
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.structured_logging import make_formatter
from cowork.db.scoped import TenantScope
from cowork.draft_host import app as draft_app_module
from cowork.draft_host.app import create_draft_app
from cowork.draft_host.token import DRAFT_TTL_SECONDS, verify_draft_token
from cowork.server import create_app

from draft_host_support import (
    ARTIFACT,
    DRAFT_HOST,
    FETCH,
    KEYS,
    ORG,
    PROJECT,
    USER,
    configure_draft_env,
    fake_review,
    make_artifact,
)

IDENTITY = {"X-User-Id": USER, "X-Organization-Id": ORG}
DASHED_ARTIFACT = str(UUID(ARTIFACT))
LINK = f"/api/v1/artifacts/drafts/{PROJECT}/{DASHED_ARTIFACT}/link"


@pytest.fixture
def draft_env(monkeypatch):
    configure_draft_env(monkeypatch)
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    """Wire both routes to one real artifact folder; returns (seen, set_artifact)."""
    seen: dict = {}
    folders = iter(range(100))

    def set_artifact(**kwargs):
        source, folder, metadata = make_artifact(tmp_path / f"artifact-{next(folders)}", **kwargs)
        review = fake_review(source, folder, metadata, seen)
        monkeypatch.setattr(workspace_ep, "review_artifact_for_request", review)
        monkeypatch.setattr(draft_app_module, "review_artifact_for_request", review)

    set_artifact()
    return seen, set_artifact


@pytest.fixture
def api(draft_env):
    return TestClient(create_app())


def _token_of(url: str) -> str:
    return url.split("/_t/", 1)[1].split("/", 1)[0]


def test_issues_a_signed_link_to_the_primary_file(api, reviewed):
    seen, _ = reviewed

    response = api.post(LINK, headers=IDENTITY)

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["url"].startswith(f"https://{DRAFT_HOST}/_t/v1.")
    assert body["url"].endswith("/index.html")
    claims = verify_draft_token(_token_of(body["url"]), env="staging", keys=KEYS)
    assert (claims["sub"], claims["org"], claims["proj"], claims["art"]) == (USER, ORG, PROJECT, ARTIFACT)
    assert body["expiresAt"] == claims["exp"]
    assert DRAFT_TTL_SECONDS == 1800
    assert seen["scope"] == TenantScope(org_mode=True, org_id=ORG, user_id=USER)


def test_an_issued_link_is_counted_but_never_logged(api, reviewed, caplog):
    with caplog.at_level("WARNING", logger="cowork.api.v1.endpoints.artifact_workspace"):
        url = api.post(LINK, headers=IDENTITY).json()["url"]

    [record] = [r for r in caplog.records if getattr(r, "event", None) == "draft_link_issued"]
    line = make_formatter().format(record)
    assert json.loads(line)["message"] == "Draft link issued"
    assert _token_of(url) not in line


def test_every_call_signs_a_new_link_from_the_current_time(api, reviewed, monkeypatch):
    clock = iter([1_900_000_000, 1_900_000_600])
    # Only this module's clock: the global time.time also feeds log records.
    monkeypatch.setattr(workspace_ep, "time", SimpleNamespace(time=lambda: next(clock)))

    first = api.post(LINK, headers=IDENTITY).json()
    second = api.post(LINK, headers=IDENTITY).json()

    assert first["url"] != second["url"]
    assert (first["expiresAt"], second["expiresAt"]) == (1_900_001_800, 1_900_002_400)


def test_the_link_opens_on_the_draft_host(api, reviewed):
    url = api.post(LINK, headers=IDENTITY).json()["url"]
    draft = TestClient(create_draft_app(), base_url=f"https://{DRAFT_HOST}")

    response = draft.get(url.removeprefix(f"https://{DRAFT_HOST}"), headers=FETCH)

    assert response.status_code == 200
    assert "draft body" in response.text


def test_a_fullstack_link_points_into_its_public_subtree(api, reviewed):
    _, set_artifact = reviewed
    set_artifact(
        metadata={"type": "fullstack-app", "primary": "static/index.html"},
        files={"static/index.html": "<html><body>app</body></html>", "backend.py": "SECRET = 1"},
    )

    response = api.post(LINK, headers=IDENTITY)

    assert response.status_code == 200, response.text
    assert response.json()["url"].endswith("/static/index.html")


def test_a_draft_the_caller_may_not_review_is_404(api, monkeypatch):
    def refuse(*_args):
        raise HTTPException(status_code=404, detail="Artifact not found")

    monkeypatch.setattr(workspace_ep, "review_artifact_for_request", refuse)

    assert api.post(LINK, headers=IDENTITY).status_code == 404


def test_the_same_path_checks_as_the_draft_route(api, reviewed):
    assert api.post(f"/api/v1/artifacts/drafts/not-a-uuid/{DASHED_ARTIFACT}/link", headers=IDENTITY).status_code == 400
    assert api.post(f"/api/v1/artifacts/drafts/{PROJECT}/not-a-uuid/link", headers=IDENTITY).status_code == 422


def test_no_identity_is_401(api, reviewed):
    assert api.post(LINK).status_code == 401


def test_no_link_while_draft_links_are_off(monkeypatch, reviewed):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_DRAFT_HOSTS", raising=False)
    get_app_settings.cache_clear()
    try:
        response = TestClient(create_app()).post(LINK, headers=IDENTITY)
    finally:
        get_app_settings.cache_clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "Draft links are not available"


def test_the_route_declares_a_permission(draft_env):
    # create_app() raises at boot on an undeclared route; say it here too.
    assert undeclared_routes(create_app()) == []


def test_health_says_the_deployment_serves_draft_links(api):
    # The web client reads this once at boot; every way into a preview (the
    # Artifacts page, the rail, a chat card) then opens drafts by link.
    assert api.get("/api/v1/health/").json()["draft_links"] is True


@pytest.mark.parametrize(
    "name, value",
    [
        ("COWORK_TENANCY_MODE", "local"),
        ("COWORK_DRAFT_HOSTS", ""),
        ("COWORK_DRAFT_LINK_KEYS", ""),
        ("COWORK_DRAFT_PORT", "0"),
    ],
    ids=["local-tenancy", "no-hosts", "no-key", "no-port"],
)
def test_health_says_no_draft_links_while_any_value_is_missing(draft_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    get_app_settings.cache_clear()

    assert TestClient(create_app()).get("/api/v1/health/").json()["draft_links"] is False

