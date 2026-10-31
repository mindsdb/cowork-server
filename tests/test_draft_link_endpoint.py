"""POST /api/v1/artifacts/drafts/{project}/{artifact}/link and /health's draft_links."""
from __future__ import annotations

import asyncio
import json
import shutil
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlmodel import Session
from starlette.testclient import TestClient

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.api.v1.endpoints import health
from cowork.api.v1.route_walker import undeclared_routes
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.structured_logging import make_formatter
from cowork.db.scoped import TenantScope
from cowork.draft_host import app as draft_app_module
from cowork.draft_host.app import create_draft_app
from cowork.draft_host.token import verify_draft_token
from cowork.server import create_app
from cowork.services.artifact_draft_review import disable_draft_review
from cowork.services.artifact_identity import ensure_full_id

import test_artifacts_api_tenancy as tenancy
from draft_host_support import (
    ARTIFACT,
    DRAFT_HOST,
    FETCH,
    KEYS,
    ORG,
    PROJECT,
    USER,
    draft_env,  # noqa: F401  (fixture)
    events,
    fake_review,
    make_artifact,
    reviewed_and_read,
)
from test_artifacts_api_tenancy import session  # noqa: F401  (fixture)

IDENTITY = {"X-User-Id": USER, "X-Organization-Id": ORG}
DASHED_ARTIFACT = str(UUID(ARTIFACT))
LINK = f"/api/v1/artifacts/drafts/{PROJECT}/{DASHED_ARTIFACT}/link"


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    """Wire both routes to one real artifact folder; returns (seen, set_artifact)."""
    seen: dict = {}
    folders = iter(range(100))

    def set_artifact(**kwargs):
        source, folder, metadata = make_artifact(tmp_path / f"artifact-{next(folders)}", **kwargs)
        review = fake_review(source, folder, metadata, seen)
        monkeypatch.setattr(workspace_ep, "review_artifact_and_read", reviewed_and_read(review))
        monkeypatch.setattr(draft_app_module, "review_artifact_and_read", reviewed_and_read(review))

    set_artifact()
    return seen, set_artifact


@pytest.fixture
def api(draft_env):
    return TestClient(create_app())


def _token_of(url: str) -> str:
    return url.split("/_t/", 1)[1].split("/", 1)[0]


def _on_the_draft_host(url: str) -> str:
    return url.removeprefix(f"https://{DRAFT_HOST}")


def test_the_link_starts_with_the_bare_draft_origin(api, reviewed, monkeypatch):
    monkeypatch.setenv("COWORK_DRAFT_ORIGIN", f"HTTPS://{DRAFT_HOST.upper()}/\n")
    get_app_settings.cache_clear()

    url = api.post(LINK, headers=IDENTITY).json()["url"]

    assert url.startswith(f"https://{DRAFT_HOST}/_t/v1.")


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
    assert seen["scope"] == TenantScope(org_mode=True, org_id=ORG, user_id=USER)


def test_an_issued_link_is_counted_but_never_logged(api, reviewed, caplog):
    with caplog.at_level("WARNING", logger="cowork.api.v1.endpoints.artifact_workspace"):
        url = api.post(LINK, headers=IDENTITY).json()["url"]

    [record] = events(caplog, "draft_link_issued")
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

    response = draft.get(_on_the_draft_host(url), headers=FETCH)

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

    monkeypatch.setattr(workspace_ep, "review_artifact_and_read", reviewed_and_read(refuse))

    assert api.post(LINK, headers=IDENTITY).status_code == 404


def test_the_path_lookup_runs_off_the_event_loop(api, reviewed, monkeypatch):
    # The path lookup resolves symlinks on the file store: blocking, on a
    # request the web app sends at every preview open and reload. The review
    # is a coroutine whose database and file store reads run in worker
    # threads of their own (tests/test_artifact_resolution_off_the_loop.py).
    on_loop: dict[str, bool] = {}
    names = ("_artifact_primary", "_primary_rel_path", "_draft_file_parts")
    for name in names:

        def watched(*args, _name=name, _original=getattr(workspace_ep, name)):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                on_loop[_name] = False
            else:
                on_loop[_name] = True
            return _original(*args)

        monkeypatch.setattr(workspace_ep, name, watched)

    assert api.post(LINK, headers=IDENTITY).status_code == 200
    assert on_loop == dict.fromkeys(names, False)


def test_an_artifact_with_no_previewable_file_gets_no_link(api, tmp_path, monkeypatch, caplog):
    source, folder, metadata = make_artifact(tmp_path)
    (folder / "index.html").unlink()
    shutil.rmtree(folder / "assets")
    monkeypatch.setattr(
        workspace_ep, "review_artifact_and_read", reviewed_and_read(fake_review(source, folder, metadata, {}))
    )

    with caplog.at_level("WARNING", logger="cowork.api.v1.endpoints.artifact_workspace"):
        response = api.post(LINK, headers=IDENTITY)

    assert response.status_code == 404
    assert response.json()["detail"] == "This artifact has no previewable file."
    assert events(caplog, "draft_link_issued") == []


def test_a_primary_outside_the_artifact_folder_gets_no_link(api, tmp_path, monkeypatch, caplog):
    source, folder, metadata = make_artifact(tmp_path)
    outside = folder.parent / "secret.txt"
    assert outside.is_file()
    monkeypatch.setattr(
        workspace_ep, "review_artifact_and_read", reviewed_and_read(fake_review(source, folder, metadata, {}))
    )
    monkeypatch.setattr(workspace_ep, "_artifact_primary", lambda *_args: outside)

    with caplog.at_level("WARNING", logger="cowork.api.v1.endpoints.artifact_workspace"):
        response = api.post(LINK, headers=IDENTITY)

    assert response.status_code == 404
    assert response.json()["detail"] == "Artifact file not found"
    assert events(caplog, "draft_link_issued") == []


@pytest.mark.parametrize(
    "scope",
    [
        TenantScope(org_mode=True, org_id=ORG, user_id=None),
        TenantScope(org_mode=True, org_id=None, user_id=USER),
    ],
    ids=["no-user", "no-org"],
)
async def test_a_scope_without_a_user_or_an_org_is_401_and_reviews_nothing(draft_env, monkeypatch, caplog, scope):
    # Reached only by a caller that builds the session itself: the request
    # dependency already answers 401 for an org request without an org.
    def never(*_args):
        raise AssertionError("the review must not run")

    monkeypatch.setattr(workspace_ep, "review_artifact_and_read", reviewed_and_read(never))

    with caplog.at_level("WARNING", logger="cowork.api.v1.endpoints.artifact_workspace"):
        with pytest.raises(HTTPException) as refused:
            await workspace_ep.create_draft_link(PROJECT, DASHED_ARTIFACT, SimpleNamespace(scope=scope))

    assert refused.value.status_code == 401
    assert events(caplog, "draft_link_issued") == []


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


def test_health_says_the_deployment_serves_draft_links(draft_env):
    # The web client reads this once at boot; every way into a preview (the
    # Artifacts page, the rail, a chat card) then opens drafts by link.
    assert health.health()["draft_links"] is True


def test_health_says_no_draft_links_while_a_value_is_missing(draft_env, monkeypatch):
    # The settings tests cover each value; this pins that /health follows them.
    monkeypatch.setenv("COWORK_DRAFT_PORT", "0")
    get_app_settings.cache_clear()

    assert health.health()["draft_links"] is False



# The tests above stub the review. These do not: they pin that
# the draft host asks the real review on every request, so a grant that is
# withdrawn stops a link that was issued while it stood.


# Draft links on and role checks granted; the review itself is not stubbed.
real_review = pytest.mark.usefixtures("draft_env", "granted_product_permissions")


async def _issue_link(row, artifact_id, scoped) -> str:
    response = await workspace_ep.create_draft_link(str(row.id), artifact_id, scoped)
    assert response.status_code == 200, response.body
    return json.loads(response.body)["url"]


@real_review
async def test_an_owner_link_opens_through_the_real_review(session, tmp_path):
    row, folder = tenancy._project_with_artifact(session, tmp_path, name="mine", org_id=tenancy.ORG_A, slug="dash")
    artifact_id, _ = ensure_full_id(folder)
    url = await _issue_link(row, artifact_id, tenancy._scoped(session, tenancy.ORG_A))
    draft = TestClient(create_draft_app(), base_url=f"https://{DRAFT_HOST}")

    response = draft.get(_on_the_draft_host(url), headers=FETCH)

    assert response.status_code == 200, response.text


@real_review
async def test_a_reviewer_link_stops_opening_when_the_grant_is_withdrawn(session, tmp_path, monkeypatch):
    row, folder, card = await tenancy._grant_draft_review(session, tmp_path, monkeypatch)
    with Session(session.get_bind()) as reviewer_session:
        reviewer = tenancy._scoped(reviewer_session, tenancy.ORG_A, tenancy.USER_A2)
        url = await _issue_link(row, card["id"], reviewer)
    draft = TestClient(create_draft_app(), base_url=f"https://{DRAFT_HOST}")
    assert draft.get(_on_the_draft_host(url), headers=FETCH).status_code == 200

    assert disable_draft_review(folder) is True

    assert draft.get(_on_the_draft_host(url), headers=FETCH).status_code == 404


@real_review
async def test_a_co_member_without_a_grant_gets_no_link(session, tmp_path):
    row, folder = tenancy._project_with_artifact(session, tmp_path, name="mine", org_id=tenancy.ORG_A, slug="dash")
    artifact_id, _ = ensure_full_id(folder)

    with Session(session.get_bind()) as other_session:
        other = tenancy._scoped(other_session, tenancy.ORG_A, tenancy.USER_A2)
        with pytest.raises(HTTPException) as refused:
            await workspace_ep.create_draft_link(str(row.id), artifact_id, other)

    assert refused.value.status_code == 404
