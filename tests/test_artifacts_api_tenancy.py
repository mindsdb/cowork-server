"""The artifact HTTP surface in org mode.

Only two endpoints are exposed, both addressed by project id and slug — never by a
server filesystem path, because a path carries no tenant and these endpoints
previously had no Principal to compare it against.

The org cases call the handlers directly with an explicitly built ScopedSession:
`cowork.server.app` is created at import time and only wires the principal
middleware when the process started in org mode, so a TestClient request would
silently run under LOCAL_SCOPE and prove nothing about isolation.
"""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.api.v1.endpoints import artifacts as ep
from cowork.services import artifacts as ep_artifacts
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.project import Project

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"


def _set_mode(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("COWORK_TENANCY_MODE", mode)
    get_app_settings.cache_clear()


@pytest.fixture
def org_mode(monkeypatch):
    _set_mode(monkeypatch, "org")
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def local_mode(monkeypatch):
    _set_mode(monkeypatch, "local")
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _scoped(session, org_id: str) -> ScopedSession:
    return ScopedSession(session, TenantScope(org_mode=True, org_id=org_id, user_id="u-1"))


def _project_with_artifact(session, tmp_path, *, name, org_id, slug, conversation=None):
    path = tmp_path / (org_id or "local") / name
    # Org projects carry the conversation segment the cloud pod writes under
    # (artifact_roots.CONVERSATIONS_DIRNAME); desktop projects do not. Mirroring
    # both real layouts here is what makes these tests exercise the resolver
    # rather than a shape only the tests believe in.
    workspace = path / "conversations" / (conversation or "c1") if org_id is not None else path
    folder = workspace / ".anton" / "artifacts" / slug
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(
        json.dumps({"slug": slug, "name": slug, "type": "html-app"})
    )
    row = Project(id=uuid.uuid4(), name=name, path=str(path), org_id=org_id)
    session.add(row)
    session.commit()
    return row, folder


@pytest.fixture
def publish_key(monkeypatch):
    async def fake_get(self):
        return "turnkey-1"

    async def fake_revoke(self):
        return None

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey.get", fake_get)
    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey.revoke", fake_revoke)


# ── list ──────────────────────────────────────────────────────────────────

async def test_list_returns_own_org_artifacts(session, tmp_path, org_mode):
    _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")

    cards = ep.artifacts_for_request(_scoped(session, ORG_A))

    assert "dash" in [c["slug"] for c in cards]


async def test_list_hides_other_org_artifacts(session, tmp_path, org_mode):
    _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")
    _project_with_artifact(session, tmp_path, name="theirs", org_id=ORG_B, slug="secret")

    slugs = [c["slug"] for c in ep.artifacts_for_request(_scoped(session, ORG_A))]

    assert "dash" in slugs          # not vacuously empty
    assert "secret" not in slugs


async def test_list_by_project_id_narrows_to_that_project(session, tmp_path, org_mode):
    row, _ = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")
    _project_with_artifact(session, tmp_path, name="other", org_id=ORG_A, slug="second")

    cards = ep.artifacts_for_request(_scoped(session, ORG_A), project_id=row.id)

    assert [c["slug"] for c in cards] == ["dash"]


async def test_list_by_foreign_project_id_is_404(session, tmp_path, org_mode):
    row, _ = _project_with_artifact(session, tmp_path, name="theirs", org_id=ORG_B, slug="secret")

    with pytest.raises(HTTPException) as err:
        ep.artifacts_for_request(_scoped(session, ORG_A), project_id=row.id)

    assert err.value.status_code == 404


async def test_list_rejects_project_path_in_org_mode(session, tmp_path, org_mode):
    row, _ = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")

    with pytest.raises(HTTPException) as err:
        ep.artifacts_for_request(_scoped(session, ORG_A), project_path=row.path)

    assert err.value.status_code == 400


async def test_list_merges_projects_and_labels_them(session, tmp_path, org_mode):
    _project_with_artifact(session, tmp_path, name="alpha", org_id=ORG_A, slug="one")
    _project_with_artifact(session, tmp_path, name="beta", org_id=ORG_A, slug="two")

    cards = ep.artifacts_for_request(_scoped(session, ORG_A))
    labels = {c["slug"]: c["projectName"] for c in cards}

    assert labels["one"] == "alpha"
    assert labels["two"] == "beta"


async def test_list_is_capped_at_eighty(session, tmp_path, org_mode):
    # The cap is pre-existing but matters more here: the org list spans every
    # project, so a large org silently loses the tail.
    row, first = _project_with_artifact(session, tmp_path, name="big", org_id=ORG_A, slug="a-000")
    # Same conversation root the helper just created — sibling of a-000.
    base = first.parent
    for i in range(1, 90):
        folder = base / f"a-{i:03d}"
        folder.mkdir()
        (folder / "index.html").write_text("<html></html>")
        (folder / "metadata.json").write_text(
            json.dumps({"slug": f"a-{i:03d}", "type": "html-app"})
        )

    cards = ep.artifacts_for_request(_scoped(session, ORG_A), project_id=row.id)

    assert len(cards) == 80


async def test_card_omits_owner_side_access_fields_in_org_mode(session, tmp_path, org_mode):
    _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")

    card = (ep.artifacts_for_request(_scoped(session, ORG_A)))[0]

    assert "accessPassword" not in card
    assert "accessEmails" not in card


# ── delete ────────────────────────────────────────────────────────────────

async def test_delete_by_slug_removes_the_folder(session, tmp_path, org_mode, publish_key):
    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")

    await ep.delete_artifact_for_request(_scoped(session, ORG_A), "dash", project_id=row.id)

    assert not folder.exists()


async def test_delete_in_foreign_project_is_404_and_keeps_files(
    session, tmp_path, org_mode, publish_key
):
    row, folder = _project_with_artifact(
        session, tmp_path, name="theirs", org_id=ORG_B, slug="secret"
    )

    with pytest.raises(HTTPException) as err:
        await ep.delete_artifact_for_request(_scoped(session, ORG_A), "secret", project_id=row.id)

    assert err.value.status_code == 404
    assert folder.exists()


async def test_delete_in_desktop_hits_the_named_project_not_the_first_one(
    session, tmp_path, local_mode, publish_key
):
    """The dangerous case: if the desktop branch ignored project_id and the handler
    took sources[0], a same-named slug in another project would be deleted instead.
    Inline chat cards carry a project_id in every mode, so this path is reachable."""
    _first, first_folder = _project_with_artifact(
        session, tmp_path, name="aaa-first", org_id=None, slug="dash"
    )
    second, second_folder = _project_with_artifact(
        session, tmp_path, name="zzz-second", org_id=None, slug="dash"
    )

    await ep.delete_artifact_for_request(
        ScopedSession(session, LOCAL_SCOPE), "dash", project_id=second.id
    )

    assert not second_folder.exists()
    assert first_folder.exists()


# ── fail closed ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [
    "/api/v1/artifacts/status?path=/x",
    "/api/v1/artifacts/preview?path=/x",
    "/api/v1/publish/",
])
def test_desktop_only_endpoints_are_501_in_org_mode(org_mode, path):
    # TestClient is fine here: require_local_tenancy reads the setting at request
    # time, so the app's build-time mode is irrelevant.
    from cowork.server import app

    assert TestClient(app).get(path).status_code == 501


def test_desktop_only_endpoints_are_reachable_in_local_mode(local_mode):
    from cowork.server import app

    assert TestClient(app).get("/api/v1/artifacts/status?path=/nope").status_code != 501


# ── desktop project_path filter ────────────────────────────────────────────

async def test_desktop_project_path_narrows_to_that_project(
    session, tmp_path, local_mode, monkeypatch
):
    """The legacy desktop addressing still works.

    The client value is matched as a normalized string, never resolved through the
    filesystem — untrusted input must not drive a filesystem access even when the
    result is only compared for equality.
    """
    _, mine = _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="dash")
    _project_with_artifact(session, tmp_path, name="other", org_id=None, slug="second")
    mine_project_dir = mine.parent.parent.parent

    monkeypatch.setattr(
        ep, "_sources_for_scan",
        lambda: [
            ep_artifacts.ProjectArtifacts(
                base=p / ".anton" / "artifacts", project_id=None, project_name=p.name,
            )
            for p in sorted((tmp_path / "local").iterdir())
        ],
    )

    cards = ep.artifacts_for_request(
        ScopedSession(session, LOCAL_SCOPE), project_path=str(mine_project_dir),
    )

    assert [c["slug"] for c in cards] == ["dash"]


async def test_desktop_project_path_that_matches_nothing_yields_nothing(
    session, tmp_path, local_mode, monkeypatch
):
    _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="dash")
    monkeypatch.setattr(
        ep, "_sources_for_scan",
        lambda: [
            ep_artifacts.ProjectArtifacts(
                base=(tmp_path / "local" / "mine") / ".anton" / "artifacts",
                project_id=None, project_name="mine",
            )
        ],
    )

    cards = ep.artifacts_for_request(
        ScopedSession(session, LOCAL_SCOPE), project_path="/nowhere/at/all",
    )

    assert cards == []
