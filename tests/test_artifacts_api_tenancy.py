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
from cowork.models.conversation import Conversation
from cowork.models.project import Project

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"
USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
# A second member of ORG_A. Same tenant, different person: the axis the org
# filter alone does not cover.
USER_A2 = "a2a2a2a2-a2a2-a2a2-a2a2-a2a2a2a2a2a2"

#: What the gateway injects on a real org-mode request.
_IDENTITY = {"X-User-Id": USER_A, "X-Organization-Id": ORG_A}


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


def _scoped(session, org_id: str, user_id: str = USER_A) -> ScopedSession:
    return ScopedSession(session, TenantScope(org_mode=True, org_id=org_id, user_id=user_id))


def _project_with_artifact(session, tmp_path, *, name, org_id, slug, owner=USER_A):
    path = tmp_path / (org_id or "local") / name
    row = Project(id=uuid.uuid4(), name=name, path=str(path), org_id=org_id)
    session.add(row)

    # Org projects carry the conversation segment the cloud pod writes under
    # (artifact_roots.CONVERSATIONS_DIRNAME); desktop projects do not. Mirroring
    # both real layouts here is what makes these tests exercise the resolver
    # rather than a shape only the tests believe in. The directory name is the
    # conversation's id and the row behind it is what says who owns the
    # artifacts inside, so both have to be real.
    workspace = path
    if org_id is not None:
        conversation = Conversation(
            id=uuid.uuid4(), topic=f"{name} chat", project_id=row.id, org_id=org_id, created_by=owner
        )
        session.add(conversation)
        workspace = path / "conversations" / str(conversation.id)

    folder = workspace / ".anton" / "artifacts" / slug
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(
        json.dumps({"slug": slug, "name": slug, "type": "html-app"})
    )
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


def test_list_hides_another_members_artifacts(session, tmp_path, org_mode):
    """Same org, different chat. Live artifacts live inside a conversation, so
    they inherit that conversation's privacy rather than the project's sharing.

    Two projects here, because `_project_with_artifact` builds one per call. The
    one-project-two-owners case, which is what a narrowing of the filter to
    per-project granularity would slip past, is pinned at the resolver instead:
    `test_artifact_roots.py::test_sources_for_project_skip_another_members_conversations`.
    """
    _project_with_artifact(
        session, tmp_path, name="shared", org_id=ORG_A, slug="theirs", owner=USER_A2
    )
    _project_with_artifact(
        session, tmp_path, name="also-mine", org_id=ORG_A, slug="mine", owner=USER_A
    )

    slugs = [c["slug"] for c in ep.artifacts_for_request(_scoped(session, ORG_A, USER_A))]

    assert "mine" in slugs
    assert "theirs" not in slugs


def test_list_by_project_id_narrows_to_that_project(session, tmp_path, org_mode):
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


async def test_delete_cannot_reach_another_members_artifact(
    session, tmp_path, org_mode, publish_key
):
    """The project is shared, so the scoped read hands it over and the org filter
    is satisfied. What stops the delete is that none of the roots under it belong
    to the caller, and a slug that resolves to no root of theirs is a miss."""
    row, folder = _project_with_artifact(
        session, tmp_path, name="shared-proj", org_id=ORG_A, slug="theirs", owner=USER_A
    )

    with pytest.raises(HTTPException) as err:
        await ep.delete_artifact_for_request(
            _scoped(session, ORG_A, USER_A2), "theirs", project_id=row.id
        )

    assert err.value.status_code == 404
    assert folder.exists()


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
def test_desktop_only_endpoints_are_403_in_org_mode(org_mode, path):
    # Build the app under the mode this test names instead of importing the
    # module-level `app`. That one is created on first import and keeps
    # whichever middleware stack the mode in force at that moment gave it, so
    # importing it makes the answer depend on which test file ran first.
    #
    # The identity headers are what the gateway injects on every real request.
    # Without them the principal middleware answers 401 before the guard is
    # reached, which says nothing about the guard.
    #
    # 403 rather than 501 so the edge books the refusal as a client error. The
    # capability 501 in handlers/responses.py is a different thing and keeps its
    # status; tests/test_no_execution_in_org_mode.py pins it.
    from cowork.server import create_app

    res = TestClient(create_app()).get(path, headers=_IDENTITY)
    assert res.status_code == 403
    # require_local answers 403 too, so the detail is what tells the two
    # refusals apart, and it is what tells the caller why.
    assert res.json()["detail"] == "not available in org deployments"


def test_desktop_only_endpoints_are_reachable_in_local_mode(local_mode):
    from cowork.server import create_app

    # /status never raises for an unknown path, so a guard that stays quiet
    # means 200. Asserting the real status rather than "not the refusal" keeps
    # this test failing if the guard ever fires in local mode. No identity
    # headers: the desktop sends none and local mode wires no middleware.
    assert TestClient(create_app()).get("/api/v1/artifacts/status?path=/nope").status_code == 200


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
