"""The artifact HTTP surface in org mode.

Every org-capable endpoint is addressed by project id plus stable artifact id —
never by a server filesystem path, because a path carries no tenant.

The org cases call the handlers directly with an explicitly built ScopedSession:
`cowork.server.app` is created at import time and only wires the principal
middleware when the process started in org mode, so a TestClient request would
silently run under LOCAL_SCOPE and prove nothing about isolation.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.api.v1.endpoints import artifacts as ep
from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.services import artifacts as ep_artifacts
from cowork.services.artifact_identity import ensure_full_id
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


def _project_with_artifact(
    session, tmp_path, *, name, org_id, slug, owner=USER_A, conversation=None
):
    path = tmp_path / (org_id or "local") / name
    row = Project(id=uuid.uuid4(), name=name, path=str(path), org_id=org_id)
    session.add(row)

    # Org projects carry the conversation segment the cloud pod writes under
    # (artifact_roots.CONVERSATIONS_DIRNAME); desktop projects do not. Mirroring
    # both real layouts here is what makes these tests exercise the resolver
    # rather than a shape only the tests believe in. The directory name is the
    # conversation's id and the row behind it is what says who owns the
    # artifacts inside, so both have to be real.
    # `conversation` pins that id when a test needs the artifact's
    # `originConversationId` to point at a chat it already knows.
    workspace = path
    if org_id is not None:
        conversation_id = UUID(str(conversation)) if conversation else uuid.uuid4()
        session.add(
            Conversation(
                id=conversation_id,
                topic=f"{name} chat",
                project_id=row.id,
                org_id=org_id,
                created_by=owner,
            )
        )
        workspace = path / "conversations" / str(conversation_id)

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

    cards = await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=None, project_path=None)

    assert "dash" in [c["slug"] for c in cards]


async def test_list_hides_other_org_artifacts(session, tmp_path, org_mode):
    _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")
    _project_with_artifact(session, tmp_path, name="theirs", org_id=ORG_B, slug="secret")

    cards = await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=None, project_path=None)
    slugs = [c["slug"] for c in cards]

    assert "dash" in slugs          # not vacuously empty
    assert "secret" not in slugs


async def test_list_hides_another_members_legacy_artifacts(session, tmp_path, org_mode):
    """Same org, different chat, LEGACY layout. Artifacts written before
    ENG-2056 live inside a conversation workspace, so they inherit that
    conversation's privacy. New artifacts land in the shared project base and
    are project-wide by decision — that case is pinned in
    `test_artifact_roots.py::test_listing_shows_project_base_artifacts_to_every_member_and_keeps_legacy_ones`.

    Two projects here, because `_project_with_artifact` builds one per call. The
    one-project-two-owners case, which is what a narrowing of the filter to
    per-project granularity would slip past, is pinned at the resolver instead:
    `test_artifact_roots.py::test_sources_for_project_skip_another_members_legacy_conversations`.
    """
    _project_with_artifact(
        session, tmp_path, name="shared", org_id=ORG_A, slug="theirs", owner=USER_A2
    )
    _project_with_artifact(
        session, tmp_path, name="also-mine", org_id=ORG_A, slug="mine", owner=USER_A
    )

    cards = await ep.list_artifacts(
        _scoped(session, ORG_A, USER_A).scope, project_id=None, project_path=None
    )
    slugs = [c["slug"] for c in cards]

    assert "mine" in slugs
    assert "theirs" not in slugs


async def test_list_by_project_id_narrows_to_that_project(session, tmp_path, org_mode):
    row, _ = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")
    _project_with_artifact(session, tmp_path, name="other", org_id=ORG_A, slug="second")

    cards = await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=row.id, project_path=None)

    assert [c["slug"] for c in cards] == ["dash"]


async def test_list_by_foreign_project_id_is_404(session, tmp_path, org_mode):
    row, _ = _project_with_artifact(session, tmp_path, name="theirs", org_id=ORG_B, slug="secret")

    with pytest.raises(HTTPException) as err:
        await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=row.id, project_path=None)

    assert err.value.status_code == 404


async def test_list_rejects_project_path_in_org_mode(session, tmp_path, org_mode):
    row, _ = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")

    with pytest.raises(HTTPException) as err:
        await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=None, project_path=row.path)

    assert err.value.status_code == 400


async def test_list_merges_projects_and_labels_them(session, tmp_path, org_mode):
    _project_with_artifact(session, tmp_path, name="alpha", org_id=ORG_A, slug="one")
    _project_with_artifact(session, tmp_path, name="beta", org_id=ORG_A, slug="two")

    cards = await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=None, project_path=None)
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

    cards = await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=row.id, project_path=None)

    assert len(cards) == 80


async def test_card_omits_owner_side_access_fields_in_org_mode(session, tmp_path, org_mode):
    _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")

    card = (await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=None, project_path=None))[0]

    assert "accessPassword" not in card
    assert "accessEmails" not in card


def _published(folder: Path, **entry) -> None:
    """A .published.json entry for the folder's index.html, as a publish
    writes it."""
    (folder / ".published.json").write_text(json.dumps({"index.html": {
        "report_id": f"r-{folder.name}", "url": f"https://view.example/r/{folder.name}",
        "published": True, **entry,
    }}))


def _audience(card: dict) -> tuple:
    """The card's audience fields, indexed so a card without one fails: the
    app reads a missing field as an empty audience, never as unknown."""
    return tuple(card[field] for field in ("orgAllowed", "ownerOnly", "allowedTeamIds", "allowedUserIds"))


# What an entry written before MindsHub mode stopped keeping a copy of the
# audience says, or what an agent in the pod could write there.
_OWNER_ONLY = {"mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True}
_TEAM_SHARED = {"mode": "restricted", "emails": [], "org_allowed": False, "owner_only": False,
                "allowed_team_ids": ["33333333-3333-4333-8333-333333333333"]}


async def test_an_org_card_says_it_does_not_know_who_a_restricted_artifact_reaches(
    session, tmp_path, org_mode
):
    """In MindsHub mode .published.json keeps no copy of a restricted
    artifact's audience: MindsHub's rule is the one record, which only the
    owner's GET .../access reads. So the list answers its audience fields as
    null, for an artifact only its owner can open and for one shared with a
    team alike, never as an empty audience it doesn't know. A public card's
    audience is no one in particular."""
    _row, owner_only = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="draft")
    team_shared = owner_only.parent / "team"
    public = owner_only.parent / "public"
    for folder in (team_shared, public):
        folder.mkdir()
        (folder / "index.html").write_text("<html></html>")
        (folder / "metadata.json").write_text(json.dumps({"slug": folder.name, "type": "html-app"}))
    _published(owner_only, **_OWNER_ONLY)
    _published(team_shared, **_TEAM_SHARED)
    _published(public, mode="public")

    cards = {
        card["slug"]: card
        for card in await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=None, project_path=None)
    }

    for slug in ("draft", "team"):
        assert cards[slug]["accessMode"] == "restricted"
        assert _audience(cards[slug]) == (None, None, None, None), slug
        assert "accessEmails" not in cards[slug]
    assert (cards["public"]["accessMode"], _audience(cards["public"])) == ("public", (False, False, [], []))


async def test_a_desktop_card_says_who_a_restricted_artifact_reaches(session, tmp_path, local_mode):
    """On the desktop the entry is the record, so the list answers it."""
    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="draft")
    _published(folder, **_OWNER_ONLY)

    [card] = await ep.list_artifacts(
        LOCAL_SCOPE, project_id=row.id, project_path=None
    )

    assert (card["accessMode"], card["accessEmails"], _audience(card)) == (
        "restricted", [], (False, True, [], []),
    )


def test_a_desktop_status_says_an_artifact_only_its_owner_can_open(session, tmp_path, monkeypatch, local_mode):
    """The preview viewer's refresh reads who a desktop artifact reaches from
    GET /status, so an artifact only its owner can open answers ownerOnly
    true there, as its card does: without it the app reads "Specific people"
    with nobody listed."""
    from cowork.server import create_app

    _row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="draft")
    _published(folder, **_OWNER_ONLY)
    source = ep_artifacts.ProjectArtifacts(
        base=folder.parent,
        project_id=None,
        project_name="mine",
        trusted_anchor=folder.parent.parent.parent,
        root_parts=(".anton", "artifacts"),
    )
    monkeypatch.setattr(ep, "artifacts_sources_for_scan", lambda *, scope: [source])

    res = TestClient(create_app()).get("/api/v1/artifacts/status", params={"path": str(folder / "index.html")})

    assert res.status_code == 200
    status = res.json()
    assert (status["accessMode"], status["accessEmails"], status["orgAllowed"], status["ownerOnly"]) == (
        "restricted", [], False, True,
    )


async def test_card_has_one_identity_for_draft_and_comments(session, tmp_path, org_mode):
    row, folder = _project_with_artifact(
        session, tmp_path, name="mine", org_id=ORG_A, slug="dash"
    )

    card = (await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=row.id, project_path=None))[0]
    artifact_id = card["id"]

    assert UUID(artifact_id).hex == artifact_id
    assert "stableId" not in card
    assert card["artifactKey"] == f"artifact/{UUID(artifact_id)}"
    assert card["draftUrl"].startswith(
        f"/api/v1/artifacts/drafts/{row.id}/{artifact_id}/"
    )
    # GET/listing derives legacy identities in memory and must not mutate the
    # agent-owned metadata tree. Identity-resolving mutation routes persist the
    # same deterministic value when they subsequently need it.
    persisted = json.loads((folder / "metadata.json").read_text())
    assert "id" not in persisted
    assert "stableId" not in persisted


async def test_org_source_read_and_manual_edit_are_project_scoped(
    session, tmp_path, org_mode
):
    row, folder = _project_with_artifact(
        session, tmp_path, name="mine", org_id=ORG_A, slug="dash"
    )
    card = (await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=row.id, project_path=None))[0]

    source = await workspace_ep.artifact_source(
        str(row.id), card["id"], _scoped(session, ORG_A).scope, path="index.html"
    )
    saved = await workspace_ep.update_artifact_source(
        str(row.id),
        card["id"],
        workspace_ep._SourceUpdateBody(
            content="<html><h1>Edited</h1></html>",
            expectedRevisionId=source["revision"]["id"],
            path="index.html",
            summary="Edit heading",
        ),
        _scoped(session, ORG_A).scope,
    )

    assert saved["revision"]["actor"] == {"kind": "manual", "id": USER_A}
    assert (folder / "index.html").read_text() == "<html><h1>Edited</h1></html>"


async def test_org_source_read_cannot_cross_org(session, tmp_path, org_mode):
    row, _folder = _project_with_artifact(
        session, tmp_path, name="theirs", org_id=ORG_B, slug="secret"
    )
    card = (await ep.list_artifacts(_scoped(session, ORG_B).scope, project_id=row.id, project_path=None))[0]

    # A raw SQLAlchemy session is intentionally single-tenant once wrapped.
    # Use another request-like session for the foreign caller.
    with Session(session.get_bind()) as foreign_session:
        with pytest.raises(HTTPException) as err:
            await workspace_ep.artifact_source(
                str(row.id), card["id"], _scoped(foreign_session, ORG_A).scope, path="index.html"
            )

        assert err.value.status_code == 404


async def _grant_draft_review(session, tmp_path, monkeypatch, *, slug="dash", org_id=ORG_A):
    """Own an artifact and open it for same-org review, the way the owner does.

    The grant is the whole point of these tests: a project is org-shared but the
    conversation workspace under it is private (ENG-1910), so without an owner
    action a co-member cannot see the artifact at all.
    """
    row, folder = _project_with_artifact(
        session, tmp_path, name="mine", org_id=org_id, slug=slug
    )
    card = (await ep.list_artifacts(_scoped(session, org_id).scope, project_id=row.id, project_path=None))[0]

    async def fake_provision(*_args, **_kwargs):
        return card["artifactKey"]

    monkeypatch.setattr(
        "cowork.services.artifact_access.provision_draft_review_access",
        fake_provision,
    )
    monkeypatch.setattr(
        "cowork.services.artifact_authorization_identity._allocate_or_adopt",
        lambda *_args: card["artifactKey"],
    )
    enabled = await workspace_ep.enable_artifact_comments(
        str(row.id), card["id"], _scoped(session, org_id).scope
    )
    assert enabled.enabled is True
    return row, folder, card


async def test_same_org_reviewer_can_preview_but_cannot_edit_source(
    session, tmp_path, org_mode, monkeypatch
):
    row, _folder, card = await _grant_draft_review(session, tmp_path, monkeypatch)
    with Session(session.get_bind()) as reviewer_session:
        reviewer = _scoped(reviewer_session, ORG_A, user_id=USER_A2)
        entry = await workspace_ep.artifact_review_entry(str(row.id), card["id"], reviewer.scope)
        assert entry["capabilities"]["role"] == "reviewer"
        assert entry["capabilities"]["canEdit"] is False

        draft = await workspace_ep.serve_private_draft(
            str(row.id), card["id"], "index.html", MagicMock(query_params={}), reviewer.scope
        )
        assert draft.status_code == 200

        with pytest.raises(HTTPException) as err:
            await workspace_ep.artifact_source(
                str(row.id), card["id"], reviewer.scope, path="index.html"
            )
        assert err.value.status_code == 403


async def test_ungranted_draft_stays_invisible_to_a_co_member(session, tmp_path, org_mode):
    """No owner action, no access — and the refusal is a 404, so a co-member
    cannot use the review routes to learn that the artifact exists."""
    row, folder = _project_with_artifact(
        session, tmp_path, name="mine", org_id=ORG_A, slug="dash"
    )
    artifact_id, _metadata = ensure_full_id(folder)
    with Session(session.get_bind()) as reviewer_session:
        reviewer = _scoped(reviewer_session, ORG_A, user_id=USER_A2)

        for call in (
            workspace_ep.artifact_review_entry(str(row.id), artifact_id, reviewer.scope),
            workspace_ep.serve_private_draft(
                str(row.id), artifact_id, "index.html", MagicMock(query_params={}), reviewer.scope
            ),
        ):
            with pytest.raises(HTTPException) as err:
                await call
            assert err.value.status_code == 404


async def test_reviewer_cannot_provision_draft_review(session, tmp_path, org_mode, monkeypatch):
    """Provisioning mints an auth rule and reopens a private workspace, so it is
    the owner's decision. A reviewer's client reads `review` instead."""
    row, _folder, card = await _grant_draft_review(session, tmp_path, monkeypatch)
    with Session(session.get_bind()) as reviewer_session:
        with pytest.raises(HTTPException) as err:
            await workspace_ep.enable_artifact_comments(
                str(row.id), card["id"], _scoped(reviewer_session, ORG_A, user_id=USER_A2).scope
            )

    assert err.value.status_code == 403


async def test_a_comparison_sides_draft_is_never_opened_to_the_org(session, tmp_path, org_mode, monkeypatch):
    """Turning review on opens the draft to the organization, and nothing
    leaves a model comparison until the user continues with a side."""
    from cowork.services.projects import COMPARISON_SANDBOX_PREFIX

    row, folder = _project_with_artifact(
        session, tmp_path, name=f"{COMPARISON_SANDBOX_PREFIX}{'c' * 32}", org_id=ORG_A, slug="dash"
    )
    card = (await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=row.id, project_path=None))[0]
    provisioned = []

    async def fake_provision(*args, **kwargs):
        provisioned.append(args)
        return card["artifactKey"]

    monkeypatch.setattr("cowork.services.artifact_access.provision_draft_review_access", fake_provision)
    monkeypatch.setattr(
        "cowork.services.artifact_authorization_identity._allocate_or_adopt",
        lambda *_args: card["artifactKey"],
    )

    with pytest.raises(HTTPException) as err:
        await workspace_ep.enable_artifact_comments(str(row.id), card["id"], _scoped(session, ORG_A).scope)

    assert err.value.status_code == 409
    assert "Continue with this side" in err.value.detail
    assert provisioned == []
    assert not (folder / ".revisions" / "draft-review.json").exists()
    with Session(session.get_bind()) as reviewer_session:
        with pytest.raises(HTTPException) as err:
            await workspace_ep.artifact_review_entry(
                str(row.id), card["id"], _scoped(reviewer_session, ORG_A, user_id=USER_A2).scope
            )
        assert err.value.status_code == 404


async def test_same_org_reviewer_receives_the_current_revision_for_comments(
    session, tmp_path, org_mode, monkeypatch
):
    row, _folder, card = await _grant_draft_review(session, tmp_path, monkeypatch)
    with Session(session.get_bind()) as reviewer_session:
        result = await workspace_ep.artifact_review_entry(
            str(row.id),
            card["id"],
            _scoped(reviewer_session, ORG_A, user_id=USER_A2).scope,
        )

    assert result["capabilities"]["role"] == "reviewer"
    assert result["capabilities"]["canEdit"] is False
    assert result["currentRevision"]["artifactId"] == card["id"]
    assert result["currentRevision"]["path"] == "index.html"


async def test_fullstack_draft_preview_cannot_read_backend_source(
    session, tmp_path, org_mode, monkeypatch
):
    row, folder, _card = await _grant_draft_review(session, tmp_path, monkeypatch, slug="app")
    (folder / "static").mkdir()
    (folder / "index.html").replace(folder / "static" / "index.html")
    (folder / "static" / "app.js").write_text("console.log('safe')")
    (folder / "backend.py").write_text("API_SECRET = 'server-only'")
    (folder / "metadata.json").write_text(json.dumps({
        "slug": "app",
        "name": "app",
        "type": "fullstack-stateless-app",
        "primary": "static/index.html",
    }))
    card = (await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=row.id, project_path=None))[0]
    with Session(session.get_bind()) as reviewer_session:
        reviewer = _scoped(reviewer_session, ORG_A, user_id=USER_A2)
        asset = await workspace_ep.serve_private_draft(
            str(row.id), card["id"], "static/app.js", MagicMock(query_params={}), reviewer.scope
        )
        assert asset.status_code == 200
        # Read the body as the server sends it: its end closes the pinned file
        # and folders the response holds, rather than leaving them to the GC.
        assert b"".join([chunk async for chunk in asset.body_iterator]) == b"console.log('safe')"

        with pytest.raises(HTTPException) as err:
            await workspace_ep.serve_private_draft(
                str(row.id), card["id"], "backend.py", MagicMock(query_params={}), reviewer.scope
            )
        assert err.value.status_code == 404


async def test_fullstack_draft_preview_refuses_a_root_level_primary(
    session, tmp_path, org_mode, monkeypatch
):
    row, folder, _card = await _grant_draft_review(
        session, tmp_path, monkeypatch, slug="legacy-app"
    )
    (folder / "backend.py").write_text("API_SECRET = 'server-only'")
    (folder / "metadata.json").write_text(json.dumps({
        "slug": "legacy-app",
        "name": "legacy-app",
        "type": "fullstack-stateless-app",
        "primary": "index.html",
    }))
    card = (await ep.list_artifacts(_scoped(session, ORG_A).scope, project_id=row.id, project_path=None))[0]
    with Session(session.get_bind()) as reviewer_session:
        reviewer = _scoped(reviewer_session, ORG_A, user_id=USER_A2)
        with pytest.raises(HTTPException) as err:
            await workspace_ep.serve_private_draft(
                str(row.id), card["id"], "index.html", MagicMock(query_params={}), reviewer.scope
            )
        assert err.value.status_code == 404


# ── delete ────────────────────────────────────────────────────────────────

async def test_delete_by_slug_removes_the_folder(session, tmp_path, org_mode, publish_key):
    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")

    await ep.delete_artifact_for_request(_scoped(session, ORG_A).scope, "dash", project_id=row.id)

    assert not folder.exists()


async def test_delete_threads_the_active_workspace_into_the_publish_key(
    session, tmp_path, org_mode, monkeypatch
):
    """Unpublishing mints its own credential (the viewer's, not a stored provider
    key), and that mint must carry the caller's picked workspace too — the same
    binding the publish and autopublish paths already carry."""
    from cowork.services.artifact_publish_key import PublishKey
    import cowork.services.artifact_autopublish as autopublish

    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="dash")
    monkeypatch.setattr(autopublish, "_active_workspace_id", lambda scope: "ws-1")

    captured = {}
    real_init = PublishKey.__init__

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        real_init(self, *args, **kwargs)

    async def fake_get(self):
        return "turnkey-1"

    async def fake_revoke(self):
        return None

    monkeypatch.setattr(PublishKey, "__init__", spy_init)
    monkeypatch.setattr(PublishKey, "get", fake_get)
    monkeypatch.setattr(PublishKey, "revoke", fake_revoke)

    await ep.delete_artifact_for_request(_scoped(session, ORG_A).scope, "dash", project_id=row.id)

    assert captured["workspace_id"] == "ws-1"


@pytest.mark.parametrize("stage", ["mint", "unpublish"])
@pytest.mark.parametrize("status", [403, 503])
async def test_delete_preserves_authority_failures_and_keeps_artifact_files(
    session, tmp_path, org_mode, monkeypatch, stage, status,
):
    import io
    from urllib.error import HTTPError
    from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable

    row, folder = _project_with_artifact(
        session, tmp_path, name="publish-denied", org_id=ORG_A, slug="dash"
    )
    record = {"index.html": {"report_id": "rid", "published": True}}
    (folder / ".published.json").write_text(json.dumps(record))
    error_type = ProductPermissionDenied if status == 403 else ProductPermissionUnavailable

    async def mint(**kwargs):
        if stage == "mint":
            raise error_type()
        return "artifact-key"

    async def revoke(*args, **kwargs):
        return True

    def unpublish(*args, **kwargs):
        assert stage == "unpublish", "Mint denial must stop before unpublishing"
        body = (
            {"code": "permission_denied"} if status == 403
            else {"error": "Auth unavailable", "code": "auth_unavailable"}
        )
        raise HTTPError("https://publish.example/delete/rid", status, "Rejected", {},
                        io.BytesIO(json.dumps(body).encode()))

    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", mint)
    monkeypatch.setattr("cowork.services.artifact_access.revoke_draft_review_access", revoke)
    monkeypatch.setattr("anton.publisher.unpublish", unpublish)
    with pytest.raises(error_type) as error:
        await ep.delete_artifact_for_request(_scoped(session, ORG_A).scope, "dash", project_id=row.id)
    assert error.value.status_code == status
    assert (folder / "index.html").read_text() == "<html></html>"
    assert json.loads((folder / ".published.json").read_text()) == record


async def test_delete_keeps_artifact_files_when_the_upload_service_is_unavailable(
    session, tmp_path, org_mode, monkeypatch,
):
    import io
    from urllib.error import HTTPError
    from cowork.services.publish import UploadServiceUnavailable

    row, folder = _project_with_artifact(
        session, tmp_path, name="publish-down", org_id=ORG_A, slug="dash"
    )
    record = {"index.html": {"report_id": "rid", "published": True}}
    (folder / ".published.json").write_text(json.dumps(record))

    async def mint(**kwargs):
        return "artifact-key"

    async def revoke(*args, **kwargs):
        return True

    def unpublish(*args, **kwargs):
        raise HTTPError("https://publish.example/delete/rid", 503, "Service Unavailable", {},
                        io.BytesIO(b"Service Unavailable"))

    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", mint)
    monkeypatch.setattr("cowork.services.artifact_access.revoke_draft_review_access", revoke)
    monkeypatch.setattr("anton.publisher.unpublish", unpublish)
    with pytest.raises(UploadServiceUnavailable):
        await ep.delete_artifact_for_request(_scoped(session, ORG_A).scope, "dash", project_id=row.id)
    assert (folder / "index.html").read_text() == "<html></html>"
    assert json.loads((folder / ".published.json").read_text()) == record


@pytest.mark.parametrize("by_id", [False, True])
async def test_delete_revokes_with_the_verified_owner_and_tenant_before_removing_files(
    session, tmp_path, org_mode, publish_key, monkeypatch, by_id
):
    row, folder = _project_with_artifact(
        session, tmp_path, name="owned", org_id=ORG_A, slug="dash"
    )
    artifact_id, _metadata = ensure_full_id(folder)
    calls = []

    async def revoke(identifier, scope):
        assert folder.exists()
        calls.append((identifier, scope.user_id, scope.org_id))
        return True

    monkeypatch.setattr("cowork.services.artifact_access.revoke_draft_review_access", revoke)

    await ep.delete_artifact_for_request(
        _scoped(session, ORG_A).scope, artifact_id if by_id else "dash", project_id=row.id
    )

    assert [(UUID(identifier).hex, owner, org) for identifier, owner, org in calls] == [
        (artifact_id, USER_A, ORG_A)
    ]
    assert not folder.exists()


@pytest.mark.parametrize("by_id", [False, True])
async def test_a_delete_past_its_draft_rule_revoke_is_never_refused_a_file_store_slot(
    session, tmp_path, org_mode, publish_key, monkeypatch, by_id
):
    """Once the draft review rule is gone, a busy file store must not refuse
    the delete: the folder would keep its review marker, which still lets
    co-members open the draft here, with no rule behind it. The lookup before
    the revoke still refuses a busy store before anything irreversible runs."""
    from cowork.db import units

    row, folder = _project_with_artifact(session, tmp_path, name="owned", org_id=ORG_A, slug="dash")
    artifact_id, _metadata = ensure_full_id(folder)

    async def revoke(identifier, scope):
        return True

    monkeypatch.setattr("cowork.services.artifact_access.revoke_draft_review_access", revoke)
    real_run_efs = units.run_efs
    delete_steps: list[tuple[str, bool]] = []

    async def recorded(fn, *, scope, deadline=True):
        delete_steps.append((getattr(fn, "func", fn).__name__, deadline))
        return await real_run_efs(fn, scope=scope, deadline=deadline)

    # The delete's own steps import run_efs when they run; the lookups bind it
    # at import, so only the delete's steps are recorded.
    monkeypatch.setattr(units, "run_efs", recorded)

    await ep.delete_artifact_for_request(
        _scoped(session, ORG_A).scope, artifact_id if by_id else "dash", project_id=row.id
    )

    assert delete_steps == [("pin", False), ("_finish_delete", False)]
    assert not folder.exists()


async def test_deletes_waiting_on_a_slow_publish_service_leave_their_member_file_store_slots(
    session, tmp_path, org_mode, publish_key, monkeypatch
):
    """A member deletes a share's worth of published artifacts while the
    publish service hangs. The take-downs wait on the network, not in file
    store calls, so that member's own artifact review still gets a slot."""
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import cowork.db.session as db_session
    from cowork.api.v1.artifact_scope import review_artifact_for_request

    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    share = db_session.settings.storage.efs_share
    deleting = []
    for n in range(share):
        row, folder = _project_with_artifact(session, tmp_path, name=f"published-{n}", org_id=ORG_A, slug="dash")
        (folder / ".published.json").write_text(
            json.dumps({"index.html": {"report_id": f"rid-{n}", "published": True}})
        )
        deleting.append((row, folder, ensure_full_id(folder)[0]))
    review_row, review_folder = _project_with_artifact(session, tmp_path, name="kept", org_id=ORG_A, slug="kept")
    review_id, _metadata = ensure_full_id(review_folder)
    publish_service = threading.Event()
    waiting = threading.Semaphore(0)
    taken_down: list[str] = []

    def unpublish(identifier, **kwargs):
        waiting.release()
        publish_service.wait(timeout=10)
        taken_down.append(identifier)

    async def revoke(identifier, scope):
        return True

    monkeypatch.setattr("anton.publisher.unpublish", unpublish)
    monkeypatch.setattr("cowork.services.artifact_access.revoke_draft_review_access", revoke)
    # The take-downs and this test's own waits run on the loop's default
    # executor, whose size follows the CPU count. Sized here, the whole share
    # hangs at once on any runner.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=share + 2))
    deletes = [
        asyncio.create_task(ep.delete_artifact_for_request(_scoped(session, ORG_A).scope, artifact_id, project_id=row.id))
        for row, _folder, artifact_id in deleting
    ]
    try:
        for _ in deleting:
            assert await asyncio.to_thread(waiting.acquire, True, 10)
        scope = TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A)
        resolved = await asyncio.wait_for(
            review_artifact_for_request(scope, str(review_row.id), review_id), timeout=5,
        )
        assert resolved.folder == review_folder
        assert taken_down == []
    finally:
        publish_service.set()
        await asyncio.gather(*deletes, return_exceptions=True)

    for task in deletes:
        assert task.exception() is None
    assert sorted(taken_down) == sorted(f"rid-{n}" for n in range(share))
    assert not any(folder.exists() for _row, folder, _id in deleting)


async def test_delete_keeps_files_when_draft_authorization_is_unavailable(
    session, tmp_path, org_mode, publish_key, monkeypatch
):
    from cowork.services.artifact_access import ArtifactAccessUnavailable

    row, folder = _project_with_artifact(
        session, tmp_path, name="owned", org_id=ORG_A, slug="dash"
    )
    artifact_id, _metadata = ensure_full_id(folder)

    async def revoke(identifier, scope):
        raise ArtifactAccessUnavailable("Could not revoke draft collaboration")

    monkeypatch.setattr("cowork.services.artifact_access.revoke_draft_review_access", revoke)

    with pytest.raises(HTTPException) as err:
        await ep.delete_artifact_for_request(
            _scoped(session, ORG_A).scope, artifact_id, project_id=row.id
        )

    assert err.value.status_code == 503
    assert (folder / "index.html").read_text() == "<html></html>"
    assert (folder / "metadata.json").exists()


async def test_delete_by_artifact_id_selects_the_exact_duplicate_slug(
    session, tmp_path, org_mode, publish_key
):
    row, first_folder = _project_with_artifact(
        session, tmp_path, name="mine", org_id=ORG_A, slug="same"
    )
    first_metadata = json.loads((first_folder / "metadata.json").read_text())
    first_metadata["id"] = "first-artifact"
    (first_folder / "metadata.json").write_text(json.dumps(first_metadata))
    second_conversation_id = uuid.uuid4()
    session.add(
        Conversation(
            id=second_conversation_id,
            topic="second task",
            project_id=row.id,
            org_id=ORG_A,
            created_by=USER_A,
        )
    )
    session.commit()
    second_folder = (
        Path(row.path)
        / "conversations"
        / str(second_conversation_id)
        / ".anton"
        / "artifacts"
        / "same"
    )
    second_folder.mkdir(parents=True)
    (second_folder / "index.html").write_text("<html>second</html>")
    (second_folder / "metadata.json").write_text(
        json.dumps({
            "id": "second-artifact",
            "slug": "same",
            "name": "same",
            "type": "html-app",
        })
    )
    second_artifact_id, _metadata = ensure_full_id(second_folder)

    await ep.delete_artifact_for_request(
        _scoped(session, ORG_A).scope, second_artifact_id, project_id=row.id
    )

    assert first_folder.exists()
    assert not second_folder.exists()


async def test_reviewer_cannot_delete_an_owners_artifact(
    session, tmp_path, org_mode, publish_key, monkeypatch
):
    """A granted reviewer sees the artifact, so the delete is refused as 403 —
    not hidden as a 404, which would read as "already gone" to a client that is
    looking at the draft."""
    row, folder, card = await _grant_draft_review(session, tmp_path, monkeypatch)

    async def revoke(identifier, scope):
        pytest.fail("A reviewer must never reach the policy mutation")

    monkeypatch.setattr("cowork.services.artifact_access.revoke_draft_review_access", revoke)

    with Session(session.get_bind()) as reviewer_session:
        with pytest.raises(HTTPException) as err:
            await ep.delete_artifact_for_request(
                _scoped(reviewer_session, ORG_A, user_id=USER_A2).scope,
                card["id"],
                project_id=row.id,
            )

    assert err.value.status_code == 403
    assert folder.exists()


async def test_ungranted_delete_by_a_co_member_is_404(
    session, tmp_path, org_mode, publish_key
):
    row, folder = _project_with_artifact(
        session, tmp_path, name="mine", org_id=ORG_A, slug="dash"
    )
    artifact_id, _metadata = ensure_full_id(folder)

    with pytest.raises(HTTPException) as err:
        await ep.delete_artifact_for_request(
            _scoped(session, ORG_A, user_id=USER_A2).scope, artifact_id, project_id=row.id
        )

    assert err.value.status_code == 404
    assert folder.exists()


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
            _scoped(session, ORG_A, USER_A2).scope, "theirs", project_id=row.id
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
        await ep.delete_artifact_for_request(_scoped(session, ORG_A).scope, "secret", project_id=row.id)

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
        ScopedSession(session, LOCAL_SCOPE).scope, "dash", project_id=second.id
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

async def test_desktop_project_id_filter_keeps_project_addressed_cards(
    session, tmp_path, local_mode
):
    row, _folder = _project_with_artifact(
        session, tmp_path, name="mine", org_id=None, slug="dash"
    )

    card = (await ep.list_artifacts(
        LOCAL_SCOPE, project_id=row.id, project_path=None
    ))[0]

    assert card["projectId"] == str(row.id)
    assert card["draftUrl"].startswith(
        f"/api/v1/artifacts/drafts/{row.id}/"
    )


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
        ep, "artifacts_sources_for_scan",
        lambda *, scope: [
            ep_artifacts.ProjectArtifacts(
                base=p / ".anton" / "artifacts", project_id=None, project_name=p.name,
            )
            for p in sorted((tmp_path / "local").iterdir())
        ],
    )

    cards = await ep.list_artifacts(
        LOCAL_SCOPE, project_id=None, project_path=str(mine_project_dir)
    )

    assert [c["slug"] for c in cards] == ["dash"]


async def test_desktop_project_path_that_matches_nothing_yields_nothing(
    session, tmp_path, local_mode, monkeypatch
):
    _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="dash")
    monkeypatch.setattr(
        ep, "artifacts_sources_for_scan",
        lambda *, scope: [
            ep_artifacts.ProjectArtifacts(
                base=(tmp_path / "local" / "mine") / ".anton" / "artifacts",
                project_id=None, project_name="mine",
            )
        ],
    )

    cards = await ep.list_artifacts(
        LOCAL_SCOPE, project_id=None, project_path="/nowhere/at/all"
    )

    assert cards == []


pytestmark = pytest.mark.usefixtures("granted_product_permissions")
