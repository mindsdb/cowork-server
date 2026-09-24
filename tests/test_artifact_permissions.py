"""Owner capabilities are a property of each artifact (ENG-2961 / ENG-2949)."""
from __future__ import annotations

import json
from contextlib import contextmanager
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from cowork.api.v1.endpoints import artifact_workspace, artifacts as artifacts_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.principal import ORG_MANAGE_ROLE, Principal
from cowork.services import artifact_ownership as ownership
from cowork.services.artifact_permissions import (
    artifact_capabilities,
    may_delete_ownerless_artifact,
    require_artifact_owner,
)
from test_artifact_ownership import (
    legacy_root,
    make_conversation,
    make_project,
    project_root,
    write_artifact,
)


@pytest.fixture(autouse=True)
def cleanup_test_projects(tmp_path):
    """Clean up projects created by tests to avoid database pollution.

    Mirrors `test_artifact_ownership.cleanup_test_projects`: autouse fixtures
    do not carry over through imports, and this file creates its own
    org_id=None Project row in `test_desktop_is_a_single_owner_boundary` that
    would otherwise leak into `tests/test_artifact_roots.py`.
    """
    from sqlmodel import select
    from cowork.models.project import Project

    yield

    with Session(get_engine(get_app_settings().database.uri)) as session:
        projects = session.exec(select(Project)).all()
        for project in projects:
            if tmp_path.as_posix() in project.path:
                session.delete(project)
        session.commit()


@contextmanager
def scoped(org_id, user_id):
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        yield ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id, user_id=user_id))


@pytest.fixture
def world(tmp_path):
    org_id, creator, member = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    with scoped(org_id, None) as session:
        ownership.record_artifact_owner(session, project.id, "mine", creator, action="create")
        ownership.record_artifact_owner(session, project.id, "theirs", member, action="create")
    return org_id, creator, member, project, source


def test_project_root_artifact_is_owned_by_its_creator(world):
    org_id, creator, member, _project, source = world
    with scoped(org_id, creator) as session:
        caps = artifact_capabilities(session, source, "mine")
        assert caps["role"] == "owner"
        assert caps["canEdit"] is True
        assert "ownerUnknown" not in caps
        assert require_artifact_owner(session, source, "mine")["canEdit"] is True


def test_each_member_owns_only_their_own_artifact(world):
    org_id, creator, member, _project, source = world
    with scoped(org_id, creator) as session:
        assert artifact_capabilities(session, source, "theirs")["role"] == "reviewer"
        with pytest.raises(HTTPException) as refused:
            require_artifact_owner(session, source, "theirs")
        assert refused.value.status_code == 403
        assert refused.value.detail == "Only the artifact owner can change this draft"
    with scoped(org_id, member) as session:
        assert artifact_capabilities(session, source, "theirs")["role"] == "owner"
        assert artifact_capabilities(session, source, "mine")["role"] == "reviewer"


def test_unknown_owner_is_a_named_state(world):
    org_id, creator, _member, _project, source = world
    with scoped(org_id, creator) as session:
        caps = artifact_capabilities(session, source, "orphan")
        assert caps == {
            "role": "reviewer",
            "canPreview": True,
            "canComment": True,
            "canEdit": False,
            "canAddressWithAgent": False,
            "canResolveComments": False,
            "ownerUnknown": True,
        }
        with pytest.raises(HTTPException) as refused:
            require_artifact_owner(session, source, "orphan")
        assert refused.value.detail == "Artifact owner is unknown"


def test_legacy_artifact_stays_owned_by_its_creator(tmp_path):
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = legacy_root(project, str(make_conversation(project, creator)))
    with scoped(org_id, creator) as session:
        assert artifact_capabilities(session, source, "old")["role"] == "owner"


def test_desktop_is_a_single_owner_boundary(tmp_path):
    project = make_project(tmp_path, None)
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        session = ScopedSession(raw, LOCAL_SCOPE)
        assert artifact_capabilities(session, project_root(project), "a")["canEdit"] is True


def _principal(org_id, user_id, *, admin):
    roles = frozenset({ORG_MANAGE_ROLE}) if admin else frozenset()
    return Principal(user_id=user_id, org_id=org_id, roles=roles)


def test_org_admin_is_a_reviewer_on_another_members_artifact(world):
    org_id, _creator, _member, _project, source = world
    admin = str(uuid4())
    with scoped(org_id, admin) as session:
        assert artifact_capabilities(session, source, "mine")["role"] == "reviewer"


def test_only_a_matching_admin_may_delete_an_ownerless_artifact(world):
    org_id, creator, _member, _project, source = world
    admin = str(uuid4())
    with scoped(org_id, admin) as session:
        principal = _principal(org_id, admin, admin=True)
        assert may_delete_ownerless_artifact(session, source, "orphan", principal) is True
        assert may_delete_ownerless_artifact(session, source, "mine", principal) is False
        assert may_delete_ownerless_artifact(
            session, source, "orphan", _principal(org_id, admin, admin=False)
        ) is False
        assert may_delete_ownerless_artifact(
            session, source, "orphan", _principal(org_id, creator, admin=True)
        ) is False
        assert may_delete_ownerless_artifact(session, source, "orphan", None) is False


def test_cards_carry_per_artifact_capabilities(world):
    org_id, creator, _member, _project, source = world
    for slug in ("mine", "theirs", "orphan"):
        write_artifact(source.base, slug)
    with scoped(org_id, creator) as session:
        cards = {c["slug"]: c for c in artifacts_ep._artifact_cards(session, [source])}
    assert cards["mine"]["capabilities"]["role"] == "owner"
    assert cards["theirs"]["capabilities"]["role"] == "reviewer"
    assert cards["orphan"]["capabilities"]["ownerUnknown"] is True


@pytest.fixture
def org_deployment(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.mark.asyncio
async def test_creator_can_open_the_workspace_of_a_project_root_artifact(
    world, org_deployment, granted_product_permissions
):
    """The ENG-2949 regression: this 403'd for every artifact since 2026-09-14."""
    org_id, creator, member, project, source = world
    local_id = uuid4().hex
    folder = source.base / "mine"
    folder.mkdir(exist_ok=True)
    (folder / "index.html").write_text("<html>mine</html>")
    (folder / "metadata.json").write_text(
        json.dumps({"id": local_id, "slug": "mine", "type": "html-app", "primary": "index.html"})
    )
    with scoped(org_id, creator) as session:
        result = await artifact_workspace.artifact_source(
            str(project.id), local_id, session, path=None
        )
        assert result["capabilities"]["role"] == "owner"
        assert result["capabilities"]["canEdit"] is True
    with scoped(org_id, member) as session:
        with pytest.raises(HTTPException) as refused:
            await artifact_workspace.artifact_source(str(project.id), local_id, session, path=None)
        assert refused.value.status_code == 403


from cowork.services import artifact_access
from cowork.services.artifact_authorization_identity import ArtifactIdentity, _session as identity_session


@pytest.fixture
def no_publish_side_effects(monkeypatch):
    async def mint(_self):
        return "test-publish-key"

    async def no_revoke(*_args, **_kwargs):
        return False

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey.get", mint)
    monkeypatch.setattr("anton.publisher.unpublish", lambda *_a, **_kw: None)
    monkeypatch.setattr(artifact_access, "revoke_draft_review_access", no_revoke)


def _write(source, slug):
    local_id = uuid4().hex
    folder = source.base / slug
    folder.mkdir(exist_ok=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(
        json.dumps({"id": local_id, "slug": slug, "type": "html-app", "primary": "index.html"})
    )
    return local_id, folder


@pytest.mark.asyncio
async def test_owner_delete_removes_the_owner_row(
    world, org_deployment, granted_product_permissions, no_publish_side_effects
):
    org_id, creator, _member, project, source = world
    local_id, folder = _write(source, "mine")
    with scoped(org_id, creator) as session:
        await artifacts_ep.delete_artifact_for_request(session, local_id, project_id=project.id)
    assert not folder.exists()
    with scoped(org_id, creator) as session:
        assert ownership.resolve_artifact_owner(session, source, "mine").unknown


@pytest.mark.asyncio
async def test_admin_may_delete_an_ownerless_artifact_but_not_an_owned_one(
    world, org_deployment, granted_product_permissions, no_publish_side_effects
):
    org_id, _creator, _member, project, source = world
    admin = str(uuid4())
    principal = _principal(org_id, admin, admin=True)
    orphan_id, orphan = _write(source, "orphan")
    owned_id, owned = _write(source, "mine")
    # An identity row naming someone else must not turn the admin delete into a 503.
    with identity_session(TenantScope(org_mode=True, org_id=org_id, user_id=admin)) as ids:
        ids.add(ArtifactIdentity(
            org_id=org_id,
            owner_keycloak_id=str(uuid4()),
            local_artifact_id=orphan_id,
            # Allocated, not pending: a pending alias refuses every read.
            canonical_artifact_id=f"artifact/{uuid4()}",
        ))
        ids.commit()
    with scoped(org_id, admin) as session:
        await artifacts_ep.delete_artifact_for_request(
            session, orphan_id, project_id=project.id, principal=principal
        )
        with pytest.raises(HTTPException) as refused:
            await artifacts_ep.delete_artifact_for_request(
                session, owned_id, project_id=project.id, principal=principal
            )
        assert refused.value.status_code == 403
    assert not orphan.exists()
    assert owned.exists()


@pytest.mark.asyncio
async def test_without_a_principal_there_is_no_admin_exception(
    world, org_deployment, granted_product_permissions, no_publish_side_effects
):
    org_id, _creator, _member, project, source = world
    orphan_id, orphan = _write(source, "orphan")
    with scoped(org_id, str(uuid4())) as session:
        with pytest.raises(HTTPException) as refused:
            await artifacts_ep.delete_artifact_for_request(session, orphan_id, project_id=project.id)
        assert refused.value.detail == "Artifact owner is unknown"
    assert orphan.exists()
