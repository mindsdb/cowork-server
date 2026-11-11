"""Real SQL aliases keep mutable metadata out of the global authorization key."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from threading import Barrier, Lock
import time
from typing import Literal
import urllib.error
from uuid import UUID, uuid4

import anton.publisher
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import httpx
from pydantic import BaseModel, Field
import pytest
from sqlmodel import Session, select
import yaml

from cowork.api.v1.endpoints import artifact_workspace, artifacts, comments
from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope, get_tenant_scope
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.schemas.artifact_access import ArtifactAccessRequest
from cowork.services import (
    artifact_access,
    artifact_authorization_identity as identities,
)
from cowork.services.artifact_access import ArtifactAccessUnavailable
from cowork.services.artifact_identity import artifact_key
from cowork.services.artifact_metadata import ArtifactMetadata
from cowork.services.publish import publish_artifact
from tests._org_sharing import TEAM_ELSEWHERE, TEAM_ENG, TEAM_OPS, TEAM_QA, member_request, mindshub_sharing_answers


class AuthIssuer:
    """HTTP boundary fake; records ownership, grants and issuance retries.

    It keeps each artifact's rule the way MindsHub does. Its upsert
    replaces a sharing field the caller sends and keeps one it leaves out,
    and echoes the org flag, teams and people; its read answers the rule as
    stored, or 404 ``no_rule``; its delete removes it. Every route reads an
    artifact id as MindsHub does, stripped, and answers 400 for a blank one,
    one holding a NUL, and one longer than MindsHub takes. ``lambda_publish`` is
    the publish lambda's own write of the emails and the org flag, and its
    delete of the rule for a public or password publish. A deleted
    team is dropped from every rule and refused from then on
    (``unknown_team``). A person who left the org stays on every rule and is
    refused on the next write that names them (``not_a_member``).
    """

    def __init__(self):
        self.calls = []
        self.bindings = {}
        self.allocations = {}
        self.failure = None
        self.drop_allocation_reply = False
        self.barrier = None
        self.lock = Lock()
        self.rules = {}
        self.deleted_teams = set()
        self.departed = set()

    def delete_team(self, team_id):
        self.deleted_teams.add(team_id)
        for rule in self.rules.values():
            rule["allowed_team_ids"] = [t for t in rule["allowed_team_ids"] if t != team_id]

    def leave_org(self, user_id):
        self.departed.add(user_id)

    def _rule(self, artifact_id):
        return self.rules.setdefault(artifact_id, {
            "org_allowed": False, "allowed_team_ids": [], "allowed_user_ids": [], "allowed_emails": [],
        })

    def lambda_publish(self, kwargs):
        """What the publish lambda writes from anton's access block: a
        restricted rule's emails and org flag, its teams and people kept.
        A public or password publish deletes the rule."""
        access = kwargs.get("access") or {}
        if access.get("mode") == "restricted":
            rule = self._rule(kwargs["artifact_key"])
            rule["allowed_emails"] = list(access.get("emails") or [])
            rule["org_allowed"] = bool(access.get("org_allowed"))
        else:
            self.rules.pop(kwargs["artifact_key"], None)

    def _upsert(self, body):
        unknown = [t for t in body.get("allowed_team_ids", []) if t in self.deleted_teams]
        if unknown:
            return httpx.Response(400, json={"code": "unknown_team", "detail": "x", "allowed_team_ids": unknown})
        outsiders = [u for u in body.get("allowed_user_ids", []) if u in self.departed]
        if outsiders:
            return httpx.Response(400, json={"code": "not_a_member", "detail": "x", "allowed_user_ids": outsiders})
        rule = self._rule(body["artifact_id"])
        for field in ("allowed_team_ids", "allowed_user_ids"):
            if field in body:
                rule[field] = list(body[field])
        return httpx.Response(200, json={
            "ok": True, **{k: rule[k] for k in ("org_allowed", "allowed_team_ids", "allowed_user_ids")},
        })

    def _read(self, body):
        rule = self.rules.get(body["artifact_id"])
        if rule is None:
            return httpx.Response(404, json={"code": "no_rule", "detail": "x"})
        return httpx.Response(200, json={"artifact_id": body["artifact_id"], **rule})

    def handle(self, request):
        body = json.loads(request.content)
        path = request.url.path
        self.calls.append((path, body))
        assert request.headers["X-Internal-Auth"] == "test-secret"
        if self.failure:
            return httpx.Response(503)
        owner = (body["owner_keycloak_id"], body["organization_id"])
        if path.endswith("/allocate/"):
            assert set(body) == {
                "owner_keycloak_id",
                "organization_id",
                "allocation_request_id",
            }
            nonce = body["allocation_request_id"]
            UUID(nonce)
            if self.barrier:
                self.barrier.wait(timeout=5)
            with self.lock:
                key = self.allocations.setdefault(nonce, artifact_key(str(uuid4())))
                self.bindings[key] = owner
            if self.drop_allocation_reply:
                self.drop_allocation_reply = False
                raise httpx.ReadTimeout("Allocation reply lost", request=request)
            return httpx.Response(200, json={"artifact_id": key})
        stripped = body["artifact_id"].strip()
        if not stripped or "\x00" in stripped or len(stripped) > 512:
            # MindsHub's serializers strip an artifact id, and refuse a blank
            # one, one holding a NUL, and one over 512 characters.
            return httpx.Response(400, json={"artifact_id": ["Not a valid artifact id."]})
        body = {**body, "artifact_id": stripped}
        key = body["artifact_id"].replace("artifact-draft/", "artifact/")
        expected = self.bindings.get(key)
        if expected is None and path.endswith("/read/") and body["artifact_id"] not in self.rules:
            # MindsHub's read answers an id nobody reserved and no rule sits
            # under as having no rule, like any other.
            return httpx.Response(404, json={"code": "no_rule", "detail": "x"})
        if expected is None:
            return httpx.Response(409, json={"code": "artifact_ownership_unclaimed"})
        if expected != owner:
            return httpx.Response(409, json={"code": "artifact_ownership_conflict"})
        if path.endswith("/claim/"):
            assert body["create_if_missing"] is False
        if path == "/v1/internal/artifact-access/":
            return self._upsert(body)
        if path.endswith("/read/"):
            return self._read(body)
        if path.endswith("/delete/"):
            self.rules.pop(body["artifact_id"], None)
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
def issuer(monkeypatch):
    fake = AuthIssuer()
    sync_client, async_client = httpx.Client, httpx.AsyncClient
    transport = httpx.MockTransport(fake.handle)
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: sync_client(transport=transport, **kw)
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: async_client(transport=transport, **kw)
    )
    settings = TurnQueueSettings(
        auth_internal_base_url="http://auth.test", auth_internal_secret="test-secret"
    )
    monkeypatch.setattr(identities, "TurnQueueSettings", lambda: settings)
    monkeypatch.setattr(artifact_access, "TurnQueueSettings", lambda: settings)
    return fake


@pytest.fixture
def scope():
    return TenantScope(org_mode=True, org_id=str(uuid4()), user_id=str(uuid4()))


@pytest.fixture
def local_id():
    return uuid4().hex


def _ensure(local_id, scope):
    return identities.ensure_authorization_key(
        local_id, scope, owner_user_id=scope.user_id
    )


def test_new_ids_are_issued_by_auth_and_reused_from_sql(issuer, scope, local_id):
    canonical = _ensure(local_id, scope)
    assert canonical != artifact_key(local_id)
    assert len(issuer.calls) == 2
    assert _ensure(local_id, scope) == canonical
    assert identities.existing_authorization_key(local_id, scope) == canonical
    assert len(issuer.calls) == 2
    with identities.authorization_session(scope) as session:
        row = identities._row(session, local_id)
        assert row.canonical_artifact_id == canonical
        assert row.owner_keycloak_id == scope.user_id
        assert issuer.calls[1][1]["allocation_request_id"] == str(row.id)


def test_existing_matching_binding_preserves_comment_identity(issuer, scope, local_id):
    issuer.bindings[artifact_key(local_id)] = (scope.user_id, scope.org_id)
    assert _ensure(local_id, scope) == artifact_key(local_id)
    assert len(issuer.calls) == 1
    assert issuer.allocations == {}


def test_foreign_global_binding_cannot_be_claimed_via_metadata(issuer, scope, local_id):
    original = (str(uuid4()), str(uuid4()))
    issuer.bindings[artifact_key(local_id)] = original
    with pytest.raises(ArtifactAccessUnavailable, match="Could not establish"):
        _ensure(local_id, scope)
    assert issuer.bindings[artifact_key(local_id)] == original
    assert issuer.allocations == {}
    assert len(issuer.calls) == 1


def test_same_local_id_in_different_organizations_gets_different_global_ids(
    issuer, scope, local_id
):
    first = _ensure(local_id, scope)
    other_scope = TenantScope(org_mode=True, org_id=str(uuid4()), user_id=scope.user_id)
    second = _ensure(local_id, other_scope)
    assert first != second
    assert identities.existing_authorization_key(local_id, scope) == first
    assert identities.existing_authorization_key(local_id, other_scope) == second


def test_copied_local_id_cannot_replace_a_same_org_owners_mapping(
    issuer, scope, local_id
):
    original = _ensure(local_id, scope)
    peer = TenantScope(org_mode=True, org_id=scope.org_id, user_id=str(uuid4()))
    before = list(issuer.calls)
    with pytest.raises(ArtifactAccessUnavailable, match="another owner"):
        _ensure(local_id, peer)
    with pytest.raises(ArtifactAccessUnavailable, match="another owner"):
        identities.existing_authorization_key(
            local_id, peer, owner_user_id=peer.user_id
        )
    assert issuer.calls == before
    assert identities.existing_authorization_key(local_id, scope) == original


def test_identity_reads_never_allocate(issuer, scope, local_id):
    assert identities.existing_authorization_key(local_id, scope) is None
    assert issuer.calls == []
    with identities.authorization_session(scope) as session:
        assert identities._row(session, local_id) is None


def test_unavailable_auth_preserves_pending_nonce_for_retry(issuer, scope, local_id):
    issuer.failure = True
    with pytest.raises(ArtifactAccessUnavailable):
        _ensure(local_id, scope)
    with identities.authorization_session(scope) as session:
        nonce = identities._row(session, local_id).id
    with pytest.raises(ArtifactAccessUnavailable, match="pending"):
        identities.existing_authorization_key(local_id, scope)
    issuer.failure = None
    canonical = _ensure(local_id, scope)
    assert issuer.allocations[str(nonce)] == canonical


def test_concurrent_first_use_reuses_one_durable_allocation_nonce(
    issuer, scope, local_id
):
    issuer.barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: _ensure(local_id, scope), range(2)))
    assert results[0] == results[1]
    assert len(issuer.allocations) == 1


def test_lost_allocation_reply_retries_the_same_issued_identity(
    issuer, scope, local_id
):
    issuer.drop_allocation_reply = True
    with pytest.raises(ArtifactAccessUnavailable):
        _ensure(local_id, scope)
    assert len(issuer.allocations) == 1
    already_issued = next(iter(issuer.allocations.values()))
    assert _ensure(local_id, scope) == already_issued
    assert len(issuer.allocations) == 1


@pytest.fixture
def owned_artifact(tmp_path, monkeypatch, scope, local_id):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        session = ScopedSession(raw, scope)
        project = Project(
            name="Owned project", path=str(tmp_path / "project"), org_id=scope.org_id
        )
        session.add(project)
        session.commit()
        conversation = Conversation(
            topic="Owner task",
            project_id=project.id,
            org_id=scope.org_id,
            created_by=scope.user_id,
        )
        session.add(conversation)
        session.commit()
        folder = (
            Path(project.path)
            / "conversations"
            / str(conversation.id)
            / ".anton"
            / "artifacts"
            / "demo"
        )
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text(
            json.dumps({"id": local_id, "slug": "demo", "type": "html-app"})
        )
        (folder / "index.html").write_text("<html>Owner bytes</html>")
        yield session, project, conversation, folder
    get_app_settings.cache_clear()


@pytest.mark.asyncio
async def test_draft_grant_publish_proxy_and_delete_use_one_server_identity(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    session, project, conversation, folder = owned_artifact
    granted = await artifact_workspace.enable_artifact_comments(
        str(project.id), local_id, session.scope
    )
    canonical = identities.existing_authorization_key(local_id, scope)
    assert granted.artifact_key == artifact_key(local_id)
    assert issuer.calls[-1][1]["artifact_id"] == canonical.replace(
        "artifact/", "artifact-draft/"
    )
    seen = {}

    def publish(_source, **kwargs):
        seen.update(kwargs)
        return {
            "artifact_key": kwargs["artifact_key"],
            "report_id": "report-1",
            "view_url": "https://view.test/1",
        }

    monkeypatch.setattr("anton.publisher.publish", publish)
    publish_artifact(
        folder,
        artifacts_base=folder.parent,
        api_key="test-publish-key",
        publish_url="https://view.test",
        scope=scope,
        project_id=str(project.id),
    )
    assert seen["artifact_key"] == canonical
    assert comments.resolve_comments_route(
        "artifact", local_id, tenant_scope=session.scope
    ) == tuple(canonical.split("/", 1))

    async def mint(_self):
        return "test-publish-key"

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey.get", mint)
    monkeypatch.setattr("anton.publisher.unpublish", lambda *_a, **_kw: None)
    await artifacts.delete_artifact_for_request(
        session.scope, local_id, project_id=project.id
    )
    assert issuer.calls[-1][0].endswith("/delete/")
    assert issuer.calls[-1][1] == {
        "artifact_id": canonical.replace("artifact/", "artifact-draft/"),
        "owner_keycloak_id": scope.user_id,
        "organization_id": scope.org_id,
    }
    assert not folder.exists()
    assert identities.existing_authorization_key(local_id, scope) == canonical


@pytest.mark.asyncio
async def test_an_authorization_outage_still_answers_503(issuer, local_id, owned_artifact):
    """An auth service that is set up but failing stays a 503, so the outage
    still shows as one."""
    session, project, _conversation, folder = owned_artifact
    issuer.failure = True

    with pytest.raises(HTTPException) as err:
        await artifact_workspace.enable_artifact_comments(str(project.id), local_id, session.scope)

    assert err.value.status_code == 503
    assert not (folder / ".revisions" / "draft-review.json").exists()


def test_publish_refuses_another_conversation_owner_before_allocating(
    issuer, scope, local_id, owned_artifact
):
    _session, project, _conversation, folder = owned_artifact
    peer = TenantScope(org_mode=True, org_id=scope.org_id, user_id=str(uuid4()))
    # The peer cannot even see the creator's private legacy root.
    with pytest.raises(ArtifactAccessUnavailable, match="does not belong|Only the artifact owner"):
        identities.publish_authorization_key(local_id, folder.parent, "demo", project.id, peer)
    assert issuer.calls == []


@pytest.mark.asyncio
async def test_live_editor_sync_preserves_canonical_identity_and_refuses_peer(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    from cowork.services.artifact_revisions import current_source

    session, project, _conversation, folder = owned_artifact
    await artifact_workspace.enable_artifact_comments(
        str(project.id), local_id, session.scope
    )
    canonical = identities.existing_authorization_key(local_id, scope)
    uploads = []

    def publish(source, **kwargs):
        uploads.append((Path(source).read_text(), kwargs))
        issuer.lambda_publish(kwargs)
        return {
            "artifact_key": kwargs["artifact_key"],
            "report_id": "report-1",
            "view_url": "https://view.test/1",
        }

    async def mint(_self):
        return "test-publish-key"

    async def revoke(_self):
        pass

    monkeypatch.setattr("anton.publisher.publish", publish)
    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey.get", mint)
    monkeypatch.setattr(
        "cowork.services.artifact_publish_key.PublishKey.revoke", revoke
    )
    publish_artifact(
        folder,
        artifacts_base=folder.parent,
        api_key="test-publish-key",
        publish_url="https://view.test",
        access=ArtifactAccessRequest(mode="restricted", allowed_emails=["reviewer@example.com"]),
        scope=scope,
        project_id=str(project.id),
    )
    metadata = ArtifactMetadata.parse((folder / "metadata.json").read_text())
    initial = current_source(folder, metadata, local_id)
    body = artifact_workspace._SourceUpdateBody(
        content="<html>Updated owner bytes</html>",
        expectedRevisionId=initial["revision"]["id"],
        path="index.html",
    )
    peer = TenantScope(org_mode=True, org_id=scope.org_id, user_id=str(uuid4()))
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        peer_session = ScopedSession(raw, peer)
        with pytest.raises(HTTPException) as excinfo:
            await artifact_workspace.update_artifact_source(
                str(project.id), local_id, body, peer_session.scope
            )
        assert excinfo.value.status_code == 403
    assert len(uploads) == 1
    authority_calls = len(issuer.calls)

    saved = await artifact_workspace.update_artifact_source(
        str(project.id), local_id, body, session.scope
    )
    assert saved["content"] == body.content
    assert len(uploads) == 2
    assert uploads[-1][0] == body.content
    assert uploads[0][1]["artifact_key"] == uploads[1][1]["artifact_key"] == canonical
    assert uploads[0][1]["access"] == uploads[1][1]["access"]
    assert uploads[1][1]["report_id"] == "report-1"
    # The sync's one call to the authority reads who the artifact is shared
    # with, from the rule the first publish stored.
    assert [(path, body["artifact_id"]) for path, body in issuer.calls[authority_calls:]] == [
        ("/v1/internal/artifact-access/read/", canonical),
    ]


def test_first_publish_allocates_before_upload_without_requiring_a_draft_grant(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    _session, project, _conversation, folder = owned_artifact
    uploads = []

    def publish(_source, **kwargs):
        key = kwargs["artifact_key"]
        assert issuer.bindings[key] == (scope.user_id, scope.org_id)
        uploads.append(key)
        return {
            "artifact_key": key,
            "report_id": "report-1",
            "view_url": "https://view.test/1",
        }

    monkeypatch.setattr("anton.publisher.publish", publish)
    result = publish_artifact(
        folder,
        artifacts_base=folder.parent,
        api_key="key",
        publish_url="https://view.test",
        scope=scope,
        project_id=str(project.id),
    )
    assert uploads == [identities.existing_authorization_key(local_id, scope)]
    assert result["artifactKey"] == uploads[0]
    assert uploads[0] != artifact_key(local_id)
    # A publish that isn't a person's save and isn't restricted asks first
    # whether the artifact has a rule (AccessModeChanged); it has none yet.
    assert [path for path, _body in issuer.calls] == [
        "/v1/internal/artifact-access/claim/",
        "/v1/internal/artifact-access/allocate/",
        "/v1/internal/artifact-access/read/",
    ]


@pytest.mark.parametrize("suffix", ["threads", "stream"])
def test_pending_allocation_cannot_fall_back_to_an_untrusted_global_id(
    issuer, scope, local_id, monkeypatch, suffix
):
    issuer.failure = True
    with pytest.raises(ArtifactAccessUnavailable):
        _ensure(local_id, scope)
    before = list(issuer.calls)
    app = FastAPI()
    app.include_router(comments.router, prefix="/comments")
    app.dependency_overrides[get_tenant_scope] = lambda: scope
    monkeypatch.setattr(comments, "_org_mode", lambda: True)

    async def forwarded(*_args):
        pytest.fail("Pending local alias must not forward its metadata id")

    monkeypatch.setattr(comments, "forward_comments_rest", forwarded)
    monkeypatch.setattr(comments, "forward_comments_stream", forwarded)
    with TestClient(app) as client:
        assert client.get(f"/comments/artifact/{local_id}/{suffix}").status_code == 503
    assert issuer.calls == before


@pytest.mark.parametrize("suffix", ["threads", "stream"])
def test_rest_and_stream_translate_sql_aliases_without_issuing_identity(
    issuer, scope, local_id, monkeypatch, suffix
):
    canonical = _ensure(local_id, scope)
    before = list(issuer.calls)
    app = FastAPI()
    app.include_router(comments.router, prefix="/comments")
    app.dependency_overrides[get_tenant_scope] = lambda: scope
    monkeypatch.setattr(comments, "_org_mode", lambda: True)
    seen = []

    async def forwarded(request, user_dir, report_id, *rest):
        from starlette.responses import JSONResponse

        seen.append((user_dir, report_id, request.headers.get("Authorization")))
        return JSONResponse({"ok": True})

    monkeypatch.setattr(comments, "forward_comments_rest", forwarded)
    monkeypatch.setattr(comments, "forward_comments_stream", forwarded)
    with TestClient(app) as client:
        response = client.get(
            f"/comments/artifact/{local_id}/{suffix}",
            headers={"Authorization": "Bearer caller-token"},
        )
        assert response.status_code == 200
    assert seen == [(*canonical.split("/", 1), "Bearer caller-token")]
    assert issuer.calls == before


def test_foreign_org_proxy_cannot_select_another_tenants_sql_alias(
    issuer, scope, local_id, monkeypatch
):
    canonical = _ensure(local_id, scope)
    foreign = TenantScope(org_mode=True, org_id=str(uuid4()), user_id=str(uuid4()))
    monkeypatch.setattr(comments, "_org_mode", lambda: True)
    assert comments.resolve_comments_route(
        "artifact", local_id, tenant_scope=foreign
    ) == ("artifact", local_id)
    assert canonical != artifact_key(local_id)
    assert len(issuer.calls) == 2


def test_unknown_external_canonical_key_remains_subject_to_upstream_authorization(
    issuer, scope, monkeypatch
):
    external = str(uuid4())
    monkeypatch.setattr(comments, "_org_mode", lambda: True)
    assert comments.resolve_comments_route(
        "artifact", external, tenant_scope=scope
    ) == ("artifact", external)
    assert issuer.calls == []


def test_identity_migration_preserves_aliases_across_application_rollback():
    from importlib import import_module
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect

    migration = import_module(
        "cowork.db.alembic.versions.e2262a14c001_artifact_identities"
    )
    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            with Operations.context(MigrationContext.configure(connection)):
                migration.upgrade()
                inspector = inspect(connection)
                assert "artifact_identities" in inspector.get_table_names()
                constraints = inspector.get_unique_constraints("artifact_identities")
                assert any(
                    item["column_names"] == ["org_id", "local_artifact_id"]
                    for item in constraints
                )
                migration.downgrade()
                assert "artifact_identities" in inspect(connection).get_table_names()
                migration.upgrade()
                assert "artifact_identities" in inspect(connection).get_table_names()
    finally:
        engine.dispose()


@pytest.fixture
def project_root_artifact(tmp_path, monkeypatch, scope, local_id):
    """The ENG-2056 layout: the artifact sits in the shared project root.

    Unlike `owned_artifact` (a legacy per-conversation root nested under
    `conversations/<uuid>/`), this fixture creates `<project.path>/.anton/
    artifacts/` directly — exactly the shape `_sources_outside_the_projects_root`
    treats as an adopted desktop project. The test DB is session-scoped (see
    `tests/conftest.py`), so without teardown this `Project` row and its
    on-disk folder would keep being listed by every later LOCAL_SCOPE query in
    the same pytest run (e.g. `tests/test_artifact_roots.py`).
    """
    from cowork.models.artifact_identity import ArtifactIdentity
    from cowork.models.shared_resource import SharedResourceAttribution
    from cowork.services.artifact_ownership import record_artifact_owner

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        session = ScopedSession(raw, scope)
        project = Project(name="Shared project", path=str(tmp_path / "shared"), org_id=scope.org_id)
        session.add(project)
        session.commit()
        folder = Path(project.path) / ".anton" / "artifacts" / "demo"
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text(
            json.dumps({"id": local_id, "slug": "demo", "type": "html-app"})
        )
        (folder / "index.html").write_text("<html>Owner bytes</html>")
        try:
            yield session, project, folder, record_artifact_owner
        finally:
            with Session(get_engine(get_app_settings().database.uri)) as cleanup:
                for row in cleanup.exec(
                    select(SharedResourceAttribution).where(
                        SharedResourceAttribution.org_id == scope.org_id,
                        SharedResourceAttribution.resource_key.startswith(f"{project.id}/"),
                    )
                ).all():
                    cleanup.delete(row)
                for row in cleanup.exec(
                    select(ArtifactIdentity).where(
                        ArtifactIdentity.org_id == scope.org_id,
                        ArtifactIdentity.local_artifact_id == UUID(local_id).hex,
                    )
                ).all():
                    cleanup.delete(row)
                stale_project = cleanup.get(Project, project.id)
                if stale_project is not None:
                    cleanup.delete(stale_project)
                cleanup.commit()
    get_app_settings.cache_clear()


def test_project_root_publish_uses_the_recorded_owner(
    issuer, scope, local_id, project_root_artifact
):
    session, project, folder, record = project_root_artifact
    record(session, project.id, "demo", scope.user_id, action="create")
    key = identities.publish_authorization_key(local_id, folder.parent, "demo", project.id, scope)
    assert key == identities.existing_authorization_key(local_id, scope)


def test_project_root_publish_without_an_owner_is_refused(
    issuer, scope, local_id, project_root_artifact
):
    from cowork.services.artifact_ownership import ArtifactOwnerUnknown

    _session, project, folder, _record = project_root_artifact
    with pytest.raises(ArtifactOwnerUnknown):
        identities.publish_authorization_key(local_id, folder.parent, "demo", project.id, scope)
    assert issuer.calls == []


def test_identity_bound_to_another_owner_is_refused(
    issuer, scope, local_id, project_root_artifact
):
    session, project, folder, record = project_root_artifact
    record(session, project.id, "demo", scope.user_id, action="create")
    other = str(uuid4())
    with identities.authorization_session(scope) as identity_session:
        identity_session.add(
            identities.ArtifactIdentity(
                org_id=scope.org_id, owner_keycloak_id=other, local_artifact_id=local_id
            )
        )
        identity_session.commit()
    with pytest.raises(ArtifactAccessUnavailable, match="bound to a different owner"):
        identities.publish_authorization_key(local_id, folder.parent, "demo", project.id, scope)


def test_project_root_publish_does_not_rediscover_roots(
    issuer, scope, local_id, project_root_artifact, monkeypatch
):
    from cowork.services import artifact_roots

    session, project, folder, record = project_root_artifact
    record(session, project.id, "demo", scope.user_id, action="create")

    def _no_discovery(*_args, **_kwargs):
        raise AssertionError("the project root must not trigger root discovery")

    monkeypatch.setattr(artifact_roots, "artifacts_sources_for_project", _no_discovery)
    assert identities.publish_authorization_key(
        local_id, folder.parent, "demo", project.id, scope
    )


def test_publish_from_a_root_outside_the_project_is_refused(
    issuer, scope, local_id, project_root_artifact, tmp_path
):
    session, project, _folder, record = project_root_artifact
    record(session, project.id, "demo", scope.user_id, action="create")
    stray = tmp_path / "elsewhere" / ".anton" / "artifacts"
    (stray / "demo").mkdir(parents=True)
    with pytest.raises(ArtifactAccessUnavailable, match="does not belong to its project"):
        identities.publish_authorization_key(local_id, stray, "demo", project.id, scope)
    assert issuer.calls == []


def test_publish_for_a_project_outside_the_scope_is_refused(
    issuer, scope, local_id, project_root_artifact
):
    _session, project, folder, _record = project_root_artifact
    other = TenantScope(org_mode=True, org_id=str(uuid4()), user_id=scope.user_id)
    with pytest.raises(ArtifactAccessUnavailable, match="could not be resolved"):
        identities.publish_authorization_key(local_id, folder.parent, "demo", project.id, other)
    with pytest.raises(ArtifactAccessUnavailable, match="could not be resolved"):
        identities.publish_authorization_key(local_id, folder.parent, "demo", "not-a-uuid", scope)
    assert issuer.calls == []


def test_publish_artifact_requires_project_id_in_org_mode(scope, project_root_artifact):
    _session, _project, folder, _record = project_root_artifact
    with pytest.raises(ValueError, match="requires project_id"):
        publish_artifact(
            folder, artifacts_base=folder.parent, api_key="k",
            publish_url="https://view.test", scope=scope,
        )


pytestmark = pytest.mark.usefixtures("granted_product_permissions")


def test_org_publish_reports_the_lambda_echo_when_it_differs(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    # Org comment routing never reads the record, so the echo is the only way
    # the renderer reaches the scope the published page uses.
    _session, project, _conversation, folder = owned_artifact
    echoed = f"artifact/{uuid4()}"
    monkeypatch.setattr(
        "anton.publisher.publish",
        lambda _source, **_kwargs: {
            "artifact_key": echoed,
            "report_id": "report-1",
            "view_url": "https://view.test/1",
        },
    )
    result = publish_artifact(
        folder,
        artifacts_base=folder.parent,
        api_key="key",
        publish_url="https://view.test",
        scope=scope,
        project_id=str(project.id),
    )
    assert result["artifactKey"] == echoed


def _org_publisher(
    issuer, folder, scope, project, monkeypatch, *, view_url="https://view.test/1", echoed_key=None,
    publish_url="https://view.test",
):
    """``(run, uploads)``: run(access, **kwargs) publishes ``folder`` in org
    mode to ``publish_url`` through anton's publisher, faked as the publish
    lambda (which writes the rule's emails and org flag), and ``uploads``
    lists what it was given. The lambda answers ``view_url``, and
    ``echoed_key`` in place of the key it was given when set."""
    uploads = []

    def publish(_source, **kwargs):
        uploads.append(kwargs)
        issuer.lambda_publish(kwargs)
        return {"artifact_key": echoed_key or kwargs["artifact_key"], "report_id": "report-1", "view_url": view_url}

    monkeypatch.setattr("anton.publisher.publish", publish)
    monkeypatch.setattr("cowork.services.artifact_autopublish._active_workspace_id", lambda scope: None)

    def run(access, **kwargs):
        if not isinstance(access, ArtifactAccessRequest):
            # The access as anton's publisher names it, teams and people under
            # MindsHub's names.
            access = ArtifactAccessRequest.of_publisher(access)
        return publish_artifact(
            folder, artifacts_base=folder.parent, api_key="key", publish_url=publish_url,
            scope=scope, project_id=str(project.id), access=access, **kwargs,
        )

    return run, uploads


def _upserts(issuer):
    return [body for path, body in issuer.calls if path == "/v1/internal/artifact-access/"]


def _entry(folder):
    return json.loads((folder / ".published.json").read_text())["index.html"]


@pytest.mark.asyncio
async def test_get_access_on_an_unpublished_artifact_asks_mindshub_nothing(
    issuer, scope, local_id, owned_artifact
):
    session, project, _conversation, _folder = owned_artifact
    state = (await artifact_workspace.artifact_access(str(project.id), local_id, session.scope)).model_dump(
        by_alias=True, mode="json"
    )
    assert state["accessMode"] == "public"
    assert issuer.calls == []


def _plant_entry(folder, **fields):
    """Overwrite ``fields`` of the artifact's .published.json entry, as an
    agent in the pod can: the folder is mounted writable."""
    published = json.loads((folder / ".published.json").read_text())
    published["index.html"].update(fields)
    (folder / ".published.json").write_text(json.dumps(published))


async def _live_sync(folder, scope, project, monkeypatch):
    """A content save's re-publish of ``folder`` (live sync,
    artifact_workspace._sync_live_artifact), with a turn key that needs no
    minting."""

    class _Key:
        async def get(self):
            return "turn-key"

        async def revoke(self):
            pass

    monkeypatch.setattr(
        artifact_workspace, "_owner_publish_context",
        lambda _session, _folder: (folder.parent, "https://view.test", _Key()),
    )
    return await artifact_workspace._sync_live_artifact(
        scope, folder, project_id=str(project.id)
    )


# What a MindsHub-mode entry never keeps: who the artifact is shared with.
_AUDIENCE_KEYS = {"emails", "org_allowed", "owner_only", "allowed_team_ids", "allowed_user_ids"}


def test_a_team_audience_is_published_owner_only_and_widened_by_mindshub(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """anton's publisher knows no teams, so a restricted publish to
    teams alone reaches it owner-only, and cowork then writes the rule's
    ``allowed_team_ids`` alone, under the ArtifactAccess names; the emails and
    the org flag are the lambda's. The entry keeps no audience. A republish
    reads the rule, sends its org flag, writes nothing and keeps the version;
    changing the teams bumps it."""
    from cowork.services.publish import published_artifact_access

    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)

    result = run({"mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": [TEAM_ENG]})
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": [], "org_allowed": False}
    path, body = issuer.calls[-1]
    assert path == "/v1/internal/artifact-access/"
    assert body["allowed_team_ids"] == [TEAM_ENG]
    assert not {"team_ids", "org_wide", "org_allowed", "allowed_emails", "allowed_user_ids"} & set(body)
    assert body["artifact_id"] == result["artifactKey"]
    first_version = body["access_version"]
    assert (result["allowedTeamIds"], result["orgAllowed"]) == ([TEAM_ENG], False)
    assert not _AUDIENCE_KEYS & set(_entry(folder))

    writes = len(_upserts(issuer))
    run(published_artifact_access(folder, artifacts_base=folder.parent, org_mode=True))
    assert len(_upserts(issuer)) == writes
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": [], "org_allowed": False}
    assert _entry(folder)["access_version"] == first_version

    result = run({"mode": "restricted", "emails": [], "org_allowed": True, "allowed_team_ids": [TEAM_OPS]})
    body = issuer.calls[-1][1]
    assert body["allowed_team_ids"] == [TEAM_OPS]
    assert body["access_version"] > first_version
    assert (result["allowedTeamIds"], result["orgAllowed"]) == ([TEAM_OPS], True)
    run(published_artifact_access(folder, artifacts_base=folder.parent, org_mode=True))
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": [], "org_allowed": True}


@pytest.mark.asyncio
async def test_a_live_sync_of_a_planted_entry_sends_what_the_rule_holds(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """.published.json sits in a folder the pod can write. An agent
    that writes everyone, an outside address and a team into the entry gets
    none of it published: a content save re-publishes the artifact (live
    sync) with the emails and the org flag MindsHub's rule holds,
    and writes no team."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False})
    _plant_entry(folder, emails=["mal@else.com"], org_allowed=True, owner_only=False, allowed_team_ids=[TEAM_ELSEWHERE])
    writes = len(_upserts(issuer))

    saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is True
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}
    assert len(_upserts(issuer)) == writes
    assert not _AUDIENCE_KEYS & set(_entry(folder))


_MODE_CHANGED = (
    {"mode": "public"},
    {"mode": "password", "access_password": "agent-chosen", "requires_password": True},
)


def _mode_changed_records(caplog):
    return [
        r for r in caplog.records
        if r.name == "cowork.services.publish"
        and r.levelno == logging.WARNING
        and getattr(r, "event", None) == "artifact_access_mode_changed"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("planted", _MODE_CHANGED)
async def test_a_live_sync_never_publishes_over_a_restricted_rule_after_an_agent_changes_the_mode(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog, planted
):
    """.published.json sits in a folder the pod can write. An agent
    that changes a restricted artifact's entry to public, or to a password it
    chose, gets nothing published: in MindsHub mode only a restricted
    publish leaves a rule, and the owner's last save left one, so the entry's
    mode changed after that save. The live sync is refused and logged before
    anything is uploaded, and the rule stays as the owner saved it."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    uploaded = len(uploads)
    _plant_entry(folder, **planted)

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is False
    assert len(uploads) == uploaded
    assert issuer.rules[key]["allowed_emails"] == ["a@else.com"]
    [record] = _mode_changed_records(caplog)
    assert record.artifact_slug == folder.name


@pytest.mark.asyncio
@pytest.mark.parametrize("planted", _MODE_CHANGED)
async def test_an_autopublish_never_publishes_over_a_restricted_rule_after_an_agent_changes_the_mode(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog, planted
):
    """The same entry, re-published by autopublish after a turn changed the
    artifact: refused before anything is uploaded."""
    from cowork.services import artifact_autopublish as ap

    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "restricted", "emails": [], "org_allowed": True}, owner_save=True)
    uploaded = len(uploads)
    _plant_entry(folder, **planted)
    decision = ap.PublishDecision("changed", published_entry=_entry(folder))

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        published = await ap._publish_one(
            folder.parent, folder.name, "key", "https://view.test", 30.0, scope,
            ap._access_for(decision), project_id=str(project.id),
        )

    assert published is False
    assert len(uploads) == uploaded
    assert len(_mode_changed_records(caplog)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("blanked", "reason"), (
    ({"artifact_key": ""}, "rule_held"),
    ({"artifact_key": "", "lambda_artifact_key": ""}, "no_rule_key"),
))
async def test_an_autopublish_with_the_metadata_removed_and_the_entry_key_blanked_publishes_nothing(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog, blanked, reason
):
    """The pod can write metadata.json too. An agent that deletes it, so the
    artifact has no identity to read the rule under, and blanks the key the
    entry recorded beside a public mode gets nothing published: the rule is
    read under the lambda's key the entry still records, and with none left
    a re-publish of the page can't be checked, so it is refused as well."""
    from cowork.services import artifact_autopublish as ap

    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    uploaded = len(uploads)
    (folder / "metadata.json").unlink()
    _plant_entry(folder, mode="public", published_mtime=0, last_md5="x", **blanked)
    decision = ap.PublishDecision("changed", published_entry=_entry(folder))

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        published = await ap._publish_one(
            folder.parent, folder.name, "key", "https://view.test", 30.0, scope,
            ap._access_for(decision), project_id=str(project.id),
        )

    assert published is False
    assert len(uploads) == uploaded
    assert issuer.rules[key]["allowed_emails"] == ["a@else.com"]
    [record] = _mode_changed_records(caplog)
    assert record.reason == reason


@pytest.mark.asyncio
async def test_a_live_sync_after_an_agent_rewrites_the_metadata_id_publishes_nothing(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog
):
    """An agent that gives the artifact a fresh id in metadata.json moves the
    identity the rule is read under to a key that holds none. The rule the
    owner saved stays under the key the entry recorded, which is read too,
    so a public mode planted beside it publishes nothing."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    uploaded = len(uploads)
    (folder / "metadata.json").write_text(json.dumps({"id": uuid4().hex, "slug": "demo", "type": "html-app"}))
    _plant_entry(folder, mode="public")

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is False
    assert len(uploads) == uploaded
    assert issuer.rules[key]["allowed_emails"] == ["a@else.com"]
    [record] = _mode_changed_records(caplog)
    assert record.reason == "rule_held"


@pytest.mark.asyncio
@pytest.mark.parametrize("planted", _MODE_CHANGED)
async def test_the_access_route_answers_the_rule_an_agent_changed_the_mode_over(
    issuer, scope, local_id, owned_artifact, monkeypatch, planted
):
    """GET /access pre-fills the owner's Share dialog. An agent that changes
    a restricted artifact's entry to public, or to a password it chose,
    changes nothing published, since every re-publish is refused while the
    rule is held. So the route answers what the rule holds: restricted, to
    the audience the owner saved, with no password."""
    session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": True, "allowed_team_ids": [TEAM_ENG]},
        owner_save=True)
    _plant_entry(folder, **planted)

    state = (await artifact_workspace.artifact_access(str(project.id), local_id, session.scope)).model_dump(
        by_alias=True, mode="json"
    )

    assert (state["accessMode"], state["accessProtected"], state["accessPassword"]) == ("restricted", False, "")
    assert (state["accessEmails"], state["orgAllowed"], state["ownerOnly"], state["allowedTeamIds"]) == (
        ["a@else.com"], True, False, [TEAM_ENG],
    )


@pytest.mark.asyncio
async def test_the_access_route_reads_the_rule_under_the_key_the_entry_recorded(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """An agent that also gives the artifact a fresh id in metadata.json
    moves its identity to a key that holds no rule. The route reads the keys
    the entry recorded too, as a re-publish does, so it still answers the
    rule the owner saved."""
    session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)
    fresh = uuid4().hex
    (folder / "metadata.json").write_text(json.dumps({"id": fresh, "slug": "demo", "type": "html-app"}))
    _plant_entry(folder, mode="public")

    state = (await artifact_workspace.artifact_access(str(project.id), fresh, session.scope)).model_dump(
        by_alias=True, mode="json"
    )

    assert (state["accessMode"], state["accessEmails"]) == ("restricted", ["a@else.com"])


def _published_before_stable_ids(issuer, folder, scope, key, *, emails, org_allowed):
    """Leave the artifact as a report published restricted before stable ids
    and not since: its rule under the report's ``<user_dir>/<report_id>`` id,
    bound to the owner, none under ``key``, and an entry that records no key.
    ``user_dir`` is the upload service's folder for the owner, the first nine
    hex digits of md5 of their id (the upload service's code, which can't be
    imported here).
    Answers the legacy id."""
    legacy = f"{hashlib.md5(scope.user_id.encode()).hexdigest()[:9]}/report-1"
    issuer.rules.pop(key, None)
    issuer.rules[legacy] = {
        "org_allowed": org_allowed, "allowed_team_ids": [], "allowed_user_ids": [], "allowed_emails": list(emails),
    }
    issuer.bindings[legacy] = (scope.user_id, scope.org_id)
    _plant_entry(folder, artifact_key="", lambda_artifact_key="")
    return legacy


@pytest.mark.asyncio
async def test_a_member_re_saving_their_legacy_artifact_shared_with_everyone_keeps_it(
    issuer, scope, local_id, owned_artifact, monkeypatch, granted_product_permissions
):
    """A report an admin shared with everyone before stable ids still has its
    rule under its old id. The check reads that rule when the stable id has
    none, so a member without artifacts.share_org who saves it with everyone
    again is not refused, and the publish plans against that rule."""
    session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    _published_before_stable_ids(issuer, folder, scope, key, emails=["a@else.com"], org_allowed=True)
    mindshub_sharing_answers(monkeypatch)

    class _Key:
        async def get(self):
            return "turn-key"

        async def revoke(self):
            pass

    monkeypatch.setattr(
        artifact_workspace, "_owner_publish_context",
        lambda _session, _folder: (folder.parent, "https://view.test", _Key()),
    )
    everyone = {"mode": "restricted", "org_allowed": True, "allowed_emails": ["a@else.com"]}

    saved = await artifact_workspace.set_artifact_access(
        str(project.id), local_id, artifact_workspace._AccessBody(access=everyone), session.scope, member_request(), principal=None,
    )

    # A refusal answers its 403 as a response of its own; a publish, its result.
    assert isinstance(saved, dict), (saved.status_code, saved.body)
    assert saved["orgAllowed"] is True
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": ["a@else.com"], "org_allowed": True}
    assert uploads[-1]["artifact_key"] == key


@pytest.mark.asyncio
async def test_the_access_route_answers_a_legacy_rule_when_the_stable_id_holds_none(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    _published_before_stable_ids(issuer, folder, scope, key, emails=["a@else.com"], org_allowed=True)

    state = (await artifact_workspace.artifact_access(str(project.id), local_id, session.scope)).model_dump(
        by_alias=True, mode="json"
    )

    assert (state["accessMode"], state["accessEmails"], state["orgAllowed"]) == ("restricted", ["a@else.com"], True)


@pytest.mark.asyncio
async def test_a_live_sync_of_a_legacy_artifact_sends_the_legacy_rules_audience(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """A content save that isn't the owner's choice takes who the artifact is
    shared with from the rule. Read under the stable id alone, a legacy
    artifact's would be nobody, and the publish would quietly take everyone
    else off it."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    _published_before_stable_ids(issuer, folder, scope, key, emails=["a@else.com"], org_allowed=True)

    saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is True
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": ["a@else.com"], "org_allowed": True}


def _another_artifact_shared_with_everyone(issuer, scope):
    """Another of the owner's artifacts, its rule shared with everyone. Answers its key."""
    other = artifact_key(uuid4().hex)
    issuer.rules[other] = {"org_allowed": True, "allowed_team_ids": [], "allowed_user_ids": [], "allowed_emails": []}
    issuer.bindings[other] = (scope.user_id, scope.org_id)
    return other


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["artifact_key", "lambda_artifact_key"])
async def test_a_live_sync_never_takes_its_audience_from_a_key_an_agent_writes_in_the_entry(
    issuer, scope, local_id, owned_artifact, monkeypatch, field
):
    """The pod can write the keys the entry recorded. A re-publish that took
    who it shares with from a rule under one of them would send another
    artifact's audience, everyone in the organization here, as this one's.
    So it reads the artifact's identity, then its report's old id."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    _published_before_stable_ids(issuer, folder, scope, key, emails=["a@else.com"], org_allowed=False)
    _plant_entry(folder, **{field: _another_artifact_shared_with_everyone(issuer, scope)})

    saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is True
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}


@pytest.mark.asyncio
async def test_a_loose_files_live_sync_never_takes_its_audience_from_an_artifact_key_an_agent_writes(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """A loose file has no metadata.json, so it has no identity, as this
    folder has none once its metadata.json is gone. This server sends the
    upload service no key for it, so its rule sits under the report's old
    id. The pod can write the entry's ``artifact_key``. A re-publish that
    took who it shares with from a rule under it would send another
    artifact's audience, everyone in the organization here, as this one's.
    So it reads the report's old id alone."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    _published_before_stable_ids(issuer, folder, scope, key, emails=["a@else.com"], org_allowed=False)
    (folder / "metadata.json").unlink()
    _plant_entry(folder, artifact_key=_another_artifact_shared_with_everyone(issuer, scope))

    saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is True
    assert uploads[-1]["artifact_key"] is None
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}


def test_a_loose_files_owner_save_writes_its_rule_under_the_key_the_upload_service_answers(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """Only the read of a loose file's audience skips the entry's key. The
    team and people write, and the owner's choice, take the key the upload
    service answers for the publish (publish._rule_key), which for a loose
    file is the report's old id, so they land on the rule the next read
    finds."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services.artifact_owner_choice import OwnerChoice, read_owner_choice

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    legacy = _published_before_stable_ids(issuer, folder, scope, key, emails=["a@else.com"], org_allowed=False)
    (folder / "metadata.json").unlink()
    # The upload service answers a key-less publish with the report's old id.
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch, echoed_key=legacy)

    run(
        {"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False, "allowed_team_ids": [TEAM_ENG]},
        owner_save=True,
    )

    assert uploads[-1]["artifact_key"] is None
    assert _upserts(issuer)[-1]["artifact_id"] == legacy
    assert issuer.rules[legacy]["allowed_team_ids"] == [TEAM_ENG]
    assert read_owner_choice(keys=[legacy], scope=scope) == OwnerChoice(AccessMode.RESTRICTED, True)


@pytest.mark.asyncio
async def test_a_live_sync_reads_the_identitys_rule_before_a_legacy_rule_left_behind(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """The upload service deletes the legacy rule once it moves, on a best
    effort, so one can be left behind. The identity's rule is the one the
    owner saved since, and a legacy rule left behind never widens it again."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)
    legacy = f"{hashlib.md5(scope.user_id.encode()).hexdigest()[:9]}/{_entry(folder)['report_id']}"
    issuer.rules[legacy] = {"org_allowed": True, "allowed_team_ids": [], "allowed_user_ids": [], "allowed_emails": []}
    issuer.bindings[legacy] = (scope.user_id, scope.org_id)

    saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is True
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}


@pytest.mark.asyncio
async def test_the_access_route_never_answers_the_rule_under_a_key_an_agent_writes_in_a_restricted_entry(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """The Share dialog, and the check a save makes, read who a restricted
    artifact is shared with the way its re-publish does, so an agent can't
    make it look shared with everyone."""
    session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    _published_before_stable_ids(issuer, folder, scope, key, emails=["a@else.com"], org_allowed=False)
    _plant_entry(folder, lambda_artifact_key=_another_artifact_shared_with_everyone(issuer, scope))

    state = (await artifact_workspace.artifact_access(str(project.id), local_id, session.scope)).model_dump(
        by_alias=True, mode="json"
    )

    assert (state["accessMode"], state["accessEmails"], state["orgAllowed"]) == ("restricted", ["a@else.com"], False)


@pytest.mark.asyncio
async def test_the_access_route_answers_when_an_agent_writes_a_report_id_too_long_for_mindshub(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """The pod can write the entry's report_id. An old id longer than
    MindsHub takes holds no rule, so it is never asked about, and the Share
    dialog answers rather than failing on MindsHub's 400."""
    session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    _published_before_stable_ids(issuer, folder, scope, key, emails=["a@else.com"], org_allowed=False)
    _plant_entry(folder, report_id="r" * 600)

    state = (await artifact_workspace.artifact_access(str(project.id), local_id, session.scope)).model_dump(
        by_alias=True, mode="json"
    )

    assert (state["accessMode"], state["accessEmails"], state["orgAllowed"]) == ("restricted", [], False)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["artifact_key", "lambda_artifact_key"])
@pytest.mark.parametrize(
    "planted",
    ["k" * 600, "   ", "\t\n\u3000", "a\x00b", "a\ud800", "a\udfff"],
    ids=["too_long", "blank", "other_whitespace", "nul", "lone_high_surrogate", "lone_low_surrogate"],
)
async def test_the_access_route_answers_when_an_agent_writes_an_entry_key_mindshub_refuses(
    issuer, scope, local_id, owned_artifact, monkeypatch, field, planted
):
    """The pod can write the keys a public entry records. One MindsHub
    refuses as an artifact id holds no rule, so the check for a rule held
    over the entry skips it, and the Share dialog answers rather than
    failing on MindsHub's 400, or before it on a body that can't be
    encoded."""
    session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "public"}, owner_save=True)
    _plant_entry(folder, **{field: planted})

    state = (await artifact_workspace.artifact_access(str(project.id), local_id, session.scope)).model_dump(
        by_alias=True, mode="json"
    )

    assert state["accessMode"] == "public"
    asked = [body["artifact_id"] for path, body in issuer.calls if path.endswith("/read/")]
    assert planted not in asked and planted.strip() not in asked


@pytest.mark.asyncio
async def test_the_access_route_refuses_when_an_agent_writes_a_key_mindshub_binds_to_another_owner(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """MindsHub's 409 for a key in the entry bound to another owner also
    answers a key it holds for repair, whose rule may be this artifact's, so
    the read fails closed and the Share dialog reports the refusal."""
    session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "public"}, owner_save=True)
    colleague = artifact_key(uuid4().hex)
    issuer.bindings[colleague] = (str(uuid4()), scope.org_id)
    _plant_entry(folder, artifact_key=colleague)

    with pytest.raises(HTTPException) as refused:
        await artifact_workspace.artifact_access(str(project.id), local_id, session.scope)

    assert refused.value.status_code == 409



@pytest.mark.asyncio
@pytest.mark.parametrize("planted", ["a\ud800", "a\x00b"], ids=["lone_surrogate", "nul"])
async def test_a_live_sync_of_a_public_entry_with_a_key_mindshub_refuses_still_publishes(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog, planted
):
    """A key the pod writes into a public entry that no request body or
    column can carry is asked about neither in MindsHub nor in the owner's
    record, so the content save still publishes the page as it was."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "public"}, owner_save=True)
    uploaded = len(uploads)
    _plant_entry(folder, artifact_key=planted)

    with caplog.at_level(logging.WARNING):
        saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is True
    assert len(uploads) == uploaded + 1
    assert uploads[-1]["access"] == {"mode": "public"}
    assert not [record for record in caplog.records if record.exc_info]


def _nest(folder, local_id, how):
    """Put the artifact's page in a subfolder, as an agent-built site does:
    a ``primary`` the metadata names, or a newer nested index.html."""
    sub = folder / ("site" if how == "hint" else "docs")
    sub.mkdir()
    (sub / "index.html").write_text("<html>nested</html>")
    if how == "hint":
        (folder / "metadata.json").write_text(
            json.dumps({"id": local_id, "slug": "demo", "type": "html-app", "primary": "site/index.html"})
        )
    else:
        old = time.time() - 600
        os.utime(folder / "index.html", (old, old))
    return sub


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["hint", "newer_nested_index"])
@pytest.mark.parametrize("republish", ["live_sync", "autopublish"])
async def test_a_page_in_a_subfolder_its_owner_made_public_keeps_syncing(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog, how, republish
):
    """A page in a subfolder of the artifact has no metadata beside it, so
    it publishes with no identity, under the key the upload service gives
    its report. A page its owner made public is still re-published by live
    sync and autopublish, never refused for having no identity."""
    from cowork.services import artifact_autopublish as ap

    _session, project, _conversation, folder = owned_artifact
    sub = _nest(folder, local_id, how)
    legacy = f"{hashlib.md5(scope.user_id.encode()).hexdigest()[:9]}/report-1"
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch, echoed_key=legacy)
    run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)
    assert uploads[-1]["artifact_key"] is None
    run({"mode": "public"}, owner_save=True)
    uploaded = len(uploads)

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        if republish == "live_sync":
            saved = await _live_sync(folder, scope, project, monkeypatch)
        else:
            entry = json.loads((sub / ".published.json").read_text())["index.html"]
            decision = ap.PublishDecision("changed", published_entry=entry)
            saved = await ap._publish_one(
                folder.parent, folder.name, "key", "https://view.test", 30.0, scope,
                ap._access_for(decision), project_id=str(project.id),
            )

    assert saved is True, [record.reason for record in _mode_changed_records(caplog)]
    assert len(uploads) == uploaded + 1
    assert uploads[-1]["access"] == {"mode": "public"}



@pytest.mark.asyncio
async def test_a_live_sync_reads_the_rule_under_a_recorded_key_after_one_too_long_for_mindshub(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog
):
    """An agent that moves the metadata id and writes a key longer than
    MindsHub takes into the entry's artifact_key gets nothing published: the
    long key is skipped, and the rule the owner saved is still read under the
    lambda's key the entry records."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    uploaded = len(uploads)
    (folder / "metadata.json").write_text(json.dumps({"id": uuid4().hex, "slug": "demo", "type": "html-app"}))
    _plant_entry(folder, mode="public", artifact_key="k" * 600)

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is False
    assert len(uploads) == uploaded
    assert issuer.rules[key]["allowed_emails"] == ["a@else.com"]
    [record] = _mode_changed_records(caplog)
    assert record.reason == "rule_held"


@pytest.mark.asyncio
async def test_a_live_sync_never_publishes_a_legacy_artifact_publicly_after_an_agent_changes_the_mode(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog
):
    """The legacy rule is a rule MindsHub holds, so an entry an agent turned
    public publishes nothing while it stands."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False})["artifactKey"]
    legacy = _published_before_stable_ids(issuer, folder, scope, key, emails=["a@else.com"], org_allowed=False)
    uploaded = len(uploads)
    _plant_entry(folder, mode="public")

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is False
    assert len(uploads) == uploaded
    assert issuer.rules[legacy]["allowed_emails"] == ["a@else.com"]
    [record] = _mode_changed_records(caplog)
    assert record.reason == "rule_held"


@pytest.mark.parametrize(
    ("entry", "owner", "expected"),
    [
        # The upload service's folder for this owner, as its own code names it.
        ({"report_id": "r1"}, "0b0b0b0b-0000-4000-8000-000000000001", "d002e645a/r1"),
        ({"report_id": "r1"}, "0B0B0B0B-0000-4000-8000-000000000001", "d002e645a/r1"),
        ({"report_id": "r1/x"}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        ({"report_id": ""}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        ({}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        ({"report_id": "r1"}, "not-a-uuid", ""),
        ({"report_id": "r1"}, None, ""),
        # Only the upload service's report-id form, in an id no longer than MindsHub takes.
        ({"report_id": "Report-1_x"}, "0b0b0b0b-0000-4000-8000-000000000001", "d002e645a/Report-1_x"),
        ({"report_id": "a" * 502}, "0b0b0b0b-0000-4000-8000-000000000001", "d002e645a/" + "a" * 502),
        ({"report_id": "a" * 503}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        ({"report_id": "r1\n"}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        ({"report_id": "r 1"}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        ({"report_id": "r1.html"}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        ({"report_id": "r1\x00"}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        ({"report_id": "r\u00e9"}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        # The Kelvin sign, which [A-Za-z] matches only when case is ignored.
        ({"report_id": "r\u212a"}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
        ({"report_id": 7}, "0b0b0b0b-0000-4000-8000-000000000001", ""),
    ],
    ids=[
        "owner", "owner_upper_case", "report_id_with_a_slash", "empty_report_id", "no_report_id",
        "owner_not_a_uuid", "no_owner", "report_id_with_a_dash_an_underscore_and_capitals",
        "longest_id_mindshub_takes", "id_longer_than_mindshub_takes", "report_id_with_a_trailing_newline",
        "report_id_with_a_space", "report_id_with_a_dot", "report_id_with_a_nul", "report_id_not_ascii",
        "report_id_with_a_kelvin_sign", "report_id_not_a_string",
    ],
)
def test_a_reports_id_from_before_stable_ids_is_the_upload_services_folder_and_report(entry, owner, expected):
    from cowork.services.publish import _report_rule_key

    assert _report_rule_key(entry, owner_user_id=owner) == expected


@pytest.mark.asyncio
async def test_an_artifact_the_owner_made_public_keeps_syncing_publicly(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog
):
    """The owner's own save from restricted to public goes through, and the
    publish lambda deletes the rule. A content save then re-publishes it
    publicly, and GET /access answers public: no rule is left, so the
    entry's mode is the owner's."""
    session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False}, owner_save=True)["artifactKey"]
    run({"mode": "public"}, owner_save=True)
    assert key not in issuer.rules

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is True
    assert uploads[-1]["access"] == {"mode": "public"}
    assert _mode_changed_records(caplog) == []
    state = (await artifact_workspace.artifact_access(str(project.id), local_id, session.scope)).model_dump(
        by_alias=True, mode="json"
    )
    assert (state["accessMode"], state["accessEmails"]) == ("public", [])


async def _autopublish_once(folder, scope, project):
    """One autopublish of ``folder`` after a turn changed it, with the access
    autopublish rebuilds from its entry (artifact_autopublish._access_for)."""
    from cowork.services import artifact_autopublish as ap

    decision = ap.PublishDecision("changed", published_entry=_entry(folder))
    return await ap._publish_one(
        folder.parent, folder.name, "key", "https://view.test", 30.0, scope,
        ap._access_for(decision), project_id=str(project.id),
    )


async def _republish(how, folder, scope, project, monkeypatch):
    if how == "live_sync":
        return await _live_sync(folder, scope, project, monkeypatch)
    return await _autopublish_once(folder, scope, project)


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ("live_sync", "autopublish"))
async def test_a_password_artifact_an_agent_makes_public_is_never_republished_publicly(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog, how
):
    """A password publish leaves MindsHub no rule, so the restricted-rule
    check can't see an agent rewrite the entry's mode to public. The owner's
    save records its mode where the pod can't write, so a re-publish in that
    mode still goes through, and one wider than it is refused and logged
    before anything is uploaded."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "password", "password": "owner-pw"}, owner_save=True)["artifactKey"]
    assert key not in issuer.rules
    assert await _republish(how, folder, scope, project, monkeypatch) is True
    assert uploads[-1]["access"]["mode"] == "password"
    uploaded = len(uploads)
    _plant_entry(folder, mode="public", requires_password=False)

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        published = await _republish(how, folder, scope, project, monkeypatch)

    assert published is False
    assert len(uploads) == uploaded
    [record] = _mode_changed_records(caplog)
    assert (record.reason, record.artifact_slug) == ("wider_than_owner_choice", folder.name)


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ("live_sync", "autopublish"))
async def test_an_artifact_recorded_unpublished_is_never_republished_when_an_agent_marks_it_published(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog, how
):
    """The record says the owner's artifact is unpublished, and the entry
    agrees, so autopublish leaves it alone. An agent that marks the entry
    published again gets nothing uploaded: every re-publish that isn't the
    owner's save is refused while the record says unpublished. The owner's
    own save publishes it again, and the record with it."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services.artifact_autopublish import REASON_UNPUBLISHED, needs_publish
    from cowork.services.artifact_owner_choice import OwnerChoice, record_owner_choice

    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "public"}, owner_save=True)["artifactKey"]
    record_owner_choice(key=key, choice=OwnerChoice(AccessMode.PUBLIC, published=False), scope=scope)
    _plant_entry(folder, published=False)
    assert needs_publish(folder, folder.parent).reason == REASON_UNPUBLISHED
    uploaded = len(uploads)
    _plant_entry(folder, published=True)

    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        published = await _republish(how, folder, scope, project, monkeypatch)

    assert published is False
    assert len(uploads) == uploaded
    [record] = _mode_changed_records(caplog)
    assert record.reason == "unpublished_by_owner"

    run({"mode": "public"}, owner_save=True)
    assert await _republish(how, folder, scope, project, monkeypatch) is True
    assert uploads[-1]["access"] == {"mode": "public"}


@pytest.mark.asyncio
async def test_an_owners_save_narrows_the_record_before_anything_uploads(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog
):
    """The owner's save of a narrower mode narrows the record before its
    upload, so a save whose upload fails leaves no wider record behind: the
    public entry the failed save left in place is not re-published publicly
    on the owner's behalf."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "public"}, owner_save=True)

    def upload_fails(_source, **kwargs):
        raise OSError("upload failed")

    monkeypatch.setattr("anton.publisher.publish", upload_fails)
    with pytest.raises(RuntimeError):
        run({"mode": "password", "password": "owner-pw"}, owner_save=True)
    assert _entry(folder)["mode"] == "public"

    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        assert await _live_sync(folder, scope, project, monkeypatch) is False
    assert uploads == []
    [record] = _mode_changed_records(caplog)
    assert record.reason == "wider_than_owner_choice"


class UploadEnding(BaseModel):
    """One way an owner's save can end once its upload is under way
    (tests/fixtures/upload_endings.yaml): the upload service's answer, or an
    error with no answer, and the error the save raises for it."""

    name: str
    status: int | None = None
    body: dict[str, str] = Field(default_factory=dict)
    raw_body: str | None = None
    raises: Literal["timeout", "connection_error", "state_publish_blocked", "os_error"] | None = None
    save_raises: str


class UploadEndings(BaseModel):
    restored: list[UploadEnding]
    narrowed: list[UploadEnding]


UPLOAD_ENDINGS = UploadEndings.model_validate(
    yaml.safe_load((Path(__file__).parent / "fixtures" / "upload_endings.yaml").read_text())
)
_SHARE_ORG_REQUIRED = next(e for e in UPLOAD_ENDINGS.restored if e.name == "share_org_required")
# anton's publisher as it ships, taken before any test replaces it.
_ANTON_PUBLISH = anton.publisher.publish
# MindsHub's own publishing service, the one publisher whose refusals put the
# owner's choice back (cowork.services.providers.is_mindshub_publish_url).
_MINDSHUB_PUBLISH_URL = "https://view.mindshub.ai"


def _upload_ends(monkeypatch, ending: UploadEnding, *, meanwhile=None) -> None:
    """The next publish runs anton's publisher, and its upload ends as
    ``ending`` says. ``meanwhile`` runs while the request is out, as another
    save of the artifact on another replica would."""

    def request(url, api_key, **_kwargs):
        if meanwhile is not None:
            meanwhile()
        if ending.raises == "timeout":
            raise TimeoutError("The read operation timed out")
        if ending.raises == "connection_error":
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        if ending.status == 200:
            return (ending.raw_body or "").encode()
        body = io.BytesIO(json.dumps(ending.body).encode())
        raise urllib.error.HTTPError(url, ending.status, "Refused", {}, body)

    def publish_fails_before_sending(_source, **_kwargs):
        if ending.raises == "state_publish_blocked":
            raise anton.publisher.StatePublishBlocked("Collections ['notes'] are missing from the new schema")
        raise OSError("Could not read the artifact's files to zip them")

    local = ending.raises in ("state_publish_blocked", "os_error")
    monkeypatch.setattr("anton.publisher.publish", publish_fails_before_sending if local else _ANTON_PUBLISH)
    monkeypatch.setattr("anton.publisher.minds_request", request)


def _owner_choice(key, scope):
    from cowork.services.artifact_owner_choice import read_owner_choice

    return read_owner_choice(keys=[key], scope=scope)


_EVERYONE = {"mode": "restricted", "emails": [], "org_allowed": True}


@pytest.mark.asyncio
async def test_a_save_mindshub_refuses_to_share_with_everyone_keeps_the_owners_public_choice(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog
):
    """An owner of a public artifact picks Everyone while MindsHub no longer
    lets them share with everyone, and MindsHub refuses the rule's write.
    Nothing was published, and the public link still works. The record goes
    back to public, so the owner's next edit still reaches that link: a live
    sync publishes it publicly, with no refusal logged."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services.artifact_owner_choice import OwnerChoice
    from cowork.services.publish import ShareOrgRequired

    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch, publish_url=_MINDSHUB_PUBLISH_URL)
    key = run({"mode": "public"}, owner_save=True)["artifactKey"]

    _upload_ends(monkeypatch, _SHARE_ORG_REQUIRED)
    with pytest.raises(ShareOrgRequired):
        run(_EVERYONE, owner_save=True)
    assert _entry(folder)["mode"] == "public"
    assert _owner_choice(key, scope) == OwnerChoice(AccessMode.PUBLIC, published=True)

    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        assert await _live_sync(folder, scope, project, monkeypatch) is True
    assert uploads[-1]["access"] == {"mode": "public"}
    assert _mode_changed_records(caplog) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", UPLOAD_ENDINGS.restored, ids=lambda ending: ending.name)
async def test_an_upload_refused_before_writing_puts_the_owners_choice_back(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog, ending
):
    """The owner of a public artifact saves a password, and the upload ends
    before MindsHub's upload service changed who can open the artifact or
    what it serves. The save raises the refusal it always did, the record
    goes back to public, and a live sync still publishes publicly."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services.artifact_owner_choice import OwnerChoice

    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch, publish_url=_MINDSHUB_PUBLISH_URL)
    key = run({"mode": "public"}, owner_save=True)["artifactKey"]

    _upload_ends(monkeypatch, ending)
    with pytest.raises(Exception) as raised:
        run({"mode": "password", "password": "owner-pw"}, owner_save=True)
    assert type(raised.value).__name__ == ending.save_raises
    assert _owner_choice(key, scope) == OwnerChoice(AccessMode.PUBLIC, published=True)

    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        assert await _live_sync(folder, scope, project, monkeypatch) is True
    assert uploads[-1]["access"] == {"mode": "public"}
    assert _mode_changed_records(caplog) == []


@pytest.mark.parametrize("ending", UPLOAD_ENDINGS.restored, ids=lambda ending: ending.name)
def test_a_refusal_from_a_publisher_that_isnt_mindshubs_keeps_the_owners_choice_narrowed(
    issuer, scope, local_id, owned_artifact, monkeypatch, ending
):
    """The organization publishes through a service other than MindsHub's,
    whose order of writes nothing here knows, and that service refuses the
    owner's save of a password. It may have written first, so the record
    keeps the password, whatever the refusal."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services.artifact_owner_choice import OwnerChoice

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch, publish_url="https://publish.example")
    key = run({"mode": "public"}, owner_save=True)["artifactKey"]

    _upload_ends(monkeypatch, ending)
    with pytest.raises(Exception) as raised:
        run({"mode": "password", "password": "owner-pw"}, owner_save=True)
    assert type(raised.value).__name__ == ending.save_raises
    assert _owner_choice(key, scope) == OwnerChoice(AccessMode.PASSWORD, published=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", UPLOAD_ENDINGS.narrowed, ids=lambda ending: ending.name)
async def test_an_upload_that_may_have_written_keeps_the_owners_choice_narrowed(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog, ending
):
    """The owner of a public artifact saves a password, and the upload ends
    in a way that can follow the upload service's writes: the password may
    be live. The record keeps the password, so a live sync of the public
    entry left behind is refused before anything is uploaded."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services.artifact_owner_choice import OwnerChoice

    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch, publish_url=_MINDSHUB_PUBLISH_URL)
    key = run({"mode": "public"}, owner_save=True)["artifactKey"]

    _upload_ends(monkeypatch, ending)
    with pytest.raises(Exception) as raised:
        run({"mode": "password", "password": "owner-pw"}, owner_save=True)
    assert type(raised.value).__name__ == ending.save_raises
    assert _owner_choice(key, scope) == OwnerChoice(AccessMode.PASSWORD, published=True)

    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        assert await _live_sync(folder, scope, project, monkeypatch) is False
    assert uploads == []
    [record] = _mode_changed_records(caplog)
    assert record.reason == "wider_than_owner_choice"


def _markdown_artifact(folder) -> None:
    """Make ``folder`` a Markdown artifact: its page is notes.md."""
    (folder / "index.html").unlink()
    (folder / "notes.md").write_text("# Notes\n")
    metadata = json.loads((folder / "metadata.json").read_text())
    (folder / "metadata.json").write_text(json.dumps({**metadata, "primary": "notes.md"}))


def test_a_markdown_render_that_fails_leaves_the_owners_choice_as_it_was(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """A Markdown artifact's save renders its page before it narrows the
    record, so a render that fails leaves the record as it was."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services.artifact_owner_choice import OwnerChoice

    _session, project, _conversation, folder = owned_artifact
    _markdown_artifact(folder)
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    key = run({"mode": "public"}, owner_save=True)["artifactKey"]

    def render_fails(_md_path, _out_dir):
        raise OSError("Could not read notes.md")

    monkeypatch.setattr("cowork.services.publish._render_markdown_to_html", render_fails)
    with pytest.raises(OSError):
        run({"mode": "password", "password": "owner-pw"}, owner_save=True)
    assert _owner_choice(key, scope) == OwnerChoice(AccessMode.PUBLIC, published=True)


def test_a_markdown_save_whose_narrowing_fails_removes_its_rendered_page(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """A Markdown artifact's save renders its page into a folder of its own,
    then can't narrow the record because the database is down. The save
    fails with that outage, and the folder is already gone while the
    failure's traceback still holds the save, not left for garbage
    collection."""
    from cowork.services import artifact_owner_choice as choices
    from cowork.services import publish as publish_module
    from cowork.services.artifact_access import ArtifactAccessUnavailable

    _session, project, _conversation, folder = owned_artifact
    _markdown_artifact(folder)
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "public"}, owner_save=True)

    rendered = []
    render = publish_module._render_markdown_to_html

    def render_and_note(md_path, out_dir):
        rendered.append(out_dir)
        return render(md_path, out_dir)

    def narrowing_fails(**_kwargs):
        raise ArtifactAccessUnavailable("Could not record the sharing the artifact's owner chose")

    monkeypatch.setattr("cowork.services.publish._render_markdown_to_html", render_and_note)
    monkeypatch.setattr(choices, "narrow_before_save", narrowing_fails)
    with pytest.raises(ArtifactAccessUnavailable) as raised:
        run({"mode": "password", "password": "owner-pw"}, owner_save=True)
    # The traceback keeps the save's frame, and with it the folder's
    # TemporaryDirectory, alive: only the save itself can have removed it.
    assert raised.value.__traceback__ is not None
    [out_dir] = rendered
    assert not out_dir.exists()


@pytest.mark.parametrize("other_save", ("records_the_same_mode", "narrows_nothing"))
def test_a_save_that_started_since_keeps_its_write_over_a_refused_saves_restore(
    issuer, scope, local_id, owned_artifact, monkeypatch, other_save
):
    """The owner's save of a password narrows a public record, and while
    its upload is out another save of the artifact starts on another
    replica: one that records the password too once its upload is live, or
    one that narrows nothing because the record is already a password, but
    whose upload may change what is live. When MindsHub then refuses the
    first save, its restore leaves the record as the other save left it. A
    restore that compared the mode alone, or a narrowing that stamped the row
    only when it changed the mode, would put public back."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services.artifact_owner_choice import OwnerChoice, narrow_before_save, record_owner_choice

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch, publish_url=_MINDSHUB_PUBLISH_URL)
    key = run({"mode": "public"}, owner_save=True)["artifactKey"]
    password = OwnerChoice(AccessMode.PASSWORD, published=True)

    def another_save():
        if other_save == "records_the_same_mode":
            record_owner_choice(key=key, choice=password, scope=scope)
        else:
            assert narrow_before_save(key=key, chosen=OwnerChoice(AccessMode.PUBLIC, True), scope=scope) is None

    _upload_ends(monkeypatch, _SHARE_ORG_REQUIRED, meanwhile=another_save)
    with pytest.raises(Exception):
        run({"mode": "password", "password": "owner-pw"}, owner_save=True)
    assert _owner_choice(key, scope) == password


def test_a_restore_that_cannot_reach_the_database_leaves_the_choice_narrowed_and_logs_it(
    issuer, scope, local_id, owned_artifact, monkeypatch, caplog
):
    """MindsHub refuses the owner's save of Everyone, and the database goes
    down before the save can put the record back. The record stays
    restricted, which only refuses wider re-publishes, a warning names the
    artifact, and the save still raises MindsHub's refusal rather than the
    database's outage."""
    from sqlalchemy.exc import OperationalError

    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services import artifact_owner_choice as choices
    from cowork.services.publish import ShareOrgRequired

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch, publish_url=_MINDSHUB_PUBLISH_URL)
    key = run({"mode": "public"}, owner_save=True)["artifactKey"]
    reachable = choices.authorization_session

    def database_down(_scope):
        raise OperationalError("SELECT", {}, Exception("database is down"))

    _upload_ends(
        monkeypatch, _SHARE_ORG_REQUIRED,
        meanwhile=lambda: monkeypatch.setattr(choices, "authorization_session", database_down),
    )
    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        with pytest.raises(ShareOrgRequired):
            run(_EVERYONE, owner_save=True)

    monkeypatch.setattr(choices, "authorization_session", reachable)
    assert _owner_choice(key, scope) == choices.OwnerChoice(AccessMode.RESTRICTED, published=True)
    [record] = [
        r for r in caplog.records
        if r.name == "cowork.services.publish" and getattr(r, "event", None) == "artifact_owner_choice_restore_failed"
    ]
    assert (record.levelno, record.artifact_slug) == (logging.WARNING, folder.name)


def test_the_owner_choice_record_judges_by_the_narrowest_key_and_fails_closed(scope, monkeypatch):
    """A re-publish is judged against the narrowest choice recorded under any
    key its rule could sit under, so a key the pod wrote into the entry can
    only narrow it. A key longer than the column holds keeps no record. A
    record that can't be read fails the re-publish rather than guessing."""
    from sqlalchemy.exc import OperationalError

    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services import artifact_owner_choice as choices

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    mine, other = artifact_key(uuid4().hex), artifact_key(uuid4().hex)
    choices.record_owner_choice(key=mine, choice=choices.OwnerChoice(AccessMode.PUBLIC, True), scope=scope)
    choices.record_owner_choice(key=other, choice=choices.OwnerChoice(AccessMode.PASSWORD, True), scope=scope)
    assert choices.republish_widening(keys=[mine], mode=AccessMode.PUBLIC, scope=scope) is None
    assert choices.republish_widening(keys=[mine, other], mode=AccessMode.PUBLIC, scope=scope) == (
        choices.Widening.WIDER_THAN_OWNER_CHOICE
    )
    # Recorded again, the owner's newest choice replaces the row.
    choices.record_owner_choice(key=mine, choice=choices.OwnerChoice(AccessMode.RESTRICTED, True), scope=scope)
    assert choices.read_owner_choice(keys=[mine], scope=scope) == choices.OwnerChoice(AccessMode.RESTRICTED, True)
    # Another organization's record of the same key is not this one's.
    elsewhere = TenantScope(org_mode=True, org_id=str(uuid4()), user_id=scope.user_id)
    assert choices.read_owner_choice(keys=[mine], scope=elsewhere) is None
    too_long = "artifact/" + "x" * 64
    choices.record_owner_choice(key=too_long, choice=choices.OwnerChoice(AccessMode.RESTRICTED, False), scope=scope)
    assert choices.read_owner_choice(keys=[too_long], scope=scope) is None

    def broken(_scope):
        raise OperationalError("SELECT", {}, Exception("database is down"))

    monkeypatch.setattr(choices, "authorization_session", broken)
    with pytest.raises(ArtifactAccessUnavailable):
        choices.republish_widening(keys=[mine], mode=AccessMode.PUBLIC, scope=scope)
    get_app_settings.cache_clear()


def test_two_saves_racing_to_record_an_artifacts_first_choice_both_succeed(scope, monkeypatch):
    """Two owner saves of one artifact can each find no row and each insert
    one. The unique key lets the first commit; the second's commit fails,
    rolls back, finds the row the first committed and records its own
    choice over it. The save doesn't fail, and one row holds the choice of
    the save that committed last."""
    from sqlmodel import func

    from cowork.models.artifact_owner_choice import AccessMode, ArtifactOwnerChoice
    from cowork.services import artifact_owner_choice as choices

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    key = artifact_key(uuid4().hex)
    stage = choices._stage_choice
    raced: list = []

    def stage_then_race(session, *, key, choice):
        stage(session, key=key, choice=choice)
        if not raced:
            # The other save found no row either, and commits its own first.
            raced.append(choice)
            with choices.authorization_session(scope) as other:
                stage(other, key=key, choice=choices.OwnerChoice(AccessMode.PUBLIC, True))
                other.commit()

    monkeypatch.setattr(choices, "_stage_choice", stage_then_race)
    choices.record_owner_choice(key=key, choice=choices.OwnerChoice(AccessMode.RESTRICTED, True), scope=scope)

    assert raced, "the race never ran"
    assert choices.read_owner_choice(keys=[key], scope=scope) == choices.OwnerChoice(AccessMode.RESTRICTED, True)
    with choices.authorization_session(scope) as session:
        rows = session.exec(
            session.select(func.count()).select_from(ArtifactOwnerChoice).where(ArtifactOwnerChoice.artifact_key == key)
        ).one()
    assert rows == 1
    get_app_settings.cache_clear()


def test_the_rule_is_read_and_written_under_one_key_whatever_the_lambda_echoes(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """The read before a publish asks the artifact's identity first
    (publish._audience_rule_keys), and the team and people write after it
    takes the rule's key from publish._rule_key, the identity first too. A
    lambda that echoes another key moves neither."""
    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(
        issuer, folder, scope, project, monkeypatch, echoed_key=f"artifact/{uuid4()}",
    )
    team = {"mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": [TEAM_ENG]}
    for teams in ([TEAM_ENG], [TEAM_OPS]):
        run({**team, "allowed_team_ids": teams}, owner_save=True)

    canonical = identities.existing_authorization_key(local_id, scope)
    rule_calls = [
        (path.rsplit("/artifact-access/", 1)[1], body["artifact_id"]) for path, body in issuer.calls
        if path.endswith(("/artifact-access/", "/read/"))
    ]
    assert rule_calls == [("read/", canonical), ("", canonical), ("read/", canonical), ("", canonical)]
    assert issuer.rules[canonical]["allowed_team_ids"] == [TEAM_OPS]


def test_a_save_plans_against_the_rule_its_check_read_without_reading_it_again(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """The access route reads the rule once, under the publish lock, to check
    a save, and hands it to publish_artifact as ``stored``: the publish
    plans against that rule and asks MindsHub for none."""
    from cowork.services.artifact_access import ArtifactRule

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    team = {"mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": [TEAM_ENG]}
    key = run(team, owner_save=True)["artifactKey"]
    calls = len(issuer.calls)

    checked = ArtifactRule(org_allowed=False, allowed_team_ids=[TEAM_ENG], allowed_user_ids=[], allowed_emails=[])
    run(team, owner_save=True, stored=checked)
    assert [path for path, _body in issuer.calls[calls:]] == []

    # The rule the check read already holds the team, so nothing is written;
    # one that doesn't is planned as a change and written.
    run(team, owner_save=True, stored=ArtifactRule.nobody())
    assert [(path, body["allowed_team_ids"]) for path, body in issuer.calls[calls:]] == [
        ("/v1/internal/artifact-access/", [TEAM_ENG]),
    ]
    assert issuer.rules[key]["allowed_team_ids"] == [TEAM_ENG]


@pytest.mark.asyncio
async def test_a_re_publish_that_cannot_read_the_rule_fails_before_anything_is_uploaded(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """A live sync or autopublish never guesses who an artifact is shared
    with: while MindsHub's rule can't be read, it fails, and the
    published version stays as it was."""
    from cowork.services.publish import published_artifact_access

    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": True})
    uploaded = len(uploads)

    issuer.failure = True
    with pytest.raises(ArtifactAccessUnavailable):
        run(published_artifact_access(folder, artifacts_base=folder.parent, org_mode=True))
    assert len(uploads) == uploaded


def test_a_refused_audience_leaves_the_rule_as_it_was_and_a_retry_bumps_the_version(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """A team MindsHub refuses (deleted since the share
    dialog listed it) leaves the rule's teams as they were. The entry's
    version moved with the publish, so the retry with a team that exists
    bumps it again, past every grant issued before."""
    from cowork.services.artifact_access import ArtifactAudienceRejected

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)

    key = run({"mode": "restricted", "emails": [], "allowed_team_ids": [TEAM_ENG]})["artifactKey"]
    granted = _entry(folder)["access_version"]

    issuer.delete_team(TEAM_OPS)
    with pytest.raises(ArtifactAudienceRejected):
        run({"mode": "restricted", "emails": [], "allowed_team_ids": [TEAM_OPS]})
    assert issuer.rules[key]["allowed_team_ids"] == [TEAM_ENG]
    refused = _entry(folder)["access_version"]
    assert refused > granted

    run({"mode": "restricted", "emails": [], "allowed_team_ids": [TEAM_QA]})
    assert issuer.rules[key]["allowed_team_ids"] == [TEAM_QA]
    assert _entry(folder)["access_version"] > refused


def test_a_republish_after_an_unpublish_sends_the_whole_audience_again(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """An unpublish deletes the artifact's rule (the publish
    lambda's delete), while the soft-deleted entry keeps its mode. The owner
    then republishes it to the same team: the stored rule holds no team, so
    the team is written again, and the team's members open it."""
    from cowork.services.publish import unpublish_artifact

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    team = {"mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": [TEAM_ENG]}
    key = run(team)["artifactKey"]
    assert issuer.rules[key]["allowed_team_ids"] == [TEAM_ENG]

    monkeypatch.setattr("anton.publisher.unpublish", lambda *_a, **_kw: issuer.rules.pop(key))
    unpublish_artifact(
        folder, artifacts_base=folder.parent, api_key="key", publish_url="https://view.test", scope=scope,
    )
    assert key not in issuer.rules
    assert (_entry(folder)["published"], _entry(folder)["mode"]) == (False, "restricted")

    writes = len(_upserts(issuer))
    result = run(team)
    assert len(_upserts(issuer)) == writes + 1
    assert _upserts(issuer)[-1]["allowed_team_ids"] == [TEAM_ENG]
    assert issuer.rules[key]["allowed_team_ids"] == [TEAM_ENG]
    assert result["allowedTeamIds"] == [TEAM_ENG]


def test_a_notifying_save_asks_mindshub_to_email_with_title_and_url(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """Only a person's save with "Notify people" chosen sends
    ``notify_recipients`` with the people to tell (and the artifact's title
    and published URL) on cowork's write of the rule. The lambda's upsert
    never carries it, and neither does a re-publish from the stored access
    (live sync, autopublish)."""
    from cowork.services.artifact_audience import ShareNotify
    from cowork.services.publish import published_artifact_access

    _session, project, _conversation, folder = owned_artifact
    metadata = json.loads((folder / "metadata.json").read_text())
    (folder / "metadata.json").write_text(json.dumps({**metadata, "name": "Q3  Revenue"}))
    run, uploads = _org_publisher(
        issuer, folder, scope, project, monkeypatch, view_url="https://view.dev.mindshub.ai/1",
    )

    person, other, stranger = str(uuid4()), str(uuid4()), str(uuid4())
    run({"mode": "restricted", "emails": ["out@else.com"], "org_allowed": False,
         "allowed_user_ids": [person, other]},
        notify=ShareNotify(user_ids=(person, stranger), emails=("out@else.com", "nope@else.com")))
    body = _upserts(issuer)[-1]
    assert (body["notify_recipients"], body["artifact_title"], body["artifact_url"]) == (
        True, "Q3 Revenue", "https://view.dev.mindshub.ai/1",
    )
    # Only the people the saved rule names: never someone it doesn't share with.
    assert (body["notify_user_ids"], body["notify_emails"]) == ([person], ["out@else.com"])
    assert body["allowed_user_ids"] == [person, other]
    # Never handed to anton's publisher (the lambda's write), never stored.
    assert all("notify_recipients" not in (u["access"] or {}) for u in uploads)
    assert "notify_recipients" not in _entry(folder)

    # The automatic re-publish from what is stored writes nothing, so it
    # notifies no one.
    count = len(_upserts(issuer))
    run(published_artifact_access(folder, artifacts_base=folder.parent, org_mode=True))
    assert len(_upserts(issuer)) == count
    # Nor when the flag rides inside the access object of a non-user publish.
    run({"mode": "restricted", "emails": [], "allowed_user_ids": [person],
         "notify": {"user_ids": [person]}})
    assert not {"notify_recipients", "notify_user_ids"} & set(_upserts(issuer)[-1])
    assert "notify" not in (uploads[-1]["access"] or {})
    # A list naming nobody the rule holds notifies no one: the people are
    # unchanged, so nothing is written at all.
    count = len(_upserts(issuer))
    run({"mode": "restricted", "emails": [], "allowed_user_ids": [person]},
        notify=ShareNotify(user_ids=(stranger,)))
    assert len(_upserts(issuer)) == count


def test_a_notifying_save_of_outside_emails_alone_still_writes_to_mindshub(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """No teams or people by id means the lambda wrote the whole rule, but a
    save that chose "Notify people" still makes cowork's write so MindsHub
    emails the addresses it added. Without anyone to notify, nothing extra is written."""
    from cowork.services.artifact_audience import ShareNotify

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(
        issuer, folder, scope, project, monkeypatch, view_url="https://view.dev.mindshub.ai/1",
    )

    access = {"mode": "restricted", "emails": ["out@else.com"], "org_allowed": False}
    run(access)
    assert _upserts(issuer) == []
    # An empty list (the app's boolean without lists) writes nothing extra.
    run(access, notify=ShareNotify())
    assert _upserts(issuer) == []
    run({**access, "emails": ["out@else.com", "new@else.com"]}, notify=ShareNotify(emails=("new@else.com",)))
    [body] = _upserts(issuer)
    # The lambda wrote the emails; this write only asks MindsHub to
    # email them.
    assert not {"allowed_emails", "org_allowed", "allowed_team_ids", "allowed_user_ids"} & set(body)
    assert (body["notify_recipients"], body["notify_user_ids"], body["notify_emails"]) == (
        True, [], ["new@else.com"],
    )
    assert (body["artifact_title"], body["artifact_url"]) == ("demo", "https://view.dev.mindshub.ai/1")
    # A public share names nobody to notify: no write.
    run({"mode": "public"}, notify=ShareNotify(emails=("new@else.com",)), owner_save=True)
    assert len(_upserts(issuer)) == 1


@pytest.mark.asyncio
async def test_a_live_sync_after_a_team_is_deleted_and_a_person_leaves_still_saves(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """MindsHub is the one writer of a rule's teams and
    people. An admin deletes the team the artifact is shared with, which the
    MindsHub drops from the rule, and a person on it leaves the org, whose
    id stays on the rule and is refused on the next write that names them. A
    content save then re-publishes the artifact (live sync): it sends
    neither, so it succeeds, with nothing written to the rule's teams or
    people. The owner's next save changes the people without the departed
    one (the access route drops them first): only that list is written, and
    the publish reports MindsHub's answer, without the deleted team."""
    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    person, departed = str(uuid4()), str(uuid4())
    key = run({"mode": "restricted", "emails": [], "allowed_team_ids": [TEAM_ENG],
               "allowed_user_ids": [person, departed]})["artifactKey"]
    issuer.delete_team(TEAM_ENG)
    issuer.leave_org(departed)
    assert issuer.rules[key]["allowed_user_ids"] == [person, departed]
    writes = len(_upserts(issuer))

    saved = await _live_sync(folder, scope, project, monkeypatch)

    assert saved is True
    assert len(_upserts(issuer)) == writes
    assert _entry(folder)["mode"] == "restricted"

    newcomer = str(uuid4())
    result = run({"mode": "restricted", "emails": [], "allowed_user_ids": [person, newcomer]})
    assert _upserts(issuer)[-1]["allowed_user_ids"] == [person, newcomer]
    assert "allowed_team_ids" not in _upserts(issuer)[-1]
    assert (result["allowedTeamIds"], result["allowedUserIds"]) == ([], [person, newcomer])
    assert not _AUDIENCE_KEYS & set(_entry(folder))


def test_the_share_email_link_is_left_to_mindshubs_allowlist(tmp_path):
    """Which viewer hosts the share email links to is MindsHub's allowlist's
    call, so every https view_url within MindsHub's 2048-character bound is
    sent as the lambda returned it, a PR environment's viewer included. A
    link that isn't https, or is longer than MindsHub takes, is left out."""
    from cowork.services.artifact_audience import ShareNotify, share_notice

    def link(url: str) -> str:
        return share_notice(tmp_path, "index.html", url, recipients=ShareNotify(emails=("a@b.co",))).artifact_url

    for url in (
        "https://view.dev.mindshub.ai/r/1", "https://view.staging.mindshub.ai/r/1", "https://view.mindshub.ai/r/1",
        "https://view-pr-x-1.dev.mindshub.ai/r/1",
    ):
        assert link(url) == url
    too_long = "https://view.mindshub.ai/r/" + "a" * 2048
    for url in ("http://view.mindshub.ai/r/1", "https:///r/1", too_long, ""):
        assert link(url) == ""
