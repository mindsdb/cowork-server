"""Real SQL aliases keep mutable metadata out of the global authorization key."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import httpx
import pytest
from sqlmodel import Session

from cowork.api.v1.endpoints import artifact_workspace, artifacts, comments
from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope, get_scoped_session
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services import (
    artifact_access,
    artifact_authorization_identity as identities,
)
from cowork.services.artifact_access import ArtifactAccessUnavailable
from cowork.services.artifact_identity import artifact_key
from cowork.services.publish import publish_artifact


class AuthIssuer:
    """HTTP boundary fake; records ownership, grants and issuance retries."""

    def __init__(self):
        self.calls = []
        self.bindings = {}
        self.allocations = {}
        self.failure = None
        self.drop_allocation_reply = False
        self.barrier = None
        self.lock = Lock()

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
        key = body["artifact_id"].replace("artifact-draft/", "artifact/")
        expected = self.bindings.get(key)
        if expected is None:
            return httpx.Response(409, json={"code": "artifact_ownership_unclaimed"})
        if expected != owner:
            return httpx.Response(409, json={"code": "artifact_ownership_conflict"})
        if path.endswith("/claim/"):
            assert body["create_if_missing"] is False
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
    with identities._session(scope) as session:
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
    with identities._session(scope) as session:
        assert identities._row(session, local_id) is None


def test_unavailable_auth_preserves_pending_nonce_for_retry(issuer, scope, local_id):
    issuer.failure = True
    with pytest.raises(ArtifactAccessUnavailable):
        _ensure(local_id, scope)
    with identities._session(scope) as session:
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
        str(project.id), local_id, session
    )
    canonical = identities.existing_authorization_key(local_id, scope)
    assert granted["artifactKey"] == artifact_key(local_id)
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
    )
    assert seen["artifact_key"] == canonical
    assert comments.resolve_comments_route(
        "artifact", local_id, session=session
    ) == tuple(canonical.split("/", 1))

    async def mint(_self):
        return "test-publish-key"

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey.get", mint)
    monkeypatch.setattr("anton.publisher.unpublish", lambda *_a, **_kw: None)
    await artifacts.delete_artifact_for_request(
        session, local_id, project_id=project.id
    )
    assert issuer.calls[-1][0].endswith("/delete/")
    assert issuer.calls[-1][1] == {
        "artifact_id": canonical.replace("artifact/", "artifact-draft/"),
        "owner_keycloak_id": scope.user_id,
        "organization_id": scope.org_id,
    }
    assert not folder.exists()
    assert identities.existing_authorization_key(local_id, scope) == canonical


def test_publish_refuses_another_conversation_owner_before_allocating(
    issuer, scope, local_id, owned_artifact
):
    _session, _project, _conversation, folder = owned_artifact
    peer = TenantScope(org_mode=True, org_id=scope.org_id, user_id=str(uuid4()))
    with pytest.raises(ArtifactAccessUnavailable, match="Only the artifact owner"):
        identities.publish_authorization_key(local_id, folder.parent, peer)
    assert issuer.calls == []


@pytest.mark.asyncio
async def test_live_editor_sync_preserves_canonical_identity_and_refuses_peer(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    from cowork.services.artifact_revisions import current_source

    session, project, _conversation, folder = owned_artifact
    await artifact_workspace.enable_artifact_comments(
        str(project.id), local_id, session
    )
    canonical = identities.existing_authorization_key(local_id, scope)
    uploads = []

    def publish(source, **kwargs):
        uploads.append((Path(source).read_text(), kwargs))
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
        access={"mode": "restricted", "emails": ["reviewer@example.com"]},
        scope=scope,
    )
    metadata = json.loads((folder / "metadata.json").read_text())
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
                str(project.id), local_id, body, peer_session
            )
        assert excinfo.value.status_code == 403
    assert len(uploads) == 1
    authority_calls = len(issuer.calls)

    saved = await artifact_workspace.update_artifact_source(
        str(project.id), local_id, body, session
    )
    assert saved["content"] == body.content
    assert len(uploads) == 2
    assert uploads[-1][0] == body.content
    assert uploads[0][1]["artifact_key"] == uploads[1][1]["artifact_key"] == canonical
    assert uploads[0][1]["access"] == uploads[1][1]["access"]
    assert uploads[1][1]["report_id"] == "report-1"
    assert len(issuer.calls) == authority_calls


def test_first_publish_allocates_before_upload_without_requiring_a_draft_grant(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    _session, _project, _conversation, folder = owned_artifact
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
    )
    assert uploads == [identities.existing_authorization_key(local_id, scope)]
    assert result["artifactKey"] == uploads[0]
    assert uploads[0] != artifact_key(local_id)
    assert len(issuer.calls) == 2


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
    app.dependency_overrides[get_scoped_session] = lambda: SimpleNamespace(scope=scope)
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
    app.dependency_overrides[get_scoped_session] = lambda: SimpleNamespace(scope=scope)
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
        "artifact", local_id, session=SimpleNamespace(scope=foreign)
    ) == ("artifact", local_id)
    assert canonical != artifact_key(local_id)
    assert len(issuer.calls) == 2


def test_unknown_external_canonical_key_remains_subject_to_upstream_authorization(
    issuer, scope, monkeypatch
):
    external = str(uuid4())
    monkeypatch.setattr(comments, "_org_mode", lambda: True)
    assert comments.resolve_comments_route(
        "artifact", external, session=SimpleNamespace(scope=scope)
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
