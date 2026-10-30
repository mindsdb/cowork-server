"""Real SQL aliases keep mutable metadata out of the global authorization key."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import httpx
import pytest
from sqlmodel import Session, select

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
    """HTTP boundary fake; records ownership, grants and issuance retries.

    It keeps each artifact's rule the way the auth service does. Its upsert
    replaces a sharing field the caller sends and keeps one it leaves out,
    and echoes the org flag, teams and people; its read answers the rule as
    stored, or 404 ``no_rule``; its delete removes it. ``lambda_publish`` is
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
        key = body["artifact_id"].replace("artifact-draft/", "artifact/")
        expected = self.bindings.get(key)
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


@pytest.mark.asyncio
async def test_an_authorization_outage_still_answers_503(issuer, local_id, owned_artifact):
    """An auth service that is set up but failing stays a 503, so the outage
    still shows as one."""
    session, project, _conversation, folder = owned_artifact
    issuer.failure = True

    with pytest.raises(HTTPException) as err:
        await artifact_workspace.enable_artifact_comments(str(project.id), local_id, session)

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
        str(project.id), local_id, session
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
        access={"mode": "restricted", "emails": ["reviewer@example.com"]},
        scope=scope,
        project_id=str(project.id),
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
    with identities._session(scope) as identity_session:
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
):
    """``(run, uploads)``: run(access, **kwargs) publishes ``folder`` in org
    mode through anton's publisher, faked as the publish lambda (which writes
    the rule's emails and org flag), and ``uploads`` lists what it was given.
    The lambda answers ``view_url``, and ``echoed_key`` in place of the key
    it was given when set."""
    uploads = []

    def publish(_source, **kwargs):
        uploads.append(kwargs)
        issuer.lambda_publish(kwargs)
        return {"artifact_key": echoed_key or kwargs["artifact_key"], "report_id": "report-1", "view_url": view_url}

    monkeypatch.setattr("anton.publisher.publish", publish)
    monkeypatch.setattr("cowork.services.artifact_autopublish._active_workspace_id", lambda scope: None)

    def run(access, **kwargs):
        return publish_artifact(
            folder, artifacts_base=folder.parent, api_key="key", publish_url="https://view.test",
            scope=scope, project_id=str(project.id), access=access, **kwargs,
        )

    return run, uploads


def _upserts(issuer):
    return [body for path, body in issuer.calls if path == "/v1/internal/artifact-access/"]


def _entry(folder):
    return json.loads((folder / ".published.json").read_text())["index.html"]


@pytest.mark.asyncio
async def test_get_access_on_an_unpublished_artifact_asks_the_auth_service_nothing(
    issuer, scope, local_id, owned_artifact
):
    session, project, _conversation, _folder = owned_artifact
    state = await artifact_workspace.artifact_access(str(project.id), local_id, session)
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
        SimpleNamespace(scope=scope), folder, project_id=str(project.id)
    )


# What an org-mode entry never keeps: who the artifact is shared with.
_AUDIENCE_KEYS = {"emails", "org_allowed", "owner_only", "allowed_team_ids", "allowed_user_ids"}


def test_a_team_audience_is_published_owner_only_and_widened_by_the_access_service(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """ENG-3787: anton's publisher knows no teams, so a restricted publish to
    teams alone reaches it owner-only, and cowork then writes the rule's
    ``allowed_team_ids`` alone, under the ArtifactAccess names; the emails and
    the org flag are the lambda's. The entry keeps no audience. A republish
    reads the rule, sends its org flag, writes nothing and keeps the version;
    changing the teams bumps it."""
    from cowork.services.publish import published_artifact_access

    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)

    result = run({"mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": ["team-eng"]})
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": [], "org_allowed": False}
    path, body = issuer.calls[-1]
    assert path == "/v1/internal/artifact-access/"
    assert body["allowed_team_ids"] == ["team-eng"]
    assert not {"team_ids", "org_wide", "org_allowed", "allowed_emails", "allowed_user_ids"} & set(body)
    assert body["artifact_id"] == result["artifactKey"]
    first_version = body["access_version"]
    assert (result["allowedTeamIds"], result["orgAllowed"]) == (["team-eng"], False)
    assert not _AUDIENCE_KEYS & set(_entry(folder))

    writes = len(_upserts(issuer))
    run(published_artifact_access(folder, artifacts_base=folder.parent, org_mode=True))
    assert len(_upserts(issuer)) == writes
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": [], "org_allowed": False}
    assert _entry(folder)["access_version"] == first_version

    result = run({"mode": "restricted", "emails": [], "org_allowed": True, "allowed_team_ids": ["team-ops"]})
    body = issuer.calls[-1][1]
    assert body["allowed_team_ids"] == ["team-ops"]
    assert body["access_version"] > first_version
    assert (result["allowedTeamIds"], result["orgAllowed"]) == (["team-ops"], True)
    run(published_artifact_access(folder, artifacts_base=folder.parent, org_mode=True))
    assert uploads[-1]["access"] == {"mode": "restricted", "emails": [], "org_allowed": True}


@pytest.mark.asyncio
async def test_a_live_sync_of_a_planted_entry_sends_what_the_rule_holds(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """ENG-3787: .published.json sits in a folder the pod can write. An agent
    that writes everyone, an outside address and a team into the entry gets
    none of it published: a content save re-publishes the artifact (live
    sync) with the emails and the org flag the auth service's rule holds,
    and writes no team."""
    _session, project, _conversation, folder = owned_artifact
    run, uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": False})
    _plant_entry(folder, emails=["mal@else.com"], org_allowed=True, owner_only=False, allowed_team_ids=["team-mal"])
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
    """ENG-3787: .published.json sits in a folder the pod can write. An agent
    that changes a restricted artifact's entry to public, or to a password it
    chose, gets nothing published: in hosted org mode only a restricted
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
    run({"mode": "restricted", "emails": ["a@else.com"], "org_allowed": True, "allowed_team_ids": ["team-eng"]},
        owner_save=True)
    _plant_entry(folder, **planted)

    state = await artifact_workspace.artifact_access(str(project.id), local_id, session)

    assert (state["accessMode"], state["accessProtected"], state["accessPassword"]) == ("restricted", False, "")
    assert (state["accessEmails"], state["orgAllowed"], state["ownerOnly"], state["allowedTeamIds"]) == (
        ["a@else.com"], True, False, ["team-eng"],
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

    state = await artifact_workspace.artifact_access(str(project.id), fresh, session)

    assert (state["accessMode"], state["accessEmails"]) == ("restricted", ["a@else.com"])


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
    state = await artifact_workspace.artifact_access(str(project.id), local_id, session)
    assert (state["accessMode"], state["accessEmails"]) == ("public", [])


def test_the_rule_is_read_and_written_under_one_key_whatever_the_lambda_echoes(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """The read before a publish and the team and people write after it both
    take the rule's key from publish._rule_key, the artifact's identity
    first. A lambda that echoes another key moves neither."""
    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(
        issuer, folder, scope, project, monkeypatch, echoed_key=f"artifact/{uuid4()}",
    )
    team = {"mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": ["team-eng"]}
    for teams in (["team-eng"], ["team-ops"]):
        run({**team, "allowed_team_ids": teams}, owner_save=True)

    canonical = identities.existing_authorization_key(local_id, scope)
    rule_calls = [
        (path.rsplit("/artifact-access/", 1)[1], body["artifact_id"]) for path, body in issuer.calls
        if path.endswith(("/artifact-access/", "/read/"))
    ]
    assert rule_calls == [("read/", canonical), ("", canonical), ("read/", canonical), ("", canonical)]
    assert issuer.rules[canonical]["allowed_team_ids"] == ["team-ops"]


def test_a_save_plans_against_the_rule_its_check_read_without_reading_it_again(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """The access route reads the rule once, under the publish lock, to check
    a save, and hands it to publish_artifact as ``stored``: the publish
    plans against that rule and asks the auth service for none."""
    from cowork.services.artifact_access import ArtifactRule

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    team = {"mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": ["team-eng"]}
    key = run(team, owner_save=True)["artifactKey"]
    calls = len(issuer.calls)

    checked = ArtifactRule(org_allowed=False, allowed_team_ids=["team-eng"], allowed_user_ids=[], allowed_emails=[])
    run(team, owner_save=True, stored=checked)
    assert [path for path, _body in issuer.calls[calls:]] == []

    # The rule the check read already holds the team, so nothing is written;
    # one that doesn't is planned as a change and written.
    run(team, owner_save=True, stored=ArtifactRule.nobody())
    assert [(path, body["allowed_team_ids"]) for path, body in issuer.calls[calls:]] == [
        ("/v1/internal/artifact-access/", ["team-eng"]),
    ]
    assert issuer.rules[key]["allowed_team_ids"] == ["team-eng"]


@pytest.mark.asyncio
async def test_a_re_publish_that_cannot_read_the_rule_fails_before_anything_is_uploaded(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """A live sync or autopublish never guesses who an artifact is shared
    with: while the auth service's rule can't be read, it fails, and the
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
    """ENG-3787: a team the auth service refuses (deleted since the share
    dialog listed it) leaves the rule's teams as they were. The entry's
    version moved with the publish, so the retry with a team that exists
    bumps it again, past every grant issued before."""
    from cowork.services.artifact_access import ArtifactAudienceRejected

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)

    key = run({"mode": "restricted", "emails": [], "allowed_team_ids": ["team-eng"]})["artifactKey"]
    granted = _entry(folder)["access_version"]

    issuer.delete_team("team-ops")
    with pytest.raises(ArtifactAudienceRejected):
        run({"mode": "restricted", "emails": [], "allowed_team_ids": ["team-ops"]})
    assert issuer.rules[key]["allowed_team_ids"] == ["team-eng"]
    refused = _entry(folder)["access_version"]
    assert refused > granted

    run({"mode": "restricted", "emails": [], "allowed_team_ids": ["team-qa"]})
    assert issuer.rules[key]["allowed_team_ids"] == ["team-qa"]
    assert _entry(folder)["access_version"] > refused


def test_a_republish_after_an_unpublish_sends_the_whole_audience_again(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """ENG-3787: an unpublish deletes the artifact's rule (the publish
    lambda's delete), while the soft-deleted entry keeps its mode. The owner
    then republishes it to the same team: the stored rule holds no team, so
    the team is written again, and the team's members open it."""
    from cowork.services.publish import unpublish_artifact

    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    team = {"mode": "restricted", "emails": [], "org_allowed": False, "allowed_team_ids": ["team-eng"]}
    key = run(team)["artifactKey"]
    assert issuer.rules[key]["allowed_team_ids"] == ["team-eng"]

    monkeypatch.setattr("anton.publisher.unpublish", lambda *_a, **_kw: issuer.rules.pop(key))
    unpublish_artifact(
        folder, artifacts_base=folder.parent, api_key="key", publish_url="https://view.test", scope=scope,
    )
    assert key not in issuer.rules
    assert (_entry(folder)["published"], _entry(folder)["mode"]) == (False, "restricted")

    writes = len(_upserts(issuer))
    result = run(team)
    assert len(_upserts(issuer)) == writes + 1
    assert _upserts(issuer)[-1]["allowed_team_ids"] == ["team-eng"]
    assert issuer.rules[key]["allowed_team_ids"] == ["team-eng"]
    assert result["allowedTeamIds"] == ["team-eng"]


def test_a_notifying_save_asks_the_access_service_to_email_with_title_and_url(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """ENG-3787: only a person's save with "Notify people" chosen sends
    ``notify_recipients`` with the people to tell (and the artifact's title
    and published URL) on cowork's write of the rule. The lambda's upsert
    never carries it, and neither does a re-publish from the stored access
    (live sync, autopublish)."""
    from cowork.services.publish import ShareNotify, published_artifact_access

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


def test_a_notifying_save_of_outside_emails_alone_still_writes_to_the_access_service(
    issuer, scope, local_id, owned_artifact, monkeypatch
):
    """No teams or people by id means the lambda wrote the whole rule, but a
    save that chose "Notify people" still makes cowork's write so the auth
    service emails the addresses it added. Without anyone to notify, nothing extra is written."""
    from cowork.services.publish import ShareNotify

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
    # The lambda wrote the emails; this write only asks the auth service to
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
    """ENG-3787: the auth service is the one writer of a rule's teams and
    people. An admin deletes the team the artifact is shared with, which the
    auth service drops from the rule, and a person on it leaves the org, whose
    id stays on the rule and is refused on the next write that names them. A
    content save then re-publishes the artifact (live sync): it sends
    neither, so it succeeds, with nothing written to the rule's teams or
    people. The owner's next save changes the people without the departed
    one (the access route drops them first): only that list is written, and
    the publish reports the auth service's answer, without the deleted team."""
    _session, project, _conversation, folder = owned_artifact
    run, _uploads = _org_publisher(issuer, folder, scope, project, monkeypatch)
    person, departed = str(uuid4()), str(uuid4())
    key = run({"mode": "restricted", "emails": [], "allowed_team_ids": ["team-eng"],
               "allowed_user_ids": [person, departed]})["artifactKey"]
    issuer.delete_team("team-eng")
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


def test_the_share_email_links_only_a_viewer_url(tmp_path):
    """The auth service links the share email only to the viewer hosts (dev
    and PR envs share dev's), so any other view_url is left out rather than
    sent."""
    from cowork.services.publish import _share_notice

    for url in ("https://view.dev.mindshub.ai/r/1", "https://view.staging.mindshub.ai/r/1", "https://view.mindshub.ai/r/1"):
        assert _share_notice(tmp_path, "index.html", url)["artifact_url"] == url
    for url in ("http://view.mindshub.ai/r/1", "https://view.test/1", "https://evil.example/view.mindshub.ai", ""):
        assert _share_notice(tmp_path, "index.html", url)["artifact_url"] == ""
