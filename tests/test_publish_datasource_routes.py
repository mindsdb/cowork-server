"""The publish routes: the pending list, and an approval that needs the app secret."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork import app_secret
from cowork.api.v1.endpoints import publish as publish_ep
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import unit_session
from cowork.models.artifact_datasource_approval import ArtifactDatasourceApproval
from cowork.services import publish
from cowork.services.datasource_approval import AllowedDataVault, DatasourceKey
from tests._artifact_metadata import write_artifact_metadata

SECRET = "ab" * 16
BODY = {"path": "/p/app", "approved_datasources": [{"engine": "Postgres", "name": "SCM"}]}


@pytest.fixture(autouse=True)
def _no_approvals():
    def clear():
        with unit_session(scope=LOCAL_SCOPE) as session:
            for row in session.exec(session.select(ArtifactDatasourceApproval)):
                session.delete(row)
    clear()
    yield
    clear()


@pytest.fixture
def client(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")
    context = (tmp_path, tmp_path, "k", "https://api.mindshub.ai")
    monkeypatch.setattr(publish_ep, "_desktop_context", lambda raw, *, scope: context)
    return TestClient(app)


@pytest.fixture
def publish_calls(monkeypatch):
    sent = mock.Mock(return_value={"url": "https://view.example/r/1"})
    monkeypatch.setattr(publish_ep, "_publish", sent)
    monkeypatch.setattr(publish_ep, "_update", sent)
    return sent


@pytest.fixture
def protected(monkeypatch):
    monkeypatch.setattr(app_secret, "_secret", SECRET)


@pytest.mark.parametrize("route", ["/api/v1/publish/", "/api/v1/publish/update"])
def test_an_approval_needs_the_app_secret(client, publish_calls, protected, route):
    res = client.post(route, json=BODY)
    assert res.status_code == 403
    publish_calls.assert_not_called()

    res = client.post(route, json=BODY, headers={app_secret.HEADER: SECRET})
    assert res.status_code == 200
    # The exact spelling: `Postgres/SCM` and `postgres/scm` can be two connections.
    assert publish_calls.call_args.kwargs["approved_datasources"] == frozenset({DatasourceKey("Postgres", "SCM")})
    assert publish_calls.call_args.kwargs["approved_datasources"] != frozenset({DatasourceKey("postgres", "scm")})


def test_a_publish_without_approval_needs_no_secret(client, publish_calls, protected):
    res = client.post("/api/v1/publish/", json={"path": "/p/app"})
    assert res.status_code == 200
    assert publish_calls.call_args.kwargs["approved_datasources"] is None


def test_a_server_without_a_secret_takes_an_approval(client, publish_calls, monkeypatch):
    monkeypatch.setattr(app_secret, "_secret", None)
    assert client.post("/api/v1/publish/", json=BODY).status_code == 200


def test_unreadable_metadata_answers_400(client, monkeypatch):
    def unreadable(*_a, **_k):
        raise ValueError("Artifact metadata is unreadable")

    monkeypatch.setattr(publish_ep, "_pending", unreadable)
    res = client.get("/api/v1/publish/datasources", params={"path": "/p/app"})
    assert res.status_code == 400


def test_an_unreadable_approval_record_answers_503(client, monkeypatch):
    from cowork.services.datasource_approval import DatasourceApprovalUnavailable

    def unavailable(*_a, **_k):
        raise DatasourceApprovalUnavailable("Could not read the connections approved for this artifact")

    monkeypatch.setattr(publish_ep, "_publish", unavailable)
    monkeypatch.setattr(publish_ep, "_update", unavailable)
    monkeypatch.setattr(publish_ep, "_pending", unavailable)
    assert client.post("/api/v1/publish/", json={"path": "/p/app"}).status_code == 503
    assert client.post("/api/v1/publish/update", json={"path": "/p/app"}).status_code == 503
    assert client.get("/api/v1/publish/datasources", params={"path": "/p/app"}).status_code == 503


class _Vault:
    records = {("postgres", "scm"): {"_user_label": "Sales DB", "password": "s3cret"}}

    def read_record(self, engine, name):
        fields = self.records.get((engine, name))
        return None if fields is None else {"engine": engine, "name": name, "fields": fields}

    def env_for(self, engine, name, *, flat=False):
        return {"DS_PASSWORD": "s3cret"} if (engine, name) in self.records else None


@pytest.fixture
def real_app(monkeypatch, tmp_path: Path):
    """A full-stack app on disk naming postgres/scm, a vault that holds it,
    and anton's upload replaced; the routes run the real publish service."""
    base = tmp_path / "proj" / ".anton" / "artifacts"
    folder = write_artifact_metadata(
        base / "planner-app",
        slug="planner-app",
        artifact_type="fullstack-stateless-app",
        datasources=[{"engine": "postgres", "name": "scm"}],
    )
    (folder / "static").mkdir()
    (folder / "static" / "index.html").write_text("<html>app</html>")
    (folder / "backend.py").write_text("app = None\n")
    (folder / "requirements.txt").write_text("fastapi\n")
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: _Vault())
    monkeypatch.setattr(
        publish, "desktop_artifact_and_base", lambda raw, *, scope: publish.DesktopArtifact(folder, base)
    )
    upload = mock.Mock(return_value={"view_url": "https://view.example/r/1", "report_id": "rid-1", "md5": "d"})
    monkeypatch.setattr("anton.publisher.publish", upload)
    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")
    context = (folder, base, "k", "https://api.mindshub.ai")
    monkeypatch.setattr(publish_ep, "_desktop_context", lambda raw, *, scope: context)
    return SimpleNamespace(client=TestClient(app), path=str(folder), upload=upload)


def test_an_approved_connection_is_sent_and_leaves_the_pending_list(real_app, protected):
    pending = real_app.client.get("/api/v1/publish/datasources", params={"path": real_app.path})
    assert pending.status_code == 200
    assert pending.json() == {"pending": [{"engine": "postgres", "name": "scm", "label": "Sales DB (postgres)"}]}

    body = {
        "path": real_app.path,
        "access": {"mode": "public"},
        "approved_datasources": [{"engine": "postgres", "name": "scm"}],
    }
    res = real_app.client.post("/api/v1/publish/", json=body, headers={app_secret.HEADER: SECRET})
    assert res.status_code == 200
    vault = real_app.upload.call_args.kwargs["vault"]
    assert isinstance(vault, AllowedDataVault)
    assert vault.env_for("postgres", "scm") == {"DS_PASSWORD": "s3cret"}

    pending = real_app.client.get("/api/v1/publish/datasources", params={"path": real_app.path})
    assert pending.json() == {"pending": []}


def test_an_unapproved_connection_answers_409_with_the_list(real_app, protected):
    # Approving another spelling leaves the saved connection pending.
    body = {
        "path": real_app.path,
        "access": {"mode": "public"},
        "approved_datasources": [{"engine": "Postgres", "name": "SCM"}],
    }
    res = real_app.client.post("/api/v1/publish/", json=body, headers={app_secret.HEADER: SECRET})
    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "datasource_approval_required"
    assert res.json()["detail"]["pending"] == [{"engine": "postgres", "name": "scm", "label": "Sales DB (postgres)"}]
    real_app.upload.assert_not_called()
