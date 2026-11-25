"""A full-stack publish sends credentials only of approved connections."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import unit_session
from cowork.models.artifact_datasource_approval import ArtifactDatasourceApproval
from cowork.schemas.artifact_access import ArtifactAccessRequest
from cowork.models.artifact_owner_choice import AccessMode
from cowork.services import publish
from cowork.services.datasource_approval import AllowedDataVault, DatasourceKey, stored_approvals
from tests._artifact_metadata import write_artifact_metadata

MINDSHUB = "https://api.mindshub.ai"
PUBLIC = ArtifactAccessRequest(mode=AccessMode.PUBLIC)
SCM = DatasourceKey("postgres", "scm")


@pytest.fixture(autouse=True)
def _no_approvals():
    def clear():
        with unit_session(scope=LOCAL_SCOPE) as session:
            for row in session.exec(session.select(ArtifactDatasourceApproval)):
                session.delete(row)
    clear()
    yield
    clear()


class FakeVault:
    records = {
        ("postgres", "scm"): {"_user_label": "Sales DB", "password": "s3cret"},
        ("mysql", "late"): {"password": "late"},
    }

    def __init__(self, records: dict | None = None) -> None:
        if records is not None:
            self.records = records

    def read_record(self, engine, name):
        fields = self.records.get((engine, name))
        return None if fields is None else {"engine": engine, "name": name, "fields": fields}

    def env_for(self, engine, name, *, flat=False):
        fields = self.records.get((engine, name))
        return None if fields is None else {f"DS_{engine}_{name}__PASSWORD".upper(): fields["password"]}

    def load(self, engine, name):
        return self.records.get((engine, name))


@pytest.fixture
def artifacts_base(tmp_path: Path) -> Path:
    base = tmp_path / "proj" / ".anton" / "artifacts"
    base.mkdir(parents=True)
    return base


def _app(base: Path, datasources: list[dict], slug: str = "planner-app") -> Path:
    folder = write_artifact_metadata(
        base / slug, slug=slug, artifact_type="fullstack-stateless-app", datasources=datasources
    )
    (folder / "static").mkdir()
    (folder / "static" / "index.html").write_text("<html>app</html>")
    (folder / "backend.py").write_text("app = None\n")
    (folder / "requirements.txt").write_text("fastapi\n")
    return folder


@pytest.fixture
def app(artifacts_base):
    return _app(artifacts_base, [{"engine": "postgres", "name": "scm"}])


@pytest.fixture
def upload(monkeypatch):
    def fake(path, **kwargs):
        # As anton's publisher does: read each connection metadata.json names.
        if path.is_dir():
            meta = json.loads((path / "metadata.json").read_text())
            for ref in meta.get("datasources", []):
                kwargs["vault"].env_for(ref["engine"], ref["name"])
        return {"view_url": "https://view.example/r/1", "report_id": "rid-1", "md5": "d"}

    sent = mock.Mock(side_effect=fake)
    monkeypatch.setattr("anton.publisher.publish", sent)
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: FakeVault())
    return sent


def _publish(folder, artifacts_base, **kwargs):
    return publish.publish_artifact(
        folder, artifacts_base=artifacts_base, api_key="k", publish_url=MINDSHUB, access=PUBLIC, **kwargs
    )


def _approved(folder, artifacts_base):
    key = publish._authorization_key(folder, artifacts_base, None, None)
    return stored_approvals(artifact_key=key, artifact_dir=folder, scope=None)


def test_unapproved_connections_upload_nothing(app, artifacts_base, upload):
    with pytest.raises(publish.DatasourceApprovalRequired) as exc:
        _publish(app, artifacts_base)
    assert exc.value.detail["pending"] == [
        {"engine": "postgres", "name": "scm", "label": "Sales DB (postgres)"}
    ]
    assert exc.value.artifact_slug == "planner-app"
    upload.assert_not_called()
    assert not (app / ".published.json").exists()


def test_an_approval_is_recorded_and_the_next_publish_needs_none(app, artifacts_base, upload):
    _publish(app, artifacts_base, approved_datasources=frozenset({SCM}))
    assert isinstance(upload.call_args.kwargs["vault"], AllowedDataVault)
    assert _approved(app, artifacts_base) == {SCM}
    _publish(app, artifacts_base)
    assert upload.call_count == 2


def test_an_approval_stays_when_the_upload_fails(app, artifacts_base, upload):
    upload.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError):
        _publish(app, artifacts_base, approved_datasources=frozenset({SCM}))
    assert _approved(app, artifacts_base) == {SCM}


def test_an_approval_never_covers_a_connection_the_app_does_not_name(app, artifacts_base, upload):
    late = DatasourceKey("mysql", "late")
    _publish(app, artifacts_base, approved_datasources=frozenset({SCM, late}))
    assert _approved(app, artifacts_base) == {SCM}


def test_a_connection_named_after_the_check_gets_no_credentials(app, artifacts_base, upload, monkeypatch):
    """`pg/ghost` is not saved in the vault, so it is missing, not unapproved."""
    original = publish.declared_datasources

    def declared_then_rewritten(folder):
        out = original(folder)
        meta = json.loads((folder / "metadata.json").read_text())
        meta["datasources"] += [{"engine": "mysql", "name": "late"}, {"engine": "pg", "name": "ghost"}]
        (folder / "metadata.json").write_text(json.dumps(meta))
        return out

    monkeypatch.setattr(publish, "declared_datasources", declared_then_rewritten)
    out = _publish(app, artifacts_base, approved_datasources=frozenset({SCM}))
    assert out["unapprovedDatasources"] == [{"engine": "mysql", "name": "late", "label": "late (mysql)"}]


def test_two_spellings_are_two_connections(artifacts_base, upload, monkeypatch):
    """`prod` and `Prod` are two vault records with their own credentials."""
    prod, cased = DatasourceKey("postgres", "prod"), DatasourceKey("postgres", "Prod")
    vault = FakeVault({("postgres", "prod"): {"password": "a"}, ("postgres", "Prod"): {"password": "b"}})
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: vault)
    folder = _app(
        artifacts_base, [{"engine": "postgres", "name": "prod"}, {"engine": "postgres", "name": "Prod"}], slug="cased"
    )

    with pytest.raises(publish.DatasourceApprovalRequired) as exc:
        _publish(folder, artifacts_base)
    assert [(p["engine"], p["name"]) for p in exc.value.detail["pending"]] == [("postgres", "prod"), ("postgres", "Prod")]

    with pytest.raises(publish.DatasourceApprovalRequired) as exc:
        _publish(folder, artifacts_base, approved_datasources=frozenset({prod}))
    assert [(p["engine"], p["name"]) for p in exc.value.detail["pending"]] == [("postgres", "Prod")]
    upload.assert_not_called()
    assert _approved(folder, artifacts_base) == frozenset()

    _publish(folder, artifacts_base, approved_datasources=frozenset({prod, cased}))
    assert _approved(folder, artifacts_base) == {prod, cased}


def test_an_approval_of_one_spelling_does_not_cover_another(artifacts_base, upload):
    folder = _app(artifacts_base, [{"engine": "postgres", "name": "scm"}], slug="cased")
    with pytest.raises(publish.DatasourceApprovalRequired):
        _publish(folder, artifacts_base, approved_datasources=frozenset({DatasourceKey("Postgres", "SCM")}))
    upload.assert_not_called()
    assert _approved(folder, artifacts_base) == frozenset()


def test_an_unsaved_connection_is_neither_pending_nor_approved(artifacts_base, upload, monkeypatch):
    """Nothing to upload, so no card; saving it later asks before it is sent."""
    ghost = DatasourceKey("pg", "ghost")
    vault = FakeVault({})
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: vault)
    folder = _app(artifacts_base, [{"engine": "pg", "name": "ghost"}], slug="ghosted")
    monkeypatch.setattr(
        publish, "desktop_artifact_and_base",
        lambda raw, *, scope, adopted=(): SimpleNamespace(artifact=folder, artifacts_base=artifacts_base),
    )

    assert publish.pending_datasources(str(folder), scope=None) == []
    _publish(folder, artifacts_base)
    _publish(folder, artifacts_base, approved_datasources=frozenset({ghost}))
    assert upload.call_count == 2
    assert upload.call_args.kwargs["vault"].env_for("pg", "ghost") is None
    assert _approved(folder, artifacts_base) == frozenset()

    vault.records = {("pg", "ghost"): {"password": "now-saved"}}
    assert publish.pending_datasources(str(folder), scope=None) == [
        {"engine": "pg", "name": "ghost", "label": "ghost (pg)"}
    ]
    with pytest.raises(publish.DatasourceApprovalRequired) as exc:
        _publish(folder, artifacts_base)
    assert exc.value.detail["pending"] == [{"engine": "pg", "name": "ghost", "label": "ghost (pg)"}]
    assert upload.call_count == 2


def test_metadata_anton_rejects_needs_no_card_and_sends_no_credentials(artifacts_base, upload):
    """anton publishes such a folder as HTML; the vault it gets still allows nothing."""
    folder = _app(artifacts_base, [], slug="partial")
    (folder / "metadata.json").write_text(json.dumps({
        "slug": "partial", "name": "partial", "type": "fullstack-stateless-app",
        "datasources": [{"engine": "postgres", "name": "scm"}],
    }))
    _publish(folder, artifacts_base)
    vault = upload.call_args.kwargs["vault"]
    assert isinstance(vault, AllowedDataVault)
    assert vault.env_for("postgres", "scm") is None


def test_a_static_page_needs_no_approval(artifacts_base, upload):
    folder = artifacts_base / "report"
    folder.mkdir()
    (folder / "report.html").write_text("<html>report</html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "report", "name": "report", "type": "html-app"}))
    _publish(folder, artifacts_base)
    upload.assert_called_once()


def test_the_pending_list_for_the_dialog(app, artifacts_base, monkeypatch):
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: FakeVault())
    monkeypatch.setattr(
        publish, "desktop_artifact_and_base",
        lambda raw, *, scope, adopted=(): SimpleNamespace(artifact=app, artifacts_base=artifacts_base),
    )
    assert publish.pending_datasources(str(app), scope=None) == [
        {"engine": "postgres", "name": "scm", "label": "Sales DB (postgres)"}
    ]
    key = publish._authorization_key(app, artifacts_base, None, None)
    from cowork.services.datasource_approval import record_approval
    record_approval(artifact_key=key, artifact_dir=app, datasources={SCM}, scope=None)
    assert publish.pending_datasources(str(app), scope=None) == []


def test_a_folder_that_turns_fullstack_after_the_check_gets_no_credentials(artifacts_base, monkeypatch):
    """cowork-server saw a static page; anton re-reads metadata.json at upload."""
    folder = artifacts_base / "flip.html"
    folder.mkdir()
    (folder / "index.html").write_text("<html>page</html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "flip", "name": "flip", "type": "html-app"}))
    seen = {}

    def fake(path, **kwargs):
        write_artifact_metadata(
            folder, slug="flip", artifact_type="fullstack-stateless-app",
            datasources=[{"engine": "postgres", "name": "scm"}],
        )
        seen["env"] = kwargs["vault"].env_for("postgres", "scm")
        return {"view_url": "https://view.example/r/1", "report_id": "rid-1", "md5": "d"}

    monkeypatch.setattr("anton.publisher.publish", mock.Mock(side_effect=fake))
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: FakeVault())
    _publish(folder, artifacts_base)
    assert seen["env"] is None


def test_a_partial_approval_raises_and_records_nothing(artifacts_base, upload):
    folder = _app(
        artifacts_base,
        [{"engine": "postgres", "name": "scm"}, {"engine": "mysql", "name": "late"}],
        slug="two",
    )
    with pytest.raises(publish.DatasourceApprovalRequired) as exc:
        _publish(folder, artifacts_base, approved_datasources=frozenset({SCM}))
    assert [p["name"] for p in exc.value.detail["pending"]] == ["late"]
    upload.assert_not_called()
    assert _approved(folder, artifacts_base) == frozenset()


def _published_before_approvals(folder: Path, artifacts_base: Path, *, history: bool = True) -> float:
    """A page published before approvals were asked for: its .published.json
    entry and, unless ``history`` is False, this server's record of that
    publish. Returns when it was published."""
    target, published_dir, key, _fullstack = publish._resolve_publish_target(folder, container_dirs=[artifacts_base])
    (published_dir / ".published.json").write_text(json.dumps({
        key: {"report_id": "rid-0", "url": "https://view.example/r/0", "published": True, "mode": "public"},
    }))
    published_at = time.time()
    if history:
        state = publish._load_state(scope=None)
        item = {
            "artifact": str(target),
            "reportId": "rid-0",
            "publishedAt": datetime.fromtimestamp(published_at, timezone.utc).isoformat(),
        }
        state["publish_history"] = [item, *state.get("publish_history", [])]
        publish._save_state(state, scope=None)
    return published_at


def _touch_metadata(folder: Path, at: float) -> None:
    os.utime(folder / "metadata.json", (at, at))


def test_a_page_published_before_approvals_keeps_its_connections(app, artifacts_base, upload, monkeypatch):
    published_at = _published_before_approvals(app, artifacts_base)
    _touch_metadata(app, published_at - 60)
    monkeypatch.setattr(
        publish, "desktop_artifact_and_base",
        lambda raw, *, scope, adopted=(): SimpleNamespace(artifact=app, artifacts_base=artifacts_base),
    )
    assert publish.pending_datasources(str(app), scope=None) == []
    # The dialog's read records nothing; the publish does.
    assert _approved(app, artifacts_base) == frozenset()
    _publish(app, artifacts_base)
    assert upload.call_args.kwargs["vault"].env_for("postgres", "scm") is not None
    assert _approved(app, artifacts_base) == {SCM}


def test_a_page_whose_metadata_changed_since_its_publish_asks(app, artifacts_base, upload):
    published_at = _published_before_approvals(app, artifacts_base)
    _touch_metadata(app, published_at + 60)
    with pytest.raises(publish.DatasourceApprovalRequired):
        _publish(app, artifacts_base)
    upload.assert_not_called()


def test_an_entry_this_server_never_published_inherits_nothing(app, artifacts_base, upload):
    """The agent can write .published.json, not the server's publish history."""
    published_at = _published_before_approvals(app, artifacts_base, history=False)
    _touch_metadata(app, published_at - 60)
    with pytest.raises(publish.DatasourceApprovalRequired):
        _publish(app, artifacts_base)
    upload.assert_not_called()


def test_a_stored_approval_stops_inheriting(artifacts_base, upload):
    folder = _app(
        artifacts_base,
        [{"engine": "postgres", "name": "scm"}, {"engine": "mysql", "name": "late"}],
        slug="two",
    )
    key = publish._authorization_key(folder, artifacts_base, None, None)
    from cowork.services.datasource_approval import record_approval
    record_approval(artifact_key=key, artifact_dir=folder, datasources={SCM}, scope=None)
    published_at = _published_before_approvals(folder, artifacts_base)
    _touch_metadata(folder, published_at - 60)
    with pytest.raises(publish.DatasourceApprovalRequired) as exc:
        _publish(folder, artifacts_base)
    assert [p["name"] for p in exc.value.detail["pending"]] == ["late"]


def test_a_page_published_since_approvals_inherits_nothing(app, artifacts_base, upload, monkeypatch):
    """A connection saved after such a publish was never sent, so it asks."""
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: FakeVault({}))
    _publish(app, artifacts_base)
    _touch_metadata(app, time.time() - 60)
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: FakeVault())
    with pytest.raises(publish.DatasourceApprovalRequired):
        _publish(app, artifacts_base)
    assert upload.call_count == 1
