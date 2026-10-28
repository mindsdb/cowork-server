"""Trusted publisher ownership on deployments with authenticated local testers."""
from __future__ import annotations

import asyncio
import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, get_ident
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, select

from cowork.api.v1.endpoints import publish as routes
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.session import get_engine
from cowork.models.published_page import PublishedPageOwner
from cowork.models.setting import Setting
from cowork.services import artifacts, published_page_ownership as ownership
from cowork.services import publish
from cowork.services.artifacts import ProjectArtifacts

ALICE = TenantScope(personal=True, org_id="44e92829-e701-599b-88f9-69f5bf1165eb", user_id="9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d")
BOB = TenantScope(personal=True, org_id="55e92829-e701-599b-88f9-69f5bf1165eb", user_id="8a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d")


@pytest.fixture
def publisher(personal_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTON_COWORK_STATE_DIR", str(tmp_path / "history"))
    get_app_settings.cache_clear()
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: object())
    endpoint = f"https://publisher-{uuid4().hex}.example"
    pages = {}
    uploads = []
    deletes = []

    def upload(path, **kwargs):
        uploads.append(copy.deepcopy(kwargs))
        identifier = kwargs.get("report_id") or uuid4().hex
        pages[identifier] = {"content": Path(path).read_text(), "access": copy.deepcopy(kwargs["access"])}
        return {"report_id": identifier, "view_url": f"{endpoint}/view/{identifier}/", "md5": "content-hash", "password_hash": "owner-password-hash"}

    def delete(identifier, **kwargs):
        deletes.append(identifier)
        pages.pop(identifier, None)
        return {"deleted": True}

    monkeypatch.setattr("anton.publisher.publish", upload)
    monkeypatch.setattr("anton.publisher.unpublish", delete)
    monkeypatch.setattr(routes, "_desktop_context", lambda path, *, scope: (Path(path), tmp_path, "test-key", endpoint))

    class Publisher:
        def page(self, name="page"):
            folder = tmp_path / name
            folder.mkdir()
            (folder / "metadata.json").write_text(json.dumps({"id": str(uuid4()), "type": "html", "primary": "index.html"}))
            target = folder / "index.html"
            target.write_text("<html><body>original</body></html>")
            return target

        def send(self, page, scope=ALICE, **kwargs):
            return publish.publish_artifact(page, artifacts_base=tmp_path, api_key="test-key", publish_url=endpoint, scope=scope, **kwargs)

        def remove(self, page, scope=ALICE):
            return asyncio.run(routes.unpublish_artifact(scope=scope, path=str(page)))

        def entry(self, page):
            return json.loads((page.parent / ".published.json").read_text())[page.name]

        def plant(self, page, entry):
            (page.parent / ".published.json").write_text(json.dumps({page.name: entry}))

    result = Publisher()
    result.pages, result.uploads, result.deletes = pages, uploads, deletes
    result.endpoint, result.root = endpoint, tmp_path
    return result


def _source(root: Path) -> ProjectArtifacts:
    return ProjectArtifacts(
        base=root, project_id="project-1", project_name=root.name,
        trusted_anchor=root.parent, root_parts=(root.name,),
    )


def _delete(publisher, page, scope, route):
    """Delete the page's whole artifact folder through one of the two delete routes."""
    if route == "slug":
        artifacts.delete_artifact_from_source(
            _source(publisher.root), page.parent.name, expected_artifact_id=None,
            api_key="test-key", publish_url=publisher.endpoint, scope=scope,
        )
    else:
        artifacts.delete_artifact(
            page.parent, artifacts_base=publisher.root, api_key="test-key",
            publish_url=publisher.endpoint, scope=scope,
        )


@pytest.mark.parametrize("operation", ["publish", "delete"])
def test_another_tester_cannot_mutate_the_original_page(publisher, operation):
    page = publisher.page()
    publisher.send(page, access={"mode": "password", "password": "owner-password"})
    before_remote = copy.deepcopy(publisher.pages)
    before_local = (page.parent / ".published.json").read_bytes()
    before_uploads = len(publisher.uploads)
    with pytest.raises(publish.PublishOwnedByOtherUser):
        publisher.send(page, BOB, access={"mode": "public"}) if operation == "publish" else publisher.remove(page, BOB)
    assert publisher.pages == before_remote
    assert (page.parent / ".published.json").read_bytes() == before_local
    assert len(publisher.uploads) == before_uploads
    assert publisher.deletes == []


def test_owner_keeps_link_and_password_hash_across_republish_and_unpublish(publisher):
    page = publisher.page()
    first = publisher.send(page, access={"mode": "password", "password": "owner-password"})
    page.write_text("<html><body>edited</body></html>")
    again = publisher.send(page, access={"mode": "password", "password": "owner-password"})
    assert again["url"] == first["url"]
    assert publisher.uploads[-1]["previous_access"]["password_hash"] == "owner-password-hash"
    publisher.remove(page)
    assert publisher.pages == {}
    restored = publisher.send(page, access={"mode": "password", "password": "owner-password"})
    assert restored["url"] == first["url"]


@pytest.mark.parametrize("copy_metadata", [False, True])
def test_a_forged_page_id_or_complete_folder_copy_publishes_a_new_page(publisher, copy_metadata):
    original = publisher.page("original")
    first = publisher.send(original)
    before = copy.deepcopy(publisher.pages)
    other = publisher.page("copy")
    if copy_metadata:
        (other.parent / "metadata.json").write_bytes((original.parent / "metadata.json").read_bytes())
    publisher.plant(other, publisher.entry(original))
    result = publisher.send(other, BOB)
    assert result["url"] != first["url"]
    assert publisher.uploads[-1]["report_id"] is None
    for identifier, value in before.items():
        assert publisher.pages[identifier] == value


@pytest.mark.parametrize("known", [False, True])
def test_unpublishing_a_foreign_or_unknown_id_only_clears_this_entry(publisher, known):
    original = publisher.page("original")
    publisher.send(original)
    copied = publisher.page("copy")
    entry = publisher.entry(original)
    if not known:
        entry["report_id"] = "unregistered-page"
        publisher.pages["unregistered-page"] = {"content": "legacy page"}
    publisher.plant(copied, entry)
    before = copy.deepcopy(publisher.pages)
    publisher.remove(copied, BOB)
    assert publisher.pages == before
    assert publisher.deletes == []
    assert copied.name not in json.loads((copied.parent / ".published.json").read_text())


def test_unknown_id_is_never_sent_to_the_publisher(publisher):
    page = publisher.page()
    publisher.pages["legacy-page"] = {"content": "legacy"}
    publisher.plant(page, {"report_id": "legacy-page", "published": True})
    result = publisher.send(page)
    assert publisher.uploads[-1]["report_id"] is None
    assert publisher.pages["legacy-page"] == {"content": "legacy"}
    assert "legacy-page" not in result["url"]


@pytest.mark.parametrize("scope", [None, LOCAL_SCOPE, replace(ALICE, user_id="invalid")])
def test_missing_tester_identity_fails_before_upload(publisher, scope):
    page = publisher.page()
    with pytest.raises(HTTPException) as caught:
        publisher.send(page, scope)
    assert caught.value.status_code == 401
    assert publisher.uploads == []


@pytest.mark.parametrize("personal_projects", [False, True])
def test_shared_and_personal_project_scopes_enforce_the_same_owner(publisher, personal_projects):
    from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES

    resources = ALL_PERSONAL_RESOURCES if personal_projects else frozenset()
    alice = replace(ALICE, personal_resources=resources)
    bob = replace(BOB, personal_resources=resources)
    page = publisher.page()
    first = publisher.send(page, alice)
    assert publisher.send(page, alice)["url"] == first["url"]
    with pytest.raises(publish.PublishOwnedByOtherUser):
        publisher.send(page, bob)


def test_simultaneous_first_publish_records_one_tester_before_the_other_uploads(publisher, monkeypatch):
    import anton.publisher

    page = publisher.page()
    entered, release = Event(), Event()
    upload = anton.publisher.publish

    def waiting_upload(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return upload(*args, **kwargs)

    monkeypatch.setattr(anton.publisher, "publish", waiting_upload)
    with ThreadPoolExecutor(max_workers=2) as pool:
        alice = pool.submit(publisher.send, page, ALICE)
        assert entered.wait(5)
        bob = pool.submit(publisher.send, page, BOB)
        release.set()
        alice.result(timeout=5)
        with pytest.raises(publish.PublishOwnedByOtherUser):
            bob.result(timeout=5)
    assert len(publisher.uploads) == 1


def test_update_keeps_another_pages_record_written_after_its_publish(publisher, monkeypatch):
    page = publisher.page()
    publisher.send(page)
    other = page.with_name("other.html")
    other.write_text("<html><body>another tester's page</body></html>")
    monkeypatch.setattr(publish, "desktop_artifact_and_base", lambda raw, *, scope: publish.DesktopArtifact(page, page.parent.parent))
    monkeypatch.setattr(publish, "desktop_publish_credential", lambda: ("test-key", publisher.endpoint))
    content_mtime = publish._content_mtime
    ready, writer_finished = Event(), Event()
    update_thread = None
    mtime_reads = 0

    def interleaved_mtime(path):
        nonlocal mtime_reads
        if get_ident() == update_thread:
            mtime_reads += 1
            if mtime_reads == 2:
                # The publish already saved its mtime under the folder lock.
                # A second refresh has read the map outside that lock: let a
                # different page publish before the stale map can be saved.
                ready.set()
                assert writer_finished.wait(5)
        return content_mtime(path)

    def update():
        nonlocal update_thread
        update_thread = get_ident()
        try:
            return publish.update_artifact(str(page), scope=ALICE)
        finally:
            ready.set()

    monkeypatch.setattr(publish, "_content_mtime", interleaved_mtime)
    with ThreadPoolExecutor(max_workers=1) as pool:
        updating = pool.submit(update)
        try:
            assert ready.wait(5)
            second = publisher.send(other, BOB)
        finally:
            writer_finished.set()
        updating.result(timeout=5)

    published = json.loads((page.parent / ".published.json").read_text())
    assert page.name in published
    assert published[other.name]["report_id"] == second["result"]["report_id"]


@pytest.mark.parametrize("operation", ["publish", "delete"])
def test_database_failure_prevents_any_upstream_mutation(publisher, monkeypatch, operation):
    page = publisher.page()
    publisher.send(page)
    before = copy.deepcopy(publisher.pages)
    before_uploads = len(publisher.uploads)

    def unavailable(*args, **kwargs):
        raise OperationalError("SELECT", {}, RuntimeError("unavailable"))

    monkeypatch.setattr(ownership, "get_engine", unavailable)
    with pytest.raises(HTTPException) as caught:
        publisher.send(page) if operation == "publish" else publisher.remove(page)
    assert caught.value.status_code == 503
    assert publisher.pages == before
    assert len(publisher.uploads) == before_uploads
    assert not publisher.deletes


def test_owner_record_survives_local_state_failure_and_reconnection(publisher, monkeypatch):
    page = publisher.page()

    def full_disk(*args):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(publish, "_write_published_map", full_disk)
        with pytest.raises(OSError, match="disk full"):
            publisher.send(page)
    identifier = next(iter(publisher.pages))
    get_engine(get_app_settings().database.uri).dispose()
    publisher.plant(page, {"report_id": identifier, "published": True})
    with pytest.raises(publish.PublishOwnedByOtherUser):
        publisher.send(page, BOB)
    assert publisher.send(page)["result"]["report_id"] == identifier


def test_failed_upload_creates_no_ownership_claim(publisher, monkeypatch):
    page = publisher.page()

    def unavailable(*args, **kwargs):
        raise RuntimeError("publisher unavailable")

    monkeypatch.setattr("anton.publisher.publish", unavailable)
    with pytest.raises(RuntimeError, match="publisher unavailable"):
        publisher.send(page)
    with Session(get_engine(get_app_settings().database.uri)) as session:
        assert session.exec(select(PublishedPageOwner).where(
            PublishedPageOwner.publisher_key == ownership.publisher_key(publisher.endpoint),
        )).all() == []


@pytest.mark.parametrize("url,mode,trusted", [
    ("https://view.mindshub.ai", "local", True),
    ("https://custom.example", "org", False),
    ("https://custom.example", "local", False),
])
def test_hosted_and_single_user_publish_do_not_open_the_local_registry(monkeypatch, tmp_path, url, mode, trusted):
    monkeypatch.setenv("COWORK_TENANCY_MODE", mode)
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", str(trusted).lower())
    get_app_settings.cache_clear()

    def unexpected(*args):
        pytest.fail("This publisher must keep its existing authorization")

    monkeypatch.setattr(ownership, "get_engine", unexpected)
    with ownership.publication_owner(tmp_path / "page.html", published_dir=tmp_path, publish_url=url, scope=None) as guard:
        assert guard is None
    get_app_settings.cache_clear()


@pytest.mark.asyncio
async def test_startup_runs_the_backfill_first_and_still_starts_when_it_cannot_finish(monkeypatch):
    from cowork import server
    from cowork.services import published_page_owner_backfill as backfill

    events, errors = [], []
    monkeypatch.setattr(server, "run_dev_setup", lambda: events.append("migrated"))
    monkeypatch.setattr(server, "consume_pending_credential_clear", lambda *args: events.append("cleared"))
    monkeypatch.setattr(server, "_warm_agent_runtime", lambda: events.append("warmed"))
    monkeypatch.setattr(server.logger, "error", lambda message, *args, **kwargs: errors.append(kwargs.get("extra")))

    def failed_backfill():
        events.append("backfill")
        raise backfill.PublishedPageBackfillUnavailable("Published page ownership backfill could not read publish history.")

    monkeypatch.setattr(backfill, "run_published_page_owner_backfill", failed_backfill)
    async with server.lifespan(server.create_app()):
        events.append("ready")
    assert events[:3] == ["migrated", "cleared", "backfill"]
    assert events.index("warmed") < events.index("ready")
    assert {"event": "published_page_backfill_unavailable",
            "reason": "Published page ownership backfill could not read publish history."} in errors


def _drop_backfill_sentinel():
    with Session(get_engine(get_app_settings().database.uri)) as session:
        for row in session.exec(select(Setting).where(Setting.key == ownership.BACKFILL_SENTINEL_KEY)).all():
            session.delete(row)
        session.commit()


@pytest.mark.parametrize("operation", ["publish", "unpublish", "slug", "path"])
def test_an_unfinished_backfill_refuses_protected_publishing_until_it_completes(publisher, monkeypatch, operation):
    from cowork.services import published_page_owner_backfill as backfill

    page = publisher.page()
    publisher.send(page)
    before = copy.deepcopy(publisher.pages)
    before_uploads = len(publisher.uploads)
    _drop_backfill_sentinel()
    attempts = []

    def unavailable():
        attempts.append("failed")
        raise backfill.PublishedPageBackfillUnavailable("Published page ownership backfill could not verify the publisher's page list.")

    def finished():
        attempts.append("finished")
        with Session(get_engine(get_app_settings().database.uri)) as session:
            session.add(Setting(key=ownership.BACKFILL_SENTINEL_KEY, value="1"))
            session.commit()

    def run():
        if operation == "publish":
            return publisher.send(page)
        if operation == "unpublish":
            return publisher.remove(page)
        return _delete(publisher, page, ALICE, operation)

    try:
        monkeypatch.setattr(backfill, "run_published_page_owner_backfill", unavailable)
        with pytest.raises(HTTPException) as caught:
            run()
        assert caught.value.status_code == 503
        assert publisher.pages == before and page.is_file()
        assert len(publisher.uploads) == before_uploads and publisher.deletes == []

        monkeypatch.setattr(backfill, "run_published_page_owner_backfill", finished)
        run()
        # Once the pass has finished, publishing no longer retries it.
        publisher.send(publisher.page("after-backfill"))
        assert attempts == ["failed", "finished"]
    finally:
        _drop_backfill_sentinel()


@pytest.mark.parametrize("route", ["slug", "path"])
def test_another_tester_cannot_delete_the_original_artifact_or_its_page(publisher, route):
    page = publisher.page()
    publisher.send(page, access={"mode": "password", "password": "owner-password"})
    before = copy.deepcopy(publisher.pages)
    with pytest.raises(publish.PublishOwnedByOtherUser):
        _delete(publisher, page, BOB, route)
    assert publisher.pages == before
    assert publisher.deletes == []
    assert page.is_file()


@pytest.mark.parametrize("route", ["slug", "path"])
@pytest.mark.parametrize("known", [False, True])
def test_deleting_a_folder_that_names_another_page_deletes_only_that_folder(publisher, route, known):
    original = publisher.page("original")
    publisher.send(original)
    planted = publisher.page("planted")
    entry = publisher.entry(original)
    if not known:
        entry["report_id"] = "unregistered-page"
        publisher.pages["unregistered-page"] = {"content": "legacy page"}
    publisher.plant(planted, entry)
    before = copy.deepcopy(publisher.pages)
    _delete(publisher, planted, BOB, route)
    assert publisher.pages == before
    assert publisher.deletes == []
    assert not planted.parent.exists() and original.is_file()


@pytest.mark.parametrize("route", ["slug", "path"])
def test_the_owner_deletes_the_page_with_the_artifact(publisher, route):
    page = publisher.page()
    identifier = publisher.send(page)["result"]["report_id"]
    _delete(publisher, page, ALICE, route)
    assert publisher.deletes == [identifier]
    assert publisher.pages == {} and not page.parent.exists()


def test_the_path_delete_route_answers_the_owner_refusal(publisher, monkeypatch):
    from cowork.api.v1.endpoints import artifacts as artifact_routes

    page = publisher.page()
    publisher.send(page)
    monkeypatch.setattr(publish, "desktop_artifact_and_base", lambda raw, *, scope: publish.DesktopArtifact(page.parent, publisher.root))
    monkeypatch.setattr(publish, "desktop_publish_credential", lambda: ("test-key", publisher.endpoint))
    with pytest.raises(HTTPException) as caught:
        artifact_routes.delete_artifact_endpoint(scope=BOB, path=str(page.parent))
    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == publish.OWNED_BY_OTHER_USER_CODE
    assert page.is_file() and publisher.deletes == []


@pytest.mark.parametrize("path", ["agent", "update", "editor"])
def test_every_publish_path_refuses_another_testers_page(publisher, monkeypatch, path):
    from types import SimpleNamespace

    from cowork.api.v1.endpoints.artifact_workspace import _sync_live_artifact
    from cowork.common.settings.user_settings import use_settings_scope
    from cowork.harnesses.anton_harness import tools

    page = publisher.page()
    publisher.send(page, access={"mode": "password", "password": "owner-password"})
    before = copy.deepcopy(publisher.pages)
    before_uploads = len(publisher.uploads)
    monkeypatch.setattr(publish, "desktop_publish_context", lambda raw, *, scope: publish.PublishContext(page, publisher.root, "test-key", publisher.endpoint))
    monkeypatch.setattr(publish, "desktop_artifact_and_base", lambda raw, *, scope: publish.DesktopArtifact(page, publisher.root))
    monkeypatch.setattr(publish, "desktop_publish_credential", lambda: ("test-key", publisher.endpoint))
    if path == "agent":
        with use_settings_scope(BOB), pytest.raises(publish.PublishOwnedByOtherUser):
            tools._publish_artifact(str(page))
    elif path == "update":
        with pytest.raises(publish.PublishOwnedByOtherUser):
            publish.update_artifact(str(page), scope=BOB)
    else:
        assert asyncio.run(_sync_live_artifact(SimpleNamespace(scope=BOB), page.parent)) is False
    assert publisher.pages == before
    assert len(publisher.uploads) == before_uploads


def test_a_moved_folder_keeps_its_owner_and_link(publisher):
    original = publisher.page("before-move")
    first = publisher.send(original, access={"mode": "password", "password": "owner-password"})
    identifier = first["result"]["report_id"]
    moved_folder = publisher.root / "after-move"
    original.parent.rename(moved_folder)
    ownership.rekey_published_page_paths(original.parent, moved_folder)
    moved = moved_folder / original.name
    with pytest.raises(publish.PublishOwnedByOtherUser):
        publisher.send(moved, BOB)
    again = publisher.send(moved, access={"mode": "password", "password": "owner-password"})
    assert again["url"] == first["url"]
    assert publisher.uploads[-1]["previous_access"]["password_hash"] == "owner-password-hash"
    publisher.remove(moved)
    assert publisher.deletes == [identifier]


@pytest.mark.usefixtures("cleanup_tmp_projects")
def test_moving_a_task_to_another_project_keeps_its_published_link(publisher):
    from cowork.db.scoped import ScopedSession
    from cowork.models.conversation import Conversation
    from cowork.models.project import Project
    from cowork.services.task_objects import TaskObjectService
    from test_artifact_ownership import make_conversation, make_project, project_root, write_artifact

    source, dest = make_project(publisher.root, ALICE.org_id), make_project(publisher.root, ALICE.org_id)
    conversation_id = make_conversation(source, ALICE.user_id)
    folder = write_artifact(project_root(source).base, "report", conversation_id)
    first = publish.publish_artifact(
        folder / "index.html", artifacts_base=project_root(source).base, api_key="test-key",
        publish_url=publisher.endpoint, scope=ALICE,
    )
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        session = ScopedSession(raw, ALICE)
        assert TaskObjectService(session).relocate_to_project(
            session.get(Conversation, conversation_id), session.get(Project, source.id), session.get(Project, dest.id),
        ) == 1
    moved = project_root(dest).base / "report" / "index.html"
    again = publish.publish_artifact(
        moved, artifacts_base=project_root(dest).base, api_key="test-key",
        publish_url=publisher.endpoint, scope=ALICE,
    )
    assert again["url"] == first["url"]
    publish.unpublish_artifact(
        moved, artifacts_base=project_root(dest).base, api_key="test-key",
        publish_url=publisher.endpoint, scope=ALICE,
    )
    assert publisher.deletes == [first["result"]["report_id"]]


@pytest.mark.usefixtures("cleanup_tmp_projects")
def test_renaming_a_project_keeps_its_published_link(publisher, monkeypatch):
    from cowork.db.scoped import ScopedSession
    from cowork.services.projects import ProjectService

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(publisher.root / "projects"))
    get_app_settings.cache_clear()
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        projects = ProjectService(ScopedSession(raw, ALICE))
        project = projects.create_project("reports")
        base = Path(project.path) / ".anton" / "artifacts"
        page = base / "report" / "index.html"
        page.parent.mkdir(parents=True)
        (page.parent / "metadata.json").write_text(json.dumps({"id": str(uuid4()), "type": "html", "primary": "index.html"}))
        page.write_text("<html><body>before the rename</body></html>")
        first = publish.publish_artifact(page, artifacts_base=base, api_key="test-key", publish_url=publisher.endpoint, scope=ALICE)
        moved_base = Path(projects.update_project(project.id, name="reports-q4").path) / ".anton" / "artifacts"
    moved = moved_base / "report" / "index.html"
    assert moved.is_file() and not page.exists()
    again = publish.publish_artifact(moved, artifacts_base=moved_base, api_key="test-key", publish_url=publisher.endpoint, scope=ALICE)
    assert again["url"] == first["url"]
    publish.unpublish_artifact(moved, artifacts_base=moved_base, api_key="test-key", publish_url=publisher.endpoint, scope=ALICE)
    assert publisher.deletes == [first["result"]["report_id"]]


@pytest.mark.usefixtures("cleanup_tmp_projects")
def test_the_slug_delete_route_checks_the_requesting_tester(publisher, monkeypatch):
    from cowork.api.v1.endpoints.artifacts import delete_artifact_for_request
    from cowork.db.scoped import ScopedSession
    from cowork.services.projects import ProjectService

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(publisher.root / "projects"))
    get_app_settings.cache_clear()
    monkeypatch.setattr(publish, "_resolve_publish_endpoint", lambda settings: (publisher.endpoint, "test-key"))
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as raw:
        project = ProjectService(ScopedSession(raw, ALICE)).create_project("shared")
        project_id, base = project.id, Path(project.path) / ".anton" / "artifacts"
    folder = base / "report"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"id": str(uuid4()), "type": "html", "primary": "index.html"}))
    (folder / "index.html").write_text("<html><body>shared</body></html>")
    published = publish.publish_artifact(
        folder / "index.html", artifacts_base=base, api_key="test-key", publish_url=publisher.endpoint, scope=ALICE,
    )
    artifact_id = json.loads((folder / "metadata.json").read_text())["id"]
    before = copy.deepcopy(publisher.pages)

    with Session(engine) as raw, pytest.raises(HTTPException) as caught:
        asyncio.run(delete_artifact_for_request(ScopedSession(raw, BOB), artifact_id, project_id=project_id))
    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == publish.OWNED_BY_OTHER_USER_CODE
    assert folder.is_dir() and publisher.pages == before and publisher.deletes == []

    with Session(engine) as raw:
        asyncio.run(delete_artifact_for_request(ScopedSession(raw, ALICE), artifact_id, project_id=project_id))
    assert publisher.deletes == [published["result"]["report_id"]]
    assert not folder.exists()


@pytest.mark.parametrize("route", ["slug", "path"])
@pytest.mark.parametrize("deleter", ["owner", "another tester"])
def test_a_delete_never_follows_a_record_key_out_of_its_folder(publisher, route, deleter):
    original = publisher.page("original")
    publisher.send(original)
    record = (original.parent / ".published.json").read_bytes()
    before = copy.deepcopy(publisher.pages)
    draft = publisher.page("draft")
    entry = publisher.entry(original)
    (draft.parent / ".published.json").write_text(json.dumps({
        "../original/index.html": entry, str(original): entry,
    }))
    _delete(publisher, draft, ALICE if deleter == "owner" else BOB, route)
    assert not draft.parent.exists()
    assert (original.parent / ".published.json").read_bytes() == record
    assert publisher.pages == before and publisher.deletes == []


def test_an_unrecorded_page_is_logged_and_the_upload_runs_outside_a_transaction(publisher, monkeypatch, caplog):
    import anton.publisher

    page = publisher.page()
    publisher.pages["legacy-page"] = {"content": "legacy"}
    old_url = "https://old-viewer.example/view/legacy-page/"
    publisher.plant(page, {"report_id": "legacy-page", "url": old_url, "published": True})
    sessions, in_transaction = [], []
    checked = ownership.PublicationOwner.checked_report_id

    def capture(self, *args, **kwargs):
        sessions.append(self.session)
        return checked(self, *args, **kwargs)

    upload = anton.publisher.publish

    def watched_upload(*args, **kwargs):
        in_transaction.append(sessions[-1].in_transaction())
        return upload(*args, **kwargs)

    monkeypatch.setattr(ownership.PublicationOwner, "checked_report_id", capture)
    monkeypatch.setattr(anton.publisher, "publish", watched_upload)
    with caplog.at_level(logging.WARNING, logger=ownership.logger.name):
        publisher.send(page)
    assert in_transaction == [False]
    assert [record.args for record in caplog.records
            if record.name == ownership.logger.name and record.levelno == logging.WARNING
            and record.msg.startswith("orphaned_publish") and "unrecorded_owner" in record.msg] == [("legacy-page", old_url)]


def test_a_failed_owner_record_after_a_new_upload_is_logged(publisher, monkeypatch, caplog):
    page = publisher.page()

    def unavailable(self, report_id):
        raise OperationalError("INSERT", {}, RuntimeError("unavailable"))

    monkeypatch.setattr(ownership.PublicationOwner, "record", unavailable)
    with caplog.at_level(logging.WARNING, logger=publish.logger.name), pytest.raises(HTTPException) as caught:
        publisher.send(page)
    assert caught.value.status_code == 503
    [identifier] = publisher.pages
    assert [record.args for record in caplog.records
            if record.name == publish.logger.name and "owner_record_failed" in str(record.msg)] == [
        (identifier, f"{publisher.endpoint}/view/{identifier}/"),
    ]


def test_rekey_moves_only_the_moved_folders_records(publisher):
    root = publisher.root.resolve()
    key = ownership.publisher_key(publisher.endpoint)
    paths = {name: str(root / name / "index.html") for name in ("report", "REPORT", "report-2")}
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        for name, path in paths.items():
            session.add(PublishedPageOwner(publisher_key=key, report_id=f"{name}-{uuid4().hex}", artifact_path=path,
                                           owner_user_id=ALICE.user_id))
        session.commit()
    ownership.rekey_published_page_paths(root / "report", root / "moved")
    with Session(engine) as session:
        stored = {row.artifact_path for row in session.exec(select(PublishedPageOwner).where(PublishedPageOwner.publisher_key == key))}
    assert stored == {str(root / "moved" / "index.html"), paths["REPORT"], paths["report-2"]}


def test_a_failed_rekey_is_logged_and_never_raises(publisher, monkeypatch, caplog):
    page = publisher.page("before-move")
    publisher.send(page)
    moved = publisher.root / "after-move"
    page.parent.rename(moved)

    def unavailable(*args, **kwargs):
        raise OperationalError("SELECT", {}, RuntimeError("unavailable"))

    monkeypatch.setattr(ownership, "get_engine", unavailable)
    with caplog.at_level(logging.WARNING, logger=ownership.logger.name):
        ownership.rekey_published_page_paths(page.parent, moved)
    # The database-error filter names the call site in place of the message.
    [record] = [record for record in caplog.records if record.name == ownership.logger.name]
    assert record.levelno == logging.WARNING and record.error_type == "OperationalError"
    assert "rekey_published_page_paths" in record.getMessage()
