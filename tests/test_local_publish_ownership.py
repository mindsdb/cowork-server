"""Trusted publisher ownership on deployments with authenticated local testers."""
from __future__ import annotations

import asyncio
import copy
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
from cowork.services import published_page_ownership as ownership
from cowork.services import publish

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
    result.endpoint = endpoint
    return result


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
async def test_startup_waits_for_backfill_before_serving_or_warming(monkeypatch):
    from cowork import server
    from cowork.services import published_page_owner_backfill as backfill

    events = []
    monkeypatch.setattr(server, "run_dev_setup", lambda: events.append("migrated"))
    monkeypatch.setattr(server, "consume_pending_credential_clear", lambda *args: events.append("cleared"))
    monkeypatch.setattr(server, "_warm_agent_runtime", lambda: events.append("warmed"))

    def failed_backfill():
        events.append("backfill")
        raise backfill.PublishedPageBackfillUnavailable("retry startup")

    monkeypatch.setattr(backfill, "run_published_page_owner_backfill", failed_backfill)
    with pytest.raises(backfill.PublishedPageBackfillUnavailable, match="retry startup"):
        async with server.lifespan(server.create_app()):
            events.append("ready")
    assert events == ["migrated", "cleared", "backfill"]
