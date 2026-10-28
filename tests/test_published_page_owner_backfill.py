"""Conservative startup ownership imports from publish_artifact's real history output."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import TenantScope
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.published_page import PublishedPageOwner
from cowork.models.setting import Setting
from cowork.services import publish, published_page_ownership
from cowork.services import published_page_owner_backfill as backfill

ENDPOINT = "https://publisher.example"


@dataclass(frozen=True)
class Page:
    partition: str
    user: str
    target: Path
    history: Path
    report_id: str
    url: str


@pytest.fixture
def store(personal_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.delenv("ANTON_COWORK_STATE_DIR", raising=False)
    get_app_settings.cache_clear()
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    inventory: dict[str, str] = {}
    calls: list[dict] = []
    monkeypatch.setattr(backfill, "get_user_settings", lambda: object())
    monkeypatch.setattr(backfill, "_resolve_publish_endpoint", lambda settings: (ENDPOINT, "test-publish-key"))

    def listing(**kwargs):
        calls.append(kwargs)
        return [{"report_id": report, "view_url": url} for report, url in inventory.items()]

    monkeypatch.setattr(backfill, "list_published", listing)

    class Store:
        def page(self, *, trust="project", report_id=None, partition=None, user=None, nested=False):
            partition, user = partition or str(uuid4()), user or str(uuid4())
            report_id = report_id or uuid4().hex
            url = f"https://viewer.example/view/{report_id}/"
            project_path = tmp_path / "shared" / partition / "projects" / uuid4().hex
            target = project_path / ".anton" / "artifacts" / "page" / ("site" if nested else "") / "index.html"
            target.parent.mkdir(parents=True)
            target.write_text("<html>Before the ownership upgrade</html>")
            with Session(engine) as session:
                project = Project(name=uuid4().hex, path=str(project_path), org_id=partition,
                                  created_by=user if trust == "project" else None)
                session.add(project)
                if trust == "setting":
                    session.add(Setting(key="theme", value="dark", scope="user", org_id=partition, user_id=user))
                if trust == "conversation":
                    session.add(Conversation(topic="history owner", project_id=project.id, org_id=partition, created_by=user))
                session.commit()
            scope = TenantScope(personal=True, org_id=partition, user_id=user,
                                personal_resources=frozenset({PersonalResource.PROJECTS}))
            # The producer writes the fixture, with only the new guard disabled
            # to represent a page published before this upgrade.
            with monkeypatch.context() as legacy:
                legacy.setattr(published_page_ownership, "protects_publisher", lambda _: False)
                legacy.setattr(publish, "vault_for_scope", lambda _: object())
                legacy.setattr("anton.publisher.publish", lambda *args, **kwargs: {
                    "report_id": report_id, "view_url": url, "md5": "fixture-content",
                })
                publish.publish_artifact(target, artifacts_base=project_path / ".anton" / "artifacts",
                                         api_key="test-publish-key", publish_url=ENDPOINT, scope=scope)
            inventory[report_id] = url
            return Page(partition, user, target, tmp_path / "shared" / partition / "publish" / "state.json", report_id, url)

        def run(self):
            return backfill.run_published_page_owner_backfill(engine)

        def rows(self):
            with Session(engine) as session:
                return list(session.exec(select(PublishedPageOwner)).all())

        def sentinel(self):
            with Session(engine) as session:
                return session.exec(select(Setting).where(Setting.key == backfill.SENTINEL_KEY)).first()

        def history_row(self, page):
            return next(row for row in json.loads(page.history.read_text())["publish_history"] if row["reportId"] == page.report_id)

        def replace_history(self, page, **updates):
            row = self.history_row(page) | updates
            page.history.write_text(json.dumps({"publish_history": [row]}))

    fixture = Store()
    fixture.engine, fixture.inventory, fixture.calls, fixture.root = engine, inventory, calls, tmp_path
    yield fixture
    engine.dispose()
    get_app_settings.cache_clear()


@pytest.mark.parametrize("trust", ["project", "setting", "conversation"])
def test_real_publish_history_imports_the_trusted_tester_and_resolved_source(store, trust):
    page = store.page(trust=trust)
    assert store.run() == backfill.BackfillSummary(recorded=1, unknown=0)
    [row] = store.rows()
    assert (row.publisher_key, row.report_id, row.artifact_path, row.owner_user_id) == (
        published_page_ownership.publisher_key(ENDPOINT), page.report_id, str(page.target.resolve()), page.user,
    )
    assert store.calls == [{"api_key": "test-publish-key", "publish_url": ENDPOINT, "ssl_verify": True}]
    assert store.sentinel() is not None


def test_restart_is_a_noop_and_never_calls_the_publisher_again(store):
    store.page()
    assert store.run().recorded == 1
    assert store.run() is None
    assert len(store.rows()) == len(store.calls) == 1


def test_an_existing_record_is_never_changed(store):
    page = store.page()
    known_user = str(uuid4())
    with Session(store.engine) as session:
        session.add(PublishedPageOwner(publisher_key=published_page_ownership.publisher_key(ENDPOINT),
                                      report_id=page.report_id, artifact_path="/previous/source.html", owner_user_id=known_user))
        session.commit()
    assert store.run().recorded == 0
    [row] = store.rows()
    assert (row.artifact_path, row.owner_user_id) == ("/previous/source.html", known_user)
    assert store.calls == []


@pytest.mark.parametrize("problem", ["unmapped", "ambiguous", "missing", "outside", "wrong_partition", "invalid_row"])
def test_unplaceable_history_stays_unknown_without_a_publisher_call(store, problem):
    page = store.page(trust=None if problem == "unmapped" else "project")
    if problem == "ambiguous":
        with Session(store.engine) as session:
            session.add(Setting(key="theme", value="dark", scope="user", org_id=page.partition, user_id=str(uuid4())))
            session.commit()
    elif problem == "missing":
        page.target.unlink()
    elif problem == "outside":
        outside = store.root / "index.html"
        outside.write_text("outside every trusted project")
        store.replace_history(page, artifact=str(outside))
    elif problem == "wrong_partition":
        other = store.page()
        store.replace_history(page, artifact=str(other.target))
        other.history.unlink()
    elif problem == "invalid_row":
        store.replace_history(page, publishedAt="not a date")
    assert store.run().recorded == 0
    assert store.rows() == [] and store.calls == []
    assert store.sentinel() is not None


@pytest.mark.parametrize("conflict", ["other_tester", "other_artifact", "unmapped_tester", "global"])
def test_conflicting_histories_never_select_the_latest_claimant(store, conflict):
    page = store.page()
    if conflict == "global":
        home = store.root / "home"
        home.mkdir(exist_ok=True)
        (home / "state.json").write_text(page.history.read_text())
    else:
        store.page(report_id=page.report_id,
                   partition=page.partition if conflict == "other_artifact" else None,
                   user=page.user if conflict == "other_artifact" else None,
                   trust=None if conflict == "unmapped_tester" else "project")
    assert store.run().recorded == 0
    assert store.rows() == [] and store.calls == []


@pytest.mark.parametrize("inventory", ["missing_id", "different_url", "duplicate_id"])
def test_inventory_must_confirm_both_the_exact_id_and_exact_view_url(store, monkeypatch, inventory):
    page = store.page()
    if inventory == "missing_id":
        store.inventory.clear()
    elif inventory == "different_url":
        store.inventory[page.report_id] = page.url.replace("viewer.example", "old-viewer.example")
    else:
        monkeypatch.setattr(backfill, "list_published", lambda **kwargs: [
            {"report_id": page.report_id, "view_url": page.url},
            {"report_id": page.report_id, "view_url": page.url + "other"},
        ])
    assert store.run().recorded == 0
    assert store.rows() == [] and store.sentinel() is not None


def test_publisher_failure_writes_nothing_and_restart_retries_without_exposing_its_body(store, monkeypatch):
    store.page()
    with monkeypatch.context() as failure:
        def unavailable(**kwargs):
            raise RuntimeError("private-key-and-publisher-body")
        failure.setattr(backfill, "list_published", unavailable)
        with pytest.raises(backfill.PublishedPageBackfillUnavailable) as caught:
            store.run()
        assert "private-key-and-publisher-body" not in str(caught.value)
        assert caught.value.__suppress_context__
    assert store.rows() == [] and store.sentinel() is None
    assert store.run().recorded == 1


@pytest.mark.parametrize("bad_history", ["broken_json", "wrong_shape", "symlink"])
def test_unreadable_history_aborts_the_whole_pass_without_a_sentinel(store, bad_history):
    good = store.page()
    bad = store.page()
    if bad_history == "broken_json":
        bad.history.write_text('{"private-history-body')
    elif bad_history == "wrong_shape":
        bad.history.write_text('{"publish_history": "private-history-body"}')
    else:
        bad.history.unlink()
        bad.history.symlink_to(good.history)
    with pytest.raises(backfill.PublishedPageBackfillUnavailable) as caught:
        store.run()
    assert "private-history-body" not in str(caught.value)
    assert store.rows() == [] and store.sentinel() is None and store.calls == []


def test_failed_flush_rolls_back_imports_and_sentinel_then_restart_retries(store):
    store.page()
    store.page()
    def fail_after_inserts(session, context):
        if session.get_bind() is store.engine and any(isinstance(row, PublishedPageOwner) for row in session.new):
            raise RuntimeError("injected after ownership inserts")
    event.listen(Session, "after_flush", fail_after_inserts)
    try:
        with pytest.raises(RuntimeError, match="injected after ownership inserts"):
            store.run()
    finally:
        event.remove(Session, "after_flush", fail_after_inserts)
    assert store.rows() == [] and store.sentinel() is None
    assert store.run().recorded == 2
    assert len(store.rows()) == 2 and store.sentinel() is not None


def test_another_folder_naming_the_page_blocks_the_import(store):
    page = store.page()
    # The first publisher's folder, whose history row rotated out of the 100 kept.
    first = store.root / "shared" / str(uuid4()) / "projects" / uuid4().hex
    folder = first / ".anton" / "artifacts" / "page"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html>The first publisher's page</html>")
    (folder / ".published.json").write_text(json.dumps({"index.html": {"report_id": page.report_id, "published": True}}))
    with Session(store.engine) as session:
        session.add(Project(name=uuid4().hex, path=str(first), org_id=str(uuid4()), created_by=str(uuid4())))
        session.commit()
    assert store.run() == backfill.BackfillSummary(recorded=0, unknown=1)
    assert store.rows() == [] and store.calls == []
    assert store.sentinel() is not None


def _other_folder(store):
    """An artifact folder in another tester's project, registered like any other."""
    project = store.root / "shared" / str(uuid4()) / "projects" / uuid4().hex
    folder = project / ".anton" / "artifacts" / "page"
    folder.mkdir(parents=True)
    with Session(store.engine) as session:
        session.add(Project(name=uuid4().hex, path=str(project), org_id=str(uuid4()), created_by=str(uuid4())))
        session.commit()
    return folder


@pytest.mark.parametrize("record", ["nul_key", "self_loop", "deep", "oversized", "not_json"])
def test_an_unreadable_record_elsewhere_is_skipped_without_breaking_the_pass(store, record):
    page = store.page()
    folder = _other_folder(store)
    path = folder / ".published.json"
    if record == "nul_key":
        path.write_text(json.dumps({"a\u0000b": {"report_id": page.report_id}}))
    elif record == "self_loop":
        (folder / "loop").symlink_to(folder / "loop")
        path.write_text(json.dumps({"loop": {"report_id": page.report_id}}))
    elif record == "deep":
        path.write_text("[" * 200_000)
    elif record == "oversized":
        path.write_text(json.dumps({"index.html": {"report_id": page.report_id, "pad": "x" * backfill._MAX_RECORD_BYTES}}))
    else:
        path.write_text("{not json")
    assert store.run().recorded == 1
    assert store.sentinel() is not None


def test_a_page_published_from_a_subfolder_stays_unclaimed(store):
    store.page(nested=True)
    assert store.run() == backfill.BackfillSummary(recorded=0, unknown=1)
    assert store.rows() == [] and store.calls == []


def test_a_linked_publish_record_elsewhere_is_not_read(store):
    page = store.page()
    first = store.root / "shared" / str(uuid4()) / "projects" / uuid4().hex
    folder = first / ".anton" / "artifacts" / "page"
    folder.mkdir(parents=True)
    (folder / ".published.json").symlink_to(page.target.parent / ".published.json")
    with Session(store.engine) as session:
        session.add(Project(name=uuid4().hex, path=str(first), org_id=str(uuid4()), created_by=str(uuid4())))
        session.commit()
    assert store.run().recorded == 1


def test_a_backfilled_owner_keeps_the_link_and_another_tester_is_refused(store, monkeypatch):
    page = store.page()
    assert store.run().recorded == 1
    monkeypatch.setattr(published_page_ownership, "get_engine", lambda uri: store.engine)
    monkeypatch.setattr(publish, "vault_for_scope", lambda scope: object())
    uploads = []

    def upload(path, **kwargs):
        uploads.append(kwargs)
        return {"report_id": kwargs["report_id"] or uuid4().hex, "view_url": page.url, "md5": "after-upgrade"}

    monkeypatch.setattr("anton.publisher.publish", upload)
    owner = TenantScope(personal=True, org_id=page.partition, user_id=page.user,
                        personal_resources=frozenset({PersonalResource.PROJECTS}))
    other = TenantScope(personal=True, org_id=str(uuid4()), user_id=str(uuid4()),
                        personal_resources=frozenset({PersonalResource.PROJECTS}))
    base = page.target.parent.parent
    with pytest.raises(publish.PublishOwnedByOtherUser):
        publish.publish_artifact(page.target, artifacts_base=base, api_key="test-publish-key", publish_url=ENDPOINT, scope=other)
    assert uploads == []
    result = publish.publish_artifact(page.target, artifacts_base=base, api_key="test-publish-key", publish_url=ENDPOINT, scope=owner)
    assert uploads[-1]["report_id"] == page.report_id
    assert result["url"] == page.url


def test_no_history_finishes_without_contacting_the_publisher(store):
    assert store.run() == backfill.BackfillSummary(recorded=0, unknown=0)
    assert store.calls == [] and store.sentinel() is not None


@pytest.mark.parametrize("setting,value", [("COWORK_TRUST_IDENTITY_HEADERS", "false"), ("COWORK_TENANCY_MODE", "org")])
def test_other_deployments_skip_before_reading_credentials(store, monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    get_app_settings.cache_clear()
    monkeypatch.setattr(backfill, "_resolve_publish_endpoint", lambda settings: pytest.fail("credential read outside trusted local mode"))
    assert store.run() is None
    assert store.sentinel() is None


def test_mindshub_publisher_skips_the_import(store, monkeypatch):
    monkeypatch.setattr(backfill, "_resolve_publish_endpoint", lambda settings: ("https://publish.mindshub.ai", "test-key"))
    assert store.run() is None
    assert store.sentinel() is None and store.calls == []


def test_hosted_publisher_without_a_key_skips_the_pass(store, monkeypatch):
    store.page()
    monkeypatch.setattr(backfill, "_resolve_publish_endpoint", lambda settings: ("https://publish.mindshub.ai", ""))
    assert store.run() is None
    assert store.sentinel() is None and store.calls == [] and store.rows() == []


def test_custom_publisher_without_a_key_and_without_candidates_finishes(store, monkeypatch):
    monkeypatch.setattr(backfill, "_resolve_publish_endpoint", lambda settings: (ENDPOINT, ""))
    assert store.run() == backfill.BackfillSummary(recorded=0, unknown=0)
    assert store.sentinel() is not None and store.calls == []


def test_custom_publisher_without_a_key_and_with_candidates_aborts_without_a_sentinel(store, monkeypatch):
    store.page()
    monkeypatch.setattr(backfill, "_resolve_publish_endpoint", lambda settings: (ENDPOINT, ""))
    with pytest.raises(backfill.PublishedPageBackfillUnavailable, match="requires a publisher credential"):
        store.run()
    assert store.sentinel() is None and store.calls == [] and store.rows() == []
