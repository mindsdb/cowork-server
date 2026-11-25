"""Which docs are shared, catching them up at startup, and the debouncer their follow-ups run on."""
from __future__ import annotations

import contextvars
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.docs import republish, sharing, sync
from cowork.docs.debounce import Debouncer
from cowork.docs.service import create_doc
from cowork.docs.store import DOC_HTML

PAGE_URL = "https://view.staging.mindshub.ai/view/215f76ea7/4bbd890f"
KEY = "artifact/814c74db-0218-48d5-a29d-50be7bbfa243"


def _publish_record(folder: Path, **entry) -> None:
    (folder / ".published.json").write_text(json.dumps({DOC_HTML: {
        "url": PAGE_URL, "published": True, "artifact_key": KEY, "report_id": "4bbd890f", **entry,
    }}))


def _settle(pending, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while pending() and time.monotonic() < end:
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _no_republish_left_waiting():
    """The republish debouncer is one for the process. A republish an earlier
    test left waiting would fire during a later one, against that test's
    stand-ins, so none crosses a test's edge."""
    republish._republishes.cancel_all()
    yield
    republish._republishes.cancel_all()


@pytest.fixture
def folder(tmp_path):
    """A doc, shared at PAGE_URL."""
    made = create_doc(tmp_path / ".anton" / "artifacts", title="Launch plan", markdown="Launch is Nov 18.").folder
    _publish_record(made)
    return made


# --- which docs are shared ---


def test_the_shared_docs_are_the_doc_folders_with_a_live_page(folder, tmp_path, monkeypatch):
    root = folder.parent
    unshared = create_doc(root, title="Not shared", markdown="x").folder
    (root / ".hidden").mkdir()
    (root / "notes").mkdir()
    monkeypatch.setattr(sharing, "artifacts_sources_for_scan",
                        lambda scope: [SimpleNamespace(base=str(root)), SimpleNamespace(base=str(tmp_path / "gone"))])
    assert sharing.shared_doc_folders() == [folder]
    _publish_record(unshared)
    assert sorted(sharing.shared_doc_folders()) == sorted([folder, unshared])


def test_the_scan_reads_no_metadata_of_a_folder_without_a_doc_page(folder, monkeypatch):
    root = folder.parent
    other = root / "chart-1234abcd"
    other.mkdir()
    (other / "index.html").write_text("<p>chart</p>")
    read = []
    real = sharing.is_doc_folder
    monkeypatch.setattr(sharing, "is_doc_folder", lambda f: read.append(f.name) or real(f))
    monkeypatch.setattr(sharing, "artifacts_sources_for_scan", lambda scope: [SimpleNamespace(base=str(root))])
    assert sharing.shared_doc_folders() == [folder]
    assert read == [folder.name]


# --- catching up at startup ---


def test_startup_republishes_a_page_that_changed_since_it_went_out(folder, monkeypatch):
    scheduled = []
    monkeypatch.setattr(republish, "schedule", lambda f, **kw: scheduled.append(f) or True)
    monkeypatch.setattr(republish, "compute_publish_md5", lambda page, **kw: "new")
    _publish_record(folder, last_md5="new")
    republish.catch_up(folder)
    assert scheduled == []  # what went out is what the page is
    _publish_record(folder, last_md5="old")  # e.g. an edit just before the app quit
    republish.catch_up(folder)
    assert scheduled == [folder]


def test_startup_catches_each_shared_doc_up_and_one_failing_holds_none_back(folder, tmp_path, monkeypatch):
    other = create_doc(tmp_path / "b" / ".anton" / "artifacts", title="Other", markdown="x").folder
    caught = []

    def catch_up(f, **kw):
        if f == folder:
            raise OSError("disk full")
        caught.append(f)

    monkeypatch.setattr(sync, "shared_doc_folders", lambda: [folder, other])
    monkeypatch.setattr(sync.republish, "catch_up", catch_up)
    sync.catch_up_shared_docs()
    assert caught == [other]


def test_a_server_several_people_share_catches_nothing_up_at_startup(monkeypatch):
    monkeypatch.setattr(sync, "tenant_scope_required", lambda: True)
    monkeypatch.setattr(sync.threading, "Thread", lambda **kw: pytest.fail("walked the projects"))
    sync.start()


# --- the debouncer ---


def test_only_the_newest_of_a_burst_runs_in_the_context_it_was_asked_in():
    runs = []
    who = contextvars.ContextVar("who", default="nobody")
    debouncer = Debouncer(0.02, what="test")
    for n in range(3):
        who.set(f"call {n}")
        debouncer.schedule("k", lambda: runs.append(who.get()))
    _settle(debouncer.pending)
    time.sleep(0.05)
    assert runs == ["call 2"]


def test_a_replaced_run_that_already_fired_stands_down():
    runs = []
    debouncer = Debouncer(0.0, what="test")
    debouncer._waiting["k"] = threading.Timer(1, lambda: None)  # a newer schedule took its place
    debouncer._fire("k", contextvars.copy_context(), lambda: runs.append("old"))
    assert runs == [] and debouncer.pending() == ["k"]
    debouncer.cancel_all()
    assert debouncer.pending() == []


def test_a_republish_asks_whether_the_doc_is_still_shared_under_the_pages_publish_lock(tmp_path, monkeypatch):
    """Stop sharing takes the page's publish lock while it takes the page down, so
    the check and the upload under it can't have an unshare land between them."""
    import contextlib

    folder = create_doc(tmp_path, title="Plan", markdown="Hi.", by="user").folder
    events = []

    @contextlib.contextmanager
    def publish_lock(artifact, *, published_dir, publish_url, scope):
        assert (artifact, published_dir) == (folder / DOC_HTML, folder)
        events.append("lock")
        yield
        events.append("unlock")

    monkeypatch.setattr(republish, "publication_owner", publish_lock)
    monkeypatch.setattr(republish, "desktop_publish_credential", lambda: ("key", "https://publish.test"))
    monkeypatch.setattr(republish, "is_shared", lambda f: events.append("shared?") or True)
    monkeypatch.setattr(republish, "update_artifact", lambda page, *, scope: events.append("upload"))
    monkeypatch.setattr(republish._republishes, "delay", 0.0)
    assert republish.schedule(folder)
    _settle(republish.pending)
    end = time.monotonic() + 2
    while "unlock" not in events and time.monotonic() < end:
        time.sleep(0.01)
    assert events == ["shared?", "lock", "shared?", "upload", "unlock"]


def test_two_runs_of_one_key_never_overlap():
    """A publish that outlasts the delay: the next one waits for it rather than racing it."""
    active, most, runs = [0], [0], []
    guard = threading.Lock()

    def run(n):
        with guard:
            active[0] += 1
            most[0] = max(most[0], active[0])
        time.sleep(0.15)
        with guard:
            active[0] -= 1
        runs.append(n)

    debouncer = Debouncer(0.0, what="test")
    debouncer.schedule("k", lambda: run(1))
    time.sleep(0.05)  # the first is running now
    debouncer.schedule("k", lambda: run(2))
    end = time.monotonic() + 2
    while len(runs) < 2 and time.monotonic() < end:
        time.sleep(0.01)
    assert runs == [1, 2] and most[0] == 1


ORG = TenantScope(org_mode=True, org_id="org-1", user_id="u1")


def _deployment(monkeypatch, *, mode="org"):
    monkeypatch.setattr(republish, "get_app_settings", lambda: SimpleNamespace(tenancy_mode=mode))
    queued = []
    monkeypatch.setattr("cowork.docs.publish_queue.schedule_enqueue", lambda job: queued.append(job) or True)
    return queued


def test_an_org_deployment_queues_the_page_in_redis(folder, monkeypatch):
    queued = _deployment(monkeypatch)
    assert republish.schedule(folder, scope=ORG, project_id="p1", artifact_id="a1")
    job = queued[0]
    assert (job.artifact_id, job.project_id, job.slug, job.org_id) == ("a1", "p1", folder.name, "org-1")
    assert job.artifacts_base == str(folder.parent)


def test_an_org_deployment_queues_nothing_without_an_org_scope_a_project_or_a_doc_id(folder, monkeypatch):
    queued = _deployment(monkeypatch)
    assert not republish.schedule(folder, scope=LOCAL_SCOPE, project_id="p1", artifact_id="a1")
    assert not republish.schedule(folder, scope=ORG, project_id=None, artifact_id="a1")
    assert not republish.schedule(folder, scope=ORG, project_id="p1", artifact_id=None)
    assert queued == []


def test_a_local_deployment_that_trusts_identity_headers_keeps_the_desktop_republish(folder, monkeypatch):
    queued = _deployment(monkeypatch, mode="local")
    monkeypatch.setattr(republish, "is_shared", lambda f: True)
    debounced = []
    monkeypatch.setattr(republish._republishes, "schedule", lambda key, fn: debounced.append(key))
    personal = TenantScope(personal=True, org_id="me", user_id="me")
    assert republish.schedule(folder, scope=personal, project_id="p1", artifact_id="a1")
    assert debounced == [str(folder / "doc.html")] and queued == []


def test_importing_the_republish_module_leaves_the_org_queue_unloaded():
    import subprocess
    import sys

    org_only = ("cowork.docs.publish_queue", "cowork.turnqueue.redis_client",
                "cowork.services.live_publish", "cowork.services.artifact_autopublish")
    code = ("import sys, cowork.docs.republish; "
            f"print(sorted(m for m in {org_only!r} if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1], timeout=120)
    assert out.stdout.strip() == "[]"
