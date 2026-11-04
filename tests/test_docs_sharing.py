"""Which docs are shared, catching them up at startup, and the debouncer their follow-ups run on."""
from __future__ import annotations

import contextvars
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

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
