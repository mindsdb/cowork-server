"""A shared doc's page, its live copy in MindsHub, and the owner's edits from the shared page."""
from __future__ import annotations

import contextlib
import json
import time
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from cowork.docs import cloud, page_editor, republish, sharing
from cowork.docs.page_editor import EDITOR_DIR_ENV, EDITOR_FILES
from cowork.docs.service import apply, create_doc, render_page
from cowork.docs.store import DOC_HTML, load_doc
from cowork.services import artifacts, publish, share_hooks

PAGE_URL = "https://view.staging.mindshub.ai/view/215f76ea7/4bbd890f"
KEY = "artifact/814c74db-0218-48d5-a29d-50be7bbfa243"
PAGE = cloud.SharedPage("215f76ea7", "4bbd890f", KEY)


def _publish_record(folder: Path, **entry) -> None:
    (folder / ".published.json").write_text(json.dumps({DOC_HTML: {
        "url": PAGE_URL, "published": True, "artifact_key": KEY, "report_id": "4bbd890f", **entry,
    }}))


def _settle(pending, timeout: float = 2.0) -> None:
    """Wait for debounced work to run."""
    end = time.monotonic() + timeout
    while pending() and time.monotonic() < end:
        time.sleep(0.01)


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """A doc, shared at PAGE_URL, with MindsHub calls captured instead of sent."""
    monkeypatch.setattr(cloud, "_endpoint", lambda: ("https://api.test/v1", "mdb_key"))
    monkeypatch.setattr(republish, "schedule", lambda folder, **kw: True)
    folder = create_doc(tmp_path / ".anton" / "artifacts", title="Launch plan",
                        markdown="Launch is Nov 18.\n\n- [ ] Book the security review").folder
    _publish_record(folder)
    monkeypatch.setattr(cloud, "shared_doc_folders", lambda: [folder])
    calls = []

    def fake(method):
        def send(url, **kw):
            calls.append((method, url, kw.get("json")))
            return type("R", (), {"status_code": 200})()
        return send

    monkeypatch.setattr(cloud.httpx, "put", fake("PUT"))
    monkeypatch.setattr(cloud.httpx, "post", fake("POST"))
    monkeypatch.setattr(cloud.httpx, "delete", fake("DELETE"))
    with cloud._lock:
        cloud._folders.clear()
    cloud._unsupported_until = 0.0
    yield folder, calls
    cloud.stop()
    cloud._unsupported_until = 0.0


def _editor_dir(tmp_path: Path) -> Path:
    src = tmp_path / "doc-editor"
    src.mkdir()
    (src / "doc-editor.js").write_text("(()=>{window.__booted=1;const k='password=1';})();\n")
    (src / "doc-editor.css").write_text(".doc-web{position:fixed}\n")
    return src


# --- which docs are shared ---


def test_a_doc_names_the_page_it_is_shared_as(shared):
    folder, _ = shared
    assert cloud.shared_page(folder) == PAGE
    assert cloud.shared_page(folder).artifact_id == "215f76ea7/4bbd890f"
    _publish_record(folder, published=False)  # sharing stopped: the entry stays, marked off
    assert cloud.shared_page(folder) is None and not sharing.is_shared(folder)


@pytest.mark.parametrize("url", [
    "https://view.test/view/../4bbd890f", "https://view.test/view/a/b%2F..", "https://view.test/elsewhere/a/b",
])
def test_a_page_address_that_isnt_plain_names_no_page(shared, url):
    folder, _ = shared
    _publish_record(folder, url=url)
    assert cloud.shared_page(folder) is None


def test_a_publish_entry_names_its_page_after_the_folder_stops_saying_so(shared):
    folder, _ = shared
    """Unsharing reads the entry before it is marked off; deleting, before the folder goes."""
    entry = json.loads((folder / ".published.json").read_text())[DOC_HTML]
    _publish_record(folder, published=False)
    assert cloud.page_from_record(entry) == PAGE
    assert cloud.page_from_record({}) is None and cloud.page_from_record(None) is None


def test_link_editing_is_off_where_several_people_share_the_server(shared, tmp_path, monkeypatch):
    folder, calls = shared
    monkeypatch.setenv(EDITOR_DIR_ENV, str(_editor_dir(tmp_path)))
    monkeypatch.setattr(page_editor, "link_editing_available", lambda: False)
    monkeypatch.setattr(cloud, "link_editing_available", lambda: False)
    render_page(folder, load_doc(folder))
    assert "doc-editor.js" not in (folder / DOC_HTML).read_text()
    assert cloud.schedule_mirror(folder) is False
    cloud.after_share(folder)
    assert calls == []


# --- mirroring this machine's copy ---


def test_mirroring_sends_the_doc_as_it_is_now(shared):
    folder, calls = shared
    assert cloud.mirror_now(folder) is True
    [(method, url, body)] = calls
    assert method == "PUT" and url == "https://api.test/v1/artifact-docs/215f76ea7/4bbd890f/state"
    assert body["rev"] == load_doc(folder)["rev"] and body["doc"]["title"] == "Launch plan"
    assert body["artifactKey"] == KEY


def test_mirroring_waits_for_writes_to_settle_and_skips_unshared_docs(shared, tmp_path, monkeypatch):
    folder, _ = shared
    sent = []
    monkeypatch.setattr(cloud, "mirror_now", lambda f: sent.append(f) or True)
    monkeypatch.setattr(cloud, "ensure_listening", lambda: None)
    for _ in range(3):
        assert cloud.schedule_mirror(folder, delay=0.05)
    _settle(cloud.pending_mirrors)
    assert sent == [folder]
    other = create_doc(tmp_path / "other" / ".anton" / "artifacts", title="Not shared", markdown="Hi").folder
    assert cloud.schedule_mirror(other) is False


def test_a_mindshub_without_shared_doc_editing_is_left_alone_for_a_while(shared, monkeypatch):
    folder, calls = shared
    monkeypatch.setattr(cloud.httpx, "put",
                        lambda url, **kw: calls.append(("PUT", url, None)) or type("R", (), {"status_code": 404})())
    assert cloud.mirror_now(folder) is False
    assert cloud.mirror_now(folder) is False
    assert len(calls) == 1  # the second write did not ask again


# --- the owner's edits from the shared page ---


def test_an_edit_from_the_shared_page_lands_as_the_owners_and_is_mirrored_first(shared, monkeypatch):
    folder, calls = shared
    order = []
    real_mirror = cloud.mirror_now
    monkeypatch.setattr(cloud, "mirror_now", lambda f: order.append("mirror") or real_mirror(f))
    doc = load_doc(folder)
    task = next(b for b in doc["tabs"][0]["blocks"] if b["type"] == "task")
    outcome = cloud.apply_edit({
        "id": "e1", "artifact_id": PAGE.artifact_id, "base_rev": doc["rev"], "client_id": "c1",
        "ops": [{"op": "update", "id": task["id"], "set": {"done": True}}],
    })
    after = load_doc(folder)
    assert outcome == {"status": "applied", "rev": after["rev"]}
    done = next(b for b in after["tabs"][0]["blocks"] if b["id"] == task["id"])
    assert done["done"] is True and done["by"] == "user"
    # The page hears "applied" only after MindsHub holds the new copy.
    assert order == ["mirror"] and calls[-1][0] == "PUT" and calls[-1][2]["rev"] == after["rev"]


def test_an_edit_to_a_block_changed_here_first_is_refused_as_stale(shared):
    folder, _ = shared
    doc = load_doc(folder)
    lead = doc["tabs"][0]["blocks"][0]
    apply(folder, [{"op": "edit_text", "find": "Nov 18", "replace": "Nov 20"}], by="agent")
    outcome = cloud.apply_edit({
        "id": "e2", "artifact_id": PAGE.artifact_id, "base_rev": doc["rev"],
        "ops": [{"op": "update", "id": lead["id"], "set": {"text": "Launch is Dec 1."}}],
    })
    assert outcome["status"] == "rejected" and outcome["error"]["code"] == "stale"
    # The refusal says which op and how to recover, as the agent's would.
    assert outcome["error"]["message"].startswith("op 0 (update")
    assert "Nov 20" in load_doc(folder)["tabs"][0]["blocks"][0]["text"]


def test_an_edit_with_a_rev_that_isnt_a_number_is_refused(shared):
    outcome = cloud.apply_edit({"id": "e4", "artifact_id": PAGE.artifact_id, "base_rev": "latest",
                                "ops": [{"op": "delete", "target": "x"}]})
    assert outcome["status"] == "rejected"


def test_an_edit_for_a_doc_not_on_this_machine_is_left_for_the_one_that_holds_it(shared):
    _, calls = shared
    edit = {"id": "e3", "artifact_id": "215f76ea7/ffffffff", "ops": [{"op": "delete", "target": "x"}]}
    assert cloud.apply_edit(edit) is None
    # The owner's other computer holds it: answering here would settle the edit as refused.
    cloud.handle_edit_event(edit)
    assert not [c for c in calls if c[0] == "POST"]


def test_an_edit_delivered_again_after_its_report_was_lost_is_reported_not_reapplied(shared):
    folder, calls = shared
    doc = load_doc(folder)
    edit = {"id": "e6", "artifact_id": PAGE.artifact_id, "base_rev": doc["rev"],
            "ops": [{"op": "insert", "after": doc["tabs"][0]["blocks"][-1]["id"], "markdown": "Added once."}]}
    first = cloud.apply_edit(edit)
    again = cloud.apply_edit(edit)
    after = load_doc(folder)
    assert first == again == {"status": "applied", "rev": after["rev"]}
    assert after["rev"] == doc["rev"] + 1
    assert [b["text"] for b in after["tabs"][0]["blocks"]].count("Added once.") == 1
    # The page rereads the copy when it hears back, so it goes out again first.
    assert calls[-1][0] == "PUT"


def test_an_edit_that_fails_unexpectedly_is_reported_and_the_stream_goes_on(shared, monkeypatch):
    _, calls = shared

    def boom(event):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(cloud, "apply_edit", boom)
    cloud.handle_edit_event({"id": "e5", "artifact_id": PAGE.artifact_id})
    method, url, body = calls[-1]
    assert method == "POST" and url.endswith("/edits/e5/result")
    assert body == {"status": "rejected", "error": {"code": "error", "message": "disk on fire"}}


def _stream_answering(*responses):
    """An httpx.stream stand-in answering each connect with the next (status, lines)."""
    answers = list(responses)

    @contextlib.contextmanager
    def stream(method, url, **kw):
        assert (method, url) == ("GET", "https://api.test/v1/artifact-docs/stream")
        assert kw["headers"]["Authorization"] == "Bearer mdb_key"
        status, lines, after = answers.pop(0)

        class Res:
            status_code = status
            request = None

            def iter_lines(self):
                yield from lines
                after()

        yield Res()

    return stream


def test_the_listener_applies_each_streamed_edit_and_reports_back(shared, monkeypatch):
    folder, calls = shared
    doc = load_doc(folder)
    task = next(b for b in doc["tabs"][0]["blocks"] if b["type"] == "task")
    edit = {"type": "doc.edit", "id": "e9", "artifact_id": PAGE.artifact_id, "base_rev": doc["rev"],
            "ops": [{"op": "update", "id": task["id"], "set": {"done": True}}]}
    listener = cloud._EditListener()
    lines = [": keepalive", "", "event: doc.edit", f"data: {json.dumps(edit)}", ""]
    monkeypatch.setattr(cloud.httpx, "stream", _stream_answering((200, lines, listener.stopping.set)))
    listener.run()
    method, url, body = calls[-1]
    assert method == "POST"
    assert url == "https://api.test/v1/artifact-docs/215f76ea7/4bbd890f/edits/e9/result"
    assert body == {"status": "applied", "rev": load_doc(folder)["rev"]}


def test_the_listener_reconnects_after_a_refused_stream_with_growing_waits(shared, monkeypatch):
    listener = cloud._EditListener()
    waits = []
    monkeypatch.setattr(listener.stopping, "wait", lambda seconds: waits.append(seconds))
    monkeypatch.setattr(cloud.httpx, "stream", _stream_answering(
        (503, [], lambda: None), (503, [], lambda: None), (200, [], listener.stopping.set)))
    listener.run()
    assert waits[:2] == [1.0, 2.0]


def test_sse_frames_decode_and_comments_are_skipped():
    lines = [": keepalive", "", "event: doc.edit", 'data: {"id": "a",', 'data:  "ops": []}', "", "data: not json", ""]
    assert list(cloud.sse_events(iter(lines))) == [{"id": "a", "ops": []}]


# --- the page carries the owner editor ---


def test_the_page_carries_the_editor_when_the_app_provides_it(shared, tmp_path, monkeypatch):
    folder, _ = shared
    render_page(folder, load_doc(folder))
    assert "doc-editor.js" not in (folder / DOC_HTML).read_text()  # no editor build: a read-only page

    src = _editor_dir(tmp_path)
    monkeypatch.setenv(EDITOR_DIR_ENV, str(src))
    render_page(folder, load_doc(folder))
    page = (folder / DOC_HTML).read_text()
    assert "<script defer src='doc-editor.js'></script>" in page
    assert "<link rel='prefetch' href='doc-editor.css' as='style'>" in page
    for name in EDITOR_FILES:
        assert (folder / name).read_bytes() == (src / name).read_bytes()


def test_only_a_shared_doc_keeps_the_editor_beside_it(shared, tmp_path, monkeypatch):
    """The editor is ~600 KB: copied into every doc it filled project folders
    (a project can be a git repository) for a page that only runs when shared."""
    folder, _ = shared
    monkeypatch.setenv(EDITOR_DIR_ENV, str(_editor_dir(tmp_path)))
    mine = create_doc(tmp_path / ".anton" / "artifacts", title="Private notes", markdown="Just mine.").folder
    assert not any((mine / name).exists() for name in EDITOR_FILES)
    assert "doc-editor.js" not in (mine / DOC_HTML).read_text()

    render_page(folder, load_doc(folder))
    assert all((folder / name).is_file() for name in EDITOR_FILES)
    _publish_record(folder, published=False)
    render_page(folder, load_doc(folder))
    assert not any((folder / name).exists() for name in EDITOR_FILES)
    assert "doc-editor.js" not in (folder / DOC_HTML).read_text()


def test_the_published_bundle_holds_the_editor_byte_for_byte(shared, tmp_path, monkeypatch):
    from anton.publisher import _zip_html

    folder, _ = shared
    src = _editor_dir(tmp_path)
    monkeypatch.setenv(EDITOR_DIR_ENV, str(src))
    render_page(folder, load_doc(folder))
    bundle = zipfile.ZipFile(BytesIO(_zip_html(folder / DOC_HTML)))
    assert sorted(bundle.namelist()) == ["doc-editor.css", "doc-editor.js", "index.html"]
    for name in EDITOR_FILES:
        assert bundle.read(name) == (src / name).read_bytes()


# --- sharing starting and stopping, by any path (button, agent, delete) ---


@pytest.fixture
def publishing(shared, monkeypatch):
    """The publish service with the upload and the remote delete stubbed out."""
    folder, calls = shared
    sent = []

    def upload(artifact, **kw):
        sent.append((folder / DOC_HTML).read_text())
        _publish_record(folder)
        return {"url": PAGE_URL}

    monkeypatch.setattr(publish, "_publish_bundle", upload)
    monkeypatch.setattr("anton.publisher.unpublish", lambda *a, **kw: None)
    monkeypatch.setattr(share_hooks, "_hooks", None)
    cloud.install()  # as the server does at startup
    _publish_record(folder, published=False)
    return folder, calls, sent


def test_sharing_a_doc_sends_its_page_fresh_with_the_editor_and_starts_its_live_copy(publishing, tmp_path, monkeypatch):
    folder, calls, sent = publishing
    render_page(folder, load_doc(folder))
    assert "doc-editor.js" not in (folder / DOC_HTML).read_text()
    monkeypatch.setenv(EDITOR_DIR_ENV, str(_editor_dir(tmp_path)))
    monkeypatch.setattr(cloud, "ensure_listening", lambda: None)
    publish.publish_artifact(folder / DOC_HTML, artifacts_base=folder.parent, api_key="k", publish_url="u")
    # A doc not edited since the editor build arrived would otherwise go out read-only.
    assert "<script defer src='doc-editor.js'></script>" in sent[0]
    assert [(m, u) for m, u, _ in calls] == [("PUT", "https://api.test/v1/artifact-docs/215f76ea7/4bbd890f/state")]


def test_unsharing_a_doc_drops_its_live_copy_and_takes_the_editor_away(publishing, tmp_path, monkeypatch):
    folder, calls, _ = publishing
    monkeypatch.setenv(EDITOR_DIR_ENV, str(_editor_dir(tmp_path)))
    monkeypatch.setattr(cloud, "ensure_listening", lambda: None)
    publish.publish_artifact(folder / DOC_HTML, artifacts_base=folder.parent, api_key="k", publish_url="u")
    assert all((folder / name).is_file() for name in EDITOR_FILES)
    calls.clear()
    # The Stop sharing button and deleting the doc both unpublish through here.
    publish.unpublish_artifact(folder / DOC_HTML, artifacts_base=folder.parent, api_key="k", publish_url="u")
    assert calls == [("DELETE", "https://api.test/v1/artifact-docs/215f76ea7/4bbd890f", None)]
    assert not any((folder / name).exists() for name in EDITOR_FILES)
    assert "doc-editor.js" not in (folder / DOC_HTML).read_text()


def test_unsharing_a_doc_that_no_longer_loads_still_drops_its_live_copy(publishing, monkeypatch):
    folder, calls, _ = publishing
    monkeypatch.setattr(cloud, "ensure_listening", lambda: None)
    publish.publish_artifact(folder / DOC_HTML, artifacts_base=folder.parent, api_key="k", publish_url="u")
    (folder / "doc.cdoc.json").write_text("{not json")
    calls.clear()
    publish.unpublish_artifact(folder / DOC_HTML, artifacts_base=folder.parent, api_key="k", publish_url="u")
    assert calls == [("DELETE", "https://api.test/v1/artifact-docs/215f76ea7/4bbd890f", None)]


def test_deleting_a_shared_doc_drops_its_live_copy(publishing, monkeypatch):
    folder, calls, _ = publishing
    monkeypatch.setattr(cloud, "ensure_listening", lambda: None)
    publish.publish_artifact(folder / DOC_HTML, artifacts_base=folder.parent, api_key="k", publish_url="u")
    calls.clear()
    monkeypatch.setattr(artifacts, "_unpublish_identifier", lambda identifier, entry, **kw: None)
    project = folder.parent.parent.parent
    source = artifacts.ProjectArtifacts(base=folder.parent, project_id="p", project_name=project.name,
                                        trusted_anchor=project, root_parts=(".anton", "artifacts"))
    artifacts.delete_artifact_from_source(source, folder.name, expected_artifact_id=None,
                                          api_key="k", publish_url="u")
    assert not folder.exists()
    assert calls == [("DELETE", "https://api.test/v1/artifact-docs/215f76ea7/4bbd890f", None)]


def test_the_services_know_a_doc_only_by_its_files():
    """cowork.docs sits above the services; they reach its behaviour through share_hooks."""
    import ast

    services = Path(publish.__file__).parent
    reached = set()
    for path in services.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("cowork.docs"):
                reached.add(node.module)
            elif isinstance(node, ast.Import):
                reached.update(a.name for a in node.names if a.name.startswith("cowork.docs"))
    assert reached == {"cowork.docs.store"}


def test_any_other_artifact_publishes_as_before(publishing, monkeypatch):
    folder, calls, _ = publishing
    seen = []
    monkeypatch.setattr(publish, "_publish_bundle", lambda artifact, **kw: seen.append(artifact) or {"url": "x"})
    other = folder.parent / "report.html"
    other.write_text("<html></html>")
    publish.publish_artifact(other, artifacts_base=folder.parent, api_key="k", publish_url="u")
    assert seen == [other] and calls == []
