"""Documents on the web artifacts' path (ENG-4687): a PDF, Word, Excel or
PowerPoint primary gets the revision journal, the workspace and quick edits a
page has, with its bytes never decoded as text.

The journal, the routes and the quick edit stream are the real ones, on a
real artifact folder; only the owner lookup, permissions and anton's editor
are stubbed, as the neighbouring route tests stub them.
"""

from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from _async_stub import awaitable
from cowork.api.v1.endpoints import artifact_quick_edit as quick_ep
from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services import artifact_revisions as revision_service
from cowork.services import quick_edit as quick_edit_service
from cowork.services.artifact_identity import ensure_full_id
from cowork.services.artifact_metadata import ArtifactMetadata
from cowork.services.artifact_revisions import (
    DOCUMENT_EXTENSIONS,
    MAX_DOCUMENT_BYTES,
    MAX_SOURCE_BYTES,
    DocumentNotTextEditable,
    RevisionConflict,
    RevisionValidationError,
    capture_agent_revision,
    current_source,
    current_workspace,
    list_revisions,
    restore_revision,
    revision_content,
    revision_with_content,
    save_source,
)
from cowork.services.quick_edit import Decisions, QuickEditBackend
from cowork.streaming.preview_checks import preview_checks

DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
CONVERSATION = "33333333-3333-4333-8333-333333333333"


def _docx(text: str) -> bytes:
    """A real Word document saying *text*."""
    from docx import Document

    document = Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _make_artifact(tmp_path, primary: str, content: bytes, *, kind: str = "document"):
    folder = tmp_path / "quarterly-report"
    folder.mkdir()
    metadata = {
        "id": "a1b2c3d4",
        "slug": "quarterly-report",
        "createdAt": "2026-11-25T12:00:00+00:00",
        "name": "Quarterly report",
        "type": kind,
        "primary": primary,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / primary).write_bytes(content)
    artifact_id, metadata = ensure_full_id(folder, ArtifactMetadata.model_validate(metadata))
    return folder, metadata, artifact_id


@pytest.fixture
def docx_artifact(tmp_path):
    return _make_artifact(tmp_path, "report.docx", _docx("First draft"))


@pytest.fixture
def pdf_artifact(tmp_path):
    return _make_artifact(tmp_path, "report.pdf", b"%PDF-1.7\n\xe2\xe3\xcf\xd3\n%%EOF\n")


@pytest.fixture
def html_artifact(tmp_path):
    return _make_artifact(tmp_path, "index.html", b"<!doctype html><h1>Sales</h1>\n", kind="html-app")


# ── the journal ───────────────────────────────────────────────────────────


def test_the_document_kinds_are_the_binary_publishable_ones():
    from cowork.services.file_page import PUBLISHABLE_FILE_SUFFIXES
    from cowork.services.publish import PUBLISH_FILE_MAX_BYTES

    assert DOCUMENT_EXTENSIONS == set(PUBLISHABLE_FILE_SUFFIXES) - {".csv", ".txt"}
    assert MAX_DOCUMENT_BYTES == PUBLISH_FILE_MAX_BYTES
    assert MAX_SOURCE_BYTES == 2_000_000


def test_a_document_is_captured_listed_and_restored_as_bytes(docx_artifact):
    folder, metadata, artifact_id = docx_artifact
    first_bytes = (folder / "report.docx").read_bytes()

    first = current_source(folder, metadata, artifact_id)
    assert first["binary"] is True
    assert "content" not in first
    assert first["contentType"] == "docx"
    assert first["path"] == "report.docx"
    assert first["revision"]["number"] == 1
    assert first["revision"]["bytes"] == len(first_bytes)

    second_bytes = _docx("Second draft")
    (folder / "report.docx").write_bytes(second_bytes)
    second = current_source(folder, metadata, artifact_id)
    assert second["revision"]["number"] == 2
    assert [r["number"] for r in list_revisions(folder, rel_path="report.docx")] == [2, 1]

    # A revision read is its metadata alone; the bytes are the content route's.
    old = revision_with_content(folder, first["revision"]["id"])
    assert "content" not in old
    assert old["id"] == first["revision"]["id"]
    found = revision_content(folder, first["revision"]["id"])
    assert (found.data, found.content_type, found.path) == (first_bytes, DOCX_TYPE, "report.docx")

    with pytest.raises(RevisionConflict):
        restore_revision(
            folder, metadata, artifact_id, first["revision"]["id"],
            expected_revision_id=first["revision"]["id"],
        )
    assert (folder / "report.docx").read_bytes() == second_bytes

    restored = restore_revision(
        folder, metadata, artifact_id, first["revision"]["id"],
        expected_revision_id=second["revision"]["id"], actor_id="user-1",
    )
    assert (folder / "report.docx").read_bytes() == first_bytes
    assert restored["binary"] is True and "content" not in restored
    assert restored["revision"]["number"] == 3
    assert restored["revision"]["summary"] == "Restored revision 1"
    assert restored["revision"]["actor"] == {"kind": "manual", "id": "user-1"}
    assert restored["revision"]["contentHash"] == first["revision"]["contentHash"]


def test_a_document_past_the_text_bound_is_versioned_up_to_its_own(docx_artifact):
    folder, metadata, artifact_id = docx_artifact
    # Larger than any page source the journal keeps, well inside a document's.
    big = _docx("Big") + os.urandom(MAX_SOURCE_BYTES + 1024)
    (folder / "report.docx").write_bytes(big)
    revision = current_source(folder, metadata, artifact_id)["revision"]
    assert revision_content(folder, revision["id"]).data == big

    with (folder / "report.docx").open("r+b") as stream:
        stream.truncate(MAX_DOCUMENT_BYTES + 1)
    with pytest.raises(RevisionValidationError, match="too large"):
        current_source(folder, metadata, artifact_id)


def test_an_agent_turns_change_to_a_document_is_recorded(docx_artifact):
    folder, metadata, artifact_id = docx_artifact
    before = current_source(folder, metadata, artifact_id)["revision"]

    (folder / "report.docx").write_bytes(_docx("The agent's draft"))
    captured = capture_agent_revision(folder, conversation_id=CONVERSATION, summary="Agent: Rewrote the summary")

    assert captured is not None
    assert captured["actor"] == {"kind": "agent", "id": None}
    assert captured["summary"] == "Agent: Rewrote the summary"
    assert captured["baseRevisionId"] == before["id"]
    assert captured["conversationId"] == CONVERSATION
    assert [r["id"] for r in list_revisions(folder)][0] == captured["id"]


def test_the_text_save_refuses_a_document(docx_artifact):
    folder, metadata, artifact_id = docx_artifact
    original = (folder / "report.docx").read_bytes()
    head = current_source(folder, metadata, artifact_id)["revision"]

    with pytest.raises(DocumentNotTextEditable):
        save_source(folder, metadata, artifact_id, content="plain text", expected_revision_id=head["id"])

    assert (folder / "report.docx").read_bytes() == original
    assert len(list_revisions(folder)) == 1


def test_a_text_source_still_wins_the_fallback_over_a_document(tmp_path):
    folder, metadata, artifact_id = _make_artifact(tmp_path, "report.docx", _docx("Doc"))
    (folder / "notes.md").write_text("# Notes\n", encoding="utf-8")
    unset = ArtifactMetadata.model_validate({**metadata.to_document(), "primary": ""})
    assert current_source(folder, unset, artifact_id)["path"] == "notes.md"
    (folder / "notes.md").unlink()
    assert current_source(folder, unset, artifact_id)["path"] == "report.docx"


def test_text_sources_answer_exactly_as_before(html_artifact):
    folder, metadata, artifact_id = html_artifact
    source = current_source(folder, metadata, artifact_id)
    assert set(source) == {"artifactId", "path", "content", "contentType", "revision"}
    assert source["content"] == "<!doctype html><h1>Sales</h1>\n"
    assert set(current_workspace(folder, metadata, artifact_id)) == {
        "artifactId", "path", "content", "contentType", "revision", "revisions",
    }
    saved = save_source(
        folder, metadata, artifact_id, content="<h1>Two</h1>", expected_revision_id=source["revision"]["id"],
    )
    assert set(saved) == {"artifactId", "path", "content", "revision"}
    old = revision_with_content(folder, source["revision"]["id"])
    assert old["content"] == "<!doctype html><h1>Sales</h1>\n"
    restored = restore_revision(
        folder, metadata, artifact_id, source["revision"]["id"], expected_revision_id=saved["revision"]["id"],
    )
    assert set(restored) == {"artifactId", "path", "content", "revision"}
    assert restored["content"] == "<!doctype html><h1>Sales</h1>\n"
    # Page bytes are never served from the API's origin.
    with pytest.raises(RevisionValidationError):
        revision_content(folder, source["revision"]["id"])


def _blobs(folder) -> set[str]:
    return {p.name for p in (folder / ".revisions" / "blobs").iterdir()}


def _new_document_version(folder, metadata, artifact_id, number: int) -> dict:
    (folder / "report.docx").write_bytes(b"PK\x03\x04" + bytes([number]) * 1000)
    return current_source(folder, metadata, artifact_id)["revision"]


def test_a_document_over_the_byte_budget_drops_its_oldest_revisions(docx_artifact, monkeypatch):
    folder, metadata, artifact_id = docx_artifact
    # Room for three 1 KB versions; the fixture's first draft is larger.
    monkeypatch.setattr(revision_service, "MAX_REVISION_BYTES", 3 * 1004)
    first = current_source(folder, metadata, artifact_id)["revision"]
    versions = [_new_document_version(folder, metadata, artifact_id, n) for n in range(1, 7)]

    kept = list_revisions(folder)
    # The newest three, the oldest gone, and their blobs with them.
    assert [r["id"] for r in kept] == [v["id"] for v in reversed(versions[-3:])]
    assert _blobs(folder) == {r["contentHash"] for r in kept}
    assert first["contentHash"] not in _blobs(folder)
    assert sum(r["bytes"] for r in kept) <= revision_service.MAX_REVISION_BYTES


def test_the_byte_budget_keeps_the_head_and_the_published_revision(docx_artifact, monkeypatch):
    folder, metadata, artifact_id = docx_artifact
    first = current_source(folder, metadata, artifact_id)["revision"]
    # The published revision is the old, large first draft.
    (folder / ".published.json").write_text(json.dumps({
        "report.docx": {"report_id": "r-1", "url": "https://x", "published_revision_id": first["id"]},
    }), encoding="utf-8")
    monkeypatch.setattr(revision_service, "MAX_REVISION_BYTES", first["bytes"] + 2 * 1004)
    versions = [_new_document_version(folder, metadata, artifact_id, n) for n in range(1, 7)]

    kept = [r["id"] for r in list_revisions(folder)]
    assert kept == [versions[-1]["id"], versions[-2]["id"], first["id"]]
    assert revision_content(folder, first["id"]).data
    entry = json.loads((folder / ".published.json").read_text())["report.docx"]
    assert revision_service.published_revision(folder, entry, "report.docx")["id"] == first["id"]

    # Even a head past the budget on its own is kept.
    monkeypatch.setattr(revision_service, "MAX_REVISION_BYTES", 10)
    head = _new_document_version(folder, metadata, artifact_id, 9)
    assert [r["id"] for r in list_revisions(folder)] == [head["id"], first["id"]]


def test_the_byte_budget_keeps_revisions_an_open_repair_names(html_artifact, monkeypatch):
    folder, metadata, artifact_id = html_artifact
    base = current_source(folder, metadata, artifact_id)["revision"]
    revision_service.create_agent_repair(
        folder, metadata, artifact_id, expected_revision_id=base["id"], comment_thread_id="t-1",
        selector="h1", thread=[{"text": "Bigger"}], conversation_id=CONVERSATION,
    )
    monkeypatch.setattr(revision_service, "MAX_REVISION_BYTES", 1)
    (folder / "index.html").write_text("<h1>Agent</h1>", encoding="utf-8")
    agent = capture_agent_revision(folder, conversation_id=CONVERSATION)
    (folder / "index.html").write_text("<h1>Later</h1>", encoding="utf-8")
    head = current_source(folder, metadata, artifact_id)["revision"]

    assert [r["id"] for r in list_revisions(folder)] == [head["id"], agent["id"], base["id"]]


def test_text_artifacts_stay_under_the_byte_budget():
    # 80 revisions of the largest page source fit, so text keeps its count rule.
    assert revision_service.MAX_REVISIONS * MAX_SOURCE_BYTES <= revision_service.MAX_REVISION_BYTES


# ── the routes ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean():
    yield
    preview_checks.reset()
    quick_edit_service.quick_edit_locks.reset()
    get_app_settings.cache_clear()


@pytest.fixture
def client(monkeypatch):
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield TestClient(create_app())


OWNER = {"canEdit": True, "canAddressWithAgent": True, "canResolveComments": True}


def _owned(monkeypatch, artifact):
    """The routes find *artifact* and its caller owns it, with every product
    permission, quick edits on and a decisions model for them."""
    folder, metadata, _ = artifact
    monkeypatch.setattr(
        workspace_ep, "_owner_workspace",
        awaitable(lambda *_a: (SimpleNamespace(project_id=None, base=str(folder.parent)), folder, metadata, dict(OWNER))),
    )
    monkeypatch.setattr(quick_ep, "_owner_workspace", awaitable(lambda *_a: (None, folder, metadata, dict(OWNER))))
    monkeypatch.setattr(workspace_ep, "_sync_live_artifact", awaitable(lambda *_a, **_k: None))
    monkeypatch.setattr(workspace_ep, "has_product_permission", awaitable(lambda scope, permission: True))
    monkeypatch.setattr(workspace_ep, "quick_edit_available", lambda: True)
    monkeypatch.setattr(workspace_ep, "direct_edit_available", awaitable(lambda scope, **_: True))


def _ooxml_edits(monkeypatch, on: bool):
    """The installed anton with or without the OOXML_EDITS flag."""
    import anton.core.artifact_editor as editor

    if on:
        monkeypatch.setattr(editor, "OOXML_EDITS", True, raising=False)
    else:
        monkeypatch.delattr(editor, "OOXML_EDITS", raising=False)


def _url(artifact_id: str, tail: str = "") -> str:
    return f"/api/v1/artifacts/workspace/local/{artifact_id}{tail}"


def test_the_workspace_describes_a_document_without_content(client, docx_artifact, monkeypatch):
    folder, _metadata, artifact_id = docx_artifact
    _owned(monkeypatch, docx_artifact)
    _ooxml_edits(monkeypatch, True)

    response = client.get(_url(artifact_id))

    assert response.status_code == 200
    body = response.json()
    assert "content" not in body
    assert body["binary"] is True
    assert (body["path"], body["contentType"], body["artifactId"]) == ("report.docx", "docx", artifact_id)
    assert body["revision"] == body["revisions"][0]
    assert body["publishedRevision"] is None
    assert body["repair"] is None
    assert (body["capabilities"]["canQuickEdit"], body["capabilities"]["quickEditDirect"]) == (True, True)


def test_the_workspace_of_a_page_keeps_its_shape(client, html_artifact, monkeypatch):
    _folder, _metadata, artifact_id = html_artifact
    _owned(monkeypatch, html_artifact)

    body = client.get(_url(artifact_id)).json()

    assert body["content"] == "<!doctype html><h1>Sales</h1>\n"
    assert "binary" not in body
    assert body["contentType"] == "html"
    revision = client.get(_url(artifact_id, f"/revisions/{body['revision']['id']}")).json()
    assert revision["content"] == "<!doctype html><h1>Sales</h1>\n"
    assert "contentUrl" not in revision
    assert client.get(_url(artifact_id, f"/revisions/{body['revision']['id']}/content")).status_code == 422


@pytest.mark.parametrize("primary,ooxml_flag,expected", [
    ("report.docx", True, (True, True)),
    # An anton without the flag, and a PDF: the bar stays, as a way into the chat.
    ("report.docx", False, (True, False)),
    ("report.pdf", True, (True, False)),
    ("index.html", False, (True, True)),
])
async def test_the_quick_edit_capability_follows_the_document_kind(monkeypatch, primary, ooxml_flag, expected):
    monkeypatch.setattr(workspace_ep, "has_product_permission", awaitable(lambda scope, permission: True))
    monkeypatch.setattr(workspace_ep, "quick_edit_available", lambda: True)
    monkeypatch.setattr(workspace_ep, "direct_edit_available", awaitable(lambda scope, **_: True))
    _ooxml_edits(monkeypatch, ooxml_flag)

    current = await workspace_ep._current_capabilities(LOCAL_SCOPE, dict(OWNER), source_path=primary)

    assert (current["canQuickEdit"], current["quickEditDirect"]) == expected


def test_a_document_revision_points_at_its_bytes(client, docx_artifact, monkeypatch):
    folder, metadata, artifact_id = docx_artifact
    _owned(monkeypatch, docx_artifact)
    first_bytes = (folder / "report.docx").read_bytes()
    first = current_source(folder, metadata, artifact_id)["revision"]
    (folder / "report.docx").write_bytes(_docx("Second draft"))

    revision = client.get(_url(artifact_id, f"/revisions/{first['id']}"))

    assert revision.status_code == 200
    body = revision.json()
    assert "content" not in body
    assert body["id"] == first["id"] and body["path"] == "report.docx"
    assert body["contentUrl"] == _url(artifact_id, f"/revisions/{first['id']}/content")

    content = client.get(body["contentUrl"])
    assert content.status_code == 200
    assert content.content == first_bytes
    assert content.headers["content-type"] == DOCX_TYPE
    assert content.headers["cache-control"] == "private, no-store"
    assert content.headers["x-content-type-options"] == "nosniff"

    assert client.get(_url(artifact_id, "/revisions/no-such-revision/content")).status_code == 404


def test_the_content_route_is_owner_authorized(client, docx_artifact, monkeypatch):
    from fastapi import HTTPException

    folder, metadata, artifact_id = docx_artifact
    revision = current_source(folder, metadata, artifact_id)["revision"]

    async def refused(*_a):
        raise HTTPException(status_code=403, detail="Only the owner can do this")

    monkeypatch.setattr(workspace_ep, "_owner_workspace", refused)
    assert client.get(_url(artifact_id, f"/revisions/{revision['id']}/content")).status_code == 403


def test_the_source_put_refuses_a_document(client, docx_artifact, monkeypatch):
    folder, metadata, artifact_id = docx_artifact
    _owned(monkeypatch, docx_artifact)
    original = (folder / "report.docx").read_bytes()
    head = current_source(folder, metadata, artifact_id)["revision"]

    response = client.put(_url(artifact_id), json={"content": "text", "expectedRevisionId": head["id"]})

    assert response.status_code == 422
    assert response.json()["detail"] == "A document can't be edited as text"
    assert (folder / "report.docx").read_bytes() == original


def test_the_restore_route_puts_a_documents_bytes_back(client, docx_artifact, monkeypatch):
    folder, metadata, artifact_id = docx_artifact
    _owned(monkeypatch, docx_artifact)
    first_bytes = (folder / "report.docx").read_bytes()
    first = current_source(folder, metadata, artifact_id)["revision"]
    (folder / "report.docx").write_bytes(_docx("Second draft"))
    second = current_source(folder, metadata, artifact_id)["revision"]

    stale = client.post(_url(artifact_id, f"/revisions/{first['id']}/restore"), json={"expectedRevisionId": first["id"]})
    assert stale.status_code == 409
    assert stale.json()["detail"]["currentRevision"]["id"] == second["id"]

    restored = client.post(
        _url(artifact_id, f"/revisions/{first['id']}/restore"), json={"expectedRevisionId": second["id"]},
    )
    assert restored.status_code == 200
    assert restored.json()["binary"] is True
    assert restored.json()["revision"]["summary"] == "Restored revision 1"
    assert (folder / "report.docx").read_bytes() == first_bytes


# ── quick edits ───────────────────────────────────────────────────────────


class _Anton:
    """anton's quick-edit API, as the contract gives it; ``edit`` decides
    what run_quick_edit does."""

    def __init__(self, edit=None):
        self.edit = edit
        self.edit_calls: list[dict] = []

    async def route_request(self, message, *, files, kind, base_url, api_key, **_extra):
        return SimpleNamespace(route="edit", confidence=0.9, reason="jev")

    async def run_quick_edit(self, folder, message, *, kind, llm, checker, history=None, on_progress=None):
        self.edit_calls.append({"kind": kind, "checker": checker})
        return await self.edit(folder)

    def api(self) -> quick_edit_service.AntonQuickEdit:
        def no_preview(kind, *, slug, preview_checker):
            raise AssertionError("a document has no page for the preview to check")

        return quick_edit_service.AntonQuickEdit(
            route_request=self.route_request,
            run_quick_edit=self.run_quick_edit,
            checkers=SimpleNamespace(select_checker=no_preview),
            workspace_cls=None,
        )


def _quick_edits(monkeypatch, artifact, anton: _Anton):
    _owned(monkeypatch, artifact)
    monkeypatch.setattr(quick_edit_service, "anton_quick_edit", lambda: anton.api())
    monkeypatch.setattr(
        quick_edit_service, "resolve_backend",
        awaitable(lambda *_a, **_k: QuickEditBackend(llm="the-llm", route_base_url="https://hub.example/v1", route_api_key="k")),
    )
    monkeypatch.setattr(
        quick_edit_service, "decisions_for_scope",
        awaitable(lambda anton, scope: ("the-user", Decisions(available=True, model="jev-1.13.0"))),
    )
    monkeypatch.setattr(quick_ep, "_model_choice", awaitable(lambda *_a: (None, None)))
    monkeypatch.setattr(quick_edit_service, "zdr_blocks", awaitable(lambda *_a, **_k: False))


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_a_quick_edit_of_a_word_document_lands_one_revision(client, docx_artifact, monkeypatch):
    folder, metadata, artifact_id = docx_artifact
    edited = _docx("Title in bold")

    async def edit(folder):
        (folder / "report.docx").write_bytes(edited)
        return SimpleNamespace(status="applied", changed=["report.docx"], summary="Made the title bold", check=None)

    anton = _Anton(edit)
    _quick_edits(monkeypatch, docx_artifact, anton)
    _ooxml_edits(monkeypatch, True)
    base = current_source(folder, metadata, artifact_id)["revision"]

    response = client.post(_url(artifact_id, "/quick-edits"), json={"message": "Make the title bold"})

    events = _events(response.text)
    names = [name for name, _ in events]
    assert "check_request" not in names
    assert anton.edit_calls == [{"kind": "document", "checker": None}]
    done = events[-1][1]
    assert events[-1][0] == "done" and done["status"] == "applied"
    revisions = list_revisions(folder)
    assert len(revisions) == 2
    assert revisions[0]["id"] == done["revisionId"]
    assert done["previousRevisionId"] == base["id"]
    assert revisions[0]["summary"] == "Quick edit: Made the title bold"
    assert revision_content(folder, done["revisionId"]).data == edited


@pytest.mark.parametrize("artifact_name,ooxml_flag", [("pdf_artifact", True), ("docx_artifact", False)])
def test_a_document_the_editor_cant_change_goes_to_the_agent(client, request, monkeypatch, artifact_name, ooxml_flag):
    artifact = request.getfixturevalue(artifact_name)
    folder, _metadata, artifact_id = artifact
    original = next(p for p in folder.iterdir() if p.suffix in DOCUMENT_EXTENSIONS).read_bytes()

    async def edit(folder):
        raise AssertionError("the editor must not run on this document")

    anton = _Anton(edit)
    _quick_edits(monkeypatch, artifact, anton)
    _ooxml_edits(monkeypatch, ooxml_flag)

    # Even an edit the app forces.
    response = client.post(_url(artifact_id, "/quick-edits"), json={"message": "Fix the typo", "forceRoute": "edit"})

    events = _events(response.text)
    assert events[0] == ("route", {"route": "agent", "confidence": 1.0, "ms": 0, "reason": "document_not_editable"})
    assert events[-1][0] == "done" and events[-1][1]["status"] == "agent"
    assert anton.edit_calls == []
    assert next(p for p in folder.iterdir() if p.suffix in DOCUMENT_EXTENSIONS).read_bytes() == original
    assert len(list_revisions(folder)) == 1


def test_ooxml_edits_reads_antons_flag_without_calling_it(monkeypatch):
    _ooxml_edits(monkeypatch, False)
    assert quick_edit_service.ooxml_edits_available() is False
    assert quick_edit_service.edits_in_place("report.docx") is False
    assert quick_edit_service.edits_in_place("index.html") is True
    _ooxml_edits(monkeypatch, True)
    assert quick_edit_service.ooxml_edits_available() is True
    assert quick_edit_service.edits_in_place("deck.pptx") is True
    assert quick_edit_service.edits_in_place("scan.pdf") is False
    assert revision_service.is_ooxml("scan.pdf") is False
