"""Docs over HTTP: create, read, edit with the stale guard, watch, export."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture
def client(monkeypatch, granted_product_permissions):
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield TestClient(create_app())
    get_app_settings.cache_clear()


@pytest.fixture
def doc(client):
    res = client.post("/api/v1/docs/", json={
        "projectId": str(GENERAL_PROJECT_ID),
        "title": "Hiring plan",
        "markdown": "We hire 4 engineers by March.\n\n| Role | Status |\n| --- | --- |\n| Backend | Open |\n| Design | Hired |",
    })
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["card"]["kind"] == "Doc" and body["card"]["format"] == "doc"
    return body


def _url(doc, suffix=""):
    return f"/api/v1/docs/{GENERAL_PROJECT_ID}/{doc['artifactId']}{suffix}"


def test_read_and_edit(client, doc):
    res = client.get(_url(doc))
    assert res.status_code == 200
    got = res.json()
    assert got["doc"]["title"] == "Hiring plan"
    assert got["capabilities"]["canEdit"] is True
    lead = got["doc"]["tabs"][0]["blocks"][0]

    res = client.post(_url(doc, "/ops"), json={"ops": [
        {"op": "update", "id": lead["id"], "set": {"text": "We hire 5 engineers by March."}}], "clientId": "tab-1"})
    assert res.status_code == 200, res.text
    assert res.json()["rev"] == 2
    assert res.json()["doc"]["tabs"][0]["blocks"][0]["by"] == "user"


def test_a_stale_write_is_a_409_naming_the_block(client, doc):
    lead = doc["doc"]["tabs"][0]["blocks"][0]
    client.post(_url(doc, "/ops"), json={"ops": [{"op": "update", "id": lead["id"], "set": {"text": "v2"}}]})
    res = client.post(_url(doc, "/ops"), json={"ifRev": 1, "ops": [
        {"op": "update", "id": lead["id"], "set": {"text": "from an old read"}}]})
    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "stale"
    assert res.json()["detail"]["ids"] == [lead["id"]]


def test_a_bad_op_is_a_422_with_the_fix(client, doc):
    res = client.post(_url(doc, "/ops"), json={"ops": [{"op": "explode"}]})
    assert res.status_code == 422
    assert "op 0 (explode" in res.json()["detail"]["message"] or "op 0" in res.json()["detail"]["message"]


def test_an_op_with_a_field_of_the_wrong_type_is_a_422_not_a_500(client, doc):
    res = client.post(_url(doc, "/ops"), json={"ops": [{"op": "insert", "tab": 1, "markdown": "x"}]})
    assert res.status_code == 422, res.text
    assert res.json()["detail"]["code"] == "bad_op" and res.json()["detail"]["op"] == 0
    assert client.get(_url(doc)).json()["doc"]["rev"] == 1


@pytest.mark.parametrize("fmt,marker", [
    ("md", b"| Role | Status |"),
    ("html", b"<table>"),
    ("docx", b"PK"),
    ("pdf", b"%PDF"),
])
def test_export(client, doc, fmt, marker):
    res = client.get(_url(doc, f"/export?format={fmt}"))
    assert res.status_code == 200, res.text
    assert marker in res.content
    assert "Hiring-plan." + fmt in res.headers["content-disposition"]


def test_a_title_in_any_script_exports_under_its_own_name(client):
    from urllib.parse import quote

    res = client.post("/api/v1/docs/", json={"projectId": str(GENERAL_PROJECT_ID), "title": "計画 Q4", "markdown": "Body."})
    exported = client.get(_url(res.json(), "/export?format=md"))
    assert exported.status_code == 200, exported.text
    assert f"filename*=UTF-8''{quote('計画-Q4.md', safe='')}" in exported.headers["content-disposition"]


@pytest.mark.parametrize("fmt", ["pdf", "docx"])
def test_a_shared_deployment_refuses_the_exports_that_fetch_images(fmt):
    """The converters fetch every http(s) image a doc names, from the server."""
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from cowork.api.v1.endpoints.docs import export_doc

    session = SimpleNamespace(scope=SimpleNamespace(partitioned=True))
    with pytest.raises(HTTPException) as refused:
        asyncio.run(export_doc("local", "0" * 32, session, format=fmt))
    assert refused.value.status_code == 403


def test_a_card_that_fails_is_logged_as_a_warning(monkeypatch, caplog, tmp_path):
    from types import SimpleNamespace

    from cowork.api.v1.endpoints import docs

    def broken_card(*_args, **_kwargs):
        raise NameError("name 'oops' is not defined")

    monkeypatch.setattr(docs, "card_for_folder", broken_card)
    source = SimpleNamespace(project_id=None, project_name="")
    with caplog.at_level("DEBUG", logger=docs.logger.name):
        assert docs._card(SimpleNamespace(scope=None), source, tmp_path) is None
    (record,) = [r for r in caplog.records if "no card for doc" in r.getMessage()]
    assert record.levelname == "WARNING" and record.exc_info


def test_a_non_doc_artifact_is_not_found(client):
    res = client.get(f"/api/v1/docs/{GENERAL_PROJECT_ID}/0f9e8d7c-6b5a-4938-8271-605f4e3d2c1b")
    assert res.status_code == 404


def test_a_doc_is_shared_as_its_rendered_page(client, doc):
    """doc.html is kept beside the doc, and the card's publish state is the page's."""
    from pathlib import Path

    card = client.get(_url(doc)).json()["card"]
    page = Path(card["htmlPath"])
    assert page.name == "doc.html" and page.is_file()
    html = page.read_text()
    assert "<h1 class='page-title'>Hiring plan</h1>" in html
    assert card["format"] == "doc" and card["publishedUrl"] == ""
    lead = doc["doc"]["tabs"][0]["blocks"][0]
    client.post(_url(doc, "/ops"), json={"ops": [{"op": "update", "id": lead["id"], "set": {"text": "Five engineers."}}]})
    assert "Five engineers." in page.read_text()


def test_the_rendered_page_is_publishable_like_any_html_artifact(client, doc):
    from pathlib import Path

    from cowork.services.publish import _resolve_publish_target, is_publishable_static

    page = Path(client.get(_url(doc)).json()["card"]["htmlPath"])
    target, published_dir, key, fullstack = _resolve_publish_target(page, container_dirs=[page.parent.parent])
    assert (target, key, fullstack) == (page, "doc.html", False)
    assert published_dir == page.parent
    assert is_publishable_static(target, page.parent.parent)


# ---------------------------------------------------------------- who may see and change what


def test_a_doc_created_over_http_is_its_creators(monkeypatch, tmp_path):
    """In org mode an artifact nobody is recorded as owning refuses every edit."""
    import asyncio
    from types import SimpleNamespace
    from uuid import uuid4

    from cowork.api.v1.endpoints import docs

    recorded = []

    async def allowed(scope, permission):
        return None

    source = SimpleNamespace(base=str(tmp_path / "artifacts"), project_id="p-1", project_name="P", external=False)
    monkeypatch.setattr(docs, "require_product_permission", allowed)
    monkeypatch.setattr(docs, "artifact_sources_for_request", lambda session, project_id, path: [source])
    monkeypatch.setattr(docs, "record_artifact_owner", lambda *a, **kw: recorded.append((a, kw)))
    monkeypatch.setattr(docs, "_card", lambda *a: None)
    session = SimpleNamespace(scope=SimpleNamespace(user_id="alice"))
    made = asyncio.run(docs.create_blank_doc(docs._CreateBody(projectId=uuid4(), title="Plan"), session))
    assert recorded == [((session, "p-1", made["slug"], "alice"), {"action": "create"})]


def test_a_reviewer_reads_the_doc_without_the_owners_card(monkeypatch, tmp_path):
    """The card carries the share's access: its password and the invited emails."""
    import asyncio
    from types import SimpleNamespace

    from cowork.api.v1.endpoints import docs
    from cowork.docs.service import create_doc

    folder = create_doc(tmp_path, title="Plan", markdown="Hi.").folder
    reviewer = {"role": "reviewer", "canPreview": True, "canComment": True, "canEdit": False,
                "canAddressWithAgent": False, "canResolveComments": False, "ownerUnknown": False}

    async def narrowed(session, capabilities):
        return capabilities

    monkeypatch.setattr(docs, "_resolve", lambda *a, **kw: (SimpleNamespace(), folder, {}, False))
    monkeypatch.setattr(docs, "artifact_capabilities", lambda *a: reviewer)
    monkeypatch.setattr(docs, "current_capabilities", narrowed)
    monkeypatch.setattr(docs, "_card", lambda *a: pytest.fail("a reviewer was handed the owner's card"))
    got = asyncio.run(docs.get_doc("local", "0" * 32, SimpleNamespace(scope=None)))
    assert got["card"] is None and got["capabilities"] == reviewer


def test_only_the_owner_writes_to_a_doc(monkeypatch, tmp_path):
    """A reviewer reads a doc; an edit from them is refused and nothing is written."""
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from cowork.api.v1.endpoints import docs
    from cowork.docs.service import create_doc
    from cowork.docs.store import DOC_FILENAME

    folder = create_doc(tmp_path, title="Plan", markdown="Hi.").folder
    before = (folder / DOC_FILENAME).read_bytes()

    async def allowed(scope, permission):
        return None

    monkeypatch.setattr(docs, "require_product_permission", allowed)
    monkeypatch.setattr(docs, "review_artifact_for_request", lambda *a: (SimpleNamespace(), folder, {}, False))
    body = docs._OpsBody(ops=[{"op": "edit_text", "find": "Hi", "replace": "Bye"}])
    with pytest.raises(HTTPException) as err:
        asyncio.run(docs.edit_doc("local", "0" * 32, body, SimpleNamespace(scope=None)))
    assert err.value.status_code == 403
    assert (folder / DOC_FILENAME).read_bytes() == before


def test_a_doc_stream_sends_announced_writes_and_ones_only_the_file_shows(monkeypatch, tmp_path):
    """An open editor hears the hub's announcements, and also a write from another
    process (a remote turn), which changes the file without one."""
    import asyncio
    import json

    from cowork.api.v1.endpoints import docs
    from cowork.docs import store
    from cowork.docs.hub import doc_topic
    from cowork.docs.ops import apply_ops
    from cowork.docs.service import announce, apply, create_doc

    monkeypatch.setattr(docs, "_POLL_S", 0.02)
    made = create_doc(tmp_path, title="Plan", markdown="Hi.", by="user")
    lead = made.doc["tabs"][0]["blocks"][0]["id"]

    class Open:
        async def is_disconnected(self):
            return False

    def frame(chunk):
        event, data = chunk.split("\n")[:2]
        return event.removeprefix("event: "), json.loads(data.removeprefix("data: "))

    async def scenario():
        stream = docs._sse(Open(), doc_topic(made.artifact_id), initial=made.folder).body_iterator
        frames = [frame(await anext(stream))]
        doc, result = apply(made.folder, [{"op": "edit_text", "find": "Hi", "replace": "Hello"}], by="user")
        announce(made.artifact_id, doc, result, by="user", client_id="tab-1")
        frames.append(frame(await asyncio.wait_for(anext(stream), 2)))
        store.mutate_doc(made.folder, lambda d: apply_ops(d, [{"op": "update", "id": lead, "set": {"text": "Remote."}}],
                                                          by="agent"), by="agent")
        frames.append(frame(await asyncio.wait_for(anext(stream), 2)))
        await stream.aclose()
        return frames

    (ready, r), (announced, a), (seen, f) = asyncio.run(scenario())
    assert (ready, r) == ("doc.ready", {"rev": 1})
    assert announced == "doc.updated" and a["rev"] == 2 and a["clientId"] == "tab-1"
    assert seen == "doc.updated" and f["rev"] == 3 and f["by"] == "agent"
    assert f["touched"] == {made.doc["tabs"][0]["id"]: [lead]}
    assert f["doc"]["tabs"][0]["blocks"][0]["text"] == "Remote."


# ---------------------------------------------------------------- conversation stream access

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _conversation_db():
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine

    import cowork.models.message_event  # noqa: F401 — resolve mappers
    from cowork.models.project import Project

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as seed:
        seed.add(Project(name="a-proj", path="/tmp/a", org_id=ORG_A))
        seed.commit()
    return engine


def test_only_the_owner_can_follow_a_conversations_docs():
    """Events carry doc titles and cards: another member, or another org,
    must not be able to watch them by knowing the conversation id."""
    from uuid import uuid4

    from sqlmodel import Session, select

    from cowork.api.v1.endpoints.docs import _may_follow_conversation
    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
    from cowork.models.project import Project
    from cowork.services.conversations import ConversationService

    engine = _conversation_db()
    alice = TenantScope(org_mode=True, org_id=ORG_A, user_id="alice")
    with Session(engine) as s:
        project_id = s.exec(select(Project)).one().id
        conv = ConversationService(ScopedSession(s, alice)).create_conversation(topic="t", project_id=project_id)
        conv_id = conv.id

    def may(scope, cid):
        with Session(engine) as s:
            return _may_follow_conversation(ScopedSession(s, scope), s, cid)

    assert may(alice, conv_id)
    assert not may(TenantScope(org_mode=True, org_id=ORG_A, user_id="bob"), conv_id)
    assert not may(TenantScope(org_mode=True, org_id=ORG_B, user_id="alice"), conv_id)
    # A new chat's id, allocated by the composer before its first turn stores it.
    assert may(alice, uuid4())
    assert may(LOCAL_SCOPE, conv_id)


def test_the_conversation_stream_refuses_a_placeholder_id(client):
    res = client.get("/api/v1/docs/events", params={"conversation_id": "tmp-123"})
    assert res.status_code == 422


# ---------------------------------------------------------------- Word and PDF exports


def _docx_paragraphs(content: bytes) -> list[tuple[str, str]]:
    import io

    from docx import Document

    return [(p.style.name, p.text) for p in Document(io.BytesIO(content)).paragraphs if p.text.strip()]


def _pdf_text(content: bytes) -> str:
    import io

    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)


def test_a_callout_after_a_heading_is_its_own_paragraph_in_word(client):
    res = client.post("/api/v1/docs/", json={
        "projectId": str(GENERAL_PROJECT_ID), "title": "Risks",
        "markdown": "## Risk\n\n> [!WARNING] Pipeline dried up.\n> Owner: Priya.\n\nTail.",
    })
    paras = _docx_paragraphs(client.get(_url(res.json(), "/export?format=docx")).content)
    assert ("Heading 2", "Risk") in paras
    assert any(style != "Heading 2" and "Pipeline dried up." in text for style, text in paras)


def test_the_title_is_text_in_word_and_pdf(client):
    res = client.post("/api/v1/docs/", json={
        "projectId": str(GENERAL_PROJECT_ID), "title": "R&D <draft> plan", "markdown": "Body.",
    })
    docx = client.get(_url(res.json(), "/export?format=docx")).content
    pdf = client.get(_url(res.json(), "/export?format=pdf")).content
    assert _docx_paragraphs(docx)[0][1] == "R&D <draft> plan"
    assert "R&D <draft> plan" in _pdf_text(pdf)


def test_an_export_never_reads_a_local_file_an_image_points_at(client, tmp_path):
    """A doc already on disk with a local image path (written before paths were
    refused, or by hand) must not copy that file into a Word or PDF export."""
    import io
    import json
    import zipfile

    secret = tmp_path / "secret.png"
    secret.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d4944415478da63f8cfc0f01f0005000201a0d5e1b70000000049454e44ae426082"))
    res = client.post("/api/v1/docs/", json={"projectId": str(GENERAL_PROJECT_ID), "title": "Pics", "markdown": "Intro."})
    body = res.json()
    folder = next(p for p in _artifact_dirs(client) if (p / "doc.cdoc.json").is_file()
                  and json.loads((p / "doc.cdoc.json").read_text())["id"] == body["artifactId"])
    doc = json.loads((folder / "doc.cdoc.json").read_text())
    doc["tabs"][0]["blocks"].append({"id": "b_localimg", "type": "image", "v": 1, "by": "user",
                                     "src": str(secret), "alt": "A local scan"})
    (folder / "doc.cdoc.json").write_text(json.dumps(doc))

    docx = client.get(_url(body, "/export?format=docx")).content
    assert not [n for n in zipfile.ZipFile(io.BytesIO(docx)).namelist() if n.startswith("word/media/")]
    assert any("A local scan" in text for _, text in _docx_paragraphs(docx))
    pdf = client.get(_url(body, "/export?format=pdf")).content
    assert b"/Subtype /Image" not in pdf and b"/Subtype/Image" not in pdf


def _artifact_dirs(client):
    import os
    from pathlib import Path

    return [f.parent for f in Path(os.environ["COWORK_HOME"]).rglob("doc.cdoc.json")]
