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


FAKE_PDF = b"%PDF-1.7\n% a stand-in for the render service's PDF\n"


class _RenderService:
    """The render service's client, answering with FAKE_PDF and keeping each request."""

    def __init__(self):
        self.requests = []

    async def render(self, *, request):
        self.requests.append(request)
        return FAKE_PDF


@pytest.fixture
def render_service(monkeypatch):
    """PDF export on, as a deployment that configures a render service has it."""
    from cowork.schemas.artifact_export import ExportPdfSettings
    from cowork.services import artifact_export

    service = _RenderService()
    monkeypatch.setattr(artifact_export, "export_pdf_settings",
                        lambda: ExportPdfSettings(engine="render", render_url="http://render.test"))
    monkeypatch.setattr(artifact_export, "_pdf_client", lambda: service)
    return service


@pytest.mark.parametrize("fmt,marker", [
    ("md", b"| Role | Status |"),
    ("html", b"<table>"),
    ("docx", b"PK"),
])
def test_export(client, doc, fmt, marker):
    res = client.get(_url(doc, f"/export?format={fmt}"))
    assert res.status_code == 200, res.text
    assert marker in res.content
    assert "Hiring-plan." + fmt in res.headers["content-disposition"]


def test_without_a_render_service_a_pdf_is_refused_and_not_offered(client, doc):
    res = client.get(_url(doc, "/export?format=pdf"))
    assert res.status_code == 422 and "PDF export isn't set up" in res.json()["detail"]
    assert client.get(_url(doc)).json()["capabilities"]["exportFormats"] == ["docx", "md", "html"]


def test_a_pdf_is_the_render_services_made_from_the_inlined_doc(client, doc, render_service):
    assert client.get(_url(doc)).json()["capabilities"]["exportFormats"] == ["docx", "pdf", "md", "html"]
    res = client.get(_url(doc, "/export?format=pdf"))
    assert res.status_code == 200, res.text
    assert res.content == FAKE_PDF and "Hiring-plan.pdf" in res.headers["content-disposition"]
    [sent] = render_service.requests
    assert "<table>" in sent.html and "<h1>Hiring plan</h1>" in sent.html


def test_a_title_in_any_script_exports_under_its_own_name(client):
    from urllib.parse import quote

    res = client.post("/api/v1/docs/", json={"projectId": str(GENERAL_PROJECT_ID), "title": "計画 Q4", "markdown": "Body."})
    exported = client.get(_url(res.json(), "/export?format=md"))
    assert exported.status_code == 200, exported.text
    assert f"filename*=UTF-8''{quote('計画-Q4.md', safe='')}" in exported.headers["content-disposition"]


@pytest.mark.parametrize("fmt", ["pdf", "docx"])
def test_while_org_export_is_off_an_organization_refuses_word_and_pdf(fmt, render_service):
    """As an artifact's export: a conversion waits for a process of its own on a shared replica."""
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from cowork.api.v1.endpoints.docs import export_doc
    from cowork.docs.export import offered_formats

    scope = SimpleNamespace(org_mode=True)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(export_doc("local", "0" * 32, scope, format=fmt))
    assert refused.value.status_code == 403
    assert offered_formats(org_mode=True) == ["md", "html"]


def test_a_card_that_fails_is_logged_as_a_warning(monkeypatch, caplog, tmp_path):
    from types import SimpleNamespace

    from cowork.api.v1.endpoints import docs

    def broken_card(*_args, **_kwargs):
        raise NameError("name 'oops' is not defined")

    monkeypatch.setattr(docs, "card_for_folder", broken_card)
    source = SimpleNamespace(project_id=None, project_name="")
    with caplog.at_level("DEBUG", logger=docs.logger.name):
        assert docs._card(None, source, tmp_path) is None
    (record,) = [r for r in caplog.records if "no card for doc" in r.getMessage()]
    assert record.levelname == "WARNING" and record.exc_info


def test_a_non_doc_artifact_is_not_found(client):
    res = client.get(f"/api/v1/docs/{GENERAL_PROJECT_ID}/0f9e8d7c-6b5a-4938-8271-605f4e3d2c1b")
    assert res.status_code == 404



def test_a_doc_can_be_started_in_a_project_kept_in_a_folder_the_person_chose(client, monkeypatch, tmp_path):
    from cowork.api.v1.endpoints import docs as endpoint
    from cowork.services.artifacts import ProjectArtifacts

    base = tmp_path / "notes" / ".anton" / "artifacts"
    base.mkdir(parents=True)
    chosen = ProjectArtifacts(base=base, project_id=None, project_name="notes", external=True)
    monkeypatch.setattr(endpoint, "_project_plan", lambda session, **kw: None)
    monkeypatch.setattr(endpoint, "readable_sources", lambda plan: [chosen])
    res = client.post("/api/v1/docs/", json={"projectId": str(GENERAL_PROJECT_ID), "title": "Notes"})
    assert res.status_code == 200, res.text
    assert [p.name for p in base.iterdir() if (p / "doc.cdoc.json").is_file()] == [res.json()["slug"]]


def test_a_doc_file_swapped_for_a_link_to_another_doc_is_not_served(client, doc, tmp_path):
    """The agent's code can write the doc's folder: a link planted at its file must not
    hand this route another doc, from this project or any other."""
    from cowork.docs.service import create_doc

    from cowork.common.paths import cowork_home
    from cowork.docs.store import find_doc_folder

    other = create_doc(tmp_path / "elsewhere", title="Someone else's", markdown="Their numbers.", by="agent")
    # Found on disk, as the agent's code would find it.
    folder = find_doc_folder(cowork_home() / "projects" / "general" / ".anton" / "artifacts", doc["artifactId"])
    (folder / "doc.cdoc.json").unlink()
    (folder / "doc.cdoc.json").symlink_to(other.folder / "doc.cdoc.json")
    res = client.get(_url(doc))
    assert res.status_code != 200 and "Their numbers" not in res.text

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
    # Every route that names the doc's folder (the workspace's access route, live sync,
    # access reads) shares the same page, never the JSON the folder names as its primary.
    target, published_dir, key, fullstack = _resolve_publish_target(page.parent, container_dirs=[page.parent.parent])
    assert (target.resolve(), published_dir.resolve(), key, fullstack) == (page.resolve(), page.parent.resolve(), "doc.html", False)


def _wait_for(pred, timeout=3.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_a_shared_doc_republishes_itself_after_edits_settle(client, doc, monkeypatch):
    """Like a shared Google Doc: the link shows the latest, with no Update step."""
    import json
    from pathlib import Path

    from cowork.docs import republish

    calls = []
    monkeypatch.setattr(republish, "update_artifact", lambda path, *, scope: calls.append(path) or {})
    monkeypatch.setattr(republish, "desktop_publish_credential", lambda: ("key", "https://publish.test"))
    monkeypatch.setattr(republish._republishes, "delay", 0.2)

    page = Path(client.get(_url(doc)).json()["card"]["htmlPath"])
    lead = doc["doc"]["tabs"][0]["blocks"][0]
    edit = lambda text: client.post(_url(doc, "/ops"), json={"ops": [  # noqa: E731
        {"op": "update", "id": lead["id"], "set": {"text": text}}]})

    edit("Not shared yet.")
    assert not republish.pending() and not _wait_for(lambda: calls, timeout=0.4)

    (page.parent / ".published.json").write_text(json.dumps(
        {"doc.html": {"url": "https://view.example/abc", "report_id": "abc", "published": True}}))
    for text in ("One", "Two", "Three"):  # a burst of typing is one republish
        assert edit(text).status_code == 200
    assert _wait_for(lambda: calls)
    assert _wait_for(lambda: not republish.pending())
    assert calls == [str(page)]

    (page.parent / ".published.json").write_text(json.dumps(
        {"doc.html": {"url": "https://view.example/abc", "report_id": "abc", "published": False}}))
    edit("Stopped sharing.")
    assert not _wait_for(lambda: len(calls) > 1, timeout=0.5)


def test_a_republish_waiting_when_sharing_stops_does_not_put_the_page_back(tmp_path, monkeypatch):
    """Unsharing in the seconds after an edit: the republish that edit scheduled must stand down."""
    import json

    from cowork.docs import republish

    calls = []
    monkeypatch.setattr(republish, "update_artifact", lambda path, *, scope: calls.append(path) or {})
    monkeypatch.setattr(republish, "desktop_publish_credential", lambda: ("key", "https://publish.test"))
    monkeypatch.setattr(republish._republishes, "delay", 0.2)
    published = tmp_path / ".published.json"
    published.write_text(json.dumps({"doc.html": {"url": "https://view.example/abc", "published": True}}))
    assert republish.schedule(tmp_path) is True
    published.write_text(json.dumps({"doc.html": {"url": "https://view.example/abc", "published": False}}))
    assert _wait_for(lambda: not republish.pending())
    assert not _wait_for(lambda: calls, timeout=0.3)


def test_org_mode_leaves_a_shared_doc_to_autopublish(tmp_path):
    import json

    from cowork.db.scoped import TenantScope
    from cowork.docs import republish, sharing

    (tmp_path / ".published.json").write_text(json.dumps({"doc.html": {"url": "https://x", "published": True}}))
    assert sharing.is_shared(tmp_path)
    assert republish.schedule(tmp_path, scope=TenantScope(org_mode=True, org_id="org")) is False


# ---------------------------------------------------------------- who may see and change what


@pytest.fixture
def inline_units(monkeypatch):
    """Database units and file store calls run inline, for a route called without the app."""
    from cowork.api.v1.endpoints import docs

    async def run_db(fn, *, scope):
        return fn("unit")

    async def run_efs(fn, *, scope, deadline=True):
        return fn()

    monkeypatch.setattr(docs, "run_db", run_db)
    monkeypatch.setattr(docs, "run_efs", run_efs)


def test_a_doc_created_over_http_is_its_creators(monkeypatch, tmp_path, inline_units):
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
    monkeypatch.setattr(docs, "_project_plan", lambda session, **kw: None)
    monkeypatch.setattr(docs, "readable_sources", lambda plan: [source])
    monkeypatch.setattr(docs, "record_artifact_owner", lambda *a, **kw: recorded.append((a, kw)))
    monkeypatch.setattr(docs, "_card", lambda *a: None)
    scope = SimpleNamespace(user_id="alice")
    made = asyncio.run(docs.create_blank_doc(docs._CreateBody(projectId=uuid4(), title="Plan"), scope))
    assert recorded == [(("unit", "p-1", made["slug"], "alice"), {"action": "create"})]


def test_a_reviewer_reads_the_doc_without_the_owners_card(monkeypatch, tmp_path, inline_units):
    """The card carries the share's access: its password and the invited emails."""
    import asyncio
    from types import SimpleNamespace

    from cowork.api.v1.endpoints import docs
    from cowork.docs.service import create_doc
    from cowork.services.artifact_metadata import ArtifactMetadata

    folder = create_doc(tmp_path, title="Plan", markdown="Hi.").folder
    reviewer = {"role": "reviewer", "canPreview": True, "canComment": True, "canEdit": False,
                "canAddressWithAgent": False, "canResolveComments": False, "ownerUnknown": False}

    async def narrowed(scope, capabilities):
        return capabilities

    async def resolved(*_a, **_kw):
        return SimpleNamespace(), folder, ArtifactMetadata(), False

    monkeypatch.setattr(docs, "_resolve", resolved)
    monkeypatch.setattr(docs, "artifact_capabilities", lambda *a: reviewer)
    monkeypatch.setattr(docs, "current_capabilities", narrowed)
    monkeypatch.setattr(docs, "_card", lambda *a: pytest.fail("a reviewer was handed the owner's card"))
    got = asyncio.run(docs.get_doc("local", "0" * 32, SimpleNamespace(org_mode=False)))
    assert got["card"] is None
    assert got["capabilities"] == {**reviewer, "exportFormats": ["docx", "md", "html"]}


def test_only_the_owner_writes_to_a_doc(monkeypatch, tmp_path, inline_units):
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

    async def reviewed(*_a):
        return SimpleNamespace(), folder, {}, False

    monkeypatch.setattr(docs, "require_product_permission", allowed)
    monkeypatch.setattr(docs, "review_artifact_for_request", reviewed)
    body = docs._OpsBody(ops=[{"op": "edit_text", "find": "Hi", "replace": "Bye"}])
    with pytest.raises(HTTPException) as err:
        asyncio.run(docs.edit_doc("local", "0" * 32, body, SimpleNamespace(org_mode=False)))
    assert err.value.status_code == 403
    assert (folder / DOC_FILENAME).read_bytes() == before


def test_a_doc_stream_sends_announced_writes_and_ones_only_the_file_shows(monkeypatch, tmp_path):
    """An open editor hears the hub's announcements, and also a write from another
    process (a remote turn), which changes the file without one."""
    import asyncio
    import json

    from cowork.api.v1.endpoints import docs
    from cowork.db.scoped import LOCAL_SCOPE
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
        stream = docs._sse(Open(), doc_topic(made.folder), initial=made.folder, scope=LOCAL_SCOPE).body_iterator
        frames = [frame(await anext(stream))]
        doc, result = apply(made.folder, [{"op": "edit_text", "find": "Hi", "replace": "Hello"}], by="user")
        announce(made.artifact_id, doc, result, folder=made.folder, by="user", client_id="tab-1")
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


def test_a_doc_whose_files_claim_another_docs_id_never_reaches_that_docs_editor(monkeypatch, tmp_path):
    """The agent's code can write doc.cdoc.json: a doc that claims another doc's id
    (another tenant's, on the same process) must not feed or read that doc's stream."""
    import asyncio
    import json

    from cowork.api.v1.endpoints import docs
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.docs.hub import doc_topic
    from cowork.docs.service import announce, apply, create_doc
    from cowork.docs.store import DOC_FILENAME

    monkeypatch.setattr(docs, "_POLL_S", 0.02)
    victim = create_doc(tmp_path / "victim", title="Theirs", markdown="Their plan.", by="user")
    forger = create_doc(tmp_path / "forger", title="Mine", markdown="Hi.", by="user")
    raw = json.loads((forger.folder / DOC_FILENAME).read_text())
    raw["id"] = victim.doc["id"]
    (forger.folder / DOC_FILENAME).write_text(json.dumps(raw))

    class Open:
        async def is_disconnected(self):
            return False

    async def scenario():
        stream = docs._sse(Open(), doc_topic(victim.folder), initial=victim.folder, scope=LOCAL_SCOPE).body_iterator
        await anext(stream)  # doc.ready
        doc, result = apply(forger.folder, [{"op": "edit_text", "find": "Hi", "replace": "Injected"}], by="user")
        announce(doc["id"], doc, result, folder=forger.folder, by="user")
        try:
            frame = await asyncio.wait_for(anext(stream), 0.3)
        except asyncio.TimeoutError:
            frame = None
        await stream.aclose()
        return frame

    frame = asyncio.run(scenario())
    assert frame is None or "Injected" not in frame


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


def test_only_the_owner_can_follow_a_conversations_docs(monkeypatch):
    """Events carry doc titles and cards: another member, or another org,
    must not be able to watch them by knowing the conversation id."""
    import asyncio
    from uuid import uuid4

    from sqlmodel import Session, select

    from cowork.api.v1.endpoints import docs
    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
    from cowork.models.project import Project
    from cowork.services.conversations import ConversationService

    engine = _conversation_db()
    alice = TenantScope(org_mode=True, org_id=ORG_A, user_id="alice")
    with Session(engine) as s:
        project_id = s.exec(select(Project)).one().id
        conv = ConversationService(ScopedSession(s, alice)).create_conversation(topic="t", project_id=project_id)
        conv_id = conv.id

    async def run_db(fn, *, scope):
        with Session(engine) as s:
            return fn(ScopedSession(s, scope))

    monkeypatch.setattr(docs, "run_db", run_db)

    def may(scope, cid):
        return asyncio.run(docs._may_follow_conversation(scope, cid))

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


def test_a_callout_after_a_heading_is_its_own_paragraph_in_word(client):
    res = client.post("/api/v1/docs/", json={
        "projectId": str(GENERAL_PROJECT_ID), "title": "Risks",
        "markdown": "## Risk\n\n> [!WARNING] Pipeline dried up.\n> Owner: Priya.\n\nTail.",
    })
    paras = _docx_paragraphs(client.get(_url(res.json(), "/export?format=docx")).content)
    assert ("Heading 2", "Risk") in paras
    assert any(style != "Heading 2" and "Pipeline dried up." in text for style, text in paras)


def test_the_title_is_text_in_word_and_pdf(client, render_service):
    res = client.post("/api/v1/docs/", json={
        "projectId": str(GENERAL_PROJECT_ID), "title": "R&D <draft> plan", "markdown": "Body.",
    })
    docx = client.get(_url(res.json(), "/export?format=docx")).content
    assert client.get(_url(res.json(), "/export?format=pdf")).status_code == 200
    assert _docx_paragraphs(docx)[0][1] == "R&D <draft> plan"
    assert "<h1>R&amp;D &lt;draft&gt; plan</h1>" in render_service.requests[0].html


def test_an_export_never_reads_a_local_file_an_image_points_at(client, tmp_path, render_service):
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
    assert client.get(_url(body, "/export?format=pdf")).status_code == 200
    assert "data:image" not in render_service.requests[0].html


def _artifact_dirs(client):
    import os
    from pathlib import Path

    return [f.parent for f in Path(os.environ["COWORK_HOME"]).rglob("doc.cdoc.json")]


_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d4944415478da63f8cfc0f01f0005000201a0d5e1b70000000049454e44ae426082")


def test_a_pasted_image_is_kept_beside_the_doc_and_goes_into_word(client, doc, monkeypatch):
    """Found in QA: a person couldn't add an image at all."""
    res = client.post(_url(doc, "/images"), content=_PNG, headers={"Content-Type": "image/png"})
    assert res.status_code == 200
    src = res.json()["src"]
    assert src.startswith("images/") and src.endswith(".png")
    ops = client.post(_url(doc, "/ops"), json={"ops": [
        {"op": "insert", "position": "end", "blocks": [{"type": "image", "src": src, "alt": "Logo"}]}]})
    assert ops.status_code == 200

    import io
    import zipfile
    docx = client.get(_url(doc, "/export?format=docx")).content
    assert [n for n in zipfile.ZipFile(io.BytesIO(docx)).namelist() if n.startswith("word/media/")]

    assert client.post(_url(doc, "/images"), content=b"<svg/>").status_code == 422
    from cowork.api.v1.endpoints import docs as docs_ep
    monkeypatch.setattr(docs_ep, "MAX_IMAGE_BYTES", 10)
    assert client.post(_url(doc, "/images"), content=_PNG).status_code == 413


def test_an_edit_answers_its_undo_and_posting_it_puts_the_doc_back(client, doc):
    first = client.get(_url(doc)).json()["doc"]
    lead = first["tabs"][0]["blocks"][0]
    res = client.post(_url(doc, "/ops"), json={"ops": [{"op": "delete", "target": [lead["id"]]}]}).json()
    assert res["undo"] and res["undo"][0]["op"] == "insert"
    back = client.post(_url(doc, "/ops"), json={"ops": res["undo"]}).json()
    assert back["doc"]["tabs"][0]["blocks"][0]["id"] == lead["id"]
    assert back["doc"]["tabs"][0]["blocks"][0]["text"] == lead["text"]
    assert back["undo"][0] == {"op": "delete", "tab": first["tabs"][0]["id"], "target": [lead["id"]]}


# ---------------------------------------------------------------- Word and PDF in the conversion process

_CONVERTED = b"<p>what the conversion process made</p>"


def _converted(**_kwargs):
    from cowork.services.export_child import Conversion

    return Conversion(data=_CONVERTED, omitted_count=0, scripts_dropped=False)


@pytest.fixture
def conversions(monkeypatch):
    """The arguments of each conversion an export asks for, answered at once with _CONVERTED."""
    from cowork.services import artifact_export

    calls = []

    def stub(**kwargs):
        calls.append(kwargs)
        return _converted()

    monkeypatch.setattr(artifact_export, "convert_in_child", stub)
    return calls


@pytest.mark.parametrize("fmt", ["docx", "pdf"])
def test_word_and_pdf_convert_the_docs_page_in_the_conversion_process(client, doc, render_service, conversions, fmt):
    from cowork.services.artifact_export import CONVERSION_MEMORY_BYTES, INLINE_BUDGET_BYTES, PDF_INLINE_BUDGET_BYTES

    res = client.get(_url(doc, f"/export?format={fmt}"))

    assert res.status_code == 200, res.text
    [call] = conversions
    request = call["request"]
    assert (request.export_format, request.suffix, request.base_parts, request.title) == (fmt, ".html", (), "Hiring plan")
    assert request.limits.max_total_bytes == (PDF_INLINE_BUDGET_BYTES if fmt == "pdf" else INLINE_BUDGET_BYTES)
    assert call["limits"].memory_bytes == CONVERSION_MEMORY_BYTES
    assert b"<h1>Hiring plan</h1>" in call["source"]
    if fmt == "docx":
        assert res.content == _CONVERTED
    else:
        # The render service gets the page the conversion process inlined, and only that.
        assert [r.html for r in render_service.requests] == [_CONVERTED.decode("utf-8")]
        assert res.content == FAKE_PDF


def test_in_org_mode_a_doc_converts_with_the_org_budget_and_memory(conversions, tmp_path):
    from cowork.common.paths import pinned_dir
    from cowork.docs.export import export_bytes
    from cowork.docs.ops import build_new_doc
    from cowork.services.artifact_export import ORG_CONVERSION_MEMORY_BYTES, ORG_INLINE_BUDGET_BYTES

    with pinned_dir(tmp_path) as pinned:
        assert export_bytes(build_new_doc("Plan"), "docx", folder=pinned, org_mode=True) == _CONVERTED

    [call] = conversions
    assert call["request"].limits.max_total_bytes == ORG_INLINE_BUDGET_BYTES
    assert call["limits"].memory_bytes == ORG_CONVERSION_MEMORY_BYTES


@pytest.mark.usefixtures("child_runs_this_checkout")
def test_a_title_past_the_requests_header_bound_still_exports_to_word(client, doc):
    import io

    from docx import Document

    from cowork.services.docx_writer import MAX_TITLE_LENGTH

    title = "計" * 20_000
    renamed = client.post(_url(doc, "/ops"), json={"ops": [{"op": "rename_doc", "title": title}]})
    assert renamed.status_code == 200, renamed.text

    exported = client.get(_url(doc, "/export?format=docx"))

    assert exported.status_code == 200, exported.text
    assert _docx_paragraphs(exported.content)[0][1] == title
    assert Document(io.BytesIO(exported.content)).core_properties.title == title[:MAX_TITLE_LENGTH]


@pytest.mark.parametrize("second_fmt", ["docx", "pdf"])
def test_a_second_export_while_one_converts_answers_429_at_once(client, doc, render_service, monkeypatch, second_fmt):
    import threading

    from cowork.services import artifact_export

    entered, release = threading.Event(), threading.Event()

    def held(**kwargs):
        entered.set()
        assert release.wait(timeout=10)
        return _converted()

    monkeypatch.setattr(artifact_export, "convert_in_child", held)
    answers = []
    first = threading.Thread(target=lambda: answers.append(client.get(_url(doc, "/export?format=docx")).status_code))
    first.start()
    try:
        assert entered.wait(timeout=10)
        # The first export is released only after this answers, so a 429 here never waited.
        second = client.get(_url(doc, f"/export?format={second_fmt}"))
    finally:
        release.set()
        first.join(timeout=60)

    assert second.status_code == 429, second.text
    assert second.json()["detail"] == "An export is already running; try again in a moment."
    assert answers == [200]
    assert client.get(_url(doc, f"/export?format={second_fmt}")).status_code == 200


def test_a_word_export_while_another_conversion_holds_the_process_answers_429(client, doc):
    """Publishing a Markdown page converts in the same one-at-a-time process."""
    from cowork.services import export_runner

    assert export_runner._child_slot.acquire(blocking=False)
    try:
        res = client.get(_url(doc, "/export?format=docx"))
    finally:
        export_runner._child_slot.release()

    assert res.status_code == 429, res.text
    assert res.json()["detail"] == "An export is already running; try again in a moment."


def test_a_page_past_the_source_bound_answers_413_and_is_never_converted(client, doc, conversions, monkeypatch):
    from cowork.services import artifact_export

    monkeypatch.setattr(artifact_export, "HTML_SOURCE_MAX_BYTES", 100)

    res = client.get(_url(doc, "/export?format=docx"))

    assert res.status_code == 413, res.text
    assert res.json()["detail"] == "This doc is too large to export."
    assert conversions == []


@pytest.mark.usefixtures("child_runs_this_checkout")
@pytest.mark.parametrize("fmt", ["docx", "pdf"])
def test_a_conversion_past_its_limits_answers_413_and_renders_nothing(client, doc, render_service, monkeypatch, fmt):
    from cowork.services import artifact_export

    monkeypatch.setattr(artifact_export, "CONVERSION_WALL_SECONDS", 0.0)

    res = client.get(_url(doc, f"/export?format={fmt}"))

    assert res.status_code == 413, res.text
    assert res.json()["detail"] == "This doc is too large to export."
    assert render_service.requests == []


def test_a_failed_conversion_answers_500(client, doc, monkeypatch):
    from cowork.services import artifact_export, export_runner

    def broken(**_kwargs):
        raise export_runner.ConversionError("ValueError")

    monkeypatch.setattr(artifact_export, "convert_in_child", broken)

    res = client.get(_url(doc, "/export?format=docx"))

    assert res.status_code == 500, res.text
    assert res.json()["detail"] == "Could not export the doc as DOCX"


def test_a_busy_file_store_answers_503_not_a_failed_export(client, doc, monkeypatch):
    from cowork.api.v1.endpoints import docs as docs_ep
    from cowork.db.units import FilesystemBusy

    def busy(*_args, **_kwargs):
        raise FilesystemBusy(limiter="run_efs")

    monkeypatch.setattr(docs_ep, "_converted", busy)

    res = client.get(_url(doc, "/export?format=docx"))

    assert res.status_code == 503, res.text
    assert "Retry-After" in res.headers


# ---------------------------------------------------------------- another replica's writes


class _Open:
    async def is_disconnected(self):
        return False


def _sse_frame(chunk):
    import json

    event, data = chunk.split("\n")[:2]
    return event.removeprefix("event: "), json.loads(data.removeprefix("data: "))


async def _no_frame(stream, wait=0.3):
    import asyncio

    try:
        chunk = await asyncio.wait_for(anext(stream), wait)
    except asyncio.TimeoutError:
        return True
    raise AssertionError(f"unexpected frame: {chunk[:200]}")


def test_a_nudge_for_the_write_on_disk_sends_the_doc_with_the_writers_client_id(monkeypatch, tmp_path):
    """Another replica's write arrives as a nudge: the stream reads the doc and sends
    it like the mtime poll would, plus the writer's clientId, since the doc read is
    exactly that write."""
    import asyncio

    from cowork.api.v1.endpoints import docs
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.docs.hub import DocHub, doc_topic
    from cowork.docs.service import apply, create_doc

    # Only the nudge can deliver a frame within the test's waits.
    monkeypatch.setattr(docs, "_POLL_S", 5.0)
    monkeypatch.setattr(docs, "hub", DocHub())
    made = create_doc(tmp_path, title="Plan", markdown="Hi.", by="user")
    topic = doc_topic(made.folder)
    lead = made.doc["tabs"][0]["blocks"][0]["id"]

    async def scenario():
        stream = docs._sse(_Open(), topic, initial=made.folder, scope=LOCAL_SCOPE).body_iterator
        frames = [_sse_frame(await anext(stream))]
        doc, _ = apply(made.folder, [{"op": "update", "id": lead, "set": {"text": "Agent."}}], by="agent")
        docs.hub.deliver_remote({"type": "doc.updated", "rev": doc["rev"], "clientId": "tab-1",
                                 "origin": "other", "topic": topic})
        frames.append(_sse_frame(await asyncio.wait_for(anext(stream), 3)))
        await _no_frame(stream)
        await stream.aclose()
        return frames

    (ready, _r), (name, f) = asyncio.run(scenario())
    assert ready == "doc.ready"
    assert name == "doc.updated" and f["rev"] == 2 and f["clientId"] == "tab-1"
    assert f["by"] == "agent" and f["touched"] == {made.doc["tabs"][0]["id"]: [lead]}
    assert f["doc"]["tabs"][0]["blocks"][0]["text"] == "Agent."


def test_two_nudges_after_both_writes_send_one_frame_without_the_first_writers_client_id(monkeypatch, tmp_path):
    """The client's own write N went through another replica, then the agent wrote
    N+1. The nudge for N already reads N+1 and sends nothing (with the client's
    clientId the client would skip the agent's write); the nudge for N+1 sends the
    doc with the agent's write's clientId, which is none."""
    import asyncio

    from cowork.api.v1.endpoints import docs
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.docs.hub import DocHub, doc_topic
    from cowork.docs.service import apply, create_doc

    monkeypatch.setattr(docs, "_POLL_S", 5.0)
    monkeypatch.setattr(docs, "hub", DocHub())
    made = create_doc(tmp_path, title="Plan", markdown="Hi.", by="user")
    topic = doc_topic(made.folder)
    lead = made.doc["tabs"][0]["blocks"][0]["id"]

    async def scenario():
        stream = docs._sse(_Open(), topic, initial=made.folder, scope=LOCAL_SCOPE).body_iterator
        await anext(stream)  # doc.ready
        own, _ = apply(made.folder, [{"op": "edit_text", "find": "Hi", "replace": "Hello"}], by="user")
        agent, _ = apply(made.folder, [{"op": "update", "id": lead, "set": {"text": "Agent."}}], by="agent")
        for rev, client_id in ((own["rev"], "tab-1"), (agent["rev"], None)):
            docs.hub.deliver_remote({"type": "doc.updated", "rev": rev, "clientId": client_id,
                                     "origin": "other", "topic": topic})
        frame = _sse_frame(await asyncio.wait_for(anext(stream), 3))
        await _no_frame(stream)
        await stream.aclose()
        return frame

    name, f = asyncio.run(scenario())
    assert name == "doc.updated" and f["rev"] == 3
    assert "clientId" not in f
    assert f["by"] == "agent" and f["doc"]["tabs"][0]["blocks"][0]["text"] == "Agent."


def test_a_nudge_that_reads_a_newer_doc_leaves_it_to_that_writes_nudge(monkeypatch, tmp_path):
    """The agent wrote N, then the client's own write N+1 went through another
    replica. The nudge for N already reads N+1: sent without a clientId, it would
    reach the client before its own write's answer and apply that write twice. So
    nudge N sends nothing, and nudge N+1 sends the doc with the client's clientId."""
    import asyncio

    from cowork.api.v1.endpoints import docs
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.docs.hub import DocHub, doc_topic
    from cowork.docs.service import apply, create_doc

    monkeypatch.setattr(docs, "_POLL_S", 5.0)
    monkeypatch.setattr(docs, "hub", DocHub())
    made = create_doc(tmp_path, title="Plan", markdown="Hi.", by="user")
    topic = doc_topic(made.folder)
    lead = made.doc["tabs"][0]["blocks"][0]["id"]

    async def scenario():
        stream = docs._sse(_Open(), topic, initial=made.folder, scope=LOCAL_SCOPE).body_iterator
        await anext(stream)  # doc.ready
        agent, _ = apply(made.folder, [{"op": "update", "id": lead, "set": {"text": "Agent."}}], by="agent")
        own, _ = apply(made.folder, [{"op": "edit_text", "find": "Agent", "replace": "Mine"}], by="user")
        docs.hub.deliver_remote({"type": "doc.updated", "rev": agent["rev"], "clientId": None,
                                 "origin": "other", "topic": topic})
        # One pending read across both nudges: cancelling it would end the stream.
        following = asyncio.ensure_future(anext(stream))
        done, _ = await asyncio.wait({following}, timeout=0.3)
        assert not done, "the nudge for the agent's write sent the newer doc"
        docs.hub.deliver_remote({"type": "doc.updated", "rev": own["rev"], "clientId": "tab-1",
                                 "origin": "other", "topic": topic})
        frame = _sse_frame(await asyncio.wait_for(following, 3))
        await _no_frame(stream)
        await stream.aclose()
        return frame

    name, f = asyncio.run(scenario())
    assert name == "doc.updated" and f["rev"] == 3 and f["clientId"] == "tab-1"
    assert f["doc"]["tabs"][0]["blocks"][0]["text"] == "Mine."


def test_a_read_older_than_the_nudge_keeps_both_marks(monkeypatch, tmp_path):
    import asyncio

    from cowork.api.v1.endpoints import docs
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.docs.hub import NUDGE
    from cowork.docs.service import create_doc

    made = create_doc(tmp_path, title="Plan", markdown="Hi.", by="user")
    monkeypatch.setattr(docs, "_mtime", lambda folder: 999)
    nudge = {"type": NUDGE, "rev": made.doc["rev"] + 1, "clientId": "tab-1"}
    result = asyncio.run(docs._nudged(made.folder, nudge, last_rev=0, last_mtime=7, pending_rev=-1,
                                      scope=LOCAL_SCOPE))
    assert result == (None, 0, 7, made.doc["rev"] + 1)


def test_after_a_lagging_read_the_mtime_poll_still_sends_the_write(monkeypatch, tmp_path):
    """This host's view of the file can lag the other replica's write: the mtime
    already moved but the read returns the old doc. The nudge then sends nothing
    and leaves last_mtime alone, so the next poll reads the file again and sends
    the write (without a clientId, as the poll always does)."""
    import asyncio

    from cowork.api.v1.endpoints import docs
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.docs.hub import DocHub, doc_topic
    from cowork.docs.service import apply, create_doc
    from cowork.docs.store import load_doc

    monkeypatch.setattr(docs, "_POLL_S", 0.2)
    monkeypatch.setattr(docs, "hub", DocHub())
    made = create_doc(tmp_path, title="Plan", markdown="Hi.", by="user")
    topic = doc_topic(made.folder)
    lead = made.doc["tabs"][0]["blocks"][0]["id"]
    seen = {"mtime": 1, "stale": None}

    def lagging_load(folder):
        stale, seen["stale"] = seen["stale"], None
        return stale if stale is not None else load_doc(folder)

    monkeypatch.setattr(docs, "_mtime", lambda folder: seen["mtime"])
    monkeypatch.setattr(docs, "load_doc", lagging_load)

    async def scenario():
        stream = docs._sse(_Open(), topic, initial=made.folder, scope=LOCAL_SCOPE).body_iterator
        await anext(stream)  # doc.ready at rev 1, last_mtime 1
        old = load_doc(made.folder)
        doc, _ = apply(made.folder, [{"op": "update", "id": lead, "set": {"text": "Agent."}}], by="agent")
        seen.update(mtime=2, stale=old)
        docs.hub.deliver_remote({"type": "doc.updated", "rev": doc["rev"], "clientId": "tab-1",
                                 "origin": "other", "topic": topic})
        frame = _sse_frame(await asyncio.wait_for(anext(stream), 3))
        await stream.aclose()
        return frame

    name, f = asyncio.run(scenario())
    assert name == "doc.updated" and f["rev"] == 2 and "clientId" not in f
    assert f["doc"]["tabs"][0]["blocks"][0]["text"] == "Agent."


def test_a_lagging_read_is_read_again_even_when_the_mtime_never_moves(monkeypatch, tmp_path):
    """This host can see the old doc under an mtime that already shows the write.
    The nudge leaves its rev pending, so the poll rereads the file until it shows
    that rev, though the mtime never changes again."""
    import asyncio

    from cowork.api.v1.endpoints import docs
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.docs.hub import DocHub, doc_topic
    from cowork.docs.service import apply, create_doc
    from cowork.docs.store import load_doc

    monkeypatch.setattr(docs, "_POLL_S", 0.2)
    monkeypatch.setattr(docs, "hub", DocHub())
    made = create_doc(tmp_path, title="Plan", markdown="Hi.", by="user")
    topic = doc_topic(made.folder)
    lead = made.doc["tabs"][0]["blocks"][0]["id"]
    stale = {"doc": None}

    def lagging_load(folder):
        doc, stale["doc"] = stale["doc"], None
        return doc if doc is not None else load_doc(folder)

    monkeypatch.setattr(docs, "_mtime", lambda folder: 1)
    monkeypatch.setattr(docs, "load_doc", lagging_load)

    async def scenario():
        stream = docs._sse(_Open(), topic, initial=made.folder, scope=LOCAL_SCOPE).body_iterator
        await anext(stream)  # doc.ready at rev 1, last_mtime 1
        old = load_doc(made.folder)
        doc, _ = apply(made.folder, [{"op": "update", "id": lead, "set": {"text": "Agent."}}], by="agent")
        stale["doc"] = old
        docs.hub.deliver_remote({"type": "doc.updated", "rev": doc["rev"], "clientId": "tab-1",
                                 "origin": "other", "topic": topic})
        frame = _sse_frame(await asyncio.wait_for(anext(stream), 3))
        await stream.aclose()
        return frame

    name, f = asyncio.run(scenario())
    assert name == "doc.updated" and f["rev"] == 2 and "clientId" not in f
