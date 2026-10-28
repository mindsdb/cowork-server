"""Importing picked SharePoint files into a project through the real route.

Graph and the SharePoint download host are answered by a fake at the HTTP
transport; the project is a real folder with a real database row. Files land
together with their names, or the project is left as it was.
"""
from __future__ import annotations

import uuid

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.api.v1.endpoints import project_files
from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.principal import Principal, get_principal
from cowork.services.connectors import sharepoint as sharepoint_service
from cowork.services.connectors.persist import persist_connection

SECRET = "sp-import-secret-2b9f"
TENANT = "contoso.onmicrosoft.com"
CLIENT_ID = "11111111-2222-3333-4444-555555555555"
SITE = "https://contoso.sharepoint.com/sites/team"
OTHER_SITE = "https://contoso.sharepoint.com/sites/finance"
SITE_ID = "contoso.sharepoint.com,aaaa-1111,bbbb-2222"
DRIVE_ID = "b!Lib_1"
SITE_PATH = "/v1.0/sites/contoso.sharepoint.com:/sites/team"
DRIVES_PATH = f"/v1.0/sites/{SITE_ID}/drives"
ITEMS_PATH = f"{DRIVES_PATH}/{DRIVE_ID}/items"
DOWNLOAD_PATH = "/sites/team/_layouts/15/download.aspx"
ORG_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_USER = "11111111-1111-4111-8111-111111111111"
STORED = {"tenant_id": TENANT, "client_id": CLIENT_ID, "client_secret": SECRET, "site_urls": SITE}


class SharePoint:
    """A fake Entra, Graph and download host serving one library of picked items."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.items: dict[str, dict] = {}
        self.children: dict[str, list[dict]] = {}
        self.bytes: dict[str, bytes] = {}
        self.fail_download: set[str] = set()
        self.stale_second_page: set[str] = set()
        self.on_content = None

    def file(self, item_id: str, name: str, data: bytes, size: int | None = None) -> dict:
        entry = {"id": item_id, "name": name, "size": len(data) if size is None else size, "file": {}}
        self.items[item_id] = entry
        self.bytes[item_id] = data
        return entry

    def folder(self, item_id: str, name: str, children: list[dict]) -> dict:
        entry = {"id": item_id, "name": name, "folder": {"childCount": len(children)}}
        self.items[item_id] = entry
        self.children[item_id] = children
        return entry

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "import-token", "expires_in": 3600})
        if request.url.host == "contoso.sharepoint.com" and path == DOWNLOAD_PATH:
            item_id = request.url.params["UniqueId"]
            if item_id in self.fail_download:
                return httpx.Response(500)
            return httpx.Response(200, content=self.bytes[item_id])
        if path == SITE_PATH:
            return httpx.Response(200, json={"id": SITE_ID})
        if path == DRIVES_PATH:
            return httpx.Response(200, json={"value": [{"id": DRIVE_ID, "name": "Documents"}]})
        if path.startswith(f"{ITEMS_PATH}/"):
            item_id, _, rest = path[len(ITEMS_PATH) + 1:].partition("/")
            if rest == "children":
                if item_id in self.stale_second_page:
                    if request.url.params.get("$skiptoken") == "p2":
                        return httpx.Response(400, json={"error": {"code": "invalidRequest"}})
                    next_link = f"https://graph.microsoft.com{ITEMS_PATH}/{item_id}/children?$skiptoken=p2"
                    return httpx.Response(200, json={"value": self.children[item_id], "@odata.nextLink": next_link})
                return httpx.Response(200, json={"value": self.children.get(item_id, [])})
            if rest == "content" and item_id in self.bytes:
                if self.on_content is not None:
                    self.on_content(item_id)
                location = f"https://contoso.sharepoint.com{DOWNLOAD_PATH}?UniqueId={item_id}&tempauth=tempauth-{item_id}"
                return httpx.Response(302, headers={"Location": location})
            if rest == "" and item_id in self.items:
                return httpx.Response(200, json=self.items[item_id])
        return httpx.Response(404, json={"error": {"code": "itemNotFound"}})

    def content_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("/content")]


class FakeTime:
    def __init__(self) -> None:
        self.now = 1000.0

    def clock(self) -> float:
        return self.now


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


@pytest.fixture
def microsoft(monkeypatch, sharepoint_stores, fake_time):
    fake = SharePoint()
    monkeypatch.setattr(
        project_files,
        "SharePointClient",
        lambda key: sharepoint_service.SharePointClient(
            key, transport=httpx.MockTransport(fake.handler), clock=fake_time.clock, sleep=lambda _s: None
        ),
    )
    persist_connection("sharepoint", None, "corp", dict(STORED), scope=LOCAL_SCOPE)
    return fake


@pytest.fixture
def project(sharepoint_stores):
    """A real project row and folder; returns (name, id, folder)."""
    name = f"import-{uuid.uuid4().hex[:8]}"
    folder = sharepoint_stores / "projects" / name
    folder.mkdir(parents=True)
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        row = Project(id=uuid.uuid4(), name=name, path=str(folder), org_id=None)
        session.add(row)
        session.commit()
        project_id = row.id
    yield name, project_id, folder
    with Session(engine) as session:
        stored = session.get(Project, project_id)
        if stored is not None:
            session.delete(stored)
            session.commit()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


def _import(project_name: str, item_ids: list[str], folder: str = "", site: str = SITE, **overrides):
    body = {"connection": "corp", "site": site, "drive_id": DRIVE_ID, "folder": folder, "item_ids": item_ids}
    body.update(overrides)
    return _client().post(f"/api/v1/projects/{project_name}/files/import/sharepoint", json=body)


def _tree(base) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).as_posix()
        found[relative] = b"<dir>" if path.is_dir() else path.read_bytes()
    return found


def test_a_file_and_a_nested_folder_land_with_their_names_and_bytes(microsoft, project):
    name, _id, folder = project
    microsoft.folder(
        "01R",
        "Reports",
        [
            microsoft.file("01a", "q3.xlsx", b"xlsx bytes"),
            microsoft.folder("01O", "Old", [microsoft.file("01b", "q2.docx", b"docx bytes")]),
            microsoft.file("01d", ".DS_Store", b"mac"),
        ],
    )
    microsoft.file("01t", "notes.txt", b"notes")

    response = _import(name, ["01R", "01t"])

    assert response.status_code == 200, response.text
    assert response.json() == {
        "imported": [
            {"path": "Reports/q3.xlsx", "size": 10},
            {"path": "Reports/Old/q2.docx", "size": 10},
            {"path": "notes.txt", "size": 5},
        ],
        "skipped": [{"path": "Reports/.DS_Store", "reason": "its name starts with a dot"}],
    }
    assert _tree(folder) == {
        "Reports": b"<dir>",
        "Reports/Old": b"<dir>",
        "Reports/Old/q2.docx": b"docx bytes",
        "Reports/q3.xlsx": b"xlsx bytes",
        "notes.txt": b"notes",
    }
    for request in microsoft.requests:
        if request.url.host == "contoso.sharepoint.com":
            assert "authorization" not in request.headers


def test_excel_and_word_files_land_byte_identical_and_are_listed(microsoft, project):
    name, _id, folder = project
    xlsx = b"PK\x03\x04" + bytes(range(256)) * 4
    docx = b"PK\x03\x04" + bytes(reversed(range(256))) * 3
    microsoft.file("01x", "test-sales.xlsx", xlsx)
    microsoft.file("01w", "test-notes.docx", docx)

    response = _import(name, ["01x", "01w"], folder="Imported")
    listing = _client().get(f"/api/v1/projects/{name}/files")

    assert response.status_code == 200, response.text
    assert (folder / "Imported" / "test-sales.xlsx").read_bytes() == xlsx
    assert (folder / "Imported" / "test-notes.docx").read_bytes() == docx
    assert listing.status_code == 200, listing.text
    listed = {entry["path"] for entry in listing.json()["files"]}
    assert {"Imported/test-sales.xlsx", "Imported/test-notes.docx"} <= listed


def test_a_download_failing_on_the_third_of_five_files_leaves_the_project_as_it_was(microsoft, project):
    name, _id, folder = project
    (folder / "b.txt").write_bytes(b"old b")
    before = _tree(folder)
    for n, letter in enumerate("abcde"):
        microsoft.file(f"01{letter}", f"{letter}.txt", f"new {letter}".encode())
    microsoft.fail_download.add("01c")

    response = _import(name, [f"01{letter}" for letter in "abcde"])

    assert response.status_code == 502
    assert "tempauth" not in response.text
    assert _tree(folder) == before


def test_over_100_files_and_over_1_gb_are_refused_by_name_before_any_download(microsoft, project):
    name, _id, folder = project
    microsoft.folder("01M", "Many", [microsoft.file(f"01m{n}", f"m{n}.txt", b"x") for n in range(101)])
    microsoft.file("01h1", "half1.bin", b"", size=2**29)
    microsoft.file("01h2", "half2.bin", b"", size=2**29 + 1)

    too_many = _import(name, ["01M"])
    too_large = _import(name, ["01h1", "01h2"])

    assert too_many.status_code == 413
    assert too_large.status_code == 413
    assert "100-file limit" in too_many.json()["detail"]
    assert "1 GB limit" in too_large.json()["detail"]
    assert too_many.json()["detail"] != too_large.json()["detail"]
    assert microsoft.content_requests() == []
    assert _tree(folder) == {}


def test_the_import_deadline_answers_504_and_rolls_back(microsoft, project, fake_time):
    name, _id, folder = project
    microsoft.file("01a", "a.txt", b"a")
    microsoft.file("01b", "b.txt", b"b")

    def late(item_id: str) -> None:
        if item_id == "01b":
            fake_time.now += sharepoint_service.IMPORT_SECONDS

    microsoft.on_content = late

    response = _import(name, ["01a", "01b"])

    assert response.status_code == 504
    assert "15 minutes" in response.json()["detail"]
    assert _tree(folder) == {}


@pytest.mark.parametrize("target", ["conversations/x", ".anton", "skills/mine", "Skills", ".x", "a/../b"])
def test_reserved_and_invalid_folders_are_refused(microsoft, project, target):
    name, _id, folder = project
    microsoft.file("01a", "a.txt", b"a")

    response = _import(name, ["01a"], folder=target)

    assert response.status_code == 400
    assert microsoft.requests == []
    assert _tree(folder) == {}


def test_a_picked_skills_folder_cannot_land_at_the_project_root(microsoft, project):
    name, _id, folder = project
    microsoft.folder("01S", "Skills", [microsoft.file("01a", "a.md", b"a")])

    response = _import(name, ["01S"])

    assert response.status_code == 400
    assert "reserved" in response.json()["detail"]
    assert microsoft.content_requests() == []


def test_a_second_import_into_the_same_project_is_refused_while_one_runs(microsoft, project):
    name, project_id, _folder = project
    microsoft.file("01a", "a.txt", b"a")

    with project_files._project_import_lock(project_id):
        response = _import(name, ["01a"])

    assert response.status_code == 409
    assert "still running" in response.json()["detail"]


def test_a_failed_import_releases_the_project_lock(microsoft, project):
    name, _id, folder = project
    microsoft.file("01a", "a.txt", b"a")
    microsoft.fail_download.add("01a")

    failed = _import(name, ["01a"])
    microsoft.fail_download.clear()
    retried = _import(name, ["01a"])

    assert failed.status_code == 502
    assert retried.status_code == 200, retried.text
    assert (folder / "a.txt").read_bytes() == b"a"


def test_the_database_session_is_closed_before_the_first_download(microsoft, project, monkeypatch):
    name, _id, _folder = project
    microsoft.file("01a", "a.txt", b"a")
    events: list[str] = []
    real_close = ScopedSession.close

    def recording_close(self):
        events.append("session closed")
        real_close(self)

    monkeypatch.setattr(ScopedSession, "close", recording_close)
    microsoft.on_content = lambda _item: events.append("download")

    response = _import(name, ["01a"])

    assert response.status_code == 200, response.text
    assert events.index("session closed") < events.index("download")


def test_a_site_the_connection_does_not_name_is_refused_before_any_graph_call(microsoft, project):
    name, _id, _folder = project

    response = _import(name, ["01a"], site=OTHER_SITE)

    assert response.status_code == 403
    assert microsoft.requests == []


@pytest.mark.parametrize(
    "overrides",
    [{"item_ids": []}, {"item_ids": "01a"}, {"item_ids": [1]}, {"folder": None}, {"refresh_token": "x"}],
)
def test_a_malformed_body_is_refused_without_echoing_it(microsoft, project, overrides):
    name, _id, _folder = project
    body = {"connection": "corp", "site": SITE, "drive_id": DRIVE_ID, "folder": "", "item_ids": ["01a"], **overrides}

    response = _client().post(f"/api/v1/projects/{name}/files/import/sharepoint", json=body)

    assert response.status_code == 422
    assert "refresh_token" not in response.text
    assert microsoft.requests == []


def test_org_mode_refuses_an_authenticated_import(microsoft, project, monkeypatch):
    name, _id, _folder = project
    microsoft.file("01a", "a.txt", b"a")
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    app = FastAPI()
    app.include_router(api_router)
    # A verified member, so the router's org-mode authentication passes and
    # only the route's own desktop-only guard can refuse.
    app.dependency_overrides[get_principal] = lambda: Principal(user_id=ORG_USER, org_id=ORG_ID)
    body = {"connection": "corp", "site": SITE, "drive_id": DRIVE_ID, "folder": "", "item_ids": ["01a"]}

    response = TestClient(app).post(f"/api/v1/projects/{name}/files/import/sharepoint", json=body)

    assert response.status_code == 403
    assert response.json()["detail"] == "not available in org deployments"
    assert microsoft.requests == []


def test_no_download_link_or_secret_reaches_a_response(microsoft, project):
    name, _id, _folder = project
    microsoft.file("01a", "a.txt", b"a")

    response = _import(name, ["01a"])

    assert response.status_code == 200, response.text
    assert "tempauth" not in response.text
    assert "download.aspx" not in response.text
    assert SECRET not in response.text


def test_every_failure_kind_maps_to_an_import_status():
    for kind in sharepoint_service.SharePointErrorKind:
        error = project_files._sharepoint_import_error(sharepoint_service.SharePointError(kind))
        assert 400 <= error.status_code < 600
    writer_errors = {
        project_files._ImportConflict,
        project_files._ImportFileInUse,
        project_files._ImportDiskError,
        project_files._ImportRestoreFailed,
    }
    assert set(project_files._IMPORT_WRITE_STATUS) == writer_errors


def test_a_folder_that_changes_while_it_is_read_asks_to_retry_the_import(microsoft, project):
    name, _id, folder = project
    microsoft.folder("01R", "Reports", [microsoft.file("01a", "q3.xlsx", b"xlsx bytes")])
    microsoft.stale_second_page.add("01R")

    response = _import(name, ["01R"])

    assert response.status_code == 400
    assert response.json()["detail"] == "The folder changed while it was read. Try the import again."
    assert _tree(folder) == {}
    assert microsoft.content_requests() == []
