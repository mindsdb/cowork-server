"""PDFs in draft previews: the pdf.js viewer's files, the viewer page for a
frame that navigates to a PDF, CORS for the desktop frame, and the address the
shim gets."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, get_tenant_scope
from cowork.draft_host import app as draft_app_module
from cowork.draft_host.app import create_draft_app
from cowork.services import pdf_viewer
from cowork.services.preview_html import prepare_preview_html
from tests.support.pinned_responses import served_body

from draft_host_support import (
    DOC,
    DRAFT_HOST,
    FETCH,
    draft_env,  # noqa: F401  (fixture)
    draft_token,
    fake_review,
    make_artifact,
    reviewed_and_read,
)

PDF = "%PDF-1.4 a stand-in for a PDF"
FILES = {"index.html": DOC, "docs/report.pdf": PDF, "assets/app.js": "console.log('draft');"}
ARTIFACT_HEX = "0123456789abcdef0123456789abcdef"
DESKTOP_ASSETS = f"/api/v1/artifacts/drafts/local/{ARTIFACT_HEX}/_anton_pdf/"


# --------------------------------------------------------------- draft host

@pytest.fixture
def host(draft_env, tmp_path, monkeypatch):  # noqa: F811
    source, folder, metadata = make_artifact(tmp_path, files=FILES)
    monkeypatch.setattr(
        draft_app_module, "review_artifact_and_read", reviewed_and_read(fake_review(source, folder, metadata, {}))
    )
    return TestClient(create_draft_app(), base_url=f"https://{DRAFT_HOST}")


def _host_get(client, rel, dest):
    token = draft_token()
    return token, client.get(f"/_t/{token}/{rel}", headers={**FETCH, "Sec-Fetch-Dest": dest})


def test_the_draft_host_serves_the_viewer_files(host):
    _token, response = _host_get(host, "_anton_pdf/pdf.min.mjs", "script")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.content == pdf_viewer.asset("pdf.min.mjs")[0]
    assert response.headers["access-control-allow-origin"] == "*"
    # The address carries the draft token, so nothing on this host is cached.
    assert response.headers["cache-control"] == "private, no-store"


@pytest.mark.parametrize("name", ["LICENSE", "fetch_pdfjs.py", "__init__.py", "missing.mjs"])
def test_the_draft_host_serves_nothing_else_from_the_viewer_folder(host, name):
    _token, response = _host_get(host, f"_anton_pdf/{name}", "script")
    assert response.status_code == 404


@pytest.mark.parametrize("dest", ["iframe", "frame", "document"])
def test_a_navigation_to_a_pdf_gets_the_viewer_page_on_the_draft_host(host, dest):
    _token, response = _host_get(host, "docs/report.pdf", dest)
    assert response.status_code == 200
    assert response.text == prepare_preview_html(pdf_viewer.viewer_page("docs/report.pdf", draft=True), comments=False)
    assert '<html lang="en" data-anton-pdf-draft>' in response.text
    assert 'href="../_anton_pdf/viewer.css"' in response.text
    assert response.headers["vary"] == "Sec-Fetch-Dest"


def test_a_fetch_of_a_pdf_gets_the_bytes_on_the_draft_host(host):
    _token, response = _host_get(host, "docs/report.pdf", "empty")
    assert response.status_code == 200
    assert response.text == PDF
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["vary"] == "Sec-Fetch-Dest"


def test_the_draft_host_gives_the_shim_its_viewer_address(host):
    token, response = _host_get(host, "index.html", "iframe")
    assert response.text == prepare_preview_html(DOC, comments=False, pdf_assets=f"/_t/{token}/_anton_pdf/")
    assert f'var PDF_ASSETS = "/_t/{token}/_anton_pdf/";' in response.text


# --------------------------------------------------------------- /drafts route

@pytest.fixture
def desktop(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _serve_the_artifact(monkeypatch, tmp_path):
    """The /drafts lookup finds the test artifact and lets the caller read it."""
    source, folder, metadata = make_artifact(tmp_path, files=FILES)
    monkeypatch.setattr(
        workspace_ep, "review_artifact_and_read",
        reviewed_and_read(lambda *_args: (source, folder, metadata, True)),
    )


async def _drafts(monkeypatch, tmp_path, rel, dest="", *, download=False):
    _serve_the_artifact(monkeypatch, tmp_path)
    request = SimpleNamespace(query_params={})
    return await workspace_ep.serve_private_draft(
        "local", ARTIFACT_HEX, rel, request, LOCAL_SCOPE, download=download, sec_fetch_dest=dest,
    )


async def test_desktop_viewer_files_answer_any_origin(desktop, tmp_path, monkeypatch):
    response = await _drafts(monkeypatch, tmp_path, "_anton_pdf/viewer.mjs", "script")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.body == pdf_viewer.asset("viewer.mjs")[0]


async def test_desktop_viewer_files_may_be_cached(desktop, tmp_path, monkeypatch):
    response = await _drafts(monkeypatch, tmp_path, "_anton_pdf/pdf.min.mjs", "script")
    assert response.headers["cache-control"] == "private, max-age=3600"


async def test_the_desktop_viewer_page_of_a_pdf_is_not_cached(desktop, tmp_path, monkeypatch):
    response = await _drafts(monkeypatch, tmp_path, "docs/report.pdf", "iframe")
    assert response.headers["cache-control"] == "private, no-store"


async def test_desktop_pdf_bytes_answer_any_origin(desktop, tmp_path, monkeypatch):
    response = await _drafts(monkeypatch, tmp_path, "docs/report.pdf", "empty")
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.headers["vary"] == "Sec-Fetch-Dest"
    assert await served_body(response) == PDF.encode()


async def test_desktop_frame_navigation_gets_the_viewer_page(desktop, tmp_path, monkeypatch):
    response = await _drafts(monkeypatch, tmp_path, "docs/report.pdf", "iframe")
    assert response.body.decode() == prepare_preview_html(
        pdf_viewer.viewer_page("docs/report.pdf", draft=True), comments=False
    )
    assert response.headers["access-control-allow-origin"] == "*"


async def test_desktop_documents_give_the_shim_the_route_address(desktop, tmp_path, monkeypatch):
    response = await _drafts(monkeypatch, tmp_path, "index.html", "iframe")
    assert f'var PDF_ASSETS = "{DESKTOP_ASSETS}";' in response.body.decode()


async def test_a_download_still_gets_the_bytes(desktop, tmp_path, monkeypatch):
    response = await _drafts(monkeypatch, tmp_path, "docs/report.pdf", "iframe", download=True)
    assert response.headers["content-disposition"].startswith("attachment")
    assert await served_body(response) == PDF.encode()


# The org-mode side of the /drafts route is tested in
# tests/test_artifact_resolution_off_the_loop.py, on that file's org fixtures:
# a direct call here would run file store work on the event loop and trip the
# blocking guard.


# --------------------------------------------------------------- whole app, desktop

@pytest.fixture
def app_client(desktop, tmp_path, monkeypatch):
    from cowork.server import create_app

    _serve_the_artifact(monkeypatch, tmp_path)
    client = TestClient(create_app())
    client.app.dependency_overrides[get_tenant_scope] = lambda: LOCAL_SCOPE
    return client


@pytest.mark.parametrize("rel", ["docs/report.pdf", "_anton_pdf/pdf.min.mjs"])
def test_the_desktop_frame_reads_pass_cors_through_the_whole_app(app_client, rel):
    response = app_client.get(
        f"/api/v1/artifacts/drafts/local/{ARTIFACT_HEX}/{rel}",
        headers={"Origin": "null", "Sec-Fetch-Dest": "empty"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    # CORSMiddleware adds these to any request with an Origin.
    assert response.headers["access-control-allow-credentials"] == "true"
