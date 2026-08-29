"""Draft previews keep authorization and file use on one pinned descriptor."""
from __future__ import annotations

import os
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.services.artifacts import ProjectArtifacts
from cowork.services.comments_layer import ACTIVATION_PARAM


def _artifact(tmp_path, *, metadata: dict | None = None):
    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "report"
    folder.mkdir(parents=True)
    source = ProjectArtifacts(
        base=base,
        project_id=None,
        project_name="project",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )
    return source, folder, metadata or {"type": "html-app"}


async def _serve(
    monkeypatch, source, folder, metadata, rel_path, *,
    comments=False, download=False, is_own=True,
):
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        lambda *_args: (source, folder, metadata, is_own),
    )
    request = SimpleNamespace(
        query_params={ACTIVATION_PARAM: "1"} if comments else {}
    )
    return await workspace_ep.serve_private_draft(
        "local",
        "0123456789abcdef0123456789abcdef",
        rel_path,
        request,
        SimpleNamespace(),
        download=download,
    )


async def _stream_body(response: StreamingResponse) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return b"".join(chunks)


@pytest.mark.parametrize(
    "rel_path",
    (
        "../outside.txt",
        "assets/../../outside.txt",
        "..\\outside.txt",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "C:secret.txt",
        "assets//app.js",
        "assets/./app.js",
    ),
)
async def test_traversal_never_reaches_a_filesystem_open(
    tmp_path, monkeypatch, rel_path
):
    source, folder, metadata = _artifact(tmp_path)

    def unexpected_open(*_args):
        raise AssertionError("invalid path reached the pinned open")

    monkeypatch.setattr(workspace_ep, "_open_pinned_draft_file", unexpected_open)

    with pytest.raises(HTTPException) as error:
        await _serve(monkeypatch, source, folder, metadata, rel_path)

    assert error.value.status_code == 400


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX symlink defence")
async def test_file_swap_after_authorization_cannot_change_streamed_bytes(
    tmp_path, monkeypatch
):
    source, folder, metadata = _artifact(tmp_path)
    served = folder / "app.js"
    served.write_text("authorized bytes")
    outside = tmp_path / "outside.js"
    outside.write_text("outside secret")

    response = await _serve(monkeypatch, source, folder, metadata, "app.js")
    assert isinstance(response, StreamingResponse)

    # FileResponse would reopen this name only after the handler returned and
    # disclose the new target. The response must retain the already-open inode.
    served.unlink()
    served.symlink_to(outside)

    assert await _stream_body(response) == b"authorized bytes"
    assert response.headers["content-length"] == str(len("authorized bytes"))
    assert response.headers["cache-control"] == "private, no-store"


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX symlink defence")
async def test_comment_layer_reads_the_pinned_file_not_its_replaced_name(
    tmp_path, monkeypatch
):
    source, folder, metadata = _artifact(tmp_path)
    served = folder / "index.html"
    served.write_text("<html><body>authorized html</body></html>")
    outside = tmp_path / "outside.html"
    outside.write_text("<html><body>outside secret</body></html>")
    read_pinned = workspace_ep._comment_layer_from_fd

    def swap_then_read(fd):
        served.unlink()
        served.symlink_to(outside)
        return read_pinned(fd)

    monkeypatch.setattr(workspace_ep, "_comment_layer_from_fd", swap_then_read)

    response = await _serve(
        monkeypatch, source, folder, metadata, "index.html", comments=True
    )

    assert isinstance(response, HTMLResponse)
    body = response.body.decode()
    assert "authorized html" in body
    assert "outside secret" not in body
    assert response.headers["cache-control"] == "private, no-store"


async def test_non_utf8_comment_preview_falls_back_to_the_pinned_byte_stream(
    tmp_path, monkeypatch
):
    source, folder, metadata = _artifact(tmp_path)
    payload = b"<html>\xffbinary</html>"
    (folder / "index.html").write_bytes(payload)

    response = await _serve(
        monkeypatch, source, folder, metadata, "index.html", comments=True
    )

    assert isinstance(response, StreamingResponse)
    assert await _stream_body(response) == payload


def test_request_components_are_replaced_with_names_read_from_disk(
    tmp_path, monkeypatch
):
    source, folder, _metadata = _artifact(tmp_path)
    (folder / "assets").mkdir()
    (folder / "assets" / "app.js").write_text("safe", encoding="utf-8")
    requested_dir = ("assets" + "x")[:-1]
    requested_file = ("app.js" + "x")[:-1]
    requested = (requested_dir, requested_file)
    opened_names = []
    original_open_child = workspace_ep.open_pinned_child
    original_dir_open = workspace_ep.dir_open

    def open_child(directory, name):
        opened_names.append(name)
        return original_open_child(directory, name)

    def open_file(directory, name, flags, mode=0o777):
        opened_names.append(name)
        return original_dir_open(directory, name, flags, mode)

    monkeypatch.setattr(workspace_ep, "open_pinned_child", open_child)
    monkeypatch.setattr(workspace_ep, "dir_open", open_file)

    resources, fd, _file_stat = workspace_ep._open_pinned_draft_file(
        source, folder, requested
    )
    try:
        assert os.read(fd, 4) == b"safe"
    finally:
        resources.close()

    assert opened_names == ["assets", "app.js"]
    assert opened_names[0] is not requested_dir
    assert opened_names[1] is not requested_file


async def test_send_failure_before_iteration_closes_the_pinned_file(tmp_path):
    path = tmp_path / "draft.txt"
    path.write_text("draft", encoding="utf-8")
    fd = os.open(path, os.O_RDONLY)
    resources = ExitStack()
    resources.callback(os.close, fd)
    response = workspace_ep._draft_stream(resources, fd, 5, "text/plain")

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        raise RuntimeError("send failed")

    with pytest.raises(RuntimeError, match="send failed"):
        await response(
            {"type": "http", "asgi": {"spec_version": "2.4"}},
            receive,
            send,
        )

    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX symlink defence")
@pytest.mark.parametrize("linked_component", ("folder", "nested", "file"))
async def test_a_symlink_at_any_artifact_component_is_not_served(
    tmp_path, monkeypatch, linked_component
):
    source, folder, metadata = _artifact(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside secret")

    if linked_component == "folder":
        folder.rmdir()
        folder.symlink_to(outside, target_is_directory=True)
        rel_path = "secret.txt"
    elif linked_component == "nested":
        (folder / "assets").symlink_to(outside, target_is_directory=True)
        rel_path = "assets/secret.txt"
    else:
        (folder / "secret.txt").symlink_to(outside / "secret.txt")
        rel_path = "secret.txt"

    with pytest.raises(HTTPException) as error:
        await _serve(monkeypatch, source, folder, metadata, rel_path)

    assert error.value.status_code == 404


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX symlink defence")
async def test_a_symlinked_artifacts_root_cannot_reuse_the_sources_authority(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    (project / ".anton").mkdir(parents=True)
    outside_base = tmp_path / "another-workspace" / "artifacts"
    outside_folder = outside_base / "report"
    outside_folder.mkdir(parents=True)
    (outside_folder / "secret.txt").write_text("another workspace's secret")
    declared_base = project / ".anton" / "artifacts"
    declared_base.symlink_to(outside_base, target_is_directory=True)
    source = ProjectArtifacts(
        base=declared_base,
        project_id=None,
        project_name="project",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )

    with pytest.raises(HTTPException) as error:
        await _serve(
            monkeypatch,
            source,
            outside_folder,
            {"type": "html-app"},
            "secret.txt",
        )

    assert error.value.status_code == 404


async def test_fullstack_preview_stays_inside_the_declared_public_subtree(
    tmp_path, monkeypatch
):
    source, folder, _metadata = _artifact(tmp_path)
    (folder / "static").mkdir()
    (folder / "static" / "app.js").write_text("public")
    (folder / "backend.py").write_text("secret")
    metadata = {
        "type": "fullstack-stateless-app",
        "primary": "static/index.html",
    }

    response = await _serve(
        monkeypatch, source, folder, metadata, "static/app.js"
    )
    assert await _stream_body(response) == b"public"

    with pytest.raises(HTTPException) as error:
        await _serve(monkeypatch, source, folder, metadata, "backend.py")
    assert error.value.status_code == 404


# ── ?download=1 (ENG-2044) ──────────────────────────────────────────────────
# On an org deployment this parameter is the only way to obtain a non-HTML
# artifact: `/serve` is desktop-only there and autopublish skips anything that
# is not HTML/Markdown. The bytes and the authorization are the preview's; only
# the disposition changes.

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


async def test_download_serves_a_binary_as_an_attachment(tmp_path, monkeypatch):
    source, folder, metadata = _artifact(tmp_path, metadata={"type": "file"})
    payload = b"PK\x03\x04" + bytes(range(64))
    (folder / "report.xlsx").write_bytes(payload)

    response = await _serve(monkeypatch, source, folder, metadata, "report.xlsx", download=True)

    assert isinstance(response, StreamingResponse)
    assert response.media_type == XLSX
    assert response.headers["content-disposition"] == (
        "attachment; filename=\"report.xlsx\"; filename*=UTF-8''report.xlsx"
    )
    # The preview's hardening headers are kept, not replaced.
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["content-length"] == str(len(payload))
    assert await _stream_body(response) == payload


async def test_without_download_the_response_is_unchanged(tmp_path, monkeypatch):
    source, folder, metadata = _artifact(tmp_path, metadata={"type": "file"})
    (folder / "report.xlsx").write_bytes(b"PK")

    response = await _serve(monkeypatch, source, folder, metadata, "report.xlsx")

    assert "content-disposition" not in response.headers


async def test_download_skips_the_comment_layer(tmp_path, monkeypatch):
    """A saved HTML file must be the file, not the review page wrapped around it."""
    source, folder, metadata = _artifact(tmp_path)
    (folder / "index.html").write_text("<h1>deck</h1>", encoding="utf-8")

    response = await _serve(
        monkeypatch, source, folder, metadata, "index.html", comments=True, download=True,
    )

    assert isinstance(response, StreamingResponse)
    assert not isinstance(response, HTMLResponse)
    assert response.headers["content-disposition"].startswith("attachment;")
    assert await _stream_body(response) == b"<h1>deck</h1>"


@pytest.mark.parametrize(
    ("name", "ascii_part", "encoded_part"),
    (
        # A quote must be escaped, not close the parameter early.
        ('evil" bad.txt', 'evil\\" bad.txt', "evil%22%20bad.txt"),
        # CR/LF are dropped: the header can never gain a second line.
        ("a\r\nX-Injected: 1.txt", "aX-Injected: 1.txt", "aX-Injected%3A%201.txt"),
        # Non-ASCII survives only in the RFC 5987 spelling; the ASCII one degrades.
        ("rapport-\u00e9.xlsx", "rapport-.xlsx", "rapport-%C3%A9.xlsx"),
    ),
)
async def test_download_filename_cannot_inject_headers(
    tmp_path, monkeypatch, name, ascii_part, encoded_part
):
    """`_relative_file_parts` lets these names through — it only guards the
    filesystem walk. The header builder is the second gate."""
    source, folder, metadata = _artifact(tmp_path, metadata={"type": "file"})
    (folder / name).write_bytes(b"x")

    response = await _serve(monkeypatch, source, folder, metadata, name, download=True)

    disposition = response.headers["content-disposition"]
    assert "\r" not in disposition and "\n" not in disposition
    assert disposition == (
        f'attachment; filename="{ascii_part}"; filename*=UTF-8\'\'{encoded_part}'
    )


async def test_download_is_served_to_a_granted_reviewer(tmp_path, monkeypatch):
    """Recorded decision (ENG-2044): a review grant already exposes every byte
    through the preview, so a reviewer may save the file too."""
    source, folder, metadata = _artifact(tmp_path, metadata={"type": "file"})
    (folder / "report.docx").write_bytes(b"PK")

    response = await _serve(
        monkeypatch, source, folder, metadata, "report.docx", download=True, is_own=False,
    )

    assert response.headers["content-disposition"].startswith("attachment;")


async def test_download_does_not_bypass_authorization(tmp_path, monkeypatch):
    """The parameter changes the disposition only; who may read is decided
    before it is ever looked at."""
    def deny(*_args):
        raise HTTPException(status_code=404, detail="Artifact not found")

    monkeypatch.setattr(workspace_ep, "review_artifact_for_request", deny)

    with pytest.raises(HTTPException) as excinfo:
        await workspace_ep.serve_private_draft(
            "local", "0123456789abcdef0123456789abcdef", "report.xlsx",
            SimpleNamespace(query_params={}), SimpleNamespace(), download=True,
        )
    assert excinfo.value.status_code == 404
