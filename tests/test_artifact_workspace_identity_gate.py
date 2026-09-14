"""The workspace routes' path-level identity gate.

The rest of the workspace suite calls the handlers directly, which skips
FastAPI's own parameter validation — so a gate that only exists in the route
signature would look tested while never running. These go over HTTP.
"""
from __future__ import annotations

import mimetypes
import os
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from cowork.api.v1 import artifact_scope
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE


@pytest.fixture
def client(monkeypatch):
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield TestClient(create_app())
    get_app_settings.cache_clear()


@pytest.fixture
def no_resolution(monkeypatch):
    """Fail loudly if a request reaches artifact resolution at all.

    That is the point of the gate: a malformed identity must be refused before
    the service layer, the identity index or the filesystem sees the string.
    """
    def explode(*_args, **_kwargs):
        raise AssertionError("resolution reached with an unvalidated identity")

    monkeypatch.setattr("cowork.api.v1.artifact_scope._resolve_in", explode)


@pytest.mark.parametrize(
    "artifact_id",
    [
        "../../../etc/passwd",
        "..%2f..%2fetc%2fpasswd",
        "a1b2c3d4",                              # legacy 8-char id, no workspace
        "not-a-uuid",
        "0123456789abcdef0123456789abcdefff",    # hex, but too long for a UUID
        "",
    ],
)
def test_a_malformed_identity_never_reaches_resolution(client, no_resolution, artifact_id):
    res = client.get(f"/api/v1/artifacts/workspace/local/{artifact_id}")

    assert res.status_code in (404, 422), res.text


def test_both_uuid_spellings_address_the_same_artifact(client, monkeypatch):
    """The dashed and undashed forms are one identity, and the handler receives
    the canonical 32-hex spelling either way — that is what metadata carries."""
    seen: list[str] = []

    def capture(_sources, artifact_id):
        seen.append(artifact_id)
        raise AssertionError("stop here; the id is what this test is about")

    monkeypatch.setattr("cowork.api.v1.artifact_scope._resolve_in", capture)

    dashed = "0f9e8d7c-6b5a-4938-8271-605f4e3d2c1b"
    undashed = dashed.replace("-", "")
    for spelling in (dashed, undashed):
        with pytest.raises(AssertionError, match="stop here"):
            client.get(f"/api/v1/artifacts/workspace/local/{spelling}")

    assert seen == [undashed, undashed]


def test_the_draft_preview_route_is_gated_too(client, no_resolution):
    res = client.get("/api/v1/artifacts/drafts/local/not-a-uuid/index.html")

    assert res.status_code == 422, res.text


def test_project_root_discovery_receives_the_scoped_catalog_id(monkeypatch):
    """The equal request spelling chooses a DB value; it is never reused as
    the UUID handed to artifact-root discovery."""
    server_id = UUID("0f9e8d7c-6b5a-4938-8271-605f4e3d2c1b")
    request_ref = (str(server_id) + "x")[:-1]
    seen = []

    class Projects:
        def __init__(self, _session):
            pass

        def list_projects(self):
            return [SimpleNamespace(id=server_id)]

    monkeypatch.setattr("cowork.services.projects.ProjectService", Projects)
    monkeypatch.setattr(
        artifact_scope,
        "artifacts_sources_for_project",
        lambda _session, project_id, **_kwargs: seen.append(project_id) or [],
    )

    assert artifact_scope._sources_for_project_ref(object(), request_ref) == []
    assert seen == [server_id]
    assert seen[0] is server_id


# ── ?download=1 is read at the route boundary (ENG-2044) ─────────────────────
# test_artifact_draft_pinned_serving.py exercises the download behaviour by
# calling the handler with `download=True`. That proves the header builder, not
# that FastAPI parses the query string into the parameter — a `Query(False)`
# default that leaks the `Query` object into a direct call, or a renamed
# parameter, would leave those tests green while every real request ignored
# the flag. These go over HTTP for the same reason the gate tests above do.

# See test_artifact_draft_pinned_serving.py: the built-in mimetypes map has no
# `.xlsx`, so the expectation is derived the way the route derives it.
_XLSX = mimetypes.guess_type("model.xlsx")[0] or "application/octet-stream"


@pytest.fixture
def served_xlsx(tmp_path, monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
    from cowork.services.artifacts import ProjectArtifacts

    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "model"
    folder.mkdir(parents=True)
    (folder / "model.xlsx").write_bytes(b"PK\x03\x04sheet")
    source = ProjectArtifacts(
        base=base, project_id=None, project_name="project",
        trusted_anchor=project, root_parts=(".anton", "artifacts"),
    )
    monkeypatch.setattr(
        workspace_ep, "review_artifact_for_request",
        lambda *_args: (source, folder, {"type": "file"}, True),
    )
    return "/api/v1/artifacts/drafts/local/0123456789abcdef0123456789abcdef/model.xlsx"


@pytest.mark.parametrize("flag", ["1", "true"])
def test_download_flag_reaches_the_handler_over_http(client, served_xlsx, flag):
    res = client.get(f"{served_xlsx}?download={flag}")

    assert res.status_code == 200, res.text
    assert res.headers["content-disposition"] == (
        "attachment; filename=\"model.xlsx\"; filename*=UTF-8''model.xlsx"
    )
    assert res.headers["content-type"].startswith(_XLSX)
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.content == b"PK\x03\x04sheet"


@pytest.mark.parametrize("query", ["", "?download=0", "?download=false"])
def test_preview_is_unchanged_without_the_flag(client, served_xlsx, query):
    res = client.get(f"{served_xlsx}{query}")

    assert res.status_code == 200, res.text
    assert "content-disposition" not in res.headers
    assert res.content == b"PK\x03\x04sheet"


# ── the cancel route's discard intent is read at the boundary ────────────────
# The service tests call cancel_agent_repair directly, which proves the
# discard, not that FastAPI parses the body into the flag. An old client posts
# a bare `{}` and must keep the queued-only behaviour, so both spellings go
# over HTTP for the same reason as the tests above.


@pytest.fixture
def cancel_route(tmp_path, monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
    from cowork.services.artifacts import ProjectArtifacts

    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "brief"
    folder.mkdir(parents=True)
    source = ProjectArtifacts(
        base=base, project_id=None, project_name="project",
        trusted_anchor=project, root_parts=(".anton", "artifacts"),
    )
    monkeypatch.setattr(
        workspace_ep, "review_artifact_for_request",
        lambda *_args: (source, folder, {"type": "document"}, True),
    )
    monkeypatch.setattr(workspace_ep, "require_artifact_owner", lambda *_args: {"role": "owner"})
    seen: list[bool] = []

    def capture(_folder, _repair_id, *, discard_ready=False):
        seen.append(discard_ready)
        return {"id": "repair-1", "status": "discarded" if discard_ready else "cancelled"}

    monkeypatch.setattr(workspace_ep, "cancel_agent_repair", capture)
    return seen


@pytest.mark.parametrize(
    ("body", "expected"),
    [({}, False), ({"discardReady": False}, False), ({"discardReady": True}, True)],
)
def test_discard_intent_reaches_the_handler_over_http(client, cancel_route, body, expected):
    res = client.post(
        "/api/v1/artifacts/workspace/local/0123456789abcdef0123456789abcdef"
        "/agent-repairs/repair-1/cancel",
        json=body,
    )

    assert res.status_code == 200, res.text
    assert cancel_route == [expected]


# ── the comment-release route resolves as a literal, and is owner-scoped ─────
# It sits next to /agent-repairs/{repair_id}/..., so "release" must not be read
# as a repair id; and it is on this router precisely because the comments
# router forwards to inference in org mode with no tenant scope, so the owner
# gate has to be exercised here.


@pytest.fixture
def release_route(tmp_path, monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
    from cowork.services.artifacts import ProjectArtifacts

    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "brief"
    folder.mkdir(parents=True)
    source = ProjectArtifacts(
        base=base, project_id=None, project_name="project",
        trusted_anchor=project, root_parts=(".anton", "artifacts"),
    )
    monkeypatch.setattr(
        workspace_ep, "review_artifact_for_request",
        lambda *_args: (source, folder, {"type": "document"}, True),
    )
    monkeypatch.setattr(workspace_ep, "require_artifact_owner", lambda *_args: {"role": "owner"})
    seen: list[str] = []
    monkeypatch.setattr(
        workspace_ep, "release_repairs_for_comment",
        lambda _folder, thread_id: seen.append(thread_id) or [{"id": "r1", "status": "discarded"}],
    )
    return seen


_RELEASE_URL = (
    "/api/v1/artifacts/workspace/local/0123456789abcdef0123456789abcdef"
    "/agent-repairs/release"
)


def test_release_route_is_not_read_as_a_repair_id(client, release_route):
    res = client.post(_RELEASE_URL, json={"commentThreadId": "thread-1"})

    assert res.status_code == 200, res.text
    assert res.json() == {"released": [{"id": "r1", "status": "discarded"}]}
    assert release_route == ["thread-1"]


def test_release_requires_a_comment_thread(client, release_route):
    res = client.post(_RELEASE_URL, json={"commentThreadId": ""})

    assert res.status_code == 422, res.text
    assert release_route == []


def test_release_is_refused_without_owner_access(client, tmp_path, monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
    from fastapi import HTTPException as _HTTPException

    def deny(*_args):
        raise _HTTPException(status_code=403, detail="Not the owner")

    monkeypatch.setattr(workspace_ep, "review_artifact_for_request", deny)
    monkeypatch.setattr(
        workspace_ep, "release_repairs_for_comment",
        lambda *_a, **_k: pytest.fail("release ran without an owner check"),
    )

    res = client.post(_RELEASE_URL, json={"commentThreadId": "thread-1"})

    assert res.status_code == 403, res.text


# ── the source path is a selector, not a path ────────────────────────────────
# `?path=` (and the PUT body's `path`) name the file to edit. The revision
# service resolves that name under the artifact folder, and its own checks
# (no `..`, containment, extension allowlist) are the inner gate. The route
# boundary is the outer one: the request string is matched against a scandir
# pass on pinned descriptors and only the OS-returned spelling travels on —
# the same rule every other filesystem-facing route in the file follows.

_WORKSPACE_URL = "/api/v1/artifacts/workspace/local/0123456789abcdef0123456789abcdef"


@pytest.fixture
def editable_artifact(tmp_path, monkeypatch):
    # Every route test below uses this fixture, and two of its entries are
    # symlinks, so without the guard the whole group errors in setup rather
    # than skipping where links are unavailable.
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("the fixture's symlink entries need a POSIX filesystem")
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
    from cowork.services.artifacts import ProjectArtifacts

    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "brief"
    (folder / "docs").mkdir(parents=True)
    (folder / "brief.md").write_text("# primary\n", encoding="utf-8")
    (folder / "docs" / "notes.md").write_text("# nested\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("# secret\n", encoding="utf-8")
    (folder / "link.md").symlink_to(outside)
    (folder / "linkdir").symlink_to(folder / "docs", target_is_directory=True)
    source = ProjectArtifacts(
        base=base, project_id=None, project_name="project",
        trusted_anchor=project, root_parts=(".anton", "artifacts"),
    )
    metadata = {"type": "file", "primary": "brief.md"}
    monkeypatch.setattr(
        workspace_ep, "_owner_workspace",
        lambda *_args: (source, folder, metadata, {"canEdit": True, "canAddressWithAgent": True, "canResolveComments": True}),
    )
    return SimpleNamespace(source=source, folder=folder, metadata=metadata)


def test_the_service_receives_the_resolved_path_on_both_routes(
    editable_artifact, monkeypatch,
):
    """Route-level value check: both routes hand the service the same path.

    Provenance is not testable here. `"/".join` builds a fresh string for any
    multi-component path, so an `is not` against the request would hold even
    for a selector that never consulted the disk. The two tests below cover
    that property where it can actually fail.
    """
    import asyncio

    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep

    received: list[object] = []

    def capture_read(folder, metadata, artifact_id, rel_path=None):
        received.append(rel_path)
        return {"path": rel_path, "content": "", "revision": {}, "revisions": []}

    def capture_save(folder, metadata, artifact_id, *, rel_path=None, **_kw):
        received.append(rel_path)
        # The id matches the expected one, so the route's live-artifact sync
        # stays out of a test about which string reaches the service.
        return {"path": rel_path, "revision": {"id": "r1"}}

    monkeypatch.setattr(workspace_ep, "current_workspace", capture_read)
    monkeypatch.setattr(workspace_ep, "active_agent_repair", lambda *_a, **_k: None)
    monkeypatch.setattr(workspace_ep, "save_source", capture_save)

    requested = "docs/notes.md"
    asyncio.run(workspace_ep.artifact_source(
        "local", "0123456789abcdef0123456789abcdef", session=SimpleNamespace(scope=LOCAL_SCOPE), path=requested,
    ))
    body = workspace_ep._SourceUpdateBody(
        content="x", expectedRevisionId="r1", path=requested,
    )
    asyncio.run(workspace_ep.update_artifact_source(
        "local", "0123456789abcdef0123456789abcdef", body=body, session=SimpleNamespace(scope=LOCAL_SCOPE),
    ))

    assert received == ["docs/notes.md", "docs/notes.md"]


def test_no_path_leaves_the_choice_to_the_service(editable_artifact, client):
    res = client.get(_WORKSPACE_URL)

    assert res.status_code == 200, res.text
    assert res.json()["path"] == "brief.md"


def test_a_nested_source_is_read_and_saved_through_the_selector(editable_artifact, client):
    res = client.get(f"{_WORKSPACE_URL}?path=docs/notes.md")
    assert res.status_code == 200, res.text
    assert res.json()["path"] == "docs/notes.md"
    assert res.json()["content"] == "# nested\n"

    saved = client.put(_WORKSPACE_URL, json={
        "content": "# edited\n",
        "expectedRevisionId": res.json()["revision"]["id"],
        "path": "docs/notes.md",
    })
    assert saved.status_code == 200, saved.text
    assert (editable_artifact.folder / "docs" / "notes.md").read_text() == "# edited\n"


@pytest.mark.parametrize(
    "path",
    [
        "../outside.md",
        "..%2Foutside.md",
        "docs/../../outside.md",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "docs\\..\\..\\outside.md",
        "brief.md%00.html",
        ".revisions/manifest.json",
    ],
)
def test_traversal_shapes_are_refused_before_the_service(
    editable_artifact, client, monkeypatch, path,
):
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep

    def explode(*_args, **_kwargs):
        raise AssertionError("the revision service saw an unvalidated path")

    monkeypatch.setattr(workspace_ep, "current_workspace", explode)
    monkeypatch.setattr(workspace_ep, "save_source", explode)

    res = client.get(f"{_WORKSPACE_URL}?path={path}")
    assert res.status_code == 422, res.text

    res = client.put(_WORKSPACE_URL, json={
        "content": "x", "expectedRevisionId": "r1", "path": path.replace("%2F", "/").replace("%00", "\x00"),
    })
    assert res.status_code == 422, res.text


@pytest.mark.parametrize(
    "path",
    [
        "missing.md",
        "docs/missing.md",
        "docs",                 # a directory, not a file
        "link.md",              # symlink to a file outside the folder
        "linkdir/notes.md",     # symlinked directory, even though it lands inside
    ],
)
def test_symlinks_and_missing_entries_are_not_found_before_the_service(
    editable_artifact, client, monkeypatch, path,
):
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep

    def explode(*_args, **_kwargs):
        raise AssertionError("the revision service saw a path that was not on disk")

    monkeypatch.setattr(workspace_ep, "current_workspace", explode)
    monkeypatch.setattr(workspace_ep, "save_source", explode)

    res = client.get(f"{_WORKSPACE_URL}?path={path}")
    assert res.status_code == 404, res.text

    res = client.put(_WORKSPACE_URL, json={
        "content": "x", "expectedRevisionId": "r1", "path": path,
    })
    assert res.status_code == 404, res.text


@pytest.fixture
def readme_backed_artifact(tmp_path, monkeypatch):
    """An artifact whose editable source is the one the service picks itself.

    `metadata["primary"]` is optional, and without it `resolve_source` takes
    the sorted-first editable file. `README.md` sorts ahead of any lowercase
    name, so it is the source the GET reports for artifacts like this one.
    """
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
    from cowork.services.artifacts import ProjectArtifacts

    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "brief"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text("# readme\n", encoding="utf-8")
    (folder / "index.html").write_text("<h1>x</h1>\n", encoding="utf-8")
    source = ProjectArtifacts(
        base=base, project_id=None, project_name="project",
        trusted_anchor=project, root_parts=(".anton", "artifacts"),
    )
    metadata = {"type": "file"}
    monkeypatch.setattr(
        workspace_ep, "_owner_workspace",
        lambda *_args: (source, folder, metadata, {"canEdit": True, "canAddressWithAgent": True, "canResolveComments": True}),
    )
    return SimpleNamespace(source=source, folder=folder, metadata=metadata)


def test_the_path_a_get_reports_can_be_saved_back(readme_backed_artifact, client):
    """The round trip a client actually performs: read, then save what it read.

    The selector must accept every path the service is willing to report. A
    boundary refusal the inner gate does not share makes the reported source
    unsaveable, and the client has no other path to send.
    """
    read = client.get(_WORKSPACE_URL)
    assert read.status_code == 200, read.text
    reported = read.json()["path"]
    assert reported == "README.md"

    saved = client.put(_WORKSPACE_URL, json={
        "content": "# edited\n",
        "expectedRevisionId": read.json()["revision"]["id"],
        "path": reported,
    })

    assert saved.status_code == 200, saved.text
    assert (readme_backed_artifact.folder / "README.md").read_text() == "# edited\n"


def test_the_names_handed_to_the_filesystem_are_the_ones_scandir_returned(
    editable_artifact, monkeypatch,
):
    """Every component reaching `openat` is the object the directory scan produced.

    `is not` against the request cannot show this for a multi-component path:
    splitting the joined string yields fresh objects whatever the selector
    then does with them. So the assertion is against the scan's own return
    values, which a selector that skipped the scan would not have.
    """
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep

    from_disk = []
    opened_names = []
    original_entry_name = workspace_ep._existing_draft_entry_name
    original_open_child = workspace_ep.open_pinned_child
    original_dir_lstat = workspace_ep.dir_lstat

    def entry_name(directory, requested, **kwargs):
        name = original_entry_name(directory, requested, **kwargs)
        from_disk.append(name)
        return name

    def open_child(directory, name):
        opened_names.append(name)
        return original_open_child(directory, name)

    def lstat(directory, name):
        opened_names.append(name)
        return original_dir_lstat(directory, name)

    monkeypatch.setattr(workspace_ep, "_existing_draft_entry_name", entry_name)
    monkeypatch.setattr(workspace_ep, "open_pinned_child", open_child)
    monkeypatch.setattr(workspace_ep, "dir_lstat", lstat)

    selected = workspace_ep._editable_source_selector(
        editable_artifact.source, editable_artifact.folder, "docs/notes.md"
    )

    assert selected == "docs/notes.md"
    assert from_disk == ["docs", "notes.md"]
    assert opened_names == ["docs", "notes.md"]
    assert all(
        opened is scanned for opened, scanned in zip(opened_names, from_disk, strict=True)
    )


def test_a_single_component_is_replaced_by_the_disk_name_too(editable_artifact):
    """The case `"/".join` cannot launder.

    Joining a one-element list returns that element, and splitting a string
    with no separator returns the string itself, so a selector that skipped
    the disk match would hand the request object straight through. Only this
    shape distinguishes the two.
    """
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep

    requested = ("brief.md" + "x")[:-1]

    selected = workspace_ep._editable_source_selector(
        editable_artifact.source, editable_artifact.folder, requested
    )

    assert selected == "brief.md"
    assert selected is not requested


def _case_insensitive(directory: Path) -> bool:
    """Ask the volume rather than the platform, as the selector does."""
    probe = directory / "CaseProbe.tmp"
    probe.write_text("x", encoding="utf-8")
    try:
        return (directory / "caseprobe.tmp").exists()
    finally:
        probe.unlink()


@pytest.fixture
def mixed_case_artifact(tmp_path, monkeypatch):
    """An artifact whose recorded primary is spelled unlike its file.

    `metadata.json` is written outside this service, so the two spellings can
    differ. `resolve_source` opens by path and inherits the volume's case
    rules, which means it reports the primary's spelling verbatim.
    """
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
    from cowork.services.artifacts import ProjectArtifacts

    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "brief"
    folder.mkdir(parents=True)
    (folder / "brief.md").write_text("# primary\n", encoding="utf-8")
    source = ProjectArtifacts(
        base=base, project_id=None, project_name="project",
        trusted_anchor=project, root_parts=(".anton", "artifacts"),
    )
    metadata = {"type": "file", "primary": "Brief.md"}
    monkeypatch.setattr(
        workspace_ep, "_owner_workspace",
        lambda *_args: (source, folder, metadata, {"canEdit": True, "canAddressWithAgent": True, "canResolveComments": True}),
    )
    return SimpleNamespace(source=source, folder=folder, metadata=metadata)


def test_a_case_mismatched_primary_reports_the_disk_spelling(mixed_case_artifact, client):
    """Both routes name the source the same way, so the journal has one key.

    This is the property that broke when only the write side was translated:
    the read reported `Brief.md`, the write recorded `brief.md`, and the
    mismatched revision id came back 409 where a 404 had been.

    Only a case-insensitive volume can show it. On a case-sensitive one
    `Brief.md` names nothing, `resolve_source` refuses it, and the artifact's
    source is unreachable through this route both before and after this
    change -- a set-but-absent primary is not the empty primary the service
    falls back on.
    """
    if not _case_insensitive(mixed_case_artifact.folder):
        pytest.skip("needs a case-insensitive volume; the mismatch cannot arise")

    read = client.get(_WORKSPACE_URL)
    assert read.status_code == 200, read.text
    reported = read.json()["path"]
    assert reported == "brief.md"

    saved = client.put(_WORKSPACE_URL, json={
        "content": "# edited\n",
        "expectedRevisionId": read.json()["revision"]["id"],
        "path": reported,
    })

    assert saved.status_code == 200, saved.text
    assert (mixed_case_artifact.folder / "brief.md").read_text() == "# edited\n"


def test_the_recorded_primary_is_accepted_on_a_case_insensitive_volume(
    mixed_case_artifact, client,
):
    """The spelling the volume itself accepts is accepted here too.

    This is the half a case-exact boundary refused. It cannot be asserted
    where the volume is case-sensitive, because there `Brief.md` names
    nothing and refusing it is correct.
    """
    if not _case_insensitive(mixed_case_artifact.folder):
        pytest.skip("needs a case-insensitive volume; the spelling names nothing here")

    res = client.get(f"{_WORKSPACE_URL}?path=Brief.md")

    assert res.status_code == 200, res.text
    assert res.json()["path"] == "brief.md"


def test_a_case_variant_is_refused_where_the_volume_is_case_sensitive(
    mixed_case_artifact, client,
):
    """The other half: accepting the volume's rules must not invent entries."""
    if _case_insensitive(mixed_case_artifact.folder):
        pytest.skip("needs a case-sensitive volume; the variant is a real file here")

    res = client.get(f"{_WORKSPACE_URL}?path=Brief.md")

    assert res.status_code == 404, res.text


def test_a_name_absent_from_disk_is_refused_whatever_the_volume(editable_artifact):
    """`native_case` widens the accepted spellings, never the accepted files."""
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep

    with pytest.raises(HTTPException) as refused:
        workspace_ep._editable_source_selector(
            editable_artifact.source, editable_artifact.folder, "nothing-here.md"
        )

    assert refused.value.status_code == 404


def test_an_oversized_path_is_refused_by_the_route(editable_artifact, client, monkeypatch):
    """Both spellings of the parameter answer the same way on length."""
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep

    def explode(*_args, **_kwargs):
        raise AssertionError("an oversized path reached the service")

    monkeypatch.setattr(workspace_ep, "current_workspace", explode)
    monkeypatch.setattr(workspace_ep, "save_source", explode)
    oversized = "a" * 1001

    assert client.get(f"{_WORKSPACE_URL}?path={oversized}").status_code == 422
    assert client.get(f"{_WORKSPACE_URL}/revisions?path={oversized}").status_code == 422
    assert client.put(_WORKSPACE_URL, json={
        "content": "x", "expectedRevisionId": "r1", "path": oversized,
    }).status_code == 422


def test_a_folder_outside_the_sources_base_is_refused(editable_artifact, tmp_path):
    """`_artifact_folder_component` is what keeps a grant folder-specific.

    Resolution hands the route a folder alongside the source that authorized
    it. Translating any other folder to its basename would let a grant for one
    resolved source select a same-named folder under a different one, so a
    folder whose parent is not the source's base is refused outright.
    """
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep

    elsewhere = tmp_path / "elsewhere" / ".anton" / "artifacts" / "brief"
    elsewhere.mkdir(parents=True)
    (elsewhere / "brief.md").write_text("# other\n", encoding="utf-8")

    with pytest.raises(HTTPException) as refused:
        workspace_ep._editable_source_selector(
            editable_artifact.source, elsewhere, "brief.md"
        )

    assert refused.value.status_code == 404
