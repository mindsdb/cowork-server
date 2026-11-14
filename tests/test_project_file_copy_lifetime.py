"""How long a cloud turn's copy of its project's files lives, and who reaches
it: a conversation deleted while its copy runs keeps nothing, a moved
conversation leaves no copy in the project it left, the user routes never
serve a copy, and the list of files a copy left out stays what anton's
strict reader accepts."""
from __future__ import annotations

import errno
import json
import logging
import os
import shutil
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import cowork.services.project_file_copy as project_file_copy
import cowork.services.project_file_view as project_file_view
from cowork.db.scoped import TenantScope
from cowork.db.units import on_efs_thread
from cowork.services.project_file_copy import (
    NOT_COPIED_FILENAME,
    NotCopiedFile,
    NotCopiedFiles,
    NotCopiedReason,
    ProjectFileCopyLimits,
    ProjectFilesOutcome,
    stage_project_files,
)

ORG = "7a1c2d3e-4f5a-4b6c-8d7e-9f0a1b2c3d83"
USER = "8b2d3e4f-5a6b-4c7d-9e8f-0a1b2c3d4e83"
SCOPE = TenantScope(org_mode=True, org_id=ORG, user_id=USER)
MEMBER = {"X-User-Id": USER, "X-Organization-Id": ORG}
LIMITS = ProjectFileCopyLimits(max_file_bytes=1000, max_files=50, max_total_bytes=10_000, seconds=30.0)

posix_only = pytest.mark.skipif(os.name == "nt", reason="org mode runs on POSIX only")

#: Every key of the list, and of one named file, that anton's strict reader
#: (_NotCopiedFiles in anton/cloud_turn/session.py) accepts, and every reason.
ANTON_LIST_KEYS = {"files", "unnamed", "listing_truncated", "copy_failed"}
ANTON_FILE_KEYS = {"path", "size", "reason"}
ANTON_REASONS = {"too_large", "too_many_files", "over_total_bytes", "out_of_time"}


def _write(root: Path, rel: str, content: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def project(org_deployment) -> Path:
    from cowork.services.projects import projects_root

    folder = projects_root(scope=SCOPE) / f"lifetime-{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    return folder


def _stage(project: Path, conv, **limits):
    return stage_project_files(project, conv, scope=SCOPE, limits=LIMITS._replace(**limits))


def _copy_of(project: Path, conv) -> Path:
    return project / "conversations" / str(conv) / "project-files"


def _copied(project: Path, conv) -> dict[str, str]:
    root = _copy_of(project, conv)
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != NOT_COPIED_FILENAME
    }


def _app():
    from cowork.server import create_app

    return create_app()


def _project_with_a_conversation(client: TestClient, prefix: str) -> tuple[dict, Path, UUID]:
    created = client.post("/api/v1/projects/", json={"name": f"{prefix}-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert created.status_code == 201, created.text
    conversation = client.post(
        "/api/v1/conversations/", json={"title": "t", "projectId": created.json()["id"]}, headers=MEMBER,
    )
    assert conversation.status_code == 201, conversation.text
    return created.json(), Path(created.json()["path"]), UUID(conversation.json()["id"])


# ── A conversation deleted while its copy runs ──────────────────────────


@posix_only
def test_a_conversation_deleted_while_its_copy_runs_gets_no_folder_or_record_again(project, monkeypatch, caplog):
    """The delete removes the conversation's folder while the copy writes
    into it. The copy's next write fails, and the copy keeps nothing: it
    makes neither the folder nor a record again, where nothing would ever
    remove them. Its WARNING names the conversation as a field, which the
    deployments' JSON log keeps, not as a template value, which it drops."""
    _write(project, "a.md", "a")
    _write(project, "b.md", "b")
    conv = uuid4()
    conversation = project / "conversations" / str(conv)
    real_copy_file = project_file_copy._copy_file

    def deleted_midway(found, target, **kwargs):
        if conversation.exists():
            shutil.rmtree(conversation)
        return real_copy_file(found, target, **kwargs)

    monkeypatch.setattr(project_file_copy, "_copy_file", deleted_midway)

    with caplog.at_level(logging.WARNING, logger=project_file_copy.logger.name):
        result = _stage(project, conv)

    assert result.outcome is ProjectFilesOutcome.CONVERSATION_GONE
    assert not conversation.exists()
    assert not project_file_copy.copy_record_path(SCOPE, conv).exists()
    [record] = [r for r in caplog.records if r.name == project_file_copy.logger.name]
    assert (record.levelname, record.args) == ("WARNING", ())
    assert (record.event, record.conversation_id) == ("project_files_conversation_gone", str(conv))


@posix_only
@pytest.mark.parametrize("made_again", [False, True])
def test_a_conversation_deleted_once_its_copy_is_done_gets_no_record(project, monkeypatch, made_again):
    """The delete lands after the copy and before its record is written.
    The folder at the conversation's name is gone, or is a new one another
    staging made, so the copy writes no record and leaves the new folder
    as it found it."""
    _write(project, "a.md", "a")
    conv = uuid4()
    conversation = project / "conversations" / str(conv)
    real_delete_removed = project_file_copy._CopyFolders.delete_removed

    def then_deleted(self, *, budget):
        real_delete_removed(self, budget=budget)
        shutil.rmtree(conversation)
        if made_again:
            conversation.mkdir()

    monkeypatch.setattr(project_file_copy._CopyFolders, "delete_removed", then_deleted)

    result = _stage(project, conv)

    assert result.outcome is ProjectFilesOutcome.CONVERSATION_GONE
    assert not project_file_copy.copy_record_path(SCOPE, conv).exists()
    if made_again:
        assert list(conversation.iterdir()) == []
    else:
        assert not conversation.exists()


@posix_only
@pytest.mark.parametrize("route", ["move", "rename by project name", "rename by project id"])
def test_a_moved_conversation_leaves_no_copy_in_the_project_it_left(org_deployment, monkeypatch, route):
    """Its turns now copy the new project's files into the new project's
    folder, so the copy in the old one goes with the move, on the file
    store's threads whichever way the move names the project."""
    client = TestClient(_app())
    left, left_path, conv = _project_with_a_conversation(client, "left")
    _write(left_path, "chapter-1.md", "The porter kept the keys.")
    _stage(left_path, conv)
    assert _copied(left_path, conv) == {"chapter-1.md": "The porter kept the keys."}
    joined = client.post("/api/v1/projects/", json={"name": f"joined-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert joined.status_code == 201, joined.text
    real_remove = project_file_copy.remove_project_files_copy
    removed_on_efs: list[bool] = []

    def remove(*args, **kwargs):
        removed_on_efs.append(on_efs_thread())
        return real_remove(*args, **kwargs)

    monkeypatch.setattr(project_file_copy, "remove_project_files_copy", remove)

    if route == "move":
        moved = client.post(
            f"/api/v1/conversations/{conv}/move", json={"projectId": joined.json()["id"]}, headers=MEMBER,
        )
    elif route == "rename by project name":
        moved = client.patch(f"/api/v1/conversations/{conv}", json={"project": joined.json()["name"]}, headers=MEMBER)
    else:
        moved = client.patch(f"/api/v1/conversations/{conv}", json={"projectId": joined.json()["id"]}, headers=MEMBER)

    assert moved.status_code == 200, moved.text
    assert removed_on_efs == [True]
    conversation = left_path / "conversations" / str(conv)
    assert not (conversation / "project-files").exists()
    assert not (conversation / project_file_copy.REMOVED_DIRNAME).exists()


# ── The user routes and the copies ──────────────────────────────────────


@posix_only
def test_the_user_routes_never_serve_or_write_a_conversations_copy(org_deployment):
    """The listing hides a conversation's project-files and
    .project-files-removed folders, and every user route answers 404 for
    them too, even to the conversation's owner: a copy a delete has not
    reached yet is never served. The rest of the owner's folder is theirs."""
    client = TestClient(_app())
    created, project, conv = _project_with_a_conversation(client, "routes")
    name = created["name"]
    _write(project, "chapter-1.md", "The porter kept the keys.")
    _stage(project, conv)
    removed = f"conversations/{conv}/{project_file_copy.REMOVED_DIRNAME}/{uuid4().hex}/chapter-1.md"
    _write(project, removed, "set aside")
    _write(project, f"conversations/{conv}/result.csv", "mine")

    for path in (f"conversations/{conv}/project-files/chapter-1.md", removed):
        assert client.get(f"/api/v1/projects/{name}/files/{path}", headers=MEMBER).status_code == 404
        assert client.get(f"/api/v1/projects/{name}/files-raw/{path}", headers=MEMBER).status_code == 404
        written = client.put(f"/api/v1/projects/{name}/files/{path}", json={"content": "x"}, headers=MEMBER)
        assert written.status_code == 404, written.text
        assert client.delete(f"/api/v1/projects/{name}/files/{path}", headers=MEMBER).status_code == 404
    assert _copied(project, conv) == {"chapter-1.md": "The porter kept the keys."}
    assert (project / removed).read_text() == "set aside"
    own = client.get(f"/api/v1/projects/{name}/files-raw/conversations/{conv}/result.csv", headers=MEMBER)
    assert own.status_code == 200 and own.content == b"mine"


@posix_only
def test_a_preview_of_a_page_in_a_conversation_never_serves_its_copy(org_deployment):
    """A preview mount taken on a page in the owner's conversation folder
    serves that folder's files, and not the conversation's copy of the
    project's files."""
    client = TestClient(_app())
    created, project, conv = _project_with_a_conversation(client, "preview")
    _write(project, "chapter-1.md", "The porter kept the keys.")
    _stage(project, conv)
    _write(project, f"conversations/{conv}/report.html", "<html></html>")
    _write(project, f"conversations/{conv}/report.css", "body {}")

    mounted = client.post(
        "/api/v1/projects/preview-mount-file",
        json={"name": created["name"], "path": f"conversations/{conv}/report.html"}, headers=MEMBER,
    )

    assert mounted.status_code == 200, mounted.text
    base = f"/api/v1/projects/preview-asset/{mounted.json()['token']}"
    assert client.get(f"{base}/report.css", headers=MEMBER).status_code == 200
    assert client.get(f"{base}/project-files/chapter-1.md", headers=MEMBER).status_code == 404


# ── What the copy reads ─────────────────────────────────────────────────


@posix_only
@pytest.mark.parametrize("swapped_for", ["a second name for a secret", "a pipe"])
def test_a_source_that_is_not_one_regular_file_once_opened_is_not_copied(project, tmp_path, monkeypatch, swapped_for):
    """The walk saw a regular file with one name, and by the time the copy
    opens it, the name is a second name for another file (such as
    .anton/.env) or a pipe. The check on the opened file refuses it, and the
    copy an earlier turn made goes."""
    _write(project, "notes.md", "first")
    conv = uuid4()
    _stage(project, conv)
    assert _copied(project, conv) == {"notes.md": "first"}
    secret = _write(project, ".anton/.env", "SECRET")
    (project / "notes.md").unlink()
    if swapped_for == "a pipe":
        os.mkfifo(project / "notes.md")
    else:
        os.link(secret, project / "notes.md")
    # What the walk saw: a regular file with one name.
    seen = os.lstat(_write(tmp_path, "as-walked.md", "first"))
    real_lstat = project_file_view.dir_lstat
    monkeypatch.setattr(
        project_file_view, "dir_lstat",
        lambda directory, name: seen if name == "notes.md" else real_lstat(directory, name),
    )

    result = _stage(project, conv)

    assert result.outcome is ProjectFilesOutcome.STAGED
    assert _copied(project, conv) == {}


# ── The not-copied list and anton's strict reader ───────────────────────


def test_the_not_copied_models_keep_to_what_anton_reads():
    """anton refuses a key it does not know, requires every key but
    copy_failed, and accepts these reasons alone. A new key or reason ships
    in anton first."""
    listed = json.loads(
        NotCopiedFiles(files=[NotCopiedFile(path="a.csv", size=1, reason=NotCopiedReason.TOO_LARGE)])
        .model_dump_json()
    )

    assert set(listed) == ANTON_LIST_KEYS
    assert set(listed["files"][0]) == ANTON_FILE_KEYS
    assert set(json.loads(NotCopiedFiles().model_dump_json())) == ANTON_LIST_KEYS
    assert {reason.value for reason in NotCopiedReason} == ANTON_REASONS


@posix_only
def test_every_list_a_copy_writes_holds_every_key_anton_requires(project, monkeypatch):
    """The list a copy writes, when it leaves files out and when it fails,
    holds every key, defaults included, and reasons anton accepts."""
    _write(project, "big.csv", "x" * 2000)
    _write(project, "small.csv", "x")
    conv = uuid4()

    _stage(project, conv)
    left_out = json.loads((_copy_of(project, conv) / NOT_COPIED_FILENAME).read_bytes())

    assert set(left_out) == ANTON_LIST_KEYS
    assert [set(entry) for entry in left_out["files"]] == [ANTON_FILE_KEYS]
    assert {entry["reason"] for entry in left_out["files"]} <= ANTON_REASONS

    def unreadable(*args, **kwargs):
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(project_file_copy, "_copy_file", unreadable)
    failing = uuid4()
    failed = _stage(project, failing)
    failed_list = json.loads((_copy_of(project, failing) / NOT_COPIED_FILENAME).read_bytes())

    assert failed.outcome is ProjectFilesOutcome.FAILED
    assert set(failed_list) == ANTON_LIST_KEYS and failed_list["copy_failed"] is True
