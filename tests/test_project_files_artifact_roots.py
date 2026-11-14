"""The project files PUT and the artifact roots it writes into.

A PUT that writes into an artifact folder records that the folder's root
holds one (held_artifact_roots.hold_root), so the artifacts list opens that
root once the held rows' backfill has finished: the project's shared root,
or the caller's own conversation's legacy root. A conversation workspace
counts only under its canonical name, str(conversation_id): another
spelling of the id is no artifact root, and in org tenancy the PUT refuses
to write there.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event
from sqlmodel import Session, select

from cowork.api.v1.endpoints import project_files
from cowork.models.conversation import Conversation
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.services.artifact_roots import workspace_conversation_id
from cowork.services.held_artifact_roots import HeldRoot
from test_org_list_scaling import (  # noqa: F401  (fixtures)
    ALICE,
    BOB,
    _bulk_paths,
    _drop_held_sentinel,
    _engine,
    _headers,
    _project_id,
    _write_artifact,
    client,
    held_roots,
    org,
)

pytestmark = pytest.mark.usefixtures("granted_product_permissions")

NOT_FOUND = {"detail": "File not found"}


def _project(api, org_id: str) -> Path:
    [path] = _bulk_paths(api, org_id, 0, 1)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _conversation(org_id: str, project_path: Path, owner: str) -> UUID:
    with Session(_engine()) as raw:
        conversation = Conversation(
            topic="task", project_id=_project_id(org_id, project_path), org_id=org_id, created_by=owner
        )
        raw.add(conversation)
        raw.commit()
        return conversation.id


def _held(org_id: str) -> set[HeldRoot]:
    with Session(_engine()) as raw:
        rows = raw.exec(select(HeldArtifactRoot).where(HeldArtifactRoot.org_id == org_id)).all()
        return {HeldRoot(row.project_id, row.conversation_id) for row in rows}


def _put(api, org_id: str, user: str, project_path: Path, rel: str, content: str):
    return api.put(
        f"/api/v1/projects/{project_path.name}/files/{rel}", json={"content": content}, headers=_headers(org_id, user),
    )


def _put_artifact(api, org_id: str, user: str, project_path: Path, root: str, slug: str) -> None:
    """An artifact folder made by hand through the PUT: its metadata.json,
    then its page, below the root at ``root``."""
    files = {
        "metadata.json": '{"slug": "%s", "name": "%s", "type": "html-app"}' % (slug, slug),
        "index.html": "<html></html>",
    }
    for name, content in files.items():
        response = _put(api, org_id, user, project_path, f"{root}/{slug}/{name}", content)
        assert response.status_code == 200, response.text


def _slugs(api, org_id: str, user: str, query: str = "") -> list[str]:
    response = api.get(f"/api/v1/artifacts/{query}", headers=_headers(org_id, user))
    assert response.status_code == 200, response.text
    return [card["slug"] for card in response.json()]


def test_a_put_into_a_projects_artifacts_root_holds_it_so_the_list_shows_the_folder(
    client, org, held_roots  # noqa: F811
):
    """The backfill found nothing to hold, so without the PUT's row the
    list, which trusts the rows from then on, never opens the root."""
    path = _project(client, org)
    held_roots()

    _put_artifact(client, org, ALICE, path, ".anton/artifacts", "handmade")

    assert _held(org) == {HeldRoot(_project_id(org, path), None)}
    assert _slugs(client, org, BOB) == ["handmade"]


def test_a_put_into_the_owners_conversation_workspace_holds_its_legacy_root(client, org, held_roots):  # noqa: F811
    """After the backfill only the PUT and a comparison side's carry record
    a legacy root, so without the PUT's row nothing would ever list the
    folder."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE)
    held_roots()

    _put_artifact(client, org, ALICE, path, f"conversations/{conversation_id}/.anton/artifacts", "old-chart")

    project_id = _project_id(org, path)
    assert _held(org) == {HeldRoot(project_id, conversation_id)}
    assert _slugs(client, org, ALICE, f"?project_id={project_id}") == ["old-chart"]
    # Still the owner's alone.
    assert _slugs(client, org, BOB, f"?project_id={project_id}") == []


@pytest.mark.parametrize(
    "rel",
    [
        "notes.txt",
        ".anton/artifacts/loose.txt",
        ".anton/artifacts/.hidden/metadata.json",
        "conversations/{conversation_id}/notes.txt",
        "conversations/not-a-conversation/.anton/artifacts/chart/metadata.json",
    ],
    ids=["project-file", "file-in-the-root", "dot-folder", "workspace-file", "no-conversations-workspace"],
)
def test_a_put_outside_an_artifact_folder_holds_no_root(client, org, rel):  # noqa: F811
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE)

    response = _put(client, org, ALICE, path, rel.format(conversation_id=conversation_id), "{}")

    assert response.status_code == 200, response.text
    assert _held(org) == set()


@contextmanager
def _held_root_inserts_fail() -> Iterator[None]:
    """Every INSERT into held_artifact_roots fails in the block, as on a lock
    timeout: hold_root adds its row and its commit raises an error it does
    not catch, which leaves its session to be rolled back."""

    def locked(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.startswith("INSERT INTO held_artifact_roots"):
            raise sqlite3.OperationalError("database is locked")

    engine = _engine()
    event.listen(engine, "before_cursor_execute", locked)
    try:
        yield
    finally:
        event.remove(engine, "before_cursor_execute", locked)


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_failed_hold_is_logged_and_leaves_the_written_file(client, org, caplog, legacy):  # noqa: F811
    """The file is on the store before the hold runs, so a failed hold fails
    no PUT: it is logged with the folder it left unheld. The PUT's unit
    commits once the PUT returns, so the PUT rolls the failed hold back
    first, or that commit fails too."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = f"conversations/{conversation_id}/.anton/artifacts" if legacy else ".anton/artifacts"

    with _held_root_inserts_fail(), caplog.at_level(logging.ERROR, logger=project_files.logger.name):
        response = _put(client, org, ALICE, path, f"{root}/handmade/index.html", "<html></html>")

    assert response.status_code == 200, response.text
    assert (path / root / "handmade" / "index.html").read_text() == "<html></html>"
    assert _held(org) == set()
    [record] = [
        record for record in caplog.records
        if record.name == project_files.logger.name
        and getattr(record, "event", None) == "held_artifact_root_not_recorded"
    ]
    assert record.levelno == logging.ERROR
    assert record.error_type == "OperationalError"
    assert (record.project_id, record.artifact_slug) == (str(_project_id(org, path)), "handmade")
    # A shared root's line names no conversation (log_context leaves a None out).
    assert getattr(record, "conversation_id", None) == (str(conversation_id) if legacy else None)


@pytest.mark.usefixtures("cleanup_tmp_projects")
def test_a_local_put_into_another_spelling_of_a_workspace_is_written(tmp_path, monkeypatch):
    """Only org tenancy refuses it: elsewhere no artifacts list reads a
    conversation workspace, so a PUT below another spelling of an id writes
    the file."""
    from fastapi.testclient import TestClient

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    try:
        path = tmp_path / f"desktop-{uuid4().hex[:8]}"
        path.mkdir()
        with Session(_engine()) as raw:
            raw.add(Project(name=path.name, path=str(path), org_id=None))
            raw.commit()
        rel = f"conversations/{str(uuid4()).upper()}/notes.txt"

        response = TestClient(create_app()).put(f"/api/v1/projects/{path.name}/files/{rel}", json={"content": "kept"})
    finally:
        get_app_settings.cache_clear()

    assert response.status_code == 200, response.text
    assert (path / rel).read_text() == "kept"


_SPELLINGS = [str.upper, lambda name: name.replace("-", ""), lambda name: "{%s}" % name]
_SPELLING_IDS = ["uppercase", "unhyphenated", "braced"]


@pytest.mark.parametrize("spelling", _SPELLINGS, ids=_SPELLING_IDS)
def test_only_the_canonical_spelling_names_a_conversation(spelling):
    conversation_id = uuid4()

    assert workspace_conversation_id(str(conversation_id)) == conversation_id
    assert workspace_conversation_id(spelling(str(conversation_id))) is None
    assert workspace_conversation_id("not-a-conversation") is None


@pytest.mark.parametrize("spelling", _SPELLINGS, ids=_SPELLING_IDS)
def test_another_spelling_of_the_owners_workspace_is_no_artifact_root(client, org, spelling):  # noqa: F811
    """The held rows name a legacy root by str(conversation_id), so a
    workspace under another spelling would show its cards to a walk and lose
    them once the backfill had run. It is no root either way."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE)
    _write_artifact(path / "conversations" / spelling(str(conversation_id)) / ".anton" / "artifacts", "chart")
    # The walk, which reads the workspace names: no held rows are trusted.
    _drop_held_sentinel()

    assert _slugs(client, org, ALICE, f"?project_id={_project_id(org, path)}") == []
    assert _slugs(client, org, ALICE) == []


@pytest.mark.parametrize("spelling", _SPELLINGS, ids=_SPELLING_IDS)
def test_a_put_into_another_spelling_of_the_owners_workspace_is_refused(client, org, spelling):  # noqa: F811
    """Refused as a co-member's workspace is: the same 404 and body, before
    anything is written."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE)
    other = spelling(str(conversation_id))
    rel = ".anton/artifacts/chart/metadata.json"

    refused = _put(client, org, ALICE, path, f"conversations/{other}/{rel}", "{}")
    co_members = _put(client, org, BOB, path, f"conversations/{conversation_id}/{rel}", "{}")

    assert (refused.status_code, refused.json()) == (404, NOT_FOUND)
    assert (co_members.status_code, co_members.json()) == (404, NOT_FOUND)
    assert not (path / "conversations").exists()
    assert _held(org) == set()


@pytest.mark.parametrize("spelling", _SPELLINGS, ids=_SPELLING_IDS)
def test_the_owner_still_reads_and_deletes_a_file_in_another_spelling_of_their_workspace(
    client, org, spelling  # noqa: F811
):
    """Only the PUT is refused there, since only a write can make such a
    folder: the owner still lists, reads and deletes what one already holds,
    and a co-member still gets the 404 a workspace of another member gets."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE)
    rel = f"conversations/{spelling(str(conversation_id))}/notes.txt"
    (path / rel).parent.mkdir(parents=True)
    (path / rel).write_text("kept")
    files_url = f"/api/v1/projects/{path.name}/files"

    def listed(user: str) -> list[str]:
        response = client.get(files_url, headers=_headers(org, user))
        assert response.status_code == 200, response.text
        return [file["path"] for file in response.json()["files"]]

    assert rel in listed(ALICE)
    assert rel not in listed(BOB)
    read = client.get(f"{files_url}/{rel}", headers=_headers(org, ALICE))
    assert (read.status_code, read.json()["content"]) == (200, "kept")
    co_members = client.get(f"{files_url}/{rel}", headers=_headers(org, BOB))
    assert (co_members.status_code, co_members.json()) == (404, NOT_FOUND)

    deleted = client.delete(f"{files_url}/{rel}", headers=_headers(org, ALICE))

    assert deleted.status_code == 200, deleted.text
    assert not (path / rel).exists()
