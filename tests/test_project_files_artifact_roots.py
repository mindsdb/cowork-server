"""The project files PUT and DELETE and the artifact roots they write into.

A PUT that writes into an artifact folder records that the folder's root
holds one (held_artifact_roots.hold_root), so the artifacts list opens that
root once the held rows' backfill has finished, and the folder's sort time,
the modification time of its metadata.json (artifact_sort_times), so the
list counts the folder among the newest once the sort times' backfill has
finished: in the project's shared root, or in the caller's own
conversation's legacy root. A failed row of a shared root asks for the
project's location check, whose listing records both. A DELETE of a
folder's metadata.json drops that folder's sort time, in either root, then
reads the folder's time again, so a PUT that wrote the file back while the
DELETE ran keeps its row. A
conversation workspace counts only under its canonical name,
str(conversation_id): another spelling of the id is no artifact root, and in
org tenancy the PUT refuses to write there.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event
from sqlmodel import Session, select

from cowork.api.v1.endpoints import project_files
from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.models.conversation import Conversation
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.services import artifact_autopublish as ap
from cowork.services.artifact_roots import workspace_conversation_id
from cowork.services.artifact_sort_times import SortTime
from cowork.services.held_artifact_roots import HeldRoot
from test_org_list_scaling import (  # noqa: F401  (fixtures)
    ALICE,
    BOB,
    _bulk_paths,
    _drop_sentinels,
    _engine,
    _headers,
    _project_id,
    _write_artifact,
    client,
    held_roots,
    org,
)
from tests._artifact_rows import held_roots_of, no_self_heals_left  # noqa: F401  (no_self_heals_left is a fixture)
from tests._logging_assertions import assert_log_fields

pytestmark = pytest.mark.usefixtures("granted_product_permissions")

NOT_FOUND = {"detail": "File not found"}
METADATA = '{"slug": "%s", "name": "%s", "type": "html-app"}'


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


def _sort_times(org_id: str) -> set[SortTime]:
    with Session(_engine()) as raw:
        rows = raw.exec(select(ArtifactSortTime).where(ArtifactSortTime.org_id == org_id)).all()
        return {SortTime(row.project_id, row.conversation_id, row.slug, row.sort_time) for row in rows}


def _artifacts_root(conversation_id: UUID | None) -> str:
    """The project-relative artifacts root: the shared one, or the legacy
    root of `conversation_id`'s workspace."""
    if conversation_id is None:
        return ".anton/artifacts"
    return f"conversations/{conversation_id}/.anton/artifacts"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _requested_location_checks() -> set[tuple[Path, str]]:
    """The root and project of each location check asked for, and not
    whether its publish was asked for too."""
    return {(heal.artifacts_base, heal.project_id) for heal in ap._self_heals_requested}


def _run_requested_location_checks() -> None:
    async def run() -> None:
        await asyncio.gather(*ap.start_due_self_heals())

    asyncio.run(run())


def _put(api, org_id: str, user: str, project_path: Path, rel: str, content: str):
    return api.put(
        f"/api/v1/projects/{project_path.name}/files/{rel}", json={"content": content}, headers=_headers(org_id, user),
    )


def _put_artifact(api, org_id: str, user: str, project_path: Path, root: str, slug: str) -> None:
    """An artifact folder made by hand through the PUT: its metadata.json,
    then its page, below the root at ``root``."""
    files = {"metadata.json": METADATA % (slug, slug), "index.html": "<html></html>"}
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
    """The backfills found nothing to hold or time, so without the PUT's
    rows the list, which trusts the rows from then on, never opens the root
    or shows the folder."""
    path = _project(client, org)
    held_roots()

    _put_artifact(client, org, ALICE, path, ".anton/artifacts", "handmade")

    project_id = _project_id(org, path)
    assert held_roots_of(org) == {HeldRoot(project_id, None)}
    metadata = path / ".anton" / "artifacts" / "handmade" / "metadata.json"
    assert _sort_times(org) == {SortTime(project_id, None, "handmade", _mtime(metadata))}
    assert _slugs(client, org, BOB) == ["handmade"]


def test_a_put_into_the_owners_conversation_workspace_holds_its_legacy_root(client, org, held_roots):  # noqa: F811
    """After the backfills only the PUT and a comparison side's carry record
    a legacy root and its folders, so without the PUT's rows nothing would
    ever list the folder."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE)
    held_roots()

    _put_artifact(client, org, ALICE, path, _artifacts_root(conversation_id), "old-chart")

    project_id = _project_id(org, path)
    assert held_roots_of(org) == {HeldRoot(project_id, conversation_id)}
    metadata = path / _artifacts_root(conversation_id) / "old-chart" / "metadata.json"
    assert _sort_times(org) == {SortTime(project_id, conversation_id, "old-chart", _mtime(metadata))}
    assert _slugs(client, org, ALICE, f"?project_id={project_id}") == ["old-chart"]
    # Still the owner's alone.
    assert _slugs(client, org, BOB, f"?project_id={project_id}") == []


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_folders_time_is_recorded_by_the_put_that_writes_its_metadata_json(
    client, org, held_roots, owned_logger, legacy  # noqa: F811
):
    """The time is the modification time of the folder's metadata.json. A
    PUT of another file into a new folder finds none yet, so it records no
    time, and the PUT of metadata.json records it. A missing metadata.json
    is no failed read: neither PUT logs a line or asks for the project's
    location check, whose listing reads every folder of the project."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = _artifacts_root(conversation_id)
    project_id = _project_id(org, path)
    held_roots()
    logged = owned_logger(project_files.logger.name, level=logging.WARNING)

    page = _put(client, org, ALICE, path, f"{root}/chart/index.html", "<html></html>")

    assert page.status_code == 200, page.text
    assert held_roots_of(org) == {HeldRoot(project_id, conversation_id)}
    assert _sort_times(org) == set()

    metadata = _put(client, org, ALICE, path, f"{root}/chart/metadata.json", METADATA % ("chart", "chart"))

    assert metadata.status_code == 200, metadata.text
    assert _sort_times(org) == {
        SortTime(project_id, conversation_id, "chart", _mtime(path / root / "chart" / "metadata.json"))
    }
    assert logged.output() == ""
    assert _requested_location_checks() == set()


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_put_of_a_metadata_json_that_does_not_parse_records_no_time(
    client, org, held_roots, owned_logger, legacy  # noqa: F811
):
    """A metadata.json that does not parse holds no id and builds no card,
    so the PUT that
    writes it holds the root but records no time for its folder, as the
    self-heal records none. Nothing failed, so nothing is logged and no
    location check is asked for."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = _artifacts_root(conversation_id)
    project_id = _project_id(org, path)
    held_roots()
    logged = owned_logger(project_files.logger.name, level=logging.WARNING)

    response = _put(client, org, ALICE, path, f"{root}/scratch/metadata.json", "{")

    assert response.status_code == 200, response.text
    assert held_roots_of(org) == {HeldRoot(project_id, conversation_id)}
    assert _sort_times(org) == set()
    assert logged.output() == ""
    assert _requested_location_checks() == set()


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_later_put_of_metadata_json_moves_its_folder_to_the_top_of_the_list(
    client, org, held_roots, owned_logger, legacy  # noqa: F811
):
    """The backfill times both folders from their metadata.json. A PUT
    rewrites the older folder's metadata.json now, and its row takes that
    newer time, so the list shows that folder first. The PUT logs no line
    and asks for no location check."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = _artifacts_root(conversation_id)
    for slug, mtime in (("first", 1_000_000.0), ("second", 2_000_000.0)):
        _write_artifact(path / root, slug)
        os.utime(path / root / slug / "metadata.json", (mtime, mtime))
    held_roots()
    query = f"?project_id={_project_id(org, path)}"
    assert _slugs(client, org, ALICE, query) == ["second", "first"]
    logged = owned_logger(project_files.logger.name, level=logging.WARNING)

    response = _put(client, org, ALICE, path, f"{root}/first/metadata.json", METADATA % ("first", "First"))

    assert response.status_code == 200, response.text
    assert _slugs(client, org, ALICE, query) == ["first", "second"]
    project_id = _project_id(org, path)
    assert _sort_times(org) == {
        SortTime(project_id, conversation_id, "first", _mtime(path / root / "first" / "metadata.json")),
        SortTime(project_id, conversation_id, "second", 2_000_000.0),
    }
    assert logged.output() == ""
    assert _requested_location_checks() == set()


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
    assert held_roots_of(org) == set()
    assert _sort_times(org) == set()


@contextmanager
def _statements_fail(start: str) -> Iterator[None]:
    """Every statement that starts with `start` fails in the block, as on a
    lock timeout: the writer adds or deletes its row and its commit raises
    an error it does not catch, which leaves its session to be rolled
    back."""

    def locked(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.startswith(start):
            raise sqlite3.OperationalError("database is locked")

    engine = _engine()
    event.listen(engine, "before_cursor_execute", locked)
    try:
        yield
    finally:
        event.remove(engine, "before_cursor_execute", locked)


def _held_root_inserts_fail():
    return _statements_fail("INSERT INTO held_artifact_roots")


def _sort_time_inserts_fail():
    return _statements_fail("INSERT INTO artifact_sort_times")


def _sort_time_deletes_fail():
    return _statements_fail("DELETE FROM artifact_sort_times")


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_failed_hold_is_logged_and_leaves_the_written_file(client, org, caplog, legacy):  # noqa: F811
    """The file is on the store before the hold runs, so a failed hold fails
    no PUT: it is logged with the folder it left unheld. The PUT's unit
    commits once the PUT returns, so the PUT rolls the failed hold back
    first, or that commit fails too. The time has its own try, so it is
    still recorded. A shared root asks for the project's location check,
    whose listing holds the root; a legacy root has no self-heal."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = _artifacts_root(conversation_id)
    project_id = _project_id(org, path)
    _write_artifact(path / root, "handmade")

    with _held_root_inserts_fail(), caplog.at_level(logging.ERROR, logger=project_files.logger.name):
        response = _put(client, org, ALICE, path, f"{root}/handmade/index.html", "<p>edited</p>")

    assert response.status_code == 200, response.text
    assert (path / root / "handmade" / "index.html").read_text() == "<p>edited</p>"
    assert held_roots_of(org) == set()
    metadata = path / root / "handmade" / "metadata.json"
    assert _sort_times(org) == {SortTime(project_id, conversation_id, "handmade", _mtime(metadata))}
    [record] = [
        record for record in caplog.records
        if record.name == project_files.logger.name
        and getattr(record, "event", None) == "held_artifact_root_not_recorded"
    ]
    assert record.levelno == logging.ERROR
    assert record.error_type == "OperationalError"
    assert (record.project_id, record.artifact_slug) == (str(project_id), "handmade")
    # A shared root's line names no conversation (log_context leaves a None out).
    assert getattr(record, "conversation_id", None) == (str(conversation_id) if legacy else None)
    if legacy:
        assert _requested_location_checks() == set()
        return
    assert _requested_location_checks() == {(path / ".anton" / "artifacts", str(project_id))}

    _run_requested_location_checks()

    assert held_roots_of(org) == {HeldRoot(project_id, None)}


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_failed_sort_time_is_logged_and_a_shared_root_asks_for_its_location_check(
    client, org, owned_logger, legacy  # noqa: F811
):
    """The hold is written and the time is not: the PUT still succeeds, and
    logs the folder at ERROR. A shared root asks for the project's location
    check, whose listing records the time of every folder it reads. A legacy
    root has no self-heal, so its time waits for the next PUT into the
    folder."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = _artifacts_root(conversation_id)
    project_id = _project_id(org, path)
    logged = owned_logger(project_files.logger.name, level=logging.ERROR)

    with _sort_time_inserts_fail():
        response = _put(client, org, ALICE, path, f"{root}/handmade/metadata.json", METADATA % ("handmade", "h"))

    assert response.status_code == 200, response.text
    assert held_roots_of(org) == {HeldRoot(project_id, conversation_id)}
    assert _sort_times(org) == set()
    fields = {"conversation_id": str(conversation_id)} if legacy else {}
    assert_log_fields(
        logged.output(), log_level="ERROR", event="artifact_sort_time_not_recorded", error_type="OperationalError",
        project_id=str(project_id), artifact_slug="handmade", **fields,
    )
    metadata = path / root / "handmade" / "metadata.json"
    if legacy:
        assert _requested_location_checks() == set()
        response = _put(client, org, ALICE, path, f"{root}/handmade/index.html", "<html></html>")
        assert response.status_code == 200, response.text
    else:
        assert _requested_location_checks() == {(path / ".anton" / "artifacts", str(project_id))}
        _run_requested_location_checks()

    assert _sort_times(org) == {SortTime(project_id, conversation_id, "handmade", _mtime(metadata))}


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_folder_whose_metadata_json_the_put_cannot_read_is_logged(
    client, org, owned_logger, legacy  # noqa: F811
):
    """The read of the folder's time fails with an error that leaves unknown
    what the folder holds (sort_times_below's unreadable). The folder gets
    no time, and the PUT logs it at WARNING with the event a turn end's
    read logs. A shared root asks for the project's location check, whose
    listing reads the folder again; a legacy root has no self-heal."""
    from cowork.services import artifact_sort_times

    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = _artifacts_root(conversation_id)
    project_id = _project_id(org, path)
    logged = owned_logger(project_files.logger.name, level=logging.WARNING)

    def unreadable(*, base, slugs):
        return artifact_sort_times.SortTimesRead(times={}, unreadable=frozenset(slugs))

    with patch.object(artifact_sort_times, "sort_times_below", unreadable):
        response = _put(client, org, ALICE, path, f"{root}/handmade/metadata.json", METADATA % ("handmade", "h"))

    assert response.status_code == 200, response.text
    assert held_roots_of(org) == {HeldRoot(project_id, conversation_id)}
    assert _sort_times(org) == set()
    fields = {"conversation_id": str(conversation_id)} if legacy else {}
    assert_log_fields(
        logged.output(), log_level="WARNING", event="artifact_sort_times_unreadable", count=1,
        project_id=str(project_id), artifact_slug="handmade", **fields,
    )
    if legacy:
        assert _requested_location_checks() == set()
    else:
        assert _requested_location_checks() == {(path / ".anton" / "artifacts", str(project_id))}


def _delete(api, org_id: str, user: str, project_path: Path, rel: str):
    return api.delete(f"/api/v1/projects/{project_path.name}/files/{rel}", headers=_headers(org_id, user))


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_delete_of_a_folders_metadata_json_drops_its_sort_time(
    client, org, held_roots, owned_logger, legacy  # noqa: F811
):
    """Without its metadata.json the folder shows no card, so its row would
    only cost the list a failed open among its newest rows, and no self-heal
    lists a legacy root to drop it. The DELETE drops that folder's row and
    leaves the others, and logs no line."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = _artifacts_root(conversation_id)
    for slug, mtime in (("kept", 1_000_000.0), ("unlisted", 2_000_000.0)):
        _write_artifact(path / root, slug)
        os.utime(path / root / slug / "metadata.json", (mtime, mtime))
    held_roots()
    project_id = _project_id(org, path)
    logged = owned_logger(project_files.logger.name, level=logging.WARNING)

    response = _delete(client, org, ALICE, path, f"{root}/unlisted/metadata.json")

    assert response.status_code == 200, response.text
    assert not (path / root / "unlisted" / "metadata.json").exists()
    assert _sort_times(org) == {SortTime(project_id, conversation_id, "kept", 1_000_000.0)}
    assert _slugs(client, org, ALICE, f"?project_id={project_id}") == ["kept"]
    assert logged.output() == ""


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_put_of_metadata_json_during_the_delete_keeps_its_folders_sort_time(
    client, org, held_roots, owned_logger, legacy  # noqa: F811
):
    """A PUT of the same metadata.json writes it again and records its row
    after the DELETE's unlink and before its drop. The DELETE reads the
    folder's time again after the drop, so the row of the live folder comes
    back and the list still shows its card; a legacy root has no self-heal
    that would write it back."""
    from cowork.services import artifact_sort_times

    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = _artifacts_root(conversation_id)
    _write_artifact(path / root, "chart")
    metadata = path / root / "chart" / "metadata.json"
    os.utime(metadata, (1_000_000.0, 1_000_000.0))
    held_roots()
    project_id = _project_id(org, path)
    logged = owned_logger(project_files.logger.name, level=logging.WARNING)
    forget_sort_time = artifact_sort_times.forget_sort_time

    def put_lands_before_the_drop(session, *, location):
        metadata.write_text(METADATA % ("chart", "Chart"))
        artifact_sort_times.record_sort_times(
            session, project_id=location.project_id, times={"chart": _mtime(metadata)},
            conversation_id=location.conversation_id,
        )
        forget_sort_time(session, location=location)

    with patch.object(artifact_sort_times, "forget_sort_time", put_lands_before_the_drop):
        response = _delete(client, org, ALICE, path, f"{root}/chart/metadata.json")

    assert response.status_code == 200, response.text
    assert _sort_times(org) == {SortTime(project_id, conversation_id, "chart", _mtime(metadata))}
    assert _slugs(client, org, ALICE, f"?project_id={project_id}") == ["chart"]
    assert logged.output() == ""


@pytest.mark.parametrize(
    "rel",
    [
        "notes.txt",
        ".anton/artifacts/loose.json",
        ".anton/artifacts/chart/index.html",
        ".anton/artifacts/chart/assets/metadata.json",
    ],
    ids=["project-file", "file-in-the-root", "page-of-the-folder", "nested-metadata-json"],
)
def test_a_delete_of_a_file_that_is_no_folders_metadata_json_drops_no_sort_time(
    client, org, held_roots, rel  # noqa: F811
):
    """Only a folder's own metadata.json decides whether the list shows its
    card, so the DELETE of any other file leaves every row as it was."""
    path = _project(client, org)
    _write_artifact(path / ".anton" / "artifacts", "chart")
    (path / rel).parent.mkdir(parents=True, exist_ok=True)
    (path / rel).write_text("{}")
    held_roots()
    before = _sort_times(org)
    assert {row.slug for row in before} == {"chart"}

    response = _delete(client, org, ALICE, path, rel)

    assert response.status_code == 200, response.text
    assert not (path / rel).exists()
    assert _sort_times(org) == before


def test_a_delete_of_an_artifact_folder_is_refused_and_keeps_its_sort_time(client, org, held_roots):  # noqa: F811
    """The DELETE never removes a folder, so a folder keeps its row until
    its metadata.json goes."""
    path = _project(client, org)
    _write_artifact(path / ".anton" / "artifacts", "chart")
    held_roots()
    before = _sort_times(org)

    response = _delete(client, org, ALICE, path, ".anton/artifacts/chart")

    assert (response.status_code, response.json()) == (400, {"detail": "Path is a directory"})
    assert (path / ".anton" / "artifacts" / "chart" / "metadata.json").is_file()
    assert _sort_times(org) == before


@pytest.mark.parametrize("legacy", [False, True], ids=["shared-root", "legacy-root"])
def test_a_failed_drop_is_logged_and_the_delete_still_succeeds(
    client, org, held_roots, owned_logger, legacy  # noqa: F811
):
    """The file is gone before the drop runs, so a failed drop fails no
    DELETE: it is rolled back and logged at WARNING with the folder whose
    row it left."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE) if legacy else None
    root = _artifacts_root(conversation_id)
    _write_artifact(path / root, "unlisted")
    held_roots()
    project_id = _project_id(org, path)
    logged = owned_logger(project_files.logger.name, level=logging.WARNING)

    with _sort_time_deletes_fail():
        response = _delete(client, org, ALICE, path, f"{root}/unlisted/metadata.json")

    assert response.status_code == 200, response.text
    assert not (path / root / "unlisted" / "metadata.json").exists()
    assert {(row.conversation_id, row.slug) for row in _sort_times(org)} == {(conversation_id, "unlisted")}
    fields = {"conversation_id": str(conversation_id)} if legacy else {}
    assert_log_fields(
        logged.output(), log_level="WARNING", event="artifact_sort_time_not_dropped", error_type="OperationalError",
        project_id=str(project_id), artifact_slug="unlisted", **fields,
    )


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


@pytest.mark.usefixtures("cleanup_tmp_projects")
def test_a_local_put_into_an_artifact_folder_records_no_row(tmp_path, monkeypatch, owned_logger):
    """Only org tenancy keeps held-root and sort-time rows: on the desktop
    the list walks every folder, so a PUT there records neither, and never
    reads the folder's time: a read that would fail is never made."""
    from fastapi.testclient import TestClient

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.server import create_app
    from cowork.services import artifact_sort_times

    def never_read(*, base, slugs):
        raise AssertionError("a local PUT read a sort time")

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    try:
        path = tmp_path / f"desktop-{uuid4().hex[:8]}"
        path.mkdir()
        with Session(_engine()) as raw:
            project = Project(name=path.name, path=str(path), org_id=None)
            raw.add(project)
            raw.commit()
            project_id = project.id
        rel = ".anton/artifacts/chart/metadata.json"
        api = TestClient(create_app())
        logged = owned_logger(project_files.logger.name, level=logging.WARNING)

        with patch.object(artifact_sort_times, "sort_times_below", never_read):
            response = api.put(
                f"/api/v1/projects/{path.name}/files/{rel}", json={"content": METADATA % ("chart", "chart")}
            )
    finally:
        get_app_settings.cache_clear()

    assert response.status_code == 200, response.text
    assert (path / rel).is_file()
    assert logged.output() == ""
    with Session(_engine()) as raw:
        assert raw.exec(select(HeldArtifactRoot).where(HeldArtifactRoot.project_id == project_id)).all() == []
        assert raw.exec(select(ArtifactSortTime).where(ArtifactSortTime.project_id == project_id)).all() == []
    assert _requested_location_checks() == set()


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
    _drop_sentinels()

    assert _slugs(client, org, ALICE, f"?project_id={_project_id(org, path)}") == []
    assert _slugs(client, org, ALICE) == []


@pytest.mark.parametrize("spelling", _SPELLINGS, ids=_SPELLING_IDS)
def test_a_put_into_another_spelling_of_the_owners_workspace_is_refused(client, org, spelling):  # noqa: F811
    """Refused as a co-member's workspace is: the same 404 and body, before
    anything is written or recorded."""
    path = _project(client, org)
    conversation_id = _conversation(org, path, ALICE)
    other = spelling(str(conversation_id))
    rel = ".anton/artifacts/chart/metadata.json"

    refused = _put(client, org, ALICE, path, f"conversations/{other}/{rel}", METADATA % ("chart", "chart"))
    co_members = _put(client, org, BOB, path, f"conversations/{conversation_id}/{rel}", METADATA % ("chart", "chart"))

    assert (refused.status_code, refused.json()) == (404, NOT_FOUND)
    assert (co_members.status_code, co_members.json()) == (404, NOT_FOUND)
    assert not (path / "conversations").exists()
    assert held_roots_of(org) == set()
    assert _sort_times(org) == set()


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
