"""The artifact_sort_times rows each writer leaves, through the code that calls
it rather than below it: a turn end through the remote turn and through the
responses handler, the self-heal's location check, the heal sweep, a task's
move and a delete.

A turn end times every folder it created, touched or its tools named as
edited, so a metadata.json rewrite alone moves a card, and a folder whose
time it could not read has its root held and asks for the project's location
check. A turn end with nothing to record, or in a scope that keeps no rows,
opens no record unit. The self-heal reads the project's path and the rows
before it lists, and its record reads the path again: a heal whose path is no
longer the project's, after a rename, records nothing. It keeps every row
when its listing found the root gone, as while a rename moves the folder, or
a folder's row when it could not read that folder, and gives each of its
writes its own try.
A failed write, or a refused unit, asks for the location check that writes
it again, and the heal sweep keeps such a project for its retries.
"""
from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlmodel import Session

import cowork.handlers.responses as responses_mod
import cowork.turnqueue.remote_turn as remote_turn_mod
from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.db import units
from cowork.db.scoped import TenantScope
from cowork.handlers.responses import ResponsesHandler, _RemoteArtifactsDir
from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.models.project import Project
from cowork.services import artifact_autopublish as ap
from cowork.services import artifact_heal_sweep as sweep
from cowork.services import (
    artifact_identity,
    artifact_locations,
    artifact_ownership,
    artifact_sort_times,
    held_artifact_roots,
    task_objects,
)
from cowork.services.artifact_locations import Location, LocationRow, find_location, record_checked_folders
from cowork.services.artifact_roots import project_artifacts_base
from cowork.services.artifact_sort_times import record_sort_times
from cowork.services.held_artifact_roots import HeldRoot
from cowork.services.task_objects import ListedLocationsUnchecked
from cowork.turnqueue.remote_turn import remote_turn_events
from test_artifact_heal_sweep import _talked_in
from test_artifact_locations import _write
from test_artifact_ownership import _engine, make_conversation, make_project, project_root, scoped, write_artifact
from test_artifact_permissions import no_publish_side_effects  # noqa: F401  (a fixture)
from test_artifact_relocation_ownership import _index, _relocate
from test_org_list_scaling import (  # noqa: F401  (fixtures)
    ALICE,
    _artifacts_at,
    _bulk_paths,
    _headers,
    _alices_conversation,
    _project_id,
    client,
    _read_in_full,
    held_roots,
    org,
)
from test_remote_turn import _drain, _fake_handler
from test_responses_remote_backend import (
    _no_workspace_or_memory_units,
    _RecBuffer,
    _remote_handler,
    _units_on_the_test_database,
)
from tests._artifact_rows import (  # noqa: F401  (no_self_heals_left is a fixture)
    held_roots_of,
    no_self_heals_left,
    sort_times_of,
    timed_artifact,
)
from tests._logging_assertions import assert_log_fields

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

_UNTIMED_EVENT = "artifact_sort_times_unreadable"


def _rewrite_metadata(folder: Path, at: float) -> None:
    """What anton's update_artifact_metadata does: rewrite metadata.json
    alone, here with a new title, leaving every content file as it was."""
    meta = folder / "metadata.json"
    data = json.loads(meta.read_text())
    data["title"] = "a new title"
    meta.write_text(json.dumps(data))
    os.utime(meta, (at, at))


def _location_check(project, org_id: str, user_id: str | None = None) -> ap.SelfHeal:
    return ap.SelfHeal(
        artifacts_base=project_root(project).base,
        scope=TenantScope(org_mode=True, org_id=org_id, user_id=user_id),
        project_id=str(project.id),
    )


def _times_that_fail_for(monkeypatch, *names: str) -> Callable[[], None]:
    """The file server stops answering for the metadata.json of each of
    `names`, as sort_times_below reads it, and only there. Returns what
    makes it answer again."""
    real = artifact_sort_times.dir_child_stat

    def child_stat(root, name, child, **kwargs):
        if name in names:
            raise OSError(errno.EIO, "the file server did not answer")
        return real(root, name, child, **kwargs)

    monkeypatch.setattr(artifact_sort_times, "dir_child_stat", child_stat)
    return lambda: monkeypatch.setattr(artifact_sort_times, "dir_child_stat", real)


# ── the self-heal ────────────────────────────────────────────────────────


async def test_a_heal_asked_for_before_a_rename_keeps_the_projects_rows_and_cards(
    client, org, held_roots,  # noqa: F811
):
    """A turn end asks for its project's self-heal with the project's path.
    The member then renames the project, which moves its folder
    (ProjectService._stage_project_rename). The heal runs later with the
    old path, which is no longer the project's shared root: it drops no row,
    and the list still shows the project's cards, newest first. It asks for
    no further check, since the project never has the old path again."""
    # The second project is Alice's, so she may rename it.
    _, path = _bulk_paths(client, org, 0, 2)
    _artifacts_at(org, path, ALICE, {"chart": 1_000_100.0, "report": 1_000_200.0})
    held_roots()
    project_id = _project_id(org, path)
    before = sort_times_of(org)
    ap.request_turn_end_self_heal(
        artifacts_base=project_artifacts_base(str(path)),
        scope=TenantScope(org_mode=True, org_id=org, user_id=ALICE),
        project_id=str(project_id),
    )

    renamed = client.patch(
        f"/api/v1/projects/{project_id}", json={"name": "renamed-reports"}, headers=_headers(org, ALICE),
    )
    assert renamed.status_code == 200, renamed.text
    assert not path.exists()
    await asyncio.gather(*ap.start_due_self_heals())

    assert set(before) == {(project_id, None, "chart"), (project_id, None, "report")}
    assert sort_times_of(org) == before
    listed = client.get(f"/api/v1/artifacts/?project_id={project_id}", headers=_headers(org, ALICE))
    assert [card["slug"] for card in listed.json()] == ["report", "chart"]
    assert ap._self_heals_requested == set()


def _listings_counted(monkeypatch) -> list[Path]:
    """The roots the self-heal lists from now on (_listed_artifact_ids)."""
    listings: list[Path] = []
    real = ap._listed_artifact_ids

    def listing(*, artifacts_base):
        listings.append(Path(artifacts_base))
        return real(artifacts_base=artifacts_base)

    monkeypatch.setattr(ap, "_listed_artifact_ids", listing)
    return listings


def _move_project(org_id: str, project, new_path: Path) -> None:
    """What a rename does once it commits: the project's folder moves to
    `new_path`, and its row names that path and the name it comes from."""
    os.rename(project.path, new_path)
    with Session(_engine()) as raw:
        row = raw.get(Project, project.id)
        row.name, row.path = new_path.name, str(new_path)
        raw.add(row)
        raw.commit()


def _project_at(org_id: str, path: Path) -> Project:
    """A project created at `path`, as one that takes a renamed project's
    old name is, holding `someone-elses`."""
    project = Project(id=uuid4(), name=path.name, path=str(path), org_id=org_id)
    with Session(_engine()) as raw:
        raw.add(project)
        raw.commit()
        raw.refresh(project)
        raw.expunge(project)
    timed_artifact(project_root(project).base, "someone-elses", 1_000_300.0)
    return project


async def test_a_heal_asked_for_before_a_rename_lists_nothing_once_another_project_has_the_old_path(
    client, org, held_roots, monkeypatch,  # noqa: F811
):
    """The member renames the project after a turn end asked for its
    self-heal, and a new project then takes the old name, so the old path,
    and makes `someone-elses` there. The heal names a path that is no
    longer its project's shared root: it lists nothing and records nothing,
    under either project, and asks for no further check."""
    _, path = _bulk_paths(client, org, 0, 2)
    _artifacts_at(org, path, ALICE, {"chart": 1_000_100.0, "report": 1_000_200.0})
    held_roots()
    project_id = _project_id(org, path)
    ap.request_turn_end_self_heal(
        artifacts_base=project_artifacts_base(str(path)),
        scope=TenantScope(org_mode=True, org_id=org, user_id=ALICE),
        project_id=str(project_id),
    )
    renamed = client.patch(
        f"/api/v1/projects/{project_id}", json={"name": "renamed-reports"}, headers=_headers(org, ALICE),
    )
    assert renamed.status_code == 200, renamed.text
    _project_at(org, path)
    before = (held_roots_of(org), sort_times_of(org))
    listings = _listings_counted(monkeypatch)

    await asyncio.gather(*ap.start_due_self_heals())

    assert listings == []
    assert (held_roots_of(org), sort_times_of(org)) == before
    assert before == (
        {HeldRoot(project_id, None)},
        {(project_id, None, "chart"): 1_000_100.0, (project_id, None, "report"): 1_000_200.0},
    )
    assert ap._self_heals_requested == set()


async def test_a_rename_that_commits_while_the_heal_lists_leaves_the_rows_alone(tmp_path, monkeypatch):
    """The heal read the project's path and its rows, and then, while it
    lists, the rename commits and a new project takes the old path and
    makes `someone-elses` there, which the listing reads. Its record reads
    the project's path again: the listing was not the project's, so it
    records nothing, and the rows of `chart` and `report` stay."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    base = project_root(project).base
    timed_artifact(base, "chart", 1_000_100.0)
    timed_artifact(base, "report", 1_000_200.0)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"chart": 1_000_100.0, "report": 1_000_200.0})
    before = sort_times_of(org_id)
    heal = _location_check(project, org_id)
    real = ap._listed_artifact_ids
    listed: list[set[str]] = []

    def rename_then_list(*, artifacts_base):
        _move_project(org_id, project, Path(project.path).with_name("renamed"))
        _project_at(org_id, Path(project.path))
        listing = real(artifacts_base=artifacts_base)
        listed.append(set(listing.ids))
        return listing

    monkeypatch.setattr(ap, "_listed_artifact_ids", rename_then_list)

    await ap._record_listed_locations(heal)

    assert listed == [{"someone-elses"}]
    assert sort_times_of(org_id) == before
    assert held_roots_of(org_id) == set()
    assert ap._self_heals_requested == set()


@pytest.mark.parametrize("when", ["after_the_listing", "during_the_stat_loop", "during_the_listing_only"])
async def test_a_heal_whose_project_folder_moves_while_it_lists_keeps_the_rows(tmp_path, monkeypatch, when):
    """A rename moves the project's folder before it commits, so the
    project's row still names the path the heal lists
    (projects.ProjectService._stage_project_rename). The folder moves once
    the listing found its folders (`after_the_listing`), so the id reader
    finds the root gone. Or it moves once the listing stat'ed `chart`'s
    metadata.json, so the stat of `report` finds it missing, and is back
    before the id reader opens the root (`during_the_stat_loop`), as when
    the rename fails and is rolled back
    (projects.ProjectService.rollback_project_rename). Or it is gone when
    the listing lists the root and back before the id reader opens it
    (`during_the_listing_only`). Each time the heal could not tell a folder
    gone from moved, so it keeps every row rather than take the folders for
    gone, and asks for no further check."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    base = project_root(project).base
    timed_artifact(base, "chart", 1_000_100.0)
    timed_artifact(base, "report", 1_000_200.0)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"chart": 1_000_100.0, "report": 1_000_200.0})
    before = sort_times_of(org_id)
    moved = Path(project.path).with_name("renamed")
    real = artifact_identity.artifact_folders_below
    real_stat = os.stat

    def stat_then_move(path, *args, **kwargs):
        stated = real_stat(path, *args, **kwargs)
        if Path(path) == base / "chart" / "metadata.json":
            os.rename(project.path, moved)
        return stated

    def listing_while_the_folder_moves(*, base):
        if when == "after_the_listing":
            listing = real(base=base)
            os.rename(project.path, moved)
            return listing
        if when == "during_the_stat_loop":
            monkeypatch.setattr(os, "stat", stat_then_move)
            try:
                listing = real(base=base)
            finally:
                monkeypatch.setattr(os, "stat", real_stat)
        else:
            os.rename(project.path, moved)
            listing = real(base=base)
        os.rename(moved, project.path)
        return listing

    monkeypatch.setattr(artifact_identity, "artifact_folders_below", listing_while_the_folder_moves)

    await ap._record_listed_locations(_location_check(project, org_id))

    assert set(before) == {(project.id, None, "chart"), (project.id, None, "report")}
    assert sort_times_of(org_id) == before
    assert ap._self_heals_requested == set()


async def test_a_row_a_turn_end_writes_while_the_heal_lists_stays(tmp_path, monkeypatch):
    """The heal lists the shared root, then a turn end makes `fresh` and
    writes its row, then the heal records what its listing found. The
    listing never saw `fresh`, but its row was not among those the heal read
    before it listed, so it stays. The row of `gone`, a folder deleted
    outside a turn before the heal began, goes."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    base = project_root(project).base
    timed_artifact(base, "older", 1_000_100.0)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"older": 1_000_100.0, "gone": 1_000_150.0})
    real = ap._listed_artifact_ids

    def listing_then_a_turn_end(*, artifacts_base):
        listed = real(artifacts_base=artifacts_base)
        timed_artifact(base, "fresh", 1_000_200.0)
        with scoped(org_id) as session:
            record_sort_times(session, project_id=project.id, times={"fresh": 1_000_200.0})
        return listed

    monkeypatch.setattr(ap, "_listed_artifact_ids", listing_then_a_turn_end)

    await ap._record_listed_locations(_location_check(project, org_id))

    assert sort_times_of(org_id) == {(project.id, None, "older"): 1_000_100.0, (project.id, None, "fresh"): 1_000_200.0}
    assert ap._self_heals_requested == set()


def _copied_brief(tmp_path) -> tuple[str, object, str]:
    """A project whose `brief` has its location row and its sort time, and
    an agent's `cp -r brief brief-copy` beside it, which no turn end claims.
    Returns the organization, the project and the id both folders hold."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    artifact_id, folder = _write(source, "brief")
    os.utime(folder / "metadata.json", (1_000_100.0, 1_000_100.0))
    _write(source, "brief-copy", artifact_id)
    os.utime(source.base / "brief-copy" / "metadata.json", (1_000_200.0, 1_000_200.0))
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(project.id, None, "brief")])
        record_sort_times(session, project_id=project.id, times={"brief": 1_000_000.0, "gone": 1_000_000.0})
    return org_id, project, artifact_id


_HEAL_WRITE_FAILURES = {
    "hold": "artifact self-heal could not record that the project's artifacts folder holds an artifact",
    "sort_times": "artifact self-heal could not record the sort times of the project's artifacts",
    "locations": "artifact self-heal could not check the locations of the project's listed artifacts",
}


def _a_write_the_database_refuses(session, *, project_id, **kwargs) -> None:
    """A write whose statement the database refuses inside the session: two
    sort-time rows for one folder of the shared root break its unique index
    at the flush. The session then writes nothing more until it is rolled
    back, and on PostgreSQL neither does its transaction."""
    for sort_time in (1.0, 2.0):
        session.add(ArtifactSortTime(project_id=project_id, conversation_id=None, slug="twice", sort_time=sort_time))
    session.flush()


def _a_write_that_raises(session, **kwargs) -> None:
    """A write that raises before it sends any statement."""
    raise RuntimeError("the write was refused")


@pytest.mark.parametrize("refusal", ["raises", "statement"])
@pytest.mark.parametrize("failing", list(_HEAL_WRITE_FAILURES))
async def test_a_heal_write_that_fails_skips_none_of_the_others_and_asks_for_the_next_check(
    tmp_path, monkeypatch, caplog, failing, refusal,
):
    """The heal's three writes (_record_listing) each run on their own: a
    hold, a reconcile of the sort times or a check of the locations that
    raises, or whose statement the database refuses, is rolled back and
    logged, and the other two still write. The heal then asks for the
    project's next location check, which writes the failed one again."""
    org_id, project, artifact_id = _copied_brief(tmp_path)
    refused = _a_write_that_raises if refusal == "raises" else _a_write_the_database_refuses
    module, name = {
        "hold": (held_artifact_roots, "hold_root"),
        "sort_times": (artifact_sort_times, "reconcile_sort_times"),
        "locations": (artifact_locations, "record_listed_folders"),
    }[failing]
    real = getattr(module, name)
    monkeypatch.setattr(module, name, refused)
    heal = _location_check(project, org_id)

    with caplog.at_level(logging.WARNING, logger=ap.logger.name):
        await ap._record_listed_locations(heal)

    [warning] = [record for record in caplog.records if record.name == ap.logger.name]
    assert warning.levelno == logging.WARNING
    if refusal == "raises":
        assert warning.getMessage() == _HEAL_WRITE_FAILURES[failing]
        assert warning.exc_info is not None
    else:
        # A database error's line keeps only its type and where it was
        # logged (common.logger.DatabaseErrorFilter).
        assert warning.error_type == "IntegrityError"
    assert ap._self_heals_requested == {heal}
    assert ap._self_heal_publishes_requested == set()
    held = {HeldRoot(project.id, None)}
    times = {(project.id, None, "brief"): 1_000_100.0, (project.id, None, "brief-copy"): 1_000_200.0}
    conflicted = LocationRow(Location(project.id, None, "brief"), conflicted=True)
    with scoped(org_id) as session:
        location = find_location(session, artifact_id)
    assert held_roots_of(org_id) == (set() if failing == "hold" else held)
    assert sort_times_of(org_id) == (
        {(project.id, None, "brief"): 1_000_000.0, (project.id, None, "gone"): 1_000_000.0}
        if failing == "sort_times"
        else times
    )
    assert location == (
        LocationRow(Location(project.id, None, "brief"), conflicted=False) if failing == "locations" else conflicted
    )

    monkeypatch.setattr(module, name, real)
    monkeypatch.setattr(ap, "SELF_HEAL_INTERVAL_S", 0.0)
    await asyncio.gather(*ap.start_due_self_heals())

    with scoped(org_id) as session:
        assert find_location(session, artifact_id) == conflicted
    assert (held_roots_of(org_id), sort_times_of(org_id)) == (held, times)


@pytest.mark.parametrize("refused_unit", ["rows", "record"])
async def test_a_heal_whose_database_unit_is_refused_asks_for_the_next_check(
    tmp_path, monkeypatch, caplog, refused_unit,
):
    """The pool refuses one of the heal's two database units: the read of
    the sort-time rows, before the listing, which then never runs; or the
    record of what the listing found. The heal wrote nothing, so it logs that
    it could not check the project and asks for the next location check,
    which writes the rows."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    listings: list[Path] = []
    real_listing = ap._listed_artifact_ids

    def listing(*, artifacts_base):
        listings.append(Path(artifacts_base))
        return real_listing(artifacts_base=artifacts_base)

    monkeypatch.setattr(ap, "_listed_artifact_ids", listing)
    refused = ap._rows_before_the_listing if refused_unit == "rows" else ap._record_listing
    real_run_db = units.run_db

    async def busy_pool(fn, *, scope):
        if getattr(fn, "func", None) is refused:
            raise units.DatabaseBusy("no database unit slot freed within POOL_TIMEOUT")
        return await real_run_db(fn, scope=scope)

    monkeypatch.setattr(units, "run_db", busy_pool)
    heal = _location_check(project, org_id)

    with caplog.at_level(logging.WARNING, logger=ap.logger.name):
        await ap._record_listed_locations(heal)

    # A database error's line keeps only its type and where it was logged
    # (common.logger.DatabaseErrorFilter).
    [warning] = [record for record in caplog.records if record.name == ap.logger.name]
    assert (warning.levelno, warning.error_type) == (logging.WARNING, "DatabaseBusy")
    assert len(listings) == (0 if refused_unit == "rows" else 1)
    assert (held_roots_of(org_id), sort_times_of(org_id)) == (set(), {})
    assert ap._self_heals_requested == {heal}
    assert ap._self_heal_publishes_requested == set()

    monkeypatch.setattr(units, "run_db", real_run_db)
    monkeypatch.setattr(ap, "SELF_HEAL_INTERVAL_S", 0.0)
    await asyncio.gather(*ap.start_due_self_heals())

    assert (held_roots_of(org_id), sort_times_of(org_id)) == (
        {HeldRoot(project.id, None)}, {(project.id, None, "report"): 1_000_100.0},
    )


def _report_unreadable(monkeypatch, base: Path, *, at: str) -> None:
    """The file server stops answering for `base`'s `report` folder with
    EIO: when a listing stats the folder (`at="stat"`), or once it has, when
    the id reader opens its metadata.json (`at="id"`)."""
    if at == "stat":
        folder, real_stat = base / "report", os.stat

        def stat(path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)) and Path(path) == folder:
                raise OSError(errno.EIO, "the file server did not answer")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", stat)
        return
    real_read = artifact_identity.read_full_id

    def read(folder, *args, **kwargs):
        if Path(folder.path).name == "report":
            raise OSError(errno.EIO, "the file server did not answer")
        return real_read(folder, *args, **kwargs)

    monkeypatch.setattr(artifact_identity, "read_full_id", read)


@pytest.mark.parametrize("at", ["stat", "id"])
async def test_a_heal_whose_listing_reads_no_folder_holds_the_root_and_keeps_the_folders_row(
    tmp_path, monkeypatch, at,
):
    """The heal lists a root whose one folder, `report`, it cannot read: it
    cannot stat the folder, or cannot read its id. The listing read no id,
    but the folder may hold an artifact the list should show, so the heal
    holds the root, which no row held, and keeps `report`'s row rather than
    taking it for a folder gone. It asks for the next location check."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    base = project_root(project).base
    timed_artifact(base, "report", 1_000_200.0)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 1_000_100.0})
    _report_unreadable(monkeypatch, base, at=at)
    heal = _location_check(project, org_id)

    await ap._record_listed_locations(heal)

    assert held_roots_of(org_id) == {HeldRoot(project.id, None)}
    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_100.0}
    assert ap._self_heals_requested == {heal}
    assert ap._self_heal_publishes_requested == set()


async def test_a_folder_the_heal_cannot_time_keeps_its_row_and_asks_for_the_next_check(
    tmp_path, monkeypatch, caplog, owned_logger,
):
    """The heal lists `stalled`, which holds an id, but the read of its
    metadata.json fails with EIO. Its row keeps the time it had, rather than
    being taken for a folder with no time and dropped, and the heal logs one
    WARNING and asks for the project's next location check, which times it
    once the file server answers again."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    base = project_root(project).base
    timed_artifact(base, "report", 1_000_200.0)
    timed_artifact(base, "stalled", 1_000_300.0)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"stalled": 1_000_000.0})
    heal = _location_check(project, org_id)
    answer_again = _times_that_fail_for(monkeypatch, "stalled")
    logged = owned_logger(ap.logger.name, level=logging.WARNING)

    await ap._record_listed_locations(heal)

    assert sort_times_of(org_id) == {
        (project.id, None, "report"): 1_000_200.0,
        (project.id, None, "stalled"): 1_000_000.0,
    }
    assert ap._self_heals_requested == {heal}
    assert ap._self_heal_publishes_requested == set()
    [warning] = [
        record for record in caplog.records
        if record.name == ap.logger.name and getattr(record, "event", None) == _UNTIMED_EVENT
    ]
    assert warning.levelno == logging.WARNING
    assert_log_fields(
        logged.output(), log_level="WARNING", event=_UNTIMED_EVENT, project_id=str(project.id), count=1,
    )

    answer_again()
    monkeypatch.setattr(ap, "SELF_HEAL_INTERVAL_S", 0.0)
    await asyncio.gather(*ap.start_due_self_heals())

    assert sort_times_of(org_id) == {
        (project.id, None, "report"): 1_000_200.0,
        (project.id, None, "stalled"): 1_000_300.0,
    }
    assert ap._self_heals_requested == set()


async def test_the_self_heal_writes_the_time_of_a_folder_with_no_row(tmp_path):
    """A folder no turn end recorded, such as one an old pod made in a
    rollout: the self-heal's location check writes its row."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    ap.request_location_check(_location_check(project, org_id))

    await asyncio.gather(*ap.start_due_self_heals())

    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_100.0}


async def test_the_heal_sweep_writes_the_time_of_a_folder_with_no_row(tmp_path, org_deployment):
    """An active project holds a folder with no row: the sweep's check of it
    writes the row, as a self-heal's does."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    _talked_in(project)

    assert await sweep.run_heal_sweep(project_ids=[project.id]) == sweep.SweepCounts(checked=1, failed=0)

    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_100.0}


@pytest.mark.parametrize("short", ["record", "time"])
async def test_a_sweep_check_that_falls_short_keeps_the_project_for_its_retries(
    tmp_path, monkeypatch, org_deployment, short,
):
    """An active project's sweep check falls short: the database refuses its
    reconcile of the sort times (`record`), or it cannot read `stalled`'s
    time (`time`). The check's request for the next location check lands in
    the sweep's own retries, never in this replica's poll, so the sweep
    counts the project as failed and checks it again."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    base = project_root(project).base
    timed_artifact(base, "report", 1_000_100.0)
    timed_artifact(base, "stalled", 1_000_200.0)
    _talked_in(project)
    if short == "record":
        monkeypatch.setattr(artifact_sort_times, "reconcile_sort_times", _a_write_the_database_refuses)
    else:
        _times_that_fail_for(monkeypatch, "stalled")

    assert await sweep.run_heal_sweep(project_ids=[project.id]) == sweep.SweepCounts(checked=0, failed=1)

    assert {heal.project_id for heal in sweep._retries} == {str(project.id)}
    assert ap._self_heals_requested == set()
    assert held_roots_of(org_id) == {HeldRoot(project.id, None)}


# ── the turn ends ────────────────────────────────────────────────────────


def _remote_turn_end(monkeypatch, *, during_turn, touched: list[str] | None) -> None:
    """A remote turn whose inputs are read and whose end is recorded in real
    units on the test database. `during_turn` runs once the turn's snapshot
    is taken, before the pod's `turn_completed`, which names `touched` as
    the artifacts its tools touched, or names none for None, so the turn end
    lists the project's whole shared root."""
    _fake_handler(monkeypatch, remote_artifacts_context=ResponsesHandler._remote_artifacts_context)
    monkeypatch.setattr(remote_turn_mod, "run_db", units.run_db)
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", AsyncMock(return_value=[]))

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        during_turn()
        yield "turn_completed", {} if touched is None else {"artifacts_touched": touched}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)


async def _run_turn(*, scope: TenantScope, conversation_id) -> None:
    _, failure = await _drain(remote_turn_events(
        scope=scope, conv_id=conversation_id, org_id=scope.org_id, user_id=scope.user_id,
        input_text="hi", model="anton", turn_rows=[],
    ))
    assert failure is None


async def _responses_turn(monkeypatch, *, project, scope: TenantScope, conversation_id, during_turn, touched) -> None:
    """The responses handler's remote turn (ResponsesHandler._produce_remote)
    in `project`, its units real and on the test database, with
    `during_turn` and `touched` as _remote_turn_end takes them."""
    source = project_root(project)
    handler = _remote_handler(monkeypatch, {})
    _units_on_the_test_database(monkeypatch, scope=scope)
    _no_workspace_or_memory_units(monkeypatch)
    monkeypatch.setattr(
        ResponsesHandler, "_remote_artifacts_context",
        staticmethod(lambda session, conv_id: _RemoteArtifactsDir(
            base=source.base, project_id=str(project.id), project_name=project.name, creator=scope.user_id,
        )),
    )
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", AsyncMock(return_value=[]))

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        during_turn()
        yield "turn_delta", {"text": "done"}
        yield "turn_completed", {} if touched is None else {"artifacts_touched": touched}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", replies)
    buffer = _RecBuffer()

    await handler._produce_remote(
        conv_id=conversation_id, input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=buffer,
    )

    assert buffer.frames[-1] == "CLOSE:completed"


class _Producer(NamedTuple):
    """A remote turn end's code, and the ERROR lines it logs when its record
    raised ListedLocationsUnchecked."""

    logger: str
    #: After a full listing: the record may have lost the location check.
    listed_lost: str
    #: After a turn whose pod named its folders: no location check was made.
    edited_lost: str


_PRODUCERS = {
    "remote_turn": _Producer(
        logger=remote_turn_mod.logger.name,
        listed_lost=(
            "Could not check the locations of the artifacts a channel turn found, or record the folder that holds"
            " them or their sort times"
        ),
        edited_lost=(
            "Could not record the sort times of the artifacts a channel turn made or changed, or the folder that"
            " holds them"
        ),
    ),
    "responses": _Producer(
        logger=responses_mod.logger.name,
        listed_lost=(
            "[responses] could not check the locations of the artifacts a remote turn found, or record the folder"
            " that holds them or their sort times"
        ),
        edited_lost=(
            "[responses] could not record the sort times of the artifacts a remote turn made or changed, or the"
            " folder that holds them"
        ),
    ),
}


async def _turn_end(
    monkeypatch, producer: str, *, project, scope: TenantScope, conversation_id, during_turn, touched,
) -> None:
    """One remote turn in `project` through `producer`'s turn end: the turn
    queue's (remote_turn) or the responses handler's (responses)."""
    if producer == "remote_turn":
        _remote_turn_end(monkeypatch, during_turn=during_turn, touched=touched)
        await _run_turn(scope=scope, conversation_id=conversation_id)
    else:
        await _responses_turn(
            monkeypatch, project=project, scope=scope, conversation_id=conversation_id,
            during_turn=during_turn, touched=touched,
        )


def _turn_end_errors(caplog, producer: str) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == _PRODUCERS[producer].logger and record.levelno == logging.ERROR
    ]


async def test_a_turn_whose_only_change_is_a_metadata_rewrite_moves_its_card_to_the_top(
    client, org, held_roots, monkeypatch,  # noqa: F811
):
    """The agent's tool rewrites only `first`'s metadata.json, as anton's
    update_artifact_metadata does, and the pod names `first` as touched.
    No content file moved, so the turn touched nothing, yet a walk sorts by
    metadata.json's time and moves the card to the top. The turn end reads
    the time of every folder its tools named, so the list does too."""
    [path] = _bulk_paths(client, org, 0, 1)
    _artifacts_at(org, path, ALICE, {"first": 1_000_100.0, "second": 1_000_200.0, "third": 1_000_300.0})
    held_roots()
    project_id = _project_id(org, path)
    listed = f"/api/v1/artifacts/?project_id={project_id}"

    def order() -> list[str]:
        return [card["slug"] for card in client.get(listed, headers=_headers(org, ALICE)).json()]

    assert order() == ["third", "second", "first"]
    _remote_turn_end(
        monkeypatch,
        during_turn=lambda: _rewrite_metadata(path / ".anton" / "artifacts" / "first", 1_000_400.0),
        touched=["first"],
    )

    await _run_turn(
        scope=TenantScope(org_mode=True, org_id=org, user_id=ALICE),
        conversation_id=_alices_conversation(org, path),
    )

    assert order() == ["first", "third", "second"]
    assert sort_times_of(org)[(project_id, None, "first")] == 1_000_400.0


@pytest.mark.parametrize("producer", list(_PRODUCERS))
async def test_an_edit_only_turn_end_holds_its_root_and_writes_the_time(tmp_path, monkeypatch, producer):
    """An old pod in a rollout made `report`, and wrote no row. A later turn
    rewrites its metadata.json and its pod names it: the turn created
    nothing and listed nothing, but it timed `report`, so its record holds
    the root and writes the row."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    folder = timed_artifact(project_root(project).base, "report", 1_000_100.0)

    await _turn_end(
        monkeypatch, producer, project=project, scope=TenantScope(org_mode=True, org_id=org_id, user_id=creator),
        conversation_id=conversation_id, during_turn=lambda: _rewrite_metadata(folder, 1_000_200.0),
        touched=["report"],
    )

    assert held_roots_of(org_id) == {HeldRoot(project.id, None)}
    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_200.0}


def _idless_folder(base: Path, slug: str, at: float) -> Path:
    """A folder whose metadata.json does not parse, last changed at `at`:
    it holds no id, so the list builds no card for it."""
    folder = base / slug
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text("{")
    os.utime(folder / "metadata.json", (at, at))
    return folder


@pytest.mark.parametrize("named", [True, False], ids=["pod-list", "no-list"])
@pytest.mark.parametrize("producer", list(_PRODUCERS))
async def test_a_turn_end_times_no_folder_that_holds_no_id(tmp_path, monkeypatch, producer, named):
    """The turn rewrites `report` and leaves `scratch`, whose metadata.json
    does not parse, so it holds no id, newer still. Its pod names both, or sends no list. The turn
    end times only `report`, as the self-heal would: `scratch` builds no
    card, so its row would only take one of the list's newest slots."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    base = project_root(project).base
    folder = timed_artifact(base, "report", 1_000_100.0)

    def during_turn() -> None:
        _rewrite_metadata(folder, 1_000_200.0)
        _idless_folder(base, "scratch", 1_000_300.0)

    await _turn_end(
        monkeypatch, producer, project=project, scope=TenantScope(org_mode=True, org_id=org_id, user_id=creator),
        conversation_id=conversation_id, during_turn=during_turn, touched=["report", "scratch"] if named else None,
    )

    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_200.0}


async def test_newest_folders_that_hold_no_id_leave_the_list_on_its_rows(
    client, org, held_roots, monkeypatch,  # noqa: F811
):
    """A pod that sends no list leaves 100 folders whose metadata.json does
    not parse, so they hold no id, newer than the project's 90 artifacts. The turn end times only
    the folders that hold an id, so the list's 160 newest rows still name 80
    folders that build a card, and it reads no root in full."""
    [path] = _bulk_paths(client, org, 0, 1)
    _artifacts_at(org, path, ALICE, {f"art-{index:03d}": 1_000_000.0 + index for index in range(90)})
    held_roots()
    base = path / ".anton" / "artifacts"

    def during_turn() -> None:
        for index in range(100):
            _idless_folder(base, f"scratch-{index:03d}", 2_000_000.0 + index)

    _remote_turn_end(monkeypatch, during_turn=during_turn, touched=None)
    await _run_turn(
        scope=TenantScope(org_mode=True, org_id=org, user_id=ALICE),
        conversation_id=_alices_conversation(org, path),
    )
    reads = _read_in_full(monkeypatch)

    response = client.get(f"/api/v1/artifacts/?project_id={_project_id(org, path)}", headers=_headers(org, ALICE))

    assert [card["slug"] for card in response.json()] == [f"art-{index:03d}" for index in range(89, 9, -1)]
    assert reads and not any(reads)


def _records_counted(monkeypatch) -> list[dict]:
    """The record units the turn ends run from now on
    (task_objects.record_new_artifacts), each still writing."""
    calls: list[dict] = []
    real = task_objects.record_new_artifacts

    def record(session, **kwargs):
        calls.append(kwargs)
        return real(session, **kwargs)

    monkeypatch.setattr(task_objects, "record_new_artifacts", record)
    return calls


@pytest.mark.parametrize("producer", list(_PRODUCERS))
async def test_a_turn_end_with_nothing_to_record_opens_no_record_unit(tmp_path, monkeypatch, producer):
    """The pod names no folder, so the turn created, listed and timed
    nothing in a project that holds `report`: the turn end opens no
    database unit to record it. The control turn, whose pod names `report`,
    which it rewrote, records once."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    folder = timed_artifact(project_root(project).base, "report", 1_000_100.0)
    scope = TenantScope(org_mode=True, org_id=org_id, user_id=creator)
    records = _records_counted(monkeypatch)

    await _turn_end(
        monkeypatch, producer, project=project, scope=scope, conversation_id=conversation_id,
        during_turn=lambda: None, touched=[],
    )

    assert records == []

    await _turn_end(
        monkeypatch, producer, project=project, scope=scope, conversation_id=conversation_id,
        during_turn=lambda: _rewrite_metadata(folder, 1_000_200.0), touched=["report"],
    )

    assert len(records) == 1


@pytest.mark.parametrize("producer", list(_PRODUCERS))
async def test_a_turn_end_in_a_persons_partition_opens_no_record_unit(tmp_path, monkeypatch, producer):
    """A person's partition keeps no rows (artifact_locations.keeps_locations).
    Its pod names `report`, whose metadata.json alone the turn rewrote: the
    turn end reads no time and opens no database unit to record it."""
    partition, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, partition)
    conversation_id = make_conversation(project, creator)
    folder = timed_artifact(project_root(project).base, "report", 1_000_100.0)
    records = _records_counted(monkeypatch)

    await _turn_end(
        monkeypatch, producer, project=project,
        scope=TenantScope(org_mode=False, org_id=partition, user_id=creator, personal=True),
        conversation_id=conversation_id, during_turn=lambda: _rewrite_metadata(folder, 1_000_200.0),
        touched=["report"],
    )

    assert records == []
    assert sort_times_of(partition) == {}


@pytest.mark.parametrize("producer", list(_PRODUCERS))
async def test_a_turn_end_whose_listing_reads_no_folder_holds_the_root_itself(tmp_path, monkeypatch, producer):
    """The pod sent no list, and the file server stops answering for the
    project's one folder during the turn. The turn end's listing finds the
    folder and reads nothing in it, so it created, listed and timed nothing,
    but the folder may hold an artifact: its record holds the root before
    any location check runs."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    source = project_root(project)
    write_artifact(source.base, "report", conversation=conversation_id)

    await _turn_end(
        monkeypatch, producer, project=project, scope=TenantScope(org_mode=True, org_id=org_id, user_id=creator),
        conversation_id=conversation_id, during_turn=lambda: _report_unreadable(monkeypatch, source.base, at="stat"),
        touched=None,
    )

    assert held_roots_of(org_id) == {HeldRoot(project.id, None)}


@pytest.mark.parametrize("producer", list(_PRODUCERS))
async def test_a_turn_end_that_cannot_time_the_folder_its_pod_named_holds_the_root_itself(
    tmp_path, monkeypatch, owned_logger, producer,
):
    """An old pod in a rollout made `report`, and wrote no row. A later
    turn rewrites its metadata.json and its pod names it, but the read of
    that file's time fails with EIO: the turn created, listed and timed
    nothing, yet `report` holds an artifact the list should show. Its record
    holds the root before any location check runs, and it logs one WARNING
    and asks for the location check that writes the time."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    folder = timed_artifact(project_root(project).base, "report", 1_000_100.0)

    def edit_while_the_server_stalls():
        _rewrite_metadata(folder, 1_000_200.0)
        _times_that_fail_for(monkeypatch, "report")

    logged = owned_logger(ap.logger.name, level=logging.WARNING)

    await _turn_end(
        monkeypatch, producer, project=project, scope=TenantScope(org_mode=True, org_id=org_id, user_id=creator),
        conversation_id=conversation_id, during_turn=edit_while_the_server_stalls, touched=["report"],
    )

    assert held_roots_of(org_id) == {HeldRoot(project.id, None)}
    assert sort_times_of(org_id) == {}
    assert ap._self_heals_requested == {_location_check(project, org_id, creator)}
    assert_log_fields(
        logged.output(), log_level="WARNING", event=_UNTIMED_EVENT, project_id=str(project.id), count=1,
    )


@pytest.mark.parametrize("producer", list(_PRODUCERS))
async def test_a_turn_end_that_cannot_time_a_folder_asks_for_the_location_check_that_does(
    tmp_path, monkeypatch, owned_logger, producer,
):
    """The pod sent no list, so the turn end lists the whole root, and
    asks for no self-heal when its record succeeds. The read of `report`'s
    metadata.json fails with EIO, so its row keeps the old time. The turn
    end logs one WARNING and asks for the project's location check, whose
    listing writes the new time once the file server answers again."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    folder = timed_artifact(project_root(project).base, "report", 1_000_100.0)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 1_000_100.0})
    answer_again: list[Callable[[], None]] = []

    def edit_while_the_server_stalls():
        _rewrite_metadata(folder, 1_000_200.0)
        answer_again.append(_times_that_fail_for(monkeypatch, "report"))

    logged = owned_logger(ap.logger.name, level=logging.WARNING)

    await _turn_end(
        monkeypatch, producer, project=project, scope=TenantScope(org_mode=True, org_id=org_id, user_id=creator),
        conversation_id=conversation_id, during_turn=edit_while_the_server_stalls, touched=None,
    )

    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_100.0}
    assert ap._self_heals_requested == {_location_check(project, org_id, creator)}
    assert ap._self_heal_publishes_requested == set()
    assert_log_fields(
        logged.output(), log_level="WARNING", event=_UNTIMED_EVENT, project_id=str(project.id), count=1,
    )

    answer_again[0]()
    await asyncio.gather(*ap.start_due_self_heals())

    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_200.0}


def _sort_time_writes_refused(monkeypatch) -> Callable[[], None]:
    """record_sort_times raises; returns what lets it write again."""
    real = artifact_sort_times.record_sort_times

    def refused(session, **kwargs):
        raise RuntimeError("the row could not be written")

    monkeypatch.setattr(artifact_sort_times, "record_sort_times", refused)
    return lambda: monkeypatch.setattr(artifact_sort_times, "record_sort_times", real)


@pytest.mark.parametrize("producer", list(_PRODUCERS))
async def test_a_turn_end_whose_sort_times_fail_asks_for_the_self_heal_that_writes_them(
    tmp_path, monkeypatch, caplog, producer,
):
    """The pod sent no list, so nothing asked for the project's self-heal
    before the diff. The record writes the turn's other rows, then raises
    ListedLocationsUnchecked for the sort times it could not write, so the
    turn end asks for the self-heal, whose listing writes them."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    source = project_root(project)
    writable_again = _sort_time_writes_refused(monkeypatch)

    with caplog.at_level(logging.ERROR, logger=_PRODUCERS[producer].logger):
        await _turn_end(
            monkeypatch, producer, project=project,
            scope=TenantScope(org_mode=True, org_id=org_id, user_id=creator), conversation_id=conversation_id,
            during_turn=lambda: write_artifact(source.base, "report", conversation=conversation_id), touched=None,
        )

    [record] = _turn_end_errors(caplog, producer)
    assert isinstance(record.exc_info[1], ListedLocationsUnchecked)
    assert record.getMessage() == _PRODUCERS[producer].listed_lost
    assert ap._self_heals_requested == {_location_check(project, org_id, creator)}
    assert held_roots_of(org_id) == {HeldRoot(project.id, None)}
    assert sort_times_of(org_id) == {}

    writable_again()
    await asyncio.gather(*ap.start_due_self_heals())

    assert set(sort_times_of(org_id)) == {(project.id, None, "report")}


@pytest.mark.parametrize("producer", list(_PRODUCERS))
async def test_an_edit_only_turn_end_whose_record_fails_names_only_what_it_lost(
    tmp_path, monkeypatch, caplog, producer,
):
    """The pod named the one folder its tool edited, so the turn end listed
    nothing and checked no location. Its record's sort-time write fails,
    and its line names the sort times and the held folder, never a location
    check it did not make."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    folder = timed_artifact(project_root(project).base, "report", 1_000_100.0)
    _sort_time_writes_refused(monkeypatch)

    with caplog.at_level(logging.ERROR, logger=_PRODUCERS[producer].logger):
        await _turn_end(
            monkeypatch, producer, project=project,
            scope=TenantScope(org_mode=True, org_id=org_id, user_id=creator), conversation_id=conversation_id,
            during_turn=lambda: _rewrite_metadata(folder, 1_000_200.0), touched=["report"],
        )

    [record] = _turn_end_errors(caplog, producer)
    assert isinstance(record.exc_info[1], ListedLocationsUnchecked)
    assert record.getMessage() == _PRODUCERS[producer].edited_lost


# ── a task's move and a delete ───────────────────────────────────────────


@pytest.mark.parametrize("failing", ["sort_times", "hold"])
async def test_a_task_move_whose_sort_times_or_hold_fail_asks_for_the_checks_that_write_them(
    tmp_path, monkeypatch, failing,
):
    """The two writes after a move run on their own: a failed move of the
    sort times still holds the destination's root, and a failed hold still
    moves the rows. Each failure asks for a location check alone of the
    project whose rows it left behind: the destination's for either, and
    the source's too for the sort times, whose row names a folder gone
    from it. Those checks write the destination's rows and drop the
    source's."""
    org_id, owner = str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation_id = make_conversation(source, owner)
    folder = write_artifact(project_root(source).base, "report", conversation_id)
    os.utime(folder / "metadata.json", (1_000_100.0, 1_000_100.0))
    _index(conversation_id, source.id, "report")
    with scoped(org_id) as session:
        artifact_ownership.record_artifact_owner(session, source.id, "report", owner, action="create")
        record_sort_times(session, project_id=source.id, times={"report": 1_000_100.0})
    module, name = (artifact_sort_times, "move_sort_times") if failing == "sort_times" else (
        held_artifact_roots, "hold_root"
    )
    real = getattr(module, name)

    def refused(session, **kwargs):
        raise RuntimeError("the rows could not be written")

    monkeypatch.setattr(module, name, refused)

    assert _relocate(org_id, owner, conversation_id, source.id, dest.id) == 1

    dest_check, source_check = _location_check(dest, org_id, owner), _location_check(source, org_id, owner)
    if failing == "sort_times":
        assert ap._self_heals_requested == {dest_check, source_check}
        assert (held_roots_of(org_id), sort_times_of(org_id)) == (
            {HeldRoot(dest.id, None)}, {(source.id, None, "report"): 1_000_100.0},
        )
    else:
        assert ap._self_heals_requested == {dest_check}
        assert (held_roots_of(org_id), sort_times_of(org_id)) == (set(), {(dest.id, None, "report"): 1_000_100.0})
    assert ap._self_heal_publishes_requested == set()
    monkeypatch.setattr(module, name, real)

    await asyncio.gather(*ap.start_due_self_heals())

    assert (held_roots_of(org_id), sort_times_of(org_id)) == (
        {HeldRoot(dest.id, None)},
        {(dest.id, None, "report"): 1_000_100.0},
    )


async def test_a_deleted_artifact_takes_its_sort_time_with_it(
    tmp_path, org_deployment, granted_product_permissions, no_publish_side_effects,  # noqa: F811
):
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    artifact_id, folder = _write(source, "mine")
    _write(source, "kept")
    with scoped(org_id) as session:
        artifact_ownership.record_artifact_owner(session, project.id, "mine", creator, action="create")
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(project.id, None, "mine")])
        record_sort_times(session, project_id=project.id, times={"mine": 1_000_100.0, "kept": 1_000_200.0})

    with scoped(org_id, creator) as session:
        await artifacts_ep.delete_artifact_for_request(session.scope, artifact_id, project_id=project.id)

    assert not folder.exists()
    assert sort_times_of(org_id) == {(project.id, None, "kept"): 1_000_200.0}
