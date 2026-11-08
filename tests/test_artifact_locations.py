"""Artifact lookups open the one folder a location row names.

Org tenancy only, where every listing and stat goes to the shared file
store's server. A lookup reads the artifact's row in its database step, then
opens that folder inside a root the caller may already read, and serves it
only when the folder's metadata names the requested id. Without a row that
answers, it lists the project's roots as before, reads every folder of the
project holding the id, and records what it found: the folder when one holds
it, the id marked conflicted when two do. Two folders in the roots that
decide the lookup answer 409, as before rows existed. A turn end, which sees
only its own folders, writes no row.
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import importlib
import json
import logging
import os
import shutil
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import DataError, IntegrityError, OperationalError
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import Session, select

import cowork.services.task_objects as task_objects
from cowork import server
from cowork.api.v1 import artifact_scope
from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness import harness as harness_module
from cowork.models.artifact_location import ArtifactLocation
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.services import (
    artifact_draft_review,
    artifact_identity,
    artifact_location_backfill,
    artifact_locations,
    artifact_ownership,
    artifact_roots,
)
from cowork.services.artifact_identity import FolderCheck
from cowork.services.artifact_locations import (
    Location,
    LocationRow,
    move_locations,
    record_checked_folders,
)
from cowork.services.task_objects import record_new_artifacts, turn_artifact_changes
from test_artifact_ownership import legacy_root, make_conversation, make_project, project_root, scoped
from test_artifact_permissions import no_publish_side_effects  # noqa: F401  (a fixture)
from test_artifact_relocation_ownership import _index, _relocate
from test_artifact_roots import _count_hops
from test_harness_scratchpad_teardown import _completes, _drain, turns  # noqa: F401  (turns is a fixture)
from test_router_probe_lifespan import boot  # noqa: F401  (a fixture)

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


def _write(source, slug: str, artifact_id: str | None = None) -> tuple[str, Path]:
    artifact_id = artifact_id or uuid4().hex
    folder = source.base / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(
        json.dumps({"id": artifact_id, "slug": slug, "type": "html-app", "primary": "index.html"})
    )
    return artifact_id, folder


def _stored(org_id: str, artifact_id: str) -> ArtifactLocation | None:
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        row = raw.exec(
            select(ArtifactLocation).where(
                ArtifactLocation.org_id == org_id, ArtifactLocation.artifact_id == artifact_id
            )
        ).first()
        if row is not None:
            raw.expunge(row)
        return row


def _row(org_id: str, artifact_id: str) -> Location | None:
    row = _stored(org_id, artifact_id)
    return Location(row.project_id, row.conversation_id, row.slug) if row is not None else None


def _conflicted(org_id: str, artifact_id: str) -> bool:
    row = _stored(org_id, artifact_id)
    return row is not None and row.conflicted


def _no_lock(_engine, _key):
    return nullcontext(True)


def _sentinel_present() -> bool:
    return artifact_location_backfill.sentinel_present(
        get_engine(get_app_settings().database.uri), artifact_location_backfill.SENTINEL_KEY
    )


@pytest.fixture
def backfill_pending():
    """No backfill has written its sentinel; a test that runs the pass gets
    its sentinel and its kept progress removed after, so one test's progress
    never skips the next test's projects."""
    yield
    keys = (artifact_location_backfill.SENTINEL_KEY, artifact_location_backfill.PROGRESS_KEY)
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        for setting in raw.exec(select(Setting).where(Setting.key.in_(keys))).all():
            raw.delete(setting)
        raw.commit()


class _FileCalls:
    """Counts filesystem calls, and the directory listings among them."""

    NAMES = ("stat", "lstat", "fstat", "open", "scandir", "listdir")

    def __init__(self, monkeypatch):
        self.count = 0
        self.listings = 0
        self.counting = False
        for name in self.NAMES:
            real = getattr(os, name)

            def counted(*args, _real=real, _name=name, **kwargs):
                if self.counting:
                    self.count += 1
                    self.listings += _name in ("scandir", "listdir")
                return _real(*args, **kwargs)

            monkeypatch.setattr(os, name, counted)


async def _for_request(scope, project_ref: str, artifact_id: str):
    return await artifact_scope.review_artifact_for_request(scope, project_ref, artifact_id)


async def _and_read(scope, project_ref: str, artifact_id: str):
    reviewed = await artifact_scope.review_artifact_and_read(
        scope, project_ref, artifact_id, read=lambda artifact: artifact.folder
    )
    assert reviewed.value == reviewed.artifact.folder
    return reviewed.artifact


@pytest.fixture(params=[_for_request, _and_read], ids=["review_artifact_for_request", "review_artifact_and_read"])
def entry_point(request):
    """Both lookups a route can make: the bare review, and the review with
    the route's own read in its file store call."""
    return request.param


async def _review(org_id: str, user_id: str, project, artifact_id: str, *, entry=_for_request):
    with scoped(org_id, user_id) as session:
        scope = session.scope
    return await entry(scope, str(project.id), artifact_id)


async def _refused(org_id: str, user_id: str, project, artifact_id: str, *, entry=_for_request) -> int:
    with pytest.raises(HTTPException) as refused:
        await _review(org_id, user_id, project, artifact_id, entry=entry)
    return refused.value.status_code


def _record_turn(org_id: str, *, project, conversation: UUID, creator: str) -> None:
    """A turn end that made every folder now in the project's shared root."""
    changes = turn_artifact_changes(
        conversation_id=conversation, artifacts_base=project_root(project).base, before=set(), before_mtimes={},
    )
    with scoped(org_id) as session:
        record_new_artifacts(
            session, conversation_id=conversation, project_id=project.id,
            slugs=changes.created, creator=creator, artifact_ids=changes.created_ids,
        )


@pytest.mark.parametrize("conversations", [1, 500])
async def test_a_warm_lookup_opens_one_folder_however_many_conversation_folders(
    tmp_path, org_deployment, monkeypatch, conversations,
):
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id, _folder = _write(project_root(project), "brief")
    # Legacy roots that each hold an artifact: the scan lists and indexes each.
    for _ in range(conversations):
        _write(legacy_root(project, str(uuid4())), "old")
    artifact_identity._clear_identity_indexes()

    first = await _review(org_id, user, project, artifact_id)  # scans, records the row
    assert _row(org_id, artifact_id) == Location(project.id, None, "brief")

    calls = _FileCalls(monkeypatch)
    calls.counting = True
    second = await _review(org_id, user, project, artifact_id)
    calls.counting = False

    assert second.folder == first.folder and second.is_own
    assert calls.count <= 40, f"{calls.count} filesystem calls for one warm lookup"
    assert calls.listings == 0


async def test_a_reviewer_reads_no_other_members_conversation_folders(tmp_path, org_deployment, monkeypatch):
    org_id, owner, reviewer = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation = make_conversation(project, owner)
    artifact_id, folder = _write(legacy_root(project, str(conversation)), "draft")
    for _ in range(20):
        _write(legacy_root(project, str(make_conversation(project, owner))), "other")
    granted = {folder}
    monkeypatch.setattr(artifact_draft_review, "draft_review_allows", lambda path, _org: Path(path) in granted)
    with scoped(org_id, owner) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(project.id, conversation, "draft")])

    calls = _FileCalls(monkeypatch)
    calls.counting = True
    resolved = await _review(org_id, reviewer, project, artifact_id)
    calls.counting = False

    assert resolved.folder == folder and not resolved.is_own
    assert calls.listings == 0

    granted.clear()
    with pytest.raises(HTTPException) as refused:
        await _review(org_id, reviewer, project, artifact_id)
    assert refused.value.status_code == 404


def _touched_by_discovery(monkeypatch) -> list[Path]:
    """Every path root discovery checks for a link, in the order checked."""
    touched: list[Path] = []
    real_is_dir = artifact_roots._is_real_directory
    real_safe = artifact_roots._storage_components_are_safe

    def is_dir(path, *, strict=False):
        touched.append(Path(path))
        return real_is_dir(path, strict=strict)

    def safe(workspace, *, may_be_absent, strict=False):
        touched.append(Path(workspace))
        return real_safe(workspace, may_be_absent=may_be_absent, strict=strict)

    monkeypatch.setattr(artifact_roots, "_is_real_directory", is_dir)
    monkeypatch.setattr(artifact_roots, "_storage_components_are_safe", safe)
    return touched


class _CoMembersWorkspaces(NamedTuple):
    """A project of `owner`'s artifact in the shared root beside 50 of
    `member`'s legacy workspaces, each holding an artifact, the first also a
    draft `member` opened for review."""

    org_id: str
    owner: str
    project: Project
    artifact_id: str
    workspaces: list[Path]
    granted_id: str
    granted: Path


def _co_members_workspaces(tmp_path) -> _CoMembersWorkspaces:
    org_id, owner, member = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id, _folder = _write(project_root(project), "brief")
    sources = [legacy_root(project, str(make_conversation(project, member))) for _ in range(50)]
    for source in sources:
        _write(source, "chart")
    granted_id, granted = _write(sources[0], "dash")
    artifact_draft_review.enable_draft_review(granted, org_id=org_id, enabled_by=member)
    artifact_identity._clear_identity_indexes()
    return _CoMembersWorkspaces(
        org_id, owner, project, artifact_id, [source.base.parent.parent for source in sources], granted_id, granted,
    )


def _in_workspaces(touched: list[Path], workspaces: list[Path]) -> list[Path]:
    return [path for path in touched if any(path.is_relative_to(workspace) for workspace in workspaces)]


async def test_an_owners_lookup_by_its_row_never_stats_a_co_members_workspace(
    tmp_path, org_deployment, monkeypatch, entry_point,
):
    """Once the owner's artifact has its row, the owner's lookup opens the
    row's folder alone and stats none of a co-member's 50 workspaces. A
    reviewer's lookup of a granted draft in one of them still finds it. The
    first lookup, with no row, scans every root, since the row it records
    answers every member's lookup (artifact_scope._scan_found)."""
    plant = _co_members_workspaces(tmp_path)
    await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id)  # scans, records the row
    assert _row(plant.org_id, plant.artifact_id) == Location(plant.project.id, None, "brief")
    touched = _touched_by_discovery(monkeypatch)

    found = await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point)

    assert found.is_own and found.folder.name == "brief"
    assert _in_workspaces(touched, plant.workspaces) == []

    reviewed = await _review(plant.org_id, plant.owner, plant.project, plant.granted_id, entry=entry_point)

    assert not reviewed.is_own and reviewed.folder == plant.granted


def test_a_turns_folder_for_a_recorded_id_marks_it_conflicted_and_keeps_the_row(tmp_path, caplog):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id = uuid4().hex
    first = Location(project.id, None, "brief")
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[first])
        # The folder the row names: no conflict.
        artifact_locations.record_made_folder(session, artifact_id=artifact_id, location=first)
        assert not _conflicted(org_id, artifact_id)
        with caplog.at_level(logging.WARNING, logger=artifact_locations.logger.name):
            artifact_locations.record_made_folder(
                session, artifact_id=artifact_id, location=Location(project.id, None, "brief-copy"),
            )

    assert _row(org_id, artifact_id) == first
    # Two folders of one project claim the id, so its lookups scan again.
    assert _conflicted(org_id, artifact_id)
    [logged] = [
        record for record in caplog.records
        if record.name == artifact_locations.logger.name and record.levelno == logging.WARNING
    ]
    assert (logged.project_id, logged.artifact_slug) == (str(project.id), "brief")


def test_rows_are_per_organization(tmp_path):
    artifact_id = uuid4().hex
    for org_id in (str(uuid4()), str(uuid4())):
        project = make_project(tmp_path, org_id)
        with scoped(org_id) as session:
            record_checked_folders(session, artifact_id=artifact_id, folders=[Location(project.id, None, "brief")])
        assert _row(org_id, artifact_id) == Location(project.id, None, "brief")


async def test_a_row_whose_folder_holds_another_id_is_refused_and_dropped(tmp_path, org_deployment):
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    artifact_id = uuid4().hex
    _write(source, "brief", uuid4().hex)  # the recorded folder now holds another id
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(project.id, None, "brief")])
    artifact_identity._clear_identity_indexes()

    with pytest.raises(HTTPException) as missing:
        await _review(org_id, user, project, artifact_id)

    assert missing.value.status_code == 404
    assert _row(org_id, artifact_id) is None


async def test_a_row_whose_folder_moved_follows_the_folder_the_scan_finds(tmp_path, org_deployment):
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id, folder = _write(project_root(project), "renamed")
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(project.id, None, "brief")])
    artifact_identity._clear_identity_indexes()

    resolved = await _review(org_id, user, project, artifact_id)

    assert resolved.folder == folder
    assert _row(org_id, artifact_id) == Location(project.id, None, "renamed")


async def test_a_row_in_another_project_is_never_served(tmp_path, org_deployment):
    """The row names a project the request didn't, whose folder still holds
    the id: the lookup scans the requested one, and the row stays."""
    org_id, user = str(uuid4()), str(uuid4())
    elsewhere, requested = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    artifact_id, _ = _write(project_root(elsewhere), "brief")
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(elsewhere.id, None, "brief")])

    with pytest.raises(HTTPException) as missing:
        await _review(org_id, user, requested, artifact_id)

    assert missing.value.status_code == 404
    assert _row(org_id, artifact_id) == Location(elsewhere.id, None, "brief")


async def test_a_lookup_without_a_row_records_the_folder_it_finds(tmp_path, org_deployment):
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation = make_conversation(project, user)
    artifact_id, _ = _write(legacy_root(project, str(conversation)), "old")

    await _review(org_id, user, project, artifact_id)

    assert _row(org_id, artifact_id) == Location(project.id, conversation, "old")


async def test_a_desktop_lookup_records_nothing(tmp_path):
    project = make_project(tmp_path, None)
    artifact_id, _ = _write(project_root(project), "brief")

    with scoped(None) as session:
        resolved = await artifact_scope.review_artifact_for_request(session.scope, str(project.id), artifact_id)

    assert resolved.folder.name == "brief"
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        assert raw.exec(select(ArtifactLocation).where(ArtifactLocation.artifact_id == artifact_id)).first() is None


@pytest.mark.parametrize("row_names", ["the_deleted_folder", "a_conflicted_id", "another_projects_folder"])
async def test_deleting_an_artifact_deletes_its_row(
    tmp_path, org_deployment, granted_product_permissions, no_publish_side_effects, row_names,  # noqa: F811
):
    """The row goes with the folder it names. A conflicted row stays, since
    another folder of the project still holds the id: here a copy in a
    co-member's workspace, which the creator's lookup does not read. So does
    a row naming another project's folder, which still holds the id."""
    org_id, creator, member = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    artifact_id, folder = _write(source, "mine")
    mine = Location(project.id, None, "mine")
    recorded = [mine]
    if row_names == "a_conflicted_id":
        copy_conversation = make_conversation(project, member)
        _write(legacy_root(project, str(copy_conversation)), "copy", artifact_id)
        recorded = [mine, Location(project.id, copy_conversation, "copy")]
    elif row_names == "another_projects_folder":
        elsewhere = make_project(tmp_path, org_id)
        _write(project_root(elsewhere), "theirs", artifact_id)
        recorded = [Location(elsewhere.id, None, "theirs")]
    with scoped(org_id) as session:
        artifact_ownership.record_artifact_owner(session, project.id, "mine", creator, action="create")
        record_checked_folders(session, artifact_id=artifact_id, folders=recorded)
    before = _stored(org_id, artifact_id)

    with scoped(org_id, creator) as session:
        await artifacts_ep.delete_artifact_for_request(session.scope, artifact_id, project_id=project.id)

    assert not folder.exists()
    after = _stored(org_id, artifact_id)
    if row_names == "the_deleted_folder":
        assert after is None
    else:
        assert after is not None
        assert (_row(org_id, artifact_id), after.conflicted) == (recorded[0], before.conflicted)


async def test_a_turns_new_folder_gets_its_row_at_its_first_lookup(tmp_path, org_deployment):
    """A turn end sees only the folders it made, and an older folder without
    a row may hold the same id, so it writes no row. The first lookup lists
    every root and records the folder."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation = make_conversation(project, creator)
    source = project_root(project)
    artifact_id, folder = _write(source, "made-this-turn")

    changes = turn_artifact_changes(conversation_id=conversation, artifacts_base=source.base, before=set(), before_mtimes={})
    assert changes.created_ids == {"made-this-turn": artifact_id}
    with scoped(org_id) as session:
        record_new_artifacts(
            session, conversation_id=conversation, project_id=project.id,
            slugs=changes.created, creator=creator, artifact_ids=changes.created_ids,
        )

    assert _row(org_id, artifact_id) is None
    assert (await _review(org_id, creator, project, artifact_id)).folder == folder
    assert _row(org_id, artifact_id) == Location(project.id, None, "made-this-turn")


def test_a_moved_folders_row_follows_it(tmp_path):
    org_id = str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    artifact_id = uuid4().hex
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(source.id, None, "brief")])
        move_locations(session, project_id=source.id, moved={"brief": "abcd1234-brief"}, to_project_id=dest.id)

    assert _row(org_id, artifact_id) == Location(dest.id, None, "abcd1234-brief")


def test_the_backfill_records_every_artifact_held_by_one_folder(tmp_path, org_deployment, backfill_pending, caplog):
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation = make_conversation(project, user)
    shared, _ = _write(project_root(project), "shared")
    legacy, _ = _write(legacy_root(project, str(conversation)), "legacy")
    twice = uuid4().hex
    _write(project_root(project), "one", twice)
    _write(project_root(project), "two", twice)
    engine = get_engine(get_app_settings().database.uri)

    with caplog.at_level(logging.WARNING, logger=artifact_location_backfill.logger.name):
        summary = artifact_location_backfill.run_artifact_location_backfill(
            engine, try_lock=_no_lock, project_ids=[project.id]
        )

    assert summary is not None and summary.recorded == 2 and summary.duplicated == [twice]
    assert _row(org_id, shared) == Location(project.id, None, "shared")
    assert _row(org_id, legacy) == Location(project.id, conversation, "legacy")
    assert not _conflicted(org_id, shared) and not _conflicted(org_id, legacy)
    # Held by two folders: marked, so no later writer records either one.
    assert _conflicted(org_id, twice)
    assert _sentinel_present()
    # Each count in fields the JSON logs keep, so a pass that recorded
    # nothing reads differently from a clean one.
    assert {
        (record.outcome, record.count)
        for record in caplog.records
        if record.name == artifact_location_backfill.logger.name
        and getattr(record, "event", None) == "artifact_location_backfill_summary"
    } == {("recorded", 2), ("duplicated", 1), ("conflicts", 0), ("unread", 0), ("skipped", 0)}
    # Once per environment.
    assert artifact_location_backfill.run_artifact_location_backfill(
        engine, try_lock=_no_lock, project_ids=[project.id]
    ) is None


def test_a_stopped_backfill_writes_no_sentinel(tmp_path, org_deployment, backfill_pending):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    stop = threading.Event()
    stop.set()
    engine = get_engine(get_app_settings().database.uri)

    assert artifact_location_backfill.run_artifact_location_backfill(
        engine, try_lock=_no_lock, project_ids=[project.id], stop=stop
    ) is None
    assert not _sentinel_present()


class _Plant(NamedTuple):
    """A member's legacy artifact, and a folder a co-member's agent wrote in
    the project's shared root claiming the same id."""

    org_id: str
    owner: str
    member: str
    project: Project
    artifact_id: str
    owners_folder: Path
    planted: Path


def _owners_artifact(tmp_path) -> _Plant:
    org_id, owner, member = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation = make_conversation(project, owner)
    artifact_id, folder = _write(legacy_root(project, str(conversation)), "owners-draft")
    return _Plant(org_id, owner, member, project, artifact_id, folder, project_root(project).base / "planted")


def _plant(plant: _Plant) -> None:
    _write(project_root(plant.project), plant.planted.name, plant.artifact_id)
    artifact_identity._clear_identity_indexes()


async def test_a_co_members_lookup_records_no_row_for_an_id_two_folders_hold(
    tmp_path, org_deployment, entry_point,
):
    """The co-member's own roots hold only the planted folder, but the scan
    reads every root of the project, so the id is marked conflicted rather
    than recorded at the plant, and the owner keeps the 409 they got before
    rows existed."""
    plant = _owners_artifact(tmp_path)
    _plant(plant)

    for _ in range(2):
        seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id, entry=entry_point)
        assert (seen.folder, seen.is_own) == (plant.planted, True)
        assert _conflicted(plant.org_id, plant.artifact_id)
        assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


async def test_a_folder_planted_after_the_owners_row_marks_the_id_conflicted(
    tmp_path, org_deployment, entry_point,
):
    plant = _owners_artifact(tmp_path)
    first = await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point)
    assert first.folder == plant.owners_folder and not _conflicted(plant.org_id, plant.artifact_id)

    _plant(plant)
    seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id, entry=entry_point)

    assert seen.folder == plant.planted
    assert _conflicted(plant.org_id, plant.artifact_id)
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


def _unreadable(monkeypatch, folder_name: str, *, strict_too: bool, error: int = errno.EIO) -> threading.Event:
    """Make the folder named ``folder_name`` fail to read with ``error`` in
    the lookup's cached identity index and, with ``strict_too``, in strict
    reads as well, while the returned event is set."""
    failing = threading.Event()
    failing.set()
    real_ensure, real_read = artifact_identity.ensure_full_id, artifact_identity.read_full_id

    def ensure(path, *args, **kwargs):
        if failing.is_set() and Path(path).name == folder_name:
            raise OSError(error, os.strerror(error))
        return real_ensure(path, *args, **kwargs)

    def read(folder, *args, **kwargs):
        if failing.is_set() and strict_too and Path(folder.path).name == folder_name:
            raise OSError(error, os.strerror(error))
        return real_read(folder, *args, **kwargs)

    monkeypatch.setattr(artifact_identity, "ensure_full_id", ensure)
    monkeypatch.setattr(artifact_identity, "read_full_id", read)
    return failing


async def test_a_folder_the_lookups_index_could_not_read_still_counts_before_a_row_is_written(
    tmp_path, org_deployment, entry_point, monkeypatch,
):
    """A read error leaves the owner's folder out of the co-member's cached
    scan, which then sees only the planted folder. The row is decided from a
    strict read of every root, which finds both, so the id is marked
    conflicted rather than recorded at the plant."""
    plant = _owners_artifact(tmp_path)
    _plant(plant)
    failing = _unreadable(monkeypatch, plant.owners_folder.name, strict_too=False)

    seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id, entry=entry_point)

    assert seen.folder == plant.planted
    assert _conflicted(plant.org_id, plant.artifact_id)
    failing.clear()
    artifact_identity._clear_identity_indexes()
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


async def test_a_scan_that_could_not_read_every_folder_records_no_row(
    tmp_path, org_deployment, entry_point, monkeypatch,
):
    """When the owner's folder fails to read in the strict read too, the
    lookup cannot know how many folders hold the id, so it writes no row,
    and the owner's lookup once the folder reads again still answers 409."""
    plant = _owners_artifact(tmp_path)
    _plant(plant)
    failing = _unreadable(monkeypatch, plant.owners_folder.name, strict_too=True)

    seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id, entry=entry_point)

    assert seen.folder == plant.planted
    assert _row(plant.org_id, plant.artifact_id) is None
    assert not _conflicted(plant.org_id, plant.artifact_id)
    failing.clear()
    artifact_identity._clear_identity_indexes()
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


async def test_a_root_the_lookups_discovery_could_not_list_still_counts_before_a_row_is_written(
    tmp_path, org_deployment, entry_point, monkeypatch,
):
    """Listing the conversation workspaces fails during a co-member's lookup,
    so its scan sees only the shared root and the plant there. The strict
    count lists them again, fails the same way, and so writes no row; the
    owner's lookup once the listing works answers 409."""
    plant = _owners_artifact(tmp_path)
    _plant(plant)
    conversations = Path(plant.project.path) / "conversations"
    real_iterdir = Path.iterdir

    def iterdir(path):
        if path == conversations:
            raise OSError(errno.EIO, "Input/output error")
        return real_iterdir(path)

    with monkeypatch.context() as unlisted:
        unlisted.setattr(Path, "iterdir", iterdir)
        seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id, entry=entry_point)

    assert (seen.folder, seen.is_own) == (plant.planted, True)
    assert _row(plant.org_id, plant.artifact_id) is None
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


def test_a_rows_folder_in_another_persons_partition_is_never_opened(tmp_path, monkeypatch):
    """A row whose project names a folder in another partition of the shared
    root is not opened. It counts as stale, so the lookup's scan moves or drops
    it, and root discovery leaves that project out of the scan."""
    shared_root = Path(os.path.realpath(get_app_settings().storage.shared_root))
    project_dir = shared_root / str(uuid4()) / "projects" / "general"
    source = artifact_scope.ProjectArtifacts(
        base=project_dir / ".anton" / "artifacts", project_id=str(uuid4()), project_name="general",
        trusted_anchor=project_dir, root_parts=(".anton", "artifacts"),
    )
    location = Location(UUID(source.project_id), None, "brief")
    lookup = artifact_scope._Lookup(
        artifact_roots.RootsPlan(partition=str(uuid4())), LocationRow(location, False), source, own=True, in_plan=True,
    )

    def refuse(*_args, **_kwargs):
        raise AssertionError("opened a folder in another person's partition")

    monkeypatch.setattr(artifact_scope, "_open_recorded", refuse)
    found = artifact_scope._open_or_scan(
        lookup, artifact_id=str(uuid4()), org_id=None, read=lambda artifact: artifact, keeps_rows=True,
    )

    assert found.opened == artifact_scope._Opened(FolderCheck.STALE)
    assert found.scanned is not None and found.scanned.refusal is not None


def test_the_backfill_reads_no_project_in_another_organizations_partition(tmp_path, org_deployment, backfill_pending):
    """A project whose stored path sits in another organization's folder of the
    shared root, as one recorded through a linked folder would, is left out of
    the backfill: its folders are not read and get no rows."""
    other_org, org_id = str(uuid4()), str(uuid4())
    shared_root = Path(os.path.realpath(get_app_settings().storage.shared_root))
    path = shared_root / other_org / "projects" / f"proj-{uuid4().hex[:8]}"
    path.mkdir(parents=True)
    project = Project(id=uuid4(), name=path.name, path=str(path), org_id=org_id)
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        raw.add(project)
        raw.commit()
        raw.refresh(project)
        raw.expunge(project)
    artifact_id, _ = _write(project_root(project), "theirs")

    artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert _row(org_id, artifact_id) is None


def _no_strict_count(monkeypatch) -> None:
    """Fail the test if a lookup counts the folders again strictly."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("a lookup that cannot write a fresh row counted the folders again")

    monkeypatch.setattr(artifact_scope, "folders_holding_strictly", refuse)


async def test_a_desktop_lookup_counts_no_folder_twice(tmp_path, monkeypatch):
    project = make_project(tmp_path, None)
    artifact_id, _ = _write(project_root(project), "brief")
    _no_strict_count(monkeypatch)

    with scoped(None) as session:
        resolved = await artifact_scope.review_artifact_for_request(session.scope, str(project.id), artifact_id)

    assert resolved.folder.name == "brief"


def _strict_counts(monkeypatch) -> list[tuple[str, ...]]:
    """Record the folder names each strict count of a lookup finds."""
    counts: list[tuple[str, ...]] = []
    real = artifact_scope.folders_holding_strictly

    def counted(*args, **kwargs):
        held = real(*args, **kwargs)
        counts.append(tuple(found.folder.name for found in held))
        return held

    monkeypatch.setattr(artifact_scope, "folders_holding_strictly", counted)
    return counts


async def test_a_conflicted_id_with_one_folder_left_clears_its_mark(
    tmp_path, org_deployment, entry_point, monkeypatch,
):
    """The row names the owner's folder and the id is conflicted. While the
    plant exists, the owner's lookup answers 409, counts nothing strictly,
    and the mark stays. Once the plant is gone, the owner's lookup finds the
    row's folder alone, and a strict count of every root agrees, so the mark
    clears and the next lookup opens only the row's folder."""
    plant = _owners_artifact(tmp_path)
    await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point)
    _plant(plant)
    await _review(plant.org_id, plant.member, plant.project, plant.artifact_id, entry=entry_point)
    assert _conflicted(plant.org_id, plant.artifact_id)
    assert _row(plant.org_id, plant.artifact_id).slug == plant.owners_folder.name
    strict_counts = _strict_counts(monkeypatch)
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409
    assert _conflicted(plant.org_id, plant.artifact_id)
    assert strict_counts == []
    shutil.rmtree(plant.planted)
    artifact_identity._clear_identity_indexes()

    seen = await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point)

    assert seen.folder == plant.owners_folder
    assert strict_counts == [(plant.owners_folder.name,)]
    assert not _conflicted(plant.org_id, plant.artifact_id)
    assert _row(plant.org_id, plant.artifact_id).slug == plant.owners_folder.name
    calls = _FileCalls(monkeypatch)
    calls.counting = True
    again = await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point)
    calls.counting = False
    assert again.folder == plant.owners_folder
    assert calls.listings == 0


async def test_a_strict_count_that_cannot_read_every_folder_keeps_the_mark(
    tmp_path, org_deployment, monkeypatch,
):
    """The plant is gone from the cached scan's view but fails to read in
    the strict count, which then cannot say one folder holds the id: the
    mark stays, and the owner's lookup is still served their folder."""
    plant = _owners_artifact(tmp_path)
    await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id)
    _plant(plant)
    await _review(plant.org_id, plant.member, plant.project, plant.artifact_id)
    assert _conflicted(plant.org_id, plant.artifact_id)
    _unreadable(monkeypatch, plant.planted.name, strict_too=True)
    artifact_identity._clear_identity_indexes()

    seen = await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id)

    assert seen.folder == plant.owners_folder
    assert _conflicted(plant.org_id, plant.artifact_id)


async def test_a_conflicted_mark_clears_where_no_caller_reads_every_root_as_their_own(
    tmp_path, org_deployment, entry_point, monkeypatch,
):
    """The owner and a member each have a legacy root, so neither reads every
    root of the project as their own. The row names the shared root's
    folder, and a turn's copy beside it marks the id conflicted, so lookups
    answer 409. Once the copy is gone, the next lookup finds the row's folder
    alone among the caller's own roots, a strict count of every root agrees,
    and the mark clears; the lookup after it opens only the row's folder."""
    org_id, owner, member = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    _write(legacy_root(project, str(make_conversation(project, owner))), "owners-old")
    _write(legacy_root(project, str(make_conversation(project, member))), "members-old")
    artifact_id, brief = _write(project_root(project), "brief")
    await _review(org_id, owner, project, artifact_id, entry=entry_point)
    assert _row(org_id, artifact_id) == Location(project.id, None, "brief")
    copy = brief.with_name("brief-copy")
    shutil.copytree(brief, copy)
    with scoped(org_id) as session:
        record_new_artifacts(
            session, conversation_id=make_conversation(project, member), project_id=project.id,
            slugs=[copy.name], creator=member, artifact_ids={copy.name: artifact_id},
        )
    artifact_identity._clear_identity_indexes()
    assert _conflicted(org_id, artifact_id)
    strict_counts = _strict_counts(monkeypatch)
    for user in (owner, member):
        assert await _refused(org_id, user, project, artifact_id, entry=entry_point) == 409
    assert strict_counts == []
    shutil.rmtree(copy)
    artifact_identity._clear_identity_indexes()

    seen = await _review(org_id, owner, project, artifact_id, entry=entry_point)

    assert seen.folder == brief
    assert strict_counts == [(brief.name,)]
    assert not _conflicted(org_id, artifact_id)
    assert _row(org_id, artifact_id) == Location(project.id, None, "brief")
    calls = _FileCalls(monkeypatch)
    calls.counting = True
    again = await _review(org_id, member, project, artifact_id, entry=entry_point)
    calls.counting = False
    assert again.folder == brief
    assert calls.listings == 0


async def test_a_lookup_counts_a_conflicted_id_strictly_at_most_once_per_window(
    tmp_path, org_deployment, monkeypatch,
):
    """The row names the owner's legacy folder, and a copy in a co-member's
    workspace marks the id conflicted. The owner's own roots hold only the
    row's folder, so the owner's lookup counts every root strictly, finds the
    copy, and the mark stays. Once the copy is gone, a lookup inside the
    window counts nothing and the mark stays; the first lookup after the
    window counts again, and the mark clears."""
    now = [1000.0]
    monkeypatch.setattr(
        artifact_scope,
        "_recount_times",
        artifact_scope._RecountTimes(
            seconds=artifact_scope._RECOUNT_SECONDS, limit=artifact_scope._RECOUNT_IDS, clock=lambda: now[0],
        ),
    )
    plant = _owners_artifact(tmp_path)
    await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id)
    members_root = legacy_root(plant.project, str(make_conversation(plant.project, plant.member)))
    _, copy = _write(members_root, "copy", plant.artifact_id)
    artifact_identity._clear_identity_indexes()
    await _review(plant.org_id, plant.member, plant.project, plant.artifact_id)
    assert _conflicted(plant.org_id, plant.artifact_id)
    strict_counts = _strict_counts(monkeypatch)

    async def owners_lookup() -> None:
        seen = await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id)
        assert seen.folder == plant.owners_folder

    await owners_lookup()
    assert [sorted(count) for count in strict_counts] == [sorted([plant.owners_folder.name, copy.name])]
    assert _conflicted(plant.org_id, plant.artifact_id)
    shutil.rmtree(copy)
    artifact_identity._clear_identity_indexes()

    now[0] += artifact_scope._RECOUNT_SECONDS - 1
    await owners_lookup()
    assert len(strict_counts) == 1
    assert _conflicted(plant.org_id, plant.artifact_id)

    now[0] += 1
    await owners_lookup()
    assert strict_counts[1:] == [(plant.owners_folder.name,)]
    assert not _conflicted(plant.org_id, plant.artifact_id)


def test_recount_times_stay_bounded_and_forget_an_id_once_its_window_ends():
    now = [0.0]
    times = artifact_scope._RecountTimes(seconds=10, limit=2, clock=lambda: now[0])

    assert times.take(org_id="o", artifact_id="a")
    assert not times.take(org_id="o", artifact_id="a")
    assert times.take(org_id="p", artifact_id="a")
    now[0] = 5
    # A third id pushes out the oldest, which may then be counted again.
    assert times.take(org_id="o", artifact_id="b")
    assert times.take(org_id="o", artifact_id="a")
    now[0] = 14
    assert not times.take(org_id="o", artifact_id="b")
    now[0] = 15
    assert times.take(org_id="o", artifact_id="b")


def test_a_strict_count_of_the_rows_folder_alone_clears_a_conflicted_mark(tmp_path):
    """record_checked_folders is also the backfill's writer, whose walk is a
    strict count of every root."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id = uuid4().hex
    brief, copy = Location(project.id, None, "brief"), Location(project.id, None, "brief-copy")
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[brief, copy])
        assert _conflicted(org_id, artifact_id)

        outcome = record_checked_folders(session, artifact_id=artifact_id, folders=[brief])

    assert outcome is artifact_locations.RecordOutcome.RECORDED
    assert _row(org_id, artifact_id) == brief
    assert not _conflicted(org_id, artifact_id)


async def test_a_turn_end_records_no_row_for_a_planted_folder(
    tmp_path, org_deployment, backfill_pending, entry_point,
):
    """The owner's legacy artifact has no row before the backfill, so a turn
    end, which sees only its own folders, could not tell the plant is a
    second folder for the id."""
    plant = _owners_artifact(tmp_path)
    _plant(plant)

    _record_turn(
        plant.org_id, project=plant.project,
        conversation=make_conversation(plant.project, plant.member), creator=plant.member,
    )

    assert _row(plant.org_id, plant.artifact_id) is None
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


async def test_a_turn_end_after_the_backfill_marks_a_recorded_id_conflicted(
    tmp_path, org_deployment, backfill_pending, entry_point,
):
    plant = _owners_artifact(tmp_path)
    artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=[plant.project.id])
    assert _row(plant.org_id, plant.artifact_id) is not None
    _plant(plant)

    _record_turn(
        plant.org_id, project=plant.project,
        conversation=make_conversation(plant.project, plant.member), creator=plant.member,
    )

    assert _conflicted(plant.org_id, plant.artifact_id)
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


async def test_a_copy_of_a_folder_without_a_row_answers_409(
    tmp_path, org_deployment, backfill_pending, entry_point,
):
    """The owner's turn ends while the backfill still walks, after it passed
    this project, so no row names the owner's folder. Once the pass has
    finished, a co-member's turn copies the folder with the same id. Neither
    turn end writes a row, so the owner's lookup lists the roots, finds both
    folders and answers 409 rather than being served the copy."""
    org_id, owner, member = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id, _folder = _write(project_root(project), "brief")
    _record_turn(org_id, project=project, conversation=make_conversation(project, owner), creator=owner)
    # The pass finishes; it walked this project before the folder existed.
    artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=[])
    assert _sentinel_present()

    _write(project_root(project), "brief-copy", artifact_id)
    with scoped(org_id) as session:
        record_new_artifacts(
            session, conversation_id=make_conversation(project, member), project_id=project.id,
            slugs=["brief-copy"], creator=member, artifact_ids={"brief-copy": artifact_id},
        )
    artifact_identity._clear_identity_indexes()

    assert _row(org_id, artifact_id) is None
    assert await _refused(org_id, owner, project, artifact_id, entry=entry_point) == 409
    assert _conflicted(org_id, artifact_id)


def _failing_backfill_read(monkeypatch, plant: _Plant, *, failure: str) -> None:
    """Make the backfill's walk fail to list the conversation workspaces, or
    to read the owner's metadata, with EIO."""
    if failure == "listing":
        conversations = Path(plant.project.path) / "conversations"
        real_iterdir = Path.iterdir

        def iterdir(path):
            if path == conversations:
                raise OSError(errno.EIO, "Input/output error")
            return real_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", iterdir)
        return
    real_read_full_id = artifact_identity.read_full_id

    def read_full_id(folder):
        if folder.path.name == plant.owners_folder.name:
            raise OSError(errno.EIO, "Input/output error")
        return real_read_full_id(folder)

    monkeypatch.setattr(artifact_identity, "read_full_id", read_full_id)


@pytest.mark.parametrize("failure", ["listing", "metadata"])
async def test_a_backfill_that_cannot_read_a_folder_records_nothing_for_its_project(
    tmp_path, org_deployment, backfill_pending, monkeypatch, caplog, failure,
):
    """A read error is not an empty folder. The pass sees only the plant, so
    it records no row for the project and writes no sentinel, and the owner
    keeps the 409."""
    plant = _owners_artifact(tmp_path)
    _plant(plant)

    with monkeypatch.context() as failing, caplog.at_level(logging.WARNING, logger=artifact_location_backfill.logger.name):
        _failing_backfill_read(failing, plant, failure=failure)
        summary = artifact_location_backfill.run_artifact_location_backfill(
            try_lock=_no_lock, project_ids=[plant.project.id]
        )

    assert _row(plant.org_id, plant.artifact_id) is None
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id) == 409
    assert not _sentinel_present()
    assert summary is not None and summary.unread == 1
    [unread] = [
        record for record in caplog.records
        if record.name == artifact_location_backfill.logger.name
        and getattr(record, "event", None) == "artifact_location_backfill_unread"
    ]
    assert unread.project_id == str(plant.project.id)


def test_a_link_or_unparseable_metadata_holds_no_id_for_the_backfill(tmp_path, org_deployment, backfill_pending):
    """Lookups skip each of these, so the pass does too, and finishes: a
    folder an agent links or breaks must not keep it from writing its
    sentinel. That includes metadata that parses to something other than an
    object, and an array nested too deeply to parse."""
    org_id, project, ids = _org_projects(tmp_path, artifacts=1)
    base = project_root(project).base
    (base / "linked").symlink_to(base / "shared-0", target_is_directory=True)
    (base / "linked-metadata").mkdir()
    (base / "linked-metadata" / "metadata.json").symlink_to(base / "shared-0" / "metadata.json")
    (base / "broken").mkdir()
    (base / "broken" / "metadata.json").write_text("{not json")
    for name, document in {
        "array": "[]", "number": "1", "null": "null", "string": '"x"', "nested": "[" * 200_000 + "]" * 200_000,
    }.items():
        (base / name).mkdir()
        (base / name / "metadata.json").write_text(document)

    summary = artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert _row(org_id, ids[0]) == Location(project.id, None, "shared-0")
    assert _sentinel_present()


async def test_the_backfill_keeps_an_id_two_folders_hold_conflicted(
    tmp_path, org_deployment, backfill_pending, entry_point,
):
    plant = _owners_artifact(tmp_path)
    _plant(plant)

    summary = artifact_location_backfill.run_artifact_location_backfill(
        try_lock=_no_lock, project_ids=[plant.project.id]
    )
    assert summary is not None and summary.duplicated == [plant.artifact_id]
    _record_turn(
        plant.org_id, project=plant.project,
        conversation=make_conversation(plant.project, plant.member), creator=plant.member,
    )
    seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id, entry=entry_point)

    assert seen.folder == plant.planted
    assert _conflicted(plant.org_id, plant.artifact_id)
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


async def test_the_backfill_marks_a_row_left_at_one_of_two_folders_conflicted(
    tmp_path, org_deployment, backfill_pending, entry_point,
):
    """A row written at the plant before the backfill, by a writer that saw
    only the plant, stops naming a folder the owner is served."""
    plant = _owners_artifact(tmp_path)
    _plant(plant)
    with scoped(plant.org_id) as session:
        record_checked_folders(session, artifact_id=plant.artifact_id, folders=[Location(plant.project.id, None, plant.planted.name)])

    artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=[plant.project.id])

    assert _conflicted(plant.org_id, plant.artifact_id)
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


async def test_a_granted_copy_in_a_co_members_workspace_never_replaces_the_owners_folder(
    tmp_path, org_deployment, monkeypatch, entry_point,
):
    """The co-member's agent writes the owner's id into the co-member's own
    conversation workspace, with a review grant it wrote itself. The
    co-member's lookup must not record that folder, or the owner's next
    lookup would read the row and be served it as a reviewer."""
    plant = _owners_artifact(tmp_path)
    member_conversation = make_conversation(plant.project, plant.member)
    _, planted = _write(legacy_root(plant.project, str(member_conversation)), "planted", plant.artifact_id)
    monkeypatch.setattr(artifact_draft_review, "draft_review_allows", lambda path, _org: Path(path) == planted)

    seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id, entry=entry_point)
    assert (seen.folder, seen.is_own) == (planted, True)
    owners = await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point)

    assert (owners.folder, owners.is_own) == (plant.owners_folder, True)
    assert _conflicted(plant.org_id, plant.artifact_id)


async def test_the_recorded_folder_counts_toward_a_conflict_when_the_scan_misses_it(
    tmp_path, org_deployment, monkeypatch, entry_point,
):
    """The owner's row names their legacy folder, which the co-member opens
    but may not be served. When listing the conversation workspaces fails
    during that lookup, its scan sees only the shared root and the plant
    there. The folder the row's open found still counts, so the id is marked
    conflicted, and the owner gets the 409 rather than a row that ignores the
    plant."""
    plant = _owners_artifact(tmp_path)
    first = await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point)
    assert first.folder == plant.owners_folder
    _plant(plant)
    conversations = Path(plant.project.path) / "conversations"
    real_iterdir = Path.iterdir

    def iterdir(path):
        if path == conversations:
            raise OSError(errno.EIO, "Input/output error")
        return real_iterdir(path)

    with monkeypatch.context() as unlisted:
        unlisted.setattr(Path, "iterdir", iterdir)
        seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id, entry=entry_point)

    assert (seen.folder, seen.is_own) == (plant.planted, True)
    assert _conflicted(plant.org_id, plant.artifact_id)
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id, entry=entry_point) == 409


def _failing_reads(monkeypatch, *, folder_name: str) -> dict[str, bool]:
    """Make every read of `folder_name`'s identity raise EIO while the
    returned switch is on."""
    real = artifact_identity.ensure_full_id
    switch = {"on": True}

    def ensure_full_id(folder, *args, **kwargs):
        if switch["on"] and Path(folder).name == folder_name:
            raise OSError(errno.EIO, "Input/output error")
        return real(folder, *args, **kwargs)

    monkeypatch.setattr(artifact_identity, "ensure_full_id", ensure_full_id)
    artifact_identity._clear_identity_indexes()
    return switch


async def test_a_read_error_on_the_recorded_folder_leaves_its_row(tmp_path, org_deployment, monkeypatch):
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id, folder = _write(project_root(project), "brief")
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(project.id, None, "brief")])
    failing = _failing_reads(monkeypatch, folder_name="brief")

    assert await _refused(org_id, user, project, artifact_id) == 404

    assert _row(org_id, artifact_id) == Location(project.id, None, "brief")
    failing["on"] = False
    assert (await _review(org_id, user, project, artifact_id)).folder == folder


async def test_a_co_members_lookup_never_moves_a_private_row(tmp_path, org_deployment, monkeypatch):
    """The owner renamed their folder, and the co-member's lookup cannot read
    the renamed one. Only the owner, who reads that root as their own, may
    move the row."""
    plant = _owners_artifact(tmp_path)
    await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id)
    recorded = _row(plant.org_id, plant.artifact_id)
    renamed = plant.owners_folder.with_name("owners-draft-2")
    plant.owners_folder.rename(renamed)
    _plant(plant)
    failing = _failing_reads(monkeypatch, folder_name=renamed.name)

    seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id)

    assert seen.folder == plant.planted
    assert _row(plant.org_id, plant.artifact_id) == recorded
    failing["on"] = False
    # Two folders hold the id, and the owner sees both.
    assert await _refused(plant.org_id, plant.owner, plant.project, plant.artifact_id) == 409


async def test_a_row_left_in_another_project_follows_the_folder_the_scan_finds(
    tmp_path, org_deployment, monkeypatch,
):
    """A task move whose row update failed leaves the row naming the old
    project. A lookup in the new one opens the old folder without serving
    it, finds it gone, and moves the row, so the next lookup lists nothing."""
    org_id, user = str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    artifact_id, folder = _write(project_root(dest), "brief")
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(source.id, None, "brief")])

    resolved = await _review(org_id, user, dest, artifact_id)

    assert resolved.folder == folder
    assert _row(org_id, artifact_id) == Location(dest.id, None, "brief")
    calls = _FileCalls(monkeypatch)
    calls.counting = True
    assert (await _review(org_id, user, dest, artifact_id)).folder == folder
    calls.counting = False
    assert calls.listings == 0


def test_deleting_a_conversation_finds_its_rows_by_an_index():
    """conversation_id carries an ON DELETE CASCADE, which Postgres does not
    index by itself."""
    assert [
        index.name for index in ArtifactLocation.__table__.indexes
        if list(index.columns.keys()) == ["conversation_id"]
    ] == ["ix_artifact_locations_conversation_id"]


def _org_projects(tmp_path, *, legacy_roots: int = 0, artifacts: int = 1):
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    ids = [_write(project_root(project), f"shared-{n}")[0] for n in range(artifacts)]
    for _ in range(legacy_roots):
        _write(legacy_root(project, str(make_conversation(project, user))), "old")
    return org_id, project, ids


def test_shutdown_stops_the_backfill_at_the_next_root(tmp_path, org_deployment, backfill_pending, monkeypatch):
    _org_id, project, _ids = _org_projects(tmp_path, legacy_roots=3)
    stop = threading.Event()
    real = artifact_location_backfill.identities_below
    walked = []

    def identities_below(source):
        walked.append(source)
        stop.set()
        return real(source)

    monkeypatch.setattr(artifact_location_backfill, "identities_below", identities_below)

    assert artifact_location_backfill.run_artifact_location_backfill(
        try_lock=_no_lock, project_ids=[project.id], stop=stop
    ) is None
    assert len(walked) == 1
    assert not _sentinel_present()


def test_shutdown_stops_the_backfill_at_the_next_id(tmp_path, org_deployment, backfill_pending, monkeypatch):
    org_id, project, ids = _org_projects(tmp_path, artifacts=3)
    stop = threading.Event()
    real = artifact_location_backfill.record_checked_folders

    def record_checked_folders(*args, **kwargs):
        stop.set()
        return real(*args, **kwargs)

    monkeypatch.setattr(artifact_location_backfill, "record_checked_folders", record_checked_folders)

    assert artifact_location_backfill.run_artifact_location_backfill(
        try_lock=_no_lock, project_ids=[project.id], stop=stop
    ) is None
    assert sum(_row(org_id, artifact_id) is not None for artifact_id in ids) == 1
    assert not _sentinel_present()


def test_a_refused_write_ends_the_backfill_without_its_sentinel(
    tmp_path, org_deployment, backfill_pending, monkeypatch,
):
    """A database that stays busy or unreachable is not a conflict: the pass
    stops at the first refused write, and the next start walks again."""
    _org_id, project, _ids = _org_projects(tmp_path, artifacts=3)
    attempts = []

    def refused(*_args, **_kwargs):
        attempts.append(1)
        raise OperationalError("INSERT INTO artifact_locations", {}, Exception("server closed the connection"))

    monkeypatch.setattr(artifact_location_backfill, "record_checked_folders", refused)

    with pytest.raises(OperationalError):
        artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=[project.id])
    assert len(attempts) == 1
    assert not _sentinel_present()


def test_a_folder_whose_project_went_away_during_the_backfill_counts_as_a_conflict(
    tmp_path, org_deployment, backfill_pending, monkeypatch,
):
    org_id, project, ids = _org_projects(tmp_path, artifacts=2)
    real = artifact_location_backfill.record_checked_folders

    def record_checked_folders(session, *, artifact_id, **kwargs):
        if artifact_id == ids[0]:
            raise IntegrityError("INSERT INTO artifact_locations", {}, Exception("foreign key"))
        return real(session, artifact_id=artifact_id, **kwargs)

    monkeypatch.setattr(artifact_location_backfill, "record_checked_folders", record_checked_folders)

    summary = artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.conflicts) == (1, 1)
    assert _row(org_id, ids[1]) == Location(project.id, None, "shared-1")
    assert _sentinel_present()


def test_moving_a_task_moves_the_rows_of_its_folders(tmp_path, org_deployment):
    org_id, owner = str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation = make_conversation(source, owner)
    artifact_id, _ = _write(project_root(source), "mine")
    _index(conversation, source.id, "mine")
    with scoped(org_id) as session:
        artifact_ownership.record_artifact_owner(session, source.id, "mine", owner, action="create")
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(source.id, None, "mine")])

    assert _relocate(org_id, owner, conversation, source.id, dest.id) == 1

    assert _row(org_id, artifact_id) == Location(dest.id, None, "mine")


async def test_an_in_process_turn_records_the_ids_of_the_folders_it_made(turns, monkeypatch):  # noqa: F811
    artifact_id = uuid4().hex
    monkeypatch.setattr(
        task_objects, "turn_artifact_changes",
        lambda **_k: task_objects.ArtifactChanges(
            created=["sales-report"], touched={"sales-report"}, created_ids={"sales-report": artifact_id},
        ),
    )

    async def no_cards(*_a, **_k):
        return []

    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", no_cards)
    real_run_db = harness_module.run_db
    recorded = {}

    async def run_db(fn, *, scope):
        if getattr(fn, "func", None) is task_objects.record_new_artifacts:
            recorded.update(fn.keywords)
            return None
        return await real_run_db(fn, scope=scope)

    monkeypatch.setattr(harness_module, "run_db", run_db)

    await _drain(turns.run(_completes))

    assert recorded["artifact_ids"] == {"sales-report": artifact_id}


async def test_the_lifespan_runs_the_location_backfill_in_org_tenancy_and_stops_it_first(boot, monkeypatch):  # noqa: F811
    app, _order, _probe = boot(tenancy="org")
    started = asyncio.Event()
    seen = {}

    async def location_backfill(stop):
        seen["stop"] = stop
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A started file store call is never abandoned, so the walk must
            # be told to stop before its task is cancelled.
            seen["stopped_before_cancel"] = stop.is_set()
            raise

    monkeypatch.setattr(server, "_run_artifact_location_backfill", location_backfill)

    async with asyncio.timeout(5):
        async with server.lifespan(app):
            await started.wait()
            assert not seen["stop"].is_set()
    assert seen["stopped_before_cancel"] is True


async def test_a_folder_the_server_may_not_read_holds_no_id_for_the_strict_count(
    tmp_path, org_deployment, monkeypatch,
):
    """Agent code can make a folder of the shared root unreadable to the
    server (mode 000). The server can never serve that folder, so the strict
    count takes it for one holding no id: the first lookup writes the row,
    and the next one opens only the row's folder."""
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id, folder = _write(project_root(project), "brief")
    _write(project_root(project), "decoy")
    artifact_identity._clear_identity_indexes()
    _unreadable(monkeypatch, "decoy", strict_too=True, error=errno.EACCES)

    assert (await _review(org_id, user, project, artifact_id)).folder == folder
    assert _row(org_id, artifact_id) == Location(project.id, None, "brief")

    calls = _FileCalls(monkeypatch)
    calls.counting = True
    assert (await _review(org_id, user, project, artifact_id)).folder == folder
    calls.counting = False
    assert calls.listings == 0


@pytest.mark.parametrize(("error", "raises"), [(errno.EACCES, False), (errno.EPERM, False), (errno.EIO, True)])
def test_strict_root_discovery_leaves_out_a_workspace_the_server_may_not_read(
    tmp_path, org_deployment, monkeypatch, error, raises,
):
    """A conversation workspace whose `.anton` the server may not stat is no
    root of the strict count, since nothing below it can be served; an error
    that may pass, such as EIO, still fails the count."""
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    readable, locked = make_conversation(project, user), make_conversation(project, user)
    for conversation in (readable, locked):
        _write(legacy_root(project, str(conversation)), "old")
    locked_anton = Path(project.path) / "conversations" / str(locked) / ".anton"
    real_stat = Path.stat

    def stat(path, *args, **kwargs):
        if path == locked_anton:
            raise OSError(error, os.strerror(error))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    row = artifact_roots.ProjectRow(id=project.id, name=project.name, path=project.path, external=False)

    if raises:
        with pytest.raises(OSError):
            artifact_roots.project_roots(row, strict=True)
        return
    roots = artifact_roots.project_roots(row, strict=True)
    assert [root.conversation_id for root in roots if root.legacy] == [readable]


def test_a_backfill_walks_again_only_the_project_it_could_not_read(
    tmp_path, org_deployment, backfill_pending, monkeypatch,
):
    """One project of three fails to read. The pass records the other two
    and keeps the third in its progress; the next start walks only that one,
    and once it reads, the pass writes its sentinel."""
    projects = [_org_projects(tmp_path, artifacts=1) for _ in range(3)]
    ids = [project.id for _org_id, project, _ids in projects]
    unreadable = ids[1]
    walked: list[UUID] = []
    failing = threading.Event()
    failing.set()
    real = artifact_location_backfill._found_locations

    def found_locations(project, *, stop):
        walked.append(project.id)
        if failing.is_set() and project.id == unreadable:
            raise OSError(errno.EIO, "Input/output error")
        return real(project, stop=stop)

    monkeypatch.setattr(artifact_location_backfill, "_found_locations", found_locations)

    def run():
        walked.clear()
        return artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=ids)

    first = run()
    assert sorted(walked) == sorted(ids)
    assert first is not None and (first.recorded, first.unread) == (2, 1)
    assert not _sentinel_present()

    second = run()
    assert walked == [unreadable]
    assert second is not None and second.unread == 1
    assert not _sentinel_present()

    failing.clear()
    third = run()
    assert walked == [unreadable]
    assert third is not None and (third.recorded, third.unread) == (1, 0)
    assert _sentinel_present()
    for org_id, project, artifact_ids in projects:
        assert _row(org_id, artifact_ids[0]) == Location(project.id, None, "shared-0")


def test_a_stopped_backfill_resumes_after_the_last_project_it_finished(
    tmp_path, org_deployment, backfill_pending, monkeypatch, caplog,
):
    projects = [_org_projects(tmp_path, artifacts=1) for _ in range(3)]
    ids = sorted(project.id for _org_id, project, _ids in projects)
    stop = threading.Event()
    walked: list[UUID] = []
    real_found, real_save = artifact_location_backfill._found_locations, artifact_location_backfill._save_progress

    def found_locations(project, *, stop):
        walked.append(project.id)
        return real_found(project, stop=stop)

    def save_progress(engine, *, progress):
        # Shutdown lands once the first project is finished.
        real_save(engine, progress=progress)
        stop.set()

    monkeypatch.setattr(artifact_location_backfill, "_found_locations", found_locations)
    monkeypatch.setattr(artifact_location_backfill, "_save_progress", save_progress)
    with caplog.at_level(logging.WARNING, logger=artifact_location_backfill.logger.name):
        assert artifact_location_backfill.run_artifact_location_backfill(
            try_lock=_no_lock, project_ids=ids, stop=stop
        ) is None

    assert walked == ids[:1]
    assert not _sentinel_present()
    [stopped] = [
        record for record in caplog.records
        if record.name == artifact_location_backfill.logger.name and record.levelno == logging.WARNING
        and getattr(record, "event", None) == "artifact_location_backfill_stopped"
    ]
    assert "resumes" in stopped.getMessage()

    monkeypatch.setattr(artifact_location_backfill, "_save_progress", real_save)
    walked.clear()
    summary = artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=ids)

    assert walked == ids[1:]
    assert summary is not None and summary.recorded == 2
    assert _sentinel_present()
    for org_id, project, artifact_ids in projects:
        assert _row(org_id, artifact_ids[0]) == Location(project.id, None, "shared-0")


@pytest.mark.parametrize("bad_first", [False, True], ids=["good_first", "bad_first"])
def test_a_folder_name_that_is_not_utf8_counts_toward_a_conflict(tmp_path, bad_first):
    """Linux hands Python a folder name that is not valid UTF-8 with a lone
    surrogate, which no row can store. It still counts, so the id is marked
    conflicted, and the row names the folder whose name it can store,
    whichever the writer found first."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id = uuid4().hex
    good, bad = Location(project.id, None, "brief"), Location(project.id, None, "brief\udcff")
    with scoped(org_id) as session:
        outcome = record_checked_folders(
            session, artifact_id=artifact_id, folders=[bad, good] if bad_first else [good, bad],
        )

    assert outcome is artifact_locations.RecordOutcome.CONFLICTED
    assert _row(org_id, artifact_id) == good
    assert _conflicted(org_id, artifact_id)


def test_a_lone_folder_name_that_is_not_utf8_writes_no_row(tmp_path):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id = uuid4().hex
    with scoped(org_id) as session:
        outcome = record_checked_folders(
            session, artifact_id=artifact_id, folders=[Location(project.id, None, "brief\udcff")],
        )

    assert outcome is artifact_locations.RecordOutcome.NOTHING
    assert _stored(org_id, artifact_id) is None


def test_a_backfill_counts_a_folder_name_that_is_not_utf8_toward_a_conflict(
    tmp_path, org_deployment, backfill_pending, monkeypatch,
):
    """The walk finds the id in a good folder and in one whose name is not
    valid UTF-8 (APFS cannot hold such a name, so the walk's listing adds
    it). The id is marked conflicted at the good folder, and the pass
    finishes."""
    org_id, project, ids = _org_projects(tmp_path, artifacts=1)
    real = artifact_location_backfill.identities_below

    def identities_below(source):
        return {
            artifact_id: (*names, "shared-0\udcff") if artifact_id == ids[0] else names
            for artifact_id, names in real(source).items()
        }

    monkeypatch.setattr(artifact_location_backfill, "identities_below", identities_below)

    summary = artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and summary.duplicated == [ids[0]]
    assert _row(org_id, ids[0]) == Location(project.id, None, "shared-0")
    assert _conflicted(org_id, ids[0])
    assert _sentinel_present()


def _progress_present() -> bool:
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        return artifact_location_backfill._progress_row(raw) is not None


def test_a_backfill_drops_an_unread_project_deleted_since_and_finishes(
    tmp_path, org_deployment, backfill_pending, monkeypatch,
):
    """A project the pass could not read is deleted before the next start.
    The query never returns it again, so the pass drops it from its
    progress, writes its sentinel and deletes the progress row."""
    projects = [_org_projects(tmp_path, artifacts=1) for _ in range(2)]
    ids = [project.id for _org_id, project, _ids in projects]
    # The first in id order, so the kept progress is already past it.
    unreadable = min(ids)
    real = artifact_location_backfill._found_locations

    def found_locations(project, *, stop):
        if project.id == unreadable:
            raise OSError(errno.EIO, "Input/output error")
        return real(project, stop=stop)

    monkeypatch.setattr(artifact_location_backfill, "_found_locations", found_locations)
    first = artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=ids)
    assert first is not None and (first.recorded, first.unread) == (1, 1)
    assert not _sentinel_present() and _progress_present()
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        raw.delete(raw.get(Project, unreadable))
        raw.commit()

    second = artifact_location_backfill.run_artifact_location_backfill(try_lock=_no_lock, project_ids=ids)

    assert second is not None and second.unread == 0
    assert _sentinel_present()
    assert not _progress_present()


@pytest.mark.parametrize(
    "refusal",
    [
        StaleDataError("UPDATE statement on table 'artifact_locations' expected to update 1 row(s); 0 were matched."),
        DataError("INSERT INTO artifact_locations", {}, Exception("value too long for type character varying(255)")),
    ],
    ids=["deleted_mid_write", "refused_value"],
)
def test_a_backfill_skips_an_id_it_cannot_write_and_finishes(
    tmp_path, org_deployment, backfill_pending, monkeypatch, caplog, refusal,
):
    org_id, project, ids = _org_projects(tmp_path, artifacts=3)
    real = artifact_location_backfill.record_checked_folders

    def record_checked_folders(session, *, artifact_id, **kwargs):
        if artifact_id == ids[1]:
            raise refusal
        return real(session, artifact_id=artifact_id, **kwargs)

    monkeypatch.setattr(artifact_location_backfill, "record_checked_folders", record_checked_folders)

    with caplog.at_level(logging.WARNING, logger=artifact_location_backfill.logger.name):
        summary = artifact_location_backfill.run_artifact_location_backfill(
            try_lock=_no_lock, project_ids=[project.id]
        )

    assert summary is not None and (summary.recorded, summary.skipped) == (2, 1)
    assert _row(org_id, ids[1]) is None
    assert _row(org_id, ids[0]) is not None and _row(org_id, ids[2]) is not None
    assert _sentinel_present()
    [skipped] = [
        record for record in caplog.records
        if record.name == artifact_location_backfill.logger.name
        and getattr(record, "event", None) == "artifact_location_backfill_skipped"
    ]
    assert (skipped.project_id, skipped.error_type) == (str(project.id), type(refusal).__name__)
    assert ("skipped", 1) in {
        (record.outcome, record.count)
        for record in caplog.records
        if record.name == artifact_location_backfill.logger.name
        and getattr(record, "event", None) == "artifact_location_backfill_summary"
    }


async def test_an_owners_lookup_of_a_conflicted_id_reads_only_the_owners_roots(
    tmp_path, org_deployment, monkeypatch,
):
    """The owner's legacy artifact has a row. A co-member's agent writes a
    folder holding its id into the co-member's own workspace, and the
    co-member's lookup marks the id conflicted. The owner's lookups, which
    answer from the owner's own roots first, find and read only those once
    the first has counted every root strictly and found the copy: inside that
    count's window, 20 more co-member workspaces cost them nothing, not even
    a stat, and nothing they find can change the row, so no record unit
    runs."""
    plant = _owners_artifact(tmp_path)
    await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id)
    member_conversation = make_conversation(plant.project, plant.member)
    _write(legacy_root(plant.project, str(member_conversation)), "copy", plant.artifact_id)
    members = [member_conversation, *(make_conversation(plant.project, plant.member) for _ in range(20))]
    for conversation in members[1:]:
        _write(legacy_root(plant.project, str(conversation)), "other")
    artifact_identity._clear_identity_indexes()
    seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id)
    assert (seen.folder.name, seen.is_own) == ("copy", True)
    assert _conflicted(plant.org_id, plant.artifact_id)
    # Warm the owner's cached indexes.
    assert (await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id)).folder == plant.owners_folder

    # The own-first shape, measured here: check the row's project sits in the
    # caller's partition, open the row's folder, discover the roots, and read
    # the owner's own.
    from cowork.services.projects import sits_in_another_partition

    row = _row(plant.org_id, plant.artifact_id)
    with scoped(plant.org_id, plant.owner) as session:
        plan = artifact_scope._plan_for_project_ref(session, project_ref=str(plant.project.id))
    rows_root = legacy_root(plant.project, str(row.conversation_id))
    calls = _FileCalls(monkeypatch)
    calls.counting = True
    sits_in_another_partition(str(rows_root.trusted_anchor), partition=plan.partition)
    artifact_identity.resolve_recorded_folder(rows_root, row.slug, plant.artifact_id)
    found = artifact_roots.discover_roots(plan, own_only=True)
    artifact_identity.artifact_folders_holding(found.sources(), artifact_id=plant.artifact_id)
    calls.counting = False
    own_first, calls.count = calls.count, 0

    async def no_record(*_args, **_kwargs):
        raise AssertionError("an owner's lookup of a conflicted id ran a record unit")

    monkeypatch.setattr(artifact_scope, "_record", no_record)
    touched = _touched_by_discovery(monkeypatch)
    calls.counting = True
    resolved = await _review(plant.org_id, plant.owner, plant.project, plant.artifact_id)
    calls.counting = False

    assert (resolved.folder, resolved.is_own) == (plant.owners_folder, True)
    assert calls.count <= own_first, f"{calls.count} filesystem calls against {own_first} for the own-first shape"
    workspaces = [Path(plant.project.path) / "conversations" / str(conversation) for conversation in members]
    assert _in_workspaces(touched, workspaces) == []
    # The co-member's lookup reads the row's folder and their own copy, and
    # records nothing either: the id is already conflicted.
    seen = await _review(plant.org_id, plant.member, plant.project, plant.artifact_id)
    assert (seen.folder.name, seen.is_own) == ("copy", True)
    assert _conflicted(plant.org_id, plant.artifact_id)


@pytest.mark.parametrize("a_folder", ["holds_the_id", "fails_to_read"])
async def test_a_lookup_from_a_project_the_row_does_not_name_records_nothing(
    tmp_path, org_deployment, monkeypatch, a_folder,
):
    """Projects A and B of one organization both hold a folder naming the
    id, and the row names A's. A lookup from B is served B's folder from a
    scan of B's roots and runs no record unit, since the row keeps naming a
    folder of A whatever B holds; the row stays at A. That holds too when A's
    folder fails to read during the lookup."""
    org_id, user = str(uuid4()), str(uuid4())
    a, b = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    artifact_id, _ = _write(project_root(a), "brief-a")
    _, b_folder = _write(project_root(b), "brief-b", artifact_id)
    with scoped(org_id) as session:
        record_checked_folders(session, artifact_id=artifact_id, folders=[Location(a.id, None, "brief-a")])
    if a_folder == "fails_to_read":
        _failing_reads(monkeypatch, folder_name="brief-a")
    hops = _count_hops(monkeypatch, artifact_scope)

    for _ in range(2):
        assert (await _review(org_id, user, b, artifact_id)).folder == b_folder

    assert hops == ["run_db", "run_efs"] * 2
    assert _row(org_id, artifact_id) == Location(a.id, None, "brief-a")


def _within(seconds: float, *, fifo: Path, call):
    """``call()`` on a daemon thread, failing the test when it has not
    returned within ``seconds``; a writer then opens ``fifo`` so a reader
    stuck on it returns and the suite goes on."""
    outcome: dict = {}

    def run():
        try:
            outcome["value"] = call()
        except BaseException as exc:  # noqa: BLE001  (re-raised on the test's thread)
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        # ENXIO when no reader holds the FIFO open: the call is slow, not stuck.
        with contextlib.suppress(OSError):
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        thread.join(5)
        pytest.fail(f"a read of the FIFO had not returned after {seconds}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def test_a_fifo_in_place_of_metadata_holds_no_id_and_holds_no_thread(tmp_path, org_deployment):
    """An agent can replace an artifact's metadata.json with a FIFO, which no
    writer on the server's node ever opens. Each read of it returns at once:
    the list leaves the artifact out, the row's folder reads as stale, and a
    cached index that still names the folder drops it."""
    from cowork.services import artifacts as artifacts_service

    project = make_project(tmp_path, str(uuid4()))
    source = project_root(project)
    artifact_id, folder = _write(source, "brief")
    other_id, _ = _write(source, "other")
    artifact_identity._clear_identity_indexes()
    # Warm the index, and the imports the list and its skip warning make on
    # first use (the log filter imports the provider SDK), so the bounds
    # below time only the reads.
    assert len(artifact_identity.artifact_folders_holding([source], artifact_id=artifact_id)) == 1
    assert len(artifacts_service.list_artifacts([source])) == 2
    importlib.import_module("anthropic")
    metadata = folder / "metadata.json"
    metadata.unlink()
    os.mkfifo(metadata)

    listed = _within(1, fifo=metadata, call=lambda: artifacts_service.list_artifacts([source]))
    assert [card["id"] for card in listed] == [other_id]
    recorded = _within(1, fifo=metadata, call=lambda: artifact_identity.resolve_recorded_folder(source, "brief", artifact_id))
    assert recorded.check is FolderCheck.STALE
    held = _within(1, fifo=metadata, call=lambda: artifact_identity.artifact_folders_holding([source], artifact_id=artifact_id))
    assert held == []


def test_a_fifo_in_place_of_metadata_gives_no_provenance_origin_at_once(tmp_path):
    """The owner backfill and a turn's new-artifact check read a folder's
    first conversation through provenance_origin. A FIFO in place of its
    metadata.json, which no writer opens, reads as no origin at once instead
    of holding the open."""
    folder = tmp_path / "brief"
    folder.mkdir()
    metadata = folder / "metadata.json"
    os.mkfifo(metadata)

    assert _within(1, fifo=metadata, call=lambda: artifact_ownership.provenance_origin(folder)) is None


async def test_shutdown_goes_on_while_the_location_backfill_is_stuck(boot, monkeypatch, caplog):  # noqa: F811
    """During a file store stall the walk can sit in one root's read when
    the teardown sets its stop, and a started file store call is never
    abandoned. The teardown waits at most LOCATION_BACKFILL_SHUTDOWN_SECONDS
    for it, then goes on to drain the channel turns and close the rest."""
    app, _order, _probe = boot(tenancy="org")
    walking, release = threading.Event(), threading.Event()

    def stuck_walk(*_args, **_kwargs):
        walking.set()
        # Ignores stop, as a read stuck on the mount does.
        release.wait(30)

    monkeypatch.setattr(artifact_location_backfill, "run_artifact_location_backfill", stuck_walk)
    loop = asyncio.get_running_loop()
    drained: list[float] = []

    async def drain_background_tasks(*_args, **_kwargs):
        drained.append(loop.time())

    monkeypatch.setattr("cowork.channels.webhooks.drain_background_tasks", drain_background_tasks)
    try:
        with caplog.at_level(logging.WARNING, logger=server.logger.name):
            async with asyncio.timeout(10):
                async with server.lifespan(app):
                    assert await asyncio.to_thread(walking.wait, 5)
                    stopping_at = loop.time()
        assert len(drained) == 1
        assert drained[0] - stopping_at <= server.LOCATION_BACKFILL_SHUTDOWN_SECONDS + 1
        assert [
            record for record in caplog.records
            if record.name == server.logger.name
            and getattr(record, "event", None) == "shutdown_location_backfill_unfinished"
        ]
    finally:
        release.set()
        backfill = app.state.artifact_location_backfill
        if backfill is not None:
            await asyncio.wait({backfill}, timeout=5)
