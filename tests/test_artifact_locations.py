"""Artifact lookups open the one folder a location row names.

Org tenancy only, where every listing and stat goes to the shared file
store's server. A lookup reads the artifact's row in its database step, then
opens that folder inside a root the caller may already read, and serves it
only when the folder's metadata names the requested id. Without a row, or
with a stale one, it lists the project's roots as before and records, or
drops, the row.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

from cowork.api.v1 import artifact_scope
from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.artifact_location import ArtifactLocation
from cowork.services import artifact_draft_review, artifact_identity, artifact_location_backfill
from cowork.services.artifact_identity import ArtifactIdentityConflict
from cowork.services.artifact_locations import (
    Location,
    move_locations,
    record_location,
)
from cowork.services.task_objects import record_new_artifacts, turn_artifact_changes
from test_artifact_ownership import legacy_root, make_conversation, make_project, project_root, scoped
from test_artifact_permissions import no_publish_side_effects  # noqa: F401  (a fixture)

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


def _row(org_id: str, artifact_id: str) -> Location | None:
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        row = raw.exec(
            select(ArtifactLocation).where(
                ArtifactLocation.org_id == org_id, ArtifactLocation.artifact_id == artifact_id
            )
        ).first()
        return Location(row.project_id, row.conversation_id, row.slug) if row is not None else None


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


async def _review(org_id: str, user_id: str, project, artifact_id: str):
    with scoped(org_id, user_id) as session:
        scope = session.scope
    return await artifact_scope.review_artifact_for_request(scope, str(project.id), artifact_id)


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
        record_location(session, artifact_id, Location(project.id, conversation, "draft"))

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


def test_a_second_folder_with_a_recorded_id_conflicts_and_keeps_the_first_row(tmp_path):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    artifact_id = uuid4().hex
    first = Location(project.id, None, "brief")
    with scoped(org_id) as session:
        record_location(session, artifact_id, first)
        record_location(session, artifact_id, first)  # the same folder again: no conflict
        with pytest.raises(ArtifactIdentityConflict):
            record_location(session, artifact_id, Location(project.id, None, "brief-copy"))

    assert _row(org_id, artifact_id) == first


def test_rows_are_per_organization(tmp_path):
    artifact_id = uuid4().hex
    for org_id in (str(uuid4()), str(uuid4())):
        project = make_project(tmp_path, org_id)
        with scoped(org_id) as session:
            record_location(session, artifact_id, Location(project.id, None, "brief"))
        assert _row(org_id, artifact_id) == Location(project.id, None, "brief")


async def test_a_row_whose_folder_holds_another_id_is_refused_and_dropped(tmp_path, org_deployment):
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    artifact_id = uuid4().hex
    _write(source, "brief", uuid4().hex)  # the recorded folder now holds another id
    with scoped(org_id) as session:
        record_location(session, artifact_id, Location(project.id, None, "brief"))
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
        record_location(session, artifact_id, Location(project.id, None, "brief"))
    artifact_identity._clear_identity_indexes()

    resolved = await _review(org_id, user, project, artifact_id)

    assert resolved.folder == folder
    assert _row(org_id, artifact_id) == Location(project.id, None, "renamed")


async def test_a_row_in_another_project_is_not_opened(tmp_path, org_deployment):
    """The row names a project the request didn't: the lookup scans the
    requested one, and the row stays."""
    org_id, user = str(uuid4()), str(uuid4())
    elsewhere, requested = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    artifact_id, _ = _write(project_root(elsewhere), "brief")
    with scoped(org_id) as session:
        record_location(session, artifact_id, Location(elsewhere.id, None, "brief"))

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


async def test_deleting_an_artifact_deletes_its_row(
    tmp_path, org_deployment, granted_product_permissions, no_publish_side_effects,  # noqa: F811
):
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    artifact_id, folder = _write(source, "mine")
    with scoped(org_id) as session:
        from cowork.services import artifact_ownership

        artifact_ownership.record_artifact_owner(session, project.id, "mine", creator, action="create")
        record_location(session, artifact_id, Location(project.id, None, "mine"))

    with scoped(org_id, creator) as session:
        await artifacts_ep.delete_artifact_for_request(session.scope, artifact_id, project_id=project.id)

    assert not folder.exists()
    assert _row(org_id, artifact_id) is None


def test_a_turns_new_folders_get_rows(tmp_path):
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation = make_conversation(project, creator)
    source = project_root(project)
    artifact_id, _ = _write(source, "made-this-turn")

    changes = turn_artifact_changes(conversation_id=conversation, artifacts_base=source.base, before=set(), before_mtimes={})
    assert changes.created_ids == {"made-this-turn": artifact_id}
    with scoped(org_id) as session:
        record_new_artifacts(
            session, conversation_id=conversation, project_id=project.id,
            slugs=changes.created, creator=creator, artifact_ids=changes.created_ids,
        )

    assert _row(org_id, artifact_id) == Location(project.id, None, "made-this-turn")


def test_a_moved_folders_row_follows_it(tmp_path):
    org_id = str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    artifact_id = uuid4().hex
    with scoped(org_id) as session:
        record_location(session, artifact_id, Location(source.id, None, "brief"))
        move_locations(session, project_id=source.id, moved={"brief": "abcd1234-brief"}, to_project_id=dest.id)

    assert _row(org_id, artifact_id) == Location(dest.id, None, "abcd1234-brief")


def test_the_backfill_records_every_artifact_held_by_one_folder(tmp_path, org_deployment):
    org_id, user = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation = make_conversation(project, user)
    shared, _ = _write(project_root(project), "shared")
    legacy, _ = _write(legacy_root(project, str(conversation)), "legacy")
    twice = uuid4().hex
    _write(project_root(project), "one", twice)
    _write(project_root(project), "two", twice)
    engine = get_engine(get_app_settings().database.uri)

    def no_lock(_engine, _key):
        from contextlib import nullcontext

        return nullcontext(True)

    summary = artifact_location_backfill.run_artifact_location_backfill(
        engine, try_lock=no_lock, project_ids=[project.id]
    )

    assert summary is not None and summary.recorded == 2 and summary.duplicated == [twice]
    assert _row(org_id, shared) == Location(project.id, None, "shared")
    assert _row(org_id, legacy) == Location(project.id, conversation, "legacy")
    assert _row(org_id, twice) is None
    # Once per environment.
    assert artifact_location_backfill.run_artifact_location_backfill(
        engine, try_lock=no_lock, project_ids=[project.id]
    ) is None
    with Session(engine) as raw:
        from cowork.models.setting import Setting

        sentinel = raw.exec(select(Setting).where(Setting.key == artifact_location_backfill.SENTINEL_KEY)).first()
        raw.delete(sentinel)
        raw.commit()


def test_a_stopped_backfill_writes_no_sentinel(tmp_path, org_deployment):
    import threading
    from contextlib import nullcontext

    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    stop = threading.Event()
    stop.set()
    engine = get_engine(get_app_settings().database.uri)

    assert artifact_location_backfill.run_artifact_location_backfill(
        engine, try_lock=lambda _e, _k: nullcontext(True), project_ids=[project.id], stop=stop
    ) is None
    assert not artifact_location_backfill._sentinel_present(engine, artifact_location_backfill.SENTINEL_KEY)
