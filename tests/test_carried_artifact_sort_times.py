"""The sort-time rows a comparison side's carry writes.

A comparison side continued into a project holds the legacy root its carried
workspace brought, when that root holds a folder, and records the sort time
of each folder the artifacts list would open there, the modification time of
its metadata.json (ComparisonService._hold_carried_workspace). The hold and
the times each have their own try, as the project files PUT's do: a failure
is rolled back and logged under its own event, the other is still written,
and the continue goes on. A folder whose metadata.json the read could not
reach is logged at WARNING, with the event a turn end's read logs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlmodel import Session

from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services import artifact_sort_times, comparisons
from cowork.services.artifact_sort_times import SortTime
from cowork.services.comparisons import ComparisonService, SideSpec
from cowork.services.held_artifact_roots import HeldRoot
from test_artifact_ownership import _engine
from test_held_root_turn_ends import _continue_a_side
from test_org_list_scaling import (  # noqa: F401  (fixtures)
    ALICE,
    _bulk_paths,
    _headers,
    _project_id,
    _write_artifact,
    client,
    held_roots,
    org,
)
from test_project_files_artifact_roots import _slugs, _sort_time_inserts_fail, _sort_times
from tests._artifact_rows import held_roots_of, no_self_heals_left  # noqa: F401  (no_self_heals_left is a fixture)
from tests._logging_assertions import assert_log_fields

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


def test_a_continued_side_records_the_sort_time_of_each_folder_its_workspace_brought(
    client, org, held_roots, owned_logger  # noqa: F811
):
    """Once the sort times' backfill has finished, the list shows a held
    root's folders from their rows alone, newest first, so the carry records
    the time of each folder the list would open, read from its
    metadata.json. The carry moves the workspace, so each file keeps its
    time. A folder with no metadata.json, for which the list shows no card,
    gets no time, and neither does a dot-named one, which the list never
    opens. A missing metadata.json is no failed read, so nothing is
    logged."""
    held_roots()
    logged = owned_logger(comparisons.logger.name, level=logging.WARNING)

    def fill(artifacts: Path) -> None:
        for slug, mtime in (("older", 1_000_000.0), ("newer", 2_000_000.0), (".hidden", 3_000_000.0)):
            _write_artifact(artifacts, slug)
            os.utime(artifacts / slug / "metadata.json", (mtime, mtime))
        (artifacts / "no-metadata").mkdir()

    destination_id, conversation_id = _continue_a_side(client, org, artifact=False, fill=fill)

    assert held_roots_of(org) == {HeldRoot(destination_id, conversation_id)}
    assert _sort_times(org) == {
        SortTime(destination_id, conversation_id, "older", 1_000_000.0),
        SortTime(destination_id, conversation_id, "newer", 2_000_000.0),
    }
    assert _slugs(client, org, ALICE, f"?project_id={destination_id}") == ["newer", "older"]
    assert logged.output() == ""


def test_a_continue_through_its_route_records_the_carried_root_on_the_file_store_threads(
    client, org, held_roots, owned_logger  # noqa: F811
):
    """The route runs the continue as one file store unit (run_efs_unit), so
    the carry's read of the folders it brought runs there too, where the
    blocking guard allows it (tests/_blocking_guard.py)."""
    held_roots()
    [path] = _bulk_paths(client, org, 0, 1)
    path.mkdir(parents=True, exist_ok=True)
    destination_id = _project_id(org, path)
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org, user_id=ALICE))
        comparison = ComparisonService(session).create_comparison(
            title="t", sides=[SideSpec("kimi"), SideSpec("qwen")]
        )
        comparison_id, side = comparison.id, comparison.sides[0]
        conversation_id = side.conversation_id
        workspace = Path(session.get(Project, side.project_id).path) / "conversations" / str(conversation_id)
    _write_artifact(workspace / ".anton" / "artifacts", "legacy-report")
    os.utime(workspace / ".anton" / "artifacts" / "legacy-report" / "metadata.json", (1_000_000.0, 1_000_000.0))
    logged = owned_logger(comparisons.logger.name, level=logging.WARNING)

    response = client.post(
        f"/api/v1/comparisons/{comparison_id}/sides/a/continue",
        json={"projectId": str(destination_id)},
        headers=_headers(org, ALICE),
    )

    assert response.status_code == 200, response.text
    assert response.json()["carriedAll"] is True
    assert held_roots_of(org) == {HeldRoot(destination_id, conversation_id)}
    assert _sort_times(org) == {SortTime(destination_id, conversation_id, "legacy-report", 1_000_000.0)}
    assert _slugs(client, org, ALICE, f"?project_id={destination_id}") == ["legacy-report"]
    assert logged.output() == ""


def test_a_continued_sides_failed_sort_time_is_logged_and_the_continue_goes_on(
    client, org, held_roots, owned_logger  # noqa: F811
):
    """The hold commits before the times are written, so it stays when the
    times fail. The failure is rolled back and logged, and the continue
    still moves the conversation."""
    held_roots()
    logged = owned_logger(comparisons.logger.name, level=logging.WARNING)

    with _sort_time_inserts_fail():
        destination_id, conversation_id = _continue_a_side(client, org, artifact=True)

    with Session(_engine()) as raw:
        assert raw.get(Conversation, conversation_id).project_id == destination_id
    assert held_roots_of(org) == {HeldRoot(destination_id, conversation_id)}
    assert _sort_times(org) == set()
    assert_log_fields(
        logged.output(), log_level="WARNING", event="artifact_sort_time_not_recorded",
        error_type="OperationalError", project_id=str(destination_id), conversation_id=str(conversation_id),
    )


def test_a_carried_folder_whose_metadata_json_cannot_be_read_is_logged(
    client, org, held_roots, owned_logger  # noqa: F811
):
    """The read of a folder's time fails with an error that leaves unknown
    what it holds (sort_times_below's unreadable): that folder gets no time,
    the others get theirs, and a WARNING with the event a turn end's read
    logs counts the folders whose time was not updated."""
    held_roots()
    real_read = artifact_sort_times.sort_times_below

    def legacy_report_unreadable(*, base, slugs):
        read = real_read(base=base, slugs=slugs)
        times = {slug: sort_time for slug, sort_time in read.times.items() if slug != "legacy-report"}
        return artifact_sort_times.SortTimesRead(times=times, unreadable=frozenset({"legacy-report"}))

    logged = owned_logger(comparisons.logger.name, level=logging.WARNING)

    with patch.object(artifact_sort_times, "sort_times_below", legacy_report_unreadable):
        destination_id, conversation_id = _continue_a_side(
            client, org, artifact=True, fill=lambda artifacts: _write_artifact(artifacts, "chart")
        )

    assert held_roots_of(org) == {HeldRoot(destination_id, conversation_id)}
    assert {row.slug for row in _sort_times(org)} == {"chart"}
    assert_log_fields(
        logged.output(), log_level="WARNING", event="artifact_sort_times_unreadable", count=1,
        project_id=str(destination_id), conversation_id=str(conversation_id),
    )
