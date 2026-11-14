"""The held-root rows a turn end, the self-heal, a task's move and a
comparison side's carry write, through the code that calls them rather than
below it.

A turn end that created, listed or timed folders in its project's shared
root, or could not read a folder's time there, holds the root
(task_objects.record_new_artifacts). A failed hold raises
ListedLocationsUnchecked once the turn's other rows are written, so the turn
end asks for the project's self-heal, and the self-heal's listing holds the
root it finds folders in (artifact_autopublish._record_listed_locations),
also when it could read none of them. A task's move holds the destination's
shared root, and a failed hold asks for the destination's location check
(TaskObjectService._relocate_artifacts). A comparison side continued into a
project holds the legacy root its carried workspace brought, when that root
holds a folder (ComparisonService._hold_carried_workspace), on the first
continue and on the retry that finishes a carry cut short; a failed hold is
logged and still lets the carry write its folders' sort times.
"""
from __future__ import annotations

import asyncio
import errno
import logging
import os
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlmodel import Session

import cowork.turnqueue.remote_turn as remote_turn_mod
from cowork.db import units
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.handlers.responses import ResponsesHandler
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.task_object import TaskObject
from cowork.services import artifact_autopublish as ap
from cowork.services import artifact_ownership as ownership
from cowork.services import comparisons, held_artifact_roots, task_objects
from cowork.services.artifact_identity import ListedIds
from cowork.services.artifact_locations import Location, LocationRow, find_location, record_checked_folders
from cowork.services.comparisons import ComparisonService, SideSpec
from cowork.services.held_artifact_roots import HeldRoot
from cowork.services.task_objects import ListedLocationsUnchecked, record_new_artifacts
from cowork.turnqueue.remote_turn import remote_turn_events
from test_artifact_ownership import _engine, make_conversation, make_project, project_root, scoped, write_artifact
from test_artifact_relocation_ownership import _index, _relocate
from test_org_list_scaling import (  # noqa: F401  (fixtures)
    ALICE,
    _bulk_paths,
    _project_id,
    _write_artifact,
    client,
    held_roots,
    org,
)
from test_project_files_artifact_roots import _held_root_inserts_fail, _slugs, _sort_times
from test_remote_turn import _drain, _fake_handler
from tests._artifact_rows import held_roots_of, no_self_heals_left  # noqa: F401  (no_self_heals_left is a fixture)

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


def _refuse_holds(monkeypatch) -> None:
    def refused(session, **kwargs):
        raise RuntimeError("the row could not be written")

    monkeypatch.setattr(held_artifact_roots, "hold_root", refused)


def _turn_end_on_the_test_database(monkeypatch, *, during_turn) -> None:
    """A remote turn whose inputs are read and whose end is recorded in real
    units on the test database. ``during_turn`` runs once the turn's
    snapshot is taken, before the pod's ``turn_completed``, which carries no
    list of the artifacts its tools touched, so the turn end lists the
    project's whole shared root."""
    _fake_handler(monkeypatch, remote_artifacts_context=ResponsesHandler._remote_artifacts_context)
    monkeypatch.setattr(remote_turn_mod, "run_db", units.run_db)
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", AsyncMock(return_value=[]))

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        during_turn()
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)


async def _run_turn(*, scope: TenantScope, conversation_id) -> None:
    _, failure = await _drain(remote_turn_events(
        scope=scope, conv_id=conversation_id, org_id=scope.org_id, user_id=scope.user_id,
        input_text="hi", model="anton", turn_rows=[],
    ))
    assert failure is None


def test_a_failed_hold_raises_once_the_turns_rows_are_written(tmp_path, monkeypatch):
    """Raised last: the index, owner and location writes all run first. The
    id of the folder the turn made, and the id of a copy its listing found,
    each have a row naming another folder of the project, so both ids are
    still marked conflicted."""
    org_id, owner = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, owner)
    made_id, listed_id = uuid4().hex, uuid4().hex
    with scoped(org_id) as session:
        for artifact_id, slug in ((made_id, "made-original"), (listed_id, "listed-original")):
            record_checked_folders(session, artifact_id=artifact_id, folders=[Location(project.id, None, slug)])
    _refuse_holds(monkeypatch)

    with scoped(org_id) as session:
        with pytest.raises(ListedLocationsUnchecked):
            record_new_artifacts(
                session, conversation_id=conversation_id, project_id=project.id, slugs=["made"], creator=owner,
                artifact_ids={"made": made_id},
                listed_ids=ListedIds(
                    ids={"listed-copy": listed_id, "listed-original": listed_id}, unreadable=frozenset(),
                ),
            )

    with scoped(org_id, owner) as session:
        indexed = session.exec(
            session.select(TaskObject.ref).where(TaskObject.conversation_id == conversation_id)
        ).all()
        owners = ownership.resolve_artifact_owners(session, project_root(project), ["made"])
        locations = {artifact_id: find_location(session, artifact_id) for artifact_id in (made_id, listed_id)}
    assert indexed == ["made"]
    assert owners["made"].owner_user_id == owner
    assert locations == {
        made_id: LocationRow(Location(project.id, None, "made-original"), conflicted=True),
        listed_id: LocationRow(Location(project.id, None, "listed-original"), conflicted=True),
    }
    assert held_roots_of(org_id) == set()


async def test_a_turn_end_whose_hold_fails_asks_for_the_projects_self_heal(tmp_path, monkeypatch, caplog):
    """The pod sent no list, so nothing asked for the self-heal before the
    diff. Without the raise only the project's next turn end would hold the
    root, and until then the list would not show the turn's artifact."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    source = project_root(project)
    _turn_end_on_the_test_database(
        monkeypatch, during_turn=lambda: write_artifact(source.base, "report", conversation=conversation_id),
    )
    _refuse_holds(monkeypatch)
    scope = TenantScope(org_mode=True, org_id=org_id, user_id=creator)

    with caplog.at_level(logging.ERROR, logger=remote_turn_mod.logger.name):
        await _run_turn(scope=scope, conversation_id=conversation_id)

    heal = ap.SelfHeal(artifacts_base=source.base, scope=scope, project_id=str(project.id))
    assert ap._self_heals_requested == {heal}
    [record] = [record for record in caplog.records if record.name == remote_turn_mod.logger.name]
    assert isinstance(record.exc_info[1], ListedLocationsUnchecked)
    # The turn created a folder, but only the hold failed, which its line names.
    assert record.getMessage() == (
        "Could not check the locations of the artifacts a channel turn found, or record the folder that holds them"
        " or their sort times"
    )
    # The turn's own rows are written all the same.
    with units.unit_session(scope=scope) as session:
        assert session.exec(
            session.select(TaskObject.ref).where(TaskObject.conversation_id == conversation_id)
        ).all() == ["report"]
    assert held_roots_of(org_id) == set()


async def test_the_self_heals_location_check_holds_the_root_its_listing_finds_a_folder_in(tmp_path):
    """The poll's location check, as start_due_self_heals runs it: its
    listing of the shared root, then the database unit that records what
    the listing found."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    write_artifact(source.base, "report")
    heal = ap.SelfHeal(
        artifacts_base=source.base, scope=TenantScope(org_mode=True, org_id=org_id), project_id=str(project.id),
    )
    ap.request_location_check(heal)

    await asyncio.gather(*ap.start_due_self_heals())

    assert held_roots_of(org_id) == {HeldRoot(project.id, None)}
    assert ap._self_heals_requested == set()


async def test_a_turn_end_whose_listing_reads_no_folder_holds_its_root_and_asks_for_the_location_check(
    tmp_path, monkeypatch
):
    """The file server stops answering for the project's one folder during
    the turn. The turn end's listing finds the folder and reads nothing in
    it, so its record holds the root (ArtifactChanges.needs_record): an
    unreadable folder may hold an artifact the list should show once it
    reads again. It asks for the project's location check, whose listing
    still cannot read the folder, and the root stays held. That check holds
    such a root on its own too (test_artifact_sort_time_writers)."""
    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    source = project_root(project)
    write_artifact(source.base, "report", conversation=conversation_id)
    unreadable = source.base / "report"
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path) == unreadable:
            raise OSError(errno.EIO, "the file server did not answer")
        return real_stat(path, *args, **kwargs)

    _turn_end_on_the_test_database(monkeypatch, during_turn=lambda: monkeypatch.setattr(os, "stat", stat))
    scope = TenantScope(org_mode=True, org_id=org_id, user_id=creator)

    await _run_turn(scope=scope, conversation_id=conversation_id)
    heal = ap.SelfHeal(artifacts_base=source.base, scope=scope, project_id=str(project.id))
    assert ap._self_heals_requested == {heal}
    assert ap._self_heal_publishes_requested == set()
    await asyncio.gather(*ap.start_due_self_heals())

    assert held_roots_of(org_id) == {HeldRoot(project.id, None)}


async def test_a_task_move_whose_hold_fails_asks_for_the_destinations_location_check(tmp_path, monkeypatch):
    """The move writes no message, so the heal sweep, which lists only
    projects with a recent message, never reaches an idle destination.
    Without the request, the moved folder would stay out of the list until a
    turn there ends."""
    org_id, owner = str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation_id = make_conversation(source, owner)
    write_artifact(project_root(source).base, "report", conversation_id)
    _index(conversation_id, source.id, "report")
    with scoped(org_id) as session:
        ownership.record_artifact_owner(session, source.id, "report", owner, action="create")
    hold_root = held_artifact_roots.hold_root
    _refuse_holds(monkeypatch)

    assert _relocate(org_id, owner, conversation_id, source.id, dest.id) == 1

    scope = TenantScope(org_mode=True, org_id=org_id, user_id=owner)
    heal = ap.SelfHeal(artifacts_base=project_root(dest).base, scope=scope, project_id=str(dest.id))
    assert ap._self_heals_requested == {heal}
    # The location check alone: the move asks for no publish.
    assert ap._self_heal_publishes_requested == set()
    assert held_roots_of(org_id) == set()
    monkeypatch.setattr(held_artifact_roots, "hold_root", hold_root)

    await asyncio.gather(*ap.start_due_self_heals())

    assert held_roots_of(org_id) == {HeldRoot(dest.id, None)}


def _continue_a_side(
    api, org_id: str, *, artifact: bool, retried: bool = False, fill: Callable[[Path], None] | None = None
):
    """Continue one of Alice's comparison sides into a project of `org_id`.
    The side's workspace holds its agent state and, when `artifact`, the
    legacy artifact folder `legacy-report`, as a PUT into the sandbox's
    workspace or an agent's own write would leave it; `fill`, when given,
    writes more into the workspace's artifacts folder. When `retried`, the
    first continue's workspace carry fails, so the sandbox is kept and no
    row is written, and a second continue carries the workspace
    (ComparisonService._finish_carry). Returns the destination's id and the
    side's conversation id."""
    [path] = _bulk_paths(api, org_id, 0, 1)
    path.mkdir(parents=True, exist_ok=True)
    destination_id = _project_id(org_id, path)
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id, user_id=ALICE))
        service = ComparisonService(session)
        comparison = service.create_comparison(title="t", sides=[SideSpec("kimi"), SideSpec("qwen")])
        side = comparison.sides[0]
        conversation_id = side.conversation_id
        workspace = Path(session.get(Project, side.project_id).path) / "conversations" / str(conversation_id)
        (workspace / ".anton").mkdir(parents=True)
        (workspace / ".anton" / "anton.md").write_text("Always use EUR.")
        if artifact:
            _write_artifact(workspace / ".anton" / "artifacts", "legacy-report")
        if fill is not None:
            fill(workspace / ".anton" / "artifacts")
        if retried:
            with patch.object(comparisons, "_carry_hosted_workspace", return_value=False):
                cut_short = service.continue_side(comparison.id, "a", destination_id)
            assert cut_short.carried_all is False
            assert held_roots_of(org_id) == set()
            assert _sort_times(org_id) == set()

        continued = service.continue_side(comparison.id, "a", destination_id)

    assert continued.carried_all is True
    assert (path / "conversations" / str(conversation_id) / ".anton" / "anton.md").is_file()
    return destination_id, conversation_id


def test_a_continued_side_holds_the_legacy_root_its_workspace_brought(client, org, held_roots):  # noqa: F811
    """The list trusts the rows once the backfill has finished, so without
    the carry's row it never opens the destination's legacy root."""
    held_roots()

    destination_id, conversation_id = _continue_a_side(client, org, artifact=True)

    assert held_roots_of(org) == {HeldRoot(destination_id, conversation_id)}
    assert _slugs(client, org, ALICE, f"?project_id={destination_id}") == ["legacy-report"]


def test_a_side_continued_again_holds_the_legacy_root_its_finished_carry_brought(
    client, org, held_roots  # noqa: F811
):
    """The first continue's carry is cut short, so the workspace stays in
    the kept sandbox and that continue holds nothing. The second continue
    finishes the carry through ComparisonService._finish_carry, which holds
    the root the workspace brought."""
    held_roots()

    destination_id, conversation_id = _continue_a_side(client, org, artifact=True, retried=True)

    assert held_roots_of(org) == {HeldRoot(destination_id, conversation_id)}
    assert _slugs(client, org, ALICE, f"?project_id={destination_id}") == ["legacy-report"]


def test_a_carry_cut_short_holds_the_legacy_root_it_already_brought(client, org, held_roots):  # noqa: F811
    """A retry's merge can stay partial (a file it cannot read), yet the
    legacy artifacts folder it copied is in the destination already: the
    carry holds its root whatever the carry reports."""
    held_roots()
    real_carry = comparisons._carry_hosted_workspace

    def carried_but_partial(*args, **kwargs):
        real_carry(*args, **kwargs)
        return False

    [path] = _bulk_paths(client, org, 0, 1)
    path.mkdir(parents=True, exist_ok=True)
    destination_id = _project_id(org, path)
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org, user_id=ALICE))
        service = ComparisonService(session)
        comparison = service.create_comparison(title="t", sides=[SideSpec("kimi"), SideSpec("qwen")])
        side = comparison.sides[0]
        conversation_id = side.conversation_id
        workspace = Path(session.get(Project, side.project_id).path) / "conversations" / str(conversation_id)
        _write_artifact(workspace / ".anton" / "artifacts", "legacy-report")
        with patch.object(comparisons, "_carry_hosted_workspace", carried_but_partial):
            continued = service.continue_side(comparison.id, "a", destination_id)

    assert continued.carried_all is False
    assert held_roots_of(org) == {HeldRoot(destination_id, conversation_id)}
    assert _slugs(client, org, ALICE, f"?project_id={destination_id}") == ["legacy-report"]


def test_a_continued_side_with_no_artifacts_folder_holds_no_root(client, org, held_roots):  # noqa: F811
    """Every hosted side's workspace holds agent state, so holding its root
    whatever it held would cost the list an open per continued side."""
    held_roots()

    destination_id, _ = _continue_a_side(client, org, artifact=False)

    assert held_roots_of(org) == set()
    assert _sort_times(org) == set()
    assert _slugs(client, org, ALICE, f"?project_id={destination_id}") == []


def test_a_continued_sides_failed_hold_is_logged_and_the_continue_goes_on(
    client, org, held_roots, caplog  # noqa: F811
):
    """After the carry the continue moves the conversation and marks the
    side in the same session, so the carry rolls its failed hold back
    first, or those writes fail too. The times have their own try, so each
    carried folder's time is still written, as the project files PUT's is
    after a failed hold."""
    held_roots()

    with _held_root_inserts_fail(), caplog.at_level(logging.WARNING, logger=comparisons.logger.name):
        destination_id, conversation_id = _continue_a_side(client, org, artifact=True)

    with Session(_engine()) as raw:
        assert raw.get(Conversation, conversation_id).project_id == destination_id
    assert held_roots_of(org) == set()
    assert {(row.project_id, row.conversation_id, row.slug) for row in _sort_times(org)} == {
        (destination_id, conversation_id, "legacy-report")
    }
    [record] = [
        record for record in caplog.records
        if record.name == comparisons.logger.name
        and getattr(record, "event", None) == "held_artifact_root_not_recorded"
    ]
    assert record.levelno == logging.WARNING
    assert record.error_type == "OperationalError"
    assert (record.project_id, record.conversation_id) == (str(destination_id), str(conversation_id))


def test_a_continued_side_whose_workspace_the_server_cannot_read_still_continues(
    client, org, held_roots, caplog  # noqa: F811
):
    """The read of the carried workspace's folders goes to the file store,
    after the workspace is already in the destination. A file server error
    there records neither row and is logged like a failed hold, and the
    continue still moves the conversation."""
    held_roots()

    def unreadable(_root):
        raise OSError(errno.EIO, "the file server did not answer")

    with (
        patch.object(held_artifact_roots, "listable_folders", unreadable),
        caplog.at_level(logging.WARNING, logger=comparisons.logger.name),
    ):
        destination_id, conversation_id = _continue_a_side(client, org, artifact=True)

    with Session(_engine()) as raw:
        assert raw.get(Conversation, conversation_id).project_id == destination_id
    assert held_roots_of(org) == set()
    assert _sort_times(org) == set()
    [record] = [
        record for record in caplog.records
        if record.name == comparisons.logger.name
        and getattr(record, "event", None) == "held_artifact_root_not_recorded"
    ]
    assert record.exc_info[0] is OSError
