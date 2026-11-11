"""A remote turn end that has the pod's list publishes only the turn's own
artifacts; the rest of the project waits for its scheduled self-heal.

With the list, the turn end opens the folders the turn touched by name and
lists nothing, so its cost no longer grows with the project. The full pass
over the project, which checks every artifact's location and retries a
publish that failed, moves to the scheduler's poll loop, at most once every
SELF_HEAL_INTERVAL_S for each member and project. The turn end asks for it
whether or not autopublish is on. A pod with no list keeps the full pass at
the turn end.
"""
from __future__ import annotations

import asyncio
import builtins
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from pydantic import BaseModel

from cowork import scheduler, server
from cowork.db.scoped import TenantScope
from cowork.services import artifact_autopublish as ap
from cowork.services.task_objects import (
    publish_and_card_turn_artifacts,
    snapshot_artifact_state,
    turn_artifact_changes,
)
from test_autopublish_reconcile import (  # noqa: F401  (fixtures)
    ORG_SCOPE,
    PROJECT_ID,
    _make,
    base,
    enabled,
    key,
    owned_slugs,
    publish_url,
    published,
)


class TurnEndCase(BaseModel, frozen=True, extra="forbid"):
    small_project: int
    large_project: int


class SelfHealCases(BaseModel, frozen=True):
    """The tables of autopublish_self_heal.yaml this file reads."""

    turn_end: TurnEndCase


CASES = SelfHealCases.model_validate(
    yaml.safe_load((Path(__file__).parent / "fixtures" / "autopublish_self_heal.yaml").read_text(encoding="utf-8"))
)

pytestmark = pytest.mark.usefixtures("publish_url")


@pytest.fixture(autouse=True)
def no_self_heals_left(monkeypatch):
    """Self-heal state of this test's own, whatever an earlier test left."""
    monkeypatch.setattr(ap, "_self_heals_requested", set())
    monkeypatch.setattr(ap, "_self_heal_publishes_requested", set())
    monkeypatch.setattr(ap, "_self_heals_started_at", {})
    monkeypatch.setattr(ap, "_self_heal_tasks", {})


@pytest.fixture(autouse=True)
def workspace(monkeypatch):
    monkeypatch.setattr(ap, "_active_workspace_id", lambda scope: None)


def _published_artifact(base: Path, *, slug: str) -> Path:  # noqa: F811
    """An artifact whose publish record is newer than its content."""
    folder = _make(base, slug, files={"index.html": f"<html>{slug}</html>"}, meta={"slug": slug, "type": "html-app"})
    (folder / ".published.json").write_text(json.dumps({
        "index.html": {
            "report_id": f"rid-{slug}", "url": "u", "published": True,
            "last_md5": "m", "published_mtime": 9_999_999_999,
        },
    }))
    return folder


class _FileCalls:
    """The file calls made below `base`, by path or through a descriptor
    opened there, and the listings of `base` itself."""

    def __init__(self, *, monkeypatch, base: Path):  # noqa: F811
        self.calls = 0
        self.base_listings = 0
        self.folders: set[str] = set()
        self.counting = False
        self._base = os.fspath(base)
        # Each open descriptor below the base, and the folder it is in ("" for
        # the base itself).
        self._fds: dict[int, str] = {}
        for name in ("stat", "lstat", "scandir", "listdir"):
            monkeypatch.setattr(os, name, self._wrapped(getattr(os, name), listing=name in ("scandir", "listdir")))
        real_open = os.open

        def opened(path, flags, mode=0o777, *, dir_fd=None):
            place = self._place(path, dir_fd=dir_fd)
            fd = real_open(path, flags, mode, dir_fd=dir_fd)
            self._fds.pop(fd, None)
            if place is not None:
                self._fds[fd] = place
                self._count(place)
            return fd

        monkeypatch.setattr(os, "open", opened)
        real_builtin_open = builtins.open

        def builtin_open(file, *args, **kwargs):
            if not isinstance(file, int):
                place = self._place(file, dir_fd=None)
                if place is not None:
                    self._count(place)
            return real_builtin_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", builtin_open)

    def _place(self, path, *, dir_fd) -> str | None:
        if isinstance(path, int):
            return self._fds.get(path)
        if not isinstance(path, (str, os.PathLike)):
            return None
        text = os.fspath(path)
        if dir_fd is not None:
            parent = self._fds.get(dir_fd)
            if parent is None:
                return None
            return text.split(os.sep)[0] if parent == "" else parent
        if text == self._base:
            return ""
        if text.startswith(self._base + os.sep):
            return text[len(self._base) + 1:].split(os.sep)[0]
        return None

    def _count(self, place: str) -> None:
        if not self.counting:
            return
        self.calls += 1
        if place:
            self.folders.add(place)

    def _wrapped(self, real, *, listing: bool):
        def wrapped(path=".", *args, **kwargs):
            place = self._place(path, dir_fd=kwargs.get("dir_fd"))
            if place is not None:
                self._count(place)
                if listing and place == "" and self.counting:
                    self.base_listings += 1
            return real(path, *args, **kwargs)

        return wrapped


async def _turn_end(base: Path, *, conversation_id, touched_by_pod: set[str] | None, before, before_mtimes):  # noqa: F811
    """A remote turn end as the remote producers run it: with the pod's
    list, the self-heal's request as its record starts, then the file store
    half, the diff, the publish and the cards."""
    tracked = touched_by_pod
    if tracked is not None:
        ap.request_turn_end_self_heal(artifacts_base=base, scope=ORG_SCOPE, project_id=PROJECT_ID)
    changes = turn_artifact_changes(
        conversation_id=conversation_id, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        tracked_new=tracked, tracked_edits=tracked, attribute_by_provenance=True,
        repair_slugs=set() if tracked is not None else None,
    )
    cards = await publish_and_card_turn_artifacts(
        base, new_slugs=changes.created, touched_slugs=changes.touched, scope=ORG_SCOPE,
        project_id=PROJECT_ID, project_name="proj", only_touched=tracked is not None,
    )
    return changes, [card.to_json() for card in cards]


async def _turn_end_editing_one_of(*, base: Path, monkeypatch, published_artifacts: int):  # noqa: F811
    """A turn end whose pod names `old-7`, which it edited, in a project of
    `published_artifacts` published artifacts. Returns the changes, the cards
    and the file calls the turn end made."""
    for index in range(published_artifacts):
        _published_artifact(base, slug=f"old-{index}")
    _published_artifact(base, slug="old-7")
    before, before_mtimes = snapshot_artifact_state(base)
    edited = base / "old-7" / "index.html"
    edited.write_text("<html>edited</html>")
    os.utime(edited, ns=(10**19, 10**19))
    (base / "old-7" / ".published.json").write_text(json.dumps({
        "index.html": {"report_id": "rid-old-7", "url": "u", "published": True, "last_md5": "m", "published_mtime": 0},
    }))

    calls = _FileCalls(monkeypatch=monkeypatch, base=base)
    calls.counting = True
    changes, cards = await _turn_end(
        base, conversation_id=uuid4(), touched_by_pod={"old-7"}, before=before, before_mtimes=before_mtimes,
    )
    calls.counting = False
    return changes, cards, calls


async def test_with_the_pods_list_the_turn_end_opens_only_its_folders_and_lists_nothing(
    tmp_path, enabled, key, published, monkeypatch,  # noqa: F811
):
    """The turn end's cost stays the same whatever the project holds: it
    lists nothing, reads no other artifact's files, and makes as many file
    calls in a large project as in a small one."""
    made: dict[int, int] = {}
    for size in (CASES.turn_end.small_project, CASES.turn_end.large_project):
        artifacts = tmp_path / str(size) / ".anton" / "artifacts"
        with monkeypatch.context() as patched:
            changes, cards, calls = await _turn_end_editing_one_of(
                base=artifacts, monkeypatch=patched, published_artifacts=size,
            )

        assert changes.touched == {"old-7"}
        assert [call["folder"].name for call in published] == ["old-7"]
        assert [card["slug"] for card in cards] == ["old-7"]
        assert calls.base_listings == 0
        assert calls.folders <= {"old-7", ".locks"}
        assert ap._self_heals_requested == {
            ap.SelfHeal(artifacts_base=artifacts, scope=ORG_SCOPE, project_id=PROJECT_ID),
        }
        made[size] = calls.calls
        published.clear()
        ap._self_heals_requested.clear()

    assert made[CASES.turn_end.small_project] == made[CASES.turn_end.large_project]


async def test_without_the_pods_list_the_turn_end_keeps_its_full_self_heal(base, enabled, key, published, monkeypatch):  # noqa: F811
    """An older pod: the turn end lists the project and publishes what an
    earlier turn left, and asks the scheduler for nothing."""
    _make(base, "left-behind", files={"index.html": "<html>old</html>"}, meta={"slug": "left-behind", "type": "html-app"})
    before, before_mtimes = snapshot_artifact_state(base)

    calls = _FileCalls(monkeypatch=monkeypatch, base=base)
    calls.counting = True
    await _turn_end(base, conversation_id=uuid4(), touched_by_pod=None, before=before, before_mtimes=before_mtimes)
    calls.counting = False

    assert [call["folder"].name for call in published] == ["left-behind"]
    assert calls.base_listings >= 1
    assert ap._self_heals_requested == set()


async def test_the_self_heal_publishes_what_the_turn_end_left(base, enabled, key, published):  # noqa: F811
    _make(base, "made", files={"index.html": "<html>new</html>"}, meta={"slug": "made", "type": "html-app"})
    _make(base, "failed-earlier", files={"index.html": "<html>old</html>"}, meta={"slug": "failed-earlier", "type": "html-app"})

    # As a producer runs a turn end with the pod's list: the request as its
    # record starts, then the publish.
    ap.request_turn_end_self_heal(artifacts_base=base, scope=ORG_SCOPE, project_id=PROJECT_ID)
    cards = [card.to_json() for card in await publish_and_card_turn_artifacts(
        base, new_slugs=["made"], touched_slugs={"made"}, scope=ORG_SCOPE,
        project_id=PROJECT_ID, project_name="proj", only_touched=True,
    )]
    assert [card["slug"] for card in cards] == ["made"]
    assert [call["folder"].name for call in published] == ["made"]

    started = ap.start_due_self_heals()
    await asyncio.gather(*started)

    assert len(started) == 1
    assert [call["folder"].name for call in published] == ["made", "failed-earlier"]


@pytest.mark.parametrize(
    ("scope", "asks"),
    [
        (ORG_SCOPE, True),
        # A channel turn's: its organization's, with no member behind it.
        (TenantScope(org_mode=True, org_id=ORG_SCOPE.org_id), True),
        # The desktop's, where artifacts have no location rows.
        (TenantScope(), False),
    ],
    ids=["member", "channel", "desktop"],
)
async def test_with_autopublish_off_a_turn_end_with_the_pods_list_still_asks_for_the_location_check(
    base, published, monkeypatch, scope, asks,  # noqa: F811
):
    """The self-heal's listing is the only one such a turn end gets, so the
    turn end asks for it wherever artifacts have location rows, whether or
    not autopublish is on, and for a channel turn too. Its run checks the
    locations and publishes nothing while autopublish is off."""
    monkeypatch.setattr(ap, "_is_enabled", lambda scope: False)
    _make(base, "made", files={"index.html": "<html>new</html>"}, meta={"slug": "made", "type": "html-app"})
    checked: list[ap.SelfHeal] = []

    async def check(heal):
        checked.append(heal)

    monkeypatch.setattr(ap, "_record_listed_locations", check)

    # As a producer runs a turn end with the pod's list: the request as its
    # record starts, then the publish.
    ap.request_turn_end_self_heal(artifacts_base=base, scope=scope, project_id=PROJECT_ID)
    await publish_and_card_turn_artifacts(
        base, new_slugs=["made"], touched_slugs={"made"}, scope=scope,
        project_id=PROJECT_ID, project_name="proj", only_touched=True,
    )
    await asyncio.gather(*ap.start_due_self_heals())

    asked = [ap.SelfHeal(artifacts_base=base, scope=scope, project_id=PROJECT_ID)] if asks else []
    assert checked == asked
    assert published == []


async def test_with_the_pods_list_autopublish_never_publishes_the_lock_folder(base, enabled, key, published):  # noqa: F811
    """Agent code can write a metadata.json into the project's `.locks`
    folder and the pod can name it. The full pass never lists that folder
    (_candidate_slugs), and a turn end with the pod's list skips it too."""
    _make(base, ".locks", files={"index.html": "<html>locks</html>"}, meta={"slug": ".locks", "type": "html-app"})
    _make(base, "made", files={"index.html": "<html>new</html>"}, meta={"slug": "made", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(
        base, ORG_SCOPE, touched={".locks", "made"}, project_id=PROJECT_ID, only_touched=True,
    )

    assert out == {"made"}
    assert [call["folder"].name for call in published] == ["made"]


async def test_a_projects_self_heal_runs_at_most_once_an_interval(base, enabled, key, published, monkeypatch):  # noqa: F811
    heal = ap.SelfHeal(artifacts_base=base, scope=ORG_SCOPE, project_id=PROJECT_ID)
    ap.request_self_heal(heal)
    await asyncio.gather(*ap.start_due_self_heals())

    ap.request_self_heal(heal)
    assert ap.start_due_self_heals() == []

    monkeypatch.setattr(ap, "SELF_HEAL_INTERVAL_S", 0.0)
    assert len(ap.start_due_self_heals()) == 1


@pytest.mark.parametrize("failing_step", ["listing", "record"])
async def test_a_location_check_that_fails_outright_asks_for_the_next_self_heal(
    base, monkeypatch, caplog, failing_step,  # noqa: F811
):
    """The self-heal's location check raises before it marks anything:
    its listing, or the database unit that records what the listing found.
    Nothing checked the project's locations, so it logs that and asks for
    the project's location check again, with no publish, which the poll
    starts once the interval has passed."""
    from cowork.db import units
    from cowork.services.artifact_identity import ListedIds

    def refused(*args, **kwargs):
        raise RuntimeError("the step was refused")

    if failing_step == "listing":
        monkeypatch.setattr(ap, "_listed_artifact_ids", refused)
    else:
        monkeypatch.setattr(
            ap, "_listed_artifact_ids", lambda *, artifacts_base: ListedIds(ids={}, unreadable=frozenset()),
        )
        monkeypatch.setattr(units, "run_db", lambda fn, *, scope: refused())
    heal = ap.SelfHeal(artifacts_base=base, scope=ORG_SCOPE, project_id=str(uuid4()))

    with caplog.at_level(logging.WARNING, logger=ap.logger.name):
        await ap._record_listed_locations(heal)

    assert [
        record.getMessage() for record in caplog.records
        if record.name == ap.logger.name and record.levelno == logging.WARNING
    ] == [
        "artifact self-heal could not check the locations of the project's artifacts",
    ]
    assert ap._self_heals_requested == {heal}
    assert ap._self_heal_publishes_requested == set()


async def test_a_location_check_asked_for_again_runs_no_publish_pass(base, enabled, monkeypatch):  # noqa: F811
    """A member's self-heal, with autopublish on, whose listing could not
    read a folder asks for the next location check alone. Started once the
    interval has passed, that check lists the project again and runs no
    publish pass, so a folder that stays unreadable costs one listing per
    interval, not the reconciler's listing and plan too. A turn end that
    asks for the self-heal meanwhile gets the publish pass back."""
    from cowork.db import units
    from cowork.services.artifact_identity import ListedIds

    listings: list[Path] = []
    passes: list[str] = []

    def unread_copy(*, artifacts_base):
        listings.append(artifacts_base)
        return ListedIds(ids={}, unreadable=frozenset({"brief-copy"}))

    async def reconciler(artifacts_base, scope, *, touched, project_id, **kwargs):
        passes.append(project_id)
        return set()

    async def recorded(fn, *, scope):
        return None

    monkeypatch.setattr(ap, "_listed_artifact_ids", unread_copy)
    monkeypatch.setattr(ap, "autopublish_project_artifacts", reconciler)
    monkeypatch.setattr(units, "run_db", recorded)
    project_id = str(uuid4())
    ap.request_turn_end_self_heal(artifacts_base=base, scope=ORG_SCOPE, project_id=project_id)
    heal = ap.SelfHeal(artifacts_base=base, scope=ORG_SCOPE, project_id=project_id)

    await asyncio.gather(*ap.start_due_self_heals())
    assert (len(listings), passes) == (1, [project_id])
    assert (ap._self_heals_requested, ap._self_heal_publishes_requested) == ({heal}, set())

    monkeypatch.setattr(ap, "SELF_HEAL_INTERVAL_S", 0.0)
    await asyncio.gather(*ap.start_due_self_heals())
    assert (len(listings), passes) == (2, [project_id])

    ap.request_turn_end_self_heal(artifacts_base=base, scope=ORG_SCOPE, project_id=project_id)
    await asyncio.gather(*ap.start_due_self_heals())
    assert (len(listings), passes) == (3, [project_id, project_id])


async def test_a_self_heal_never_starts_beside_its_running_one(base, monkeypatch):  # noqa: F811
    release = asyncio.Event()
    runs: list[ap.SelfHeal] = []

    async def held(heal):
        runs.append(heal)
        await release.wait()

    monkeypatch.setattr(ap, "_run_self_heal", held)
    monkeypatch.setattr(ap, "SELF_HEAL_INTERVAL_S", 0.0)
    heal = ap.SelfHeal(artifacts_base=base, scope=ORG_SCOPE, project_id=PROJECT_ID)
    ap.request_self_heal(heal)
    first = ap.start_due_self_heals()
    await asyncio.sleep(0)

    ap.request_self_heal(heal)
    assert ap.start_due_self_heals() == []
    release.set()
    await asyncio.gather(*first)
    assert len(ap.start_due_self_heals()) == 1
    assert runs[0] == heal


async def test_each_member_gets_a_self_heal_of_their_own(base, monkeypatch):  # noqa: F811
    """A self-heal publishes only what its member owns, so two members of one
    project each get theirs."""
    monkeypatch.setattr(ap, "_run_self_heal", lambda heal: asyncio.sleep(0))
    other = TenantScope(org_mode=True, org_id=ORG_SCOPE.org_id, user_id="user-2")
    ap.request_self_heal(ap.SelfHeal(artifacts_base=base, scope=ORG_SCOPE, project_id=PROJECT_ID))
    ap.request_self_heal(ap.SelfHeal(artifacts_base=base, scope=other, project_id=PROJECT_ID))

    started = ap.start_due_self_heals()
    await asyncio.gather(*started)

    assert len(started) == 2


async def test_the_scheduler_poll_starts_the_due_self_heals(monkeypatch):
    polls: list[int] = []
    starts: list[int] = []

    async def poll(fn, *, scope):
        polls.append(len(polls))
        return []

    def start_due_self_heals():
        starts.append(len(polls))
        return []

    monkeypatch.setattr(scheduler, "_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(scheduler, "run_db", poll)
    monkeypatch.setattr(ap, "start_due_self_heals", start_due_self_heals)
    polling = asyncio.create_task(scheduler._scheduler_loop())
    try:
        async with asyncio.timeout(2):
            while len(starts) < 2:
                await asyncio.sleep(0)
    finally:
        polling.cancel()
        await asyncio.gather(polling, return_exceptions=True)

    assert starts[:2] == [1, 2]


async def test_a_shutdown_cancels_the_running_self_heals_and_waits_for_them(base, monkeypatch):  # noqa: F811
    unwound: list[bool] = []

    async def held(heal):
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            unwound.append(True)

    class _Registry:
        def close(self):
            pass

        async def shutdown(self, *, timeout_seconds):
            return 0

    monkeypatch.setattr(ap, "_run_self_heal", held)
    monkeypatch.setattr(server, "registry", _Registry())
    monkeypatch.setattr(server, "stop_scheduler", lambda: None)
    ap.request_self_heal(ap.SelfHeal(artifacts_base=base, scope=ORG_SCOPE, project_id=PROJECT_ID))
    [running] = ap.start_due_self_heals()
    await asyncio.sleep(0)

    await server._end_remote_turns()

    assert running.cancelled()
    assert unwound == [True]
