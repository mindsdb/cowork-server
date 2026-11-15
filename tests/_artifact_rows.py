"""The artifact rows of an org as the list reads them (held roots and sort
times), the sort-time backfill's sentinel, an artifact folder with a set
sort time, and a fixture that empties this replica's self-heal and heal
sweep state, for the tests of those rows' writers and readers."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
from sqlmodel import Session, select

from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.services import artifact_autopublish as ap
from cowork.services import artifact_heal_sweep as sweep
from cowork.services.artifact_sort_times import SortTimeSentinel, read_sort_sentinel
from cowork.services.held_artifact_roots import HeldRoot
from test_artifact_ownership import _engine, write_artifact


@pytest.fixture(autouse=True)
def no_self_heals_left(monkeypatch):
    """Self-heal and heal sweep state of this test's own, whatever an
    earlier test left. It runs for every test of a module that imports it."""
    monkeypatch.setattr(ap, "_self_heals_requested", set())
    monkeypatch.setattr(ap, "_self_heal_publishes_requested", set())
    monkeypatch.setattr(ap, "_self_heals_started_at", {})
    monkeypatch.setattr(ap, "_self_heal_tasks", {})
    monkeypatch.setattr(sweep, "_retries", set())
    monkeypatch.setattr(sweep, "_last_pass_at", None)


def held_roots_of(org_id: str) -> set[HeldRoot]:
    """The roots `org_id`'s held-root rows name."""
    with Session(_engine()) as raw:
        rows = raw.exec(select(HeldArtifactRoot).where(HeldArtifactRoot.org_id == org_id)).all()
        return {HeldRoot(row.project_id, row.conversation_id) for row in rows}


def sort_times_of(org_id: str) -> dict[tuple[UUID, UUID | None, str], float]:
    """`org_id`'s sort-time rows: each folder's project, conversation and
    slug, and the time the list sorts it by."""
    with Session(_engine()) as raw:
        rows = raw.exec(select(ArtifactSortTime).where(ArtifactSortTime.org_id == org_id)).all()
        return {(row.project_id, row.conversation_id, row.slug): row.sort_time for row in rows}


def sort_sentinel() -> SortTimeSentinel | None:
    """The sort-time backfill's sentinel, or None before a pass has walked
    every project."""
    with Session(_engine()) as raw:
        return read_sort_sentinel(ScopedSession(raw, SYSTEM_SCOPE))


def timed_artifact(base: Path, slug: str, at: float) -> Path:
    """An artifact folder `slug` under `base` whose metadata.json was last
    changed at `at`, the time the list sorts it by."""
    folder = write_artifact(base, slug)
    os.utime(folder / "metadata.json", (at, at))
    return folder
