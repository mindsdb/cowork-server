"""newest_folders' read of a run of tied times on PostgreSQL: the rest of a
run the limit cuts comes in the code point order of slug, then id, as the
first query cuts it, whatever the database's collation. These run when
COWORK_TEST_POSTGRES_URI points at a disposable database, as CI's service
tests do (postgres_service)."""
from __future__ import annotations

import pytest
from sqlmodel import Session, select

from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.services import artifact_sort_times
from cowork.services.artifact_sort_times import newest_folders, record_sort_times
from cowork.services.held_artifact_roots import HeldRoot
from test_artifact_sort_times_postgres import (  # noqa: F401  (engine is a fixture)
    POSTGRES_URI,
    _project,
    _sentinel_written,
    _session,
    engine,
)

pytestmark = [
    pytest.mark.postgres_service,
    pytest.mark.skipif(not POSTGRES_URI, reason="set COWORK_TEST_POSTGRES_URI to a disposable database"),
]

# Names a locale collation such as en_US.utf8 orders differently from code
# points ("a-report" before "B-report", "salesforce" before "sales-q1").
_NAMES = ("B-report", "a-report", "report", "sales-q1", "salesforce")


@pytest.mark.parametrize("cap", [10, 9], ids=["read-whole", "past-the-cap"])
def test_a_run_the_limit_cuts_comes_whole_in_code_point_order(engine, monkeypatch, cap):  # noqa: F811
    """Two projects' shared roots hold the same five names at one time,
    beside a newer row and an older one. The limit cuts that run of ten:
    the rows kept and the rest of the run are every row at least as new as
    it, in code point order of slug, then id, each once. With room for nine
    rows of the run, none of the rest comes back, and ties_read is false."""
    project_id, other_id = _project(engine), _project(engine)
    monkeypatch.setattr(artifact_sort_times, "_TIED_ROWS", cap)
    _sentinel_written(engine)
    session = _session(engine)
    try:
        record_sort_times(session, project_id=project_id, times={"newest": 3.0} | dict.fromkeys(_NAMES, 2.0))
        record_sort_times(session, project_id=other_id, times=dict.fromkeys(_NAMES, 2.0) | {"oldest": 1.0})
        newest = newest_folders(session, roots=[HeldRoot(project_id, None), HeldRoot(other_id, None)], limit=4)
    finally:
        session.close()
    with Session(engine) as raw:
        ids = {(row.project_id, row.slug): row.id for row in raw.exec(select(ArtifactSortTime)).all()}
    run = sorted((name, ids[(project, name)], project) for project in (project_id, other_id) for name in _NAMES)
    in_order = [("newest", project_id)] + [(name, project) for name, _id, project in run]

    assert [(row.slug, row.project_id) for row in newest.rows] == in_order[:4]
    assert not newest.complete
    if cap == 10:
        assert [(row.slug, row.project_id) for row in newest.rows + newest.tied] == in_order
        assert newest.ties_read
    else:
        assert newest.tied == () and not newest.ties_read
