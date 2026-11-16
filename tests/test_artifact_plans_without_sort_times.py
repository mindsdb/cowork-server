"""Only the artifacts list reads the sort-time rows. The delete by an old
folder name and the live-app duplicate plan a project's roots through the
same step as the list of one project (the artifacts endpoint's
_scoped_project_plan), and read no newest rows, which they have no use for."""
from __future__ import annotations

from _async_stub import awaitable
from tests._live_apps import B
from test_live_apps_sharing import _client, _duplicate, _share, people, stack  # noqa: F401  (fixtures)
from test_org_list_scaling import (  # noqa: F401  (client, org and held_roots are fixtures)
    ALICE,
    _artifacts_at,
    _bulk_paths,
    _headers,
    _project_id,
    client,
    held_roots,
    org,
)


def _newest_reads(monkeypatch) -> list[list]:
    """The roots of each read of the newest rows
    (artifact_sort_times.newest_folders)."""
    from cowork.services import artifact_sort_times

    reads: list[list] = []
    real = artifact_sort_times.newest_folders

    def newest_folders(session, *, roots, limit):
        reads.append(list(roots))
        return real(session, roots=roots, limit=limit)

    monkeypatch.setattr(artifact_sort_times, "newest_folders", newest_folders)
    return reads


def test_a_delete_by_an_old_folder_name_reads_no_newest_rows(client, org, held_roots, monkeypatch):  # noqa: F811
    """Once both backfills have run, a delete by an old folder name plans
    the project's held roots and reads no newest rows; the list of the same
    project reads them."""
    from cowork.api.v1.endpoints import artifacts as artifacts_endpoint

    [path] = _bulk_paths(client, org, 0, 1)
    _artifacts_at(org, path, ALICE, {"report": 1_000_100.0})
    held_roots()
    project_id = _project_id(org, path)
    reads = _newest_reads(monkeypatch)
    plans = []

    def legacy_artifact_in_plan(plan, folder_name):
        plans.append(plan)
        return None, None

    monkeypatch.setattr(artifacts_endpoint, "require_product_permission", awaitable(lambda *_args: None))
    monkeypatch.setattr(artifacts_endpoint, "_legacy_artifact_in_plan", legacy_artifact_in_plan)

    deleted = client.delete(f"/api/v1/artifacts/report?project_id={project_id}", headers=_headers(org, ALICE))

    assert deleted.status_code == 404, deleted.text
    [plan] = plans
    assert plan.held is not None and plan.newest is None
    assert reads == []

    listed = client.get(f"/api/v1/artifacts/?project_id={project_id}", headers=_headers(org, ALICE))

    assert [card["slug"] for card in listed.json()] == ["report"]
    assert len(reads) == 1


def test_the_live_app_duplicate_plans_its_project_without_the_newest_rows(people, monkeypatch):  # noqa: F811
    """The duplicate answers the copy's card from a plan of the destination
    project that never asks for the newest rows (the endpoint's
    _with_newest); the list of that project asks."""
    from cowork.api.v1.endpoints import artifacts as artifacts_endpoint

    asked = []
    real = artifacts_endpoint._with_newest

    def with_newest(session, plan, *, sources_of):
        asked.append(plan)
        return real(session, plan, sources_of=sources_of)

    monkeypatch.setattr(artifacts_endpoint, "_with_newest", with_newest)
    http = _client()
    share = _share(http, people["a_project"], email="bob@example.com").json()

    duplicated = _duplicate(http, share["shareId"], people["b_project"])

    assert duplicated.status_code == 200, duplicated.text
    assert asked == []

    listed = http.get("/api/v1/artifacts/", params={"project_id": str(people["b_project"])}, headers=B)

    assert listed.status_code == 200, listed.text
    assert duplicated.json()["slug"] in [card["slug"] for card in listed.json()]
    assert len(asked) == 1
