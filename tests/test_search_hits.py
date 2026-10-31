"""The search route's results: typed hits inside, plain objects on the wire."""
from __future__ import annotations

from cowork.api.v1.endpoints import search
from cowork.api.v1.endpoints.search import SearchHit, _artifact_rows, _RowMatches, search_cowork
from cowork.db.scoped import LOCAL_SCOPE


def _hit(*, kind: str, score: int) -> SearchHit:
    return SearchHit(type=kind, id=f"{kind}-1", title=f"Quarterly {kind}", subtitle=kind, route=kind, score=score)


def test_an_artifact_card_becomes_a_typed_hit():
    card = {"title": "Quarterly forecast", "description": "", "path": "/p/dash/index.html", "kind": "HTML"}

    assert _artifact_rows([card], "quarterly") == [SearchHit(
        type="artifact",
        id="/p/dash/index.html",
        title="Quarterly forecast",
        subtitle="/p/dash/index.html",
        route="artifacts",
        score=11,
    )]


async def test_the_route_lists_hits_by_score_and_keeps_the_order_of_a_tie(monkeypatch):
    """The database hits come split where the artifact hits go between them,
    so hits of equal score keep that order. The answer carries each hit as
    the six-key object clients read."""
    task, pin = _hit(kind="task", score=11), _hit(kind="pin", score=11)
    project = _hit(kind="project", score=30)

    async def run_db(fn, *, scope):
        return _RowMatches(before_artifacts=[task, project], after_artifacts=[pin])

    async def run_efs(fn, *, scope, deadline=True):
        return [{"title": "Quarterly forecast", "description": "", "path": "/p/dash/index.html", "kind": "HTML"}]

    monkeypatch.setattr(search, "run_db", run_db)
    monkeypatch.setattr(search, "run_efs", run_efs)

    found = await search_cowork(LOCAL_SCOPE, q="quarterly", limit=25)

    assert [(hit["type"], hit["score"]) for hit in found["results"]] == [
        ("project", 30), ("task", 11), ("artifact", 11), ("pin", 11),
    ]
    assert all(list(hit) == ["type", "id", "title", "subtitle", "route", "score"] for hit in found["results"])
    assert found["results"][1] == task.model_dump()
