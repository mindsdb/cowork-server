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


async def test_the_route_scores_real_rows_read_in_its_unit():
    """Through the real run_db: a task's subtitle is its project's name, read
    through conv.project inside the unit, and the hits come by score, a tie
    keeping the task ahead of the project."""
    import shutil
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from cowork.db.units import unit_session
    from cowork.models.conversation import Conversation
    from cowork.models.pin import Pin
    from cowork.models.project import Project
    from cowork.models.schedule import Schedule
    from cowork.services.conversations import ConversationService
    from cowork.services.pins import PinService
    from cowork.services.projects import ProjectService
    from cowork.services.schedules import ScheduleService

    token = f"qz{uuid4().hex[:8]}"
    with unit_session(scope=LOCAL_SCOPE) as session:
        project = ProjectService(session).create_project(f"{token}-desk")
        project_id, project_name, project_path = project.id, project.name, project.path
        conversation_id = ConversationService(session).create_conversation(
            topic=f"{token} launch plan", project_id=project_id,
        ).id
        schedule_id = ScheduleService(session).create_schedule(
            title=f"weekly {token} review", prompt="summarize the week", cadence="daily",
            next_run_at=datetime.now(timezone.utc) + timedelta(days=1), model="anton",
        ).id
        pin_id = PinService(session).pin_item("task", str(conversation_id), title=f"{token} pinned").id
    try:
        found = await search_cowork(LOCAL_SCOPE, q=token, limit=25)
    finally:
        with unit_session(scope=LOCAL_SCOPE) as session:
            for model, row_id in ((Pin, pin_id), (Schedule, schedule_id), (Conversation, conversation_id)):
                session.delete(session.get(model, row_id))
            session.commit()
            session.delete(session.get(Project, project_id))
        shutil.rmtree(project_path, ignore_errors=True)

    hits = found["results"]
    assert [(hit["type"], hit["score"]) for hit in hits] == [
        # The newest pin gets the most recent pin's bonus of 5.
        ("pin", 16), ("task", 12), ("project", 12), ("schedule", 11),
    ]
    task = hits[1]
    assert (task["id"], task["title"], task["subtitle"]) == (
        str(conversation_id), f"{token} launch plan", project_name,
    )
