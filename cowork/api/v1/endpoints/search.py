"""Search endpoint — local search across cowork resources."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.services.artifacts import list_artifacts as _list_artifacts
from cowork.services.conversations import ConversationService
from cowork.services.pins import PinService
from cowork.services.projects import ProjectService
from cowork.services.schedules import ScheduleService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def _score(text: str, query: str) -> int:
    haystack = text.lower()
    terms = [term for term in query.lower().split() if term]
    if not terms:
        return 0
    score = 0
    for term in terms:
        if term in haystack:
            score += 10
        score += haystack.count(term)
    return score


@router.get("")
async def search_cowork(
    session: SessionDep,
    q: str = Query(default=""),
    limit: int = Query(default=25),
):
    query = q.strip()
    if not query:
        return {"results": []}

    results: list[dict] = []

    # Conversations (tasks)
    for conv in ConversationService(session).list_conversations(limit=500, all_projects=True):
        project_label = ""
        if conv.project:
            project_label = conv.project.name
        text = " ".join([conv.topic or "", project_label])
        score = _score(text, query)
        if score:
            results.append({
                "type": "task",
                "id": str(conv.id),
                "title": conv.topic or "Untitled task",
                "subtitle": project_label or "Task",
                "route": "task",
                "score": score,
            })

    # Projects
    for project in ProjectService(session).list_projects():
        text = " ".join([project.name or "", project.path or ""])
        score = _score(text, query)
        if score:
            results.append({
                "type": "project",
                "id": project.name,
                "title": project.name or "Project",
                "subtitle": project.path or "Project",
                "route": "project",
                "score": score,
            })

    # Artifacts
    for artifact in _list_artifacts(project_path=None):
        text = " ".join([
            artifact.get("title") or "",
            artifact.get("description") or "",
            artifact.get("path") or "",
            artifact.get("kind") or "",
        ])
        score = _score(text, query)
        if score:
            results.append({
                "type": "artifact",
                "id": artifact.get("path") or artifact.get("id") or artifact.get("title"),
                "title": artifact.get("title") or "Artifact",
                "subtitle": artifact.get("description") or artifact.get("path") or "Artifact",
                "route": "artifacts",
                "score": score,
            })

    # Schedules
    for schedule in ScheduleService(session).list_schedules():
        text = " ".join([schedule.title or "", schedule.prompt or ""])
        score = _score(text, query)
        if score:
            results.append({
                "type": "schedule",
                "id": str(schedule.id),
                "title": schedule.title or "Scheduled task",
                "subtitle": str(schedule.next_run_at) if schedule.next_run_at else "Schedule",
                "route": "scheduled",
                "score": score,
            })

    # Pins
    for idx, pin in enumerate(PinService(session).list_pins()):
        text = " ".join([pin.title or "", str(pin.item_id) or "", pin.item_type or ""])
        score = _score(text, query)
        if score:
            results.append({
                "type": "pin",
                "id": str(pin.item_id),
                "title": pin.title or str(pin.item_id) or "Pinned item",
                "subtitle": f"Pinned {pin.item_type or 'item'}",
                "route": pin.item_type or "task",
                "score": score + max(0, 5 - idx),
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return {"results": results[:limit]}
