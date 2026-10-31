"""Search endpoint — local search across cowork resources."""
from __future__ import annotations

from typing import Annotated, Literal, NamedTuple

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSession, TenantScope, get_tenant_scope
from cowork.db.units import run_db, run_efs
from cowork.services.artifact_roots import artifacts_sources_for_scan as _sources_for_scan
from cowork.services.artifacts import list_artifacts as _list_artifacts
from cowork.services.conversations import ConversationService
from cowork.services.pins import PinService
from cowork.services.projects import ProjectService
from cowork.services.schedules import ScheduleService

# AuthenticatedInOrgMode, declared explicitly: the conversation/project/
# schedule/pin listings already fail closed on their own
# (MissingTenantScopeError -> 401, cowork/db/scoped.py) whenever org mode has
# no org in scope. Declaring it too makes the requirement visible to a route
# walker instead of something only discoverable by reading scoped.py. (The
# artifact listing is a separate, desktop-only filesystem scan of the caller's
# own projects root; org mode skips it explicitly below.)
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


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


class SearchHit(BaseModel):
    """One search result, as the route lists it."""

    type: Literal["task", "project", "schedule", "pin", "artifact"]
    id: str | None
    title: str
    subtitle: str
    # The client view that opens the result.
    route: str
    score: int


class _RowMatches(NamedTuple):
    """The database matches, split where the artifact matches go between
    them, so a tie in score keeps the order it always had."""

    before_artifacts: list[SearchHit]
    after_artifacts: list[SearchHit]


def _search_rows(session: ScopedSession, query: str) -> _RowMatches:
    """The scored conversation, project, schedule and pin matches, as one
    unit. Their rows never leave it: ``conv.project`` loads lazily."""
    results: list[SearchHit] = []

    # Conversations (tasks)
    for conv in ConversationService(session).list_conversations(limit=500, all_projects=True):
        project_label = ""
        if conv.project:
            project_label = conv.project.name
        text = " ".join([conv.topic or "", project_label])
        score = _score(text, query)
        if score:
            results.append(SearchHit(
                type="task",
                id=str(conv.id),
                title=conv.topic or "Untitled task",
                subtitle=project_label or "Task",
                route="task",
                score=score,
            ))

    # Projects
    for project in ProjectService(session).list_projects():
        text = " ".join([project.name or "", project.path or ""])
        score = _score(text, query)
        if score:
            results.append(SearchHit(
                type="project",
                id=project.name,
                title=project.name or "Project",
                subtitle=project.path or "Project",
                route="project",
                score=score,
            ))
    before_artifacts, results = results, []

    # Schedules
    for schedule in ScheduleService(session).list_schedules():
        text = " ".join([schedule.title or "", schedule.prompt or ""])
        score = _score(text, query)
        if score:
            results.append(SearchHit(
                type="schedule",
                id=str(schedule.id),
                title=schedule.title or "Scheduled task",
                subtitle=str(schedule.next_run_at) if schedule.next_run_at else "Schedule",
                route="scheduled",
                score=score,
            ))

    # Pins
    for idx, pin in enumerate(PinService(session).list_pins()):
        text = " ".join([pin.title or "", str(pin.item_id) or "", pin.item_type or ""])
        score = _score(text, query)
        if score:
            results.append(SearchHit(
                type="pin",
                id=str(pin.item_id),
                title=pin.title or str(pin.item_id) or "Pinned item",
                subtitle=f"Pinned {pin.item_type or 'item'}",
                route=pin.item_type or "task",
                score=score + max(0, 5 - idx),
            ))
    return _RowMatches(before_artifacts=before_artifacts, after_artifacts=results)


def _artifact_rows(artifacts: list[dict], query: str) -> list[SearchHit]:
    results: list[SearchHit] = []
    for artifact in artifacts:
        text = " ".join([
            artifact.get("title") or "",
            artifact.get("description") or "",
            artifact.get("path") or "",
            artifact.get("kind") or "",
        ])
        score = _score(text, query)
        if score:
            results.append(SearchHit(
                type="artifact",
                id=artifact.get("path") or artifact.get("id") or artifact.get("title"),
                title=artifact.get("title") or "Artifact",
                subtitle=artifact.get("description") or artifact.get("path") or "Artifact",
                route="artifacts",
                score=score,
            ))
    return results


@router.get("")
async def search_cowork(
    scope: ScopeDep,
    q: str = Query(default=""),
    limit: int = Query(default=25),
):
    query = q.strip()
    if not query:
        return {"results": []}

    matches = await run_db(lambda session: _search_rows(session, query), scope=scope)
    results = list(matches.before_artifacts)

    # Artifacts: the scan of the caller's own projects root (a person's
    # partition where projects are personal). Hosted search has never listed
    # artifacts, and the scan of an org's projects root would start to, so org
    # mode skips it.
    if not scope.org_mode:
        artifacts = await run_efs(
            lambda: _list_artifacts(_sources_for_scan(scope=scope)), scope=scope
        )
        results.extend(_artifact_rows(artifacts, query))
    results.extend(matches.after_artifacts)

    results.sort(key=lambda hit: hit.score, reverse=True)
    return {"results": [hit.model_dump() for hit in results[:limit]]}
