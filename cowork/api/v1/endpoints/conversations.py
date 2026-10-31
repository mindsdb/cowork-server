from functools import partial
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSession, ScopedSessionDep, TenantScope, get_tenant_scope
from cowork.db.units import run_db, run_efs_unit
from cowork.models.project import Project
from cowork.schemas.conversations import (
    ConversationCreateRequest,
    ConversationListItem,
    ConversationMoveRequest,
    ConversationUpdateRequest,
)
from cowork.services.conversations import ConversationService, InvalidPaginationParams
from cowork.services.task_objects import TaskObjectService

# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

# A route below that works on the file store runs that work as one unit on the
# file store's threads (run_efs_unit), never on anyio's default threads. A
# list or create that names no project by id, and an update that names its
# project by name, may resolve the general project, which checks that
# project's folder. A request that names its project by id, asks for every
# project, or changes only a title needs only the database and runs as a
# run_db unit.


def _serialize_conversation(c, updated_at=None):
    # `updated_at` is the conversation's last-activity time, derived from its
    # messages (ENG-961) — the stored `modified_at` only moves on rename/move,
    # so it can't be trusted as "recent". Callers pass the derived value;
    # falling back to `created_at` keeps a message-less conversation stable.
    return ConversationListItem.serialize({
        "id": c.id,
        "title": c.topic,
        "preview": c.topic,
        "updated_at": updated_at or c.created_at,
        "created_at": c.created_at,
        "project": c.project.name if c.project else None,
        "project_path": c.project.path if c.project else None,
        "project_id": c.project_id,
        "harness": c.harness,
        "model": c.model,
    })


@router.get("/")
async def list_conversations(
    scope: ScopeDep,
    project_id: UUID | None = None,
    project: str | None = None,
    limit: int = 50,
):
    unit = run_db if project_id is not None or project == "all" else run_efs_unit
    return await unit(
        partial(_list_conversations, project_id=project_id, project=project, limit=limit), scope=scope
    )


def _list_conversations(
    scoped: ScopedSession, *, project_id: UUID | None, project: str | None, limit: int
) -> dict:
    all_projects = project == "all"
    resolved_project_id = project_id
    if not all_projects and resolved_project_id is None and project:
        from cowork.services.projects import ProjectService
        proj = ProjectService(scoped).get_project_by_name_or_none(project)
        if proj is not None:
            resolved_project_id = proj.id
    convs = ConversationService(scoped).list_conversations_with_activity(
        project_id=resolved_project_id, limit=limit, all_projects=all_projects,
    )
    return {"conversations": [_serialize_conversation(c, updated_at=activity) for c, activity in convs]}


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_conversation(body: ConversationCreateRequest, scope: ScopeDep):
    unit = run_db if body.project_id is not None else run_efs_unit
    return await unit(partial(_create_conversation, body=body), scope=scope)


def _create_conversation(scoped: ScopedSession, *, body: ConversationCreateRequest) -> dict:
    svc = ConversationService(scoped)
    project_id = body.project_id
    if project_id is None and body.project:
        project = svc.project_by_name(body.project)
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        project_id = project.id
    try:
        conversation = svc.create_conversation(
            topic=body.topic or body.title or "Untitled task",
            project_id=project_id,
            harness=body.harness,
            model=body.model,
        )
    except ValueError as e:
        # e.g. a project_id that isn't visible in this scope
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _serialize_conversation(conversation)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: UUID, scoped: ScopedSessionDep):
    svc = ConversationService(scoped)
    try:
        conversation = svc.get_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _serialize_conversation(conversation, updated_at=svc.last_message_at(conversation_id))


@router.patch("/{conversation_id}")
async def update_conversation(conversation_id: UUID, body: ConversationUpdateRequest, scope: ScopeDep):
    # A project named by name may be the general one, provisioned on first use.
    unit = run_efs_unit if body.project_id is None and body.project else run_db
    return await unit(partial(_update_conversation, conversation_id=conversation_id, body=body), scope=scope)


def _update_conversation(
    scoped: ScopedSession, *, conversation_id: UUID, body: ConversationUpdateRequest
) -> dict:
    svc = ConversationService(scoped)
    project_id = body.project_id
    if project_id is None and body.project:
        project = svc.project_by_name(body.project)
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        project_id = project.id
    try:
        conversation = svc.update_conversation(
            conversation_id, topic=body.topic or body.title, project_id=project_id
        )
        return _serialize_conversation(conversation, updated_at=svc.last_message_at(conversation_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{conversation_id}/move")
async def move_conversation(conversation_id: UUID, body: ConversationMoveRequest, scope: ScopeDep):
    """Move a task to another project. With `move_objects` (default), the
    artifacts the task created are relocated into the destination project;
    otherwise only the task's project pointer changes. Attachment files
    follow the conversation automatically — purpose tags are keyed by
    conversation id, not project (ENG-338). The destination project must
    already exist (the client creates a new one first, then moves to its
    id)."""
    return await run_efs_unit(partial(_move_conversation, conversation_id=conversation_id, body=body), scope=scope)


def _move_conversation(scoped: ScopedSession, *, conversation_id: UUID, body: ConversationMoveRequest) -> dict:
    svc = ConversationService(scoped)
    try:
        conversation = svc.get_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    dest = None
    if body.project_id is not None:
        dest = scoped.get(Project, body.project_id)
    elif body.project:
        dest = svc.project_by_name(body.project)
    if dest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Destination project not found")

    source = conversation.project
    if body.move_objects and source is not None and dest.id != source.id:
        TaskObjectService(scoped).relocate_to_project(conversation, source, dest)

    conversation = svc.update_conversation(conversation_id, project_id=dest.id)
    return _serialize_conversation(conversation, updated_at=svc.last_message_at(conversation_id))


@router.get("/{conversation_id}/items")
def get_messages(
    conversation_id: UUID,
    scoped: ScopedSessionDep,
    limit: int | None = None,
    before: str | None = None,
):
    """Omitting both `limit` and `before` returns the full, unbounded
    history as a bare list — unchanged from before this endpoint supported
    pagination, so an existing caller that doesn't pass either param (e.g.
    cowork_evals, outside this repo) keeps working exactly as today. Passing
    either opts into the cursor-paginated envelope."""
    svc = ConversationService(scoped)
    try:
        if limit is None and before is None:
            return svc.get_messages(conversation_id)
        page_kwargs = {"before": before}
        if limit is not None:
            page_kwargs["limit"] = limit
        page = svc.get_messages_page(conversation_id, **page_kwargs)
    except InvalidPaginationParams as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return page.model_dump(by_alias=True)


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: UUID, scope: ScopeDep):
    found = await run_efs_unit(
        lambda session: ConversationService(session).delete_conversation(conversation_id), scope=scope
    )
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"ok": True}


@router.delete("/{conversation_id}/turns/{message_id}")
async def delete_conversation_turn(conversation_id: UUID, message_id: UUID, scope: ScopeDep):
    """Delete a turn (user+assistant exchange) and everything after it.

    message_id anchors the turn: the visible assistant message it produced,
    or (for a turn stopped/failed before any answer) the opening user
    message itself. A positional index doesn't survive lazy-loaded/
    paginated history, so this took over from an earlier
    `turn_index: int` path param — an old client still sending an int 422s
    here, and a new client sending a UUID would have 422d against the old
    route, so the break is fail-closed both directions.
    """
    try:
        deleted = await run_efs_unit(
            lambda session: ConversationService(session).delete_turn(conversation_id, message_id), scope=scope
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"ok": True, "deleted": deleted}
