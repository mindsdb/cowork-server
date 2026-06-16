from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.harnesses.memory.registry import MemorySlot
from cowork.schemas.memory import MemoryResponse, MemoryScope, MemoryUpdateRequest
from cowork.services.memory import MemoryService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


# ---------------------------------------------------------------------------
# SHIM:client-compat — Helpers to bridge the server's (scope, category,
# project_id) model to the client's (scope, relativePath, projectPath,
# sections/files) model.  Remove when the client is updated to use the
# canonical schema.
# ---------------------------------------------------------------------------

# _CATEGORY_TO_PATH = {"lesson": "lessons.md", "rule": "rules.md"}
# _PATH_TO_CATEGORY = {v: k for k, v in _CATEGORY_TO_PATH.items()}
#
#
# def _relative_path(category: str) -> str:
#     return _CATEGORY_TO_PATH.get(category, f"topics/{category}.md")
#
#
# def _category_from_path(relative_path: str) -> str:
#     if relative_path in _PATH_TO_CATEGORY:
#         return _PATH_TO_CATEGORY[relative_path]
#     # topics/<slug>.md → slug
#     if relative_path.startswith("topics/") and relative_path.endswith(".md"):
#         return relative_path[len("topics/"):-len(".md")]
#     return relative_path
#
#
# def _scope_label(scope: MemoryScope) -> str:
#     return "Global" if scope == MemoryScope.global_ else "Project"
#
#
# def _build_sections(items: list[MemoryResponse], session: Session) -> list[dict]:
#     """Group flat MemoryResponse items into the {scope, projectName, projectPath, files} sections the client expects."""
#     # Collect project info for project-scoped items
#     project_cache: dict[UUID, Project] = {}
#     for item in items:
#         if item.project_id and item.project_id not in project_cache:
#             proj = session.get(Project, item.project_id)
#             if proj:
#                 project_cache[item.project_id] = proj
#
#     # Group into sections keyed by (scope_label, project_id|None)
#     section_map: dict[tuple, dict] = {}
#     for item in items:
#         scope_label = _scope_label(item.scope)
#         proj = project_cache.get(item.project_id) if item.project_id else None
#         key = (scope_label, item.project_id)
#
#         if key not in section_map:
#             section_map[key] = {
#                 "scope": scope_label,
#                 "projectName": proj.name if proj else None,
#                 "projectPath": proj.path if proj else None,
#                 "files": [],
#             }
#
#         rel_path = _relative_path(item.category)
#         section_map[key]["files"].append({
#             "relativePath": rel_path,
#             "content": item.content,
#             "preview": (item.content or "")[:200],
#             "scope": scope_label,
#             "projectName": proj.name if proj else None,
#             "projectPath": proj.path if proj else None,
#         })
#
#     return list(section_map.values())


# ---------------------------------------------------------------------------
# Legacy Endpoints
# ---------------------------------------------------------------------------

# @router.get("/")
# async def list_memory(
#     session: SessionDep,
#     project_path: str | None = Query(default=None),
# ):
#     items = await MemoryService(session).list_memory()
#
#     # If the client passes project_path, filter to global + that project only.
#     if project_path:
#         project = session.exec(
#             select(Project).where(Project.path == project_path)
#         ).first()
#         project_id = project.id if project else None
#         items = [
#             i for i in items
#             if i.scope == MemoryScope.global_ or i.project_id == project_id
#         ]
#
#     sections = _build_sections(items, session)
#     return {"sections": sections}
#
#
# @router.post("/")
# async def save_memory(body: dict, session: SessionDep):
#     """Accept the client's {scope, relativePath, content, projectPath} shape."""
#     scope_raw = body.get("scope", "Global")
#     relative_path = body.get("relativePath", "")
#     content = body.get("content", "")
#     project_path = body.get("projectPath")
#
#     scope = MemoryScope.global_ if scope_raw == "Global" else MemoryScope.project
#     category = _category_from_path(relative_path)
#
#     project_id = None
#     if scope == MemoryScope.project and project_path:
#         project = session.exec(
#             select(Project).where(Project.path == project_path)
#         ).first()
#         if not project:
#             raise HTTPException(status_code=404, detail=f"Project not found for path: {project_path}")
#         project_id = project.id
#
#     await MemoryService(session).update_memory(
#         scope=scope,
#         category=category,
#         content=content,
#         project_id=project_id,
#     )
#     return {"ok": True}
#
#
# @router.delete("/")
# async def delete_memory(
#     session: SessionDep,
#     scope: str = Query(...),
#     relative_path: str = Query(...),
#     project_path: str | None = Query(default=None),
# ):
#     scope_enum = MemoryScope.global_ if scope == "Global" else MemoryScope.project
#     category = _category_from_path(relative_path)
#
#     project_id = None
#     if scope_enum == MemoryScope.project and project_path:
#         project = session.exec(
#             select(Project).where(Project.path == project_path)
#         ).first()
#         if not project:
#             raise HTTPException(status_code=404, detail=f"Project not found for path: {project_path}")
#         project_id = project.id
#
#     try:
#         await MemoryService(session).delete_memory(
#             scope=scope_enum,
#             category=category,
#             project_id=project_id,
#         )
#     except ValueError as e:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
#     return {"ok": True}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=list[MemoryResponse])
async def list_memory(
    session: SessionDep,
    project_id: UUID | None = Query(default=None),
):
    return await MemoryService(session).list_memory(project_id=project_id)


@router.put("/", response_model=MemoryResponse)
async def update_memory(body: MemoryUpdateRequest, session: SessionDep):
    try:
        return await MemoryService(session).update_memory(
            scope=body.scope,
            category=body.category,
            content=body.content,
            project_id=body.project_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/")
async def delete_memory(
    session: SessionDep,
    scope: MemoryScope,
    category: MemorySlot,
    project_id: UUID | None = Query(default=None),
):
    try:
        await MemoryService(session).delete_memory(
            scope=scope,
            category=category,
            project_id=project_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"ok": True}

