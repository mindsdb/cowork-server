from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.schemas.skills import SkillCreateRequest, SkillResponse, SkillUpdateRequest
from cowork.services.skills import SkillService

router = APIRouter()

# Every route resolves the tenant scope — the skill store is org-keyed.
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("/")
def list_skills(scope: ScopeDep):
    # Seeded here for an org that opens this menu before it has ever chatted.
    # The turn payload seeds too, so whichever comes first wins; see
    # `build_turn_skills`.
    skill_service = SkillService(scope)
    skill_service.ensure_builtin_skills()
    skills = skill_service.list_skills()
    return {"skills": [SkillResponse.serialize(s) for s in skills]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_skill(body: SkillCreateRequest, scope: ScopeDep):
    try:
        skill = SkillService(scope).create_skill(
            label=body.label,
            name=body.name,
            instructions=body.instructions or "",
            description=body.description,
            enabled=body.enabled,
            projects=body.projects,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return SkillResponse.serialize(skill)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_skill(file: UploadFile, scope: ScopeDep):
    raw = await file.read()
    try:
        skill = SkillService(scope).import_skill(raw, filename=file.filename)
    except FileExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    return SkillResponse.serialize(skill)


@router.get("/{skill_id}")
def get_skill(skill_id: str, scope: ScopeDep):
    try:
        return SkillResponse.serialize(SkillService(scope).get_skill(skill_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/{skill_id}")
def update_skill(skill_id: str, body: SkillUpdateRequest, scope: ScopeDep):
    try:
        skill = SkillService(scope).update_skill(
            skill_id,
            label=body.label,
            name=body.name,
            description=body.description,
            instructions=body.instructions,
            enabled=body.enabled,
            projects=body.projects,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return SkillResponse.serialize(skill)


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill(skill_id: str, scope: ScopeDep):
    if SkillService(scope).delete_skill(skill_id):
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found.")
