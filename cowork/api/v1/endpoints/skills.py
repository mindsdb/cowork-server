from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.skills import SkillCreateRequest, SkillResponse, SkillUpdateRequest
from cowork.services.skills import SkillService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/")
def list_skills(session: SessionDep):
    skills = SkillService(session).list_skills()
    return {"skills": [SkillResponse.serialize(s) for s in skills]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_skill(body: SkillCreateRequest, session: SessionDep):
    try:
        skill = SkillService(session).create_skill(
            label=body.label,
            name=body.name,
            instructions=body.instructions or "",
            description=body.description,
            when_to_use=body.when_to_use,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return SkillResponse.serialize(skill)


@router.get("/{skill_id}")
def get_skill(skill_id: UUID, session: SessionDep):
    try:
        return SkillResponse.serialize(SkillService(session).get_skill(skill_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/{skill_id}")
def update_skill(skill_id: UUID, body: SkillUpdateRequest, session: SessionDep):
    try:
        skill = SkillService(session).update_skill(
            skill_id,
            label=body.label,
            name=body.name,
            description=body.description,
            when_to_use=body.when_to_use,
            instructions=body.instructions,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return SkillResponse.serialize(skill)


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill(skill_id: str, session: SessionDep):
    svc = SkillService(session)
    try:
        uid = UUID(skill_id)
    except ValueError:
        uid = None
    if uid and svc.delete_skill(uid):
        return
    # Fall back to lookup by name/label
    if svc.delete_skill_by_name(skill_id):
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found.")
