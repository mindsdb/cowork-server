from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.skills import SkillCreateRequest, SkillResponse, SkillUpdateRequest
from cowork.services.skills import SkillService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/")
def list_skills(session: SessionDep):
    skills = SkillService().list_skills()
    return {"skills": [SkillResponse.serialize(s) for s in skills]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_skill(body: SkillCreateRequest, session: SessionDep):
    try:
        skill = SkillService().create_skill(
            label=body.label,
            name=body.name,
            instructions=body.instructions or "",
            description=body.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return SkillResponse.serialize(skill)


@router.get("/{skill_id}")
def get_skill(skill_id: str, session: SessionDep):
    try:
        return SkillResponse.serialize(SkillService().get_skill(skill_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/{skill_id}")
def update_skill(skill_id: str, body: SkillUpdateRequest, session: SessionDep):
    try:
        skill = SkillService().update_skill(
            skill_id,
            label=body.label,
            name=body.name,
            description=body.description,
            instructions=body.instructions,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return SkillResponse.serialize(skill)


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill(skill_id: str, session: SessionDep):
    svc = SkillService()
    if svc.delete_skill(skill_id):
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found.")
