"""Desktop personal-skill editing; the store and task snapshots stay canonical."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cowork.api.v1.permissions import LoopbackDesktopOnly, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.models.skill import Skill
from cowork.services.skills import (
    CodeSkillService,
    SkillAlreadyExistsError,
    SkillImmutableError,
    SkillNotFoundError,
)

logger = logging.getLogger(__name__)
# Keep interactive editor/import requests small (UTF-8 bytes). This is separate
# from skills.py's 200 KB JSON-wire per-file cap for Cowork turn attachments.
MAX_SKILL_BYTES = 120_000


class _PersonalSkillRoute(APIRoute):
    """Return string errors the desktop can display, only on this surface."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                error = exc.errors()[0]
                field = {
                    "name": "Name",
                    "description": "Description",
                    "instructions": "Instructions",
                    "content": "Skill file",
                    "enabled": "Enabled",
                }.get(error["loc"][-1], "Skill")
                message = error["msg"].removeprefix("Value error, ")
                raise HTTPException(400, f"{field}: {message}") from exc

        return handle


router = APIRouter(
    route_class=_PersonalSkillRoute,
    dependencies=[Depends(require(LoopbackDesktopOnly))],
)


def _bounded_content(value: object) -> object:
    if isinstance(value, str) and len(value.encode("utf-8")) > MAX_SKILL_BYTES:
        raise ValueError("Keep the skill under 120 KB.")
    return value


class PersonalSkillWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1_024)
    instructions: str = Field(min_length=1, max_length=MAX_SKILL_BYTES)
    enabled: bool = True

    @field_validator("name", "description", "instructions")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a value, not just spaces.")
        return value

    @field_validator("instructions", mode="before")
    @classmethod
    def bounded_instructions(cls, value: object) -> object:
        return _bounded_content(value)


class PersonalSkillImport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=MAX_SKILL_BYTES)

    @field_validator("content", mode="before")
    @classmethod
    def bounded_content(cls, value: object) -> object:
        return _bounded_content(value)


class PersonalSkillResponse(BaseModel):
    id: str
    name: str
    description: str
    instructions: str
    enabled: bool
    projects: list[str]


def _personal_store(scope: Annotated[TenantScope, Depends(get_tenant_scope)]):
    try:
        yield CodeSkillService(scope)
    except SkillImmutableError as exc:
        raise HTTPException(
            403,
            "MindsHub skills cannot be changed in the personal skill editor.",
        ) from exc
    except PermissionError as exc:
        logger.exception("Personal skill files are not writable")
        raise HTTPException(
            500,
            "Could not access the skill files on this computer. Check folder permissions, then try again.",
        ) from exc
    except (SkillAlreadyExistsError, FileExistsError) as exc:
        raise HTTPException(
            409,
            "A skill with that name already exists. Edit it or use a different name.",
        ) from exc
    except SkillNotFoundError as exc:
        raise HTTPException(404, "This personal skill no longer exists.") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        logger.exception("Personal skill storage failed")
        raise HTTPException(
            500,
            "Could not save the skill on this computer. Check free space and folder permissions, then try again.",
        ) from exc


StoreDep = Annotated[CodeSkillService, Depends(_personal_store)]


def _skill_response(skill: Skill) -> PersonalSkillResponse:
    return PersonalSkillResponse(
        id=skill.name,
        name=skill.display_name,
        description=skill.description,
        instructions=skill.instructions,
        enabled=skill.enabled,
        projects=skill.projects,
    )


@router.post("", response_model=PersonalSkillResponse, status_code=201)
def create_personal_skill(body: PersonalSkillWrite, store: StoreDep):
    skill = store.create_skill(
        label=body.name,
        name=body.name.strip(),
        description=body.description,
        instructions=body.instructions,
        enabled=body.enabled,
    )
    return _skill_response(skill)


@router.post("/import", response_model=PersonalSkillResponse, status_code=201)
def import_personal_skill(body: PersonalSkillImport, store: StoreDep):
    return _skill_response(store.import_skill(body.content.encode("utf-8"), "SKILL.md"))


@router.get("/{skill_id}", response_model=PersonalSkillResponse)
def get_personal_skill(skill_id: str, store: StoreDep):
    return _skill_response(store.get_skill(skill_id))


@router.put("/{skill_id}", response_model=PersonalSkillResponse)
def update_personal_skill(skill_id: str, body: PersonalSkillWrite, store: StoreDep):
    # Keep the stable directory id, project restrictions and supporting files.
    skill = store.update_skill(
        skill_id,
        name=body.name.strip(),
        description=body.description,
        instructions=body.instructions,
        enabled=body.enabled,
    )
    return _skill_response(skill)


@router.delete("/{skill_id}", status_code=204)
def delete_personal_skill(skill_id: str, store: StoreDep):
    if not store.delete_skill(skill_id):
        raise HTTPException(404, "This personal skill no longer exists.")
