"""PostHog connector project discovery."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.services.connectors.posthog import PostHogDiscoveryError, discover_projects

router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])


class DiscoverPostHogProjectsRequest(BaseModel):
    personal_api_key: str
    host: str
    custom_host: str | None = None


@router.post("/projects")
async def discover_posthog_projects(req: DiscoverPostHogProjectsRequest) -> dict:
    try:
        projects = await discover_projects(
            personal_api_key=req.personal_api_key,
            host=req.host,
            custom_host=req.custom_host,
        )
    except PostHogDiscoveryError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"projects": [{"id": project.id, "name": project.name} for project in projects]}
