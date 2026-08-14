"""PostHog connector project discovery."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from cowork.services.connectors.posthog import PostHogDiscoveryError, discover_projects

router = APIRouter()


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
