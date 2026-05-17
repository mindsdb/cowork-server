"""
API v1 router for the Minds API.

This module aggregates all v1 endpoints into a single router
that can be included in the main FastAPI application.
"""

from fastapi import APIRouter

from cowork.api.v1.endpoints import (
    conversations,
    files,
    projects,
    responses,
    schedules,
)

# Create the v1 API router
api_router = APIRouter(prefix="/api/v1")

# Include all endpoint routers
# api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
api_router.include_router(responses.router, prefix="/responses", tags=["responses"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(schedules.router, prefix="/schedules", tags=["schedules"])

