"""
API v1 router for the Minds API.

This module aggregates all v1 endpoints into a single router
that can be included in the main FastAPI application.
"""

from fastapi import APIRouter

from cowork.api.v1.endpoints import (
    projects,
)

# Create the v1 API router
api_router = APIRouter(prefix="/api/v1")

# Include all endpoint routers
# api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])

