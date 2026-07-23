"""
API v1 router for the Cowork Server.

This module aggregates all v1 endpoints into a single router
that can be included in the main FastAPI application.
"""

from fastapi import APIRouter

from cowork.api.v1.endpoints import (
    approvals,
    artifacts,
    onboarding,
    rules,
    comments,
    conversations,
    files,
    health,
    memory,
    pins,
    project_files,
    projects,
    publish,
    responses,
    schedules,
    search,
    settings,
    skills,
)
from cowork.api.v1.endpoints.connectors import (
    connections,
    oauth,
    specs,
    submissions,
)
from cowork.api.v1.endpoints import (
    channels,
    conversations,
    files,
    memory,
    pins,
    projects,
    responses,
    schedules,
    settings,
    skills
)

# SHIM:client-compat — compat imports; remove this block and the
# "Compat routes" section below when the client is updated.
from cowork.api.v1.endpoints.compat.stubs import (
    attachments_router,
    browse_router,
    integrations_router,
    scratchpad_router,
)

# Create the v1 API router
api_router = APIRouter(prefix="/api/v1")

# ── Canonical routes ─────────────────────────────────────────────────
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(approvals.router, prefix="/approvals", tags=["approvals"])
api_router.include_router(onboarding.router, prefix="/onboarding", tags=["onboarding"])
api_router.include_router(rules.router, prefix="/rules", tags=["rules"])
api_router.include_router(specs.router, prefix="/connectors/specs", tags=["connectors"])
api_router.include_router(submissions.router, prefix="/connectors/submissions", tags=["connectors"])
api_router.include_router(connections.router, prefix="/connectors/connections", tags=["connectors"])
api_router.include_router(oauth.router, prefix="/connectors/oauth", tags=["connectors"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(project_files.router, prefix="/projects", tags=["project-files"])
api_router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
api_router.include_router(responses.router, prefix="/responses", tags=["responses"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(schedules.router, prefix="/schedules", tags=["schedules"])
api_router.include_router(pins.router, prefix="/pins", tags=["pins"])
api_router.include_router(skills.router, prefix="/skills", tags=["skills"])
api_router.include_router(memory.router, prefix="/memory", tags=["memory"])
api_router.include_router(channels.router, prefix="/channels", tags=["channels"])
api_router.include_router(artifacts.router, prefix="/artifacts", tags=["artifacts"])
api_router.include_router(comments.router, prefix="/artifact-comments", tags=["artifact-comments"])
api_router.include_router(publish.router, prefix="/publish", tags=["publish"])
api_router.include_router(settings.router, prefix="/settings", tags=["settings"])
api_router.include_router(search.router, prefix="/search", tags=["search"])

# ── Compat routes (SHIM:client-compat — delete this section) ────────
api_router.include_router(integrations_router, prefix="/integrations", tags=["compat"])
api_router.include_router(attachments_router, prefix="/attachments", tags=["compat"])
api_router.include_router(scratchpad_router, prefix="/scratchpad", tags=["compat"])
api_router.include_router(browse_router, prefix="/browse", tags=["compat"])
