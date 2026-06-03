"""
Cowork Server — FastAPI Application.

This module sets up the FastAPI application with middleware, routing,
and all necessary configurations for the Cowork service.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cowork.api.v1.router import api_router as v1_router
from cowork.common.logger import setup_logging
from cowork.common.settings.app_settings import get_app_settings
from cowork.dev_setup import run_dev_setup
from cowork.scheduler import start_scheduler


# Set up logging
logger = setup_logging()

_TOKEN_REFRESH_INTERVAL_SECONDS = 30 * 60  # 30 minutes
_token_refresh_task: asyncio.Task | None = None


async def _token_refresh_loop() -> None:
    """Background loop that refreshes Google OAuth tokens every 30 minutes."""
    from cowork.services.connectors.oauth.google import refresh_google_oauth_tokens

    await asyncio.sleep(60)  # initial delay — let the server finish startup
    while True:
        try:
            await asyncio.to_thread(refresh_google_oauth_tokens)
        except Exception:
            logger.exception("Token refresh loop error")
        await asyncio.sleep(_TOKEN_REFRESH_INTERVAL_SECONDS)


def _start_token_refresh() -> None:
    global _token_refresh_task
    if _token_refresh_task is not None and not _token_refresh_task.done():
        return
    _token_refresh_task = asyncio.create_task(_token_refresh_loop())
    logger.info("Google OAuth token refresh background task created")


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_dev_setup()
    start_scheduler()
    _start_token_refresh()
    yield


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """

    settings = get_app_settings()

    # Create FastAPI app
    app = FastAPI(
        title="Cowork API",
        description="Cowork server — OpenAI-compatible Responses API with pluggable harness backends",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow any origin. This will be controlled by the ingress controller.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )

    # Include v1 API routes
    app.include_router(v1_router)

    logger.info("Cowork application created successfully")
    return app


# Create the application instance
app = create_app()
