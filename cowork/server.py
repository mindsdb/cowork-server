"""
Minds FastAPI Application Server.

This module sets up the FastAPI application with middleware, routing,
and all necessary configurations for the Minds service.
"""

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_dev_setup()
    start_scheduler()
    try:
        yield
    finally:
        from cowork.common.http_client import close_proxy_client
        from cowork.services.artifacts import shutdown_launched_backends
        from cowork.services.scratchpad_runtime import close_all as close_scratchpads

        shutdown_launched_backends()
        await close_scratchpads()
        await close_proxy_client()


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
        description="FastAPI-based service providing OpenAI-compatible chat completions with MindsDB integration",
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
