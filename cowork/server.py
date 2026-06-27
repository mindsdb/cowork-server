"""
Cowork Server — FastAPI Application.

This module sets up the FastAPI application with middleware, routing,
and all necessary configurations for the Cowork service.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cowork.api.v1.router import api_router as v1_router
from cowork.auth_middleware import BearerTokenMiddleware, ensure_auth_token
from cowork.common.logger import setup_logging
from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings, get_app_settings
from cowork.dev_setup import run_dev_setup
from cowork.scheduler import start_scheduler
from cowork.services.connectors.oauth.google import google_service


# Set up logging
logger = setup_logging()


async def _token_refresh_loop() -> None:
    while True:
        await asyncio.sleep(30 * 60)
        logger.info("Running Google token refresh check")
        try:
            google_service.refresh_all_tokens(ConnectorSettings(), OAuthSettings())
        except Exception:
            logger.exception("Token refresh loop error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        token_refresh_task = asyncio.create_task(_token_refresh_loop())
    except Exception:
        logger.exception("Failed to start token refresh loop")
        token_refresh_task = None
    run_dev_setup()
    # Seal any turn buffers left open by a previous process (crash/restart)
    # so reconnecting clients get a clean Interrupted end-of-stream rather
    # than hanging. GC of old buffers happens lazily; cheap no-op when none.
    try:
        from cowork.streaming import get_streams_dir
        from cowork.streaming.recovery import gc_old_buffers, seal_orphan_buffers
        seal_orphan_buffers(get_streams_dir())
        gc_old_buffers(get_streams_dir(), max_age_days=7)
    except Exception:
        logger.exception("turn-buffer boot recovery failed (non-fatal)")
    start_scheduler()
    await app.state.channel_adapters.refresh_all()
    from cowork.channels.ingress import sync_channel_ingress
    from cowork.channels.registry import get_registry

    for plugin in get_registry().all():
        await sync_channel_ingress(
            app.state.channel_ingress, app.state.channel_adapters, plugin.channel_type
        )
    try:
        yield
    finally:
        from cowork.channels.webhooks import drain_background_tasks
        from cowork.common.http_client import close_proxy_client
        from cowork.services.artifacts import shutdown_launched_backends
        from cowork.services.scratchpad_runtime import close_all as close_scratchpads

        await app.state.channel_ingress.stop_all()
        await drain_background_tasks()
        await app.state.channel_adapters.shutdown()
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
        description="Cowork server — OpenAI-compatible Responses API with pluggable harness backends",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )

    # Optional bearer-token auth.  Off by default; enabled when
    # COWORK_REQUIRE_AUTH=true.  Token is auto-generated on first startup
    # when COWORK_AUTH_TOKEN is not set, then persisted to ~/.cowork/.env
    # so the desktop app and subsequent server runs share the same secret.
    if settings.require_auth:
        env_path = Path.home() / ".cowork" / ".env"
        token = settings.auth_token or ensure_auth_token(env_path)
        app.add_middleware(BearerTokenMiddleware, token=token)
        logger.info("auth: bearer-token authentication enabled")

    # Include v1 API routes
    app.include_router(v1_router)

    _install_channels(app)

    logger.info("Cowork application created successfully")
    return app


def _install_channels(app: FastAPI) -> None:
    """Discover channel plugins, mount their webhook routes, and build the
    Anton-only channel runtime + live-adapter registry.

    The registry/runtime are stashed on ``app.state`` so the lifespan can build
    adapters from stored credentials at startup and tear them down on shutdown.
    Webhook routes resolve the live adapter synchronously through the registry's
    cache, which the lifespan populates — so routes are mounted here but only
    serve once a channel is configured (otherwise the route ACK-ignores: 204).
    """
    from cowork.channels.ingress import IngressManager
    from cowork.channels.registry import get_registry, load_first_party_plugins
    from cowork.channels.runtime import AntonChannelRuntime, LiveAdapterRegistry
    from cowork.channels.webhooks import build_channel_webhook_router

    load_first_party_plugins()
    adapters = LiveAdapterRegistry()
    runtime = AntonChannelRuntime(adapters)
    for plugin in get_registry().all():
        if not plugin.webhooks:
            continue
        app.include_router(
            build_channel_webhook_router(plugin, resolver=adapters.get, sink=runtime.handle),
            prefix="/api/v1/channels",
        )
    app.state.channel_adapters = adapters
    app.state.channel_runtime = runtime
    app.state.channel_ingress = IngressManager(sink=runtime.handle)


# Create the application instance
app = create_app()
