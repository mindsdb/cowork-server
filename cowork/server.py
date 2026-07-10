"""
Cowork Server — FastAPI Application.

This module sets up the FastAPI application with middleware, routing,
and all necessary configurations for the Cowork service.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import MutableHeaders

from cowork.api.v1.router import api_router as v1_router
from cowork.auth_middleware import BearerTokenMiddleware, ensure_auth_token, sync_auth_token
from cowork.common.logger import setup_logging
from cowork.common.settings.app_settings import get_app_settings
from cowork.dev_setup import run_dev_setup
from cowork.scheduler import start_scheduler


# Set up logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
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


class _NoStoreMiddleware:
    """Stamp ``Cache-Control: no-store`` on responses under the given path
    prefixes so API keys those responses carry are never written to a client's
    on-disk HTTP cache — e.g. Electron's Cache_Data, where plaintext keys were
    found lingering (ENG-462). Pure ASGI (not BaseHTTPMiddleware) so it never
    buffers or breaks the SSE streams.
    """

    def __init__(self, app, prefixes: tuple[str, ...]) -> None:
        self.app = app
        self.prefixes = prefixes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith(self.prefixes):
            await self.app(scope, receive, send)
            return

        async def send_with_no_store(message):
            if message["type"] == "http.response.start":
                MutableHeaders(raw=message["headers"])["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_no_store)


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

    # Optional bearer-token auth.  Off by default; enabled when
    # COWORK_REQUIRE_AUTH=true.  Token is auto-generated on first startup
    # when COWORK_AUTH_TOKEN is not set, then persisted to ~/.cowork/.env
    # so the desktop app and subsequent server runs share the same secret.
    #
    # Registered BEFORE CORS so CORS ends up the outer layer (Starlette applies
    # the last-added middleware outermost): a 401 from the auth layer still
    # flows back through CORS and carries Access-Control-Allow-Origin, so the
    # browser sees the 401 rather than an opaque CORS failure.
    #
    # External channel webhooks carry their own signature, not the bearer
    # token; _install_channels fills this set with their paths so the auth
    # layer lets them through.
    channel_webhook_paths: set[str] = set()
    if settings.require_auth:
        env_path = Path.home() / ".cowork" / ".env"
        token = settings.auth_token or ensure_auth_token(env_path)
        sync_auth_token(env_path, token)
        app.add_middleware(
            BearerTokenMiddleware, token=token, exempt_paths=channel_webhook_paths
        )
        logger.info("auth: bearer-token authentication enabled")

    # Configure CORS middleware (added last → outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )

    # Keep secret-bearing settings responses (reveal-key, raw .env, the
    # providers list) out of clients' on-disk HTTP caches (ENG-462). The chat
    # and submission SSE streams set no-store at their own routes. OAuth is
    # swept in too: GET .../oauth/{engine}/credentials returns a raw
    # client_secret and, being a plain GET with no explicit cache directive,
    # is cacheable by default wherever it's fetched from.
    app.add_middleware(_NoStoreMiddleware, prefixes=("/api/v1/settings", "/api/v1/connectors/oauth"))

    # Include v1 API routes
    app.include_router(v1_router)

    _install_channels(app, channel_webhook_paths)

    logger.info("Cowork application created successfully")
    return app


def _install_channels(app: FastAPI, webhook_paths: set[str]) -> None:
    """Discover channel plugins, mount their webhook routes, and build the
    Anton-only channel runtime + live-adapter registry.

    The registry/runtime are stashed on ``app.state`` so the lifespan can build
    adapters from stored credentials at startup and tear them down on shutdown.
    Webhook routes resolve the live adapter synchronously through the registry's
    cache, which the lifespan populates — so routes are mounted here but only
    serve once a channel is configured (otherwise the route ACK-ignores: 204).

    Every mounted webhook path is recorded in ``webhook_paths`` so the bearer
    auth layer exempts it — these endpoints are called by external platforms
    that authenticate with their own signature, not the Cowork token.
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
        # Mirrors the route path built in webhooks._add_webhook_route:
        # f"/{channel_type}{webhook.path}" under the /api/v1/channels prefix.
        webhook_paths.update(
            f"/api/v1/channels/{plugin.channel_type}{webhook.path}"
            for webhook in plugin.webhooks
        )
    app.state.channel_adapters = adapters
    app.state.channel_runtime = runtime
    app.state.channel_ingress = IngressManager(sink=runtime.handle)


# Create the application instance
app = create_app()
