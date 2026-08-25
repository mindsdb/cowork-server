"""
Cowork Server — FastAPI Application.

This module sets up the FastAPI application with middleware, routing,
and all necessary configurations for the Cowork service.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import MutableHeaders

from cowork.api.v1.router import api_router as v1_router
from starlette.responses import JSONResponse

from cowork.auth_middleware import BearerTokenMiddleware, ensure_auth_token, sync_auth_token
from cowork.db.scoped import MissingTenantScopeError
from cowork.principal import TrustedHeaderMiddleware
from cowork.common.logger import setup_logging
from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import get_app_settings
from cowork.dev_setup import run_dev_setup
from cowork.scheduler import start_scheduler


# Set up logging
logger = setup_logging()


async def _start_channels(app: FastAPI) -> None:
    """Build live adapters from stored credentials and start ingress.

    Org mode: channels are local-mode only — no adapters, no provider
    connections, no ingress (the config endpoints answer 403 and webhook
    routes are not mounted, see _install_channels)."""
    if get_app_settings().tenancy_mode == "org":
        return
    await app.state.channel_adapters.refresh_all()
    from cowork.channels.ingress import sync_channel_ingress
    from cowork.channels.registry import get_registry

    for plugin in get_registry().all():
        await sync_channel_ingress(
            app.state.channel_ingress, app.state.channel_adapters, plugin.channel_type
        )


# Hard ceiling on the boot-time model-map warm. This runs during lifespan
# startup, i.e. BEFORE uvicorn binds the port, so an unbounded fetch against a
# degraded MindsHub would make the desktop app unreachable — and past the
# client's 180s start cap (see cowork src/shared/server-status.ts), a hard
# start failure. Bounded, the worst case is a short boot delay after which we
# bind with the last-known-good map and let GET /recommended-models warm it.
_BOOT_WARM_TIMEOUT_S = 3.0


async def _warm_model_map_on_boot() -> bool:
    """Warm the MindsHub availability map at boot when a key is already stored
    (desktop, returning user), so the FIRST turn heals a stored paid pin instead
    of 402'ing against an empty map (ENG-748). Since ENG-1652 the unset default
    is already free (floored by ``role_defaults``); the map is what steers a
    stored *paid pin* off an unaffordable model on a free-tier wallet.

    Desktop only: org mode stores no key and is floored by ``role_defaults``.

    Bounded (``_BOOT_WARM_TIMEOUT_S``) and fail-open — never raises, never blocks
    the socket bind past the ceiling, and never clobbers a known-good map. On a
    fresh desktop sign-in the credential is written to ``.env`` before the server
    (re)starts, so ``run_dev_setup``'s env→DB migration seeds the key ahead of
    this warm; the returning-user case already has the key stored. Returns True
    iff the stored map was updated.
    """
    if get_app_settings().tenancy_mode == "org":
        return False
    from cowork.db.session import get_open_session
    from cowork.services.providers import warm_enabled_model_map

    warm_session = get_open_session()
    try:
        return await asyncio.wait_for(
            warm_enabled_model_map(warm_session), timeout=_BOOT_WARM_TIMEOUT_S
        )
    except Exception as exc:
        # Message only, never exc_info: the warm frames hold the MindsHub API
        # key, and RICH_LOGGING's tracebacks_show_locals would render it.
        logger.warning("model-map boot warm failed (non-fatal): %s", exc)
        return False
    finally:
        warm_session.close()


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
    # Reclaim scratchpad namespace snapshots whose conversation is gone (ENG-1124).
    # `delete_conversation` prunes its own from now on; this catches everything
    # deleted before that shipped, plus any path that drops a conversation row
    # without going through ConversationService.
    try:
        from cowork.db.session import get_open_session
        from cowork.services.scratchpad_sessions import sweep_orphan_sessions

        sweep_session = get_open_session()
        try:
            sweep_orphan_sessions(sweep_session)
        finally:
            sweep_session.close()
    except Exception:
        logger.exception("scratchpad-session boot sweep failed (non-fatal)")
    # Release scheduled runs left in `running` by a previous process (crash/
    # restart). Otherwise the due-check treats the stale row as an in-flight
    # run and never fires that schedule again.
    try:
        from cowork.db.scoped import ScopedSession, SYSTEM_SCOPE
        from cowork.db.session import get_open_session
        from cowork.services.schedules import ScheduleRunService

        recovery_session = ScopedSession(get_open_session(), SYSTEM_SCOPE)
        try:
            reaped = ScheduleRunService(recovery_session).reap_orphaned_runs()
            if reaped:
                logger.warning(f"Reaped {reaped} orphaned scheduled run(s) on boot")
        finally:
            recovery_session.close()
    except Exception:
        logger.exception("scheduled-run boot recovery failed (non-fatal)")
    # Warm the MindsHub availability map at boot (desktop, returning or
    # freshly-signed-in user) so the empty-map state — which keeps a stored
    # paid pin and 402s the first message on a free-tier wallet (ENG-748) —
    # is closed before the first turn. Bounded and fail-open; see the helper.
    if await _warm_model_map_on_boot():
        logger.info("warmed MindsHub model-availability map on boot")
    start_scheduler()
    await _start_channels(app)
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

    # Org-scoped data touched without an org in scope (e.g. audit mode with no
    # identity) is an auth problem, not a server error — answer 401, not 500.
    @app.exception_handler(MissingTenantScopeError)
    async def _missing_tenant_scope(request, exc):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    # Optional bearer-token auth.  Off by default; enabled when
    # COWORK_REQUIRE_AUTH=true.  Token is auto-generated on first startup
    # when COWORK_AUTH_TOKEN is not set, then persisted to <cowork_home>/.env
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

    # Org mode: build a per-request Principal from gateway-injected identity
    # headers. Added first so it sits inner of the bearer/CORS layers.
    if settings.tenancy_mode == "org":
        enforce = settings.identity_enforce == "enforce"
        app.add_middleware(
            TrustedHeaderMiddleware,
            exempt_paths=channel_webhook_paths,
            enforce=enforce,
        )
        logger.info(
            "auth: org tenancy mode — principal middleware enabled (%s)",
            settings.identity_enforce,
        )
        # No explicit shared root → org data sits on the ephemeral pod FS.
        # Warn, don't fail: dev deployments predate the mount. model_fields_set
        # covers env and dotenv sources alike.
        if "shared_root" not in settings.storage.model_fields_set:
            logger.warning(
                "storage: org mode without COWORK_SHARED_DIR — org-keyed stores "
                "fall back to %s (ephemeral in cloud; data is lost on redeploy)",
                settings.storage.shared_root,
            )
        else:
            logger.info("storage: org-keyed shared root at %s", settings.storage.shared_root)

    if settings.require_auth:
        # The token is mirrored into cowork_home()/.env so a desktop user can
        # read it back. On an org deployment cowork_home() is shared storage
        # that every organization's agent pod can read and write, so mirroring
        # a bearer token there would publish it to every tenant and let any of
        # them overwrite it. Inert today only because require_auth defaults off
        # and no values file sets it; guarded so turning it on is not a trap.
        if settings.tenancy_mode == "org":
            raise RuntimeError(
                "COWORK_REQUIRE_AUTH is not supported in org tenancy mode: the bearer token "
                "would be mirrored into shared storage readable by every organization. "
                "Org deployments authenticate at the ingress instead."
            )
        env_path = cowork_home() / ".env"
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

    # Org mode: channels are local-mode only — mount no auth-exempt webhook
    # routes and load no plugins. The empty registry/manager keep the
    # lifespan start/stop paths inert.
    local_mode = get_app_settings().tenancy_mode != "org"
    if local_mode:
        load_first_party_plugins()
    adapters = LiveAdapterRegistry()
    runtime = AntonChannelRuntime(adapters)
    if local_mode:
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
    app.state.channel_webhook_paths = webhook_paths


# Create the application instance
app = create_app()
