"""
Cowork Server — FastAPI Application.

This module sets up the FastAPI application with middleware, routing,
and all necessary configurations for the Cowork service.
"""

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from starlette.datastructures import MutableHeaders

from cowork.api.v1.route_walker import contradictory_routes, route_key, undeclared_routes
from cowork.api.v1.router import api_router as v1_router
from starlette.responses import JSONResponse

from cowork.auth_middleware import BearerTokenMiddleware, ensure_auth_token, sync_auth_token
from cowork.coding.inference_proxy import INFERENCE_PATHS
from cowork.db.personal_resources import PersonalResource, check_personal_resource_dependencies
from cowork.db.scoped import MissingTenantScopeError
from cowork.db.units import DatabaseBusy, busy_retry_seconds
from cowork.handlers.turn_errors import SERVER_BUSY_CODE, server_busy_message
from cowork.managed import (
    ManagedFolder,
    SharedRegistryError,
    cortex_takes_managed_layer,
    managed_folder,
)
from cowork.shared_skills import read_shared_skills, shared_skills_file
from cowork.principal import TrustedHeaderMiddleware
from cowork.schemas.refusals import TURN_IN_PROGRESS, Refusal
from cowork.services.connectors.layered_vault import SharedConnectionError
from cowork.streaming import TurnInProgress
from cowork.common.logger import find_database_error, relayed_error, setup_logging
from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.dev_setup import run_dev_setup
from cowork.scheduler import start_scheduler
from cowork.services.pending_credential_clear import consume_pending_credential_clear


# Set up logging
logger = setup_logging()


async def _start_channels(app: FastAPI) -> None:
    """Build live adapters from stored credentials and start ingress.

    Local mode only: adapters/ingress are boot-time and singleton, built from
    the one deployment-global installation. Org mode resolves adapters
    per-request instead (LiveAdapterRegistry.resolve_org_bridge), since
    credentials are per-org and there is no single set to preload at boot."""
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


async def _cancel_and_wait(task: asyncio.Task | None) -> None:
    """Stop one background task at shutdown and wait for it to finish."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _run_artifact_owner_backfill() -> None:
    """Background, one-time owner backfill (ENG-2961). Never affects startup."""
    try:
        from cowork.services.artifact_owner_backfill import run_artifact_owner_backfill

        await asyncio.to_thread(run_artifact_owner_backfill)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("artifact owner backfill failed (non-fatal)")


# Imported lazily by the first agent turn otherwise, adding their import time to
# that turn's time to first text.
AGENT_RUNTIME_MODULES = (
    "anton.core.llm.openai",
    "anton.core.llm.anthropic",
    "anton.minds_client",
    "anton.core.session",
    "anton.core.tools.tool_handlers",
)


def _warm_agent_runtime() -> None:
    """Import the agent runtime at startup. A module that fails is left for the turn."""
    import importlib

    for name in AGENT_RUNTIME_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            logger.debug("agent runtime warm-up skipped %s", name, exc_info=True)


def _validate_turn_queue_settings() -> None:
    """Build ``TurnQueueSettings`` once so an invalid env value stops the boot.

    Handlers build these settings per request, so without this a bad value
    boots healthy and then fails every remote turn. Raises ``RuntimeError``
    naming the invalid fields only.
    """
    try:
        TurnQueueSettings()
    except ValidationError as exc:
        fields = ", ".join(".".join(str(part) for part in err["loc"]) for err in exc.errors())
        # The pydantic error echoes input values, and some of these fields hold secrets.
        raise RuntimeError(f"invalid turn-queue settings: {fields}") from None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_turn_queue_settings()
    run_dev_setup()
    # After migrations and the .env seed, before anything reads a credential.
    app_settings = get_app_settings()
    consume_pending_credential_clear(app_settings.database.uri, app_settings.tenancy_mode)
    # Before the server reports ready, so the first turn does not pay for imports.
    await asyncio.to_thread(_warm_agent_runtime)
    # History recovery also retries buffers sealed by an earlier failed sweep.
    try:
        from cowork.db.scoped import ScopedSession, SYSTEM_SCOPE
        from cowork.db.session import get_open_session
        from cowork.streaming import get_streams_dir
        from cowork.streaming.recovery import seal_orphan_turns_in_history

        history_session = ScopedSession(get_open_session(), SYSTEM_SCOPE)
        try:
            seal_orphan_turns_in_history(history_session, get_streams_dir())
        finally:
            history_session.close()
    except Exception:
        logger.exception("turn-history boot recovery failed (non-fatal)")
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
    app.state.channel_ingress_reconciler = None
    app.state.artifact_owner_backfill = None
    if get_app_settings().tenancy_mode == "org":
        from cowork.channels.ingress import start_reconciler

        app.state.channel_ingress_reconciler = start_reconciler(
            app.state.channel_ingress, app.state.channel_adapters
        )
        # Background so the port binds without waiting for an EFS walk.
        app.state.artifact_owner_backfill = asyncio.create_task(
            _run_artifact_owner_backfill()
        )
    try:
        yield
    finally:
        from cowork.channels.webhooks import drain_background_tasks
        from cowork.common.chat_session import drain_scratchpad_closes
        from cowork.common.http_client import close_proxy_client
        from cowork.coding.inference_proxy import close_inference_client
        from cowork.services.artifacts import shutdown_launched_backends
        from cowork.services.scratchpad_runtime import close_all as close_scratchpads
        from cowork.coding.service import get_coding_service
        from cowork.streaming.registry import registry

        # Uvicorn's request drain is bounded by the image command. Persist
        # interrupted turns before their database and runtime resources close.
        cancelled = await registry.shutdown()
        if cancelled:
            logger.warning("Cancelled %d in-flight turn(s) for shutdown", cancelled)

        await _cancel_and_wait(getattr(app.state, "channel_ingress_reconciler", None))
        await _cancel_and_wait(getattr(app.state, "artifact_owner_backfill", None))

        await app.state.channel_ingress.stop_all()
        await drain_background_tasks()
        # After both turn drains above: each turn they unwound has queued the
        # close of its scratchpad processes, and those must finish before the
        # event loop stops.
        await drain_scratchpad_closes()
        await app.state.channel_adapters.shutdown()
        get_coding_service().close_all()
        shutdown_launched_backends()
        await close_scratchpads()
        await close_proxy_client()
        await close_inference_client()


@asynccontextmanager
async def _lifespan_without_database_text(app: FastAPI):
    """Run ``lifespan``, keeping a database error's text out of Uvicorn's log.

    Starlette hands a failed startup or shutdown to Uvicorn as
    ``traceback.format_exc()`` text, and Uvicorn logs that text as a plain
    message, which no exception filter can read. A migration, the seed rows,
    or a reboot that starts Cowork before Postgres would print SQL and driver
    detail on every restart. The owned handlers log the error instead, as its
    type, SQLSTATE and the cowork frame it failed in, and Uvicorn gets an
    error that carries no database text.
    """
    phase = "startup"
    try:
        async with lifespan(app):
            phase = "shutdown"
            yield
    except Exception as exc:
        if find_database_error(exc=exc) is None:
            raise
        logger.error("Server %s failed", phase, exc_info=True, extra=relayed_error())
        raise RuntimeError(f"Database operation failed during server {phase}") from None


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


def _refuse_unreadable_managed_stores(managed: ManagedFolder) -> None:
    """Refuse to start when the server's user can't read the managed skills or
    data-vault folder, or a managed skill folder or its SKILL.md. Each would
    otherwise crash a page, a turn, or the restart the runbook names as the
    remedy, rather than surface here with the path and the fix."""
    from anton.core.tools.skill_format import SKILL_FILE

    for folder in (managed.skills_dir, managed.vault_dir):
        if not os.access(folder, os.R_OK | os.X_OK):
            raise RuntimeError(
                f"COWORK_MANAGED_DIR's {folder.name} folder {folder} can't be read: the "
                "server's user needs read and search permission on it. Run "
                f"`chmod -R a+rX {folder}` on the host copy (and `chown -R 1000:1000 "
                f"{folder}` if it is owned by another user)."
            )
    for slug_dir in sorted(managed.skills_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        if not os.access(slug_dir, os.R_OK | os.X_OK):
            raise RuntimeError(
                f"COWORK_MANAGED_DIR's skill folder {slug_dir} can't be read: run "
                f"`chmod -R a+rX {managed.skills_dir}` on the host copy."
            )
        skill_file = slug_dir / SKILL_FILE
        if skill_file.exists() and not os.access(skill_file, os.R_OK):
            raise RuntimeError(
                f"COWORK_MANAGED_DIR's skill file {skill_file} can't be read: run "
                f"`chmod -R a+rX {managed.skills_dir}` on the host copy."
            )
    # A managed connection file the server's user can't read vanishes for
    # everyone and frees its name, so a tester could save their own under it.
    # Refuse at start rather than silently share less.
    for vault_file in sorted(managed.vault_dir.iterdir()):
        if vault_file.is_file() and not os.access(vault_file, os.R_OK):
            raise RuntimeError(
                f"COWORK_MANAGED_DIR's connection file {vault_file} can't be read: run "
                f"`chown 1000:1000 {vault_file}` on the host copy so it is not freed and "
                "re-saved by a tester."
            )


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
        lifespan=_lifespan_without_database_text,
    )

    # Org-scoped data touched without an org in scope (e.g. audit mode with no
    # identity) is an auth problem, not a server error — answer 401, not 500.
    @app.exception_handler(MissingTenantScopeError)
    async def _missing_tenant_scope(request, exc):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    # No database connection freed within POOL_TIMEOUT. DatabaseBusy, a unit
    # that waited for its slot or its connection, subclasses the pool timeout,
    # so this answers both. The web UI waits for Retry-After before it offers
    # a retry, and the sentence names the wait for clients that don't read it.
    @app.exception_handler(PoolTimeoutError)
    async def _database_busy(request, exc):
        retry_after = busy_retry_seconds()
        logger.warning(
            "%s %s refused with 503: no database connection freed in time",
            request.method, request.url.path,
        )
        return JSONResponse(
            Refusal(detail=server_busy_message(retry_after), code=SERVER_BUSY_CODE).model_dump(),
            status_code=DatabaseBusy.status_code,
            headers={"Retry-After": str(retry_after)},
        )

    # A second question into a conversation whose turn is still answering.
    @app.exception_handler(TurnInProgress)
    async def _turn_in_progress(request, exc):
        return JSONResponse(TURN_IN_PROGRESS.model_dump(), status_code=exc.status_code)

    # A save or delete that reached a connection the caller can't change, or
    # a create under a name the registry still shares (LayeredDataVault). The
    # connections routes refuse a change up front; this answers the same for
    # a save whose derived name lands on one, and the create.
    @app.exception_handler(SharedConnectionError)
    async def _shared_connection(request, exc: SharedConnectionError):
        return JSONResponse({"detail": exc.detail}, status_code=403)

    # Bearer-token auth. On by default in local mode (see AppSettings.
    # require_auth). Token is auto-generated on first startup when
    # COWORK_AUTH_TOKEN is not set, then persisted to <cowork_home>/.env so
    # the desktop app and subsequent server runs share the same secret.
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

    # Trusted identity headers name a person in local tenancy. Only a request
    # that already passed the bearer token may name one, and each person's
    # partition needs every resource its rows point at, and memory of its own.
    if settings.trust_identity_headers:
        if settings.tenancy_mode == "org":
            raise RuntimeError(
                "COWORK_TRUST_IDENTITY_HEADERS is not supported in org tenancy mode: "
                "org deployments already build each request's principal from the "
                "gateway's identity headers."
            )
        if not settings.require_auth:
            raise RuntimeError(
                "COWORK_TRUST_IDENTITY_HEADERS requires COWORK_REQUIRE_AUTH: without the "
                "bearer token check, any caller could name any person in the identity headers."
            )
        check_personal_resource_dependencies(settings.personal_resources)
        if PersonalResource.MEMORY not in settings.personal_resources:
            raise RuntimeError(
                "COWORK_TRUST_IDENTITY_HEADERS needs memory in COWORK_PERSONAL_RESOURCES. "
                "Memory left off is one store that every person's turns learn into and "
                "read back, so each person would read the others' lessons and profile "
                "lines. Add memory to the list."
            )

    # The operator's managed folder reaches every turn read-only. A missing
    # folder stops the start: its skill and connection names would otherwise
    # stop being reserved, and its rules would silently reach no turn.
    managed = managed_folder()
    if managed is not None:
        if settings.tenancy_mode == "org":
            raise RuntimeError(
                "COWORK_MANAGED_DIR is not supported in org tenancy mode: turns run on "
                "remote workers, which receive none of the managed folder."
            )
        missing = managed.missing_dirs()
        if missing:
            raise RuntimeError(
                "COWORK_MANAGED_DIR names missing folders: "
                + ", ".join(str(path) for path in missing)
                + ". Create or mount each one; the managed folder needs memory/, "
                "skills/ and data-vault/."
            )
        # anton reads memory/ on every turn, and a folder it can't search
        # fails each turn on its first rule.
        if not os.access(managed.memory_dir, os.R_OK | os.X_OK):
            raise RuntimeError(
                f"COWORK_MANAGED_DIR's memory folder {managed.memory_dir} can't be read: "
                "the server's user needs read and search permission on it (for example "
                "chmod 755), or every turn fails reading its rules."
            )
        # A skills/ or data-vault/ folder the server's user can't read, or a
        # managed skill folder or its SKILL.md it can't read, would crash every
        # person's Skills or Connections page and every turn. Refuse at start,
        # with the path and the fix, rather than loop on a crash.
        _refuse_unreadable_managed_stores(managed)
        if not cortex_takes_managed_layer():
            raise RuntimeError(
                "COWORK_MANAGED_DIR needs an anton whose Cortex takes managed_hc, the "
                "read-only memory layer. The installed anton has none, so the "
                "managed rules would reach no turn."
            )
        # A malformed connection registry stops the start, where the operator
        # sees it. One broken later shares nothing until fixed
        # (SharedRegistry.read_live).
        try:
            shared_connections = managed.shared_connections()
        except SharedRegistryError as error:
            raise RuntimeError(
                f"COWORK_MANAGED_DIR's shared registry is malformed: {error}. "
                "Fix or remove the file; a missing one shares nothing."
            ) from error
        from anton.core.datasources.data_vault import LocalDataVault

        logger.info(
            "managed folder: %s (%d skills, %d connections, %d shared connections)",
            managed.root,
            len(managed.skill_slugs()),
            len(LocalDataVault(managed.vault_dir).list_connections()),
            len(shared_connections.root),
        )

    # The skills people share, which Cowork writes itself. A malformed list
    # never stops the start: it shares nothing, with a warning naming the
    # file (read_shared_skills), until the next share or hide replaces it.
    if settings.trust_identity_headers and PersonalResource.SKILLS in settings.personal_resources:
        logger.info(
            "shared skills: %d in %s", len(read_shared_skills().root), shared_skills_file()
        )

    # A remote turn runs on a worker that reads and learns memory in the
    # project's own folder, not in a person's tier, and receives none of the
    # managed folder.
    if TurnQueueSettings().is_remote and (settings.trust_identity_headers or managed is not None):
        raise RuntimeError(
            "COWORK_TURN_BACKEND=remote is not supported with COWORK_TRUST_IDENTITY_HEADERS "
            "or COWORK_MANAGED_DIR: remote turns keep memory in each project's own folder, "
            "not in each person's, and receive none of the managed folder. Leave "
            "COWORK_TURN_BACKEND unset to run turns in this process."
        )

    # Org mode, or local tenancy that trusts identity headers: build a
    # per-request Principal from the identity headers. Added first so it sits
    # inner of the bearer/CORS layers: in local tenancy it reads the headers
    # only on a request that passed the bearer token.
    if settings.tenancy_mode == "org" or settings.trust_identity_headers:
        enforce = settings.identity_enforce == "enforce"
        app.add_middleware(
            TrustedHeaderMiddleware,
            exempt_paths=channel_webhook_paths,
            enforce=enforce,
            personal_partitions=settings.trust_identity_headers,
        )
    if settings.trust_identity_headers:
        logger.info(
            "auth: local tenancy, trusted identity headers enabled "
            "(identity=%s, personal resources=%s)",
            settings.identity_enforce,
            ",".join(sorted(settings.personal_resources)) or "none",
        )
    elif settings.tenancy_mode == "org":
        logger.info(
            "auth: org tenancy mode — principal middleware enabled "
            "(identity=%s)",
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
        # Codex has a separate, process-local inference credential, not the
        # desktop's API token. These exact routes retain their own constant-time
        # credential check and LoopbackDesktopOnly guard. Do not exempt the
        # coding prefix: that would expose task/filesystem APIs to the agent.
        channel_webhook_paths.update(
            f"/api/v1/coding/inference/{path}" for path in INFERENCE_PATHS
        )
        app.add_middleware(
            BearerTokenMiddleware, token=token, exempt_paths=channel_webhook_paths
        )
        logger.info("auth: bearer-token authentication enabled")

    # Configure CORS outside the authentication and principal layers. The
    # no-store wrapper added below remains outermost.
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
    # is cacheable by default wherever it's fetched from. Capabilities are
    # included so a browser never reuses a stale rollout state.
    app.add_middleware(
        _NoStoreMiddleware,
        prefixes=(
            "/api/v1/settings",
            "/api/v1/connectors/oauth",
            "/api/v1/capabilities",
        ),
    )

    # Include v1 API routes
    app.include_router(v1_router)

    _install_channels(app, channel_webhook_paths)

    # ENG-2094: every route must declare a Permission (require(...)) so a
    # walker can tell "open on purpose" from "someone forgot the line" — see
    # route_walker.py and its docstring. Boot-time, not just CI, so a gap
    # can't reach any environment undetected.
    gaps = undeclared_routes(app)
    if gaps:
        # route_key, not route.methods: APIWebSocketRoute has no .methods, so
        # reading it here would raise AttributeError over the top of the
        # message that names the offending route.
        names = ", ".join(f"{list(methods)} {path}" for path, methods in map(route_key, gaps))
        raise RuntimeError(f"routes with no declared Permission (ENG-2094): {names}")

    contradictions = contradictory_routes(app)
    if contradictions:
        names = ", ".join(f"{list(methods)} {path}" for path, methods in map(route_key, contradictions))
        raise RuntimeError(
            "routes declaring OpenByDesign alongside an identity Permission (ENG-2094): "
            f"{names} — a route-level dependency is ADDED to its router's, never substituted for it, "
            "so the stricter check still runs and the route is not open"
        )

    _warn_if_the_front_door_outranks_the_identity_layer(app)

    logger.info("Cowork application created successfully")
    return app


def _warn_if_the_front_door_outranks_the_identity_layer(app: FastAPI) -> None:
    """Say out loud when the audit rollback lever has stopped covering things.

    ``identity_enforce=audit`` tells TrustedHeaderMiddleware to log a missing
    principal and let the request through. The ENG-2094 permission classes do
    not read that flag — deliberately, see AuthenticatedInOrgMode's docstring
    — so in org mode they still 401 every route that requires identity. That
    is the right layering and the wrong surprise: an operator who reaches for
    audit to unblock traffic would get a 401 storm from a layer the flag never
    named. Naming the count at boot is what turns that into a decision instead
    of an incident.

    A warning rather than a refusal: no deployment runs this state, and a boot
    failure over a combination nothing uses buys less than it costs.
    """
    settings = get_app_settings()
    if settings.tenancy_mode != "org" or settings.identity_enforce == "enforce":
        return

    from fastapi.routing import APIRoute, APIWebSocketRoute

    from cowork.api.v1.permissions import Authenticated
    from cowork.api.v1.route_walker import declared_permissions

    still_enforcing = sum(
        1
        for route in app.routes
        if isinstance(route, (APIRoute, APIWebSocketRoute))
        and any(issubclass(cls, Authenticated) for cls in declared_permissions(route))
    )
    logger.warning(
        "identity_enforce=%s applies to TrustedHeaderMiddleware only. %d route(s) "
        "declare a Permission that requires identity and will still answer 401 "
        "without it (ENG-2094). Audit mode is not a full rollback in org tenancy.",
        settings.identity_enforce,
        still_enforcing,
    )


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

    Mounted in both local and org mode: org mode resolves the org per-request
    from the payload (LiveAdapterRegistry.resolve_org_bridge) rather than
    from one boot-time adapter, so there's no global set of credentials to
    preload here the way local mode's lifespan startup does (_start_channels).
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
            build_channel_webhook_router(
                plugin, resolver=adapters.get, sink=runtime.handle,
                org_resolver=adapters.resolve_org_bridge,
            ),
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
