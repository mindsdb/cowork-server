"""The draft-preview host's own port, next to the API server in one process.

ingress-nginx runs with use-forwarded-headers, so a client picks the Host the
pod sees by sending X-Forwarded-Host, and the load balancer also takes
requests that skip the Cloudflare worker. A Host check on the API's port
would let such a request reach the API app, whose TrustedHeaderMiddleware
trusts X-User-*. So the draft sub-app gets a socket of its own and the draft
Ingress routes to that port only: the API app is not on it.

The listener is a second uvicorn server in the API server's event loop, which
the API app's lifespan starts and stops (cowork/server.py). The image command,
the probes and the API server's own flags stay as they are.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging

import uvicorn
from starlette.types import ASGIApp

from cowork.common.settings.app_settings import get_app_settings

logger = logging.getLogger(__name__)

# Upper bound on draining in-flight draft responses at shutdown.
_GRACEFUL_SHUTDOWN_SECONDS = 2


class _DraftServer(uvicorn.Server):
    """A uvicorn server that leaves SIGINT and SIGTERM to the API server.

    ``Server.serve()`` installs handlers for both. Here the API server keeps
    them and stops this server from its lifespan shutdown.
    """

    @contextlib.contextmanager
    def capture_signals(self):
        yield


class DraftListener:
    """A running draft listener: its server and the task that runs it."""

    def __init__(self, server: uvicorn.Server, task: asyncio.Task) -> None:
        self.server = server
        self.task = task

    @property
    def port(self) -> int:
        """The bound port, also when port 0 asked for any free one."""
        return self.server.servers[0].sockets[0].getsockname()[1]

    async def stop(self) -> None:
        self.server.should_exit = True
        await self.task


async def serve_draft_app(app: ASGIApp, *, host: str, port: int) -> DraftListener:
    """Run ``app`` on its own uvicorn server and return once it listens.

    A port that cannot be bound makes uvicorn exit the process, as it does for
    the API port, so a pod never runs without its draft port.
    """
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        lifespan="off",
        # None keeps the process's logging (setup_logging), whose access
        # handler drops the request path, and with it the link token. Any
        # other value makes uvicorn replace those handlers for the whole
        # process with its own, which print every path.
        log_config=None,
        timeout_graceful_shutdown=_GRACEFUL_SHUTDOWN_SECONDS,
    )
    server = _DraftServer(config)
    task = asyncio.create_task(server.serve(), name="draft-listener")
    while not server.started:
        if task.done():
            task.result()
            raise RuntimeError("the draft listener stopped before it listened")
        await asyncio.sleep(0.01)
    return DraftListener(server, task)


async def start_draft_listener() -> DraftListener | None:
    """Serve the draft sub-app on COWORK_DRAFT_PORT; None when no port is set."""
    settings = get_app_settings()
    if settings.draft_port <= 0:
        return None
    if not settings.draft_links_enabled():
        logger.warning(
            "draft_host: COWORK_DRAFT_PORT is set, but org tenancy, COWORK_DRAFT_HOSTS, "
            "COWORK_DRAFT_ORIGIN, COWORK_WEB_ORIGIN or the COWORK_DRAFT_LINK_KID key is "
            "missing; the draft host answers 404 and previews stay on srcdoc",
            extra={"event": "draft_host_misconfigured"},
        )
    # Imported here: app.py pulls in the API endpoint modules.
    from cowork.draft_host.app import create_draft_app

    listener = await serve_draft_app(create_draft_app(), host=settings.host, port=settings.draft_port)
    logger.info("Draft host listening", extra={"event": "draft_host_listening"})
    return listener


async def stop_draft_listener(listener: DraftListener | None) -> None:
    """Stop the listener; a failure is logged so the rest of shutdown still runs."""
    if listener is None:
        return
    try:
        await listener.stop()
    except Exception:
        logger.exception("Draft listener stop failed", extra={"event": "draft_host_stop_failed"})
