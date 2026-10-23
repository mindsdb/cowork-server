"""The draft host's own port: a second uvicorn server in the API process.

These tests use a real socket. In-process ASGI tests cannot show that the API
app is absent from the draft port; only a request to that port can.
"""
from __future__ import annotations

import inspect
import io
import json
import logging
import signal
import socket

import httpx
import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.structured_logging import console_handler, make_formatter
from cowork.draft_host import app as draft_app_module
from cowork.draft_host.app import create_draft_app
from cowork.draft_host.listener import (
    serve_draft_app,
    start_draft_listener,
    stop_draft_listener,
)
from cowork.principal import TrustedHeaderMiddleware

from draft_host_support import (
    DRAFT_CSP,
    DRAFT_HOST,
    FETCH,
    ORG,
    USER,
    configure_draft_env,
    draft_token,
    fake_review,
    make_artifact,
)

COWORK_HOST = "cowork.staging.mindshub.ai"
# What a client sends to reach the API app through the draft Ingress.
FORGED = {
    "X-Forwarded-Host": COWORK_HOST,
    "X-User-Id": "99999999-9999-4999-8999-999999999999",
    "X-Organization-Id": ORG,
    "X-User-Email": "attacker@example.com",
    "X-Forwarded-User": "attacker",
}


@pytest.fixture
def draft_env(monkeypatch):
    configure_draft_env(monkeypatch)
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def middleware_calls(monkeypatch):
    """Every path the API app's identity middleware sees."""
    calls: list[str] = []
    original = TrustedHeaderMiddleware.dispatch

    async def recording(self, request, call_next):
        calls.append(request.url.path)
        return await original(self, request, call_next)

    monkeypatch.setattr(TrustedHeaderMiddleware, "dispatch", recording)
    return calls


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _client(port: int) -> httpx.AsyncClient:
    # trust_env=False: a proxy from the environment must not answer for the pod.
    return httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False)


async def test_without_a_draft_port_no_listener_starts(draft_env, monkeypatch):
    monkeypatch.delenv("COWORK_DRAFT_PORT", raising=False)
    get_app_settings.cache_clear()

    assert await start_draft_listener() is None
    await stop_draft_listener(None)


async def test_the_draft_port_serves_the_draft_app_and_nothing_else(
    draft_env, middleware_calls, tmp_path, monkeypatch
):
    recorded: dict = {}
    source, folder, metadata = make_artifact(tmp_path)
    monkeypatch.setattr(
        draft_app_module, "review_artifact_for_request", fake_review(source, folder, metadata, recorded)
    )
    listener = await serve_draft_app(create_draft_app(), host="127.0.0.1", port=0)
    try:
        async with _client(listener.port) as client:
            served = await client.get(
                f"/_t/{draft_token()}/index.html", headers={"Host": DRAFT_HOST, **FETCH, **FORGED}
            )
            refused = [
                await client.get("/api/v1/health/", headers={"Host": DRAFT_HOST, **FORGED}),
                await client.get("/api/v1/health/", headers={"Host": COWORK_HOST, **FORGED}),
                await client.get(
                    f"/_t/{draft_token()}/index.html", headers={"Host": COWORK_HOST, **FETCH, **FORGED}
                ),
            ]
    finally:
        await listener.stop()

    assert served.status_code == 200
    assert recorded["scope"].user_id == USER
    for response in refused:
        # The draft app's own 404: the API app is not on this socket.
        assert response.status_code == 404
        assert response.headers["content-security-policy"] == DRAFT_CSP
    assert middleware_calls == []


async def test_the_api_server_keeps_the_process_signals(draft_env):
    # uvicorn.Server.serve() takes SIGINT and SIGTERM for itself. The API
    # server must keep them: it stops this server from its lifespan.
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    listener = await serve_draft_app(create_draft_app(), host="127.0.0.1", port=0)
    try:
        during = {sig: signal.getsignal(sig) for sig in before}
    finally:
        await listener.stop()

    assert during == before


async def test_stop_closes_the_port(draft_env):
    listener = await serve_draft_app(create_draft_app(), host="127.0.0.1", port=0)
    port = listener.port

    await listener.stop()

    assert listener.task.done()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


async def test_the_configured_port_is_served(draft_env, monkeypatch):
    port = _free_port()
    monkeypatch.setenv("COWORK_DRAFT_PORT", str(port))
    get_app_settings.cache_clear()

    listener = await start_draft_listener()
    try:
        assert listener is not None
        assert listener.port == port
        async with _client(port) as client:
            response = await client.get("/", headers={"Host": DRAFT_HOST})
    finally:
        await stop_draft_listener(listener)

    assert response.status_code == 404
    assert response.headers["content-security-policy"] == DRAFT_CSP


async def test_a_half_configured_draft_host_is_reported_at_boot(draft_env, monkeypatch, caplog):
    monkeypatch.setenv("COWORK_DRAFT_PORT", str(_free_port()))
    monkeypatch.setenv("COWORK_DRAFT_LINK_KID", "missing")
    get_app_settings.cache_clear()

    with caplog.at_level("WARNING", logger="cowork.draft_host.listener"):
        listener = await start_draft_listener()
    await stop_draft_listener(listener)

    assert listener is not None
    [record] = [r for r in caplog.records if r.name == "cowork.draft_host.listener"]
    assert record.event == "draft_host_misconfigured"
    assert "draft host answers 404" in json.loads(make_formatter().format(record))["message"]


async def test_the_draft_port_access_line_never_carries_the_token(draft_env, monkeypatch):
    # log_config=None is what keeps the process's access handler (the one
    # setup_logging installs), which writes no request path. A draft server
    # that installed uvicorn's own logging would print every path, token and
    # all, for the whole process.
    access = logging.getLogger("uvicorn.access")
    out = io.StringIO()
    handler = console_handler(access=True)
    handler.setStream(out)
    monkeypatch.setattr(access, "handlers", [handler])
    level = access.level
    access.setLevel(logging.INFO)
    token = draft_token()
    try:
        listener = await serve_draft_app(create_draft_app(), host="127.0.0.1", port=0)
        try:
            during = list(access.handlers)
            async with _client(listener.port) as client:
                response = await client.get(f"/_t/{token}/index.html", headers={"Host": DRAFT_HOST})
        finally:
            await listener.stop()
    finally:
        access.setLevel(level)

    assert response.status_code == 401
    assert during == [handler]
    text = out.getvalue()
    assert token not in text
    lines = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    assert any(line.get("event") == "http_access" and line.get("status_code") == 401 for line in lines)


async def test_a_failed_stop_is_logged_and_shutdown_goes_on(caplog):
    class _Broken:
        async def stop(self):
            raise RuntimeError("the draft server broke")

    with caplog.at_level("ERROR", logger="cowork.draft_host.listener"):
        await stop_draft_listener(_Broken())

    [record] = [r for r in caplog.records if r.name == "cowork.draft_host.listener"]
    assert record.event == "draft_host_stop_failed"


def test_the_api_lifespan_starts_the_listener_and_stops_it_first():
    # Read from the source, like the other lifespan-order tests: running the
    # whole lifespan here would run boot work against the test database.
    from cowork import server

    source = inspect.getsource(server.lifespan)
    start = source.index("await start_draft_listener()")
    stop = source.index("await stop_draft_listener(")

    assert start < source.index("yield") < stop < source.index("await registry.shutdown()")
