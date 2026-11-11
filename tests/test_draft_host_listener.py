"""The draft host's own port: a second uvicorn server in the API process.

These tests use a real socket. In-process ASGI tests cannot show that the API
app is absent from the draft port; only a request to that port can.
"""
from __future__ import annotations

import asyncio
import inspect
import io
import json
import logging
import signal
import socket
from types import SimpleNamespace

import httpx
import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.structured_logging import console_handler, make_formatter
from cowork.draft_host import app as draft_app_module
from cowork.draft_host.app import create_draft_app
from cowork.draft_host.listener import (
    DraftListener,
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
    draft_env,  # noqa: F401  (fixture)
    draft_token,
    events,
    fake_review,
    make_artifact,
    reviewed_and_read,
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
def fresh_settings():
    """Settings rebuilt from the test's own environment, and dropped after it."""
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def bound_on_a_free_port(monkeypatch):
    """What start_draft_listener asks serve_draft_app to bind.

    The socket is bound on port 0 instead: a port picked beforehand can be taken
    by the time uvicorn binds it, and uvicorn then ends the whole test process.
    """
    asked: dict = {}

    async def serve(app, *, host, port):
        asked.update(host=host, port=port)
        return await serve_draft_app(app, host=host, port=0)

    monkeypatch.setattr("cowork.draft_host.listener.serve_draft_app", serve)
    return asked


@pytest.fixture
async def listener(draft_env):
    """A draft listener on a free loopback port; stopping it twice is harmless."""
    running = await serve_draft_app(create_draft_app(), host="127.0.0.1", port=0)
    yield running
    await running.stop()


@pytest.fixture
def serve_stub(monkeypatch):
    """What start_draft_listener asked serve_draft_app for, without binding it."""
    asked: dict = {}

    async def serve(app, *, host, port):
        asked.update(host=host, port=port)

    monkeypatch.setattr("cowork.draft_host.listener.serve_draft_app", serve)
    return asked


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


def _client(port: int) -> httpx.AsyncClient:
    # trust_env=False: a proxy from the environment must not answer for the pod.
    return httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False)


async def test_without_a_draft_port_no_listener_starts(draft_env, monkeypatch):
    monkeypatch.delenv("COWORK_DRAFT_PORT", raising=False)
    get_app_settings.cache_clear()

    assert await start_draft_listener() is None
    await stop_draft_listener(None)


async def test_the_draft_port_serves_the_draft_app_and_nothing_else(
    listener, middleware_calls, tmp_path, monkeypatch
):
    recorded: dict = {}
    source, folder, metadata = make_artifact(tmp_path)
    monkeypatch.setattr(
        draft_app_module, "review_artifact_and_read", reviewed_and_read(fake_review(source, folder, metadata, recorded))
    )
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

    assert served.status_code == 200
    assert recorded["scope"].user_id == USER
    for response in refused:
        # The draft app's own 404: the API app is not on this socket.
        assert response.status_code == 404
        assert response.headers["content-security-policy"] == DRAFT_CSP
    assert middleware_calls == []


async def test_the_api_server_keeps_the_process_signals(draft_env):
    # uvicorn.Server.serve() takes SIGINT and SIGTERM for itself. The API
    # server must keep them: it stops this server from its lifespan. Not the
    # listener fixture: the handlers are read before the server starts.
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    listener = await serve_draft_app(create_draft_app(), host="127.0.0.1", port=0)
    try:
        during = {sig: signal.getsignal(sig) for sig in before}
    finally:
        await listener.stop()

    assert during == before


async def test_stop_closes_the_port(listener):
    port = listener.port

    await listener.stop()

    assert listener.task.done()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


async def test_the_configured_port_is_served(draft_env, bound_on_a_free_port, monkeypatch):
    monkeypatch.setenv("COWORK_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("COWORK_DRAFT_PORT", "9123")
    get_app_settings.cache_clear()

    listener = await start_draft_listener()
    try:
        assert listener is not None
        assert bound_on_a_free_port == {"host": "127.0.0.1", "port": 9123}
        async with _client(listener.port) as client:
            response = await client.get("/", headers={"Host": DRAFT_HOST})
    finally:
        await stop_draft_listener(listener)

    assert response.status_code == 404
    assert response.headers["content-security-policy"] == DRAFT_CSP


@pytest.mark.parametrize(
    "name, value",
    [
        ("COWORK_DRAFT_LINK_KID", "missing"),
        ("COWORK_DRAFT_LINK_KEYS", json.dumps({"k1": "too-short"})),
    ],
    ids=["unknown-kid", "short-secret"],
)
async def test_a_half_configured_draft_host_is_reported_at_boot(
    draft_env, serve_stub, monkeypatch, caplog, name, value
):
    monkeypatch.setenv(name, value)
    get_app_settings.cache_clear()

    with caplog.at_level("WARNING", logger="cowork.draft_host.listener"):
        await start_draft_listener()

    # Still started: the draft app answers 404 until the value is fixed.
    assert serve_stub["port"] == 9011
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


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
async def test_a_draft_listener_on_a_loopback_address_is_reported_at_boot(
    draft_env, serve_stub, monkeypatch, caplog, host
):
    monkeypatch.setenv("COWORK_SERVER_HOST", host)
    get_app_settings.cache_clear()

    with caplog.at_level("WARNING", logger="cowork.draft_host.listener"):
        await start_draft_listener()

    [record] = events(caplog, "draft_host_loopback")
    assert record.levelno == logging.WARNING
    message = json.loads(make_formatter().format(record))["message"]
    assert "COWORK_SERVER_HOST" in message and "0.0.0.0" in message
    assert host not in message
    assert serve_stub["host"] == host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.1.2.3"])
async def test_a_draft_listener_on_a_pod_address_is_not_reported(draft_env, serve_stub, monkeypatch, caplog, host):
    monkeypatch.setenv("COWORK_SERVER_HOST", host)
    get_app_settings.cache_clear()

    with caplog.at_level("WARNING", logger="cowork.draft_host.listener"):
        await start_draft_listener()

    assert events(caplog, "draft_host_loopback") == []
    assert serve_stub["host"] == host


async def test_a_loopback_address_is_not_reported_while_the_draft_host_is_off(draft_env, serve_stub, caplog, monkeypatch):
    monkeypatch.setenv("COWORK_DRAFT_LINK_KID", "missing")
    get_app_settings.cache_clear()

    with caplog.at_level("WARNING", logger="cowork.draft_host.listener"):
        await start_draft_listener()

    assert events(caplog, "draft_host_loopback") == []


async def test_a_listener_that_stops_by_itself_is_logged(listener, caplog):
    with caplog.at_level("ERROR", logger="cowork.draft_host.listener"):
        # Ends the server without going through stop(), as a failure would.
        listener.server.should_exit = True
        await listener.task
        await asyncio.sleep(0)

    [record] = events(caplog, "draft_host_stopped")
    assert record.levelno == logging.ERROR
    assert json.loads(make_formatter().format(record))["message"] == "Draft host stopped"


async def test_a_crashed_listener_is_logged_with_its_error_type(caplog):
    async def crash():
        raise RuntimeError("the draft server broke")

    listener = DraftListener(SimpleNamespace(should_exit=False), asyncio.create_task(crash()))

    with caplog.at_level("ERROR", logger="cowork.draft_host.listener"):
        with pytest.raises(RuntimeError):
            await listener.task
        await asyncio.sleep(0)

    [record] = events(caplog, "draft_host_stopped")
    assert json.loads(make_formatter().format(record))["error_type"] == "RuntimeError"


async def test_a_deliberate_stop_is_not_logged_as_a_failure(listener, caplog):
    with caplog.at_level("INFO", logger="cowork.draft_host.listener"):
        await listener.stop()
        await asyncio.sleep(0)

    assert events(caplog, "draft_host_stopped") == []


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


async def test_a_listener_that_fails_to_start_still_runs_the_shutdown(fresh_settings, monkeypatch, tmp_path):
    from fastapi import FastAPI

    from cowork import server
    from cowork.common import http_client

    calls: list[str] = []

    async def refuse() -> None:
        raise RuntimeError("the draft listener could not start")

    async def stopped(listener) -> None:
        calls.append(f"stop_draft_listener({listener})")

    async def closed() -> None:
        calls.append("close_proxy_client")

    async def nothing(*args) -> bool:
        return False

    def boot_database_is_off():
        raise RuntimeError("no database in this test")

    # Boot work that needs the database, the model map or the channels is out of
    # scope here; the steps that remain run as they do in the server.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()
    monkeypatch.setattr(server, "run_dev_setup", lambda: None)
    monkeypatch.setattr(server, "consume_pending_credential_clear", lambda *args: None)
    monkeypatch.setattr(server, "_warm_agent_runtime", lambda: None)
    monkeypatch.setattr(server, "_warm_model_map_on_boot", nothing)
    monkeypatch.setattr(server, "_start_channels", nothing)
    monkeypatch.setattr(server, "start_scheduler", lambda: None)
    monkeypatch.setattr("cowork.db.session.get_open_session", boot_database_is_off)
    monkeypatch.setattr(server, "start_draft_listener", refuse)
    monkeypatch.setattr(server, "stop_draft_listener", stopped)
    monkeypatch.setattr(http_client, "close_proxy_client", closed)
    app = FastAPI()
    app.state.channel_ingress = SimpleNamespace(stop_all=nothing)
    app.state.channel_adapters = SimpleNamespace(shutdown=nothing)

    with pytest.raises(RuntimeError, match="the draft listener could not start"):
        async with server.lifespan(app):
            pass

    assert calls == ["stop_draft_listener(None)", "close_proxy_client"]
