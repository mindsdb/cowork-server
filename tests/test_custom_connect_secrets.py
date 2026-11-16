"""A submitted value never reaches a log record or a response, whatever the connect does."""

from __future__ import annotations

import asyncio
import base64
import gzip
import ipaddress
import logging
from urllib.parse import quote

import httpx
import pytest

from cowork.services.connectors import custom_connect, egress
from tests import test_custom_connect as connect
from tests import test_pinned_get_request as pinned

SECRET = connect.SECRET
_client = connect._client
_connect = connect._connect
engine = connect.engine
world = connect.world
#: A key whose URL encoding differs from its text, so an echoed encoded copy is a distinct leak.
ENCODED_SECRET = "sk+live/planted=value never echoed"


class _Every(logging.Handler):
    """Collect every record any logger emits, formatted with its arguments."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())
        if record.exc_info:
            self.lines.append(logging.Formatter().formatException(record.exc_info))


@pytest.fixture()
def records():
    """Attach to the root and to every named logger, whatever their propagation."""
    handler = _Every()
    loggers = [logging.getLogger()] + [
        logger for logger in logging.root.manager.loggerDict.values() if isinstance(logger, logging.Logger)
    ]
    saved = [(logger, logger.level) for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    yield handler.lines
    for logger, level in saved:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _echo(status: int) -> tuple[int, bytes]:
    return status, f'{{"error":"bad key {SECRET}","key":"{SECRET}"}}'.encode()


@pytest.mark.parametrize(
    ("setup", "body"),
    [
        ("pass", {}),
        ("401", {}),
        ("500", {}),
        ("unprotected", {}),
        ("gzip", {}),
        ("dropped", {}),
        ("refused-address", {"address": f"{SECRET}.mindshub.ai"}),
        ("trailing-newline", {"values": {"api_key": SECRET + "\n"}}),
        ("non-ascii", {"values": {"api_key": SECRET + "é"}}),
        ("unknown-field", {"values": {"api_key": SECRET, SECRET: "x"}}),
    ],
)
def test_a_planted_value_reaches_no_log_and_no_response(engine, world, monkeypatch, records, setup, body):
    if setup == "401":
        world["protected"] = _echo(401)
    elif setup == "500":
        world["protected"] = _echo(500)
    elif setup == "unprotected":
        world["unprotected"] = _echo(200)
    elif setup == "gzip":
        world["protected"] = (500, gzip.compress(_echo(500)[1]))
    elif setup == "dropped":
        async def drop(request):
            raise httpx.ReadError(f"connection reset while sending {request.headers['authorization']}")

        world["protected"] = drop

    res = _connect(_client(engine, monkeypatch), **body)

    assert res.status_code in (201, 409, 422)
    assert SECRET not in res.text
    assert records, "the capture saw nothing; the check would pass for nothing"
    assert not [line for line in records if SECRET in line]


def _leaks(text: str, *values: str) -> list[str]:
    """Every form of ``values`` a member's screen or a log could show."""
    forms = {form for value in values for form in (value, quote(value, safe=""), quote(value))}
    return [form for form in forms if form in text]


@pytest.mark.parametrize("setup", ["query-401-echoes-url", "query-500-echoes-encoded", "query-pass"])
def test_a_query_key_reaches_no_log_and_no_response(engine, world, monkeypatch, records, setup):
    if setup == "query-401-echoes-url":
        async def echo_url(request):
            return httpx.Response(401, headers={"Location": str(request.url)}, stream=httpx.ByteStream(
                f"you asked for {request.url}".encode()))

        world["protected"] = echo_url
    elif setup == "query-500-echoes-encoded":
        world["protected"] = (500, f'{{"key":"{quote(ENCODED_SECRET, safe="")}"}}'.encode())
    monkeypatch.setattr(connect, "SECRET", ENCODED_SECRET)

    res = _connect(_client(engine, monkeypatch), "query_api", values={"access_key": ENCODED_SECRET})

    assert res.status_code in (201, 422), res.text
    assert records, "the capture saw nothing; the check would pass for nothing"
    assert not _leaks(res.text, ENCODED_SECRET)
    assert not [line for line in records if _leaks(line, ENCODED_SECRET)]


def test_a_basic_token_echoed_alone_reaches_no_log_and_no_response(engine, world, monkeypatch, records):
    token = base64.b64encode(f"me@x.io/token:{SECRET}".encode()).decode()
    world["protected"] = (401, f'{{"seen":"{token}","trimmed":"{token.rstrip("=")}"}}'.encode())

    res = _connect(_client(engine, monkeypatch), "basic_api", values={"email": "me@x.io", "api_token": SECRET})

    assert res.json()["detail"]["code"] == "authentication_failed"
    assert token.rstrip("=") not in res.text and SECRET not in res.text
    assert not [line for line in records if SECRET in line or token.rstrip("=") in line]


def test_a_value_with_a_trailing_space_is_refused_before_anything_is_sent(engine, world, monkeypatch, records):
    res = _connect(_client(engine, monkeypatch), values={**connect.VALUES, "api_key": SECRET + " "})

    assert (res.status_code, res.json()["detail"]["code"]) == (422, "invalid_values")
    assert world["provider"] == [] and SECRET not in res.text
    assert not [line for line in records if SECRET in line]


async def _guarded_echo_origin(reader, writer):
    """401 without the key; with it, 301 repeating the URL in Location, a header and the body. Logs nothing."""
    head = await reader.readuntil(b"\r\n\r\n")
    target = head.split(b" ", 2)[1]
    if b"access_key=" not in target:
        writer.write(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    else:
        body = b"you asked for " + target
        writer.write(
            b"HTTP/1.1 301 Moved Permanently\r\nLocation: https://" + pinned._HOST.encode() + target
            + b"\r\nX-Echo: " + target + b"\r\nContent-Length: " + str(len(body)).encode()
            + b"\r\nConnection: close\r\n\r\n" + body
        )
    await writer.drain()
    writer.close()


@pytest.fixture
async def guarded_origin(tmp_path, monkeypatch):
    """A local TLS origin that wants the key, reached through the real per-send transport."""
    server_context, client_context = pinned._certificates(tmp_path)
    server = await asyncio.start_server(_guarded_echo_origin, "127.0.0.1", 0, ssl=server_context)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(egress, "_RESOLVER", pinned._resolve_to("127.0.0.1", port))
    monkeypatch.setattr(egress, "is_public_address", lambda address: address == ipaddress.ip_address("127.0.0.1"))
    monkeypatch.setattr(egress, "_VERIFY", client_context)
    yield f"https://{pinned._HOST}:{port}"
    server.close()
    await server.wait_closed()


async def test_a_query_check_through_the_real_transport_leaves_the_key_in_no_record(guarded_origin, records):
    template = custom_connect.StoredTemplate(
        path="/v1/status", auth={"type": "query", "params": {"access_key": "{access_key}"}},
    )

    failure = await custom_connect.run_check(guarded_origin, template, {"access_key": ENCODED_SECRET})
    logging.getLogger("httpcore.http11").debug("library records pass again after the check")

    assert (failure.code, failure.provider_status) == ("provider_error", 301)
    assert "library records pass again after the check" in records
    assert not [line for line in records if _leaks(line, ENCODED_SECRET)]
    assert not _leaks(failure.excerpt, ENCODED_SECRET)
