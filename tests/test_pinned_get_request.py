"""The exact request a pinned GET sends, and that no library logs while it runs.

A key placed in a query rides in the URL, and a provider may echo it in a
redirect or a header; httpx and httpcore write both into their own records.
The real-transport tests run a local TLS origin with a throwaway CA.
"""
from __future__ import annotations

import ast
import asyncio
import datetime
import ipaddress
import logging
import socket
import ssl
import time
from pathlib import Path

import httpcore
import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from cowork.services.connectors import egress
from cowork.services.connectors.egress import LIBRARY_LOGGERS, EgressRefused, pinned_get

_HOST = "api.example.test"
_KEY = "k3y+PLANTED/value="


def _resolve_to(address: str, port: int = 443):
    def resolve(host, *args, **kwargs):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 0, "", (address, port))]

    return resolve


@pytest.fixture
def mocked(monkeypatch):
    """Route pinned_get through a fake transport; returns the requests seen."""
    requests: list[httpx.Request] = []

    def use(handler=None):
        async def record(request):
            requests.append(request)
            if handler is not None:
                return await handler(request)
            return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

        monkeypatch.setattr(egress, "_RESOLVER", _resolve_to("93.184.216.34"))
        monkeypatch.setattr(egress, "_TRANSPORT", httpx.MockTransport(record))
        return requests

    return use


class _Chunks(httpx.AsyncByteStream):
    """A body sent in many chunks, so a capped read stops before its end."""

    async def __aiter__(self):
        for _ in range(50):
            yield b"chunk"


async def test_a_send_closes_its_response_and_its_own_transport(monkeypatch):
    responses: list[httpx.Response] = []
    closed: list[httpx.AsyncBaseTransport] = []

    def answer(request):
        responses.append(httpx.Response(200, stream=_Chunks()))
        return responses[-1]

    class PerSend(httpx.MockTransport):
        def __init__(self, **kwargs):
            super().__init__(answer)

        async def aclose(self):
            closed.append(self)

    monkeypatch.setattr(egress, "_RESOLVER", _resolve_to("93.184.216.34"))
    monkeypatch.setattr(egress.httpx, "AsyncHTTPTransport", PerSend)

    got = await pinned_get("https://api.example.com", "/v1/me", {}, total_seconds=2.0, max_body_bytes=8)

    assert got.truncated
    assert responses[0].is_closed
    assert len(closed) == 1


async def test_the_request_carries_exactly_the_callers_headers_and_ours(mocked):
    requests = mocked()

    await pinned_get("https://api.example.com", "/v1/me", {"X-Api-Key": "k"}, total_seconds=2.0, max_body_bytes=64)

    assert [(name.decode(), value.decode()) for name, value in requests[0].headers.raw] == [
        ("X-Api-Key", "k"),
        ("Accept", "*/*"),
        ("User-Agent", "MindsHub-Connector/1"),
        ("Host", "api.example.com"),
        ("Accept-Encoding", "identity"),
    ]


async def test_a_callers_user_agent_replaces_ours_and_it_cannot_set_host_or_encoding(mocked):
    requests = mocked()
    headers = {"user-agent": "Mine/2", "ACCEPT": "application/json", "HOST": "evil.example", "accept-encoding": "gzip"}

    await pinned_get("https://api.example.com", "/v1/me", headers, total_seconds=2.0, max_body_bytes=64)

    assert [(name.decode(), value.decode()) for name, value in requests[0].headers.raw] == [
        ("user-agent", "Mine/2"),
        ("ACCEPT", "application/json"),
        ("Host", "api.example.com"),
        ("Accept-Encoding", "identity"),
    ]


@pytest.mark.parametrize(
    ("path", "params", "target"),
    [
        ("/v1/me", [], b"/v1/me"),
        ("/v1/me", [("access_key", _KEY)], b"/v1/me?access_key=k3y%2BPLANTED%2Fvalue%3D"),
        ("/v1/me?a=1;b=2", [("k", "v")], b"/v1/me?a=1;b=2&k=v"),
        ("/v1/me?pretty", [("k", "v")], b"/v1/me?pretty&k=v"),
        ("/v1/me?", [("k", "a b&c")], b"/v1/me?k=a%20b%26c"),
        ("/v1/me", [("a", "1"), ("b", "2")], b"/v1/me?a=1&b=2"),
    ],
)
async def test_params_follow_the_paths_own_query_as_written(mocked, path, params, target):
    requests = mocked()

    await pinned_get("https://api.example.com", path, {}, total_seconds=2.0, max_body_bytes=64, params=params)

    assert requests[0].url.raw_path == target
    assert requests[0].url.host == "93.184.216.34"


async def test_each_attempt_carries_its_own_connect_limit(mocked, monkeypatch):
    async def refuse_first(request):
        if len(requests) == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

    requests = mocked(refuse_first)
    monkeypatch.setattr(
        egress,
        "_RESOLVER",
        lambda host, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.35", 443)),
        ],
    )

    await pinned_get("https://api.example.com", "/v1/me", {}, total_seconds=3.0, max_body_bytes=64)

    assert [request.extensions["timeout"] for request in requests] == [
        {"connect": 1.0, "read": 3.0, "write": 3.0, "pool": 3.0},
        {"connect": 2.0, "read": 3.0, "write": 3.0, "pool": 3.0},
    ]


def _library_logger_names() -> set[str]:
    """Every logger name httpx and httpcore ask for, refusing anything not a literal."""
    names: set[str] = set()
    for package in (httpx, httpcore):
        for source in Path(package.__file__).parent.rglob("*.py"):
            for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
                if not isinstance(node, ast.Call):
                    continue
                called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert called != "getChild", f"{source} derives a logger with getChild"
                if called != "getLogger":
                    continue
                assert node.args and isinstance(node.args[0], ast.Constant), f"{source} names a logger at runtime"
                names.add(node.args[0].value)
    return names


def test_the_quiet_rule_covers_every_logger_httpx_and_httpcore_write_to():
    assert _library_logger_names() == set(LIBRARY_LOGGERS)


def _certificates(tmp_path: Path) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """A throwaway CA and a certificate for _HOST: the server's context and the client's."""
    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _HOST)]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(_HOST)]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    cert_file, key_file = tmp_path / "leaf.pem", tmp_path / "leaf.key"
    cert_file.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    server = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server.load_cert_chain(cert_file, key_file)
    client = ssl.create_default_context(cadata=ca.public_bytes(serialization.Encoding.PEM).decode())
    return server, client


async def _echo_origin(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Answer 301 with the request target in Location, a header and the body; logs nothing."""
    head = await reader.readuntil(b"\r\n\r\n")
    target = head.split(b" ", 2)[1]
    body = b"you asked for " + target
    writer.write(
        b"HTTP/1.1 301 Moved Permanently\r\nLocation: https://" + _HOST.encode() + target
        + b"\r\nX-Echo: " + target + b"\r\nContent-Length: " + str(len(body)).encode()
        + b"\r\nConnection: close\r\n\r\n" + body
    )
    await writer.drain()
    writer.close()


@pytest.fixture
async def tls_origin(tmp_path, monkeypatch):
    """A local TLS origin for _HOST, reached through the real per-send transport."""
    server_context, client_context = _certificates(tmp_path)
    server = await asyncio.start_server(_echo_origin, "127.0.0.1", 0, ssl=server_context)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(egress, "_RESOLVER", _resolve_to("127.0.0.1", port))
    monkeypatch.setattr(egress, "is_public_address", lambda address: address == ipaddress.ip_address("127.0.0.1"))
    monkeypatch.setattr(egress, "_VERIFY", client_context)
    yield f"https://{_HOST}:{port}"
    server.close()
    await server.wait_closed()


@pytest.fixture
def every_record(caplog):
    """Capture every record at DEBUG, the libraries' own loggers included."""
    caplog.set_level(logging.DEBUG)
    for name in LIBRARY_LOGGERS:
        caplog.set_level(logging.DEBUG, logger=name)
    return caplog


def _rendered(caplog) -> list[str]:
    return [f"{record.name}: {record.getMessage()} {record.args!r}" for record in caplog.records]


async def test_a_key_echoed_by_the_origin_reaches_no_record(tls_origin, every_record):
    response = await pinned_get(
        tls_origin, "/v1/me", {"X-Api-Key": _KEY}, total_seconds=5.0, max_body_bytes=4096,
        params=[("access_key", _KEY)],
    )

    assert response.status_code == 301
    assert b"access_key=k3y%2BPLANTED%2Fvalue%3D" in response.body
    for line in _rendered(every_record):
        assert "k3y" not in line, line
    assert not [record for record in every_record.records if record.name in LIBRARY_LOGGERS]


async def test_library_records_pass_again_after_a_send_returns_or_raises(tls_origin, every_record):
    library = logging.getLogger("httpcore.http11")

    await pinned_get(tls_origin, "/v1/me", {}, total_seconds=5.0, max_body_bytes=64)
    library.debug("after a send that returned")
    with pytest.raises(EgressRefused):
        await pinned_get(tls_origin.replace("https://", "http://"), "/v1/me", {}, total_seconds=5.0, max_body_bytes=64)
    library.debug("after a send that raised")

    messages = [record.getMessage() for record in every_record.records if record.name == "httpcore.http11"]
    assert messages == ["after a send that returned", "after a send that raised"]


async def test_a_plain_client_call_beside_a_pinned_send_keeps_its_records(mocked, every_record):
    started = asyncio.Event()

    async def slow(request):
        started.set()
        await asyncio.sleep(0.2)
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

    mocked(slow)

    async def plain():
        await started.wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(204))) as client:
            await client.get("https://plain.example/ping")

    await asyncio.gather(
        pinned_get("https://api.example.com", "/v1/me", {}, total_seconds=2.0, max_body_bytes=64), plain()
    )

    assert any("plain.example/ping" in record.getMessage() for record in every_record.records if record.name == "httpx")


async def test_a_stalled_handshake_gives_up_at_its_connect_limit_and_the_next_address_is_tried(monkeypatch):
    accepted: list[socket.socket] = []

    async def stall(reader, writer):
        accepted.append(writer.get_extra_info("socket"))
        await asyncio.sleep(10)

    server = await asyncio.start_server(stall, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    record = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", port))
    monkeypatch.setattr(egress, "_RESOLVER", lambda host, *a, **k: [record, record])
    monkeypatch.setattr(egress, "is_public_address", lambda address: True)
    started = time.monotonic()
    try:
        # The last attempt's limit is the time left, so the deadline may end it first.
        with pytest.raises((httpx.ConnectTimeout, egress.EgressTimeout)):
            await pinned_get(f"https://{_HOST}:{port}", "/v1/me", {}, total_seconds=2.5, max_body_bytes=64)
    finally:
        server.close()

    # Without a per-attempt connect limit the first handshake holds the whole deadline.
    assert len(accepted) == 2
    assert time.monotonic() - started < 3.0
