"""The render-service PDF client: its wire, its statuses, its budget and its limits."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer

import httpx
import pytest
import yaml
from pypdf import PdfWriter

from cowork.schemas.artifact_export import ExportPdfSettings, RenderPdfRequest, RenderPdfResult
from cowork.services.render_pdf_client import FULL_JOB_SECONDS, RenderPdfFailed, RenderServicePdf
from _no_fetch import recording_server

FIXTURES = Path(__file__).parent / "fixtures"
CASES = yaml.safe_load((FIXTURES / "render_pdf_cases.yaml").read_text(encoding="utf-8"))
REQUEST_JSON = (FIXTURES / "render_pdf" / "request.json").read_bytes()
RESPONSE_JSON = (FIXTURES / "render_pdf" / "response.json").read_bytes()
REQUEST = RenderPdfRequest.model_validate_json(REQUEST_JSON)
FIXTURE_PDF = base64.b64decode(RenderPdfResult.model_validate_json(RESPONSE_JSON).pdf_base64, validate=True)

REDIRECT_STATUSES = [case["upstream"] for case in CASES["upstream_statuses"] if 300 <= case["upstream"] < 400]

Handler = Callable[[httpx.Request], httpx.Response]


def _settings(
    *,
    render_url: str = "http://render.test:9000",
    timeout_seconds: float = 60.0,
    max_pdf_bytes: int = 48 * 1024 * 1024,
) -> ExportPdfSettings:
    return ExportPdfSettings(
        engine="render",
        render_url=render_url,
        timeout_seconds=timeout_seconds,
        max_pdf_bytes=max_pdf_bytes,
    )


def _client(handler: Handler, **settings: float | int | str) -> RenderServicePdf:
    return RenderServicePdf(settings=_settings(**settings), transport=httpx.MockTransport(handler))


def _answer(pdf: bytes) -> bytes:
    return RenderPdfResult(pdf_base64=base64.b64encode(pdf).decode(), refused=[]).model_dump_json().encode()


def _ok(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=RESPONSE_JSON, headers={"Content-Type": "application/json"})


async def _status(client: RenderServicePdf) -> int:
    with pytest.raises(RenderPdfFailed) as failure:
        await client.render(request=REQUEST)
    return failure.value.status


def _blank_pages(count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(count):
        writer.add_blank_page(width=72, height=72)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _pdf(*objects: str) -> bytes:
    """A PDF with a classic cross-reference table. Object 1 is the catalog."""
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


CATALOG = "<< /Type /Catalog /Pages 2 0 R >>"
PAGE = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] >>"


@dataclass(frozen=True)
class _StubAnswer:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


class _StubHandler(BaseHTTPRequestHandler):
    server: _StubRenderService

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.paths.append(self.path)
        answer = self.server.answer
        self.send_response(answer.status)
        for name, value in answer.headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(answer.body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(answer.body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _StubRenderService(ThreadingHTTPServer):
    """A 127.0.0.1 stand-in for a render service that gives every POST one answer."""

    daemon_threads = True

    def __init__(self, *, answer: _StubAnswer) -> None:
        self.answer = answer
        self.paths: list[str] = []
        super().__init__(("127.0.0.1", 0), _StubHandler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind looks up the host's name; skip that lookup.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    @property
    def url(self) -> str:
        return f"http://{self.server_name}:{self.server_port}"


@contextmanager
def _stub_render_service(answer: _StubAnswer) -> Iterator[_StubRenderService]:
    server = _StubRenderService(answer=answer)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _closed_port_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


# The wire.


async def test_render_posts_the_request_to_v1_pdf_and_returns_the_pdf():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(request)

    pdf = await _client(handler).render(request=REQUEST)

    assert pdf == FIXTURE_PDF
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == "http://render.test:9000/v1/pdf"
    assert seen[0].headers["Content-Type"] == "application/json"


async def test_the_request_body_is_the_fixture_wire():
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return _ok(request)

    await _client(handler).render(request=REQUEST)

    assert RenderPdfRequest.model_validate_json(bodies[0]) == REQUEST
    assert json.loads(bodies[0]) == json.loads(REQUEST_JSON)


async def test_non_ascii_text_goes_out_as_utf8_not_escapes():
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return _ok(request)

    request = RenderPdfRequest(html="<p>Résumé 数据</p>", scripts=False)
    await _client(handler).render(request=request)

    assert "Résumé 数据".encode() in bodies[0]
    assert b"\\u" not in bodies[0]


async def test_the_render_url_keeps_its_path_prefix():
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return _ok(request)

    await _client(handler, render_url="http://render.test:9000/base/").render(request=REQUEST)

    assert urls == ["http://render.test:9000/base/v1/pdf"]


# Statuses.


@pytest.mark.parametrize(
    "case", CASES["upstream_statuses"], ids=lambda case: f"{case['upstream']}-answers-{case['answer']}"
)
async def test_each_status_maps_to_the_routes_answer_with_no_retry(case):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(case["upstream"], headers={"Location": "http://elsewhere.test/v1/pdf"})

    assert await _status(_client(handler)) == case["answer"]
    assert len(seen) == 1


@pytest.mark.parametrize("status", REDIRECT_STATUSES)
async def test_a_redirect_answers_502_and_its_target_sees_nothing(status):
    with recording_server() as target:
        answer = _StubAnswer(status=status, headers={"Location": f"{target.url}/v1/pdf"})
        with _stub_render_service(answer) as service:
            client = RenderServicePdf(settings=_settings(render_url=service.url))
            assert await _status(client) == 502
            assert service.paths == ["/v1/pdf"]
        assert target.hits() == []


@pytest.mark.parametrize("case", CASES["unusable_bodies"], ids=lambda case: case["name"])
async def test_a_200_without_a_usable_pdf_answers_502(case):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=case["body"].encode())

    assert await _status(_client(handler)) == 502


async def test_a_pdf_after_other_bytes_answers_502():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_answer(b"<html>" + FIXTURE_PDF))

    assert await _status(_client(handler)) == 502


async def test_an_unreachable_render_service_answers_502():
    client = RenderServicePdf(settings=_settings(render_url=_closed_port_url()))

    assert await _status(client) == 502


# Busy and the budget.


async def test_a_503_then_a_pdf_retries_once_and_returns_the_pdf():
    statuses = iter([503, 200])
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status = next(statuses)
        return _ok(request) if status == 200 else httpx.Response(status)

    assert await _client(handler).render(request=REQUEST) == FIXTURE_PDF
    assert len(seen) == 2


async def test_503_twice_answers_503_after_exactly_one_retry():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503)

    assert await _status(_client(handler)) == 503
    assert len(seen) == 2


async def test_503_with_less_budget_than_a_full_job_answers_503_without_a_retry():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503)

    assert await _status(_client(handler, timeout_seconds=FULL_JOB_SECONDS - 1)) == 503
    assert len(seen) == 1


async def test_a_slow_answer_past_the_budget_answers_504():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return _ok(request)

    started = time.monotonic()
    assert await _status(_client(handler, timeout_seconds=0.2)) == 504
    assert time.monotonic() - started < 2


async def test_a_body_that_trickles_past_the_budget_answers_504():
    # Two seconds of one byte every 20 ms: each read is quick, the whole body isn't.
    async def trickle() -> AsyncIterator[bytes]:
        for _ in range(100):
            await asyncio.sleep(0.02)
            yield b" "

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=trickle())

    started = time.monotonic()
    assert await _status(_client(handler, timeout_seconds=0.3)) == 504
    assert time.monotonic() - started < 2


# Size and pages.


async def test_an_answer_over_the_byte_cap_answers_413_and_stops_reading():
    sent: list[int] = []

    async def large() -> AsyncIterator[bytes]:
        for _ in range(64):
            sent.append(1024)
            yield b"x" * 1024

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=large())

    assert await _status(_client(handler, max_pdf_bytes=4096)) == 413
    assert sum(sent) <= 8 * 1024


async def test_a_pdf_at_the_page_limit_passes():
    pdf = _blank_pages(500)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_answer(pdf))

    assert await _client(handler).render(request=REQUEST) == pdf


async def test_a_pdf_over_the_page_limit_answers_413():
    pdf = _blank_pages(501)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_answer(pdf))

    assert await _status(_client(handler)) == 413


def _one_page_listed(times: int) -> bytes:
    return _pdf(CATALOG, f"<< /Type /Pages /Kids [{' '.join(['3 0 R'] * times)}] /Count 1 >>", PAGE)


def _count_that_lies(pages: int) -> bytes:
    kids = " ".join(f"{number} 0 R" for number in range(3, pages + 3))
    return _pdf(CATALOG, f"<< /Type /Pages /Kids [{kids}] /Count 1 >>", *[PAGE] * pages)


def _looping_tree() -> bytes:
    return _pdf(
        CATALOG,
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Pages /Parent 2 0 R /Kids [2 0 R] /Count 1 >>",
    )


def _long_chain(depth: int) -> bytes:
    nodes = [f"<< /Type /Pages /Kids [{number + 1} 0 R] /Count 1 >>" for number in range(2, depth + 2)]
    return _pdf(CATALOG, *nodes, PAGE)


@pytest.mark.parametrize(
    ("pdf", "answer"),
    [
        pytest.param(_one_page_listed(501), 413, id="one-page-listed-501-times"),
        pytest.param(_count_that_lies(501), 413, id="count-of-1-over-501-pages"),
        pytest.param(_looping_tree(), 502, id="page-tree-that-loops"),
        pytest.param(_long_chain(1100), 502, id="chain-past-the-node-limit"),
    ],
)
async def test_a_hostile_page_tree_costs_a_bounded_walk(pdf, answer):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_answer(pdf))

    assert await _status(_client(handler)) == answer


async def test_a_short_chain_is_counted():
    pdf = _long_chain(10)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_answer(pdf))

    assert await _client(handler).render(request=REQUEST) == pdf


async def test_an_encrypted_pdf_answers_502():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt(user_password="", owner_password="owner", algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_answer(out.getvalue()))

    assert await _status(_client(handler)) == 502


# Proxies.

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


async def test_proxy_settings_never_reroute_the_request(monkeypatch):
    with recording_server() as proxy:
        for name in PROXY_VARIABLES:
            monkeypatch.setenv(name, proxy.url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        with _stub_render_service(_StubAnswer(status=200, body=RESPONSE_JSON)) as service:
            client = RenderServicePdf(settings=_settings(render_url=service.url))

            assert await client.render(request=REQUEST) == FIXTURE_PDF
            assert service.paths == ["/v1/pdf"]
            assert proxy.hits() == []

            # The same request from a client that reads the environment goes
            # to the proxy, so the check above can fail.
            async with httpx.AsyncClient(trust_env=True) as leaky:
                await leaky.post(
                    f"{service.url}/v1/pdf",
                    content=REQUEST.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )
            assert service.paths == ["/v1/pdf"]
            hits = proxy.hits()
    assert len(hits) == 1
    assert hits[0].startswith(f"POST {service.url}/v1/pdf ")
