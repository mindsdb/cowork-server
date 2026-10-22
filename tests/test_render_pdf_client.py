"""The render-service PDF client: its wire, its statuses, its budget and its limits."""
from __future__ import annotations

import asyncio
import base64
import gzip
import io
import json
import logging
import math
import socket
import sys
import threading
import time
import traceback
import tracemalloc
import warnings
import zlib
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from types import FrameType, SimpleNamespace
from typing import Literal

import httpx
import pytest
import yaml
from pypdf import PdfWriter
from pypdf import _configuration as pypdf_configuration
from pypdf.generic import create_string_object

from pydantic import BaseModel

from cowork.schemas.artifact_export import ExportPdfSettings, RenderPdfRequest, RenderPdfResult
from cowork.services import render_pdf_client
from cowork.services.render_pdf_client import FULL_JOB_SECONDS, RenderPdfFailed, RenderServicePdf, pdf_request
from _no_fetch import recording_server

HostileShape = Literal[
    "one_page_listed",
    "count_that_lies",
    "looping_tree",
    "long_chain",
    "indirect_types",
    "unknown_type_with_kids",
]

StreamFilter = Literal["/FlateDecode", "/ASCII85Decode"]


class UpstreamStatusCase(BaseModel, frozen=True, extra="forbid"):
    upstream: int
    answer: int


class UnusableBody(BaseModel, frozen=True, extra="forbid"):
    name: str
    body: str


class HostilePageTree(BaseModel, frozen=True, extra="forbid"):
    name: str
    shape: HostileShape
    size: int
    answer: int


class NoClientSettings(BaseModel, frozen=True, extra="forbid"):
    name: str
    settings: ExportPdfSettings


class ScriptsBySource(BaseModel, frozen=True, extra="forbid"):
    source_name: str
    scripts: bool


class RenderPdfClientCases(BaseModel, frozen=True):
    """The tables of render_pdf_cases.yaml this file reads."""

    upstream_statuses: list[UpstreamStatusCase]
    unusable_bodies: list[UnusableBody]
    hostile_page_trees: list[HostilePageTree]
    no_client_settings: list[NoClientSettings]
    scripts_by_source: list[ScriptsBySource]
    proxy_variables: list[str]


FIXTURES = Path(__file__).parent / "fixtures"
CASES = RenderPdfClientCases.model_validate(
    yaml.safe_load((FIXTURES / "render_pdf_cases.yaml").read_text(encoding="utf-8"))
)
REQUEST_JSON = (FIXTURES / "render_pdf" / "request.json").read_bytes()
RESPONSE_JSON = (FIXTURES / "render_pdf" / "response.json").read_bytes()
REQUEST = RenderPdfRequest.model_validate_json(REQUEST_JSON)
FIXTURE_PDF = base64.b64decode(RenderPdfResult.model_validate_json(RESPONSE_JSON).pdf_base64, validate=True)

REDIRECT_STATUSES = [case.upstream for case in CASES.upstream_statuses if 300 <= case.upstream < 400]

MIB = 1024 * 1024

Handler = Callable[[httpx.Request], httpx.Response]


def _settings(
    *,
    render_url: str = "http://render.test:9000",
    timeout_seconds: float | None = None,
    max_pdf_bytes: int | None = None,
) -> ExportPdfSettings:
    """Settings for ``render_url``. A limit left as None keeps the schema's default."""
    limits = {"timeout_seconds": timeout_seconds, "max_pdf_bytes": max_pdf_bytes}
    return ExportPdfSettings(
        engine="render",
        render_url=render_url,
        **{name: value for name, value in limits.items() if value is not None},
    )


def _client(
    *,
    handler: Handler,
    render_url: str = "http://render.test:9000",
    timeout_seconds: float | None = None,
    max_pdf_bytes: int | None = None,
) -> RenderServicePdf:
    settings = _settings(render_url=render_url, timeout_seconds=timeout_seconds, max_pdf_bytes=max_pdf_bytes)
    return RenderServicePdf(settings=settings, transport=httpx.MockTransport(handler))


def _streamed_200(*, body: bytes, headers: dict[str, str] | None = None) -> httpx.Response:
    """A 200 whose body streams the way a real transport's does.

    A Response built from ``content=`` bytes arrives already read, and the
    client reads the body as sent, which a read body no longer offers.
    """
    return httpx.Response(200, stream=httpx.ByteStream(body), headers=headers)


def _answer(*, pdf: bytes) -> bytes:
    return RenderPdfResult(pdf_base64=base64.b64encode(pdf).decode(), refused=[]).model_dump_json().encode()


def _ok(_request: httpx.Request) -> httpx.Response:
    return _streamed_200(body=RESPONSE_JSON, headers={"Content-Type": "application/json"})


def _answering(*, pdf: bytes) -> Handler:
    """A handler that answers every request with ``pdf``."""
    answer = _answer(pdf=pdf)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _streamed_200(body=answer)

    return handler


async def _status(*, client: RenderServicePdf) -> int:
    with pytest.raises(RenderPdfFailed) as failure:
        await client.render(request=REQUEST)
    return failure.value.status


async def _render_once_the_worker_is_free() -> None:
    """Render the fixture with a full budget.

    A test that leaves a page count running in its worker thread ends with
    this, so the next test doesn't wait for that thread.
    """
    assert await _client(handler=_ok).render(request=REQUEST) == FIXTURE_PDF


def _blank_pages(*, count: int) -> bytes:
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


STREAM_ENCODERS: dict[StreamFilter, Callable[[bytes], bytes]] = {
    "/FlateDecode": lambda data: zlib.compress(data, 9),
    "/ASCII85Decode": lambda data: base64.a85encode(data, adobe=True),
}


def _packed_pdf(*, kids: int, body: str, filters: tuple[StreamFilter, ...] = ("/FlateDecode",)) -> bytes:
    """A PDF whose page-tree root lists ``kids`` objects packed in one object stream.

    Object 1 is the catalog, 2 the page-tree root, 3 the object stream,
    encoded with ``filters`` in the order pypdf decodes them, and 4 the
    cross-reference stream. Objects 5 and up sit in the object stream, all at
    the same offset, so each one is ``body`` and pypdf parses ``body`` once
    for each of them.
    """
    packed = list(range(5, 5 + kids))
    index = " ".join(f"{number} 0" for number in packed) + " "
    encoded = index.encode() + body.encode()
    for name in reversed(filters):
        encoded = STREAM_ENCODERS[name](encoded)
    root_kids = " ".join(f"{number} 0 R" for number in packed)
    out = bytearray(b"%PDF-1.7\n")
    offsets = [len(out)]
    out += f"1 0 obj\n{CATALOG}\nendobj\n".encode()
    offsets.append(len(out))
    out += f"2 0 obj\n<< /Type /Pages /Kids [{root_kids}] /Count {kids} >>\nendobj\n".encode()
    offsets.append(len(out))
    out += (
        f"3 0 obj\n<< /Type /ObjStm /N {kids} /First {len(index)} /Filter [{' '.join(filters)}] "
        f"/Length {len(encoded)} >>\nstream\n"
    ).encode()
    out += encoded + b"\nendstream\nendobj\n"
    offsets.append(len(out))
    # Each row: type 1 (offset in the file) or type 2 (object stream 3, index in it).
    rows = bytearray(bytes(6))
    for offset in offsets:
        rows += b"\x01" + offset.to_bytes(4, "big") + b"\x00"
    for position in range(kids):
        rows += b"\x02" + (3).to_bytes(4, "big") + position.to_bytes(1, "big")
    out += (
        f"4 0 obj\n<< /Type /XRef /Size {5 + kids} /W [1 4 1] /Root 1 0 R "
        f"/Length {len(rows)} >>\nstream\n"
    ).encode()
    out += bytes(rows) + b"\nendstream\nendobj\n"
    out += f"startxref\n{offsets[3]}\n%%EOF\n".encode()
    return bytes(out)


def _zeros(*, count: int) -> str:
    return "0 " * count


def _one_long_node() -> bytes:
    """A page-tree node in an object stream with three million entries.

    pypdf parses them in one call: seconds of work on a fast machine, and
    about 6 MB inflated, under the inflate cap.
    """
    return _packed_pdf(kids=1, body=f"<< /Type /Pages /Kids [{_zeros(count=3_000_000)}] >>")


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
def _stub_render_service(*, answer: _StubAnswer) -> Iterator[_StubRenderService]:
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


# Settings.


@pytest.mark.parametrize("case", CASES.no_client_settings, ids=lambda case: case.name)
def test_no_client_without_the_render_engine_and_a_url(case):
    with pytest.raises(ValueError):
        RenderServicePdf(settings=case.settings, transport=httpx.MockTransport(_ok))


# The wire.


@pytest.mark.parametrize("case", CASES.scripts_by_source, ids=lambda case: case.source_name)
def test_scripts_run_only_for_an_html_source(case):
    request = pdf_request(source_name=case.source_name, html="<p>Report</p>")

    assert request == RenderPdfRequest(html="<p>Report</p>", scripts=case.scripts)


async def test_render_posts_the_request_to_v1_pdf_and_returns_the_pdf():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(request)

    pdf = await _client(handler=handler).render(request=REQUEST)

    assert pdf == FIXTURE_PDF
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == "http://render.test:9000/v1/pdf"
    assert seen[0].headers["Content-Type"] == "application/json"
    assert seen[0].headers["Accept-Encoding"] == "identity"


async def test_the_request_body_is_the_fixture_wire():
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return _ok(request)

    await _client(handler=handler).render(request=REQUEST)

    assert RenderPdfRequest.model_validate_json(bodies[0]) == REQUEST
    assert json.loads(bodies[0]) == json.loads(REQUEST_JSON)


async def test_non_ascii_text_goes_out_as_utf8_not_escapes():
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return _ok(request)

    request = RenderPdfRequest(html="<p>Résumé 数据</p>", scripts=False)
    await _client(handler=handler).render(request=request)

    assert "Résumé 数据".encode() in bodies[0]
    assert b"\\u" not in bodies[0]


async def test_the_render_url_keeps_its_path_prefix():
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return _ok(request)

    await _client(handler=handler, render_url="http://render.test:9000/base/").render(request=REQUEST)

    assert urls == ["http://render.test:9000/base/v1/pdf"]


# Statuses.


@pytest.mark.parametrize(
    "case", CASES.upstream_statuses, ids=lambda case: f"{case.upstream}-answers-{case.answer}"
)
async def test_each_status_maps_to_the_routes_answer_with_no_retry(case):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(case.upstream, headers={"Location": "http://elsewhere.test/v1/pdf"})

    assert await _status(client=_client(handler=handler)) == case.answer
    assert len(seen) == 1


@pytest.mark.parametrize("status", REDIRECT_STATUSES)
async def test_a_redirect_answers_502_and_its_target_sees_nothing(status):
    with recording_server() as target:
        answer = _StubAnswer(status=status, headers={"Location": f"{target.url}/v1/pdf"})
        with _stub_render_service(answer=answer) as service:
            client = RenderServicePdf(settings=_settings(render_url=service.url))
            assert await _status(client=client) == 502
            assert service.paths == ["/v1/pdf"]
        assert target.hits() == []


@pytest.mark.parametrize("case", CASES.unusable_bodies, ids=lambda case: case.name)
async def test_a_200_without_a_usable_pdf_answers_502(case):
    def handler(_request: httpx.Request) -> httpx.Response:
        return _streamed_200(body=case.body.encode())

    assert await _status(client=_client(handler=handler)) == 502


async def test_a_pdf_after_other_bytes_answers_502():
    assert await _status(client=_client(handler=_answering(pdf=b"<html>" + FIXTURE_PDF))) == 502


async def test_an_unreachable_render_service_answers_502():
    client = RenderServicePdf(settings=_settings(render_url=_closed_port_url()))

    assert await _status(client=client) == 502


async def test_an_unreachable_render_service_leaves_its_url_out_of_the_failure():
    url = _closed_port_url().replace("http://", "http://user:secret@")
    client = RenderServicePdf(settings=_settings(render_url=url))

    with pytest.raises(RenderPdfFailed) as failure:
        await client.render(request=REQUEST)

    assert failure.value.status == 502
    assert "secret" not in str(failure.value)
    assert "127.0.0.1" not in str(failure.value)


async def test_a_compressed_answer_answers_502_before_it_is_inflated():
    # Two gzip layers: a few hundred bytes on the wire, 32 MiB once inflated.
    body = gzip.compress(gzip.compress(bytes(32 * MIB)))
    answer = _StubAnswer(
        status=200,
        body=body,
        headers={"Content-Type": "application/json", "Content-Encoding": "gzip, gzip"},
    )
    with _stub_render_service(answer=answer) as service:
        client = RenderServicePdf(settings=_settings(render_url=service.url, max_pdf_bytes=4 * MIB))
        tracemalloc.start()
        try:
            status = await _status(client=client)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    assert status == 502
    assert peak < 2 * 4 * MIB


async def test_an_answer_that_names_a_content_coding_answers_502():
    def handler(_request: httpx.Request) -> httpx.Response:
        return _streamed_200(body=RESPONSE_JSON, headers={"Content-Encoding": "gzip"})

    assert await _status(client=_client(handler=handler)) == 502


async def test_an_identity_content_encoding_is_read_as_sent():
    def handler(_request: httpx.Request) -> httpx.Response:
        return _streamed_200(body=RESPONSE_JSON, headers={"Content-Encoding": "identity"})

    assert await _client(handler=handler).render(request=REQUEST) == FIXTURE_PDF


# What a failure carries.

SERVICE_TEXT = "PageChosenText"


def _protocol_error(request: httpx.Request) -> httpx.Response:
    # h11 quotes the bytes of a status or header line it can't parse.
    raise httpx.RemoteProtocolError(f"illegal status line: b'{SERVICE_TEXT}'", request=request)


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(_protocol_error, id="protocol-error"),
        pytest.param(
            lambda _request: _streamed_200(body=f'{{"pdf_base64": 7, "refused": "{SERVICE_TEXT}"}}'.encode()),
            id="answer-that-isnt-a-result",
        ),
        # pypdf quotes the bytes around an object it can't parse.
        pytest.param(
            _answering(pdf=_pdf(CATALOG, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", SERVICE_TEXT)),
            id="pdf-pypdf-cant-read",
        ),
    ],
)
async def test_a_failure_never_carries_the_render_services_text(handler):
    with pytest.raises(RenderPdfFailed) as failure:
        await _client(handler=handler).render(request=REQUEST)

    # A log call with exc_info prints the whole chain, so it carries no cause.
    assert failure.value.status == 502
    assert failure.value.__cause__ is None
    assert SERVICE_TEXT not in "".join(traceback.format_exception(failure.value))


# Busy and the budget.


async def test_a_503_then_a_pdf_retries_once_and_returns_the_pdf():
    statuses = iter([503, 200])
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status = next(statuses)
        return _ok(request) if status == 200 else httpx.Response(status)

    assert await _client(handler=handler).render(request=REQUEST) == FIXTURE_PDF
    assert len(seen) == 2


async def test_503_twice_answers_503_after_exactly_one_retry():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503)

    assert await _status(client=_client(handler=handler)) == 503
    assert len(seen) == 2


async def test_503_with_less_budget_than_a_full_job_answers_503_without_a_retry():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503)

    assert await _status(client=_client(handler=handler, timeout_seconds=FULL_JOB_SECONDS - 1)) == 503
    assert len(seen) == 1


async def test_the_retry_after_a_503_shares_the_first_attempts_budget(monkeypatch):
    # A 1 s budget and a 0.2 s job: the 503 after 0.4 s leaves room for a
    # retry, and the retry's 0.8 s runs past what is left of the one budget,
    # though not past a fresh one.
    monkeypatch.setattr(render_pdf_client, "FULL_JOB_SECONDS", 0.2)
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            await asyncio.sleep(0.4)
            return httpx.Response(503)
        await asyncio.sleep(0.8)
        return _ok(request)

    assert await _status(client=_client(handler=handler, timeout_seconds=1.0)) == 504
    assert len(seen) == 2


async def test_a_slow_answer_past_the_budget_answers_504():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return _ok(request)

    started = time.monotonic()
    assert await _status(client=_client(handler=handler, timeout_seconds=0.2)) == 504
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
    assert await _status(client=_client(handler=handler, timeout_seconds=0.3)) == 504
    assert time.monotonic() - started < 2


async def test_an_answer_whose_parsing_runs_past_the_budget_answers_504():
    # A million refusal records: about a second of parsing, ten times the
    # budget, all of it after the body arrived. The parser holds the GIL, so
    # the 504 can come only once it ends; before, the parse ran outside the
    # budget and returned the PDF.
    refused = ",".join(['{"url":"","reason":""}'] * 1_000_000)
    answer = f'{{"pdf_base64":"{base64.b64encode(FIXTURE_PDF).decode()}","refused":[{refused}]}}'.encode()

    def handler(_request: httpx.Request) -> httpx.Response:
        return _streamed_200(body=answer)

    assert await _status(client=_client(handler=handler, timeout_seconds=0.1)) == 504
    await _render_once_the_worker_is_free()


async def test_a_page_count_past_the_budget_answers_504():
    pdf = _one_long_node()

    started = time.monotonic()
    assert await _status(client=_client(handler=_answering(pdf=pdf), timeout_seconds=0.3)) == 504
    assert time.monotonic() - started < 3
    await _render_once_the_worker_is_free()


async def test_a_page_count_left_running_past_the_budget_stops_with_it():
    # Sixty objects in one object stream, each a 100,000-entry parse. pypdf
    # parses every object in the stream in one call: seconds of work.
    pdf = _packed_pdf(kids=60, body=f"<< /Type /Page /Junk [{_zeros(count=100_000)}] >>")

    assert await _status(client=_client(handler=_answering(pdf=pdf), timeout_seconds=0.3)) == 504
    # The next export waits for that count's thread, which stopped inside
    # pypdf at the deadline instead of finishing the stream.
    started = time.monotonic()
    await _render_once_the_worker_is_free()
    assert time.monotonic() - started < 3


async def test_a_page_count_over_its_own_time_answers_502(monkeypatch):
    monkeypatch.setattr(render_pdf_client, "PAGE_COUNT_SECONDS", 0.3)
    pdf = _one_long_node()

    started = time.monotonic()
    assert await _status(client=_client(handler=_answering(pdf=pdf))) == 502
    assert time.monotonic() - started < 3


def _stop_as_the_hook_comes_off(*, stop_at: float) -> Callable[[FrameType, str, object], None]:
    """A profile hook that raises only at the call that removes it.

    It stands in for the clock passing the stop time just after the count
    returned: the next event the hook sees is that call.
    """

    def profile(_frame: FrameType, event: str, arg: object) -> None:
        if event == "c_call" and getattr(arg, "__name__", "") == "setprofile":
            raise render_pdf_client._PageCountStopped

    return profile


async def test_a_stop_as_the_page_count_hook_comes_off_answers_502(monkeypatch):
    monkeypatch.setattr(render_pdf_client, "_stop_after", _stop_as_the_hook_comes_off)

    assert await _status(client=_client(handler=_ok)) == 502


def test_a_stop_as_the_page_count_hook_comes_off_restores_the_previous_profile(monkeypatch):
    monkeypatch.setattr(render_pdf_client, "_stop_after", _stop_as_the_hook_comes_off)

    def previous(_frame: FrameType, _event: str, _arg: object) -> None:
        pass

    outer = sys.getprofile()
    sys.setprofile(previous)
    try:
        with pytest.raises(RenderPdfFailed):
            render_pdf_client._count_pages(pdf=FIXTURE_PDF, max_pages=500, deadline=time.monotonic() + 60)
        restored = sys.getprofile()
    finally:
        sys.setprofile(outer)
    assert restored is previous


class _ClockPastTheStopInsideLogging:
    """A clock that reads past any stop while this thread holds logging's module lock.

    Python 3.12's logging takes that lock and only then enters the ``try``
    that releases it. A stop raised in between leaves the lock held, and every
    later logging call in the process waits for it forever.
    """

    @staticmethod
    def monotonic() -> float:
        return math.inf if logging._lock._is_owned() else 0.0


def test_a_page_count_never_stops_while_logging_holds_its_lock(monkeypatch):
    # pypdf warns about the repeated key, and each pypdf warning takes
    # logging's lock in logging.getLogger.
    pdf = _pdf("<< /Type /Catalog /Pages 2 0 R /A 1 /A 1 >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", PAGE)
    monkeypatch.setattr(render_pdf_client, "time", _ClockPastTheStopInsideLogging)
    stopped: list[int] = []
    pages = 0
    left_held = 0

    try:
        pages = render_pdf_client._count_pages(pdf=pdf, max_pages=500, deadline=60.0)
    except RenderPdfFailed as failure:
        stopped.append(failure.status)
    finally:
        # Free a lock the count left held, so the rest of the run can log.
        while logging._lock._is_owned():
            logging._lock.release()
            left_held += 1

    assert left_held == 0
    assert (pages, stopped) == (1, [])


def test_the_page_count_hook_stops_only_inside_pypdf():
    previous = sys.getprofile()
    try:
        sys.setprofile(render_pdf_client._stop_after(stop_at=-math.inf))
        # Past the stop, code outside pypdf still runs to its end...
        logging.getLogger(__name__).isEnabledFor(logging.DEBUG)
        # ...and pypdf stops as it next starts a function or calls a builtin.
        with pytest.raises(render_pdf_client._PageCountStopped):
            create_string_object("x")
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("event", ["return", "c_return", "c_exception"])
def test_the_page_count_hook_never_stops_pypdf_once_its_code_ran(event):
    # These events come after a function or builtin ran, which may have taken
    # a lock that the code right after it releases.
    pypdf_frame = SimpleNamespace(f_globals={"__name__": "pypdf._reader"})

    assert render_pdf_client._stop_after(stop_at=-math.inf)(pypdf_frame, event, None) is None


def test_a_page_count_stops_inside_one_long_pypdf_loop():
    # pypdf reads a literal string byte by byte in one function that starts no
    # other pypdf function: seconds of work for a 16 MB string.
    catalog = "<< /Type /Catalog /Pages 2 0 R /X (" + "x" * 16_000_000 + ") >>"
    pdf = _pdf(catalog, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", PAGE)

    started = time.monotonic()
    with pytest.raises(RenderPdfFailed) as failure:
        render_pdf_client._count_pages(pdf=pdf, max_pages=500, deadline=started + 0.3)
    assert failure.value.status == 504
    assert time.monotonic() - started < 3


async def test_a_page_count_left_running_holds_off_the_next_export(monkeypatch):
    release = threading.Event()
    counted: list[int] = []
    real_count_pages = render_pdf_client._count_pages

    def count_pages_on_release(*, pdf: bytes, max_pages: int, deadline: float) -> int:
        counted.append(len(pdf))
        release.wait(timeout=10)
        return real_count_pages(pdf=pdf, max_pages=max_pages, deadline=deadline)

    monkeypatch.setattr(render_pdf_client, "_count_pages", count_pages_on_release)

    assert await _status(client=_client(handler=_ok, timeout_seconds=0.2)) == 504
    # The first count still runs in its thread. The second export waits for it
    # and runs out of budget without starting a count of its own.
    assert await _status(client=_client(handler=_ok, timeout_seconds=0.2)) == 504
    # Its thread stops waiting at the same deadline; give it time to.
    await asyncio.sleep(0.2)
    assert len(counted) == 1

    release.set()
    await _render_once_the_worker_is_free()
    assert len(counted) == 2


# Size and pages.


async def test_an_answer_over_the_byte_cap_answers_413_and_stops_reading():
    sent: list[int] = []

    async def large() -> AsyncIterator[bytes]:
        for _ in range(64):
            sent.append(1024)
            yield b"x" * 1024

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=large())

    assert await _status(client=_client(handler=handler, max_pdf_bytes=4096)) == 413
    assert sum(sent) <= 8 * 1024


async def test_a_pdf_at_the_page_limit_passes():
    pdf = _blank_pages(count=500)

    assert await _client(handler=_answering(pdf=pdf)).render(request=REQUEST) == pdf


async def test_a_pdf_over_the_page_limit_answers_413():
    assert await _status(client=_client(handler=_answering(pdf=_blank_pages(count=501)))) == 413


def _one_page_listed(*, times: int) -> bytes:
    return _pdf(CATALOG, f"<< /Type /Pages /Kids [{' '.join(['3 0 R'] * times)}] /Count 1 >>", PAGE)


def _count_that_lies(*, pages: int) -> bytes:
    kids = " ".join(f"{number} 0 R" for number in range(3, pages + 3))
    return _pdf(CATALOG, f"<< /Type /Pages /Kids [{kids}] /Count 1 >>", *[PAGE] * pages)


def _looping_tree(*, nodes: int) -> bytes:
    """``nodes`` page-tree nodes, each the only kid of the one before, the last one's kid the first."""
    kids = [*range(3, nodes + 2), 2]
    return _pdf(CATALOG, *[f"<< /Type /Pages /Kids [{kid} 0 R] /Count 1 >>" for kid in kids])


def _long_chain(*, depth: int) -> bytes:
    nodes = [f"<< /Type /Pages /Kids [{number + 1} 0 R] /Count 1 >>" for number in range(2, depth + 2)]
    return _pdf(CATALOG, *nodes, PAGE)


def _indirect_types(*, pages: int) -> bytes:
    """Every page-tree node's /Type is a reference: object 3 is /Pages, object 4 is /Page."""
    kids = " ".join(f"{number} 0 R" for number in range(5, pages + 5))
    page = "<< /Type 4 0 R /Parent 2 0 R /MediaBox [0 0 72 72] >>"
    return _pdf(CATALOG, f"<< /Type 3 0 R /Kids [{kids}] /Count {pages} >>", "/Pages", "/Page", *[page] * pages)


def _unknown_type_with_kids(*, pages: int) -> bytes:
    kids = " ".join(f"{number} 0 R" for number in range(3, pages + 3))
    return _pdf(CATALOG, f"<< /Type /Foo /Kids [{kids}] /Count {pages} >>", *[PAGE] * pages)


HOSTILE_PAGE_TREES: dict[HostileShape, Callable[[int], bytes]] = {
    "one_page_listed": lambda size: _one_page_listed(times=size),
    "count_that_lies": lambda size: _count_that_lies(pages=size),
    "looping_tree": lambda size: _looping_tree(nodes=size),
    "long_chain": lambda size: _long_chain(depth=size),
    "indirect_types": lambda size: _indirect_types(pages=size),
    "unknown_type_with_kids": lambda size: _unknown_type_with_kids(pages=size),
}


@pytest.mark.parametrize("case", CASES.hostile_page_trees, ids=lambda case: case.name)
async def test_a_hostile_page_tree_costs_a_bounded_walk(case):
    pdf = HOSTILE_PAGE_TREES[case.shape](case.size)

    assert await _status(client=_client(handler=_answering(pdf=pdf))) == case.answer


async def test_a_short_chain_is_counted():
    pdf = _long_chain(depth=10)

    assert await _client(handler=_answering(pdf=pdf)).render(request=REQUEST) == pdf


async def test_indirect_types_under_the_page_limit_are_counted():
    pdf = _indirect_types(pages=3)

    assert await _client(handler=_answering(pdf=pdf)).render(request=REQUEST) == pdf


async def test_an_object_stream_past_the_decode_cap_answers_502_without_parsing_it():
    # Five million entries, about 10 MiB once inflated: seconds of parsing if pypdf got that far.
    pdf = _packed_pdf(kids=1, body=f"<< /Type /Pages /Kids [{_zeros(count=5_000_000)}] >>")

    started = time.monotonic()
    assert await _status(client=_client(handler=_answering(pdf=pdf))) == 502
    assert time.monotonic() - started < 2


def test_an_object_stream_past_the_inflate_cap_answers_502_in_a_new_thread():
    # The page tree is fine: only the padding after the page takes the stream
    # past the cap, and pypdf's own cap of 75 MB would read the one page and
    # return the PDF. A new thread starts without any limit a context variable
    # holds in another thread.
    pdf = _packed_pdf(kids=1, body=PAGE + " " * (render_pdf_client.PYPDF_STREAM_CAP + MIB))
    failures: list[RenderPdfFailed] = []

    def count() -> None:
        try:
            render_pdf_client._read_pdf(answer=_answer(pdf=pdf), max_pages=500, deadline=time.monotonic() + 60)
        except RenderPdfFailed as failure:
            failures.append(failure)

    thread = threading.Thread(target=count)
    thread.start()
    thread.join(timeout=30)

    assert [(failure.status, "LimitReachedError" in str(failure)) for failure in failures] == [(502, True)]


@pytest.mark.parametrize(
    "filters",
    [("/ASCII85Decode",), ("/FlateDecode", "/ASCII85Decode")],
    ids=["ascii85", "ascii85_inside_flate"],
)
def test_an_ascii85_stream_that_could_decode_past_the_cap_answers_502_without_decoding_it(filters):
    # The page tree is fine: only the zeros after the page take the stream
    # past the cap. ASCII85 writes four zero bytes as one "z", and pypdf hands
    # the stream to the standard library's decoder, which no pypdf cap bounds:
    # pypdf would decode past the cap, read the one page and return the PDF.
    # Inside a FlateDecode, the PDF itself is a few KB.
    pdf = _packed_pdf(kids=1, body=PAGE + "\0" * (render_pdf_client.PYPDF_STREAM_CAP + MIB), filters=filters)

    started = time.monotonic()
    with pytest.raises(RenderPdfFailed) as failure:
        render_pdf_client._read_pdf(answer=_answer(pdf=pdf), max_pages=500, deadline=started + 60)
    assert (failure.value.status, "ASCII85" in str(failure.value)) == (502, True)
    assert time.monotonic() - started < 2


async def test_an_ascii85_stream_under_the_cap_is_counted():
    pdf = _packed_pdf(kids=1, body=PAGE, filters=("/ASCII85Decode",))

    assert await _client(handler=_answering(pdf=pdf)).render(request=REQUEST) == pdf


def test_counting_pages_sets_no_deprecated_pypdf_limit(monkeypatch):
    # pypdf warns about each deprecated limit only once per process.
    monkeypatch.setattr(pypdf_configuration, "WARNED_LEGACY_OVERWRITES", set())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render_pdf_client._read_pdf(
            answer=_answer(pdf=FIXTURE_PDF), max_pages=500, deadline=time.monotonic() + 60
        )

    assert [str(warning.message) for warning in caught if issubclass(warning.category, DeprecationWarning)] == []


async def test_an_encrypted_pdf_answers_502():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt(user_password="", owner_password="owner", algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)

    assert await _status(client=_client(handler=_answering(pdf=out.getvalue()))) == 502


async def test_pypdf_warnings_about_the_pdf_stay_out_of_the_log(caplog):
    # pypdf reads past the token it can't parse and warns, quoting the PDF's
    # bytes around it.
    pdf = _pdf(CATALOG, "<< /Type /Pages /Kids [3 0 R] /X @@@ /Count 1 >>", PAGE)

    assert await _client(handler=_answering(pdf=pdf)).render(request=REQUEST) == pdf
    assert [
        record for record in caplog.records if record.name.startswith("pypdf") and record.levelno >= logging.WARNING
    ] == []


# Proxies and certificates.


async def test_certificate_variables_are_ignored(monkeypatch, tmp_path):
    # A client that reads the environment loads SSL_CERT_FILE as it starts,
    # and this one fails to load, so reading it would stop the export.
    bundle = tmp_path / "bundle.pem"
    bundle.write_text("not a certificate\n", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    with _stub_render_service(answer=_StubAnswer(status=200, body=RESPONSE_JSON)) as service:
        client = RenderServicePdf(settings=_settings(render_url=service.url))

        assert await client.render(request=REQUEST) == FIXTURE_PDF


async def test_proxy_settings_never_reroute_the_request(monkeypatch):
    with recording_server() as proxy:
        for name in CASES.proxy_variables:
            monkeypatch.setenv(name, proxy.url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        with _stub_render_service(answer=_StubAnswer(status=200, body=RESPONSE_JSON)) as service:
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
