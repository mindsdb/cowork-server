"""Turn an exported document into a PDF through a render service.

The server never makes a PDF itself and never falls back to another
converter: ``RenderServicePdf`` posts a ``RenderPdfRequest`` to
``<render_url>/v1/pdf`` and reads back a ``RenderPdfResult``. It reads no
proxy from the environment and follows no redirect, so the document goes to
the configured service and nowhere else. Ignoring the environment also skips
``SSL_CERT_FILE`` and ``SSL_CERT_DIR``: an https render service is checked
against certifi's public certificate authorities only.

One deadline, ``ExportPdfSettings.timeout_seconds`` (60 s), covers the whole
call, the retry and reading the PDF included. The proxies in front of a
self-hosted stack wait far longer: its edge allows 3600 s to read an answer,
and its recording proxy has no read timeout on its connection to this server.
So the server's own 504 reaches the browser before any proxy gives up.

The client is async. The export route is async, and waiting on the render
service is network I/O that needs no worker thread. An ``asyncio`` timeout
also holds one deadline across connecting, the retry and a slow body, which
httpx's per-operation timeouts can't: a service that sends a byte now and
then never trips them.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import PurePosixPath
from types import FrameType
from typing import Literal

import httpx
import pypdf.filters
from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

from cowork.schemas.artifact_export import ExportPdfSettings, RenderPdfRequest, RenderPdfResult

logger = logging.getLogger(__name__)

# pypdf's lenient parser logs a warning for each fault it works around, and
# the warnings quote the PDF's own bytes: one hostile PDF can write megabytes
# of them. A fault that matters reaches this client as an exception instead.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# pypdf inflates each stream it reads up to 75 MB. Counting pages inflates only
# object streams and cross-reference streams, which a render service keeps far
# smaller, and this client is the only pypdf user in the server's process
# (agent cells run in child processes). A lower cap bounds what one hostile
# stream costs; past it, pypdf raises and the export answers 502.
PYPDF_STREAM_CAP = 8 * 1024 * 1024
pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH = PYPDF_STREAM_CAP
pypdf.filters.LZW_MAX_OUTPUT_LENGTH = PYPDF_STREAM_CAP
pypdf.filters.RUN_LENGTH_MAX_OUTPUT_LENGTH = PYPDF_STREAM_CAP
pypdf.filters.MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH = PYPDF_STREAM_CAP

# The render service stops a print job 30 s after it starts. A retry after a
# 503 is worth sending only while the budget left covers that whole job; with
# less, answering busy now beats a 504 later.
FULL_JOB_SECONDS = 30.0

# Counting a render service's pages takes pypdf well under a second. A count
# still running after this long is pypdf working through a hostile structure,
# and stopping it here also bounds the memory that structure takes.
PAGE_COUNT_SECONDS = 10.0

PDF_MAGIC = b"%PDF-"

# Source files whose PDF runs the page's own scripts.
_SCRIPTED_SOURCES = frozenset({".html", ".htm"})

_MALFORMED_PAGE_TREE = "the render service's PDF has a malformed page tree"
_NO_PDF_IN_TIME = "no PDF arrived within the time budget"

# Held by the worker thread that reads a PDF until that thread returns. See
# _read_pdf.
_READ_LOCK = threading.Lock()

RenderPdfFailureStatus = Literal[413, 422, 502, 503, 504]


class RenderPdfFailed(Exception):
    """No usable PDF came back. ``status`` is what the export route answers.

    The message says why, for the log. It never carries the render service's
    own text, which the exported page can influence, and neither does its
    cause: it is raised ``from None``, because httpx, pydantic and pypdf errors
    quote the bytes they couldn't read, and a log call with ``exc_info``
    prints the whole chain.
    """

    def __init__(self, *, status: RenderPdfFailureStatus, reason: str) -> None:
        super().__init__(reason)
        self.status = status


def pdf_request(*, source_name: str, html: str) -> RenderPdfRequest:
    """The render request for ``html``, the inlined document of ``source_name``.

    An HTML page may draw its charts with scripts, so the render service runs
    them. Markdown and text become HTML that needs none, so it runs none.
    """
    suffix = PurePosixPath(source_name).suffix.lower()
    return RenderPdfRequest(html=html, scripts=suffix in _SCRIPTED_SOURCES)


class RenderServicePdf:
    """Client for a render service's ``/v1/pdf``."""

    def __init__(
        self,
        *,
        settings: ExportPdfSettings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if settings.engine != "render" or settings.render_url is None:
            raise ValueError("PDF export needs the render engine and a render service URL")
        self._settings = settings
        self._transport = transport
        self._endpoint = f"{str(settings.render_url).rstrip('/')}/v1/pdf"

    async def render(self, *, request: RenderPdfRequest) -> bytes:
        """Return the PDF for ``request``, or raise ``RenderPdfFailed``."""
        # model_dump_json writes non-ASCII text as UTF-8; json.dumps would
        # escape it and can triple the body against the service's size cap.
        body = request.model_dump_json().encode()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settings.timeout_seconds
        try:
            async with asyncio.timeout_at(deadline):
                async with httpx.AsyncClient(
                    transport=self._transport,
                    trust_env=False,
                    follow_redirects=False,
                ) as client:
                    answer = await self._post(client=client, body=body, remaining=deadline - loop.time())
                    if answer is None:
                        if deadline - loop.time() < FULL_JOB_SECONDS:
                            raise RenderPdfFailed(status=503, reason="the render service is busy")
                        logger.info("The render service is busy; retrying the PDF export once")
                        answer = await self._post(client=client, body=body, remaining=deadline - loop.time())
                    if answer is None:
                        raise RenderPdfFailed(
                            status=503, reason="the render service is still busy after one retry"
                        )
                # Parsing the answer and counting the PDF's pages is CPU work
                # on up to tens of MiB, so it runs off the event loop, and
                # inside the deadline: a hostile PDF can make pypdf work far
                # longer than the budget.
                return await asyncio.to_thread(
                    _read_pdf,
                    answer=answer,
                    max_pages=self._settings.max_pages,
                    deadline=time.monotonic() + (deadline - loop.time()),
                )
        except (TimeoutError, httpx.TimeoutException):
            raise RenderPdfFailed(status=504, reason=_NO_PDF_IN_TIME) from None
        except httpx.RequestError as error:
            raise RenderPdfFailed(
                status=502, reason=f"the render service can't be reached ({type(error).__name__})"
            ) from None

    async def _post(self, *, client: httpx.AsyncClient, body: bytes, remaining: float) -> bytes | None:
        """Send one request. Return the 200 body, or None for a 503."""
        async with client.stream(
            "POST",
            self._endpoint,
            content=body,
            headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
            timeout=httpx.Timeout(remaining),
        ) as response:
            status = response.status_code
            if status == 200:
                return await self._read_bounded(response=response)
            if status == 503:
                return None
            if status in (413, 422):
                raise RenderPdfFailed(status=status, reason=f"the render service answered {status}")
            # Any redirect lands here too: following one would send the
            # document to an address nobody configured.
            raise RenderPdfFailed(status=502, reason=f"the render service answered {status}")

    async def _read_bounded(self, *, response: httpx.Response) -> bytes:
        """Read the body as sent, stopping as soon as it passes ``max_pdf_bytes``.

        The request asks for no content coding, and an answer that has one is
        refused before its body is read: httpx inflates every layer of a
        coding like ``gzip, gzip`` in one step, before any byte of the result
        counts against the cap.
        """
        encoding = response.headers.get("Content-Encoding")
        if encoding is not None and encoding.strip().lower() != "identity":
            raise RenderPdfFailed(status=502, reason="the render service's answer has a content coding")
        limit = self._settings.max_pdf_bytes
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_raw():
            size += len(chunk)
            if size > limit:
                raise RenderPdfFailed(status=413, reason=f"the render service's answer is over {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)


def _read_pdf(*, answer: bytes, max_pages: int, deadline: float) -> bytes:
    """Return the PDF a 200 answer carries once its pages are counted, or raise.

    It runs in a worker thread, which the render call stops waiting for at
    the deadline but can't stop: the thread runs on until its own checks stop
    it. So each call holds ``_READ_LOCK`` until it returns, waiting for it no
    later than ``deadline``, and a call the export gave up on never runs
    beside a new one. ``deadline`` is a ``time.monotonic()`` value.
    """
    if not _READ_LOCK.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise RenderPdfFailed(status=504, reason=_NO_PDF_IN_TIME)
    try:
        pdf = _decode_pdf(answer=answer)
        pages = _count_pages(pdf=pdf, max_pages=max_pages, deadline=deadline)
    finally:
        _READ_LOCK.release()
    if pages > max_pages:
        raise RenderPdfFailed(status=413, reason=f"the PDF has over {max_pages} pages")
    return pdf


def _decode_pdf(*, answer: bytes) -> bytes:
    """Return the PDF a 200 answer carries, or raise 502."""
    try:
        result = RenderPdfResult.model_validate_json(answer)
    except ValidationError:
        raise RenderPdfFailed(status=502, reason="the render service's answer isn't a PDF result") from None
    try:
        pdf = base64.b64decode(result.pdf_base64, validate=True)
    except ValueError:  # binascii.Error is a ValueError
        raise RenderPdfFailed(status=502, reason="the render service's PDF isn't valid base64") from None
    if not pdf.startswith(PDF_MAGIC):
        raise RenderPdfFailed(status=502, reason="the render service's bytes aren't a PDF")
    return pdf


class _PageCountStopped(BaseException):
    """The page count ran out of time.

    A BaseException, so that pypdf's lenient ``except Exception`` handlers
    can't swallow it and parse on.
    """


def _stop_after(*, stop_at: float) -> Callable[[FrameType, str, object], None]:
    """A profile function that raises at the first event past ``stop_at``."""

    def profile(_frame: FrameType, _event: str, _arg: object) -> None:
        if time.monotonic() > stop_at:
            raise _PageCountStopped

    return profile


def _count_pages(*, pdf: bytes, max_pages: int, deadline: float) -> int:
    """Count the PDF's pages with pypdf within the time left, or raise.

    pypdf parses a whole object in one call, and one hostile object can hold
    millions of entries, or an object stream thousands of objects. Nothing
    stops that call from outside, so a profile hook on this thread stops it
    from inside: it raises at the first call pypdf makes past ``deadline`` (a
    ``time.monotonic()`` value, answered 504) or past ``PAGE_COUNT_SECONDS``
    of counting (answered 502). The hook slows pypdf down several times, which
    a count of well under a second can afford.
    """
    stop_at = min(deadline, time.monotonic() + PAGE_COUNT_SECONDS)
    previous = sys.getprofile()
    sys.setprofile(_stop_after(stop_at=stop_at))
    # The hook also sees the call that removes it, so a stop can come from the
    # finally after the count is done. The outer try catches that stop too.
    # Python turns off a hook that raises, and a stop raised at that call
    # skips the restore, so the handler restores the previous hook.
    try:
        try:
            return _walk_page_tree(pdf=pdf, max_pages=max_pages, stop_at=stop_at)
        finally:
            sys.setprofile(previous)
    except _PageCountStopped:
        sys.setprofile(previous)
        if stop_at >= deadline:
            raise RenderPdfFailed(status=504, reason=_NO_PDF_IN_TIME) from None
        raise RenderPdfFailed(
            status=502, reason=f"counting the render service's PDF pages took over {PAGE_COUNT_SECONDS} s"
        ) from None


def _walk_page_tree(*, pdf: bytes, max_pages: int, stop_at: float) -> int:
    """Count the PDF's pages with pypdf, stopping once past ``max_pages``.

    pypdf's own count lists every page in the tree, recursing however large
    or deep the tree is. This walk is a loop instead: it visits at most
    ``2 * max_pages + 2`` page-tree nodes and stops at the first page past
    ``max_pages``. A tree of ``max_pages`` pages where each node has two or
    more children needs fewer nodes than that, so only a hostile tree (a loop,
    a long chain) reaches the node limit. A node whose /Type is neither
    /Pages nor /Page is refused, where pypdf would skip it and its pages. A
    render service never encrypts its output, so an encrypted PDF is refused
    too. Each node also checks ``stop_at``, in case code the profile hook ran
    in swallowed its exception, which turns the hook off.
    """
    node_limit = 2 * max_pages + 2
    try:
        reader = PdfReader(io.BytesIO(pdf))
        if reader.is_encrypted:
            raise RenderPdfFailed(status=502, reason="the render service's PDF is encrypted")
        catalog = reader.trailer["/Root"]
        if not isinstance(catalog, DictionaryObject):
            raise RenderPdfFailed(status=502, reason=_MALFORMED_PAGE_TREE)
        pending: list[object] = [catalog.raw_get("/Pages")]
        visited = 0
        pages = 0
        while pending:
            if time.monotonic() > stop_at:
                raise _PageCountStopped
            entry = pending.pop()
            visited += 1
            if visited > node_limit:
                raise RenderPdfFailed(status=502, reason=_MALFORMED_PAGE_TREE)
            node = entry.get_object() if isinstance(entry, IndirectObject) else entry
            if not isinstance(node, DictionaryObject):
                raise RenderPdfFailed(status=502, reason=_MALFORMED_PAGE_TREE)
            # Indexing resolves a /Type stored as a reference; .get() wouldn't.
            kind = node["/Type"] if "/Type" in node else None
            # pypdf's rule: a node without /Type is a page unless it has /Kids.
            if kind == "/Pages" or (kind is None and "/Kids" in node):
                kids = node["/Kids"]
                if not isinstance(kids, ArrayObject):
                    raise RenderPdfFailed(status=502, reason=_MALFORMED_PAGE_TREE)
                pending.extend(kids)
            elif kind == "/Page" or kind is None:
                pages += 1
                if pages > max_pages:
                    return pages
            else:
                raise RenderPdfFailed(status=502, reason=_MALFORMED_PAGE_TREE)
        return pages
    except RenderPdfFailed:
        raise
    except Exception as error:
        # A hostile PDF can make pypdf raise nearly anything: its own errors,
        # KeyError, TypeError, RecursionError. Each one means the bytes aren't
        # a PDF this server accepts.
        raise RenderPdfFailed(
            status=502, reason=f"pypdf can't read the render service's PDF ({type(error).__name__})"
        ) from None
