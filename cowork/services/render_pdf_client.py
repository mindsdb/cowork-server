"""Turn an exported document into a PDF through a render service.

The server never makes a PDF itself and never falls back to another
converter: ``RenderServicePdf`` posts a ``RenderPdfRequest`` to
``<render_url>/v1/pdf`` and reads back a ``RenderPdfResult``. It reads no
proxy from the environment and follows no redirect, so the document goes to
the configured service and nowhere else.

One deadline, ``ExportPdfSettings.timeout_seconds`` (60 s), covers the whole
call, the retry included. The proxies in front of a self-hosted stack wait far
longer: its edge allows 3600 s to read an answer, and its recording proxy has
no read timeout on its connection to this server. So the server's own 504
reaches the browser before any proxy gives up.

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
from typing import Literal

import httpx
from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

from cowork.schemas.artifact_export import ExportPdfSettings, RenderPdfRequest, RenderPdfResult

logger = logging.getLogger(__name__)

# The render service stops a print job 30 s after it starts. A retry after a
# 503 is worth sending only while the budget left covers that whole job; with
# less, answering busy now beats a 504 later.
FULL_JOB_SECONDS = 30.0

PDF_MAGIC = b"%PDF-"

_MALFORMED_PAGE_TREE = "the render service's PDF has a malformed page tree"

RenderPdfFailureStatus = Literal[413, 422, 502, 503, 504]


class RenderPdfFailed(Exception):
    """No usable PDF came back. ``status`` is what the export route answers.

    The message says why, for the log. It never carries the render service's
    own text, which the exported page can influence.
    """

    def __init__(self, *, status: RenderPdfFailureStatus, reason: str) -> None:
        super().__init__(reason)
        self.status = status


class RenderServicePdf:
    """Client for a render service's ``/v1/pdf``."""

    def __init__(
        self,
        *,
        settings: ExportPdfSettings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if settings.render_url is None:
            raise ValueError("PDF export has no render service URL")
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
        except (TimeoutError, httpx.TimeoutException) as error:
            raise RenderPdfFailed(status=504, reason="no PDF arrived within the time budget") from error
        except httpx.RequestError as error:
            raise RenderPdfFailed(
                status=502, reason=f"the render service can't be reached ({type(error).__name__})"
            ) from error

        pdf = _decode_pdf(answer)
        # Parsing is CPU work on up to tens of MiB, so it runs off the event loop.
        pages = await asyncio.to_thread(_count_pages, pdf=pdf, max_pages=self._settings.max_pages)
        if pages > self._settings.max_pages:
            raise RenderPdfFailed(status=413, reason=f"the PDF has over {self._settings.max_pages} pages")
        return pdf

    async def _post(self, *, client: httpx.AsyncClient, body: bytes, remaining: float) -> bytes | None:
        """Send one request. Return the 200 body, or None for a 503."""
        async with client.stream(
            "POST",
            self._endpoint,
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(remaining),
        ) as response:
            status = response.status_code
            if status == 200:
                return await self._read_bounded(response)
            if status == 503:
                return None
            if status in (413, 422):
                raise RenderPdfFailed(status=status, reason=f"the render service answered {status}")
            # Any redirect lands here too: following one would send the
            # document to an address nobody configured.
            raise RenderPdfFailed(status=502, reason=f"the render service answered {status}")

    async def _read_bounded(self, response: httpx.Response) -> bytes:
        """Read the body, stopping as soon as it passes ``max_pdf_bytes``."""
        limit = self._settings.max_pdf_bytes
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > limit:
                raise RenderPdfFailed(status=413, reason=f"the render service's answer is over {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)


def _decode_pdf(answer: bytes) -> bytes:
    """Return the PDF a 200 answer carries, or raise 502."""
    try:
        result = RenderPdfResult.model_validate_json(answer)
    except ValidationError as error:
        raise RenderPdfFailed(status=502, reason="the render service's answer isn't a PDF result") from error
    try:
        pdf = base64.b64decode(result.pdf_base64, validate=True)
    except ValueError as error:  # binascii.Error is a ValueError
        raise RenderPdfFailed(status=502, reason="the render service's PDF isn't valid base64") from error
    if not pdf.startswith(PDF_MAGIC):
        raise RenderPdfFailed(status=502, reason="the render service's bytes aren't a PDF")
    return pdf


def _count_pages(*, pdf: bytes, max_pages: int) -> int:
    """Count the PDF's pages with pypdf, stopping once past ``max_pages``.

    pypdf's own count lists every page in the tree, recursing however large
    or deep the tree is. This walk is a loop instead: it visits at most
    ``2 * max_pages + 2`` page-tree nodes and stops at the first page past
    ``max_pages``. A tree of ``max_pages`` pages where each node has two or
    more children needs fewer nodes than that, so only a hostile tree (a loop,
    a long chain) reaches the node limit. A render service never encrypts its
    output, so an encrypted PDF is refused too.
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
            entry = pending.pop()
            visited += 1
            if visited > node_limit:
                raise RenderPdfFailed(status=502, reason=_MALFORMED_PAGE_TREE)
            node = entry.get_object() if isinstance(entry, IndirectObject) else entry
            if not isinstance(node, DictionaryObject):
                raise RenderPdfFailed(status=502, reason=_MALFORMED_PAGE_TREE)
            # pypdf's rule: a node without /Type is a page unless it has /Kids.
            kind = node.get("/Type")
            if kind == "/Pages" or (kind is None and "/Kids" in node):
                kids = node["/Kids"]
                if not isinstance(kids, ArrayObject):
                    raise RenderPdfFailed(status=502, reason=_MALFORMED_PAGE_TREE)
                pending.extend(kids)
            elif kind == "/Page" or kind is None:
                pages += 1
                if pages > max_pages:
                    return pages
        return pages
    except RenderPdfFailed:
        raise
    except Exception as error:
        # A hostile PDF can make pypdf raise nearly anything: its own errors,
        # KeyError, TypeError, RecursionError. Each one means the bytes aren't
        # a PDF this server accepts.
        raise RenderPdfFailed(status=502, reason="pypdf can't read the render service's PDF") from error
