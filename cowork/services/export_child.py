"""Convert one export's source in a process cowork-server can kill.

Python-Markdown spends its time in regular-expression calls that hold the GIL
and that nothing in a process can interrupt, so one hostile Markdown file
would hold every request a server process runs. The export route therefore
converts in a process of its own, ``python -I -B -m
cowork.services.export_child``, one per export, and kills it past its time
or memory limit (``cowork.services.export_runner``).

The process gets no path and no folder. It reads the source from its standard
input, asks the server for each file the document names, and writes the
result back, one frame per message (``cowork.schemas.export_child``). It
limits itself first where the OS lets it: no core dump, no data written to a
file, no process started, a CPU-time limit just past the server's wall clock,
and on Linux an address-space limit, the first process the kernel kills when
memory runs out, and a lower CPU priority. An audit hook refuses network and process
starts too. That is a check on our own code, not a sandbox: native code in a
parser can bypass it, and the process runs as the server's user.

``convert`` is the conversion itself, and the server's tests call it in their
own process too. Publishing a Markdown artifact renders its page here as well
(``publish_markdown_body``), for the same reason.
"""
from __future__ import annotations

import codecs
import html
import json
import os
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO, NamedTuple, NoReturn

import markdown
from pydantic import BaseModel, TypeAdapter

from cowork.common.structured_logging import safe_exception_traceback
from cowork.schemas.artifact_export import ExportFormat, InlineMode, OmittedReason
from cowork.schemas.export_child import (
    AssetReply,
    AssetRequest,
    ChildMessage,
    ConversionFailed,
    Converted,
    ConvertRequest,
    PublishMarkdownRequest,
    ServerMessage,
)
from cowork.services.artifact_files import (
    MAX_PATH_PARTS,
    ArtifactFileContent,
    ArtifactFileRefused,
)
from cowork.services.artifact_inline import AssetReader, inline_html
from cowork.services.docx_writer import write_docx
from cowork.services.markdown_render import MarkdownRefused, render_markdown

MARKDOWN_SOURCES = frozenset({".md", ".markdown"})
HTML_SOURCES = frozenset({".html", ".htm"})
INLINE_MODES: dict[ExportFormat, InlineMode] = {
    "html": InlineMode.HTML,
    "docx": InlineMode.WORD,
    "pdf": InlineMode.PDF,
}

# How the process ends when it refuses the source: Markdown that
# ``render_markdown`` refuses, or memory that ran out. The server answers 413.
EXIT_REFUSED = 3
# How it ends after it reported a ``ConversionFailed``. The server answers 500.
EXIT_FAILED = 4

# The longest message a frame may carry, as JSON. A message names at most a
# path and a title; a longer one breaks the protocol.
MAX_HEADER_BYTES = 64 * 1024
# The largest payload either side accepts. The largest is an HTML export: a
# 10 MiB source plus a 25 MiB inline budget, which base64 grows by a third.
# Serializing the page again can grow its text up to five times ("&" becomes
# "&amp;"), so an output past this bound is refused (413), never sent.
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_LENGTH = struct.Struct(">I")

_SERVER_MESSAGE: TypeAdapter[ConvertRequest | PublishMarkdownRequest | AssetReply] = TypeAdapter(
    ServerMessage
)
CHILD_MESSAGE: TypeAdapter[AssetRequest | Converted | ConversionFailed] = TypeAdapter(
    ChildMessage
)

# Audit events the conversion never raises: a connection or a lookup, and a
# process start. ``socket.`` covers every socket event.
_DENIED_EVENTS = (
    "socket.",
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "os.startfile",
    # Windows starts a multiprocessing worker through this, not subprocess.
    "_winapi.CreateProcess",
)


@dataclass(frozen=True)
class Conversion:
    """A finished conversion: the output, and what the export reports about it."""

    data: bytes
    omitted_count: int
    scripts_dropped: bool


class FrameError(Exception):
    """A frame that breaks the protocol: too long, cut short, or not a known message."""


class Frame[MessageT](NamedTuple):
    """One frame read: its validated message and the raw bytes it carries."""

    message: MessageT
    payload: bytes


def convert(*, request: ConvertRequest, source: bytes, read_asset: AssetReader) -> Conversion:
    """Convert *source* to *request*'s format, reading every file it names through *read_asset*.

    For a PDF the output is the inlined HTML, which the server sends to the
    render service. Raises ``MarkdownRefused`` for Markdown that
    ``render_markdown`` refuses.
    """
    inlined = inline_html(
        html=source_html(data=source, suffix=request.suffix, title=request.title),
        read_asset=read_asset,
        base_parts=request.base_parts,
        mode=INLINE_MODES[request.export_format],
        limits=request.limits,
    )
    if request.export_format == "docx":
        data = write_docx(document=inlined, title=request.title)
    elif request.export_format == "pdf":
        data = inlined.html.encode("utf-8")
    else:
        data = html_export(inlined.html)
    return Conversion(
        data=data, omitted_count=len(inlined.omitted), scripts_dropped=inlined.scripts_dropped
    )


def publish_markdown_body(*, text: str) -> str:
    """A published Markdown page's body: Python-Markdown with the extensions publish has always used."""
    return markdown.markdown(
        text, extensions=["fenced_code", "tables", "toc", "sane_lists"], output_format="html5"
    )


def source_html(*, data: bytes, suffix: str, title: str) -> bytes:
    """The source as one HTML document for the inliner.

    An HTML page stays as it is. Markdown is rendered and text becomes a
    ``pre`` block, and either is wrapped as the body of a minimal document
    titled *title*, which is what an HTML export of it writes. Raises
    ``MarkdownRefused`` for Markdown that ``render_markdown`` refuses.
    """
    if suffix in HTML_SOURCES:
        return data
    text = source_text(data)
    if suffix in MARKDOWN_SOURCES:
        body = render_markdown(text=text)
    else:
        body = f"<pre>{html.escape(text)}</pre>"
    return wrap_html_document(body, title).encode("utf-8")


def source_text(data: bytes) -> str:
    """A Markdown or text source's bytes as text.

    They are read as UTF-8, and a UTF-8 byte order mark is dropped, so a first
    line such as ``# Title`` stays a heading. A UTF-16 byte order mark, which
    Windows editors write for "Unicode" text, reads the rest as UTF-16. Bytes
    that don't decode become U+FFFD.
    """
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def html_export(document: str) -> bytes:
    """The inlined document as the bytes of an ``.html`` file.

    Written as the inliner serialized it, never parsed again: lxml's HTML
    parser stops at about 10 MB unless its limits are lifted, and inlined
    images can take a document past that.
    """
    if not document.lstrip()[:9].lower().startswith("<!doctype"):
        document = "<!DOCTYPE html>" + document
    return document.encode("utf-8")


def wrap_html_document(body_html: str, title: str) -> str:
    """Wrap rendered body HTML in a minimal, print-friendly document."""
    safe_title = html.escape(title or "Document")
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{safe_title}</title>"
        "<style>"
        "body{font-family:'Helvetica','Arial',sans-serif;font-size:11pt;"
        "line-height:1.5;color:#1a1a1a;margin:2.5em;}"
        "h1,h2,h3{line-height:1.25;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}"
        "code,pre{font-family:'Courier New',monospace;background:#f5f5f5;}"
        "pre{padding:10px;overflow-x:auto;white-space:pre-wrap;}"
        "</style></head><body>"
        f"{body_html}"
        "</body></html>"
    )


def write_frame(*, stream: BinaryIO, message: BaseModel, payload: bytes = b"") -> None:
    """Write *message* and its *payload* as one frame, and flush it.

    The message goes as JSON with every non-ASCII character escaped, so a
    name holding a lone surrogate, which a document or a disk name can carry,
    survives the trip.
    """
    header = _header(message)
    stream.write(_LENGTH.pack(len(header)))
    stream.write(header)
    stream.write(_LENGTH.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _header(message: BaseModel) -> bytes:
    return json.dumps(message.model_dump(mode="python"), ensure_ascii=True).encode("ascii")


def read_frame[MessageT](
    *, stream: BinaryIO, adapter: TypeAdapter[MessageT]
) -> Frame[MessageT] | None:
    """The next frame's message, validated by *adapter*, and its payload.

    None when the stream ends between frames. Raises ``FrameError`` for a
    frame over ``MAX_HEADER_BYTES`` or ``MAX_PAYLOAD_BYTES``, checked before
    it is read, for one the stream cuts short, and for a message *adapter*
    refuses.
    """
    head = stream.read(_LENGTH.size)
    if not head:
        return None
    header = _read_exact(stream=stream, size=_bounded_length(head=head, limit=MAX_HEADER_BYTES))
    payload_size = _bounded_length(
        head=_read_exact(stream=stream, size=_LENGTH.size), limit=MAX_PAYLOAD_BYTES
    )
    payload = _read_exact(stream=stream, size=payload_size)
    try:
        return Frame(message=adapter.validate_python(json.loads(header)), payload=payload)
    except ValueError as exc:
        raise FrameError("not a known message") from exc


def _bounded_length(*, head: bytes, limit: int) -> int:
    if len(head) != _LENGTH.size:
        raise FrameError("the stream ended inside a frame")
    (size,) = _LENGTH.unpack(head)
    if size > limit:
        raise FrameError("a frame over its size limit")
    return size


def _read_exact(*, stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise FrameError("the stream ended inside a frame")
    return data


def _server_reader(*, frames_in: BinaryIO, frames_out: BinaryIO) -> AssetReader:
    """Read each file the document names by asking the server for it.

    A name the server would refuse by its text alone is refused here, before
    it is asked for: one nested past the server's depth, or one too long for
    a frame, which would otherwise break the protocol and fail the export.
    """

    def read(parts: tuple[str, ...]) -> ArtifactFileContent:
        if len(parts) > MAX_PATH_PARTS:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "path nests too deep")
        asked = AssetRequest(parts=parts)
        if len(_header(asked)) > MAX_HEADER_BYTES:
            raise ArtifactFileRefused(OmittedReason.MISSING, "name too long")
        write_frame(stream=frames_out, message=asked)
        received = read_frame(stream=frames_in, adapter=_SERVER_MESSAGE)
        if received is None or not isinstance(received.message, AssetReply):
            raise FrameError("expected the server's answer to a read")
        reply, data = received
        if reply.refused is not None:
            raise ArtifactFileRefused(reply.refused, "refused by the server")
        return ArtifactFileContent(data=data, device=reply.device, inode=reply.inode)

    return read


def _limit_self(*, memory_bytes: int, cpu_seconds: int) -> None:
    """Set this process's own limits, as far as its OS allows.

    Windows has none of these: there the server's wall clock and memory watch
    are the limits. macOS refuses an address-space limit below what a process
    already maps, so the server's memory watch is the limit there too.
    """
    if sys.platform == "win32":
        return
    import resource

    if sys.platform.startswith("linux"):
        # Before the file-size limit below: it must not apply to this write.
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as score:
            score.write("1000")
        _lower_limit(resource.RLIMIT_AS, soft=memory_bytes, hard=memory_bytes)
    _lower_limit(resource.RLIMIT_CORE, soft=0, hard=0)
    _lower_limit(resource.RLIMIT_FSIZE, soft=0, hard=0)
    _lower_limit(resource.RLIMIT_NPROC, soft=0, hard=0)
    # SIGXCPU at the soft limit, SIGKILL at the hard one.
    _lower_limit(resource.RLIMIT_CPU, soft=cpu_seconds, hard=cpu_seconds + 1)
    os.nice(10)


def _lower_limit(limit: int, *, soft: int, hard: int) -> None:
    """Lower *limit* to *soft* and *hard*, never above the hard limit it already has."""
    import resource

    current_hard = resource.getrlimit(limit)[1]
    if current_hard != resource.RLIM_INFINITY:
        soft, hard = min(soft, current_hard), min(hard, current_hard)
    resource.setrlimit(limit, (soft, hard))


def _deny_network_and_processes() -> Callable[[], int]:
    """Refuse every audit event in ``_DENIED_EVENTS`` from now on; return their count so far."""
    denied = 0

    def refuse(event: str, _args: tuple[object, ...]) -> None:
        nonlocal denied
        if event.startswith(_DENIED_EVENTS):
            denied += 1
            raise RuntimeError(f"{event} is refused in an export conversion")

    sys.addaudithook(refuse)
    return lambda: denied


def _exit_refused(*, why: str) -> NoReturn:
    """End as refused, which the server answers 413, leaving *why* on the error output it logs."""
    try:
        sys.stderr.write(f"{why}\n")
        sys.stderr.flush()
    finally:
        os._exit(EXIT_REFUSED)


def main() -> None:
    """Run one conversion: ``argv`` gives the memory and CPU-time limits, in bytes and seconds.

    Everything after the imports runs inside the ``try``, so a limit the host
    refuses is reported like any other failure, and the server logs why.
    """
    frames_in, frames_out = sys.stdin.buffer, sys.stdout.buffer
    try:
        memory_bytes, cpu_seconds = (int(value) for value in sys.argv[1:3])
        _limit_self(memory_bytes=memory_bytes, cpu_seconds=cpu_seconds)
        denied = _deny_network_and_processes()
        received = read_frame(stream=frames_in, adapter=_SERVER_MESSAGE)
        if received is None or isinstance(received.message, AssetReply):
            raise FrameError("expected a conversion request")
        request, source = received
        if isinstance(request, PublishMarkdownRequest):
            body = publish_markdown_body(text=source.decode("utf-8"))
            conversion = Conversion(data=body.encode("utf-8"), omitted_count=0, scripts_dropped=False)
        else:
            conversion = convert(
                request=request,
                source=source,
                read_asset=_server_reader(frames_in=frames_in, frames_out=frames_out),
            )
        if len(conversion.data) > MAX_PAYLOAD_BYTES:
            # More than the server reads in one frame: refused as too large.
            _exit_refused(why="the output is past the frame bound")
        write_frame(
            stream=frames_out,
            message=Converted(
                omitted_count=conversion.omitted_count,
                scripts_dropped=conversion.scripts_dropped,
                denied_events=denied(),
            ),
            payload=conversion.data,
        )
    except (MarkdownRefused, MemoryError):
        why = ""
        try:
            # Types and locations only, so a real bug behind a refusal shows
            # in the server's log without the document's values.
            why = safe_exception_traceback(sys.exc_info()) or ""
        finally:
            _exit_refused(why=why)
    except BaseException:
        try:
            # Types and locations only: an exception's values can repeat the document.
            write_frame(
                stream=frames_out,
                message=ConversionFailed(
                    error_type=sys.exc_info()[0].__name__,
                    detail=safe_exception_traceback(sys.exc_info()) or "",
                ),
            )
        finally:
            os._exit(EXIT_FAILED)


if __name__ == "__main__":
    main()
