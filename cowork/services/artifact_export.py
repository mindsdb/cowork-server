"""Export a document artifact's primary file to HTML, Word or PDF, beside the source.

The export reads the primary through the pinned artifact folder, bounded by
size. Markdown and text become HTML first. ``inline_artifact_html`` then makes
the document self-contained: it keeps only files from the artifact's own folder
and never reaches the network. A PDF comes from the render service the process
environment names, and from nothing else. The result is written as a new file
next to the source, under a name nothing else holds, so an export never
replaces a file.
"""

from __future__ import annotations

import codecs
import errno
import html
import json
import os
import secrets
import stat
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import anyio

from cowork.common.paths import (
    O_NOFOLLOW,
    PinnedDir,
    dir_open,
    dir_stat,
    dir_unlink,
    open_pinned_child,
)
from cowork.common.settings.app_settings import get_export_pdf_settings
from cowork.schemas.artifact_export import (
    ExportFormat,
    ExportPdfSettings,
    ExportResult,
    InlineLimits,
    InlineMode,
    OmittedReason,
)
from cowork.services import artifact_files
from cowork.services.artifact_identity import (
    _atomic_json,
    _read_no_follow,
    opened_artifact_folder,
)
from cowork.services.artifact_inline import inline_artifact_html
from cowork.services.artifacts import (
    ProjectArtifacts,
    _org_mode,
    _pick_primary,
    _pinned_directory_path,
    _user_files,
    draft_url_for,
    serve_url_for,
)
from cowork.services.docx_writer import write_docx
from cowork.services.markdown_render import MarkdownRefused, render_markdown
from cowork.services.render_pdf_client import RenderServicePdf, pdf_request

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope


_MARKDOWN_SOURCES = frozenset({".md", ".markdown"})
_HTML_SOURCES = frozenset({".html", ".htm"})
# HTML export is offered only where it makes a new document: an HTML source
# would only be copied.
_HTML_EXPORT_SOURCES = _MARKDOWN_SOURCES | {".txt"}
_WORD_EXPORT_SOURCES = _HTML_EXPORT_SOURCES | _HTML_SOURCES

# The most of a source an export reads. A page carries its own markup and
# styles, so HTML gets more room than Markdown or text. Python-Markdown's slow
# shapes cost more than their size grows, so a smaller Markdown bound shortens
# the longest stall a source can cause; agent-written reports stay far below it.
HTML_SOURCE_MAX_BYTES = 10 * 1024 * 1024
TEXT_SOURCE_MAX_BYTES = 512 * 1024
# The inliner's budget for everything it embeds. Org mode serves many people
# from small replicas, so it embeds less.
INLINE_BUDGET_BYTES = 25 * 1024 * 1024
ORG_INLINE_BUDGET_BYTES = 10 * 1024 * 1024
# A PDF's document travels to the render service as base64 inside JSON, so it
# embeds less.
PDF_INLINE_BUDGET_BYTES = 5 * 1024 * 1024
# How many "name (n).ext" spellings an export tries before it gives up.
_MAX_OUTPUT_NAMES = 100
# The longest file name the volumes Cowork runs on all take: Linux allows 255
# bytes, macOS 255 characters and Windows 255 UTF-16 units, and a name of 255
# UTF-8 bytes fits each.
_MAX_NAME_BYTES = 255
# What link() answers on a volume with no hard links: ENOTSUP on macOS exFAT
# and FAT32, and EPERM on Linux vfat. EOPNOTSUPP is ENOTSUP's other spelling.
_NO_HARD_LINKS = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM})

_OUTPUT_EXTENSIONS: dict[ExportFormat, str] = {"html": ".html", "docx": ".docx", "pdf": ".pdf"}
_INLINE_MODES: dict[ExportFormat, InlineMode] = {
    "html": InlineMode.HTML,
    "docx": InlineMode.WORD,
    "pdf": InlineMode.PDF,
}
_METADATA = "metadata.json"
_BUSY = "An export is already running; try again in a moment."
_NOT_OFFERED = "This file can't be exported to that format."

# Org mode offers no export format, and the route answers 403 there. Measured
# with docker --cpus=1 --memory=1g from a server's own baseline, single exports
# within the org caps peak past 300 MiB: a 10 MiB page of one-word paragraphs
# past 500 MiB, and a 512 KiB Markdown table or a report with five 2 MiB
# images at 330 to 350 MiB. Short unclosed code fences still stall the event
# loop for seconds and grow faster than their size;
# scripts/measure_export_capacity.py holds the cases. A replica serves many
# people at once, so org export waits until the conversion runs in a process
# of its own that can be killed.
ORG_EXPORT_ENABLED = False

# HTML and Word export share one conversion at a time, PDF export has its own,
# and each user and each organization runs one export at a time. All of them
# hold per process, so every replica admits its own. Nothing waits for a slot:
# a busy server answers at once.
_conversion_limiter = anyio.CapacityLimiter(1)
_pdf_limiter = anyio.CapacityLimiter(1)
_running_jobs: set[str] = set()


class ExportRefusal(StrEnum):
    """Why an export was refused before anything was written."""

    NOT_OFFERED = "not_offered"
    TOO_LARGE = "too_large"
    NOT_FOUND = "not_found"
    BUSY = "busy"


class ExportRefused(Exception):
    """An export the route answers with a 4xx. Nothing was written."""

    def __init__(self, refusal: ExportRefusal, detail: str) -> None:
        super().__init__(detail)
        self.refusal = refusal


@dataclass(frozen=True)
class ExportSource:
    """What decides an artifact's export formats.

    ``source_name`` is the primary file's path inside the artifact folder,
    "/"-separated, or "" when the artifact has no primary file.
    """

    source_name: str
    is_backend: bool


def is_backend_type(artifact_type: object) -> bool:
    """Whether metadata's ``type`` names a full-stack app, which never exports."""
    return str(artifact_type or "").startswith("fullstack-")


def export_formats(
    *, source_name: str, is_backend: bool, can_edit: bool, pdf: ExportPdfSettings
) -> list[ExportFormat]:
    """The formats an artifact's Export offers this caller, in the order html, docx, pdf."""
    if is_backend or not can_edit:
        return []
    segments = source_name.replace("\\", "/").split("/")
    if not _read_takes_name(segments):
        return []
    suffix = PurePosixPath(segments[-1]).suffix.lower()
    formats: list[ExportFormat] = []
    if suffix in _HTML_EXPORT_SOURCES:
        formats.append("html")
    if suffix in _WORD_EXPORT_SOURCES:
        formats.append("docx")
        if pdf.engine == "render" and pdf.render_url is not None:
            formats.append("pdf")
    return formats


def _read_takes_name(segments: list[str]) -> bool:
    """Whether the export's pinned read takes a path of these segments, by text alone.

    It refuses a name Windows would read as another file, such as ``aux.md``
    or ``notes:q3.md``, on every platform, and a path nested past its depth.
    It refuses housekeeping names as the read does: any as the first segment,
    and a store-owned one such as ``metadata.json`` at any depth, so
    ``docs/README.md`` exports.
    """
    if len(segments) > artifact_files.MAX_PATH_PARTS:
        return False
    try:
        artifact_files.refuse_unsafe_parts(parts=tuple(segments))
        artifact_files.refuse_housekeeping(parts=tuple(segments))
    except artifact_files.ArtifactFileRefused:
        return False
    return True


def export_pdf_settings() -> ExportPdfSettings:
    """The PDF settings exports run with, read once from the process environment.

    Only a deployment that sets ``COWORK_EXPORT_PDF_ENGINE=render`` and a render
    service URL there offers PDF.
    """
    return get_export_pdf_settings()


def org_export_refused(*, org_mode: bool) -> bool:
    """Whether export is off for this request: org mode while ``ORG_EXPORT_ENABLED`` is off."""
    return org_mode and not ORG_EXPORT_ENABLED


def offered_export_formats(
    *, source: ExportSource, can_edit: bool, org_mode: bool
) -> list[ExportFormat]:
    """The formats this caller may export: ``export_formats``, or none while org export is off."""
    if org_export_refused(org_mode=org_mode):
        return []
    return export_formats(
        source_name=source.source_name,
        is_backend=source.is_backend,
        can_edit=can_edit,
        pdf=export_pdf_settings(),
    )


def _primary_parts(*, folder: Path, metadata: dict) -> tuple[str, ...] | None:
    """The primary file's path inside *folder*, picked the way the artifact card picks it."""
    hint = metadata.get("primary")
    primary = _pick_primary(
        folder, _user_files(folder), primary_hint=hint if isinstance(hint, str) else None
    )
    if primary is None:
        return None
    try:
        relative = primary.resolve(strict=False).relative_to(folder.resolve(strict=False))
    except (OSError, ValueError):
        return None
    return relative.parts or None


def _exportable_primary_parts(*, folder: Path, metadata: dict) -> tuple[str, ...] | None:
    """The primary file's path inside *folder*, or None when the export's read would refuse it.

    A primary the read refuses for what it is on disk names no source: a file
    that isn't regular, or one with a second hard link, which a sync or backup
    tool can give it. So the workspace response offers no format for it, and
    the route answers 422, as for any format the artifact doesn't offer.
    """
    parts = _primary_parts(folder=folder, metadata=metadata)
    if parts is not None and not _single_link_file(folder.joinpath(*parts)):
        return None
    return parts


def primary_export_source(*, folder: Path, metadata: dict) -> ExportSource:
    """The export source of the artifact in *folder*: its primary file and its type."""
    parts = _exportable_primary_parts(folder=folder, metadata=metadata)
    return ExportSource(
        source_name="/".join(parts or ()),
        is_backend=is_backend_type(metadata.get("type")),
    )


def _single_link_file(path: Path) -> bool:
    try:
        found = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(found.st_mode) and found.st_nlink == 1


def card_export_source(card: dict) -> ExportSource:
    """The export source an artifact card describes, from its ``path``, ``folder`` and ``type``.

    A primary with a second hard link names no source, as in
    ``primary_export_source``. The ``lstat`` that finds one is skipped while
    org export is off, as the workspace response skips its walk: no card
    offers a format then.
    """
    try:
        relative = Path(str(card.get("path") or "")).relative_to(
            Path(str(card.get("folder") or ""))
        )
    except ValueError:
        relative = Path()
    if (
        relative.parts
        and not org_export_refused(org_mode=_org_mode())
        and not _single_link_file(Path(str(card["path"])))
    ):
        relative = Path()
    return ExportSource(
        source_name="/".join(relative.parts),
        is_backend=is_backend_type(card.get("type")),
    )


def _job_keys(scope: TenantScope | None) -> frozenset[str]:
    if scope is None:
        return frozenset()
    keys = set()
    if scope.user_id:
        keys.add(f"user:{scope.user_id}")
    if scope.org_id:
        keys.add(f"org:{scope.org_id}")
    # A deployment that names no person, such as the desktop app, serves one,
    # so its exports run one at a time too, whatever their format.
    return frozenset(keys or {"local"})


@asynccontextmanager
async def export_slot(
    *, scope: TenantScope | None, export_format: ExportFormat
) -> AsyncIterator[None]:
    """Admit one export now, or raise ``ExportRefused`` as busy. Nothing waits."""
    keys = _job_keys(scope)
    if keys & _running_jobs:
        raise ExportRefused(ExportRefusal.BUSY, _BUSY)
    limiter = _pdf_limiter if export_format == "pdf" else _conversion_limiter
    # Its own borrower, so the token is this job's whichever task holds it.
    borrower = object()
    try:
        limiter.acquire_on_behalf_of_nowait(borrower)
    except anyio.WouldBlock as exc:
        raise ExportRefused(ExportRefusal.BUSY, _BUSY) from exc
    _running_jobs.update(keys)
    try:
        yield
    finally:
        _running_jobs.difference_update(keys)
        limiter.release_on_behalf_of(borrower)


def _pdf_client() -> RenderServicePdf:
    return RenderServicePdf(settings=export_pdf_settings())


def _render_pdf(*, source_name: str, html: str) -> bytes:
    """The PDF of the inlined *html*, from the render service.

    Blocking, like the rest of the export: it runs in the export's worker
    thread and hands the async render call to the event loop, so the thread
    waits while the loop does the network I/O. ``RenderPdfFailed`` carries
    the status the route answers.
    """
    request = pdf_request(source_name=source_name, html=html)
    client = _pdf_client()
    return anyio.from_thread.run(lambda: client.render(request=request))


def _read_source(*, folder: PinnedDir, parts: tuple[str, ...], suffix: str) -> bytes:
    max_bytes = HTML_SOURCE_MAX_BYTES if suffix in _HTML_SOURCES else TEXT_SOURCE_MAX_BYTES
    try:
        return artifact_files.read_artifact_file(folder=folder, parts=parts, max_bytes=max_bytes)
    except artifact_files.ArtifactFileRefused as exc:
        if exc.reason is OmittedReason.TOO_LARGE:
            raise ExportRefused(ExportRefusal.TOO_LARGE, "This file is too large to export.") from exc
        if exc.reason is OmittedReason.HOUSEKEEPING:
            raise ExportRefused(ExportRefusal.NOT_OFFERED, _NOT_OFFERED) from exc
        raise ExportRefused(ExportRefusal.NOT_FOUND, "Artifact source not found") from exc


def _source_html(*, data: bytes, suffix: str, title: str) -> bytes:
    """The source as one HTML document for the inliner.

    An HTML page stays as it is. Markdown is rendered and text becomes a
    ``pre`` block, and either is wrapped as the body of a minimal document
    titled *title*, which is what an HTML export of it writes. Markdown that
    ``render_markdown`` refuses is answered as a source over its read bound
    is, before anything is written.
    """
    if suffix in _HTML_SOURCES:
        return data
    text = _source_text(data)
    if suffix in _MARKDOWN_SOURCES:
        try:
            body = render_markdown(text=text)
        except MarkdownRefused as exc:
            raise ExportRefused(ExportRefusal.TOO_LARGE, "This file can't be exported.") from exc
    else:
        body = f"<pre>{html.escape(text)}</pre>"
    return _wrap_html_document(body, title).encode("utf-8")


def _source_text(data: bytes) -> str:
    """A Markdown or text source's bytes as text.

    They are read as UTF-8, and a UTF-8 byte order mark is dropped, so a first
    line such as ``# Title`` stays a heading. A UTF-16 byte order mark, which
    Windows editors write for "Unicode" text, reads the rest as UTF-16. Bytes
    that don't decode become U+FFFD.
    """
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def _html_export(document: str) -> bytes:
    """The inlined document as the bytes of an ``.html`` file.

    Written as the inliner serialized it, never parsed again: lxml's HTML
    parser stops at about 10 MB unless its limits are lifted, and inlined
    images can take a document past that.
    """
    if not document.lstrip()[:9].lower().startswith("<!doctype"):
        document = "<!DOCTYPE html>" + document
    return document.encode("utf-8")


def _wrap_html_document(body_html: str, title: str) -> str:
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


def _primary_names_a_file(*, folder: PinnedDir, primary: object) -> bool:
    """Whether metadata's *primary* names a file the card shows.

    The card reads the hint through ``_pick_primary``; with no files to fall
    back on, it returns a file only when the hint names one.
    """
    io_folder = _pinned_directory_path(folder)
    if not isinstance(primary, str) or not primary or io_folder is None:
        return False
    return _pick_primary(io_folder, [], primary_hint=primary) is not None


def _record_primary(*, folder: PinnedDir, primary: str) -> None:
    """Record the primary the card shows when ``metadata.json`` names none.

    Without it, the new file could become the card's primary: the card picks an
    HTML file over Markdown, and otherwise the newest file. So it runs before
    the new file takes its name. A primary that names no file, such as one the
    agent renamed, counts as none, because the card falls back the same way.
    The primary is merged into the latest document and written atomically, as
    ``ensure_full_id`` widens an id, and the file's timestamps are put back
    because the artifact's content did not change. If the export then fails,
    the record stays: it names the file the card already shows. A primary
    that names a file is never changed.
    """
    latest = json.loads(_read_no_follow(folder, _METADATA))
    if not isinstance(latest, dict) or _primary_names_a_file(
        folder=folder, primary=latest.get("primary")
    ):
        return
    before = dir_stat(folder, _METADATA, follow_symlinks=False)
    _atomic_json(folder, _METADATA, {**latest, "primary": primary})
    times = (before.st_atime_ns, before.st_mtime_ns)
    try:
        if folder.fd is not None:
            os.utime(_METADATA, ns=times, dir_fd=folder.fd, follow_symlinks=False)
        else:
            # Windows, the only place a pinned folder has no descriptor, can't
            # set times without following a link: follow_symlinks=False raises
            # NotImplementedError there. _atomic_json has just put a regular
            # file at this name.
            os.utime(folder.path / _METADATA, ns=times)
    except OSError:
        pass


def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written == 0:
            raise OSError("export write made no progress")
        remaining = remaining[written:]


def _link_without_replacing(folder: PinnedDir, source_name: str, destination_name: str) -> None:
    """Give *source_name* a second name, refusing one that already exists.

    A hard link fails on any existing entry, a symlink or a dangling one
    included, so nothing is written through it. Windows has no directory
    descriptor, and its rename refuses an existing destination the same way.
    """
    if folder.fd is not None:
        os.link(
            source_name,
            destination_name,
            src_dir_fd=folder.fd,
            dst_dir_fd=folder.fd,
            follow_symlinks=False,
        )
    else:
        os.rename(folder.path / source_name, folder.path / destination_name)


def _create_without_replacing(folder: PinnedDir, name: str, data: bytes) -> None:
    """Write *data* to a new file at *name*, refusing one that already exists.

    For a volume with no hard links, such as exFAT or FAT32. ``O_EXCL |
    O_NOFOLLOW`` refuses any existing entry, a symlink included, as a link
    does. Unlike a link, the file shows at its name while it is written, so a
    failed write removes it.
    """
    fd = dir_open(folder, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW, 0o644)
    try:
        try:
            _write_all(fd, data)
        finally:
            os.close(fd)
    except BaseException:
        try:
            dir_unlink(folder, name)
        except OSError:
            pass
        raise


def _output_stem(*, stem: str, extension: str) -> str:
    """*stem*, cut so that every output name fits ``_MAX_NAME_BYTES``.

    The longest output name is ``stem (100).ext``. The cut falls between
    characters, never inside one.
    """
    room = _MAX_NAME_BYTES - len(os.fsencode(f" ({_MAX_OUTPUT_NAMES}){extension}"))
    while len(os.fsencode(stem)) > room:
        stem = stem[:-1]
    return stem


def _write_beside(
    *,
    folder: PinnedDir,
    parts: tuple[str, ...],
    extension: str,
    data: bytes,
    before_placing: Callable[[], None],
) -> str:
    """Write *data* next to the source at *parts* under the first free name.

    The names run ``report.docx``, ``report (2).docx``, ``report (3).docx``,
    and never the source's own name. The bytes go to a temporary file created
    with ``O_EXCL | O_NOFOLLOW``. Then *before_placing* runs, and only once it
    returns does the file take the free name, without replacing anything; on
    a volume with no hard links the bytes are written to the free name
    itself, created the same way. The temporary name is always removed, as
    soon as the file has its own, so a failed export leaves no file behind.
    Returns the name.

    Without a directory descriptor, as on Windows, each folder on the way is
    opened by a joined path, so each gets the read's ``lstat`` checks. The
    first refuses a link before the write opens through it. The second, just
    before the file takes its name, refuses a walk that changed since. A
    folder swapped for a link and back between the two is still followed,
    as it is on the read.
    """
    source_name = parts[-1]
    stem = _output_stem(stem=PurePosixPath(source_name).stem, extension=extension)
    checked: list[artifact_files.CheckedName] = []
    with ExitStack() as opened:
        parent = folder
        for requested in parts[:-1]:
            entry = artifact_files.existing_draft_entry_name(parent, requested)
            seen = artifact_files.refuse_link_by_lstat(directory=parent, name=entry)
            if seen is not None:
                checked.append(seen)
            parent = open_pinned_child(parent, entry)
            opened.callback(parent.close)
        temporary = f".cowork-export-{secrets.token_hex(12)}.tmp"
        fd = dir_open(parent, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW, 0o644)
        name = None
        try:
            try:
                _write_all(fd, data)
            finally:
                os.close(fd)
            before_placing()
            artifact_files.refuse_a_changed_walk(checked=checked)
            name = _place_temporary(
                parent=parent,
                temporary=temporary,
                stem=stem,
                extension=extension,
                source_name=source_name,
                data=data,
            )
        finally:
            _remove_temporary(parent=parent, temporary=temporary, placed=name)
        return name


def _remove_temporary(*, parent: PinnedDir, temporary: str, placed: str | None) -> None:
    """Remove the temporary name, and the placed file too if that fails.

    Until the temporary name goes, a linked file has two links, and the drafts
    route never serves a file with a second link. So an export whose
    temporary name stays fails, and leaves no file at an output name.
    """
    try:
        dir_unlink(parent, temporary)
    except FileNotFoundError:
        pass
    except OSError:
        if placed is not None:
            try:
                dir_unlink(parent, placed)
            except OSError:
                pass
        raise


def _place_temporary(
    *, parent: PinnedDir, temporary: str, stem: str, extension: str, source_name: str, data: bytes
) -> str:
    """Give *temporary* the first free output name and return that name.

    Where the volume refuses a hard link, *data* is written to the free name
    directly instead, still never over an existing entry.
    """
    link = True
    for number in range(1, _MAX_OUTPUT_NAMES + 1):
        name = f"{stem}{extension}" if number == 1 else f"{stem} ({number}){extension}"
        if name.casefold() == source_name.casefold():
            continue
        try:
            if link:
                try:
                    _link_without_replacing(parent, temporary, name)
                    return name
                except OSError as exc:
                    if exc.errno not in _NO_HARD_LINKS:
                        raise
                    link = False
            _create_without_replacing(parent, name, data)
        except FileExistsError:
            continue
        return name
    raise FileExistsError(f"no free name for {stem}{extension}")


def export_primary_file(
    *,
    source: ProjectArtifacts,
    folder_name: str,
    metadata: dict,
    artifact_id: str,
    export_format: ExportFormat,
    can_edit: bool,
) -> ExportResult:
    """Export the artifact's primary file, writing the result beside it.

    Blocking: the route runs it in an anyio worker thread (``run_in_threadpool``),
    which a PDF export needs to reach the event loop. Raises ``ExportRefused``
    for a format the artifact doesn't offer, a source over its size bound,
    Markdown that ``render_markdown`` refuses or a source that can't be
    opened, and ``RenderPdfFailed`` when no usable PDF came back, and writes
    nothing then.
    """
    org_mode = _org_mode()
    with opened_artifact_folder(source, folder_name) as folder:
        io_folder = _pinned_directory_path(folder)
        if io_folder is None:
            raise ExportRefused(ExportRefusal.NOT_FOUND, "Artifact source not found")
        parts = _exportable_primary_parts(folder=io_folder, metadata=metadata)
        offered = offered_export_formats(
            source=ExportSource(
                source_name="/".join(parts or ()),
                is_backend=is_backend_type(metadata.get("type")),
            ),
            can_edit=can_edit,
            org_mode=org_mode,
        )
        extension = _OUTPUT_EXTENSIONS.get(export_format)
        if parts is None or export_format not in offered or extension is None:
            raise ExportRefused(ExportRefusal.NOT_OFFERED, _NOT_OFFERED)
        suffix = PurePosixPath(parts[-1]).suffix.lower()
        title = PurePosixPath(parts[-1]).stem
        data = _read_source(folder=folder, parts=parts, suffix=suffix)
        if export_format == "pdf":
            budget = PDF_INLINE_BUDGET_BYTES
        else:
            budget = ORG_INLINE_BUDGET_BYTES if org_mode else INLINE_BUDGET_BYTES
        inlined = inline_artifact_html(
            html=_source_html(data=data, suffix=suffix, title=title),
            folder=folder,
            base_parts=parts[:-1],
            mode=_INLINE_MODES[export_format],
            limits=InlineLimits(max_total_bytes=budget),
        )
        if export_format == "docx":
            output = write_docx(document=inlined, title=title)
        elif export_format == "pdf":
            output = _render_pdf(source_name=parts[-1], html=inlined.html)
        else:
            output = _html_export(inlined.html)
        name = _write_beside(
            folder=folder,
            parts=parts,
            extension=extension,
            data=output,
            before_placing=lambda: _record_primary(folder=folder, primary="/".join(parts)),
        )
    rel_path = "/".join((*parts[:-1], name))
    return ExportResult(
        filename=name,
        draft_url=draft_url_for(
            project_id=source.project_id, artifact_id=artifact_id, rel_path=rel_path
        ),
        # "" in org mode, where the server serves no artifact file by path.
        serve_url=serve_url_for(
            Path(source.base) / folder_name / rel_path,
            scope=None,
            artifacts_base=Path(source.base),
            project_name=source.project_name,
        ),
        omitted_count=len(inlined.omitted),
        scripts_dropped=inlined.scripts_dropped,
    )
