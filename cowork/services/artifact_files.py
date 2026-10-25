"""Open a file inside an artifact folder without following anything out of it.

An artifact folder is agent-writable, and the paths that reach it come from
an HTTP request or from a document the agent wrote. So every path walks down
from a pinned folder one segment at a time. Each segment must match a name
``scandir`` returned, so a descriptor-relative open only ever receives a name
found on disk. Each open uses ``O_NOFOLLOW``, so a symlink at any segment is
refused instead of followed, and ``O_NONBLOCK``, so a FIFO can't hold the open.
The file at the end must be a regular file with exactly one link: a hard link
inside the folder can name another artifact's file.

Windows has no ``dir_fd`` and no ``O_NOFOLLOW``, so ``PinnedDir.fd`` is None
there and an open joins the name onto a path. There the segment rules are the
guard, plus an ``lstat`` of every segment that refuses a link or a reparse
point before anything opens through it, and again once the file is open,
which refuses a walk whose names changed in between. A folder swapped for a
junction, or the file for a link, and back again between the two checks is
still followed. Comparing the open file's ``fstat`` with its ``lstat`` would
catch that, but on Windows the two calls read different file-information
records, so the walk compares ``lstat`` with ``lstat`` until a Windows test
shows they agree.
"""
from __future__ import annotations

import errno
import ntpath
import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from anton.core.artifacts.internal_files import (
    BACKEND_LOG_FILENAME,
    GENERATION_INPUT_FILES,
    NON_CONTENT_NAMES,
    README_FILENAME,
    REVISIONS_DIRNAME,
)

from cowork.common.paths import (
    O_DIRECTORY,
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    dir_lstat,
    dir_open,
    dir_scandir,
    open_pinned_child,
    read_bounded,
)
from cowork.schemas.artifact_export import OmittedReason
from cowork.services.artifact_identity import opened_artifact_folder

if TYPE_CHECKING:
    from cowork.services.artifacts import ProjectArtifacts

# Housekeeping names, compared casefolded: on a case-insensitive volume
# ``Metadata.JSON`` opens ``metadata.json``. Every one is refused as the first
# segment. anton matches its names on the first segment only, and the
# generation inputs, README.md and backend.log are content inside a subfolder
# (``docs/README.md``), so only the store-owned rest is refused at any depth.
_HOUSEKEEPING_NAMES = frozenset(
    name.casefold() for name in (*NON_CONTENT_NAMES, REVISIONS_DIRNAME)
)
_HOUSEKEEPING_AT_ANY_DEPTH = _HOUSEKEEPING_NAMES - frozenset(
    name.casefold() for name in (*GENERATION_INPUT_FILES, README_FILENAME, BACKEND_LOG_FILENAME)
)

# Names Windows maps to a device whatever follows the first dot, so ``nul.txt``
# can open NUL. Windows reserves the 0 and superscript-digit ports too.
_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{digit}" for digit in "0123456789¹²³"}
    | {f"LPT{digit}" for digit in "0123456789¹²³"}
)

# The deepest relative path an export read follows. Every segment costs a
# directory open and a scan, and one document can name thousands of
# references, so a chain thousands of folders deep would hold the export for
# seconds. The drafts route serves one file per request and has no cap.
MAX_PATH_PARTS = 64

# The most directory entries one export read scans while it matches names. A
# folder the agent fills with entries would otherwise make every segment of
# every reference read all of them. The drafts route reads every entry, so a
# folder of map tiles still serves.
_MAX_SCANNED_ENTRIES = 20_000

_PATH_SEPARATORS = re.compile(r"[/\\]")


class ArtifactFileRefused(Exception):
    """A path the pinned walk refused, with the reason an export records."""

    def __init__(self, reason: OmittedReason, detail: str = "") -> None:
        super().__init__(detail or reason.value)
        self.reason = reason


@dataclass(frozen=True)
class ArtifactFileContent:
    """The bytes of one artifact file and the identity of the inode they came from."""

    data: bytes
    device: int
    inode: int


class PinnedDraftFile(NamedTuple):
    """An open draft file, plus the descriptors that must outlive it.

    Closing ``resources`` closes ``fd`` and the artifact folder it was found in.
    """

    resources: ExitStack
    fd: int
    file_stat: os.stat_result


class _OpenedFile(NamedTuple):
    fd: int
    file_stat: os.stat_result


class CheckedName(NamedTuple):
    """A name the walk opened by a joined path, and the file its ``lstat`` saw."""

    directory: PinnedDir
    name: str
    identity: tuple[int, int]  # (st_dev, st_ino)


@dataclass
class _ScanBudget:
    """The directory entries one walk may still read."""

    remaining: int

    def spend(self) -> None:
        if self.remaining <= 0:
            raise ArtifactFileRefused(OmittedReason.BUDGET, "too many directory entries")
        self.remaining -= 1


def _is_plain_segment(*, segment: str) -> bool:
    """Whether *segment* is one name: not empty, ``.`` or ``..``, with no separator, NUL or drive."""
    return (
        segment not in {"", ".", ".."}
        and not any(character in segment for character in "/\\\x00")
        and not ntpath.splitdrive(segment)[0]
    )


def split_relative_path(*, text: str, skip_dot_segments: bool = False) -> tuple[str, ...]:
    """Split a relative path a request or a document names into plain segments.

    Every relative path goes through here, and every segment through the rule
    the pinned read checks again. Backslashes separate too, so the rules mean
    the same on the Windows desktop and the Linux service. NUL is refused, and
    so is a drive: a UNC or device path is a drive that starts with two
    separators. A leading separator leaves an empty first segment, so it is
    refused with the rest. A ``.`` segment is refused unless
    *skip_dot_segments*: in a document, ``./chart.png`` names ``chart.png``.

    Raises ``ArtifactFileRefused`` with the reason an export records.
    """
    if "\x00" in text:
        raise ArtifactFileRefused(OmittedReason.TRAVERSAL, "NUL")
    if not text:
        raise ArtifactFileRefused(OmittedReason.MISSING, "empty path")
    drive = ntpath.splitdrive(text)[0]
    if drive:
        reason = OmittedReason.UNC_PATH if drive[0] in "/\\" else OmittedReason.TRAVERSAL
        raise ArtifactFileRefused(reason, "drive or UNC path")
    parts = tuple(
        segment
        for segment in _PATH_SEPARATORS.split(text)
        if not (skip_dot_segments and segment == ".")
    )
    if not parts or not all(_is_plain_segment(segment=segment) for segment in parts):
        raise ArtifactFileRefused(OmittedReason.TRAVERSAL, "segment is not a plain name")
    return parts


def relative_file_parts(value: str) -> tuple[str, ...]:
    """Split an untrusted request path into safe, single components.

    ``split_relative_path`` holds the rules; a refusal here is a
    ``ValueError``. The returned strings are passed to descriptor-relative
    opens; no request-derived string is ever joined onto an absolute
    filesystem path.
    """
    try:
        return split_relative_path(text=value)
    except ArtifactFileRefused as exc:
        raise ValueError("invalid path") from exc


def artifact_folder_name(*, source: ProjectArtifacts, folder: Path) -> str:
    """Return the direct child of ``source.base`` selected by resolution.

    Identity lookup indexes a resolved artifacts root, so a normal result can
    be parented by either the declared root or its resolved spelling. Anything
    else must not be translated to ``folder.name``: doing so could turn a grant
    for one resolved folder into a same-named folder under another source.
    """
    base = Path(source.base)
    try:
        parents = {base, base.resolve(strict=False)}
    except OSError as exc:
        raise ArtifactFileRefused(OmittedReason.MISSING, "artifacts root unresolvable") from exc
    if folder.parent not in parents:
        raise ArtifactFileRefused(OmittedReason.MISSING, "folder outside the source")
    name = folder.name
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ArtifactFileRefused(OmittedReason.MISSING, "folder name is not a component")
    return name


def existing_draft_entry_name(
    directory: PinnedDir,
    requested: str,
    *,
    native_case: bool = False,
    budget: _ScanBudget | None = None,
) -> str:
    """Return the pinned directory's own name for a request selector.

    Validation makes ``requested`` a single component, but it still originated
    in an HTTP path.  Compare it with entries already discovered below the
    pinned directory and return ``DirEntry.name`` so no request-derived string
    is ever supplied to ``openat``.  A replacement after the scan remains safe:
    the subsequent descriptor-relative open uses ``O_NOFOLLOW``.

    ``native_case`` also accepts a spelling the filesystem itself accepts. It
    exists because ``resolve_source`` opens by path and so inherits the
    volume's own case rules, and a boundary stricter than the gate behind it
    makes a reported source unsaveable. The filesystem decides, never the
    platform: the requested spelling has to name the same inode as the entry,
    which on a case-sensitive volume it cannot. The name returned is still the
    one the scan produced.

    It refuses no name by itself: the source editor may name ``README.md`` or
    ``prd.md``, so the housekeeping refusal lives in the file walk instead.
    With a ``budget``, every entry read spends one, and running out refuses
    the walk.
    """
    insensitive_matches: list[str] = []
    with dir_scandir(directory) as entries:
        for entry in entries:
            if budget is not None:
                budget.spend()
            if entry.name == requested:
                return entry.name
            if native_case and entry.name.lower() == requested.lower():
                insensitive_matches.append(entry.name)
    # Ambiguous only on a case-sensitive volume, where the exact match above
    # is the only correct answer anyway.
    if len(insensitive_matches) == 1:
        if _names_one_inode(directory, requested, insensitive_matches[0]):
            return insensitive_matches[0]
    raise FileNotFoundError(requested)


def _names_one_inode(directory: PinnedDir, requested: str, discovered: str) -> bool:
    """Whether the filesystem resolves both spellings to the same file.

    Neither stat follows a link, so a symlink whose name differs only in case
    from a real entry compares unequal and is refused rather than matched.
    """
    try:
        probe = dir_lstat(directory, requested)
        found = dir_lstat(directory, discovered)
    except OSError:
        return False
    return (probe.st_dev, probe.st_ino) == (found.st_dev, found.st_ino)


def _windows_rewrites(*, name: str) -> bool:
    """Whether Windows would open some other file than *name* names.

    It strips a trailing dot or space (``metadata.json.`` opens
    ``metadata.json``), reads ``name:stream`` as an alternate data stream and
    maps a reserved name to a device whatever follows its first dot.
    """
    return (
        name[-1:] in {".", " "}
        or ":" in name
        or name.partition(".")[0].rstrip(" ").upper() in _RESERVED_DEVICE_NAMES
    )


def refuse_unsafe_parts(*, parts: tuple[str, ...]) -> None:
    """Refuse a path whose text alone makes it unsafe, before any file call.

    Two leading separators make a UNC path, and ``\\\\?\\`` or ``\\\\.\\`` a
    device path: on Windows even a stat of one can send the user's credentials
    to another host. Each segment must then be one plain name, by the rule
    ``split_relative_path`` splits with, and a name Windows would rewrite is
    refused on every platform.
    """
    if "/".join(parts).replace("\\", "/").startswith("//"):
        raise ArtifactFileRefused(OmittedReason.UNC_PATH, "UNC or device path")
    if not parts:
        raise ArtifactFileRefused(OmittedReason.TRAVERSAL, "empty path")
    for segment in parts:
        if not _is_plain_segment(segment=segment) or _windows_rewrites(name=segment):
            raise ArtifactFileRefused(OmittedReason.TRAVERSAL, "segment is not a plain name")


def refuse_housekeeping(*, parts: tuple[str, ...]) -> None:
    if parts[0].casefold() in _HOUSEKEEPING_NAMES or any(
        segment.casefold() in _HOUSEKEEPING_AT_ANY_DEPTH for segment in parts[1:]
    ):
        raise ArtifactFileRefused(OmittedReason.HOUSEKEEPING, "housekeeping name")


def _disk_name(*, directory: PinnedDir, requested: str, budget: _ScanBudget | None) -> str:
    """The directory's own spelling of *requested*, or a refusal.

    Without a directory descriptor the name is joined onto a path, so a name
    Windows would rewrite is refused there before anything stats it: the
    drafts route doesn't run the text rules, and ``metadata.json.`` would
    otherwise open ``metadata.json``. A name ``lstat`` can't find is missing
    without a scan. The scan still decides every name that exists, so an open
    only ever receives the spelling the directory returned. A name the
    filesystem can't encode, such as one holding a lone surrogate, names
    nothing on disk either.
    """
    if directory.fd is None and _windows_rewrites(name=requested):
        raise ArtifactFileRefused(OmittedReason.TRAVERSAL, "segment is not a plain name")
    try:
        dir_lstat(directory, requested)
        return existing_draft_entry_name(directory, requested, budget=budget)
    except (OSError, ValueError) as exc:
        raise ArtifactFileRefused(OmittedReason.MISSING, "no such entry") from exc


# The reparse-tag bit Windows sets on a reparse point that stands for another
# path (IsReparseTagNameSurrogate).
_NAME_SURROGATE_TAG = 0x20000000


def refuse_link_by_lstat(*, directory: PinnedDir, name: str) -> CheckedName | None:
    """Without a directory descriptor, refuse a link before opening through it.

    ``lstat`` reports the link itself. A symlink is refused as one. A junction
    or mount point is a reparse point whose tag marks it a name surrogate, one
    that stands for another path. Other reparse points, such as cloud-sync
    placeholders and deduplicated files, are files in place and are read.
    Returns what ``lstat`` saw, for ``refuse_a_changed_walk``; with a
    directory descriptor there is nothing to check, and it returns None.
    """
    if directory.fd is not None:
        return None
    try:
        entry_stat = dir_lstat(directory, name)
    except OSError as exc:
        raise ArtifactFileRefused(OmittedReason.MISSING, "no such entry") from exc
    if stat.S_ISLNK(entry_stat.st_mode):
        raise ArtifactFileRefused(OmittedReason.SYMLINK, "symlink")
    is_reparse_point = getattr(entry_stat, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    if is_reparse_point and getattr(entry_stat, "st_reparse_tag", 0) & _NAME_SURROGATE_TAG:
        raise ArtifactFileRefused(OmittedReason.REPARSE_POINT, "link")
    return CheckedName(
        directory=directory, name=name, identity=(entry_stat.st_dev, entry_stat.st_ino)
    )


def refuse_a_changed_walk(*, checked: list[CheckedName]) -> None:
    """Once the file is open, refuse a walk whose names changed since their ``lstat``.

    Each name was opened by a joined path after its ``lstat``, so a folder on
    the way could have turned into a junction, or the file into a link, in
    between. Every name must still be no link and the same file it was.
    """
    for entry in checked:
        current = refuse_link_by_lstat(directory=entry.directory, name=entry.name)
        if current is None or current.identity != entry.identity:
            raise ArtifactFileRefused(OmittedReason.MISSING, "changed during the walk")


def _open_failure(*, directory: PinnedDir, name: str, exc: OSError) -> ArtifactFileRefused:
    """Name the reason an ``O_NOFOLLOW`` open failed.

    Linux answers ``ELOOP`` for a symlink, but macOS answers ``ENOTDIR`` when
    the open also asks for a directory, so a failure that isn't ``ELOOP`` is
    checked with ``lstat`` before it is called missing.
    """
    if exc.errno == errno.ELOOP:
        return ArtifactFileRefused(OmittedReason.SYMLINK, "symlink")
    try:
        if stat.S_ISLNK(dir_lstat(directory, name).st_mode):
            return ArtifactFileRefused(OmittedReason.SYMLINK, "symlink")
    except OSError:
        pass
    return ArtifactFileRefused(OmittedReason.MISSING, "cannot open")


def _descend(*, directory: PinnedDir, name: str, checked: list[CheckedName]) -> PinnedDir:
    """Open direct child directory *name*, returning a ``PinnedDir`` to close.

    Without a directory descriptor, what ``lstat`` saw joins *checked*.
    """
    seen = refuse_link_by_lstat(directory=directory, name=name)
    if seen is not None:
        checked.append(seen)
        return open_pinned_child(directory, name)
    try:
        fd = dir_open(directory, name, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK)
    except OSError as exc:
        raise _open_failure(directory=directory, name=name, exc=exc) from exc
    return PinnedDir(fd, directory.path / name)


def _open_regular_file(
    *,
    folder: PinnedDir,
    parts: tuple[str, ...],
    resources: ExitStack,
    budget: _ScanBudget | None,
) -> _OpenedFile:
    """Walk *parts* down from *folder* and open the regular file at the end.

    The file's descriptor goes on *resources*, which the caller closes. Each
    directory on the way is closed once its child is open, so the walk holds
    at most two descriptors of its own at a time. With a *budget*, every
    directory entry the walk reads spends one; without one it reads them all.
    """
    if not parts:
        raise ArtifactFileRefused(OmittedReason.TRAVERSAL, "empty path")
    refuse_housekeeping(parts=parts)
    checked: list[CheckedName] = []
    current = folder
    try:
        for requested in parts[:-1]:
            child = _descend(
                directory=current,
                name=_disk_name(directory=current, requested=requested, budget=budget),
                checked=checked,
            )
            # Swap before closing, so the finally below still closes the child
            # if closing the parent fails.
            parent, current = current, child
            if parent is not folder:
                parent.close()
        name = _disk_name(directory=current, requested=parts[-1], budget=budget)
        seen = refuse_link_by_lstat(directory=current, name=name)
        if seen is not None:
            checked.append(seen)
        try:
            fd = dir_open(current, name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        except OSError as exc:
            raise _open_failure(directory=current, name=name, exc=exc) from exc
        resources.callback(os.close, fd)
        refuse_a_changed_walk(checked=checked)
    finally:
        if current is not folder:
            current.close()
    try:
        file_stat = os.fstat(fd)
    except OSError as exc:
        raise ArtifactFileRefused(OmittedReason.MISSING, "cannot stat") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ArtifactFileRefused(OmittedReason.MISSING, "not a regular file")
    if file_stat.st_nlink > 1:
        raise ArtifactFileRefused(OmittedReason.SYMLINK, "file has more than one link")
    if file_stat.st_nlink == 0:
        raise ArtifactFileRefused(OmittedReason.MISSING, "file was unlinked")
    return _OpenedFile(fd=fd, file_stat=file_stat)


def open_pinned_draft_file(
    source: ProjectArtifacts, folder: Path, parts: tuple[str, ...]
) -> PinnedDraftFile:
    """Open one regular draft file without following any writable symlink.

    Authorization chose ``source`` and ``folder`` before this call. The source
    retains its server-owned anchor, and ``opened_artifact_folder`` walks from
    that anchor to the artifact with ``O_NOFOLLOW`` on every component. The
    request path is then walked the same way. Returning the ``ExitStack`` keeps
    every descriptor alive until the response has consumed the final file.
    The walk has no depth or scan cap: the route opens one file per request,
    and a folder of thousands of files must still serve.

    Raises ``ArtifactFileRefused`` for every refusal, with every descriptor
    already closed.
    """
    resources = ExitStack()
    try:
        folder_name = artifact_folder_name(source=source, folder=folder)
        try:
            current = resources.enter_context(opened_artifact_folder(source, folder_name))
        except (OSError, ValueError) as exc:
            raise ArtifactFileRefused(OmittedReason.MISSING, "artifact folder") from exc
        opened = _open_regular_file(
            folder=current, parts=parts, resources=resources, budget=None
        )
    except BaseException:
        resources.close()
        raise
    return PinnedDraftFile(resources=resources, fd=opened.fd, file_stat=opened.file_stat)


def read_artifact_file_entry(
    *, folder: PinnedDir, parts: tuple[str, ...], max_bytes: int
) -> ArtifactFileContent:
    """Read the regular file at *parts* below *folder*, at most *max_bytes* long.

    Raises ``ArtifactFileRefused`` with the reason: a path that isn't plain
    names is ``TRAVERSAL``, a housekeeping name as the first segment or a
    store-owned one at any depth is ``HOUSEKEEPING``, a symlink
    or a second hard link is ``SYMLINK``, a UNC or device path is
    ``UNC_PATH``, a junction or other reparse point that stands for another
    path is ``REPARSE_POINT``, anything absent or not a regular file is
    ``MISSING``, a file over *max_bytes* is ``TOO_LARGE``,
    and a path over ``MAX_PATH_PARTS`` segments or a walk that would read
    over ``_MAX_SCANNED_ENTRIES`` directory entries is ``BUDGET``. The read
    stops one byte past *max_bytes*, so a file that grows after the size
    check is still refused. Every descriptor is closed before it returns or
    raises.
    """
    refuse_unsafe_parts(parts=parts)
    if len(parts) > MAX_PATH_PARTS:
        raise ArtifactFileRefused(OmittedReason.BUDGET, "path nests too deep")
    with ExitStack() as resources:
        opened = _open_regular_file(
            folder=folder,
            parts=parts,
            resources=resources,
            budget=_ScanBudget(remaining=_MAX_SCANNED_ENTRIES),
        )
        if opened.file_stat.st_size > max_bytes:
            raise ArtifactFileRefused(OmittedReason.TOO_LARGE, "over the size cap")
        try:
            data = b"".join(read_bounded(opened.fd, max_bytes + 1))
        except OSError as exc:
            raise ArtifactFileRefused(OmittedReason.MISSING, "read failed") from exc
    if len(data) > max_bytes:
        raise ArtifactFileRefused(OmittedReason.TOO_LARGE, "grew past the size cap")
    return ArtifactFileContent(
        data=data, device=opened.file_stat.st_dev, inode=opened.file_stat.st_ino
    )


def read_artifact_file(*, folder: PinnedDir, parts: tuple[str, ...], max_bytes: int) -> bytes:
    """The bytes of :func:`read_artifact_file_entry`, under the same refusals."""
    return read_artifact_file_entry(folder=folder, parts=parts, max_bytes=max_bytes).data
