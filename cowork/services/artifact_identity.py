"""Canonical artifact identity and project-scoped resolution.

Artifact folders can move between projects and published URLs can change.  The
32-hex ``id`` stored in ``metadata.json`` is the identity that survives those
changes; filesystem paths and the slug's eight-character ``id[:8]`` suffix are
compatibility details.

The widening rules live in ``anton.core.artifacts.models``: anton derives them
in memory on every read, this service persists them. Both sides must agree on
the value, so the derivation is imported rather than reimplemented.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import secrets
import stat
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, NamedTuple

# Re-exported: `artifact_key()` is the canonical `artifact/<uuid>` key the
# comments API, the auth rules and the upload lambda all agree on. Callers here
# import it from this module because that is where the rest of the identity
# vocabulary lives.
from anton.core.artifacts.models import (
    artifact_key as artifact_key,
    canonical_artifact_id,
    extend_legacy_id,
    resolve_artifact_id,
)
from cowork.code_sandbox import anchored_below_a_trusted_root
from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    dir_atomic_write,
    dir_open,
    dir_scandir,
    dir_stat,
    open_pinned_child,
    pinned_dir,
)
from cowork.services.artifact_locks import LOCKS_DIRNAME

if TYPE_CHECKING:
    from cowork.services.artifacts import ProjectArtifacts


logger = logging.getLogger(__name__)


class ArtifactIdentityConflict(RuntimeError):
    """More than one scoped folder claims the same artifact identity."""


def _component(value: str) -> str:
    """Validate one filesystem component before handing it to ``openat``."""
    value = str(value)
    # ``basename`` is both the exact transformation wanted here and a
    # path-traversal sanitizer recognized by SAST. Return its value, rather than
    # merely comparing against it, so taint cannot continue into joinpath/openat.
    name = os.path.basename(value)
    if not value or value in (".", "..") or name != value:
        raise ValueError("Artifact path component is invalid")
    return name


def _normalized_path(path: Path) -> str:
    """Normalize lexically; resolving here would follow the link we must reject."""
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


@contextmanager
def opened_artifact_root(source: "ProjectArtifacts") -> Iterator[PinnedDir]:
    """Pin an authorized artifacts root without following writable links.

    Resolvers created by :mod:`artifact_roots` carry a server-owned project
    directory plus the relative storage components.  Opening the project
    directory from the server's trusted root, and those components below it,
    one by one with ``O_NOFOLLOW`` closes both the ordinary symlink escape and
    the check/swap/use race (``_opened_from_trusted_root``).  Explicit
    legacy/local sources have no anchor and pin their base itself with
    ``O_NOFOLLOW`` instead.
    """
    base = Path(source.base)
    anchor_value = getattr(source, "trusted_anchor", None)
    parts = tuple(getattr(source, "root_parts", ()) or ())
    if anchor_value is None:
        if parts:
            raise ValueError("Artifact root parts require a trusted anchor")
        with pinned_dir(base, nofollow_base=True) as root:
            yield root
        return

    anchor = Path(anchor_value)
    parts = tuple(_component(part) for part in parts)
    normalized_anchor_parts = Path(_normalized_path(anchor)).parts
    normalized_base_parts = Path(_normalized_path(base)).parts
    if not parts or normalized_base_parts != (*normalized_anchor_parts, *parts):
        raise ValueError("Artifact root does not match its trusted anchor")

    # At no point is the already-authorized path re-walked as one joined string.
    with _opened_from_trusted_root(anchor, *parts) as root:
        yield root


@contextmanager
def _opened_from_trusted_root(anchor: Path, *names: str) -> Iterator[PinnedDir]:
    """Pin ``anchor/<names...>``, opening every folder below the server's
    trusted root that holds ``anchor`` relative to descriptors with
    O_NOFOLLOW.

    Every folder between that root and ``anchor`` is storage agent code could
    write before the code sandbox: a person's partition folder in the shared
    root, its ``projects`` store, the project folder. Any of them may be a
    link, and a partition folder linked to another person's would hand over
    that person's projects. The trusted root itself
    (code_sandbox.anchored_below_a_trusted_root) is opened as configured, a
    link included. An anchor outside every trusted root, such as an adopted
    folder, trusts its parent instead.
    """
    lexical = Path(os.path.abspath(anchor))
    anchored = anchored_below_a_trusted_root(lexical)
    if anchored is None:
        start, below = lexical.parent, (lexical.name,)
    else:
        # The root as ``anchor`` spells it, so every path opened below keeps
        # the form the caller passed in.
        start = Path(*lexical.parts[: len(lexical.parts) - len(anchored.names)])
        below = anchored.names
    with pinned_dir(start, nofollow_base=anchored is None) as top, ExitStack() as opened:
        folder = top
        for name in (*below, *names):
            try:
                folder = opened.enter_context(_opened_child_directory(folder, name))
            except OSError as error:
                # A missing folder is ordinary (a project with no artifacts yet);
                # a link or a file is not, and the routes answer it as missing.
                if isinstance(error, NotADirectoryError) or error.errno == errno.ELOOP:
                    logger.warning(
                        "artifacts: a folder on the way to an artifacts folder is a link "
                        "or not a folder, so the artifacts below it are left out",
                        extra={"event": "artifact_folder_refused", "reason": "linked_folder"},
                    )
                raise
        yield folder


@contextmanager
def _opened_child_directory(parent: PinnedDir, name: str) -> Iterator[PinnedDir]:
    name = _component(name)
    # This explicit no-follow stat gives the Windows path fallback the same
    # discovery semantics.  POSIX security comes from the subsequent openat,
    # so a replacement after this probe is still refused rather than followed.
    if not stat.S_ISDIR(dir_stat(parent, name, follow_symlinks=False).st_mode):
        raise NotADirectoryError(name)
    child = open_pinned_child(parent, name)
    try:
        yield child
    finally:
        child.close()


@contextmanager
def opened_artifact_folder(
    source: "ProjectArtifacts", folder_name: str
) -> Iterator[PinnedDir]:
    """Pin one direct, non-symlink artifact folder below ``source``."""
    with opened_artifact_root(source) as root:
        with _opened_child_directory(root, folder_name) as folder:
            yield folder


def _resolved_id(metadata: dict, folder: Path) -> str:
    """The canonical id this metadata document should carry.

    ``stableId`` is the retired second field of the two-field era; where it
    exists it already keyed published versions, auth rules and comment threads,
    so it decides for a record whose ``id`` is still the short form.
    """
    raw_id = str(metadata.get("id") or "")
    inherited = str(metadata.get("stableId") or "")
    created_at = str(metadata.get("createdAt") or "")
    try:
        if raw_id or inherited:
            return resolve_artifact_id(raw_id, inherited, created_at)
        # No identity field at all — a record anton itself would refuse to load.
        # Derive one from whatever names the folder, deterministically, so the
        # artifact still gets an identity instead of vanishing from every list.
        # Not routed through `resolve_artifact_id`: a slug is not an id, so it
        # must not be judged against the shapes a real id is allowed to take.
        return extend_legacy_id(str(metadata.get("slug") or folder.name), created_at)
    except (ValueError, TypeError, AttributeError) as exc:
        # A present-but-invalid identity is corruption, not a legacy record.
        # Replacing it would silently detach published versions and existing
        # comment threads from the artifact.
        raise ValueError("Artifact identity is invalid") from exc


def _is_migrated(metadata: dict, resolved: str) -> bool:
    return metadata.get("id") == resolved and not metadata.get("stableId")


class _NotARegularFile(OSError):
    """A document an artifact folder names is not a regular file, such as a
    FIFO an agent left in its place. It holds no id (_holds_no_id)."""


def _read_no_follow(folder: PinnedDir, name: str) -> str:
    """Read a direct child, refusing to follow it if it is a symlink.

    An artifact folder is agent-writable, so `metadata.json` can be replaced
    with a link to something outside the artifact. Reading through it would turn
    identity resolution into a file read of the attacker's choosing, and the
    caller here treats an unreadable identity as "skip this folder", which is
    the correct outcome for a link too. `O_NOFOLLOW` raises `ELOOP` (an
    `OSError`) in that case, which every caller of this module already handles.

    The open never waits on a FIFO (`O_NONBLOCK`), and anything but a regular
    file raises _NotARegularFile, so a FIFO in place of the document cannot
    hold a file store thread until a writer appears.
    """
    fd = dir_open(folder, _component(name), os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _NotARegularFile(errno.EINVAL, "Artifact document is not a regular file", name)
        stream = os.fdopen(fd, "r", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise
    with stream:
        return stream.read()


def _load_metadata(*, folder: PinnedDir, name: str = "metadata.json") -> dict:
    """Parse the folder's metadata document. A document nested too deeply
    to parse, or one that is not a JSON object, raises ValueError, which
    every caller treats as a folder holding no id."""
    try:
        metadata = json.loads(_read_no_follow(folder, name))
    except RecursionError as exc:
        raise ValueError("Artifact metadata nests too deeply") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Artifact metadata is not an object")
    return metadata


def read_full_id(folder: PinnedDir) -> tuple[str, dict]:
    """Read and normalize one artifact identity without mutating its metadata.

    Artifact listing is a read operation. Legacy metadata still needs to expose
    the same deterministic full id as :func:`ensure_full_id`, but persisting that
    widening makes an HTTP GET reach ``os.replace`` and ``os.utime``. This entry
    point has a deliberately separate call graph: it reads through the pinned
    descriptor, derives the canonical id in memory, and returns the normalized
    metadata that callers historically received after a successful migration.
    """
    metadata = _load_metadata(folder=folder)
    resolved = _resolved_id(metadata, folder.path)
    if _is_migrated(metadata, resolved):
        return resolved, metadata
    normalized = {key: value for key, value in metadata.items() if key != "stableId"}
    normalized["id"] = resolved
    return resolved, normalized


def _replaced_mode(folder: PinnedDir, name: str) -> int | None:
    """The permission bits of the regular file at *name*, which a write replaces.

    The replacement takes them, so whoever could read the old document can
    read the new one; a document written for the first time stays private.
    Windows keeps no such bits beyond read-only, so it gets None.
    """
    if folder.fd is None:
        return None
    try:
        existing = dir_stat(folder, name, follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISREG(existing.st_mode):
        return None
    return stat.S_IMODE(existing.st_mode) & 0o777


def _atomic_json(folder: PinnedDir, name: str, payload: dict) -> None:
    """Replace the document *name* with *payload*. A symlinked
    `metadata.json` is replaced by the real document, never written through
    (dir_atomic_write)."""
    name = _component(name)
    dir_atomic_write(
        folder,
        name,
        content=(json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        mode=_replaced_mode(folder, name),
    )


def _ensure_full_id(
    folder: PinnedDir, metadata: dict | None = None
) -> tuple[str, dict]:
    """Descriptor-relative implementation for :func:`ensure_full_id`."""
    name = "metadata.json"
    path = folder.path / name
    if metadata is None:
        metadata = _load_metadata(folder=folder, name=name)
    resolved = _resolved_id(metadata, folder.path)
    if _is_migrated(metadata, resolved):
        return resolved, metadata

    # The caller may have loaded metadata before another subsystem updated it.
    # Merge into the latest durable document instead of writing the stale
    # snapshot back over unrelated metadata fields.
    try:
        latest = _load_metadata(folder=folder, name=name)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Artifact metadata is unreadable") from exc
    resolved = _resolved_id(latest, folder.path)
    if _is_migrated(latest, resolved):
        return resolved, latest
    updated = {key: value for key, value in latest.items() if key != "stableId"}
    updated["id"] = resolved
    # metadata.json's mtime is a recency signal: the artifacts list
    # (cowork.services.artifacts.list_artifacts) orders folders by it, newest
    # first. Widening an id changes no user-visible content, so restore the
    # original timestamps or the first card/index build after an upgrade
    # would move every legacy artifact to the top as just updated.
    try:
        before = dir_stat(folder, name, follow_symlinks=False)
    except OSError:
        before = None
    try:
        _atomic_json(folder, name, updated)
    except OSError:
        # Persisting is an optimization, not the contract: the id is derived
        # deterministically, so a read-only or full artifacts root still
        # resolves to the same value on every read — exactly what anton does.
        # Propagating would drop the artifact from every listing (both callers
        # treat an identity error as "skip this folder"), which reads as a
        # deletion of files that are sitting right there.
        logger.warning("Could not persist widened artifact id for %s", folder.path, exc_info=True)
        return resolved, updated
    if before is not None:
        try:
            # The name is resolved against the still-pinned folder.  Combined
            # with `follow_symlinks=False`, neither a directory swap nor a
            # planted metadata link can steer the timestamp restoration.
            if folder.fd is not None:
                os.utime(
                    name,
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                    dir_fd=folder.fd,
                    follow_symlinks=False,
                )
            else:
                os.utime(
                    path,
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                    follow_symlinks=False,
                )
        except OSError:
            pass
    return resolved, updated


def ensure_full_id(
    folder: Path,
    metadata: dict | None = None,
    *,
    _pinned: PinnedDir | None = None,
) -> tuple[str, dict]:
    """Return the folder's canonical 32-hex id, persisting a legacy widening.

    The widening is deterministic, so concurrent readers choose the same value
    even before either atomic metadata write wins.
    """
    # Internal pinned callers stay on this public function so instrumentation
    # and the cache's one-read contract continue to observe every identity
    # load. The handle remains an implementation detail, not a second API.
    if _pinned is not None:
        return _ensure_full_id(_pinned, metadata)

    # Pinning the folder itself makes supplied metadata safe too: even when no
    # read is necessary, a symlink folder is refused before a legacy migration
    # can write through it.
    with pinned_dir(folder, nofollow_base=True) as pinned:
        return _ensure_full_id(pinned, metadata)


DirectoryClock = tuple[int, int, int, int]


def _directory_version(base: PinnedDir) -> DirectoryClock:
    """An invalidation clock tied to the directory inode we actually opened."""
    try:
        value = os.fstat(base.fd) if base.fd is not None else base.path.stat(
            follow_symlinks=False
        )
    except OSError:
        return (0, 0, 0, 0)
    # Device + inode prevent a replacement container with coincidentally equal
    # timestamps from reusing an index built for the directory it displaced.
    return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)


# One entry per artifacts root. A hosted replica serves every project of every
# organization on it, and each legacy conversation root is a root of its own,
# so a cap near the number of projects evicted entries before their next use.
_IDENTITY_INDEX_LIMIT = 4096
_identity_indexes: OrderedDict[
    tuple[str, DirectoryClock], dict[str, tuple[str, ...]]
] = OrderedDict()
_identity_indexes_lock = RLock()


def _holds_no_id(exc: OSError) -> bool:
    """Whether a read error means the entry holds no artifact id: it is gone,
    is not a directory, is a link, which no reader follows (ELOOP from
    ``O_NOFOLLOW``), or names a document that is not a regular file
    (_read_no_follow). Any other error leaves unknown what the entry holds."""
    return (
        isinstance(exc, (FileNotFoundError, NotADirectoryError, _NotARegularFile))
        or exc.errno == errno.ELOOP
    )


def may_not_read(*, exc: OSError) -> bool:
    """Whether a read error says the server may not read the entry at all
    (EACCES, EPERM). The server can never serve what sits below such an
    entry, so a strict count leaves it out rather than failing on it."""
    return exc.errno in (errno.EACCES, errno.EPERM)


def _strictly_holds_no_id(*, exc: OSError) -> bool:
    """_holds_no_id, plus a folder the server may not read (may_not_read):
    it is no second folder for a row, since no lookup can serve it. A strict
    read still raises on EIO, ESTALE and the like, which may pass."""
    return _holds_no_id(exc) or may_not_read(exc=exc)


def _build_identity_index(
    base: PinnedDir, *, widen: bool = True, strict: bool = False
) -> dict[str, tuple[str, ...]]:
    """Build an id-to-folder-name index below one pinned container. With
    ``widen`` off, a legacy short id is widened in memory only, and nothing is
    written.

    A folder that is gone, is a link, has no metadata file, or holds one that
    does not parse holds no id. A folder that fails to read for any other
    reason is left out too, unless ``strict``: then the error raises, for a
    caller that records which folders hold an id and must not take a folder
    it could not read for one holding none. A folder the server may not read
    at all (EACCES, EPERM) holds no id even then (_strictly_holds_no_id)."""
    matches: dict[str, list[str]] = {}
    try:
        with dir_scandir(base) as entries:
            folder_names = []
            for entry in entries:
                try:
                    if (
                        not entry.is_symlink()
                        and entry.is_dir(follow_symlinks=False)
                    ):
                        folder_names.append(entry.name)
                except OSError as exc:
                    if strict and not _strictly_holds_no_id(exc=exc):
                        raise
                    continue
    except OSError as exc:
        if strict and not _strictly_holds_no_id(exc=exc):
            raise
        return {}
    # DirEntry's no-follow probe is an early filter; each openat below is the
    # race-safe decision if an entry changes after the scan.
    for folder_name in sorted(folder_names):
        try:
            with _opened_child_directory(base, folder_name) as folder:
                metadata_stat = dir_stat(
                    folder, "metadata.json", follow_symlinks=False
                )
                if not stat.S_ISREG(metadata_stat.st_mode):
                    continue
                if widen:
                    found, _metadata = ensure_full_id(folder.path, _pinned=folder)
                else:
                    found, _metadata = read_full_id(folder)
        except (ValueError, json.JSONDecodeError):
            continue
        except OSError as exc:
            if strict and not _strictly_holds_no_id(exc=exc):
                raise
            continue
        matches.setdefault(found, []).append(folder_name)
    return {artifact_id: tuple(folders) for artifact_id, folders in matches.items()}


def _identity_index(
    base_value: str,
    directory_clock: DirectoryClock,
    base: PinnedDir,
    *,
    force_refresh: bool = False,
) -> tuple[dict[str, tuple[str, ...]], bool]:
    """Return one bounded LRU entry and whether it was already cached.

    The directory clock invalidates ordinary artifact creation, deletion and
    moves. A hit still revalidates the target metadata below; this cache only
    avoids rereading every unrelated artifact on each workspace request.
    """
    key = (base_value, directory_clock)
    if not force_refresh:
        with _identity_indexes_lock:
            cached = _identity_indexes.pop(key, None)
            if cached is not None:
                _identity_indexes[key] = cached
                return cached, True

    index = _build_identity_index(base)
    with _identity_indexes_lock:
        for stale_key in tuple(_identity_indexes):
            if stale_key[0] == base_value and stale_key != key:
                _identity_indexes.pop(stale_key, None)
        _identity_indexes[key] = index
        _identity_indexes.move_to_end(key)
        while len(_identity_indexes) > _IDENTITY_INDEX_LIMIT:
            _identity_indexes.popitem(last=False)
    return index, False


def _clear_identity_indexes() -> None:
    """Clear the process index (used by tests and explicit refresh paths)."""
    with _identity_indexes_lock:
        _identity_indexes.clear()


def _refresh_identity_index(
    cache_key: tuple[str, DirectoryClock], base: PinnedDir
) -> dict[str, tuple[str, ...]]:
    return _identity_index(*cache_key, base, force_refresh=True)[0]


def _validated_index_records(
    base: PinnedDir,
    artifact_id: str,
    folder_names: tuple[str, ...],
) -> tuple[tuple[Path, dict], ...]:
    records: list[tuple[Path, dict]] = []
    for folder_name in folder_names:
        try:
            with _opened_child_directory(base, folder_name) as folder:
                found, metadata = ensure_full_id(folder.path, _pinned=folder)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if found == artifact_id:
            records.append((base.path / folder_name, metadata))
    return tuple(records)


def _root_still_current(source: "ProjectArtifacts", opened: PinnedDir) -> bool:
    """Reopen the source and ensure its name still denotes the pinned inode."""
    try:
        with opened_artifact_root(source) as current:
            if opened.fd is not None and current.fd is not None:
                before = os.fstat(opened.fd)
                after = os.fstat(current.fd)
            else:
                before = opened.path.stat(follow_symlinks=False)
                after = current.path.stat(follow_symlinks=False)
    except OSError:
        return False
    return (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def _indexed_artifacts(
    source: "ProjectArtifacts", artifact_id: str
) -> tuple[tuple[Path, dict], ...]:
    """Revalidate one identity copied from this source's own index.

    ``artifact_id`` is deliberately not a request value.  Public resolution
    first snapshots the identities found on disk, compares the requested UUID
    with those values in memory, and passes the matching *indexed* spelling
    here.  Keeping that distinction structural prevents a request parameter
    from selecting a directory name inside this filesystem-facing helper.
    """
    try:
        with opened_artifact_root(source) as base:
            # The lexical key deliberately does not resolve symlinks.  The
            # descriptor is the authority, and its device/inode are part of the
            # clock, so a replaced root cannot inherit another root's cache.
            cache_key = (_normalized_path(base.path), _directory_version(base))
            index, was_cached = _identity_index(*cache_key, base)
            folders = index.get(artifact_id)
            if folders is not None:
                records = _validated_index_records(base, artifact_id, folders)
                if len(records) == len(folders):
                    return records if _root_still_current(source, base) else ()

            # A freshly-built index is already authoritative for this point in
            # time. Only cached misses or stale positives need one rebuild.
            if not was_cached:
                return ()

            # Writing metadata inside an already-existing artifact folder does
            # not necessarily tick the container. A cached miss or failed
            # target revalidation therefore rebuilds once.
            refreshed = _refresh_identity_index(cache_key, base)
            records = _validated_index_records(
                base, artifact_id, refreshed.get(artifact_id, ())
            )
            return records if _root_still_current(source, base) else ()
    except (OSError, ValueError):
        # Missing roots and every refused symlink have the same externally
        # observable result: this authorized source contains no such artifact.
        return ()


def _indexed_identities(
    source: "ProjectArtifacts", *, force_refresh: bool = False
) -> tuple[tuple[str, ...], bool]:
    """Snapshot server-discovered identities without accepting a lookup key.

    The boolean says whether the ordinary read reused a cache entry.  Callers
    use it to preserve the existing one-refresh-on-miss behavior for metadata
    created inside an already-existing artifact folder (which need not change
    the artifacts directory's mtime).
    """
    try:
        with opened_artifact_root(source) as base:
            cache_key = (_normalized_path(base.path), _directory_version(base))
            index, was_cached = _identity_index(
                *cache_key,
                base,
                force_refresh=force_refresh,
            )
            if not _root_still_current(source, base):
                return (), was_cached
            return tuple(index), was_cached
    except (OSError, ValueError):
        return (), False


def _matching_indexed_identity(
    indexed_identities: tuple[str, ...], wanted: str
) -> str | None:
    """Return the server-derived spelling equal to ``wanted``.

    Returning ``known`` rather than ``wanted`` is the taint boundary: the
    request UUID participates only in a constant-time in-memory comparison and
    is never handed to root discovery, directory opening, or path creation.
    """
    for known in indexed_identities:
        if secrets.compare_digest(known, wanted):
            return known
    return None


def _resolved_from_source(
    source: "ProjectArtifacts", wanted: str
) -> tuple[tuple[Path, dict], ...]:
    """Match a request identity to an index before filesystem revalidation."""
    identities, was_cached = _indexed_identities(source)
    known = _matching_indexed_identity(identities, wanted)
    if known is not None:
        return _indexed_artifacts(source, known)

    # A metadata file appearing inside an existing folder may not update the
    # root directory clock.  Refresh a cached miss once, just as the previous
    # target-keyed implementation did, then still pass only the index's value
    # into the filesystem-facing validator.
    if was_cached:
        identities, _ = _indexed_identities(source, force_refresh=True)
        known = _matching_indexed_identity(identities, wanted)
        if known is not None:
            return _indexed_artifacts(source, known)
    return ()


def identities_below(source: "ProjectArtifacts") -> dict[str, tuple[str, ...]]:
    """Every id found in ``source``'s root, with the folders holding it, read
    without writing. Empty for a missing root, a link in its chain, or a root
    the server may not read. Any other error opening the root or reading a
    folder raises (the index's ``strict``), so the caller never takes a
    folder it could not read for one holding no id."""
    try:
        with opened_artifact_root(source) as base:
            return _build_identity_index(base, widen=False, strict=True)
    except OSError as exc:
        if _strictly_holds_no_id(exc=exc):
            return {}
        raise
    except ValueError:
        return {}


class NamedArtifactFolders(NamedTuple):
    """What artifact_folders_named found among the names it was given. A name
    in neither set holds no artifact folder."""

    # Plain folders holding a plain metadata.json.
    found: set[str]
    # Names it could not read: a read error other than a missing entry or a
    # link (_holds_no_id). Every name when opening ``base`` fails that way.
    unreadable: set[str]


def artifact_folders_named(*, base: Path, slugs: set[str]) -> NamedArtifactFolders:
    """The names among ``slugs`` that are artifact folders directly below
    ``base``: plain folders holding a plain ``metadata.json``. A name that
    isn't one path component, a link, a missing entry or a folder without that
    file holds none, and so does every name when ``base`` is missing or a
    folder on the way to it is a link (``_opened_from_trusted_root``). Stats
    only the named folders, never the whole root."""
    found: set[str] = set()
    unreadable: set[str] = set()
    try:
        with _opened_from_trusted_root(Path(base)) as root:
            for slug in slugs:
                try:
                    with _opened_child_directory(root, slug) as folder:
                        if stat.S_ISREG(dir_stat(folder, "metadata.json", follow_symlinks=False).st_mode):
                            found.add(_component(slug))
                except ValueError:
                    continue
                except OSError as exc:
                    if not _holds_no_id(exc):
                        unreadable.add(slug)
    except OSError as exc:
        return NamedArtifactFolders(found=set(), unreadable=set() if _holds_no_id(exc) else set(slugs))
    return NamedArtifactFolders(found=found, unreadable=unreadable)


class ArtifactListing(NamedTuple):
    """What artifact_folders_below found directly below a folder. An entry
    in neither set holds no artifact folder."""

    # Folders holding a metadata.json file. A link to the folder or to the
    # file is followed.
    found: frozenset[str]
    # Entries whose stat, the folder's own or its metadata.json's, failed
    # with an error other than a missing entry, a name that is not a folder,
    # or a link loop (_holds_no_id).
    unreadable: frozenset[str]
    # The error listing the folder itself failed with, when it was such an
    # error. No entry below it was seen, so both sets are empty.
    base_error: OSError | None = None


def artifact_folders_below(*, base: Path) -> ArtifactListing:
    """List the artifact folders directly below ``base``, a project's
    artifacts folder, following links as a full listing always has: each
    entry is stat'ed, then its ``metadata.json``. A missing ``base``, or one
    that is not a folder, holds none. The project's lock folder
    (``LOCKS_DIRNAME``) is never an artifact, even when agent code put a
    ``metadata.json`` there, so it is in neither set. The listing's errors
    are sorted by errno rather than left to ``Path.is_dir``, whose answer to
    an EIO differs between Python versions (it raises on 3.12 and 3.13, and
    answers False from 3.14). ENOENT, ENOTDIR and ELOOP mean the entry holds
    no artifact folder; any other error leaves it in ``unreadable``, or, for
    ``base`` itself, in ``base_error``, so a caller never takes an entry it
    could not read for one that is gone."""
    base = Path(base)
    try:
        with os.scandir(base) as entries:
            names = sorted(entry.name for entry in entries if entry.name != LOCKS_DIRNAME)
    except OSError as exc:
        return ArtifactListing(found=frozenset(), unreadable=frozenset(), base_error=None if _holds_no_id(exc) else exc)
    found: set[str] = set()
    unreadable: set[str] = set()
    for name in names:
        folder = base / name
        try:
            if not stat.S_ISDIR(os.stat(folder).st_mode):
                continue
            if not stat.S_ISREG(os.stat(folder / "metadata.json").st_mode):
                continue
        except OSError as exc:
            if not _holds_no_id(exc):
                unreadable.add(name)
            continue
        found.add(name)
    return ArtifactListing(found=frozenset(found), unreadable=frozenset(unreadable))


class ListedIds(NamedTuple):
    """What artifact_ids_below read from the folders it was given, or
    listed_folder_ids from a full listing. A folder in neither holds no id."""

    # The id each folder holding one holds.
    ids: dict[str, str]
    # The folders it could not read: opening one or reading its id failed
    # with an error other than a missing entry, a link or a document that is
    # not a regular file (_holds_no_id). Every folder when opening ``base``
    # fails that way. From a full listing, also the entries the listing could
    # not stat (ArtifactListing.unreadable), less those the id reader refuses
    # as links (listed_folder_ids).
    unreadable: frozenset[str]
    # Whether opening or listing ``base`` itself failed that way, so the
    # folders below it were not all read, even when ``unreadable`` names none.
    # Never set for a ``base`` the id reader refuses, such as a link
    # (listed_folder_ids).
    base_unreadable: bool = False


def artifact_ids_below(*, base: Path, slugs) -> ListedIds:
    """The id each of ``slugs``' folders directly below ``base`` holds, read
    without writing. A folder that is gone, is a link, or holds no metadata
    file or one that does not parse holds no id, and so does every folder
    when a folder on the way to ``base`` is missing or is a link
    (``_opened_from_trusted_root``). Any other read error leaves the folder
    in ``unreadable``, so a caller never takes a folder it could not read
    for one that is gone or holds no id."""
    read = _ids_below_pinned_base(base=base, slugs=slugs)
    return read if read is not None else ListedIds(ids={}, unreadable=frozenset())


def _ids_below_pinned_base(*, base: Path, slugs) -> ListedIds | None:
    """artifact_ids_below's read, or None when ``base`` itself holds no id:
    it is missing, or a folder on the way to it is a link
    (``_opened_from_trusted_root``)."""
    ids: dict[str, str] = {}
    unreadable: set[str] = set()
    try:
        with _opened_from_trusted_root(Path(base)) as root:
            for slug in slugs:
                try:
                    with _opened_child_directory(root, slug) as folder:
                        ids[slug] = read_full_id(folder)[0]
                except ValueError:
                    # json.JSONDecodeError is a ValueError too.
                    continue
                except OSError as exc:
                    if not _holds_no_id(exc):
                        unreadable.add(slug)
    except OSError as exc:
        if _holds_no_id(exc):
            return None
        return ListedIds(ids={}, unreadable=frozenset(slugs), base_unreadable=True)
    return ListedIds(ids=ids, unreadable=frozenset(unreadable))


def listed_folder_ids(*, base: Path, listing: ArtifactListing) -> ListedIds:
    """The id each folder a full listing of ``base`` found holds, read
    without writing (artifact_ids_below), with the entries the listing could
    not stat among the unreadable folders, and ``base_unreadable`` when it
    could not list a plain ``base`` at all.

    The listing follows links and the id reader never does, so an entry the
    reader refuses as holding no id is left out of the unreadable folders,
    however the listing's stat of it failed: every entry when ``base`` is
    missing or a folder on the way to it is a link, and an entry that is a
    link, or whose ``metadata.json`` is one (_refused_as_a_link). Such an
    entry can never give an id, so it never asks for a self-heal. The same
    holds for ``base`` itself: a link the listing could not list, such as one
    to a folder the server may not read, is refused before its listing's
    error counts, so only a plain ``base`` sets ``base_unreadable``."""
    read = _ids_below_pinned_base(base=base, slugs=listing.found)
    if read is None:
        return ListedIds(ids={}, unreadable=frozenset())
    if listing.base_error is not None:
        return ListedIds(ids={}, unreadable=frozenset(), base_unreadable=True)
    unstatted = frozenset(name for name in listing.unreadable if not _refused_as_a_link(base=Path(base), name=name))
    return read._replace(unreadable=read.unreadable | unstatted)


def _refused_as_a_link(*, base: Path, name: str) -> bool:
    """Whether the entry ``name`` directly below ``base``, or its
    ``metadata.json``, is a link, which the id reader refuses as holding no
    id (``O_NOFOLLOW``). An entry whose own lstat fails is not known to be
    one."""
    try:
        if stat.S_ISLNK(os.lstat(base / name).st_mode):
            return True
        return stat.S_ISLNK(os.lstat(base / name / "metadata.json").st_mode)
    except OSError:
        return False


class FolderCheck(Enum):
    """What opening the folder a location row names found."""

    #: The folder holds the requested id.
    HOLDS = "holds"
    #: The row is stale: the folder or its metadata file is gone, is not a
    #: plain folder or file, or the folder holds another id.
    STALE = "stale"
    #: The folder could not be read: any other read error, a document that
    #: does not parse, or a root replaced while it was open.
    UNREADABLE = "unreadable"


class RecordedFolder(NamedTuple):
    """The folder a location row names and its metadata, when `check` is
    HOLDS."""

    check: FolderCheck
    folder: Path | None = None
    metadata: dict | None = None


def resolve_recorded_folder(
    source: "ProjectArtifacts", slug: str, artifact_id: str
) -> RecordedFolder:
    """Open the folder a location row names and check it still holds
    ``artifact_id``.

    ``source`` is a root the caller may already read, and ``slug`` comes from
    a row the server wrote from a folder name it found or made; it is opened
    as one validated, no-follow component below the pinned root. The request's
    id is only compared with the folder's own. A failed read is UNREADABLE,
    never STALE, so a transient file store error does not move or drop the
    row.
    """
    wanted = canonical_artifact_id(artifact_id)
    try:
        with opened_artifact_root(source) as base:
            with _opened_child_directory(base, slug) as folder:
                found, metadata = ensure_full_id(folder.path, _pinned=folder)
            if not secrets.compare_digest(found, wanted):
                return RecordedFolder(FolderCheck.STALE)
            if not _root_still_current(source, base):
                return RecordedFolder(FolderCheck.UNREADABLE)
            return RecordedFolder(FolderCheck.HOLDS, base.path / _component(slug), metadata)
    except OSError as exc:
        return RecordedFolder(FolderCheck.STALE if _holds_no_id(exc) else FolderCheck.UNREADABLE)
    except ValueError:
        # json.JSONDecodeError is a ValueError too.
        return RecordedFolder(FolderCheck.UNREADABLE)


class HeldFolder(NamedTuple):
    """A folder holding an artifact id, the root it was found in, and its
    metadata."""

    source: "ProjectArtifacts"
    folder: Path
    metadata: dict


def artifact_folders_holding(
    sources: list["ProjectArtifacts"], *, artifact_id: str
) -> list[HeldFolder]:
    """Every folder in ``sources`` that holds ``artifact_id``, in the order of
    ``sources``. The roots must already be authorized for the caller; see
    resolve_artifact_folder."""
    wanted = canonical_artifact_id(artifact_id)
    return [
        HeldFolder(source, folder, metadata)
        for source in sources
        for folder, metadata in _resolved_from_source(source, wanted)
    ]


class FolderInRoot(NamedTuple):
    """A folder found directly below one of a lookup's roots."""

    source: "ProjectArtifacts"
    folder: Path


def folders_holding_strictly(
    sources: list["ProjectArtifacts"], *, artifact_id: str
) -> list[FolderInRoot]:
    """Every folder in ``sources`` that holds ``artifact_id``, read past the
    identity cache and without writing (identities_below). An error that
    leaves unknown what a root or folder holds raises, so the list never
    misses a folder because its read failed. The roots must already be
    authorized for the caller."""
    wanted = canonical_artifact_id(artifact_id)
    held: list[FolderInRoot] = []
    for source in sources:
        identities = identities_below(source)
        known = _matching_indexed_identity(tuple(identities), wanted)
        if known is not None:
            held.extend(FolderInRoot(source, source.base / name) for name in identities[known])
    return held


def resolve_artifact_folder(
    sources: list["ProjectArtifacts"], artifact_id: str
) -> HeldFolder:
    """Resolve an id only inside roots already authorized for the caller.

    Root discovery and identity indexing never receive ``artifact_id``.  The
    request value selects a server-discovered identity in memory; only that
    trusted copy can enter the descriptor-relative revalidation path.
    """
    matches = artifact_folders_holding(sources, artifact_id=artifact_id)

    if not matches:
        raise FileNotFoundError("Artifact not found")
    if len(matches) > 1:
        raise ArtifactIdentityConflict("Artifact identity is duplicated")
    return matches[0]
