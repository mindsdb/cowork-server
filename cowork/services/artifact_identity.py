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

import json
import logging
import os
import secrets
import stat
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

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
from cowork.common.paths import (
    O_NOFOLLOW,
    PinnedDir,
    dir_open,
    dir_scandir,
    dir_stat,
    dir_unlink,
    open_pinned_child,
    opened_subdir_nofollow,
    pinned_dir,
)

if TYPE_CHECKING:
    from cowork.services.artifacts import ProjectArtifacts


logger = logging.getLogger(__name__)


class ArtifactIdentityConflict(RuntimeError):
    """More than one scoped folder claims the same artifact identity."""


def _component(value: str) -> str:
    """Validate one filesystem component before handing it to ``openat``."""
    value = str(value)
    if not value or value in (".", "..") or Path(value).name != value:
        raise ValueError("Artifact path component is invalid")
    return value


def _normalized_path(path: Path) -> str:
    """Normalize lexically; resolving here would follow the link we must reject."""
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


@contextmanager
def opened_artifact_root(source: "ProjectArtifacts") -> Iterator[PinnedDir]:
    """Pin an authorized artifacts root without following writable links.

    Resolvers created by :mod:`artifact_roots` carry a server-owned project
    directory plus the relative storage components.  Opening those components
    one by one with ``O_NOFOLLOW`` closes both the ordinary symlink escape and
    the check/swap/use race.  Explicit legacy/local sources have no anchor and
    pin their base itself with ``O_NOFOLLOW`` instead.
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
    if not parts or _normalized_path(anchor.joinpath(*parts)) != _normalized_path(base):
        raise ValueError("Artifact root does not match its trusted anchor")
    with opened_subdir_nofollow(anchor, *parts) as root:
        yield root


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


def _read_no_follow(folder: PinnedDir, name: str) -> str:
    """Read a direct child, refusing to follow it if it is a symlink.

    An artifact folder is agent-writable, so `metadata.json` can be replaced
    with a link to something outside the artifact. Reading through it would turn
    identity resolution into a file read of the attacker's choosing, and the
    caller here treats an unreadable identity as "skip this folder", which is
    the correct outcome for a link too. `O_NOFOLLOW` raises `ELOOP` (an
    `OSError`) in that case, which every caller of this module already handles.
    """
    fd = dir_open(folder, _component(name), os.O_RDONLY | O_NOFOLLOW)
    try:
        stream = os.fdopen(fd, "r", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise
    with stream:
        return stream.read()


def _atomic_json(folder: PinnedDir, name: str, payload: dict) -> None:
    name = _component(name)
    tmp = f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        # `O_EXCL | O_NOFOLLOW`: the name carries a uuid4 so a collision is not
        # realistic, but creating rather than opening is what makes "write to a
        # file somebody planted here" impossible rather than improbable.
        fd = dir_open(
            folder,
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        # `os.replace` acts on the link itself, never its target, so a symlinked
        # `metadata.json` is replaced by the real document instead of writing
        # through it.
        if folder.fd is not None:
            os.replace(tmp, name, src_dir_fd=folder.fd, dst_dir_fd=folder.fd)
        else:
            os.replace(folder.path / tmp, folder.path / name)
    finally:
        try:
            dir_unlink(folder, tmp)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _ensure_full_id(
    folder: PinnedDir, metadata: dict | None = None
) -> tuple[str, dict]:
    """Descriptor-relative implementation for :func:`ensure_full_id`."""
    name = "metadata.json"
    path = folder.path / name
    if metadata is None:
        metadata = json.loads(_read_no_follow(folder, name))
    resolved = _resolved_id(metadata, folder.path)
    if _is_migrated(metadata, resolved):
        return resolved, metadata

    # The caller may have loaded metadata before another subsystem updated it.
    # Merge into the latest durable document instead of writing the stale
    # snapshot back over unrelated metadata fields.
    try:
        latest = json.loads(_read_no_follow(folder, name))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Artifact metadata is unreadable") from exc
    resolved = _resolved_id(latest, folder.path)
    if _is_migrated(latest, resolved):
        return resolved, latest
    updated = {key: value for key, value in latest.items() if key != "stableId"}
    updated["id"] = resolved
    # metadata.json's mtime is a turn-recency signal: channel delivery
    # (artifacts_since) treats a fresh mtime as "this turn touched the
    # artifact". Widening an id changes no user-visible content, so restore the
    # original timestamps or the first card/index build after an upgrade
    # would mark every legacy artifact as just-updated and deliver stale
    # attachments to the chat.
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


_IDENTITY_INDEX_LIMIT = 256
_identity_indexes: OrderedDict[
    tuple[str, DirectoryClock], dict[str, tuple[str, ...]]
] = OrderedDict()
_identity_indexes_lock = RLock()


def _build_identity_index(base: PinnedDir) -> dict[str, tuple[str, ...]]:
    """Build an id-to-folder-name index below one pinned container."""
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
                except OSError:
                    continue
    except OSError:
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
                found, _metadata = ensure_full_id(folder.path, _pinned=folder)
        except (OSError, ValueError, json.JSONDecodeError):
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


def resolve_artifact_folder(
    sources: list["ProjectArtifacts"], artifact_id: str
) -> tuple["ProjectArtifacts", Path, dict]:
    """Resolve an id only inside roots already authorized for the caller.

    Root discovery and identity indexing never receive ``artifact_id``.  The
    request value selects a server-discovered identity in memory; only that
    trusted copy can enter the descriptor-relative revalidation path.
    """
    wanted = canonical_artifact_id(artifact_id)
    matches: list[tuple["ProjectArtifacts", Path, dict]] = []
    for source in sources:
        for folder, metadata in _resolved_from_source(source, wanted):
            matches.append((source, folder, metadata))

    if not matches:
        raise FileNotFoundError("Artifact not found")
    if len(matches) > 1:
        raise ArtifactIdentityConflict("Artifact identity is duplicated")
    return matches[0]
