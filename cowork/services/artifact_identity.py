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
import uuid
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

# Re-exported: `artifact_key()` is the canonical `artifact/<uuid>` key the
# comments API, the auth rules and the upload lambda all agree on. Callers here
# import it from this module because that is where the rest of the identity
# vocabulary lives.
from anton.core.artifacts.models import (
    artifact_key,
    canonical_artifact_id,
    extend_legacy_id,
    resolve_artifact_id,
)

if TYPE_CHECKING:
    from cowork.services.artifacts import ProjectArtifacts


logger = logging.getLogger(__name__)


class ArtifactIdentityConflict(RuntimeError):
    """More than one scoped folder claims the same artifact identity."""


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


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def ensure_full_id(folder: Path, metadata: dict | None = None) -> tuple[str, dict]:
    """Return the folder's canonical 32-hex id, persisting a legacy widening.

    The widening is deterministic, so concurrent readers choose the same value
    even before either atomic metadata write wins.
    """
    path = folder / "metadata.json"
    if metadata is None:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    resolved = _resolved_id(metadata, folder)
    if _is_migrated(metadata, resolved):
        return resolved, metadata

    # The caller may have loaded metadata before another subsystem updated it.
    # Merge into the latest durable document instead of writing the stale
    # snapshot back over unrelated metadata fields.
    try:
        latest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Artifact metadata is unreadable") from exc
    resolved = _resolved_id(latest, folder)
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
        before = path.stat()
    except OSError:
        before = None
    try:
        _atomic_json(path, updated)
    except OSError:
        # Persisting is an optimization, not the contract: the id is derived
        # deterministically, so a read-only or full artifacts root still
        # resolves to the same value on every read — exactly what anton does.
        # Propagating would drop the artifact from every listing (both callers
        # treat an identity error as "skip this folder"), which reads as a
        # deletion of files that are sitting right there.
        logger.warning("Could not persist widened artifact id for %s", folder, exc_info=True)
        return resolved, updated
    if before is not None:
        try:
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        except OSError:
            pass
    return resolved, updated


def _directory_version(base: Path) -> tuple[int, int]:
    """A cheap invalidation clock for one artifacts container."""
    try:
        stat = base.stat()
    except OSError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_ctime_ns)


_IDENTITY_INDEX_LIMIT = 256
_identity_indexes: OrderedDict[
    tuple[str, tuple[int, int]], dict[str, tuple[str, ...]]
] = OrderedDict()
_identity_indexes_lock = RLock()


def _build_identity_index(base_value: str) -> dict[str, tuple[str, ...]]:
    """Build an id-to-folder index for one artifacts container."""
    base = Path(base_value)
    matches: dict[str, list[str]] = {}
    try:
        folders = sorted(base.iterdir())
    except OSError:
        return {}
    for folder in folders:
        if not folder.is_dir() or not (folder / "metadata.json").is_file():
            continue
        try:
            found, _metadata = ensure_full_id(folder)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        matches.setdefault(found, []).append(str(folder))
    return {artifact_id: tuple(folders) for artifact_id, folders in matches.items()}


def _identity_index(
    base_value: str,
    directory_clock: tuple[int, int],
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

    index = _build_identity_index(base_value)
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
    cache_key: tuple[str, tuple[int, int]],
) -> dict[str, tuple[str, ...]]:
    return _identity_index(*cache_key, force_refresh=True)[0]


def _validated_index_records(
    base: Path,
    artifact_id: str,
    folder_values: tuple[str, ...],
) -> tuple[tuple[Path, dict], ...]:
    records: list[tuple[Path, dict]] = []
    for folder_value in folder_values:
        folder = Path(folder_value)
        try:
            folder.resolve(strict=False).relative_to(base)
            found, metadata = ensure_full_id(folder)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if found == artifact_id:
            records.append((folder, metadata))
    return tuple(records)


def _indexed_artifacts(base: Path, artifact_id: str) -> tuple[tuple[Path, dict], ...]:
    resolved = base.resolve(strict=False)
    cache_key = (str(resolved), _directory_version(resolved))
    index, was_cached = _identity_index(*cache_key)
    folders = index.get(artifact_id)
    if folders is not None:
        records = _validated_index_records(resolved, artifact_id, folders)
        if len(records) == len(folders):
            return records

    # A freshly-built index is already authoritative for this point in time.
    # Only cached misses or stale positive entries need one forced rebuild.
    if not was_cached:
        return ()

    # Writing metadata inside an already-existing artifact folder does not
    # necessarily tick the artifacts container itself. A miss or failed target
    # revalidation therefore rebuilds once before declaring the identity
    # absent. Positive lookups remain the one-target fast path.
    refreshed = _refresh_identity_index(cache_key)
    return _validated_index_records(resolved, artifact_id, refreshed.get(artifact_id, ()))


def resolve_artifact_folder(
    sources: list["ProjectArtifacts"], artifact_id: str
) -> tuple["ProjectArtifacts", Path, dict]:
    """Resolve an id only inside roots already authorized for the caller."""
    wanted = canonical_artifact_id(artifact_id)
    matches: list[tuple["ProjectArtifacts", Path, dict]] = []
    for source in sources:
        base = Path(source.base).resolve(strict=False)
        for folder, metadata in _indexed_artifacts(base, wanted):
            matches.append((source, folder, metadata))

    if not matches:
        raise FileNotFoundError("Artifact not found")
    if len(matches) > 1:
        raise ArtifactIdentityConflict("Artifact identity is duplicated")
    return matches[0]
