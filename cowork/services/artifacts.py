"""Artifacts service — filesystem operations for agent-produced outputs.

Each artifact is a folder under `<project>/.anton/artifacts/<slug>/`
containing a `metadata.json` and user files. This module handles
listing, resolving paths, and managing preview mounts.

Ported from cowork/server/routes/artifacts.py with projects_store
replaced by the DB-backed project service.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from urllib.parse import quote

from cowork.common.paths import (
    O_NOFOLLOW,
    PinnedDir,
    dir_open,
    dir_rmtree,
    dir_scandir,
    dir_stat,
    dir_unlink,
)
from cowork.common.settings.app_settings import get_app_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectArtifacts:
    """One project's artifacts root plus the identity the client sees.

    Callers resolve this (see services.artifact_roots) and hand it in; the
    service never discovers roots on its own, because discovery cannot tell
    which tenant is asking.
    """

    base: Path
    project_id: str | None
    project_name: str
    # When present, ``base`` is reopened from this server-owned directory one
    # component at a time.  The relative components are agent-writable in org
    # mode, so each open uses O_NOFOLLOW; carrying the anchor separately avoids
    # resolving a symlink before the identity service has a chance to refuse it.
    # Optional defaults preserve callers which construct an explicit local root.
    trusted_anchor: Path | None = None
    root_parts: tuple[str, ...] = ()
    # True only for a desktop project pointed at a folder the user chose. The
    # filesystem scan cannot find those, and their folder basename is not their
    # project name, so a serve URL has to be built from this source rather than
    # rediscovered. Never set for a scanned or an explicitly constructed root.
    external: bool = False


def _org_mode() -> bool:
    """True on the multi-tenant deployment.

    In org mode, artifact files live on shared EFS and are written by any
    org's agent. Executing them, or handing them to the desktop's file
    manager, would run untrusted code inside cowork-server. `noexec` on the
    mount does not stop this: it blocks `./script`, not `python script.py`.

    The same predicate also gates what a card may carry: there is no in-app
    server to serve a live artifact from, and the desktop-only publish
    credentials (`accessPassword`/`accessEmails`) must not reach a client that
    shares its artifacts root with other members of the org.
    """
    return get_app_settings().tenancy_mode == "org"


_NO_EXEC_DETAIL = (
    "Live artifact backends are not available on this deployment. "
    "Open the published version instead."
)


class ExecutionRefused(RuntimeError):
    """Raised instead of running something, because `_org_mode()` is true.

    A distinct type, not a bare RuntimeError, so callers can tell a deployment
    policy apart from a genuine failure. The /artifacts/reveal endpoint maps
    this to 403; any other RuntimeError out of `reveal_in_file_manager` (a
    broken `open`, a platform call that blew up) still has to read as a 500,
    or a real fault would be reported to the client as "not available on this
    deployment" and never looked at again. Subclasses RuntimeError so existing
    `except RuntimeError` callers keep catching the refusal.
    """


# In-memory registry: deterministic token → parent dir of an artifact.
# Used for both static (HTML asset) and proxy (fullstack backend) mounts;
# `kind` field on the preview-mount response payload discriminates.
_PREVIEW_MOUNTS: dict[str, Path] = {}

# Launched-by-cowork-server backend tracking, keyed by artifact slug.
# Shape matches anton's launcher: {"proc", "port", "pid", "log_path"}.
# Used to avoid double-launching and to reap on shutdown.
_LAUNCHED_BACKENDS: dict[str, dict] = {}

# Per-slug mutex so two parallel `preview-mount` requests (React
# StrictMode double-effects, a double-click) can't both decide the port
# is dead and spawn two backends side by side.
_BACKEND_LAUNCH_LOCKS: dict[str, asyncio.Lock] = {}

# ─── Type / kind mapping ──────────────────────────────────────────

ARTIFACT_TYPES = {
    "html-app", "document", "dataset", "image", "mixed",
    "fullstack-stateless-app", "fullstack-stateful-app",
}

KIND_BY_TYPE = {
    "html-app": "Dashboard",
    "document": "Document",
    "dataset": "Data",
    "image": "Image",
    "mixed": "Bundle",
    "fullstack-stateless-app": "App",
    "fullstack-stateful-app": "App",
}

KIND_BY_EXT = {
    ".html": "Dashboard", ".md": "Document", ".txt": "Document",
    ".pdf": "Document", ".csv": "Data", ".json": "Data",
    ".png": "Image", ".jpg": "Image", ".jpeg": "Image", ".svg": "Image",
}

BG_CYCLE = [
    "linear-gradient(135deg, var(--stone-100), var(--surface-03))",
    "linear-gradient(135deg, var(--ocean-50), #fff)",
    "linear-gradient(135deg, var(--sage-50), #fff)",
    "linear-gradient(135deg, #fff, var(--stone-150))",
]

# Files that aren't user content for the `modified` badge's mtime gate.
# Keep in sync with anton.publisher._FULLSTACK_EXCLUDED — the running
# backend's runtime log (`backend.log`) is excluded from the published
# bundle there, so it must not count toward content mtime here either,
# or it would constantly trip the gate and force a false badge.
_HOUSEKEEPING_FILES = {"metadata.json", "README.md", "backend.log", ".published.json", ".revisions"}

TEXT_EXTENSIONS = {
    ".html", ".md", ".txt", ".csv", ".json", ".py", ".js",
    ".ts", ".tsx", ".css", ".log",
}


# ─── Helpers ──────────────────────────────────────────────────────

def _human_mtime(ts: float) -> str:
    """ts <= 0 means "no user content files yet" (see _content_mtime's
    ENG-372 empty-folder case) — there is no real age to report, so this
    returns "" rather than a garbage "updated 19000+ days ago" computed
    from the Unix epoch. Callers fall back to displaying "—" for "".
    """
    if ts <= 0:
        return ""
    secs = time.time() - ts
    if secs < 60:
        return "updated just now"
    if secs < 3600:
        return f"updated {int(secs // 60)}m ago"
    if secs < 86400:
        return f"updated {int(secs // 3600)}h ago"
    return f"updated {int(secs // 86400)}d ago"


def _projects_root() -> Path:
    # Unkeyed: in org mode this dir is empty (projects live org-first under the
    # shared root), so the in-app artifact API sees nothing there. Tracked in
    # next.md §1 (artifacts org-scoping).
    return Path(get_app_settings().project.root_dir)


def _registered_project_dirs() -> list[Path]:
    """All project directories under the projects root, **resolved**.

    The resolution is part of the contract, not an implementation detail.
    ``_artifact_root_for_project`` matches these against a resolved candidate
    with bare set membership, so an unresolved path here simply fails to match
    and the caller 404s.

    **A test that patches this function must return resolved paths too.** On
    macOS ``tempfile`` hands back ``/var/folders/...``, a symlink to
    ``/private/var/folders/...``; a stub returning the raw temp path passes on
    Linux and fails on a Mac, which is how that state went unnoticed.
    """
    root = _projects_root().resolve(strict=False)
    if not root.is_dir():
        return []
    out: list[Path] = []
    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                candidate = child.resolve(strict=False)
                candidate.relative_to(root)
                out.append(candidate)
            except (ValueError, OSError):
                continue
    except OSError:
        pass
    return out


def _scan_artifact_dirs() -> list[Path]:
    """Every project's `.anton/artifacts/` dir that exists."""
    dirs: dict[str, Path] = {}
    for project_dir in _registered_project_dirs():
        candidate = project_dir / ".anton" / "artifacts"
        if candidate.is_dir():
            dirs[str(candidate.resolve())] = candidate
    return list(dirs.values())


def _iter_artifact_folders(project_path: str | None = None) -> Iterator[Path]:
    """Yield artifact folders containing readable metadata.json."""
    roots: list[Path]
    if project_path is not None:
        if not project_path or "\x00" in project_path:
            return
        try:
            requested = Path(project_path).expanduser().resolve(strict=False)
        except (OSError, ValueError, RuntimeError):
            return
        registered = set(_registered_project_dirs())
        if requested not in registered:
            return
        candidate = requested / ".anton" / "artifacts"
        if not candidate.is_dir():
            return
        roots = [candidate]
    else:
        roots = _scan_artifact_dirs()
    for root in roots:
        try:
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                if not (child / "metadata.json").is_file():
                    continue
                yield child
        except OSError:
            continue


def _load_metadata(folder: Path) -> dict | None:
    path = folder / "metadata.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Skipping artifact with unreadable metadata: %s", path, exc_info=True)
        return None


def origin_conversation_id(meta: dict | None) -> str:
    """The conversation that first created this artifact, from its metadata
    `provenance` (written by the shared ArtifactStore for every harness) — the
    creating conversation is the first entry. Empty when unknown: artifacts
    predate provenance, and a folder can be edited by later conversations
    without the first entry ever changing.

    Takes the already-parsed metadata so the card builder does not read
    `metadata.json` a second time; `task_objects` passes what it loaded.
    """
    provenance = (meta or {}).get("provenance") or []
    if not isinstance(provenance, list) or not provenance:
        return ""
    first = provenance[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("conversation") or "")


def _user_files(folder: Path) -> list[Path]:
    """All non-housekeeping files inside an artifact folder, sorted by mtime desc."""
    out: list[Path] = []
    try:
        for p in folder.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(folder)
            top = rel.parts[0] if rel.parts else ""
            if top in _HOUSEKEEPING_FILES:
                continue
            out.append(p)
    except OSError:
        return []
    try:
        out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        pass
    return out


def _pick_primary(
    folder: Path,
    files: list[Path],
    primary_hint: str | None = None,
    *,
    preserve_folder_path: bool = False,
) -> Path | None:
    """The "open this" file for an artifact card."""
    if primary_hint:
        try:
            candidate = folder / primary_hint
            target = candidate.resolve()
            target.relative_to(folder.resolve())
            if target.is_file():
                # A pinned Linux folder is represented through /proc/self/fd.
                # Keep that stable descriptor path rather than returning the
                # resolved, swappable storage name after the containment check.
                return candidate if preserve_folder_path else target
        except (ValueError, OSError):
            pass
    if not files:
        return None
    index = next((f for f in files if f.name == "index.html"), None)
    if index is not None:
        return index
    html = next((f for f in files if f.suffix.lower() == ".html"), None)
    if html is not None:
        return html
    return files[0]


def _load_published_map(folder: Path) -> dict:
    """Read a folder's `.published.json` into a dict, or {} if absent/unreadable."""
    path = folder / ".published.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_published_map_pinned(folder: PinnedDir) -> dict:
    """Read the direct publish record without following a planted link."""
    try:
        fd = dir_open(folder, ".published.json", os.O_RDONLY | O_NOFOLLOW)
    except OSError:
        return {}
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return {}
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            data = json.load(stream)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    finally:
        if fd >= 0:
            os.close(fd)
    return data if isinstance(data, dict) else {}


def _write_published_map_pinned(folder: PinnedDir, payload: dict) -> None:
    """Best-effort atomic publish-record update relative to a pinned folder."""
    name = ".published.json"
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        fd = dir_open(
            folder,
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
        if folder.fd is not None:
            os.replace(
                temporary,
                name,
                src_dir_fd=folder.fd,
                dst_dir_fd=folder.fd,
            )
        else:
            os.replace(folder.path / temporary, folder.path / name)
    except Exception:
        logger.warning("Could not write pinned publish record in %s", folder.path, exc_info=True)
    finally:
        try:
            dir_unlink(folder, temporary)
        except OSError:
            pass


def _content_mtime(folder: Path) -> int:
    """Max mtime (int seconds) across an artifact's user content files.

    Disk-derived, so it reflects in-place edits the metadata.json mtime
    misses. Housekeeping files (`metadata.json`, `README.md`,
    `.published.json`) are excluded — they're not user content. Used both as
    the renderer's cache-bust token and as the cheap "changed since publish"
    gate for the `modified` badge.
    """
    try:
        return int(max((p.stat().st_mtime for p in _user_files(folder)), default=0.0))
    except OSError:
        return 0


# Public alias: task_objects and the autopublish reconciler need this exact
# basis for their "changed since publish" gate. Keeping one implementation means
# the gate, the `modified` badge, and the card's cache-bust token can never
# disagree.
content_mtime = _content_mtime


def content_mtime_ns(folder: Path) -> int:
    """Nanosecond content clock for efficient before/after turn snapshots.

    This uses the same user-file set as :func:`content_mtime` but preserves the
    filesystem's full timestamp precision, avoiding whole-file hashing merely
    to notice two edits that landed in the same second.
    """
    try:
        return max((p.stat().st_mtime_ns for p in _user_files(folder)), default=0)
    except OSError:
        return 0


def load_published_map(folder: Path) -> dict:
    """The `.published.json` record for an artifact folder, `{}` when absent or
    unreadable. Public because the autopublish reconciler needs the same view of
    publish state that the card builder uses."""
    return _load_published_map(folder)


def _published_url_for(
    folder: Path,
    primary: Path | None,
    *,
    published_map: dict | None = None,
) -> str:
    if primary is None:
        return ""
    pmap = published_map if published_map is not None else _load_published_map(folder)
    entry = pmap.get(primary.name)
    if isinstance(entry, dict):
        # `published: False` is a soft-deleted record (kept so re-publish can
        # reuse report_id) — it must not surface as a live URL. Legacy entries
        # have no `published` field; a url means they're live.
        if not entry.get("published", True):
            return ""
        return entry.get("url", "") or ""
    return ""


def _modified_state(
    folder: Path,
    primary: Path | None,
    content_mtime: int,
    *,
    artifacts_base: Path | None = None,
    published_map: dict,
) -> tuple[bool, bool]:
    """Return ``(modified, should_heal_mtime)`` without writing anything.

    Hybrid mtime→md5 (see the 2026-06-23 design):
      1. Not published → not modified.
      2. Cheap gate: content_mtime <= published_mtime → not modified.
      3. Exact: recompute the bundle md5; compare to last_md5.
         - differ → modified;
         - equal (mtime bumped, content identical) → not modified, and
           self-heal published_mtime so the next listing hits the cheap gate.
    A md5 we can't recompute (None) is treated as "can't tell" → not modified,
    so the badge never appears on a false positive.
    """
    if primary is None:
        return False, False
    entry = published_map.get(primary.name)
    if not isinstance(entry, dict) or not entry.get("published", True):
        return False, False
    if not entry.get("report_id"):
        return False, False

    published_mtime = entry.get("published_mtime")
    if isinstance(published_mtime, (int, float)) and content_mtime <= published_mtime:
        return False, False  # cheap gate — nothing touched since publish

    # Local import: publish imports this module, so import lazily to avoid a
    # circular import (mirrors _unpublish_folder below).
    from cowork.services.publish import compute_publish_md5

    # `folder.parent` IS the artifacts root: artifacts always live at
    # `<base>/<slug>/`. Passing it explicitly is what makes the badge work in org
    # mode, where the module-level FS scan finds no roots at all.
    current_md5 = compute_publish_md5(
        folder, artifacts_base=artifacts_base or folder.parent
    )
    if current_md5 is None:
        return False, False  # can't tell — don't raise a false "modified"
    if current_md5 != entry.get("last_md5"):
        return True, False
    return False, True


def _is_modified_read_only(
    folder: Path,
    primary: Path | None,
    content_mtime: int,
    *,
    artifacts_base: Path | None,
    published_map: dict,
) -> bool:
    """Read-only modified badge calculation used exclusively by listing."""
    modified, _should_heal = _modified_state(
        folder,
        primary,
        content_mtime,
        artifacts_base=artifacts_base,
        published_map=published_map,
    )
    return modified


def _is_modified(
    folder: Path,
    primary: Path | None,
    content_mtime: int,
    *,
    artifacts_base: Path | None = None,
    published_map: dict | None = None,
    pinned_folder: PinnedDir | None = None,
) -> bool:
    """Modified badge calculation with the historical mtime self-heal."""
    pmap = published_map if published_map is not None else _load_published_map(folder)
    modified, should_heal = _modified_state(
        folder,
        primary,
        content_mtime,
        artifacts_base=artifacts_base,
        published_map=pmap,
    )
    if modified or not should_heal or primary is None:
        return modified

    # Content identical despite the bumped mtime — heal the snapshot so we
    # don't re-zip on every future listing. Best-effort.
    entry = pmap[primary.name]
    entry["published_mtime"] = content_mtime
    pmap[primary.name] = entry
    if pinned_folder is not None:
        _write_published_map_pinned(pinned_folder, pmap)
    else:
        try:
            (folder / ".published.json").write_text(
                json.dumps(pmap, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass
    return False


def _published_access_for(
    folder: Path,
    primary: Path | None,
    *,
    published_map: dict | None = None,
) -> dict:
    """Owner-side access state for the primary file, from `.published.json`.

    Returns ``accessMode`` (public|password|restricted) plus the mode-specific
    state needed to pre-fill the publish dialog on re-publish:
    ``accessProtected``/``accessPassword`` (password) and
    ``accessEmails``/``orgAllowed``/``ownerOnly`` (restricted). The plaintext password and
    the email list are owner-only — `.published.json` never enters the
    published bundle — so callers must only return this to the artifact's owner
    (the local/authenticated session).
    """
    out = {
        "accessMode": "public",
        "accessProtected": False,
        "accessPassword": "",
        "accessEmails": [],
        "orgAllowed": False,
        "ownerOnly": False,
        # Composite comments scope {user_dir}/{report_id} (Plan 5); "" when
        # unpublished or published before the key was persisted.
        "artifactKey": "",
    }
    if primary is None:
        return out
    try:
        pmap = published_map if published_map is not None else _load_published_map(folder)
        entry = pmap.get(primary.name)
        # A soft-deleted record (published=False) is no longer live, so it must
        # not report a password/restricted mode — that would draw a lock icon on
        # an artifact whose publishedUrl is empty. Legacy entries have no flag.
        if isinstance(entry, dict) and entry.get("published", True):
            # `mode` is authoritative; fall back to the legacy requires_password
            # flag for artifacts published before the mode field existed.
            mode = entry.get("mode") or ("password" if entry.get("requires_password") else "public")
            out["accessMode"] = mode
            out["artifactKey"] = entry.get("artifact_key", "") or ""
            if mode == "password":
                out["accessProtected"] = True
                out["accessPassword"] = entry.get("access_password", "") or ""
            elif mode == "restricted":
                out["accessEmails"] = entry.get("emails", []) or []
                out["orgAllowed"] = bool(entry.get("org_allowed"))
                out["ownerOnly"] = bool(entry.get("owner_only"))
    except Exception:
        pass
    return out


def _external_project_artifacts_base(project_name: str, session) -> Path | None:
    """The artifacts dir of an adopted-folder project, addressed by row name.

    Scoped read, so it cannot reach another tenant's project. Returns None for
    every project the filesystem scan already resolves, leaving that path
    untouched.
    """
    from cowork.services.projects import ProjectService

    service = ProjectService(session)
    project = service.get_project_by_name_or_none(project_name)
    if project is None or not service.directory_is_external(project):
        return None
    base = Path(project.path) / ".anton" / "artifacts"
    return base if base.is_dir() else None


def _project_artifacts_base(project_name: str, session=None) -> Path | None:
    """Resolve a project name to its `.anton/artifacts` dir, only when it
    maps to a registered project. Returns None for unknown projects or
    path-traversal attempts.

    The scan below derives the directory as `<root>/<name>`, which is only true
    for a project the scan can see. A project pointed at a folder the user
    chose is resolved from its row instead, when a session is available.
    """
    if (not project_name or "\x00" in project_name
            or "/" in project_name or "\\" in project_name
            or project_name in (".", "..")):
        return None
    if session is not None:
        base = _external_project_artifacts_base(project_name, session)
        if base is not None:
            return base
    # Compare canonical paths on both sides. On macOS, temporary and user
    # paths commonly cross aliases such as /var -> /private/var; comparing a
    # resolved candidate with raw registry entries incorrectly rejects a
    # genuinely registered project in that case.
    registered: set[Path] = set()
    for project_dir in _registered_project_dirs():
        try:
            registered.add(project_dir.resolve(strict=False))
        except (OSError, ValueError):
            continue
    root = _projects_root().resolve(strict=False)
    try:
        candidate = (root / project_name).resolve(strict=False)
    except (OSError, ValueError):
        return None
    if candidate not in registered:
        return None
    base = candidate / ".anton" / "artifacts"
    return base if base.is_dir() else None


def serve_url_for(
    path: str | Path,
    *,
    artifacts_base: Path | None = None,
    project_name: str | None = None,
) -> str:
    """Origin-relative `/api/v1/artifacts/serve/...` URL for a file under a
    project's `.anton/artifacts` tree. Returns "" when the path isn't
    inside such a tree.

    Always "" in org mode: there the server does not serve artifact content at
    all. The only route to content is the published URL, which carries an access
    check — so there is no local URL to build.

    A caller that already knows the artifacts root and the project's name says
    so. The scan below can only find projects inside the projects root, so for
    an adopted folder it is both wrong about the URL segment (it would use the
    folder's basename) and unable to find the tree at all.
    """
    if _org_mode():
        return ""
    try:
        p = Path(path).resolve(strict=False)
    except (OSError, ValueError):
        return ""
    if artifacts_base is not None and project_name:
        try:
            rel = p.relative_to(artifacts_base.resolve())
        except (ValueError, OSError):
            return ""
        if not rel.parts:
            return ""
        rel_str = "/".join(quote(part) for part in rel.parts)
        return f"/api/v1/artifacts/serve/{quote(project_name)}/{rel_str}"
    for project_dir in _registered_project_dirs():
        base = project_dir / ".anton" / "artifacts"
        try:
            rel = p.relative_to(base.resolve())
        except (ValueError, OSError):
            continue
        if not rel.parts:
            return ""
        rel_str = "/".join(quote(part) for part in rel.parts)
        return f"/api/v1/artifacts/serve/{quote(project_dir.name)}/{rel_str}"
    return ""


def _candidate_relative_artifacts(raw_path: str) -> list[Path]:
    text = (raw_path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    parts = [p for p in text.split("/") if p]
    if not text or any(p in (".", "..") for p in parts):
        return []
    if text.startswith("artifacts/"):
        text = text[len("artifacts/"):]
    matches: dict[str, Path] = {}
    for art_root in _scan_artifact_dirs():
        try:
            target = (art_root / text).resolve()
            target.relative_to(art_root.resolve())
        except ValueError:
            continue
        if target.is_file() or (target.is_dir() and (target / "metadata.json").exists()):
            matches[str(target)] = target
    return list(matches.values())


def resolve_artifact_path(raw_path: str, *, allow_dir: bool = False) -> Path | None:
    """Turn an artifact request path into an absolute path on disk.

    Returns None if invalid, or the resolved Path if found.
    Raises ValueError with a message for 400/404 cases.

    When `allow_dir` is set, an absolute path that resolves to an artifact
    *root directory* (one carrying `metadata.json`) is also accepted — used
    by publish/unpublish so a folder-based artifact can be addressed by its
    folder. The relative-path branch stays file-only (the client always
    sends absolute folder paths).
    """
    if "\x00" in raw_path:
        raise ValueError("Invalid artifact path")
    try:
        target = Path(raw_path).expanduser()
    except Exception as exc:
        raise ValueError("Invalid artifact path") from exc
    if not str(target).strip():
        raise ValueError("Invalid artifact path")

    if target.is_absolute():
        resolved = target.resolve()
        for art_root in _scan_artifact_dirs():
            try:
                resolved.relative_to(art_root.resolve())
            except ValueError:
                continue
            if resolved.is_file() or (resolved.is_dir() and (resolved / "metadata.json").exists()):
                return resolved
            if allow_dir and resolved.is_dir() and (resolved / "metadata.json").is_file():
                return resolved
        raise FileNotFoundError("Artifact is not in a known artifacts directory")

    matches = _candidate_relative_artifacts(raw_path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Artifact path matches multiple project artifact roots; pass an absolute path")
    raise FileNotFoundError("Artifact is not in a known artifacts directory")


def _artifact_root_for(path: Path) -> Path:
    """Climb from an artifact file to the folder that holds its
    `metadata.json` — the artifact root.

    The primary file isn't always at the root: backend+frontend apps
    keep their frontend in a `static/` subdir (so the backend can mount
    it with `StaticFiles`), which puts the primary one level below the
    root. `path.parent` then points at `static/`, where there's no
    `metadata.json`, and callers that look there miss the backend port
    entirely. We walk up until we find the dir carrying `metadata.json`,
    bounded by the registered artifact container dirs
    (`<base>/.anton/artifacts/`) so a metadata-less tree can't send us
    climbing into the rest of the disk. Falls back to `path.parent`.
    """
    containers = {str(d.resolve()) for d in _scan_artifact_dirs()}
    current = path.parent.resolve()
    while True:
        if (current / "metadata.json").is_file():
            return current
        # Stop at a container root (its direct children are the artifact
        # roots — it has no metadata.json of its own) or the fs root.
        if str(current) in containers or current.parent == current:
            return path.parent.resolve()
        current = current.parent


def _fullstack_types() -> frozenset[str]:
    """The artifact types anton's publisher bundles as fullstack apps.

    Imported lazily so publish/preview degrade to static-HTML-only
    behaviour if the anton package is unavailable, rather than 500ing.
    """
    try:
        from anton.publisher import FULLSTACK_ARTIFACT_TYPES
        return frozenset(FULLSTACK_ARTIFACT_TYPES)
    except Exception:
        return frozenset()


def _unpublish_folder(
    folder: Path, *, artifacts_base: Path, api_key: str, publish_url: str
) -> None:
    """Unpublish every published file in an artifact folder.

    Reads `.published.json` and unpublishes each recorded file from the
    remote. Raises if any unpublish fails so the caller can abort the
    delete and leave the artifact intact.
    """
    published_map = _load_published_map(folder)
    if not published_map:
        # Absent or unreadable record — nothing actionable to unpublish.
        return

    # Local import to avoid a circular dependency: publish imports artifacts.
    from cowork.services.publish import unpublish_artifact

    for name, entry in published_map.items():
        if not isinstance(entry, dict):
            continue
        # Soft-deleted records keep their report_id so a re-publish can reuse
        # the URL, but they're already gone from the remote — re-unpublishing
        # would fire a redundant delete (and a transient 5xx/timeout would
        # raise and block the artifact delete).
        if entry.get("published") is False:
            continue
        if not (entry.get("report_id") or entry.get("last_md5")):
            continue
        if not (folder / name).is_file():
            # The path-based unpublish needs the file present, so this record
            # cannot be cleared upstream — and once the folder is gone there is
            # nothing left pointing at the remote copy. No metrics backend
            # exists here, so this log line is the metric; keep the prefix.
            logger.warning(
                "orphaned_publish identifier=%s url=%s reason=primary_missing",
                entry.get("report_id") or entry.get("last_md5"),
                entry.get("url", ""),
            )
            continue
        unpublish_artifact(
            folder / name, artifacts_base=artifacts_base,
            api_key=api_key, publish_url=publish_url,
        )


def _unpublish_identifier(
    identifier: str, entry: dict, *, api_key: str, publish_url: str
) -> None:
    """Revoke one known remote identifier without reopening a local path."""
    if not api_key:
        raise ValueError("Unpublishing requires an API key")
    try:
        from anton.publisher import unpublish
    except Exception as exc:
        from cowork.services.publish import PublisherUnavailable

        raise PublisherUnavailable("Anton publisher is unavailable") from exc

    ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true"
    try:
        unpublish(
            identifier,
            api_key=api_key,
            publish_url=publish_url,
            ssl_verify=ssl_verify,
        )
    except Exception as exc:
        message = str(exc) or "Unpublishing failed."
        if "404" in message or "not found" in message.lower():
            logger.warning(
                "orphaned_publish identifier=%s url=%s reason=unpublish_404",
                identifier,
                entry.get("url", ""),
            )
            return
        logger.exception("Unpublishing failed (identifier=%s)", identifier)
        raise RuntimeError(f"Unpublishing failed: {message}") from exc


def _unpublish_pinned_folder(
    folder: PinnedDir, *, api_key: str, publish_url: str
) -> None:
    """Unpublish records read and updated through one pinned artifact folder."""
    published_map = _load_published_map_pinned(folder)
    for name, entry in published_map.items():
        if not isinstance(entry, dict) or entry.get("published") is False:
            continue
        identifier = entry.get("report_id") or entry.get("last_md5")
        if not identifier:
            continue
        if (
            not name
            or name in {".", ".."}
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            logger.warning(
                "orphaned_publish identifier=%s url=%s reason=primary_invalid",
                identifier,
                entry.get("url", ""),
            )
            continue
        try:
            primary_stat = dir_stat(folder, name, follow_symlinks=False)
        except OSError:
            primary_stat = None
        if primary_stat is None or not stat.S_ISREG(primary_stat.st_mode):
            logger.warning(
                "orphaned_publish identifier=%s url=%s reason=primary_missing",
                identifier,
                entry.get("url", ""),
            )
            continue

        _unpublish_identifier(
            str(identifier), entry, api_key=api_key, publish_url=publish_url
        )
        entry["published"] = False
        published_map[name] = entry
        # Match the historical partial-progress behavior: if a later remote
        # revoke fails, earlier successful records remain marked unpublished.
        _write_published_map_pinned(folder, published_map)


def delete_artifact(
    artifact: Path, *, artifacts_base: Path, api_key: str, publish_url: str
) -> None:
    """Permanently delete an artifact folder from disk.

    Published files are unpublished first; if any unpublish fails the artifact is
    left on disk and the error propagates. Containment is checked against the
    caller-supplied root — the only thing that ties the request to a tenant.
    """
    if artifact.is_dir() and (artifact / "metadata.json").exists():
        folder = artifact
    elif artifact.is_file():
        folder = artifact.parent
        if not (folder / "metadata.json").exists():
            raise ValueError("Not a valid artifact folder")
    else:
        raise FileNotFoundError("Artifact not found")

    try:
        folder.resolve().relative_to(Path(artifacts_base).resolve())
    except (ValueError, OSError):
        raise FileNotFoundError("Artifact is not in a known artifacts directory")

    # Unpublish before deleting; if this raises, the artifact stays.
    _unpublish_folder(
        folder, artifacts_base=artifacts_base, api_key=api_key, publish_url=publish_url,
    )
    shutil.rmtree(folder)


def artifact_id_for_folder(source: ProjectArtifacts, folder_name: str) -> str:
    """Read one direct child's canonical identity through its authorized root.

    This is used by the legacy slug delete path before it revokes review access.
    The returned id comes from a pinned, non-symlink folder; the request's slug
    is never converted into an absolute path.
    """
    from cowork.services.artifact_identity import (
        ensure_full_id,
        opened_artifact_folder,
    )

    with opened_artifact_folder(source, folder_name) as folder:
        artifact_id, _metadata = ensure_full_id(folder.path, _pinned=folder)
        return artifact_id


def delete_artifact_from_source(
    source: ProjectArtifacts,
    folder_name: str,
    *,
    expected_artifact_id: str | None,
    api_key: str,
    publish_url: str,
) -> None:
    """Delete one direct artifact child through pinned directory descriptors.

    The older :func:`delete_artifact` remains the desktop/path-oriented public
    helper. HTTP deletion has stronger information: an authorized
    ``ProjectArtifacts`` source and a direct child selected by identity or by a
    validated legacy name. Keeping the root and folder descriptors open across
    validation and unpublishing prevents a writable root/folder symlink swap
    from redirecting the eventual removal to another tenant's storage.
    """
    from cowork.services.artifact_identity import (
        _opened_child_directory,
        canonical_artifact_id,
        ensure_full_id,
        opened_artifact_root,
    )

    expected = (
        canonical_artifact_id(expected_artifact_id)
        if expected_artifact_id is not None
        else None
    )
    with opened_artifact_root(source) as root:
        with _opened_child_directory(root, folder_name) as folder:
            try:
                metadata_stat = dir_stat(
                    folder, "metadata.json", follow_symlinks=False
                )
            except OSError as exc:
                raise FileNotFoundError("Artifact not found") from exc
            if not stat.S_ISREG(metadata_stat.st_mode):
                raise FileNotFoundError("Artifact metadata is not a regular file")

            if expected is not None:
                found, _metadata = ensure_full_id(folder.path, _pinned=folder)
                if found != expected:
                    raise FileNotFoundError("Artifact identity changed before deletion")

            # Unpublish before deletion. A remote failure propagates while the
            # exact folder remains present, preserving the established contract.
            _unpublish_pinned_folder(
                folder,
                api_key=api_key,
                publish_url=publish_url,
            )

            opened_stat = (
                os.fstat(folder.fd)
                if folder.fd is not None
                else folder.path.stat(follow_symlinks=False)
            )
            try:
                current_stat = dir_stat(root, folder_name, follow_symlinks=False)
            except OSError as exc:
                raise FileNotFoundError("Artifact changed before deletion") from exc
            if (
                not stat.S_ISDIR(current_stat.st_mode)
                or (current_stat.st_dev, current_stat.st_ino)
                != (opened_stat.st_dev, opened_stat.st_ino)
            ):
                raise FileNotFoundError("Artifact changed before deletion")

        # POSIX ``shutil.rmtree(..., dir_fd=...)`` performs its own fd/stat
        # symlink-attack checks while walking. The root descriptor also keeps
        # this operation in the authorized container if its lexical path moves.
        dir_rmtree(root, folder_name)


def reveal_in_file_manager(path: Path) -> None:
    """Open the OS file manager on `path`. In org mode this always refuses;
    see `_org_mode`."""
    if _org_mode():
        raise ExecutionRefused(_NO_EXEC_DETAIL)
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=False)


# ─── Public API ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _PreparedArtifactCard:
    """Pure card result plus the inputs needed for its modified badge."""

    card: dict
    io_folder: Path
    io_root: Path | None
    primary: Path | None
    mtime_seconds: int


def _prepare_artifact_card(
    folder: Path,
    idx: int = 0,
    *,
    artifact_id: str,
    meta: dict,
    published_map: dict,
    project_id: str | None,
    project_name: str,
    pinned_folder: PinnedDir | None,
    pinned_root: PinnedDir | None,
    artifacts_base: Path | None = None,
) -> _PreparedArtifactCard | None:
    """Assemble the shared card shape without any filesystem mutation."""
    logical_folder = folder
    io_folder = _pinned_directory_path(pinned_folder) if pinned_folder else folder
    io_root = _pinned_directory_path(pinned_root) if pinned_root else None
    if io_folder is None or (pinned_root is not None and io_root is None):
        return None
    files = _user_files(io_folder)
    primary = _pick_primary(
        io_folder,
        files,
        primary_hint=meta.get("primary"),
        preserve_folder_path=pinned_folder is not None,
    )
    logical_primary = None
    if primary is not None:
        try:
            logical_primary = logical_folder / primary.relative_to(io_folder)
        except ValueError:
            primary = None
    primary_path = str(logical_primary) if logical_primary is not None else str(logical_folder)
    primary_ext = logical_primary.suffix.lower() if logical_primary is not None else ""
    artifact_type = meta.get("type") or "mixed"
    kind = KIND_BY_TYPE.get(artifact_type) or KIND_BY_EXT.get(primary_ext, "File")
    is_live = False
    if primary is not None:
        try:
            is_live = (time.time() - primary.stat().st_mtime) < 300
        except OSError:
            is_live = False

    # Max mtime across the artifact's content files — a precise
    # "content changed" signal for the renderer's preview viewer to
    # cache-bust/reload on (ENG-375), and the cheap gate for `modified`.
    # Named `mtime_seconds` so it does not shadow the module-level
    # `content_mtime` alias other services import.
    mtime_seconds = _content_mtime(io_folder)

    card = {
        # The one identity: drafts, published versions, revisions, comments
        # and the workspace API all key off it.
        "id": artifact_id,
        "slug": meta.get("slug") or folder.name,
        "title": meta.get("name") or folder.name,
        "description": meta.get("description") or "",
        "type": artifact_type,
        "kind": kind,
        "ext": primary_ext,
        "updated": _human_mtime(mtime_seconds),
        "mtime": mtime_seconds,
        "live": is_live,
        "bg": BG_CYCLE[idx % len(BG_CYCLE)],
        "fileCount": len(files),
        # `folder`/`path` stay in the payload even in org mode: the renderer uses
        # `path` as an opaque state key (the in-flight `busyPaths` set, live-row
        # matching, title fallbacks), and it is no secret — a client of its own
        # organization already knows `<root>/<org_id>`. Isolation comes from
        # `projectId` plus the server-side scope, not from hiding the path.
        "folder": str(logical_folder),
        "path": primary_path,
        "primary": meta.get("primary") or None,
        "projectId": project_id,
        "projectName": project_name,
        # The conversation that produced the artifact, so a comment addressed
        # with the agent from the artifacts list resumes that chat instead of
        # opening a fresh one. Empty for artifacts written before provenance —
        # the client then falls back to a new conversation. Whether the chat is
        # still reachable is the client's call: it already knows its own
        # conversations and can fetch the rest.
        "originConversationId": origin_conversation_id(meta),
        "publishedUrl": _published_url_for(
            io_folder, primary, published_map=published_map
        ),
        # Filled by one of the two callers below. Keeping the slot here preserves
        # the canonical response shape and key order.
        "modified": False,
        # Owner-side access state (lock badge + eye-reveal). accessPassword
        # is the plaintext, returned only to the owner's own session.
        **_published_access_for(
            io_folder, primary, published_map=published_map
        ),
        "serveUrl": serve_url_for(
            primary_path, artifacts_base=artifacts_base, project_name=project_name
        ),
    }
    from cowork.services.artifact_identity import artifact_key

    # Drafts and every published version share this key. A legacy
    # `.published.json` key is intentionally overridden here so comments do not
    # fork when an artifact is re-published under a different report URL.
    card["artifactKey"] = artifact_key(artifact_id)
    if primary is not None:
        project_ref = project_id or "local"
        # macOS commonly exposes the same temporary directory through both
        # ``/var`` and ``/private/var``. Canonicalize both sides before deriving
        # the URL path so an absolute primary hint cannot fail on that alias.
        # A primary that resolves outside its own folder has no draft URL, and
        # one odd artifact must not take the whole list down with a ValueError:
        # this builder runs per card, and `GET /artifacts/` would 500 instead of
        # dropping the single card. `_user_files` and `_pick_primary` both keep
        # the primary inside, so reaching the warning means one of them changed.
        try:
            rel = primary.resolve(strict=False).relative_to(
                io_folder.resolve(strict=False)
            ).as_posix()
        except ValueError:
            logger.warning("Artifact primary resolves outside its folder: %s", folder)
        else:
            card["draftUrl"] = (
                f"/api/v1/artifacts/drafts/{quote(str(project_ref))}/"
                f"{quote(artifact_id)}/{quote(rel, safe='/')}"
            )
    if _org_mode():
        # Dropped at the single card builder so inline chat cards are covered
        # too: they call this function as well, and a filter applied only at the
        # list endpoint would still hand the plaintext password to the chat.
        # Access in org mode is always restricted-to-org, so neither field has a
        # consumer there. `accessMode` stays — ArtifactStatus draws its badge
        # from it.
        card.pop("accessPassword", None)
        card.pop("accessEmails", None)
    return _PreparedArtifactCard(
        card=card,
        io_folder=io_folder,
        io_root=io_root,
        primary=primary,
        mtime_seconds=mtime_seconds,
    )


def card_for_folder(
    folder: Path,
    idx: int = 0,
    *,
    project_id: str | None = None,
    project_name: str = "",
    _pinned_folder: PinnedDir | None = None,
    _pinned_root: PinnedDir | None = None,
    artifacts_base: Path | None = None,
) -> dict | None:
    """Build one card, retaining legacy metadata and publish-map self-heals.

    Inline chat cards and status refreshes historically migrate legacy ids and
    heal stale publish mtimes. HTTP artifact listing uses the separate pinned,
    read-only entry point below; both share only the mutation-free assembler.
    """
    from cowork.services.artifact_identity import ensure_full_id

    try:
        if _pinned_folder is not None:
            artifact_id, meta = ensure_full_id(folder, _pinned=_pinned_folder)
            published_map = _load_published_map_pinned(_pinned_folder)
        else:
            meta = _load_metadata(folder)
            if meta is None:
                return None
            artifact_id, meta = ensure_full_id(folder, meta)
            published_map = _load_published_map(folder)
    except (OSError, ValueError) as exc:
        # No stack: this is a handled skip, and it runs on a polled list
        # endpoint. A full traceback per occurrence per poll is what buried
        # the real signal during the 2026-08-31 incident.
        logger.warning("Skipping artifact with invalid identity: %s (%s)", folder, exc)
        return None

    prepared = _prepare_artifact_card(
        folder,
        idx,
        artifact_id=artifact_id,
        meta=meta,
        published_map=published_map,
        project_id=project_id,
        project_name=project_name,
        pinned_folder=_pinned_folder,
        pinned_root=_pinned_root,
        artifacts_base=artifacts_base,
    )
    if prepared is None:
        return None
    prepared.card["modified"] = _is_modified(
        prepared.io_folder,
        prepared.primary,
        prepared.mtime_seconds,
        artifacts_base=prepared.io_root,
        published_map=published_map,
        pinned_folder=_pinned_folder,
    )
    return prepared.card


def _listed_card_for_pinned_folder(
    folder: Path,
    idx: int,
    *,
    project_id: str | None,
    project_name: str,
    pinned_folder: PinnedDir,
    pinned_root: PinnedDir,
    artifacts_base: Path | None = None,
) -> dict | None:
    """Build one list card through a call graph containing no artifact writes."""
    from cowork.services.artifact_identity import read_full_id

    try:
        artifact_id, meta = read_full_id(pinned_folder)
        published_map = _load_published_map_pinned(pinned_folder)
    except (OSError, ValueError) as exc:
        # No stack: this is a handled skip, and it runs on a polled list
        # endpoint. A full traceback per occurrence per poll is what buried
        # the real signal during the 2026-08-31 incident.
        logger.warning("Skipping artifact with invalid identity: %s (%s)", folder, exc)
        return None

    prepared = _prepare_artifact_card(
        folder,
        idx,
        artifact_id=artifact_id,
        meta=meta,
        published_map=published_map,
        project_id=project_id,
        project_name=project_name,
        pinned_folder=pinned_folder,
        pinned_root=pinned_root,
        artifacts_base=artifacts_base,
    )
    if prepared is None:
        return None
    prepared.card["modified"] = _is_modified_read_only(
        prepared.io_folder,
        prepared.primary,
        prepared.mtime_seconds,
        artifacts_base=prepared.io_root,
        published_map=published_map,
    )
    return prepared.card


def _pinned_directory_path(directory: PinnedDir | None) -> Path | None:
    """A stable path view of an open directory for path-oriented card helpers.

    Org mode runs on Linux, where procfs exposes an open directory descriptor as
    a traversable path.  macOS/Windows are local-only and retain their ordinary
    path behavior.  A Linux deployment without procfs fails closed instead of
    falling back to a swappable shared-EFS name.
    """
    if directory is None:
        return None
    if directory.fd is None or not sys.platform.startswith("linux"):
        return directory.path
    candidate = Path("/proc/self/fd") / str(directory.fd)
    try:
        return candidate if candidate.is_dir() else None
    except OSError:
        return None


def _blank_artifact_status() -> dict:
    """The stable response for an unknown or no-longer-readable artifact."""
    return {
        "publishedUrl": "", "modified": False, "accessMode": "public",
        "accessProtected": False, "accessEmails": [], "orgAllowed": False,
        "ownerOnly": False,
    }


def artifact_status_for_resolved(artifact: Path) -> dict:
    """Fresh owner-side status for an already resolved artifact path.

    The HTTP boundary resolves an untrusted request through
    :func:`resolve_artifact_path` once, then calls this function. Keeping the
    resolved-path operation separate prevents status computation from walking a
    request string a second time after its containment decision.

    Reuses ``card_for_folder`` so the preview viewer's in-place refresh can
    never disagree with the listing. Returns the published/modified/access
    subset only — the cheap read the viewer polls on window focus to light up
    the "Update" button when the artifact changes underneath an open preview.
    """
    blank = _blank_artifact_status()
    if not isinstance(artifact, Path) or not artifact.is_absolute():
        return dict(blank)
    # Fullstack artifacts keep their primary file in a `static/` subdir, so a
    # naive `artifact.parent` would land there instead of the artifact root
    # (where metadata.json + .published.json actually live) — climb via
    # `_artifact_root_for` like `mount_preview`/`html_artifacts` do.
    folder = artifact if artifact.is_dir() else _artifact_root_for(artifact)
    card = card_for_folder(folder)
    if card is None:
        # Loose / legacy file (no metadata.json anywhere up to the artifacts
        # container root).
        card = {
            "publishedUrl": _published_url_for(folder, artifact),
            "modified": False,
            **_published_access_for(folder, artifact),
        }
    # One response shape for BOTH branches: explicitly the published /
    # modified / access subset, and NEVER `accessPassword` — that owner-only
    # plaintext (which `_published_access_for` and the card both carry) must
    # not leave this endpoint.
    return {
        "publishedUrl": card.get("publishedUrl", ""),
        "modified": bool(card.get("modified")),
        "accessMode": card.get("accessMode", "public"),
        "accessProtected": bool(card.get("accessProtected")),
        "accessEmails": card.get("accessEmails", []),
        "orgAllowed": bool(card.get("orgAllowed")),
        "ownerOnly": bool(card.get("ownerOnly")),
        "artifactKey": card.get("artifactKey", ""),
    }


def artifact_status(raw_path: str) -> dict:
    """Compatibility wrapper accepting the historical raw path string."""
    try:
        artifact = resolve_artifact_path(raw_path, allow_dir=True)
    except Exception:
        return _blank_artifact_status()
    if artifact is None:
        return _blank_artifact_status()
    return artifact_status_for_resolved(artifact)


def list_artifacts(sources: list[ProjectArtifacts]) -> list[dict]:
    """Every artifact under the given roots, newest first, capped at 80.

    Roots come from the caller (see services.artifact_roots) — this function
    never discovers them, because a filesystem scan cannot tell which tenant is
    asking. The 80-item cap is pre-existing but matters more in org mode, where
    `sources` spans every project of the organization instead of one tree.
    """
    cards: list[dict] = []
    from cowork.services.artifact_identity import (
        _opened_child_directory,
        opened_artifact_root,
    )

    for source in sources:
        try:
            with opened_artifact_root(source) as root:
                with dir_scandir(root) as entries:
                    child_names = []
                    for entry in entries:
                        # An artifact folder never starts with a dot, while the
                        # root also holds housekeeping directories that do:
                        # `.locks` from cowork.services.artifact_locks, and the
                        # `.{name}.{uuid}.tmp` scratch directories atomic writes
                        # mint. Skipping them here costs nothing. Admitting them
                        # means opening each one and discovering the missing
                        # metadata.json by exception, once per root per request.
                        if entry.name.startswith("."):
                            continue
                        try:
                            if (
                                not entry.is_symlink()
                                and entry.is_dir(follow_symlinks=False)
                            ):
                                child_names.append(entry.name)
                        except OSError:
                            continue
                for child_name in sorted(child_names):
                    try:
                        with _opened_child_directory(root, child_name) as pinned_folder:
                            folder = root.path / child_name
                            card = _listed_card_for_pinned_folder(
                                folder,
                                len(cards),
                                project_id=source.project_id,
                                project_name=source.project_name,
                                pinned_folder=pinned_folder,
                                pinned_root=root,
                                artifacts_base=(
                                    source.base if source.external else None
                                ),
                            )
                            if card is None:
                                continue
                            try:
                                card["_sortTs"] = dir_stat(
                                    pinned_folder, "metadata.json"
                                ).st_mtime
                            except OSError:
                                card["_sortTs"] = 0.0
                            cards.append(card)
                    except (OSError, ValueError):
                        continue
        except (OSError, ValueError):
            continue

    cards.sort(key=lambda c: c["_sortTs"], reverse=True)
    for c in cards:
        c.pop("_sortTs", None)
    return cards[:80]


def preview_artifact(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        raise ValueError("Preview is available for text, Markdown, code, JSON, CSV, and HTML files")
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path),
        "title": path.name,
        "kind": KIND_BY_EXT.get(suffix, "File"),
        "mime": mimetypes.guess_type(str(path))[0] or "text/plain",
        "content": text[:200_000],
        "truncated": len(text) > 200_000,
    }


async def mount_preview(path: Path) -> dict:
    """Register an artifact for iframe preview.

    Two payload shapes share a `kind` discriminator:
      - `kind: "static"` (HTML asset bundles) — token + relUrl that
        the client loads against `/artifacts/preview-asset/`.
      - `kind: "proxy"` (fullstack apps with a `port` in metadata.json)
        — token + artifactDir + backend status; the route layer builds
        the absolute proxyUrl pointing at our forwarder.
    """
    parent = path.parent.resolve()
    # The artifact root (where metadata.json lives) is not always the
    # primary file's parent — fullstack apps keep their frontend in a
    # `static/` subdir. Resolve it explicitly for all backend lookups so
    # we read the `port` from the root, not from `static/`.
    root = _artifact_root_for(path)

    # Backend+frontend artifacts: detect them by a `port` field in the
    # root's metadata.json. The iframe will load through our proxy
    # endpoint instead of preview-asset.
    backend_port: int | None = None
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw_port = meta.get("port")
            if isinstance(raw_port, int) and 0 < raw_port < 65536:
                backend_port = raw_port
        except Exception:
            backend_port = None

    if backend_port is not None:
        # Proxy mode. In org mode this must refuse before registering
        # anything: the token minted below maps into _PREVIEW_MOUNTS, and
        # cowork.services.preview_proxy re-reads `port` from this artifact's
        # own (agent-writable) metadata.json on every proxied request, then
        # issues an httpx call to 127.0.0.1:<port> carrying the caller's
        # method, path, query, headers and body. _ensure_backend_running
        # already refuses to launch a backend here (see `_org_mode`), but
        # that alone does not stop the registration: any org's agent could
        # still point `port` at an unrelated loopback listener inside this
        # pod and use the mount as an SSRF pivot, backend running or not.
        if _org_mode():
            raise ValueError(_NO_EXEC_DETAIL)
        # Register the artifact root (where metadata.json and the live port
        # live) so the proxy endpoint reads a current port by token, then
        # auto-launch the backend if dead. Returns without `proxyUrl`; the
        # route layer fills it in using the incoming Request URL so the
        # absolute URL matches whatever host/scheme the client used to
        # reach us.
        root_token = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        _PREVIEW_MOUNTS[root_token] = root
        running, launch_detail, current_port = await _ensure_backend_running(
            root, backend_port
        )
        return {
            "kind": "proxy",
            "token": root_token,
            "artifactDir": str(root),
            "port": current_port if running else backend_port,
            "backendRunning": running,
            "launchError": "" if running else launch_detail,
            # Fullstack apps publish from the artifact root, `.published.json`
            # keyed by the primary file name — surface the published state so
            # the viewer shows the "Published" pill for backend artifacts too.
            "publishedUrl": _published_url_for(root, path),
            **_published_access_for(root, path),
        }

    # Static (HTML) branch — same behaviour as before, with an explicit
    # `kind` discriminator so the client doesn't have to infer.
    if path.suffix.lower() != ".html":
        raise ValueError("Preview mount is only available for HTML artifacts")
    token = hashlib.sha256(str(parent).encode("utf-8")).hexdigest()[:16]
    _PREVIEW_MOUNTS[token] = parent

    return {
        "kind": "static",
        "token": token,
        "entry": path.name,
        "relUrl": f"/artifacts/preview-asset/{token}/{path.name}",
        "serveUrl": serve_url_for(path),
        # Route through _published_url_for so a soft-deleted (published=False)
        # record reports an empty URL — matching the artifact grid and the
        # fullstack branch, instead of surfacing a dead 4nton.ai link.
        "publishedUrl": _published_url_for(parent, path),
        **_published_access_for(parent, path),
    }


def get_preview_mount(token: str) -> Path | None:
    return _PREVIEW_MOUNTS.get(token)


def html_artifacts() -> list[dict]:
    """List every publishable file (HTML + Markdown) under every project's
    artifacts tree.

    `.md` files publish as rendered HTML pages (see `publish.py`), so they
    belong in this list alongside `.html`. Fullstack apps keep their pages
    inside `static/`; they're surfaced as a single entry per artifact root
    (titled by the root's metadata), not one row per page.
    """
    out = []
    seen: set[str] = set()
    seen_roots: set[str] = set()
    fullstack_types = _fullstack_types()
    for art_root in _scan_artifact_dirs():
        if not art_root.exists():
            continue
        candidates = [p for ext in ("*.html", "*.md") for p in art_root.rglob(ext)]
        for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            key = str(path.resolve())
            if key in seen:
                continue

            # Group fullstack apps by their artifact root — one entry per app.
            artifact_root = _artifact_root_for(path)
            meta = _load_metadata(artifact_root) if (artifact_root / "metadata.json").is_file() else None
            if (meta or {}).get("type") in fullstack_types:
                root_key = str(artifact_root.resolve())
                if root_key in seen_roots:
                    continue
                seen_roots.add(root_key)
                primary_hint = meta.get("primary") or ""
                entry_path = (artifact_root / primary_hint) if primary_hint else path
                if not entry_path.is_file():
                    entry_path = path
                seen.add(str(entry_path.resolve()))
                out.append({
                    "title": meta.get("name") or artifact_root.name.replace("_", " ").replace("-", " ").title(),
                    "path": str(entry_path),
                    "bytes": entry_path.stat().st_size if entry_path.is_file() else 0,
                    "publishedUrl": _published_url_for(artifact_root, entry_path),
                })
                continue

            seen.add(key)
            out.append({
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "path": str(path),
                "bytes": path.stat().st_size,
                "publishedUrl": _published_url_for(path.parent, path),
            })
    return out[:40]


# ─── Backend-artifact auto-launch ─────────────────────────────────
#
# When the user opens preview for a `fullstack-stateful-app` artifact,
# its `metadata.json` records the TCP port the backend bound to. That
# backend may or may not still be alive — the session that launched it
# could be gone, the server might have been restarted, the process
# might have crashed. Rather than refuse to preview, we probe the port
# and try to bring the backend back up if it's down: delegate to anton's
# `launch_artifact_backend` so the spawn semantics (slug-keyed venv,
# requirements.txt install, `--port` flag, HTTP+TCP readiness probe)
# match Anton's own `launch_backend` tool exactly. The new port is
# persisted back to metadata.json so the proxy and future opens see it.

def _launch_lock(key: str) -> asyncio.Lock:
    lock = _BACKEND_LAUNCH_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _BACKEND_LAUNCH_LOCKS[key] = lock
    return lock


def _probe_port(port: int, *, timeout: float = 0.3) -> bool:
    """True iff something is accepting TCP connections on 127.0.0.1:<port>."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_project_root(artifact_dir: Path) -> Path | None:
    """The registered project root that owns this artifact dir, if any.

    `artifact_dir` is the parent of the primary file (e.g.
    `<project>/.anton/artifacts/<slug>/`). We walk back to the registered
    project root by checking ancestors against `_registered_project_dirs()`.
    Returns None when the dir isn't under any registered project — in
    which case auto-launch is a non-starter anyway.
    """
    try:
        artifact_resolved = artifact_dir.resolve()
    except OSError:
        return None
    registered = _registered_project_dirs()
    for parent in (artifact_resolved, *artifact_resolved.parents):
        if parent in registered:
            return parent
    return None


async def _ensure_backend_running(
    artifact_dir: Path, port: int
) -> tuple[bool, str, int]:
    """Bring up the artifact's backend if it isn't already listening.

    Returns `(running, detail, port)`:
      - `running=True`  → port is alive; `detail` is a short label
        ("already_running" or "launched"); `port` may differ from the
        input when the helper had to allocate a fresh free port.
      - `running=False` → backend is down and we couldn't start it;
        `detail` carries the reason; `port` echoes the input port.

    In org mode this always refuses; see `_org_mode`.
    """
    if _org_mode():
        return False, _NO_EXEC_DETAIL, 0

    slug = artifact_dir.name
    if _probe_port(port):
        return True, "already_running", port

    # Serialize launches per-slug. Whichever request wins the lock does
    # the actual work; the rest just re-probe after it releases.
    async with _launch_lock(slug):
        if _probe_port(port):
            return True, "already_running", port
        return await _launch_backend_locked(artifact_dir, slug)


async def _launch_backend_locked(
    artifact_dir: Path, slug: str
) -> tuple[bool, str, int]:
    """Spawn the artifact's backend via anton's shared launcher.

    The slug-keyed scratchpad venv (provisioned by Anton when the agent
    built the artifact) is the python interpreter; `requirements.txt`
    in the artifact folder is installed before spawn; the launcher
    picks a free port and passes `--port <port>` to the script. New
    port is persisted into `metadata.json` so the proxy reads a current
    value on its next request.
    """
    from anton.core.artifacts.backend_launcher import launch_artifact_backend

    from cowork.services.scratchpad_runtime import WorkspaceScopedPool

    project_root = _resolve_project_root(artifact_dir)
    if project_root is None:
        return False, "Artifact is not in a registered project.", 0

    pool = WorkspaceScopedPool(str(project_root))

    # Inject the secrets of datasources the artifact declared in metadata.json
    # into the backend subprocess only — NOT the cowork server's global
    # os.environ. The backend is a separate subprocess, so we build an
    # explicit env mapping and let the launcher merge it for the spawn.
    extra_env: dict[str, str] = {}
    try:
        meta = _load_metadata(artifact_dir) or {}
        datasources = meta.get("datasources") or []
        if datasources:
            from anton.core.datasources.data_vault import LocalDataVault

            vault = LocalDataVault(Path(get_app_settings().connector.vault_dir))
            for ds in datasources:
                engine, name = ds.get("engine"), ds.get("name")
                if not engine or not name:
                    continue
                env = vault.env_for(engine, name)
                if env is None:
                    logger.warning(
                        "Datasource %s/%s declared by artifact %s not found in vault — skipping",
                        engine, name, slug,
                    )
                    continue
                extra_env.update(env)
    except Exception:
        logger.warning(
            "Could not build datasource env for backend launch of %s", slug, exc_info=True
        )

    # anton's default health_timeout is 10s — too short for artifacts
    # that do slow IO (HTTP fetches with retry/backoff, large model
    # loads, etc.) before binding their port. The launcher would
    # otherwise terminate a perfectly healthy backend just because it
    # didn't finish startup yet. 45s leaves room for retries without
    # making the user wait forever on a truly stuck script — anton
    # terminates the proc on timeout.
    # ds_env replaces the inherited DS_* instead of merging over them, so the
    # backend sees only what it declared. Absent on an older anton pin.
    import inspect

    if "ds_env" in inspect.signature(launch_artifact_backend).parameters:
        env_kwargs = {"ds_env": extra_env}
    else:
        env_kwargs = {"extra_env": extra_env}

    result = await launch_artifact_backend(
        slug=slug,
        artifact_folder=artifact_dir,
        scratchpad_pool=pool,
        tracked_backends=_LAUNCHED_BACKENDS,
        **env_kwargs,
        health_timeout=45.0,
    )
    if isinstance(result, str):
        # Helper returned an error string. Strip the redundant "Error: "
        # prefix so the message reads naturally in the preview pane.
        detail = result[len("Error: "):] if result.startswith("Error: ") else result
        return False, detail, 0

    new_port = int(result["port"])
    # Persist the new port directly to metadata.json. The proxy reads
    # metadata.json on every request — without this write it would keep
    # dialing the stale port even though the backend is healthy on a
    # different one.
    try:
        meta_path = artifact_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["port"] = new_port
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        # Metadata write failure shouldn't abort an otherwise-working
        # relaunch — the backend is up and we return the new port. But
        # the proxy will keep dialing the stale port until the next
        # successful write.
        logger.warning("Could not persist backend port to metadata: %s", exc)

    logger.info(
        "Auto-launched artifact backend via anton helper: slug=%s port=%d pid=%s",
        slug, new_port, result.get("pid"),
    )
    return True, "launched", new_port


def shutdown_launched_backends() -> None:
    """Terminate every backend cowork-server itself launched.

    Synchronous: we schedule `proc.terminate()` (which is non-blocking on
    `asyncio.subprocess.Process`) without awaiting `proc.wait()`. The
    server is exiting anyway, and PR_SET_PDEATHSIG on Linux already makes
    the kernel SIGTERM the backends when we go. macOS relies on the
    explicit `terminate()` call.
    """
    for slug, entry in list(_LAUNCHED_BACKENDS.items()):
        proc = entry.get("proc")
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except (OSError, ProcessLookupError):
                pass
        _LAUNCHED_BACKENDS.pop(slug, None)
