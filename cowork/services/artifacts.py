"""Artifacts service — filesystem operations for agent-produced outputs.

Each artifact is a folder under `<project>/.anton/artifacts/<slug>/`
containing a `metadata.json` and user files. This module handles
listing, resolving paths, and managing preview mounts.

Ported from cowork/server/routes/artifacts.py with projects_store
replaced by the DB-backed project service.
"""
from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import logging
import mimetypes
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

from urllib.parse import quote

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.encryption import _load_or_create_master_key

logger = logging.getLogger(__name__)

# In-memory registry: short-lived opaque token -> parent dir of an artifact.
# Used for both static (HTML asset) and proxy (fullstack backend) mounts;
# `kind` field on the preview-mount response payload discriminates.
_PREVIEW_MOUNTS: dict[str, Path] = {}
_PREVIEW_MOUNT_EXPIRES: dict[str, float] = {}
_PREVIEW_MOUNT_TTL_SECONDS = 60 * 60
_SERVE_URL_TTL_SECONDS = 15 * 60

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

_HOUSEKEEPING_FILES = {"metadata.json", "README.md", ".published.json"}

TEXT_EXTENSIONS = {
    ".html", ".md", ".txt", ".csv", ".json", ".py", ".js",
    ".ts", ".tsx", ".css", ".log",
}


# ─── Helpers ──────────────────────────────────────────────────────

def register_preview_mount(parent: str | Path, *, salt: str | None = None, ttl_seconds: int = _PREVIEW_MOUNT_TTL_SECONDS) -> str:
    """Expose a folder through the tokenized preview-asset endpoint."""
    root = Path(parent).expanduser().resolve(strict=False)
    # `salt` used to make preview tokens deterministic for diff previews.
    # Browser-visible preview URLs are bearer grants, so they must stay
    # unguessable even when the mounted path or version hash is known.
    _ = salt
    token = secrets.token_urlsafe(18)
    _PREVIEW_MOUNTS[token] = root
    _PREVIEW_MOUNT_EXPIRES[token] = time.time() + max(60, ttl_seconds)
    return token


def sign_serve_url_path(path: str | Path, *, ttl_seconds: int = _SERVE_URL_TTL_SECONDS) -> str:
    try:
        resolved = str(Path(path).expanduser().resolve(strict=False))
    except (OSError, ValueError):
        resolved = str(path)
    expires = int(time.time() + max(60, ttl_seconds))
    message = f"{resolved}:{expires}".encode("utf-8")
    signature = hmac.new(_load_or_create_master_key(), message, hashlib.sha256).hexdigest()
    return f"{expires}.{signature}"


def verify_serve_url_token(path: str | Path, token: str | None) -> bool:
    if not token or "." not in token:
        return False
    expires_text, signature = token.split(".", 1)
    try:
        expires = int(expires_text)
    except ValueError:
        return False
    if expires < int(time.time()):
        return False
    try:
        resolved = str(Path(path).expanduser().resolve(strict=False))
    except (OSError, ValueError):
        return False
    message = f"{resolved}:{expires}".encode("utf-8")
    expected = hmac.new(_load_or_create_master_key(), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _human_mtime(path: Path) -> str:
    secs = time.time() - path.stat().st_mtime
    if secs < 60:    return "updated just now"
    if secs < 3600:  return f"updated {int(secs // 60)}m ago"
    if secs < 86400: return f"updated {int(secs // 3600)}h ago"
    return f"updated {int(secs // 86400)}d ago"


def _projects_root() -> Path:
    return Path(get_app_settings().project.root_dir)


def _registered_project_dirs() -> list[Path]:
    """All project directories under the projects root."""
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


def _pick_primary(folder: Path, files: list[Path], primary_hint: str | None = None) -> Path | None:
    """The "open this" file for an artifact card."""
    if primary_hint:
        try:
            target = (folder / primary_hint).resolve()
            target.relative_to(folder.resolve())
            if target.is_file():
                return target
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


def _published_url_for(folder: Path, primary: Path | None) -> str:
    if primary is None:
        return ""
    entry = _load_published_map(folder).get(primary.name)
    if isinstance(entry, dict):
        # `published: False` is a soft-deleted record (kept so re-publish can
        # reuse report_id) — it must not surface as a live URL. Legacy entries
        # have no `published` field; a url means they're live.
        if not entry.get("published", True):
            return ""
        return entry.get("url", "") or ""
    return ""


def _published_version_for(folder: Path, primary: Path | None) -> dict:
    if primary is None:
        return {}
    entry = _load_published_map(folder).get(primary.name)
    if not isinstance(entry, dict):
        return {}
    payload = {
        "publishedVersionId": entry.get("version_id") or "",
        "publishedFilesHash": entry.get("files_hash") or "",
        "publishedManifestHash": entry.get("manifest_hash") or "",
        "publishedVersionNumber": entry.get("version_number"),
    }
    return {key: value for key, value in payload.items() if value not in ("", None)}


def _is_modified(folder: Path, primary: Path | None, content_mtime: int) -> bool:
    """Whether a *published* artifact's content diverged from what was published.

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
        return False
    pmap = _load_published_map(folder)
    entry = pmap.get(primary.name)
    if not isinstance(entry, dict) or not entry.get("published", True):
        return False
    if not entry.get("report_id"):
        return False

    published_mtime = entry.get("published_mtime")
    if isinstance(published_mtime, (int, float)) and content_mtime <= published_mtime:
        return False  # cheap gate — nothing touched since publish

    # Local import: publish imports this module, so import lazily to avoid a
    # circular import (mirrors _unpublish_folder below).
    from cowork.services.publish import compute_publish_md5

    current_md5 = compute_publish_md5(str(folder))
    if current_md5 is None:
        return False  # can't tell — don't raise a false "modified"
    if current_md5 != entry.get("last_md5"):
        return True

    # Content identical despite the bumped mtime — heal the snapshot so we
    # don't re-zip on every future listing. Best-effort.
    entry["published_mtime"] = content_mtime
    pmap[primary.name] = entry
    try:
        (folder / ".published.json").write_text(
            json.dumps(pmap, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        pass
    return False


def _published_access_for(folder: Path, primary: Path | None) -> dict:
    """Owner-side access state for the primary file, from `.published.json`.

    Returns ``accessMode`` (public|password|restricted) plus the mode-specific
    state needed to pre-fill the publish dialog on re-publish:
    ``accessProtected``/``accessPassword`` (password) and
    ``accessEmails``/``orgAllowed`` (restricted). The plaintext password and
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
    }
    if primary is None:
        return out
    published_index = folder / ".published.json"
    if not published_index.is_file():
        return out
    try:
        pmap = json.loads(published_index.read_text(encoding="utf-8"))
        entry = pmap.get(primary.name)
        # A soft-deleted record (published=False) is no longer live, so it must
        # not report a password/restricted mode — that would draw a lock icon on
        # an artifact whose publishedUrl is empty. Legacy entries have no flag.
        if isinstance(entry, dict) and entry.get("published", True):
            # `mode` is authoritative; fall back to the legacy requires_password
            # flag for artifacts published before the mode field existed.
            mode = entry.get("mode") or ("password" if entry.get("requires_password") else "public")
            out["accessMode"] = mode
            if mode == "password":
                out["accessProtected"] = True
                out["accessPassword"] = entry.get("access_password", "") or ""
            elif mode == "restricted":
                out["accessEmails"] = entry.get("emails", []) or []
                out["orgAllowed"] = bool(entry.get("org_allowed"))
    except Exception:
        pass
    return out


def _last_good_preview_for(folder: Path) -> dict:
    try:
        from sqlmodel import Session, select

        from cowork.db.session import get_engine
        from cowork.models.artifact import Artifact, ArtifactVersion
        from cowork.services.artifact_versions import ArtifactVersionService
    except Exception:
        return {}
    try:
        engine = get_engine(get_app_settings().database.uri)
        folder_key = str(folder.expanduser().resolve(strict=False))
        with Session(engine) as session:
            artifact = session.exec(select(Artifact).where(Artifact.path == folder_key)).first()
            if artifact is None or artifact.last_known_good_version_id is None:
                return {}
            version = session.get(ArtifactVersion, artifact.last_known_good_version_id)
            if version is None:
                return {}
            service = ArtifactVersionService(session)
            target = service.store_root / "previews" / "last-good" / str(artifact.id) / str(version.id)
            service.materialize_version(version.id, target, clean=True)
            service.write_version_housekeeping(version.id, target)
            metadata = _load_metadata(target) if (target / "metadata.json").is_file() else {}
            files = _user_files(target)
            primary = _pick_primary(target, files, primary_hint=metadata.get("primary"))
            preview_path = str(primary) if primary is not None else str(target)
            return {
                "lastKnownGoodVersionId": str(version.id),
                "lastGoodPath": preview_path,
                "lastGood": {
                    "versionId": str(version.id),
                    "path": preview_path,
                    "label": version.label,
                },
            }
    except Exception:
        logger.debug("Could not resolve last-known-good preview for %s", folder, exc_info=True)
        return {}


def _review_summary_for(folder: Path, *, viewer_email: str | None = None) -> dict:
    try:
        from sqlmodel import Session, select

        from cowork.db.session import get_engine
        from cowork.models.artifact import Artifact, ArtifactActivityEvent, ArtifactComment
        from cowork.models.project_collaboration import ProjectCollaborator
        from cowork.services.project_collaboration import normalize_email
        from cowork.services.project_permissions import role_allows
    except Exception:
        return _empty_review_summary()
    try:
        engine = get_engine(get_app_settings().database.uri)
        folder_key = str(folder.expanduser().resolve(strict=False))
        with Session(engine) as session:
            artifact = session.exec(select(Artifact).where(Artifact.path == folder_key)).first()
            if artifact is None:
                return _empty_review_summary()
            comments = session.exec(
                select(ArtifactComment).where(ArtifactComment.artifact_id == artifact.id)
            ).all()
            events = session.exec(
                select(ArtifactActivityEvent)
                .where(ArtifactActivityEvent.artifact_id == artifact.id)
                .order_by(ArtifactActivityEvent.created_at.desc())
                .limit(50)
            ).all()
            viewer_state = _review_viewer_state(
                session,
                artifact,
                comments,
                events,
                viewer_email=viewer_email,
                collaborator_model=ProjectCollaborator,
                normalize_email_fn=normalize_email,
                role_allows_fn=role_allows,
            )
    except Exception:
        logger.debug("Could not resolve review summary for %s", folder, exc_info=True)
        return _empty_review_summary()

    open_statuses = {"open"}
    open_comments = [comment for comment in comments if comment.status in open_statuses]
    comments_count = sum(1 for comment in open_comments if comment.kind == "comment")
    suggestions_count = sum(1 for comment in open_comments if comment.kind == "suggestion")
    review_requests_count = sum(1 for comment in open_comments if comment.kind == "review")
    latest = max((comment.created_at for comment in comments if comment.created_at is not None), default=None)
    return {
        "open": len(open_comments),
        "unresolved": len(open_comments),
        "comments": comments_count,
        "suggestions": suggestions_count,
        "reviewRequests": review_requests_count,
        "resolved": sum(1 for comment in comments if comment.status == "resolved"),
        "accepted": sum(1 for comment in comments if comment.status == "accepted"),
        "rejected": sum(1 for comment in comments if comment.status == "rejected"),
        "needsReview": bool(suggestions_count or review_requests_count),
        "latestAt": latest.isoformat() if latest is not None else None,
        "viewerState": viewer_state,
    }


def _empty_review_summary() -> dict:
    return {
        "open": 0,
        "unresolved": 0,
        "comments": 0,
        "suggestions": 0,
        "reviewRequests": 0,
        "resolved": 0,
        "accepted": 0,
        "rejected": 0,
        "needsReview": False,
        "latestAt": None,
        "viewerState": _empty_review_viewer_state(),
    }


def _empty_review_viewer_state() -> dict:
    return {
        "available": False,
        "unreadComments": 0,
        "unreadActivity": 0,
        "needsAction": 0,
        "reviewRequests": {
            "open": 0,
            "needsAction": 0,
            "unread": 0,
        },
    }


def _review_viewer_state(
    session,
    artifact,
    comments,
    events,
    *,
    viewer_email: str | None,
    collaborator_model,
    normalize_email_fn,
    role_allows_fn,
) -> dict:
    if not viewer_email or artifact.project_id is None:
        return _empty_review_viewer_state()
    try:
        email = normalize_email_fn(viewer_email)
    except ValueError:
        return _empty_review_viewer_state()
    from sqlmodel import select

    collaborator = session.exec(
        select(collaborator_model)
        .where(collaborator_model.project_id == artifact.project_id)
        .where(collaborator_model.email == email)
    ).first()
    if collaborator is None:
        return _empty_review_viewer_state()

    artifact_state = dict((collaborator.notification_state or {}).get("artifacts", {}).get(str(artifact.id), {}) or {})
    last_read_at = _parse_review_dt(artifact_state.get("lastReadAt"))
    role = collaborator.role
    unread_comments = 0
    unread_activity = 0
    open_review_requests = 0
    needs_action = 0
    unread_review_requests = 0

    for comment in comments:
        own_comment = _same_email(_comment_actor_email(comment), collaborator.email)
        unread = bool(not own_comment and (last_read_at is None or _created_after(comment.created_at, last_read_at)))
        closed = comment.status in {"resolved", "accepted", "rejected"}
        review_request = comment.kind == "review" and not closed
        needs = bool(review_request and role_allows_fn(role, "review") and not own_comment)
        if unread:
            unread_comments += 1
        if review_request:
            open_review_requests += 1
            if unread:
                unread_review_requests += 1
        if needs:
            needs_action += 1

    for event in events:
        own_event = _same_email(_event_actor_email(event), collaborator.email)
        if not own_event and (last_read_at is None or _created_after(event.created_at, last_read_at)):
            unread_activity += 1

    return {
        "available": True,
        "collaboratorId": str(collaborator.id),
        "role": role,
        "lastReadAt": last_read_at.isoformat() if last_read_at else None,
        "unreadComments": unread_comments,
        "unreadActivity": unread_activity,
        "needsAction": needs_action,
        "reviewRequests": {
            "open": open_review_requests,
            "needsAction": needs_action,
            "unread": unread_review_requests,
        },
    }


def _comment_actor_email(comment) -> str | None:
    state = comment.notification_state if isinstance(comment.notification_state, dict) else {}
    return state.get("actorEmail") or state.get("actor_email")


def _event_actor_email(event) -> str | None:
    details = event.details if isinstance(event.details, dict) else {}
    return details.get("actorEmail") or details.get("actor_email")


def _same_email(left: str | None, right: str | None) -> bool:
    return bool(left and right and left.strip().lower() == right.strip().lower())


def _created_after(created_at: datetime | None, marker: datetime | None) -> bool:
    if created_at is None or marker is None:
        return False
    created = _normalize_review_dt(created_at)
    marked = _normalize_review_dt(marker)
    return bool(created and marked and created > marked)


def _parse_review_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return _normalize_review_dt(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None


def _normalize_review_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _project_artifacts_base(project_name: str) -> Path | None:
    """Resolve a project name to its `.anton/artifacts` dir, only when it
    maps to a registered project. Returns None for unknown projects or
    path-traversal attempts."""
    if (not project_name or "\x00" in project_name
            or "/" in project_name or "\\" in project_name
            or project_name in (".", "..")):
        return None
    registered = set(_registered_project_dirs())
    root = _projects_root().resolve(strict=False)
    try:
        candidate = (root / project_name).resolve(strict=False)
    except (OSError, ValueError):
        return None
    if candidate not in registered:
        return None
    base = candidate / ".anton" / "artifacts"
    return base if base.is_dir() else None


def serve_url_for(path: str | Path, *, signed: bool = True) -> str:
    """Origin-relative `/api/v1/artifacts/serve/...` URL for a file under a
    project's `.anton/artifacts` tree. Returns "" when the path isn't
    inside such a tree."""
    try:
        p = Path(path).resolve(strict=False)
    except (OSError, ValueError):
        return ""
    for project_dir in _registered_project_dirs():
        base = project_dir / ".anton" / "artifacts"
        try:
            rel = p.relative_to(base.resolve())
        except (ValueError, OSError):
            continue
        if not rel.parts:
            return ""
        rel_str = "/".join(quote(part) for part in rel.parts)
        url = f"/api/v1/artifacts/serve/{quote(project_dir.name)}/{rel_str}"
        if signed:
            return f"{url}?token={quote(sign_serve_url_path(p))}"
        return url
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


def _unpublish_folder(folder: Path) -> None:
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
        # The path-based unpublish needs the file present; a stale record
        # for a missing file can't be unpublished this way, so skip it.
        if not (folder / name).is_file():
            continue
        unpublish_artifact(str(folder / name))


def delete_artifact(raw_path: str) -> None:
    """DEPRECATED — unused in production; hard-deletes without recovery.

    The delete_artifact_endpoint uses inline tombstone logic (rename to a
    `.delete-<uuid>` folder, recoverable via the "deleted" activity event),
    NOT this function. This path does an unrecoverable ``shutil.rmtree`` and
    is retained only because tests still exercise it. Do not wire it into new
    code; prefer the endpoint's tombstone flow.

    If the artifact has published files, they are unpublished from the
    remote first. If any unpublish fails, the artifact is left on disk
    and the error propagates to the caller.
    """
    target = resolve_artifact_path(raw_path)
    if target is None:
        raise ValueError("Invalid artifact path")

    if target.is_dir() and (target / "metadata.json").exists():
        folder = target
    elif target.is_file():
        folder = target.parent
        if not (folder / "metadata.json").exists():
            raise ValueError("Not a valid artifact folder")
    else:
        raise FileNotFoundError("Artifact not found")

    for art_root in _scan_artifact_dirs():
        try:
            folder.relative_to(art_root.resolve())
        except ValueError:
            continue
        from cowork.db.session import get_open_session
        from cowork.services.artifact_versions import ArtifactVersionService

        with get_open_session(get_app_settings().database.uri) as session:
            ArtifactVersionService(session).snapshot_artifact(
                folder,
                operation_type="pre_delete",
                label="Before delete",
            )
        # Unpublish before deleting; if this raises, the artifact stays.
        _unpublish_folder(folder)
        shutil.rmtree(folder)
        return
    raise FileNotFoundError("Artifact is not in a known artifacts directory")


def reveal_in_file_manager(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=False)


# ─── Public API ───────────────────────────────────────────────────

def card_for_folder(folder: Path, idx: int = 0, *, viewer_email: str | None = None) -> dict | None:
    """The artifact card for a single folder, or ``None`` if its metadata is
    unreadable. This is the canonical card shape — used both by the artifacts
    list and by the inline chat cards (see services.task_objects), so the two
    can never disagree about how an artifact is named, typed, or opened.

    `idx` only selects a background gradient (cosmetic)."""
    meta = _load_metadata(folder)
    if meta is None:
        return None
    files = _user_files(folder)
    primary = _pick_primary(folder, files, primary_hint=meta.get("primary"))
    primary_path = str(primary) if primary is not None else str(folder)
    primary_ext = primary.suffix.lower() if primary is not None else ""
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
    content_mtime = _content_mtime(folder)

    return {
        "id": meta.get("id") or folder.name,
        "slug": meta.get("slug") or folder.name,
        "title": meta.get("name") or folder.name,
        "description": meta.get("description") or "",
        "type": artifact_type,
        "kind": kind,
        "ext": primary_ext,
        "updated": _human_mtime(folder / "metadata.json"),
        "mtime": content_mtime,
        "live": is_live,
        "bg": BG_CYCLE[idx % len(BG_CYCLE)],
        "fileCount": len(files),
        "folder": str(folder),
        "path": primary_path,
        "primary": meta.get("primary") or None,
        "publishedUrl": _published_url_for(folder, primary),
        **_published_version_for(folder, primary),
        "modified": _is_modified(folder, primary, content_mtime),
        # Owner-side access state (lock badge + eye-reveal). accessPassword
        # is the plaintext, returned only to the owner's own session.
        **_published_access_for(folder, primary),
        **_last_good_preview_for(folder),
        "reviewSummary": _review_summary_for(folder, viewer_email=viewer_email),
        "serveUrl": serve_url_for(primary_path),
    }


def list_artifacts(project_path: str | None = None, *, viewer_email: str | None = None) -> list[dict]:
    """Every artifact across all projects, newest first."""
    cards: list[dict] = []
    for folder in _iter_artifact_folders(project_path):
        card = card_for_folder(folder, len(cards), viewer_email=viewer_email)
        if card is None:
            continue
        try:
            card["_sortTs"] = (folder / "metadata.json").stat().st_mtime
        except OSError:
            card["_sortTs"] = 0.0
        cards.append(card)

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
        # Proxy mode. Register the artifact root (where metadata.json +
        # the live port live) so the proxy endpoint reads a current port
        # by token, then auto-launch the backend if dead. Returns without
        # `proxyUrl` — the route layer fills it in using the incoming
        # Request URL so the absolute URL matches whatever host/scheme
        # the client used to reach us.
        root_token = register_preview_mount(root, ttl_seconds=_PREVIEW_MOUNT_TTL_SECONDS)
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
            **_published_version_for(root, path),
            **_published_access_for(root, path),
        }

    # Static (HTML) branch — same behaviour as before, with an explicit
    # `kind` discriminator so the client doesn't have to infer.
    if path.suffix.lower() != ".html":
        raise ValueError("Preview mount is only available for HTML artifacts")
    token = register_preview_mount(parent)

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
        **_published_version_for(parent, path),
        **_published_access_for(parent, path),
    }


def get_preview_mount(token: str) -> Path | None:
    expires = _PREVIEW_MOUNT_EXPIRES.get(token)
    if expires is None:
        return None
    if expires < time.time():
        _PREVIEW_MOUNTS.pop(token, None)
        _PREVIEW_MOUNT_EXPIRES.pop(token, None)
        return None
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
                    **_published_version_for(artifact_root, entry_path),
                })
                continue

            seen.add(key)
            out.append({
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "path": str(path),
                "bytes": path.stat().st_size,
                "publishedUrl": _published_url_for(path.parent, path),
                **_published_version_for(path.parent, path),
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
    """
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
    result = await launch_artifact_backend(
        slug=slug,
        artifact_folder=artifact_dir,
        scratchpad_pool=pool,
        tracked_backends=_LAUNCHED_BACKENDS,
        extra_env=extra_env,
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
        # Atomic write: a concurrent reader (the proxy reads metadata.json on
        # every request) must never observe a truncated file, or the artifact
        # vanishes from listings. Write a temp file in the same dir, then
        # os.replace it into place (atomic on the same filesystem).
        payload = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
        with NamedTemporaryFile(
            "w",
            prefix=".metadata.",
            suffix=".tmp",
            dir=meta_path.parent,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(payload)
        try:
            os.replace(tmp_path, meta_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
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
