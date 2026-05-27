"""Artifacts service — filesystem operations for agent-produced outputs.

Each artifact is a folder under `<project>/.anton/artifacts/<slug>/`
containing a `metadata.json` and user files. This module handles
listing, resolving paths, and managing preview mounts.

Ported from cowork/server/routes/artifacts.py with projects_store
replaced by the DB-backed project service.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

from cowork.common.settings.app_settings import get_app_settings

logger = logging.getLogger(__name__)

# In-memory registry: deterministic token → parent dir of an HTML artifact.
_PREVIEW_MOUNTS: dict[str, Path] = {}

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


def _published_url_for(folder: Path, primary: Path | None) -> str:
    if primary is None:
        return ""
    published_index = folder / ".published.json"
    if not published_index.is_file():
        return ""
    try:
        pmap = json.loads(published_index.read_text(encoding="utf-8"))
        entry = pmap.get(primary.name)
        if isinstance(entry, dict):
            return entry.get("url", "") or ""
    except Exception:
        pass
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
        if target.is_file():
            matches[str(target)] = target
    return list(matches.values())


def resolve_artifact_path(raw_path: str) -> Path | None:
    """Turn an artifact request path into an absolute path on disk.

    Returns None if invalid, or the resolved Path if found.
    Raises ValueError with a message for 400/404 cases.
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
            if resolved.is_file():
                return resolved
        raise FileNotFoundError("Artifact is not in a known artifacts directory")

    matches = _candidate_relative_artifacts(raw_path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Artifact path matches multiple project artifact roots; pass an absolute path")
    raise FileNotFoundError("Artifact is not in a known artifacts directory")


def reveal_in_file_manager(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=False)


# ─── Public API ───────────────────────────────────────────────────

def list_artifacts(project_path: str | None = None) -> list[dict]:
    """Every artifact across all projects, newest first."""
    cards: list[dict] = []
    for folder in _iter_artifact_folders(project_path):
        meta = _load_metadata(folder)
        if meta is None:
            continue
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
        idx = len(cards) % len(BG_CYCLE)
        sort_ts: float
        try:
            sort_ts = (folder / "metadata.json").stat().st_mtime
        except OSError:
            sort_ts = 0.0

        cards.append({
            "id": meta.get("id") or folder.name,
            "slug": meta.get("slug") or folder.name,
            "title": meta.get("name") or folder.name,
            "description": meta.get("description") or "",
            "type": artifact_type,
            "kind": kind,
            "ext": primary_ext,
            "updated": _human_mtime(folder / "metadata.json"),
            "live": is_live,
            "bg": BG_CYCLE[idx],
            "fileCount": len(files),
            "folder": str(folder),
            "path": primary_path,
            "primary": meta.get("primary") or None,
            "publishedUrl": _published_url_for(folder, primary),
            "_sortTs": sort_ts,
        })

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


def mount_preview(path: Path) -> dict:
    """Register an HTML artifact's parent dir for iframe preview."""
    if path.suffix.lower() != ".html":
        raise ValueError("Preview mount is only available for HTML artifacts")
    parent = path.parent.resolve()
    token = hashlib.sha256(str(parent).encode("utf-8")).hexdigest()[:16]
    _PREVIEW_MOUNTS[token] = parent

    published_url = ""
    published_path = parent / ".published.json"
    if published_path.is_file():
        try:
            pmap = json.loads(published_path.read_text(encoding="utf-8"))
            entry = pmap.get(path.name)
            if isinstance(entry, dict):
                published_url = entry.get("url", "") or ""
        except Exception:
            pass

    return {
        "token": token,
        "entry": path.name,
        "relUrl": f"/artifacts/preview-asset/{token}/{path.name}",
        "publishedUrl": published_url,
    }


def get_preview_mount(token: str) -> Path | None:
    return _PREVIEW_MOUNTS.get(token)


def html_artifacts() -> list[dict]:
    """List every HTML file under every project's artifacts tree for publish."""
    out = []
    seen: set[str] = set()
    for art_root in _scan_artifact_dirs():
        if not art_root.exists():
            continue
        for path in sorted(art_root.rglob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            published_path = path.parent / ".published.json"
            published: dict = {}
            if published_path.is_file():
                try:
                    published = json.loads(published_path.read_text(encoding="utf-8")).get(path.name, {})
                except Exception:
                    published = {}
            out.append({
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "path": str(path),
                "bytes": path.stat().st_size,
                "publishedUrl": published.get("url", "") if isinstance(published, dict) else "",
            })
    return out[:40]
