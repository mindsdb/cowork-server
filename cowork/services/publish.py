"""Publish service — publish HTML artifacts to 4nton.ai.

Ported from cowork/server/routes/utilities.py (publish section).
Uses a local JSON state file for publish history tracking.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from cowork.common.settings.user_settings import get_user_settings
from cowork.services.artifacts import (
    _artifact_root_for,
    _fullstack_types,
    _load_metadata,
    _pick_primary,
    _user_files,
    html_artifacts,
    resolve_artifact_path,
)

logger = logging.getLogger(__name__)


def _cowork_state_dir() -> Path:
    base = os.environ.get("ANTON_COWORK_STATE_DIR")
    if base:
        path = Path(base).expanduser()
    else:
        path = Path.home() / ".anton" / "cowork"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path() -> Path:
    return _cowork_state_dir() / "state.json"


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)


def _secret_str(val: SecretStr | str | None) -> str:
    """Unwrap a SecretStr (or plain string) to a plain string, defaulting to ''."""
    if val is None:
        return ""
    if isinstance(val, SecretStr):
        return val.get_secret_value()
    return str(val)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_publish_target(artifact: Path) -> tuple[Path, Path, str, bool]:
    """Decide what to publish given a resolved artifact path (file OR dir).

    Folder-based artifacts are addressed by their folder; legacy loose-HTML
    (and chat-bubble / Utilities-list) artifacts by their file. In both cases:
      - fullstack (metadata.json type ∈ FULLSTACK_ARTIFACT_TYPES) → publish
        the artifact *directory* (anton bundles backend.py + static/ +
        requirements.txt only when handed a dir), `.published.json` at root;
      - static → publish the single primary *file* (anton's `_zip_html`
        renames it to index.html + pulls referenced siblings; handing it a
        dir would over-bundle metadata.json/data and skip the rename),
        `.published.json` in that file's parent.
    The map is always keyed by the primary file name — matching how
    `_published_url_for` / `list_artifacts` read it back.

    Returns (publish_target, published_dir, published_key, is_fullstack).
    """
    if artifact.is_dir():
        # Folder addressed directly — its primary file lives inside.
        artifact_root = artifact
        meta = _load_metadata(artifact_root) or {}
        primary = _pick_primary(artifact_root, _user_files(artifact_root), primary_hint=meta.get("primary"))
    else:
        # File addressed directly — it *is* the primary; climb to its root.
        artifact_root = _artifact_root_for(artifact)
        meta = _load_metadata(artifact_root) if (artifact_root / "metadata.json").is_file() else None
        primary = artifact

    if (meta or {}).get("type") in _fullstack_types():
        key = primary.name if primary else "index.html"
        return artifact_root, artifact_root, key, True
    if primary:
        return primary, primary.parent, primary.name, False
    return artifact_root, artifact_root, "index.html", False


def list_publishable() -> dict:
    settings = get_user_settings()
    state = _load_state()
    return {
        "artifacts": html_artifacts(),
        "publishReady": bool(_secret_str(settings.minds_api_key)),
        "publishUrl": settings.publish_url or "https://4nton.ai",
        "history": state.get("publish_history", [])[:40],
    }


def publish_artifact(raw_path: str, password: str | None = None) -> dict:
    settings = get_user_settings()
    api_key = _secret_str(settings.minds_api_key)
    if not api_key:
        raise ValueError("Configure your Minds API key in Settings before publishing")

    # The request path is either the artifact folder (folder-based
    # artifacts) or a single file (legacy loose-HTML / chat-bubble / the
    # Utilities per-page list). `_resolve_publish_target` normalizes both.
    artifact = resolve_artifact_path(raw_path, allow_dir=True)
    publish_target, published_dir, published_key, is_fullstack = _resolve_publish_target(artifact)
    if not is_fullstack and publish_target.suffix.lower() != ".html":
        raise ValueError("Only HTML artifacts can be published")

    try:
        from anton.publisher import publish
    except Exception as exc:
        raise RuntimeError("Anton publisher is unavailable") from exc

    published_json = published_dir / ".published.json"
    published_map: dict[str, Any] = {}
    if published_json.is_file():
        try:
            published_map = json.loads(published_json.read_text(encoding="utf-8"))
        except Exception:
            published_map = {}
    previous = published_map.get(published_key)
    report_id = previous.get("report_id") if isinstance(previous, dict) else None

    # Resolve access state. A non-empty password publishes (or keeps) the
    # artifact password-protected; empty/None publishes it public. Bump
    # pwd_version whenever the password value changes so access cookies
    # issued for the old password stop validating in the viewer.
    password = (password or "").strip() or None
    prev_password = previous.get("access_password") if isinstance(previous, dict) else None
    prev_version = previous.get("pwd_version", 0) if isinstance(previous, dict) else 0
    pwd_version = (prev_version + 1) if password and password != prev_password else (prev_version or 1)

    publish_url = settings.publish_url or "https://4nton.ai"
    ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true"
    try:
        result = publish(
            publish_target,
            api_key=api_key,
            report_id=report_id,
            publish_url=publish_url,
            ssl_verify=ssl_verify,
            password=password,
            pwd_version=pwd_version,
        )
    except Exception as exc:
        logger.exception("Publishing failed")
        raise RuntimeError("Publishing failed. Check your Minds credentials and try again.") from exc

    view_url = result.get("view_url", "")
    returned_report_id = result.get("report_id", "")
    if returned_report_id:
        history_item = {
            "artifact": str(publish_target),
            "artifactName": published_key,
            "url": view_url,
            "reportId": returned_report_id,
            "publishedAt": _utc_now_iso(),
        }
        entry: dict[str, Any] = {
            "report_id": returned_report_id,
            "url": view_url,
            "last_md5": result.get("md5", ""),
            "requires_password": bool(password),
        }
        if password:
            # Owner-side only — .published.json never enters the bundle.
            entry["access_password"] = password
            entry["pwd_version"] = pwd_version
        published_map[published_key] = entry
        try:
            published_json.write_text(json.dumps(published_map, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        state = _load_state()
        state["publish_history"] = [history_item, *state.get("publish_history", [])][:100]
        _save_state(state)

    return {
        "status": "ok",
        "url": view_url,
        "accessProtected": bool(password),
        "result": {k: v for k, v in result.items() if k != "file_payload"},
    }


def unpublish_artifact(raw_path: str) -> dict:
    settings = get_user_settings()
    api_key = _secret_str(settings.minds_api_key)
    if not api_key:
        raise ValueError("Configure your Minds API key in Settings before unpublishing")

    artifact = resolve_artifact_path(raw_path, allow_dir=True)
    # Mirror publish: resolve the same .published.json location + key
    # (primary file name) whether a folder or a file was passed.
    _publish_target, published_dir, published_key, _is_fullstack = _resolve_publish_target(artifact)
    published_json = published_dir / ".published.json"
    if not published_json.is_file():
        raise FileNotFoundError("Artifact has no publish record")

    try:
        published_map: dict[str, Any] = json.loads(published_json.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("Could not read publish record") from exc

    entry = published_map.get(published_key)
    identifier = None
    if isinstance(entry, dict):
        identifier = entry.get("report_id") or entry.get("last_md5") or None
    if not identifier:
        raise FileNotFoundError("No published version on file")

    try:
        from anton.publisher import unpublish
    except Exception as exc:
        raise RuntimeError("Anton publisher is unavailable") from exc

    publish_url = settings.publish_url or "https://4nton.ai"
    ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true"
    try:
        unpublish(
            identifier,
            api_key=api_key,
            publish_url=publish_url,
            ssl_verify=ssl_verify,
        )
    except Exception as exc:
        msg = str(exc) or "Unpublishing failed."
        if "404" in msg or "not found" in msg.lower():
            pass  # Already gone upstream — clear local record below
        else:
            logger.exception("Unpublishing failed (identifier=%s)", identifier)
            raise RuntimeError(f"Unpublishing failed: {msg}") from exc

    published_map.pop(published_key, None)
    try:
        if published_map:
            published_json.write_text(json.dumps(published_map, indent=2) + "\n", encoding="utf-8")
        else:
            published_json.unlink()
    except Exception:
        pass
    return {"status": "ok"}
