from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_CONNECTOR_SPECS_DIR = Path(__file__).resolve().parent / "specs"
_CACHE: dict[str, dict] | None = None


def _load_all() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not _CONNECTOR_SPECS_DIR.is_dir():
        return out
    for path in sorted(_CONNECTOR_SPECS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # A broken file shouldn't break the whole registry.
            continue
        if not isinstance(data, dict):
            continue
        cid = data.get("id") or path.stem
        data["id"] = cid
        out[cid] = data
    return out


def all_connectors() -> dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_all()
    return _CACHE


def get_connector(cid: str) -> dict | None:
    return all_connectors().get(cid)


def list_summaries() -> list[dict[str, Any]]:
    """Lightweight records suitable for picker UIs — no field schemas."""
    out: list[dict[str, Any]] = []
    for c in all_connectors().values():
        out.append({
            "id": c.get("id"),
            "label": c.get("label", c.get("id")),
            "description": c.get("description", ""),
            "category": c.get("category", "other"),
            "logo": c.get("logo"),
            "logo_url": c.get("logo_url"),
            "logo_color": c.get("logo_color"),
            "aliases": c.get("aliases", []),
            "featured": c.get("featured", False),
        })
    out.sort(key=lambda x: (x.get("label") or "").lower())
    return out


def reload_connectors() -> None:
    """Force a re-read from disk. Useful in dev/testing."""
    global _CACHE
    _CACHE = None
