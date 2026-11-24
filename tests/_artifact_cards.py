"""A card for tests that fake what the card builder returns."""
from __future__ import annotations

from cowork.schemas.artifact_card import ArtifactCard


def make_card(**fields) -> ArtifactCard:
    values = {
        "id": "0123456789abcdef0123456789abcdef", "slug": "dash", "title": "Dash", "description": "",
        "type": "html-app", "kind": "Dashboard", "ext": ".html", "updated": "just now", "mtime": 0,
        "live": False, "bg": "#ffffff", "file_count": 1, "folder": "/p/dash", "path": "/p/dash/index.html",
        "primary": "index.html", "project_id": None, "project_name": "", "origin_conversation_id": "",
        "serve_url": "",
    }
    values.update(fields)
    return ArtifactCard(**values)
