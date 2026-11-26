"""A metadata.json anton's Artifact model accepts, for tests that need one."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

_STAMP = "2026-11-23T00:00:00+00:00"


def write_artifact_metadata(
    folder: Path, *, slug: str, artifact_type: str, datasources: list[dict] | None = None
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(json.dumps({
        "id": uuid.uuid4().hex,
        "slug": slug,
        "createdAt": _STAMP,
        "updatedAt": _STAMP,
        "name": slug,
        "description": f"{slug} test artifact",
        "type": artifact_type,
        "datasources": datasources or [],
    }))
    return folder
