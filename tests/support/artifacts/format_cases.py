"""Shared artifact shapes and builders for format-boundary checks."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from cowork.services.artifacts import ProjectArtifacts


def _write_artifact(base: Path, slug: str, files: dict[str, str], **metadata) -> tuple[str, Path]:
    artifact_id = uuid4().hex
    folder = base / slug
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (folder / "metadata.json").write_text(
        json.dumps({"id": artifact_id, "slug": slug, "name": slug, **metadata}), encoding="utf-8"
    )
    return artifact_id, folder

def _local_source(tmp_path: Path) -> ProjectArtifacts:
    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    base.mkdir(parents=True)
    return ProjectArtifacts(
        base=base,
        project_id=None,
        project_name="project",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )


#: slug -> (files, metadata, the formats its card lists)
_LOCAL_ARTIFACTS = {
    "report": ({"report.md": "# R\n"}, {"type": "document", "primary": "report.md"}, ["html", "docx"]),
    "notes": ({"notes.txt": "n\n"}, {"type": "document"}, ["html", "docx"]),
    "page": ({"index.html": "<h1>P</h1>"}, {"type": "html-app", "primary": "index.html"}, ["docx"]),
    "sheet": ({"data.csv": "a\n"}, {"type": "data", "primary": "data.csv"}, []),
    "app": (
        {"static/index.html": "<h1>A</h1>", "backend.py": "x = 1\n"},
        {"type": "fullstack-stateless-app", "primary": "static/index.html"},
        [],
    ),
    "readme": ({"README.md": "# R\n"}, {"type": "document", "primary": "README.md"}, []),
    "nested-readme": (
        {"docs/README.md": "# R\n"},
        {"type": "document", "primary": "docs/README.md"},
        ["html", "docx"],
    ),
    # Windows reads aux.md as a device, so the export's read refuses the name.
    "device": ({"aux.md": "# A\n"}, {"type": "document", "primary": "aux.md"}, []),
    "revision": (
        {".revisions/old.md": "# R\n", "current.md": "# C\n"},
        {"type": "document", "primary": ".revisions/old.md"},
        [],
    ),
}
