"""Hermes artifact tools (ENG-287).

The handlers reuse anton-core's ArtifactStore, so what they write must be
exactly what the (harness-agnostic) artifacts service lists — that contract
is what puts Hermes outputs in the Artifacts UI. These tests exercise the
handlers the way run_agent's registry dispatch does: plain calls with the
run's task_id passed as a kwarg.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cowork.harnesses.hermes_harness.tools import (
    _hermes_create_artifact,
    _hermes_list_artifacts,
    finalize_artifact_run_context,
    register_artifact_tools,
    set_artifact_run_context,
)

TASK_ID = "test-conversation-1"


def _make_ctx(tmp: Path) -> Path:
    root = tmp / ".anton" / "artifacts"
    set_artifact_run_context(
        TASK_ID,
        artifacts_root=root,
        conversation_id=TASK_ID,
        conversation_title="Quarterly numbers",
        turn_summary="build me a dashboard",
    )
    return root


def test_create_artifact_writes_anton_convention_folder():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_ctx(Path(tmp))
        try:
            result = json.loads(
                _hermes_create_artifact(
                    {
                        "name": "Sales Dashboard",
                        "description": "Q2 sales by region",
                        "type": "html-app",
                        "primary": "dashboard.html",
                    },
                    task_id=TASK_ID,
                )
            )
            assert result["slug"] == "sales-dashboard"
            folder = Path(result["path"])
            assert folder == root / "sales-dashboard"
            metadata = json.loads((folder / "metadata.json").read_text())
            assert metadata["name"] == "Sales Dashboard"
            assert metadata["type"] == "html-app"
            assert metadata["primary"] == "dashboard.html"
            # Provenance recorded for the conversation.
            assert metadata["provenance"][0]["conversation"] == TASK_ID
            assert (folder / "README.md").is_file()
        finally:
            finalize_artifact_run_context(TASK_ID)


def test_finalize_rescans_files_written_by_the_agent():
    with tempfile.TemporaryDirectory() as tmp:
        _make_ctx(Path(tmp))
        result = json.loads(
            _hermes_create_artifact(
                {"name": "Report", "description": "d", "type": "document"},
                task_id=TASK_ID,
            )
        )
        folder = Path(result["path"])
        # Hermes writes files with its own file tools — the store only
        # learns about them at finalize time.
        (folder / "report.md").write_text("# hi")
        finalize_artifact_run_context(TASK_ID)
        metadata = json.loads((folder / "metadata.json").read_text())
        assert [f["path"] for f in metadata["files"]] == ["report.md"]


def test_list_artifacts_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        _make_ctx(Path(tmp))
        try:
            _hermes_create_artifact(
                {"name": "One", "description": "d", "type": "dataset"},
                task_id=TASK_ID,
            )
            listed = json.loads(_hermes_list_artifacts({}, task_id=TASK_ID))
            assert [a["slug"] for a in listed] == ["one"]
            assert listed[0]["type"] == "dataset"
        finally:
            finalize_artifact_run_context(TASK_ID)


def test_handlers_fail_softly_without_run_context():
    result = json.loads(_hermes_create_artifact({"name": "x", "description": "d", "type": "document"}, task_id="missing"))
    assert "error" in result
    result = json.loads(_hermes_list_artifacts({}, task_id="missing"))
    assert "error" in result


def test_create_rejects_unknown_type():
    with tempfile.TemporaryDirectory() as tmp:
        _make_ctx(Path(tmp))
        try:
            result = json.loads(
                _hermes_create_artifact(
                    {"name": "x", "description": "d", "type": "spreadsheet"},
                    task_id=TASK_ID,
                )
            )
            assert "error" in result and "type" in result["error"]
        finally:
            finalize_artifact_run_context(TASK_ID)


def test_register_artifact_tools_is_idempotent():
    register_artifact_tools()
    register_artifact_tools()
    from tools.registry import registry

    entry = registry.get_entry("create_artifact")
    assert entry is not None
    schema_types = entry.schema["parameters"]["properties"]["type"]["enum"]
    # Schema enum mirrors anton-core's closed type set.
    from anton.core.artifacts.models import ARTIFACT_TYPES

    assert schema_types == sorted(ARTIFACT_TYPES)
    assert registry.get_entry("list_artifacts") is not None


def test_registry_dispatch_forwards_task_id_to_handler():
    """run_agent invokes tools via registry.dispatch(name, args, task_id=...);
    pin that the context lookup works through that exact path."""
    register_artifact_tools()
    from tools.registry import registry

    with tempfile.TemporaryDirectory() as tmp:
        _make_ctx(Path(tmp))
        try:
            result = json.loads(
                registry.dispatch(
                    "create_artifact",
                    {"name": "Via Dispatch", "description": "d", "type": "document"},
                    task_id=TASK_ID,
                )
            )
            assert result["slug"] == "via-dispatch"
        finally:
            finalize_artifact_run_context(TASK_ID)


def test_created_artifact_appears_in_cowork_listing():
    """End contract: the artifacts service (which feeds the UI) must list
    what the Hermes tool created. The service only scans registered
    projects, so this uses the seeded `general` project."""
    from cowork.services import artifacts as artifacts_service
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.session import get_engine
    from cowork.models.project import Project
    from cowork.services.projects import GENERAL_PROJECT_ID
    from sqlmodel import Session

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = Path(session.get(Project, GENERAL_PROJECT_ID).path)

    set_artifact_run_context(
        TASK_ID,
        artifacts_root=project / ".anton" / "artifacts",
        conversation_id=TASK_ID,
        conversation_title="t",
        turn_summary="s",
    )
    result = json.loads(
        _hermes_create_artifact(
            {
                "name": "Hermes Dash",
                "description": "made by hermes",
                "type": "html-app",
                "primary": "index.html",
            },
            task_id=TASK_ID,
        )
    )
    (Path(result["path"]) / "index.html").write_text("<html></html>")
    finalize_artifact_run_context(TASK_ID)

    listed = artifacts_service.list_artifacts(str(project))
    # The general project is shared across the test session — assert on
    # our artifact rather than the full listing.
    entry = next((a for a in listed if a["slug"] == "hermes-dash"), None)
    assert entry is not None
    assert entry["title"] == "Hermes Dash"
    assert entry["primary"] == "index.html"
