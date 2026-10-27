"""Which formats an artifact's Export offers, and where ``exportFormats`` shows.

``export_formats`` is the one place that decides. The list cards and the
workspace response carry its answer in ``capabilities.exportFormats``, and the
export route accepts exactly the formats that list holds.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.schemas.artifact_export import ExportPdfSettings
from cowork.services import artifact_export, artifact_permissions
from cowork.services import artifact_ownership as ownership
from cowork.services.artifact_export import (
    ExportSource,
    card_export_source,
    export_formats,
    export_pdf_settings,
)
from cowork.services.artifact_permissions import artifact_capabilities
from cowork.services.artifacts import ProjectArtifacts
from test_artifact_ownership import make_project, project_root

FIXTURES = Path(__file__).parent / "fixtures" / "artifact_export"
FORMAT_CASES = yaml.safe_load((FIXTURES / "formats.yaml").read_text(encoding="utf-8"))["cases"]

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


@pytest.mark.parametrize("case", FORMAT_CASES, ids=[case["id"] for case in FORMAT_CASES])
def test_export_formats_follow_the_primary_file_and_the_caller(case):
    pdf = case.get("pdf") or {"engine": None, "render_url": None}

    offered = export_formats(
        source_name=case["source_name"],
        is_backend=case.get("is_backend", False),
        can_edit=case.get("can_edit", True),
        pdf=ExportPdfSettings(**pdf),
    )

    assert offered == case["expected"]


@pytest.mark.parametrize(("depth", "expected"), [(64, ["html", "docx"]), (65, [])])
def test_a_primary_nested_past_the_read_depth_offers_nothing(depth, expected):
    """The export's read follows at most 64 path segments."""
    source_name = "/".join(["d"] * (depth - 1) + ["report.md"])

    offered = export_formats(
        source_name=source_name, is_backend=False, can_edit=True, pdf=export_pdf_settings()
    )

    assert offered == expected


def test_no_deployment_offers_pdf_until_a_render_service_is_configured():
    settings = export_pdf_settings()

    assert settings.engine is None
    assert settings.render_url is None
    assert "pdf" not in export_formats(
        source_name="report.md", is_backend=False, can_edit=True, pdf=settings
    )


@pytest.mark.parametrize(
    ("primary", "artifact_type", "expected"),
    [
        ("report.md", "document", ExportSource(source_name="report.md", is_backend=False)),
        ("docs/report.md", "mixed", ExportSource(source_name="docs/report.md", is_backend=False)),
        (
            "static/index.html",
            "fullstack-stateful-app",
            ExportSource(source_name="static/index.html", is_backend=True),
        ),
        # A card with no primary file carries its folder as its path.
        (None, "mixed", ExportSource(source_name="", is_backend=False)),
    ],
)
def test_a_card_names_its_primary_file_and_type(tmp_path, primary, artifact_type, expected):
    """The primary file is on disk, since a card checks it has one link."""
    folder = tmp_path / "artifact"
    folder.mkdir()
    path = folder
    if primary is not None:
        path = folder / primary
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    card = {"folder": str(folder), "path": str(path), "type": artifact_type}

    assert card_export_source(card) == expected


# ── capabilities ─────────────────────────────────────────────────────────────


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


@pytest.fixture
def local_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def test_local_cards_carry_export_formats(tmp_path, local_mode):
    source = _local_source(tmp_path)
    for slug, (files, metadata, _expected) in _LOCAL_ARTIFACTS.items():
        _write_artifact(source.base, slug, files, **metadata)

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        cards = artifacts_ep._artifact_cards(ScopedSession(raw, LOCAL_SCOPE), [source])

    listed = {card["slug"]: card["capabilities"]["exportFormats"] for card in cards}
    assert listed == {slug: expected for slug, (_f, _m, expected) in _LOCAL_ARTIFACTS.items()}


def test_the_local_workspace_response_carries_export_formats(tmp_path, local_mode, monkeypatch):
    source = _local_source(tmp_path)
    artifact_id, folder = _write_artifact(
        source.base, "report", {"report.md": "# R\n"}, type="document", primary="report.md"
    )
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        lambda *_args: (source, folder, json.loads((folder / "metadata.json").read_text()), True),
    )
    from cowork.server import create_app

    response = TestClient(create_app()).get(f"/api/v1/artifacts/workspace/local/{artifact_id}")

    assert response.status_code == 200, response.text
    assert response.json()["capabilities"]["exportFormats"] == ["html", "docx"]


def test_the_workspace_response_decides_export_formats_where_artifact_capabilities_does(
    tmp_path, local_mode, monkeypatch
):
    """The workspace response already holds its owner's capabilities, and adds
    exportFormats through the helper artifact_capabilities() adds it with. A
    change to how the list is decided then reaches both answers alike."""
    source = _local_source(tmp_path)
    artifact_id, folder = _write_artifact(
        source.base, "report", {"report.md": "# R\n"}, type="document", primary="report.md"
    )
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        lambda *_args: (source, folder, json.loads((folder / "metadata.json").read_text()), True),
    )
    monkeypatch.setattr(artifact_permissions, "offered_export_formats", lambda **_kwargs: ["docx"])
    from cowork.server import create_app

    response = TestClient(create_app()).get(f"/api/v1/artifacts/workspace/local/{artifact_id}")
    capabilities = artifact_capabilities(
        SimpleNamespace(scope=LOCAL_SCOPE),
        source,
        "report",
        export_source=ExportSource(source_name="report.md", is_backend=False),
    )

    assert response.status_code == 200, response.text
    assert capabilities["exportFormats"] == ["docx"]
    assert response.json()["capabilities"]["exportFormats"] == ["docx"]


def test_a_hard_linked_primary_offers_nothing_anywhere_and_the_route_answers_422(
    tmp_path, local_mode, monkeypatch
):
    """A sync or backup tool can give the primary a second hard link. The
    export's read refuses such a file, so its card and the workspace response
    offer no format, and the route answers 422, as for any format not offered."""
    source = _local_source(tmp_path)
    artifact_id, folder = _write_artifact(
        source.base, "report", {"report.md": "# R\n"}, type="document", primary="report.md"
    )
    os.link(folder / "report.md", tmp_path / "backup.md")
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        cards = artifacts_ep._artifact_cards(ScopedSession(raw, LOCAL_SCOPE), [source])
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        lambda *_args: (source, folder, json.loads((folder / "metadata.json").read_text()), True),
    )
    from cowork.server import create_app

    client = TestClient(create_app())
    response = client.get(f"/api/v1/artifacts/workspace/local/{artifact_id}")
    files_before = sorted(path.name for path in folder.iterdir())

    assert [card["capabilities"]["exportFormats"] for card in cards] == [[]]
    assert response.status_code == 200, response.text
    assert response.json()["capabilities"]["exportFormats"] == []
    for export_format in ("html", "docx"):
        refused = client.post(
            f"/api/v1/artifacts/workspace/local/{artifact_id}/export", json={"format": export_format}
        )
        assert refused.status_code == 422, refused.text
    assert sorted(path.name for path in folder.iterdir()) == files_before


def test_capabilities_without_an_export_source_leave_the_field_out():
    """Only the cards and the workspace response know the primary file; any
    other capabilities answer leaves the field out rather than claim none."""
    capabilities = artifact_capabilities(SimpleNamespace(scope=LOCAL_SCOPE), None, "report")

    assert "exportFormats" not in capabilities


# ── org mode ─────────────────────────────────────────────────────────────────


@contextmanager
def _scoped(org_id: str, user_id: str):
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        yield ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id, user_id=user_id))


@pytest.fixture
def org_world(tmp_path, org_deployment):
    org_id, creator, member = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    ids = {}
    for slug, owner in (("mine", creator), ("theirs", member), ("orphan", None)):
        ids[slug], _folder = _write_artifact(
            source.base, slug, {"report.md": "# R\n"}, type="document", primary="report.md"
        )
        if owner is not None:
            with _scoped(org_id, owner) as session:
                ownership.record_artifact_owner(session, project.id, slug, owner, action="create")
    return SimpleNamespace(
        org_id=org_id, creator=creator, member=member, project=project, source=source, ids=ids
    )


@pytest.fixture
def org_export_on(monkeypatch):
    """Turn org export on, as the release that lifts the switch will."""
    monkeypatch.setattr(artifact_export, "ORG_EXPORT_ENABLED", True)


def _org_card_formats(world) -> dict[str, list[str]]:
    with _scoped(world.org_id, world.creator) as session:
        cards = artifacts_ep._artifact_cards(session, [world.source])
    return {card["slug"]: card["capabilities"]["exportFormats"] for card in cards}


def test_org_cards_offer_formats_to_the_owner_only(org_world, org_export_on):
    assert _org_card_formats(org_world) == {"mine": ["html", "docx"], "theirs": [], "orphan": []}


def test_org_cards_offer_nothing_while_org_export_is_off(org_world):
    assert _org_card_formats(org_world) == {"mine": [], "theirs": [], "orphan": []}


def _role_permissions(monkeypatch, *, artifact_manage: bool):
    from cowork.services import product_permissions

    async def decide(_scope, permission):
        return artifact_manage if permission == "artifact.manage" else True

    monkeypatch.setattr(product_permissions, "has_product_permission", decide)
    monkeypatch.setattr(workspace_ep, "has_product_permission", decide)


@pytest.mark.parametrize(
    ("org_export", "offered"), [(True, ["html", "docx"]), (False, [])], ids=["on", "off"]
)
async def test_the_org_workspace_response_offers_formats_to_an_owner_who_may_manage(
    org_world, monkeypatch, org_export, offered
):
    monkeypatch.setattr(artifact_export, "ORG_EXPORT_ENABLED", org_export)
    _role_permissions(monkeypatch, artifact_manage=True)
    with _scoped(org_world.org_id, org_world.creator) as session:
        result = await workspace_ep.artifact_source(
            str(org_world.project.id), org_world.ids["mine"], session, path=None
        )

    assert result["capabilities"]["exportFormats"] == offered


async def test_while_org_export_is_off_the_workspace_response_never_walks_the_folder(
    org_world, monkeypatch
):
    """Org export off offers nothing whatever the folder holds, so the
    response skips the walk that finds the primary."""
    monkeypatch.setattr(
        workspace_ep, "primary_export_source", Mock(side_effect=AssertionError("walked the folder"))
    )
    _role_permissions(monkeypatch, artifact_manage=True)
    with _scoped(org_world.org_id, org_world.creator) as session:
        result = await workspace_ep.artifact_source(
            str(org_world.project.id), org_world.ids["mine"], session, path=None
        )

    assert result["capabilities"]["exportFormats"] == []


async def test_the_workspace_response_offers_nothing_once_artifact_manage_is_gone(
    org_world, monkeypatch
):
    """The owner can still read the source, but the export route would refuse
    them, so the live recheck empties the list with ``canEdit``."""
    _role_permissions(monkeypatch, artifact_manage=False)
    with _scoped(org_world.org_id, org_world.creator) as session:
        result = await workspace_ep.artifact_source(
            str(org_world.project.id), org_world.ids["mine"], session, path=None
        )

    assert result["capabilities"]["canEdit"] is False
    assert result["capabilities"]["exportFormats"] == []


async def test_a_reviewer_never_reaches_the_workspace_response(org_world, monkeypatch):
    _role_permissions(monkeypatch, artifact_manage=True)
    with _scoped(org_world.org_id, org_world.member) as session:
        with pytest.raises(HTTPException) as refused:
            await workspace_ep.artifact_source(
                str(org_world.project.id), org_world.ids["mine"], session, path=None
            )

    assert refused.value.status_code == 403


@pytest.mark.parametrize("artifact_manage", [True, False])
async def test_the_live_recheck_keeps_formats_only_with_artifact_manage(
    monkeypatch, artifact_manage
):
    _role_permissions(monkeypatch, artifact_manage=artifact_manage)
    capabilities = {
        "canEdit": True,
        "canAddressWithAgent": True,
        "canResolveComments": True,
        "exportFormats": ["html", "docx"],
    }

    current = await workspace_ep._current_capabilities(
        SimpleNamespace(scope=TenantScope(org_mode=True, org_id="org", user_id="user")),
        capabilities,
    )

    assert current["exportFormats"] == (["html", "docx"] if artifact_manage else [])
    assert capabilities["exportFormats"] == ["html", "docx"]


# ── the list holds exactly what the route accepts ────────────────────────────


@pytest.mark.parametrize("slug", sorted(_LOCAL_ARTIFACTS))
def test_the_route_accepts_exactly_the_formats_the_card_lists(tmp_path, local_mode, monkeypatch, slug):
    files, metadata, _expected = _LOCAL_ARTIFACTS[slug]
    source = _local_source(tmp_path)
    artifact_id, folder = _write_artifact(source.base, slug, files, **metadata)
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        (card,) = artifacts_ep._artifact_cards(ScopedSession(raw, LOCAL_SCOPE), [source])
    listed = card["capabilities"]["exportFormats"]
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        lambda *_args: (source, folder, json.loads((folder / "metadata.json").read_text()), True),
    )
    from cowork.server import create_app

    client = TestClient(create_app())
    for export_format in ("html", "docx", "pdf"):
        response = client.post(
            f"/api/v1/artifacts/workspace/local/{artifact_id}/export",
            json={"format": export_format},
        )
        expected = 200 if export_format in listed else 422
        assert response.status_code == expected, (export_format, response.text)
