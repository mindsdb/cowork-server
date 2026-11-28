"""Artifact format decisions, card capabilities and scoped permission checks."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import yaml
from fastapi import HTTPException
from sqlmodel import Session

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.schemas.artifact_export import ExportPdfSettings
from cowork.services import artifact_export
from cowork.services import artifact_ownership as ownership
from cowork.services.artifact_export import (
    ExportSource,
    card_export_source,
    export_formats,
    export_pdf_settings,
)
from cowork.services.artifact_permissions import artifact_capabilities
from test_artifact_ownership import make_project, project_root
from _artifact_cards import make_card
from tests.support.artifacts.format_cases import _LOCAL_ARTIFACTS, _local_source, _write_artifact
from tests.support.artifacts.format_fixtures import local_mode as local_mode

FIXTURES = Path(__file__).parents[2] / "fixtures" / "artifact_export"
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
    card = make_card(folder=str(folder), path=str(path), type=artifact_type)

    assert card_export_source(card) == expected


def test_local_cards_carry_export_formats(tmp_path, local_mode):
    source = _local_source(tmp_path)
    for slug, (files, metadata, _expected) in _LOCAL_ARTIFACTS.items():
        _write_artifact(source.base, slug, files, **metadata)

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        cards = [card.to_json() for card in artifacts_ep._artifact_cards(ScopedSession(raw, LOCAL_SCOPE), [source])]

    listed = {card["slug"]: card["capabilities"]["exportFormats"] for card in cards}
    assert listed == {slug: expected for slug, (_f, _m, expected) in _LOCAL_ARTIFACTS.items()}


def test_capabilities_without_an_export_source_leave_the_field_out():
    """Only the cards and the workspace response know the primary file; any
    other capabilities answer leaves the field out rather than claim none."""
    capabilities = artifact_capabilities(SimpleNamespace(scope=LOCAL_SCOPE), None, "report")

    assert "exportFormats" not in capabilities


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
        cards = [card.to_json() for card in artifacts_ep._artifact_cards(session, [world.source])]
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
            str(org_world.project.id), org_world.ids["mine"], session.scope, path=None
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
            str(org_world.project.id), org_world.ids["mine"], session.scope, path=None
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
            str(org_world.project.id), org_world.ids["mine"], session.scope, path=None
        )

    assert result["capabilities"]["canEdit"] is False
    assert result["capabilities"]["exportFormats"] == []


async def test_a_reviewer_never_reaches_the_workspace_response(org_world, monkeypatch):
    _role_permissions(monkeypatch, artifact_manage=True)
    with _scoped(org_world.org_id, org_world.member) as session:
        with pytest.raises(HTTPException) as refused:
            await workspace_ep.artifact_source(
                str(org_world.project.id), org_world.ids["mine"], session.scope, path=None
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
        TenantScope(org_mode=True, org_id="org", user_id="user"),
        capabilities,
    )

    assert current["exportFormats"] == (["html", "docx"] if artifact_manage else [])
    assert capabilities["exportFormats"] == ["html", "docx"]
