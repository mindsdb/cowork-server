"""The export service: what it fetches, and how it admits work.

The no-fetch case runs the whole export in the test's own thread, inside
``tests._no_fetch.no_fetch()``, with a 127.0.0.1 recording server started
before the guard opens. It needs no event loop, so it means the same on every
platform.
"""
from __future__ import annotations

import io
import json
import os
import stat
import sys
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path
from uuid import uuid4

import anyio
import docx
import pytest
import yaml

from cowork.common.paths import PinnedDir
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.schemas.artifact_export import OmittedReason
from cowork.services import artifact_export, artifact_identity
from cowork.services.artifact_export import ExportRefusal, ExportRefused, export_slot
from cowork.services.artifact_files import ArtifactFileRefused
from cowork.services.artifacts import ProjectArtifacts
from test_artifact_export_route import _snapshot, padded_png
from test_artifact_files import _symlink
from tests._no_fetch import no_fetch, recording_server

FIXTURES = Path(__file__).parent / "fixtures" / "artifact_export"
NO_FETCH_CASES = yaml.safe_load(
    (FIXTURES / "no_fetch_sources.yaml").read_text(encoding="utf-8")
)["cases"]
SENTINEL = b"SENTINEL-OTHER-PARTITION-7f3a"
_RELATIONSHIPS = "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
_HYPERLINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"


@pytest.fixture
def local_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def partitions(tmp_path):
    """Two people's project folders side by side; the second holds a sentinel."""
    mine = tmp_path / "partitions" / "a" / "general"
    theirs = tmp_path / "partitions" / "b" / "general"
    (mine / ".anton" / "artifacts").mkdir(parents=True)
    secret = theirs / ".anton" / "artifacts" / "private" / "secret.png"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(padded_png(4096, filler=SENTINEL))
    return mine, secret


def _write_report(project: Path, primary: str, source: str) -> tuple[ProjectArtifacts, Path, str]:
    artifact_id = uuid4().hex
    folder = project / ".anton" / "artifacts" / "report"
    folder.mkdir()
    (folder / primary).write_text(source, encoding="utf-8")
    (folder / "metadata.json").write_text(
        json.dumps({"id": artifact_id, "slug": "report", "type": "document", "primary": primary}),
        encoding="utf-8",
    )
    source_root = ProjectArtifacts(
        base=project / ".anton" / "artifacts",
        project_id=None,
        project_name="general",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )
    return source_root, folder, artifact_id


@pytest.mark.parametrize("case", NO_FETCH_CASES, ids=[case["id"] for case in NO_FETCH_CASES])
def test_a_word_export_fetches_nothing_and_takes_nothing_from_another_partition(
    local_mode, partitions, case
):
    mine, secret = partitions
    with recording_server() as server:
        folder_path = mine / ".anton" / "artifacts" / "report"
        source_text = (
            case["source"]
            .replace("{recording}", server.url)
            .replace("{other_relative}", os.path.relpath(secret, folder_path).replace(os.sep, "/"))
            .replace("{other_absolute}", secret.as_posix())
        )
        source, folder, artifact_id = _write_report(mine, case["primary"], source_text)

        with no_fetch() as recorder:
            result = artifact_export.export_primary_file(
                source=source,
                folder_name=folder.name,
                metadata=json.loads((folder / "metadata.json").read_text()),
                artifact_id=artifact_id,
                export_format="docx",
                can_edit=True,
            )

        assert recorder.calls == []
        assert server.hits() == []

    assert result.omitted_count > 0
    written = (folder / result.filename).read_bytes()
    docx.Document(io.BytesIO(written))
    with zipfile.ZipFile(io.BytesIO(written)) as package:
        for name in package.namelist():
            assert SENTINEL not in package.read(name), name
        relationships = ElementTree.fromstring(package.read("word/_rels/document.xml.rels"))
    external = [
        rel.attrib for rel in relationships.iter(_RELATIONSHIPS) if rel.get("TargetMode") == "External"
    ]
    assert external, "the document's own link should stay a hyperlink"
    assert all(rel["Type"] == _HYPERLINK for rel in external), external
    assert {rel["Target"] for rel in external} == {"https://example.com/report"}


def test_a_first_export_records_the_primary_the_card_shows(local_mode, partitions):
    mine, _secret = partitions
    source, folder, artifact_id = _write_report(mine, "report.md", "# R\n")
    metadata = json.loads((folder / "metadata.json").read_text())
    del metadata["primary"]
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    result = artifact_export.export_primary_file(
        source=source,
        folder_name=folder.name,
        metadata=metadata,
        artifact_id=artifact_id,
        export_format="docx",
        can_edit=True,
    )

    assert result.filename == "report.docx"
    assert json.loads((folder / "metadata.json").read_text())["primary"] == "report.md"


# ── placement ────────────────────────────────────────────────────────────────
# These run on every platform: POSIX names the file with a hard link, and
# Windows, which has no directory descriptor, with a rename.


def _export_docx(source: ProjectArtifacts, folder: Path, artifact_id: str) -> str:
    return artifact_export.export_primary_file(
        source=source,
        folder_name=folder.name,
        metadata=json.loads((folder / "metadata.json").read_text()),
        artifact_id=artifact_id,
        export_format="docx",
        can_edit=True,
    ).filename


def test_each_export_takes_the_next_free_name_and_replaces_none(local_mode, partitions):
    mine, _secret = partitions
    source, folder, artifact_id = _write_report(mine, "report.md", "# R\n")
    (folder / "report (2).docx").write_bytes(b"planted bytes")

    names = [_export_docx(source, folder, artifact_id) for _ in range(2)]
    first = (folder / "report.docx").read_bytes()
    names.append(_export_docx(source, folder, artifact_id))

    assert names == ["report.docx", "report (3).docx", "report (4).docx"]
    assert (folder / "report.docx").read_bytes() == first
    assert (folder / "report (2).docx").read_bytes() == b"planted bytes"
    assert not [path.name for path in folder.iterdir() if path.name.startswith(".cowork-export-")]


def test_an_export_with_no_free_name_raises_and_writes_nothing(local_mode, partitions):
    mine, _secret = partitions
    source, folder, artifact_id = _write_report(mine, "report.md", "# R\n")
    for number in range(1, 101):
        name = "report.docx" if number == 1 else f"report ({number}).docx"
        (folder / name).write_bytes(b"old")
    before = _snapshot(folder)

    with pytest.raises(FileExistsError):
        _export_docx(source, folder, artifact_id)

    assert _snapshot(folder) == before


@pytest.mark.skipif(sys.platform == "win32", reason="a pinned folder has no descriptor on Windows")
@pytest.mark.parametrize("export_format", ["docx", "html"])
def test_a_parent_folder_swapped_for_a_link_mid_export_is_never_written_through(
    local_mode, tmp_path, monkeypatch, export_format
):
    """The source's folder is read, then swapped for a link to another folder
    while the conversion runs. The write descends with O_NOFOLLOW, so the
    export fails and writes nothing anywhere."""
    project = tmp_path / "project"
    folder = project / ".anton" / "artifacts" / "report"
    (folder / "docs").mkdir(parents=True)
    (folder / "docs" / "report.md").write_text("# R\n", encoding="utf-8")
    artifact_id = uuid4().hex
    metadata = {"id": artifact_id, "slug": "report", "type": "document", "primary": "docs/report.md"}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    source = ProjectArtifacts(
        base=project / ".anton" / "artifacts",
        project_id=None,
        project_name="project",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    inline = artifact_export.inline_artifact_html

    def swap_after_inlining(**kwargs):
        inlined = inline(**kwargs)
        (folder / "docs").rename(folder / "docs-moved")
        (folder / "docs").symlink_to(outside, target_is_directory=True)
        return inlined

    monkeypatch.setattr(artifact_export, "inline_artifact_html", swap_after_inlining)

    with pytest.raises(OSError):
        artifact_export.export_primary_file(
            source=source,
            folder_name=folder.name,
            metadata=metadata,
            artifact_id=artifact_id,
            export_format=export_format,
            can_edit=True,
        )

    assert list(outside.iterdir()) == []
    assert [path.name for path in (folder / "docs-moved").iterdir()] == ["report.md"]
    assert sorted(path.name for path in folder.iterdir()) == ["docs", "docs-moved", "metadata.json"]


def _swap_docs_for_a_link(*, folder: Path, outside: Path) -> None:
    (folder / "docs").rename(folder / "docs-moved")
    _symlink(link=folder / "docs", target=outside, directory=True)


@pytest.mark.parametrize("swapped", ["before_the_write", "before_the_temporary_opens"])
def test_without_a_directory_descriptor_a_parent_folder_swapped_for_a_link_is_never_written_through(
    tmp_path, monkeypatch, swapped
):
    """The write Windows runs, where ``PinnedDir.fd`` is None and opens join paths.

    The source's folder turns into a link to another folder while the
    conversion runs, or after the write checked it and before the temporary
    file opens. Either way the export is refused and leaves nothing outside
    the artifact.
    """
    art = tmp_path / "art"
    (art / "docs").mkdir(parents=True)
    (art / "docs" / "report.md").write_text("# R\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(artifact_export, "O_NOFOLLOW", 0)
    if swapped == "before_the_write":
        _swap_docs_for_a_link(folder=art, outside=outside)
    else:
        real_dir_open = artifact_export.dir_open

        def swapping_open(directory, name, flags, mode=0o777):
            if name.startswith(".cowork-export-"):
                _swap_docs_for_a_link(folder=art, outside=outside)
            return real_dir_open(directory, name, flags, mode)

        monkeypatch.setattr(artifact_export, "dir_open", swapping_open)

    with pytest.raises(ArtifactFileRefused) as refused:
        artifact_export._write_beside(
            folder=PinnedDir(None, art),
            parts=("docs", "report.md"),
            extension=".docx",
            data=b"export bytes",
            before_placing=lambda: None,
        )

    assert refused.value.reason is OmittedReason.SYMLINK
    assert list(outside.iterdir()) == []
    assert [path.name for path in (art / "docs-moved").iterdir()] == ["report.md"]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows keeps no permission bits beyond read-only")
@pytest.mark.parametrize("writer", ["export", "ensure_full_id"])
def test_rewriting_metadata_keeps_its_permission_bits(local_mode, partitions, writer):
    """Recording a primary and widening a legacy id both replace
    metadata.json; the new document keeps the old one's mode, as it keeps
    its timestamps."""
    mine, _secret = partitions
    source, folder, artifact_id = _write_report(mine, "report.md", "# R\n")
    path = folder / "metadata.json"
    metadata = json.loads(path.read_text())
    if writer == "export":
        del metadata["primary"]
    else:
        metadata["id"] = artifact_id[:8]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    os.chmod(path, 0o644)

    if writer == "export":
        artifact_export.export_primary_file(
            source=source,
            folder_name=folder.name,
            metadata=metadata,
            artifact_id=artifact_id,
            export_format="docx",
            can_edit=True,
        )
        assert json.loads(path.read_text())["primary"] == "report.md"
    else:
        artifact_identity.ensure_full_id(folder)
        assert len(json.loads(path.read_text())["id"]) == 32

    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_a_volume_that_refuses_a_mode_change_still_records_the_primary(
    local_mode, partitions, monkeypatch
):
    """Linux vfat answers chmod with EPERM; the mode is kept where it can be."""
    mine, _secret = partitions
    source, folder, artifact_id = _write_report(mine, "report.md", "# R\n")
    metadata = json.loads((folder / "metadata.json").read_text())
    del metadata["primary"]
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def refuse(*_args):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", refuse, raising=False)

    result = artifact_export.export_primary_file(
        source=source,
        folder_name=folder.name,
        metadata=metadata,
        artifact_id=artifact_id,
        export_format="docx",
        can_edit=True,
    )

    assert result.filename == "report.docx"
    assert json.loads((folder / "metadata.json").read_text())["primary"] == "report.md"


def test_recording_a_primary_without_a_directory_descriptor_sets_times_by_path(tmp_path, monkeypatch):
    """Windows has no directory descriptor, and its utime refuses follow_symlinks=False."""
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"id": "x"}), encoding="utf-8")
    os.utime(metadata, ns=(1_000_000_000, 2_000_000_000))
    real_utime = os.utime

    def windows_utime(path, *args, follow_symlinks=True, **kwargs):
        if not follow_symlinks:
            raise NotImplementedError("utime: follow_symlinks unavailable on this platform")
        return real_utime(path, *args, **kwargs)

    monkeypatch.setattr(os, "utime", windows_utime)

    artifact_export._record_primary(folder=PinnedDir(None, tmp_path), primary="report.md")

    assert json.loads(metadata.read_text())["primary"] == "report.md"
    assert metadata.stat().st_mtime_ns == 2_000_000_000


# ── admission ────────────────────────────────────────────────────────────────


def _org_scope(org_id: str, user_id: str) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org_id, user_id=user_id)


async def test_one_export_per_user_and_per_org(monkeypatch):
    """With room for more conversions, the user and the org still hold one
    export each, and anyone else is admitted."""
    monkeypatch.setattr(artifact_export, "_conversion_limiter", anyio.CapacityLimiter(3))
    alice = _org_scope("org-1", "alice")

    async with export_slot(scope=alice, export_format="docx"):
        for other in (alice, _org_scope("org-1", "bob"), _org_scope("org-2", "alice")):
            with pytest.raises(ExportRefused) as busy:
                async with export_slot(scope=other, export_format="docx"):
                    pass
            assert busy.value.refusal is ExportRefusal.BUSY
        async with export_slot(scope=_org_scope("org-2", "carol"), export_format="docx"):
            pass

    async with export_slot(scope=alice, export_format="docx"):
        pass


async def test_html_and_word_export_run_one_conversion_at_a_time():
    async with export_slot(scope=_org_scope("org-1", "alice"), export_format="docx"):
        for other in (_org_scope("org-2", "bob"), LOCAL_SCOPE, None):
            with pytest.raises(ExportRefused) as busy:
                async with export_slot(scope=other, export_format="docx"):
                    pass
            assert busy.value.refusal is ExportRefusal.BUSY

    async with export_slot(scope=LOCAL_SCOPE, export_format="docx"):
        pass


@pytest.mark.parametrize(
    ("running", "second"), [("pdf", "docx"), ("docx", "pdf"), ("pdf", "html"), ("html", "pdf")]
)
async def test_one_person_and_one_org_run_one_export_across_formats(running, second):
    async with export_slot(scope=_org_scope("org-1", "alice"), export_format=running):
        for other in (_org_scope("org-1", "alice"), _org_scope("org-1", "bob")):
            with pytest.raises(ExportRefused) as busy:
                async with export_slot(scope=other, export_format=second):
                    pass
            assert busy.value.refusal is ExportRefusal.BUSY


async def test_a_deployment_that_names_no_person_runs_one_export_across_formats():
    async with export_slot(scope=LOCAL_SCOPE, export_format="pdf"):
        with pytest.raises(ExportRefused) as busy:
            async with export_slot(scope=LOCAL_SCOPE, export_format="docx"):
                pass
        assert busy.value.refusal is ExportRefusal.BUSY


async def test_a_failed_export_frees_its_slot():
    alice = _org_scope("org-1", "alice")

    with pytest.raises(RuntimeError):
        async with export_slot(scope=alice, export_format="docx"):
            raise RuntimeError("conversion failed")

    async with export_slot(scope=alice, export_format="docx"):
        pass
