"""Size bounds on a skill upload, and on the files a turn reads from a skill.

A skill upload is never held in memory whole. The upload route hands the
upload's own file to SkillService.import_skill_file. A zip's central directory
is measured from its end record before zipfile.ZipFile builds an object for
each of its entries, and a SKILL.md past SKILL_MD_MAX_BYTES is refused before
anything reads it whole. A turn checks each skill file's size on disk before
it reads the file (_skill_wire_files).
"""

from __future__ import annotations

import io
import struct
import tracemalloc
import zipfile
from pathlib import Path

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

import cowork.services.skills as skills_mod
from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.skills import (
    SKILL_ARCHIVE_TOO_MANY_FILES,
    SKILL_MD_TOO_LARGE,
    SkillService,
    SkillTooLargeError,
    _skill_wire_files,
)


def _skill_md(*, name: str = "limits-skill", size: int | None = None) -> bytes:
    """A SKILL.md that parses, padded with body text to ``size`` bytes when given."""
    head = f"---\nname: {name}\ndescription: Checks the upload bounds.\n---\n".encode()
    if size is None:
        return head + b"Step 1. do the thing\n"
    return head + b"x" * (size - len(head))


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buf.getvalue()


def _past_directory_bound() -> bytes:
    """A real archive whose central directory is just past 4 MiB: 65 empty
    entries named with 65,000 characters, each directory line 46 bytes plus
    its name. Long names reach the bound in few entries, so it builds fast."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as archive:
        for number in range(65):
            archive.writestr(f"{number:03d}" + "x" * 64_997, b"")
    return buf.getvalue()


def _with_zip64_end_record(files: dict[str, bytes]) -> bytes:
    """An archive with a ZIP64 end record, whose classic end record holds
    0xFFFFFFFF for the directory's size, as an archive past 4 GiB does. A
    reader then takes the size from the ZIP64 record."""
    with pytest.MonkeyPatch.context() as patch:
        # Past this many entries the writer adds the ZIP64 records.
        patch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 0)
        data = bytearray(_zip(files))
    # The classic record is the archive's last 22 bytes (it has no comment),
    # and the directory's size sits 12 bytes into it.
    struct.pack_into("<L", data, len(data) - zipfile.sizeEndCentDir + 12, 0xFFFFFFFF)
    return bytes(data)


def _write_large_skill_zip(*, path: Path, name: str, size: int) -> None:
    """A stored zip holding a small SKILL.md and one ``size``-byte data file,
    written a MiB at a time, so building it never holds it in memory."""
    chunk = bytes(1024 * 1024)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("SKILL.md", _skill_md(name=name))
        with archive.open("data.bin", "w") as entry:
            for _ in range(size // len(chunk)):
                entry.write(chunk)


@pytest.fixture
def svc(tmp_path: Path) -> SkillService:
    """A local store under ``tmp_path``, as tests/test_skill_import.py makes one."""
    service = SkillService()
    service.root = tmp_path / "store"
    service.root.mkdir()
    return service


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """A single-user deployment whose stores are under ``tmp_path``, set up as
    tests/test_skill_routes_wire.py sets one up."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def client(stores) -> TestClient:
    from cowork.server import create_app

    # No `with`: skipping the lifespan skips boot migrations and background workers.
    return TestClient(create_app())


# ── a zip's central directory ───────────────────────────────────────────────


def test_a_zip_whose_central_directory_is_past_the_bound_is_refused_before_zipfile_opens_it(svc, monkeypatch):
    many = _past_directory_bound()
    with zipfile.ZipFile(io.BytesIO(many)) as archive:
        directory = sum(
            zipfile.sizeCentralDir + len(info.filename.encode()) + len(info.extra) + len(info.comment)
            for info in archive.infolist()
        )
    assert SkillService._ZIP_MAX_DIRECTORY_BYTES == 4 * 1024 * 1024
    assert directory > SkillService._ZIP_MAX_DIRECTORY_BYTES
    few = _zip({"SKILL.md": _skill_md()})
    real_zipfile = zipfile.ZipFile
    opened: list[object] = []

    def recording_zipfile(file, *args, **kwargs):
        opened.append(file)
        return real_zipfile(file, *args, **kwargs)

    monkeypatch.setattr(zipfile, "ZipFile", recording_zipfile)

    with pytest.raises(SkillTooLargeError) as refused:
        svc.import_skill(many, filename="many.zip")

    assert str(refused.value) == SKILL_ARCHIVE_TOO_MANY_FILES
    assert opened == []
    assert list(svc.root.iterdir()) == []
    # The recorder sees an archive the bound lets through.
    svc.import_skill(few, filename="few.zip")
    assert len(opened) == 1


def test_a_zip64_archive_is_measured_by_its_zip64_end_record(svc):
    """A reader of the classic end record alone would take 0xFFFFFFFF for the
    directory's size and refuse this small archive."""
    data = _with_zip64_end_record({"SKILL.md": _skill_md(), "helper.py": b"print(1)\n"})

    skill = svc.import_skill(data, filename="pack.zip")

    assert (svc.root / skill.name / "helper.py").read_bytes() == b"print(1)\n"


def test_a_zip_with_a_few_files_imports_from_a_file_on_disk(svc, tmp_path):
    path = tmp_path / "pack.zip"
    path.write_bytes(
        _zip({"SKILL.md": _skill_md(), "assets/helper.py": b"print(1)\n", "reference.md": b"# Notes\n"})
    )

    with path.open("rb") as source:
        skill = svc.import_skill_file(source=source, filename="pack.zip")

    stored = svc.root / skill.name
    assert skill.name == "limits-skill"
    assert (stored / "assets" / "helper.py").read_bytes() == b"print(1)\n"
    assert (stored / "reference.md").read_bytes() == b"# Notes\n"


def test_the_upload_route_answers_413_for_a_zip_whose_directory_is_past_the_bound(client):
    response = client.post(
        "/api/v1/skills/upload", files={"file": ("many.zip", _past_directory_bound(), "application/zip")}
    )

    assert (response.status_code, response.json()["detail"]) == (413, SKILL_ARCHIVE_TOO_MANY_FILES)


# ── a SKILL.md's size ───────────────────────────────────────────────────────


def test_an_md_upload_past_the_skill_md_bound_answers_413_and_one_at_it_imports(client, monkeypatch):
    monkeypatch.setattr(skills_mod, "SKILL_MD_MAX_BYTES", 1024)

    over = client.post(
        "/api/v1/skills/upload",
        files={"file": ("SKILL.md", _skill_md(name="one-past", size=1025), "text/markdown")},
    )
    at = client.post(
        "/api/v1/skills/upload",
        files={"file": ("SKILL.md", _skill_md(name="at-the-bound", size=1024), "text/markdown")},
    )

    assert (over.status_code, over.json()["detail"]) == (413, SKILL_MD_TOO_LARGE)
    assert not (SkillService(LOCAL_SCOPE).root / "one-past").exists()
    assert (at.status_code, at.json()["id"]) == (201, "at-the-bound"), at.text


def test_an_md_upload_is_read_one_byte_past_the_bound_and_no_further(svc, monkeypatch):
    monkeypatch.setattr(skills_mod, "SKILL_MD_MAX_BYTES", 1024)
    source = io.BytesIO(_skill_md(size=64 * 1024))

    with pytest.raises(SkillTooLargeError):
        svc.import_skill_file(source=source, filename="SKILL.md")

    assert source.tell() == 1025


def test_an_md_upload_cut_inside_a_character_at_the_bound_answers_413_not_422(svc, monkeypatch):
    """The read stops one byte past the bound, which can split a character
    UTF-8 writes in several bytes. The size refusal comes before the decode,
    so such an upload is refused for its size and not as text that isn't
    UTF-8."""
    monkeypatch.setattr(skills_mod, "SKILL_MD_MAX_BYTES", 1024)
    text = _skill_md(size=1022) + "é".encode("utf-8") * 64

    with pytest.raises(SkillTooLargeError):
        svc.import_skill_file(source=io.BytesIO(text), filename="SKILL.md")


@pytest.mark.parametrize(
    "member",
    ["SKILL.md", "pack/SKILL.md", "notes.md"],
    ids=["at-the-root", "in-a-wrapping-folder", "a-lone-md-renamed"],
)
def test_a_zip_whose_skill_md_is_past_the_bound_is_refused_before_it_is_parsed(svc, monkeypatch, member):
    monkeypatch.setattr(skills_mod, "SKILL_MD_MAX_BYTES", 1024)
    parsed: list[Path] = []
    monkeypatch.setattr(skills_mod, "parse_skill_dir", parsed.append)

    with pytest.raises(SkillTooLargeError) as refused:
        svc.import_skill(_zip({member: _skill_md(size=1025)}), filename="pack.zip")

    assert str(refused.value) == SKILL_MD_TOO_LARGE
    assert parsed == []
    assert list(svc.root.iterdir()) == []


def test_a_zip_whose_skill_md_is_at_the_bound_imports(svc, monkeypatch):
    monkeypatch.setattr(skills_mod, "SKILL_MD_MAX_BYTES", 1024)

    skill = svc.import_skill(_zip({"SKILL.md": _skill_md(size=1024)}), filename="pack.zip")

    assert skill.name == "limits-skill"


# ── memory ──────────────────────────────────────────────────────────────────


async def _upload_from_disk(*, path: Path) -> dict:
    """The upload route, called as FastAPI calls it in a single-user
    deployment, with an upload whose file is ``path`` on disk."""
    with path.open("rb") as handle:
        return await skills_ep.upload_skill(
            UploadFile(file=handle, filename=path.name), LOCAL_SCOPE, principal=None, shared=False, viewer=None
        )


async def test_a_zip_upload_is_read_from_its_file_and_never_held_in_memory(stores, tmp_path):
    """A 64 MiB skill zip uploaded from a file on disk peaks under 16 MiB of
    Python memory: the route hands the file itself to import_skill_file,
    which reads the zip in place and extracts each entry in chunks. A route
    that read the upload into memory first peaks past 64 MiB."""
    size = 64 * 1024 * 1024
    archive = tmp_path / "large.zip"
    _write_large_skill_zip(path=archive, name="large-skill", size=size)
    warm = tmp_path / "warm.zip"
    warm.write_bytes(_zip({"SKILL.md": _skill_md(name="warm-skill")}))
    # A first upload loads what the route imports and caches on first use,
    # so the peak below is the large upload's own.
    await _upload_from_disk(path=warm)

    already_tracing = tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before, _ = tracemalloc.get_traced_memory()
        created = await _upload_from_disk(path=archive)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()

    assert created["id"] == "large-skill"
    assert (SkillService(LOCAL_SCOPE).root / "large-skill" / "data.bin").stat().st_size == size
    growth = peak - before
    assert growth < 16 * 1024 * 1024, f"peak grew {growth / 2**20:.1f} MiB"


# ── the files a turn reads ──────────────────────────────────────────────────


@pytest.fixture
def reads(monkeypatch) -> list[Path]:
    """Every path read with Path.read_text from here on, in order."""
    read: list[Path] = []
    real_read_text = Path.read_text

    def recording_read_text(self, *args, **kwargs):
        read.append(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)
    return read


def _skill_folder(root: Path, *, files: dict[str, bytes]) -> Path:
    folder = root / "demo"
    for rel, content in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(content)
    return folder


def test_a_turn_leaves_a_file_past_the_cap_unread_and_keeps_the_rest(tmp_path, reads):
    cap = skills_mod._TURN_SKILL_FILE_MAX
    folder = _skill_folder(
        tmp_path,
        files={"SKILL.md": _skill_md(), "references/tip.md": b"- tip", "data.txt": b"x" * (cap + 1)},
    )

    files = _skill_wire_files(folder, tmp_path)

    assert files == {"SKILL.md": _skill_md().decode(), "references/tip.md": "- tip"}
    assert folder / "data.txt" not in reads
    assert folder / "references" / "tip.md" in reads


def test_a_turn_still_reads_and_skips_a_file_under_the_cap_on_disk_but_past_it_on_the_wire(tmp_path, reads):
    """Each newline is one byte on disk and two in _wire_len's JSON."""
    cap = skills_mod._TURN_SKILL_FILE_MAX
    folder = _skill_folder(tmp_path, files={"SKILL.md": _skill_md(), "lines.txt": b"\n" * (cap // 2 + 1)})

    files = _skill_wire_files(folder, tmp_path)

    assert set(files) == {"SKILL.md"}
    assert folder / "lines.txt" in reads


def test_a_turn_drops_a_skill_whose_skill_md_is_past_the_cap_without_reading_it(tmp_path, reads):
    cap = skills_mod._TURN_SKILL_FILE_MAX
    folder = _skill_folder(tmp_path, files={"SKILL.md": _skill_md(size=cap + 1), "helper.py": b"print(1)\n"})

    assert _skill_wire_files(folder, tmp_path) is None
    assert folder / "SKILL.md" not in reads
