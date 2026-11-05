"""Reading a file inside an artifact folder without following anything out of it."""
from __future__ import annotations

import ast
import inspect
import os
import stat
import sys
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.common.paths import PinnedDir, pinned_dir
from cowork.schemas.artifact_export import OmittedReason
from cowork.services import artifact_files
from cowork.services.artifact_files import (
    ArtifactFileContent,
    ArtifactFileRefused,
    read_artifact_file,
    read_artifact_file_entry,
)
from cowork.services.artifacts import ProjectArtifacts

FIXTURES = Path(__file__).parent / "fixtures" / "artifact_files"


def _fixture_cases(*, name: str) -> list[dict]:
    return yaml.safe_load((FIXTURES / name).read_text(encoding="utf-8"))


REFUSED_PATHS = _fixture_cases(name="refused_paths.yaml")
WINDOWS_REWRITTEN_NAMES = _fixture_cases(name="windows_rewritten_names.yaml")
MISSING_PATHS = _fixture_cases(name="missing_paths.yaml")
REPARSE_POINTS = _fixture_cases(name="reparse_points.yaml")
DRAFTS_ROUTE_HOUSEKEEPING_PATHS = _fixture_cases(name="drafts_route_housekeeping_paths.yaml")
CONTENT_NAMES_AT_DEPTH = _fixture_cases(name="content_names_at_depth.yaml")
API_IMPORT_FORMS = _fixture_cases(name="api_import_forms.yaml")

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(32))
SENTINEL = b"SENTINEL-OTHER-PARTITION"


@pytest.fixture
def art(tmp_path) -> Path:
    folder = tmp_path / "art"
    folder.mkdir()
    return folder


@pytest.fixture
def outside(tmp_path) -> Path:
    """Another partition's folder, holding a file the artifact must never read."""
    folder = tmp_path / "other"
    folder.mkdir()
    (folder / "secret.txt").write_bytes(SENTINEL)
    return folder


def _read(*, folder: Path, parts: tuple[str, ...], max_bytes: int = 1024) -> ArtifactFileContent:
    with pinned_dir(folder) as pinned:
        return read_artifact_file_entry(folder=pinned, parts=parts, max_bytes=max_bytes)


def _refusal(*, folder: Path, parts: tuple[str, ...], max_bytes: int = 1024) -> OmittedReason:
    with pytest.raises(ArtifactFileRefused) as refused:
        _read(folder=folder, parts=parts, max_bytes=max_bytes)
    return refused.value.reason


def _symlink(*, link: Path, target: Path, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this platform can't create a symlink here: {exc}")


def _hard_link(*, target: Path, link: Path) -> None:
    try:
        os.link(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this filesystem can't create a hard link here: {exc}")


def _unexpected_file_call(*_args, **_kwargs):
    raise AssertionError("a refused path reached a file call")


# Every call that touches the filesystem, from the walk's helpers down to os.
_FILE_CALLS = (
    (artifact_files, "dir_scandir"),
    (artifact_files, "dir_open"),
    (artifact_files, "dir_lstat"),
    (artifact_files, "open_pinned_child"),
    (os, "open"),
    (os, "lstat"),
    (os, "stat"),
    (os, "fstat"),
    (os, "scandir"),
)


def _refusal_without_file_calls(
    *, folder: Path, parts: tuple[str, ...], monkeypatch
) -> OmittedReason:
    with pinned_dir(folder) as pinned:
        with monkeypatch.context() as patch:
            for module, name in _FILE_CALLS:
                patch.setattr(module, name, _unexpected_file_call)
            with pytest.raises(ArtifactFileRefused) as refused:
                read_artifact_file_entry(folder=pinned, parts=parts, max_bytes=1024)
    return refused.value.reason


@pytest.mark.parametrize("case", REFUSED_PATHS, ids=[case["id"] for case in REFUSED_PATHS])
def test_a_path_refused_by_its_text_never_reaches_the_filesystem(art, monkeypatch, case):
    reason = _refusal_without_file_calls(
        folder=art, parts=tuple(case["parts"]), monkeypatch=monkeypatch
    )

    assert reason is OmittedReason(case["reason"])


def test_a_path_deeper_than_the_cap_is_refused_before_any_file_call(art, monkeypatch):
    parts = ("d",) * artifact_files.MAX_PATH_PARTS + ("chart.png",)

    reason = _refusal_without_file_calls(folder=art, parts=parts, monkeypatch=monkeypatch)

    assert reason is OmittedReason.BUDGET


def test_a_path_at_the_depth_cap_reads(art):
    folders = ("d",) * (artifact_files.MAX_PATH_PARTS - 1)
    art.joinpath(*folders).mkdir(parents=True)
    art.joinpath(*folders, "chart.png").write_bytes(PNG)

    assert _read(folder=art, parts=(*folders, "chart.png")).data == PNG


def test_a_nested_file_reads_with_the_device_and_inode_it_came_from(art):
    (art / "sub" / "deeper").mkdir(parents=True)
    target = art / "sub" / "deeper" / "chart.png"
    target.write_bytes(PNG)
    on_disk = os.stat(target)

    content = _read(folder=art, parts=("sub", "deeper", "chart.png"))

    assert content == ArtifactFileContent(
        data=PNG, device=on_disk.st_dev, inode=on_disk.st_ino
    )


def test_read_artifact_file_returns_the_bytes_alone(art):
    (art / "notes.txt").write_bytes(b"plain text")

    with pinned_dir(art) as pinned:
        data = read_artifact_file(folder=pinned, parts=("notes.txt",), max_bytes=1024)

    assert data == b"plain text"


def test_a_file_at_max_bytes_reads_and_one_byte_more_is_too_large(art):
    (art / "exact.png").write_bytes(b"x" * 10)
    (art / "big.png").write_bytes(b"x" * 11)

    assert _read(folder=art, parts=("exact.png",), max_bytes=10).data == b"x" * 10
    assert _refusal(folder=art, parts=("big.png",), max_bytes=10) is OmittedReason.TOO_LARGE


def _fstat_reporting(*, monkeypatch, **fields: int) -> None:
    """Make ``os.fstat`` report *fields* in place of the real values."""
    real_fstat = os.fstat

    def reported(fd):
        values = list(real_fstat(fd)[:10])
        for name, value in fields.items():
            values[getattr(stat, f"ST_{name.upper()}")] = value
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", reported)


def test_a_file_that_grows_past_max_bytes_after_its_size_check_is_too_large(
    art, monkeypatch
):
    (art / "growing.png").write_bytes(b"x" * 64)
    _fstat_reporting(monkeypatch=monkeypatch, size=8)
    read_sizes = []
    real_read_bounded = artifact_files.read_bounded

    def recording_read_bounded(fd, size, *args):
        read_sizes.append(size)
        return real_read_bounded(fd, size, *args)

    monkeypatch.setattr(artifact_files, "read_bounded", recording_read_bounded)

    assert _refusal(folder=art, parts=("growing.png",), max_bytes=10) is OmittedReason.TOO_LARGE
    assert read_sizes == [11]


def test_a_file_unlinked_after_its_open_is_missing(art, monkeypatch):
    (art / "chart.png").write_bytes(PNG)
    _fstat_reporting(monkeypatch=monkeypatch, nlink=0)

    assert _refusal(folder=art, parts=("chart.png",)) is OmittedReason.MISSING


@pytest.mark.parametrize("case", MISSING_PATHS, ids=[case["id"] for case in MISSING_PATHS])
def test_a_missing_file_or_directory_is_missing(art, case):
    (art / "sub").mkdir()

    assert _refusal(folder=art, parts=tuple(case["parts"])) is OmittedReason.MISSING


@pytest.mark.parametrize(
    "parts", [("\ud800.png",), ("\ud800", "chart.png")], ids=["file", "folder"]
)
def test_a_name_holding_a_lone_surrogate_is_missing(art, tmp_path, parts):
    """The filesystem can't encode it, so it names nothing, and only a refusal leaves."""
    source, folder = _draft_artifact(tmp_path=tmp_path)

    with pytest.raises(ArtifactFileRefused) as drafts:
        artifact_files.open_pinned_draft_file(source, folder, parts)

    assert _refusal(folder=art, parts=parts) is OmittedReason.MISSING
    assert drafts.value.reason is OmittedReason.MISSING


def test_a_directory_at_the_end_is_missing(art):
    (art / "sub").mkdir()

    assert _refusal(folder=art, parts=("sub",)) is OmittedReason.MISSING


def test_a_missing_name_is_missing_without_reading_the_directory(art, monkeypatch):
    monkeypatch.setattr(artifact_files, "dir_scandir", _unexpected_file_call)

    assert _refusal(folder=art, parts=("absent.png",)) is OmittedReason.MISSING
    assert _refusal(folder=art, parts=("absent", "chart.png")) is OmittedReason.MISSING


def _scripted_scandir(*, listings: dict[str, list[str]], read: list[str]):
    """Stand in for ``dir_scandir``, listing set names per folder and logging each read."""

    @contextmanager
    def scan(directory: PinnedDir):
        def entries():
            for name in listings[directory.path.name]:
                read.append(name)
                yield SimpleNamespace(name=name)

        yield entries()

    return scan


def test_one_walk_stops_reading_entries_at_the_scan_cap(art, monkeypatch):
    """The cap counts entries across every folder of one walk, not per folder."""
    (art / "sub").mkdir()
    (art / "sub" / "chart.png").write_bytes(PNG)
    listings = {"art": ["other", "sub"], "sub": ["more", "extra", "chart.png"]}
    read: list[str] = []
    monkeypatch.setattr(
        artifact_files, "dir_scandir", _scripted_scandir(listings=listings, read=read)
    )

    monkeypatch.setattr(artifact_files, "_MAX_SCANNED_ENTRIES", 5)
    assert _read(folder=art, parts=("sub", "chart.png")).data == PNG

    read.clear()
    monkeypatch.setattr(artifact_files, "_MAX_SCANNED_ENTRIES", 3)
    assert _refusal(folder=art, parts=("sub", "chart.png")) is OmittedReason.BUDGET
    assert read == ["other", "sub", "more", "extra"]


def test_the_shipped_scan_cap_reads_20000_entries_and_refuses_the_next(art, monkeypatch):
    """The cap as shipped, unpatched, so raising it fails a test."""
    (art / "chart.png").write_bytes(PNG)
    others = [f"tile{index}.png" for index in range(20_000)]
    listings = {"art": [*others[:-1], "chart.png"]}
    read: list[str] = []
    monkeypatch.setattr(
        artifact_files, "dir_scandir", _scripted_scandir(listings=listings, read=read)
    )

    assert _read(folder=art, parts=("chart.png",)).data == PNG
    listings["art"] = [*others, "chart.png"]
    assert _refusal(folder=art, parts=("chart.png",)) is OmittedReason.BUDGET


def test_the_shipped_depth_cap_reads_64_segments_and_refuses_65(art):
    """The cap as shipped, unpatched, so raising it fails a test."""
    art.joinpath(*("d",) * 63).mkdir(parents=True)
    art.joinpath(*("d",) * 63, "chart.png").write_bytes(PNG)
    art.joinpath(*("d",) * 64).mkdir()
    art.joinpath(*("d",) * 64, "chart.png").write_bytes(PNG)

    assert _read(folder=art, parts=(*("d",) * 63, "chart.png")).data == PNG
    assert _refusal(folder=art, parts=(*("d",) * 64, "chart.png")) is OmittedReason.BUDGET


@pytest.mark.parametrize("linked", ["leaf", "middle"])
def test_a_symlink_at_any_segment_is_refused(art, outside, linked):
    if linked == "leaf":
        _symlink(link=art / "secret.txt", target=outside / "secret.txt")
        parts = ("secret.txt",)
    else:
        (art / "sub").mkdir()
        _symlink(link=art / "sub" / "linked", target=outside, directory=True)
        parts = ("sub", "linked", "secret.txt")

    assert _refusal(folder=art, parts=parts) is OmittedReason.SYMLINK


def _record_dir_opens(*, monkeypatch) -> list[str]:
    opened: list[str] = []
    real_dir_open = artifact_files.dir_open

    def recording_open(directory, name, flags, mode=0o777):
        opened.append(name)
        return real_dir_open(directory, name, flags, mode)

    monkeypatch.setattr(artifact_files, "dir_open", recording_open)
    return opened


@pytest.mark.parametrize("linked", ["leaf", "middle"])
def test_without_a_directory_descriptor_a_symlink_is_refused_before_any_open(
    art, outside, monkeypatch, linked
):
    """The walk Windows runs, where ``PinnedDir.fd`` is None and opens join paths."""
    if linked == "leaf":
        _symlink(link=art / "secret.txt", target=outside / "secret.txt")
        parts = ("secret.txt",)
    else:
        (art / "sub").mkdir()
        _symlink(link=art / "sub" / "linked", target=outside, directory=True)
        parts = ("sub", "linked", "secret.txt")
    opened = _record_dir_opens(monkeypatch=monkeypatch)

    with pytest.raises(ArtifactFileRefused) as refused:
        read_artifact_file_entry(folder=PinnedDir(None, art), parts=parts, max_bytes=1024)

    assert refused.value.reason is OmittedReason.SYMLINK
    assert opened == []


@pytest.mark.parametrize("case", REPARSE_POINTS, ids=[case["id"] for case in REPARSE_POINTS])
def test_without_a_directory_descriptor_only_a_link_is_refused_by_lstat_before_any_open(
    art, monkeypatch, case
):
    parts = tuple(case["parts"])
    reparse_name = case["reparse_name"]
    (art / "junction").mkdir()
    (art / "junction" / "secret.txt").write_bytes(SENTINEL)
    (art / "sub").mkdir()
    (art / "sub" / "placeholder.png").write_bytes(PNG)
    real_dir_lstat = artifact_files.dir_lstat

    def reparse_lstat(directory, name):
        found = real_dir_lstat(directory, name)
        if name != reparse_name:
            return found
        return SimpleNamespace(
            st_mode=found.st_mode,
            st_dev=found.st_dev,
            st_ino=found.st_ino,
            st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
            st_reparse_tag=case["reparse_tag"],
        )

    monkeypatch.setattr(artifact_files, "dir_lstat", reparse_lstat)
    if not case["refused"]:
        assert read_artifact_file(folder=PinnedDir(None, art), parts=parts, max_bytes=1024) == PNG
        return
    opened = _record_dir_opens(monkeypatch=monkeypatch)

    with pytest.raises(ArtifactFileRefused) as refused:
        read_artifact_file_entry(folder=PinnedDir(None, art), parts=parts, max_bytes=1024)

    assert refused.value.reason is OmittedReason.REPARSE_POINT
    assert opened == []


def _swap_before_the_open(*, monkeypatch, opened: str, swap) -> None:
    """Run *swap* just before the walk opens *opened*, as a racing process could."""
    real_dir_open = artifact_files.dir_open

    def swapping_open(directory, name, flags, mode=0o777):
        if name == opened:
            swap()
        return real_dir_open(directory, name, flags, mode)

    monkeypatch.setattr(artifact_files, "dir_open", swapping_open)


@pytest.mark.parametrize("swapped", ["folder", "file"])
def test_without_a_directory_descriptor_a_link_swapped_in_during_the_walk_is_refused(
    art, outside, monkeypatch, swapped
):
    """The walk Windows runs: each lstat passes, then a link takes a name before the open.

    A joined path follows a link in a folder, and without ``O_NOFOLLOW``, as
    on Windows, a link at the file too.
    """
    (art / "assets").mkdir()
    (art / "assets" / "secret.txt").write_bytes(b"inside")
    monkeypatch.setattr(artifact_files, "O_NOFOLLOW", 0)

    def swap() -> None:
        if swapped == "folder":
            (art / "assets").rename(art / "assets-moved")
            _symlink(link=art / "assets", target=outside, directory=True)
        else:
            (art / "assets" / "secret.txt").rename(art / "assets" / "moved.txt")
            _symlink(link=art / "assets" / "secret.txt", target=outside / "secret.txt")

    _swap_before_the_open(monkeypatch=monkeypatch, opened="secret.txt", swap=swap)

    with pytest.raises(ArtifactFileRefused) as refused:
        read_artifact_file_entry(
            folder=PinnedDir(None, art), parts=("assets", "secret.txt"), max_bytes=1024
        )

    assert refused.value.reason is OmittedReason.SYMLINK


def test_without_a_directory_descriptor_a_folder_replaced_during_the_walk_is_refused(
    art, monkeypatch
):
    """A folder that is no link but no longer the one its lstat saw."""
    (art / "assets").mkdir()
    (art / "assets" / "chart.png").write_bytes(PNG)
    (art / "decoy").mkdir()
    (art / "decoy" / "chart.png").write_bytes(b"decoy")

    def swap() -> None:
        (art / "assets").rename(art / "assets-moved")
        (art / "decoy").rename(art / "assets")

    _swap_before_the_open(monkeypatch=monkeypatch, opened="chart.png", swap=swap)

    with pytest.raises(ArtifactFileRefused) as refused:
        read_artifact_file_entry(
            folder=PinnedDir(None, art), parts=("assets", "chart.png"), max_bytes=1024
        )

    assert refused.value.reason is OmittedReason.MISSING


@pytest.mark.skipif(sys.platform != "win32", reason="a junction is a Windows reparse point")
def test_a_junction_is_refused_as_a_reparse_point_on_windows(art, outside):
    import _winapi

    _winapi.CreateJunction(str(outside), str(art / "linked"))
    (art / "chart.png").write_bytes(PNG)

    with pinned_dir(art) as pinned:
        assert pinned.fd is None
        assert read_artifact_file(folder=pinned, parts=("chart.png",), max_bytes=1024) == PNG
        with pytest.raises(ArtifactFileRefused) as refused:
            read_artifact_file_entry(
                folder=pinned, parts=("linked", "secret.txt"), max_bytes=1024
            )

    assert refused.value.reason is OmittedReason.REPARSE_POINT


@pytest.mark.skipif(sys.platform == "win32", reason="Windows can't create these names by path")
@pytest.mark.parametrize(
    "case", WINDOWS_REWRITTEN_NAMES, ids=[case["id"] for case in WINDOWS_REWRITTEN_NAMES]
)
def test_without_a_directory_descriptor_a_name_windows_rewrites_is_refused(
    art, monkeypatch, case
):
    """The drafts route skips the text rules, so the walk itself refuses these names."""
    parts = tuple(case["parts"])
    art.joinpath(*parts[:-1]).mkdir(parents=True, exist_ok=True)
    art.joinpath(*parts).write_bytes(PNG)
    stated: list[str] = []
    real_dir_lstat = artifact_files.dir_lstat

    def recording_lstat(directory, name):
        stated.append(name)
        return real_dir_lstat(directory, name)

    monkeypatch.setattr(artifact_files, "dir_lstat", recording_lstat)
    opened = _record_dir_opens(monkeypatch=monkeypatch)

    with ExitStack() as resources, pytest.raises(ArtifactFileRefused) as refused:
        artifact_files._open_regular_file(
            folder=PinnedDir(None, art), parts=parts, resources=resources, budget=None
        )

    assert refused.value.reason is OmittedReason.TRAVERSAL
    assert case["refused"] not in stated
    assert case["refused"] not in opened


@pytest.mark.skipif(sys.platform != "win32", reason="Windows strips a trailing dot")
def test_a_trailing_dot_is_refused_where_windows_would_open_the_bare_name(art):
    (art / "chart.png").write_bytes(PNG)
    assert (art / "chart.png.").read_bytes() == PNG

    assert _refusal(folder=art, parts=("chart.png.",)) is OmittedReason.TRAVERSAL


@pytest.mark.parametrize("linked_from", ["inside", "outside"])
def test_a_file_with_a_second_hard_link_is_refused(art, outside, linked_from):
    if linked_from == "inside":
        (art / "chart.png").write_bytes(PNG)
        _hard_link(target=art / "chart.png", link=art / "copy.png")
        name = "chart.png"
    else:
        _hard_link(target=outside / "secret.txt", link=art / "secret.txt")
        name = "secret.txt"

    assert _refusal(folder=art, parts=(name,)) is OmittedReason.SYMLINK


def _reason_without_hanging(
    *, folder: Path, parts: tuple[str, ...], fifo: Path
) -> OmittedReason | None:
    outcome: dict[str, OmittedReason] = {}

    def read() -> None:
        try:
            _read(folder=folder, parts=parts)
        except ArtifactFileRefused as exc:
            outcome["reason"] = exc.reason

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    worker.join(timeout=5)
    if worker.is_alive():
        # Give the blocked open a writer so the thread ends, then fail.
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        worker.join(timeout=5)
        pytest.fail("opening a FIFO blocked")
    return outcome.get("reason")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_a_fifo_is_refused_without_blocking(art):
    fifo = art / "pipe.png"
    os.mkfifo(fifo)

    reason = _reason_without_hanging(folder=art, parts=("pipe.png",), fifo=fifo)

    assert reason is OmittedReason.MISSING


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_a_fifo_in_the_middle_is_refused_without_blocking(art):
    """This proves the refusal, not ``O_NONBLOCK``.

    A segment in the middle opens with ``O_DIRECTORY``, which fails on a FIFO
    before the open can wait for a writer, so this passes without
    ``O_NONBLOCK`` too. The leaf test above is the one that proves the flag.
    """
    fifo = art / "pipe"
    os.mkfifo(fifo)

    reason = _reason_without_hanging(folder=art, parts=("pipe", "chart.png"), fifo=fifo)

    assert reason is OmittedReason.MISSING


def test_only_names_from_the_directory_scan_reach_an_open(art, monkeypatch):
    (art / "sub").mkdir()
    (art / "sub" / "chart.png").write_bytes(PNG)
    # Built at runtime so each is a distinct object from the name on disk.
    requested = (("sub" + "x")[:-1], ("chart.png" + "x")[:-1])
    scanned: list[str] = []
    opened: list[str] = []
    real_entry_name = artifact_files.existing_draft_entry_name
    real_dir_open = artifact_files.dir_open
    real_open_child = artifact_files.open_pinned_child

    def entry_name(directory, name, **kwargs):
        found = real_entry_name(directory, name, **kwargs)
        scanned.append(found)
        return found

    def recording_open(directory, name, flags, mode=0o777):
        opened.append(name)
        return real_dir_open(directory, name, flags, mode)

    def recording_child(directory, name, **kwargs):
        opened.append(name)
        return real_open_child(directory, name, **kwargs)

    monkeypatch.setattr(artifact_files, "existing_draft_entry_name", entry_name)
    monkeypatch.setattr(artifact_files, "dir_open", recording_open)
    monkeypatch.setattr(artifact_files, "open_pinned_child", recording_child)

    with pinned_dir(art) as pinned:
        content = read_artifact_file_entry(folder=pinned, parts=requested, max_bytes=1024)

    assert content.data == PNG
    assert opened == ["sub", "chart.png"]
    assert all(name is found for name, found in zip(opened, scanned, strict=True))
    assert all(name is not asked for name, asked in zip(opened, requested, strict=True))


def _open_descriptor_count() -> int:
    return len(os.listdir("/dev/fd"))


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd to count descriptors")
def test_every_descriptor_is_closed_on_every_outcome(art, outside):
    (art / "sub").mkdir()
    (art / "sub" / "chart.png").write_bytes(PNG)
    (art / "sub" / "big.png").write_bytes(b"x" * 2048)
    (art / "sub" / "folder").mkdir()
    _hard_link(target=outside / "secret.txt", link=art / "sub" / "hard.txt")
    _symlink(link=art / "sub" / "soft.txt", target=outside / "secret.txt")
    _symlink(link=art / "sub" / "linked", target=outside, directory=True)
    cases = [
        ("sub", "chart.png"),
        ("sub", "big.png"),
        ("sub", "folder"),
        ("sub", "hard.txt"),
        ("sub", "soft.txt"),
        ("sub", "linked", "secret.txt"),
        ("sub", "absent.png"),
    ]

    with pinned_dir(art) as pinned:
        before = _open_descriptor_count()
        for parts in cases:
            try:
                read_artifact_file_entry(folder=pinned, parts=parts, max_bytes=1024)
            except ArtifactFileRefused:
                pass
        after = _open_descriptor_count()

    assert after == before


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd to count descriptors")
def test_a_deep_walk_holds_at_most_two_descriptors_of_its_own(art, monkeypatch):
    folders = ("d",) * 20
    art.joinpath(*folders).mkdir(parents=True)
    art.joinpath(*folders, "chart.png").write_bytes(PNG)
    counts: list[int] = []
    real_dir_open = artifact_files.dir_open
    real_dir_scandir = artifact_files.dir_scandir

    def counting_open(directory, name, flags, mode=0o777):
        fd = real_dir_open(directory, name, flags, mode)
        counts.append(_open_descriptor_count())
        return fd

    def counting_scandir(directory):
        entries = real_dir_scandir(directory)
        counts.append(_open_descriptor_count())
        return entries

    monkeypatch.setattr(artifact_files, "dir_open", counting_open)
    monkeypatch.setattr(artifact_files, "dir_scandir", counting_scandir)

    with pinned_dir(art) as pinned:
        before = _open_descriptor_count()
        content = read_artifact_file_entry(
            folder=pinned, parts=(*folders, "chart.png"), max_bytes=1024
        )

    assert content.data == PNG
    assert len(counts) == 2 * (len(folders) + 1)
    assert max(counts) - before <= 2


@pytest.mark.parametrize("name", ["README.md", "prd.md"])
def test_the_source_name_match_still_accepts_a_housekeeping_name(art, name):
    """The source editor may name README.md or prd.md, so the match refuses no name."""
    (art / name).write_text("# source\n", encoding="utf-8")

    with pinned_dir(art) as pinned:
        found = artifact_files.existing_draft_entry_name(pinned, name, native_case=True)

    assert found == name


# The drafts route walks the same way, so it hides what the walk refuses.


def _draft_artifact(*, tmp_path) -> tuple[ProjectArtifacts, Path]:
    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "report"
    folder.mkdir(parents=True)
    source = ProjectArtifacts(
        base=base,
        project_id=None,
        project_name="project",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )
    return source, folder


async def _serve_draft(*, monkeypatch, source, folder, rel_path):
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        lambda *_args: (source, folder, {"type": "html-app"}, True),
    )
    return await workspace_ep.serve_private_draft(
        "local",
        "0123456789abcdef0123456789abcdef",
        rel_path,
        SimpleNamespace(query_params={}),
        SimpleNamespace(),
        download=False,
    )


async def _body(*, response: StreamingResponse) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return b"".join(chunks)


@pytest.mark.parametrize(
    "case",
    DRAFTS_ROUTE_HOUSEKEEPING_PATHS,
    ids=[case["id"] for case in DRAFTS_ROUTE_HOUSEKEEPING_PATHS],
)
async def test_the_drafts_route_hides_a_housekeeping_name_at_any_depth(
    tmp_path, monkeypatch, case
):
    rel_path = case["rel_path"]
    source, folder = _draft_artifact(tmp_path=tmp_path)
    hidden = folder / rel_path
    hidden.parent.mkdir(parents=True)
    hidden.write_text("{}", encoding="utf-8")
    # A plain file in the same subfolder still serves.
    beside = f"{rel_path.split('/')[0]}/notes.txt"
    (folder / beside).write_text("content", encoding="utf-8")

    served = await _serve_draft(
        monkeypatch=monkeypatch, source=source, folder=folder, rel_path=beside
    )
    assert await _body(response=served) == b"content"

    with pytest.raises(HTTPException) as error:
        await _serve_draft(
            monkeypatch=monkeypatch, source=source, folder=folder, rel_path=rel_path
        )
    assert error.value.status_code == 404


@pytest.mark.parametrize(
    "case", CONTENT_NAMES_AT_DEPTH, ids=[case["id"] for case in CONTENT_NAMES_AT_DEPTH]
)
async def test_a_top_level_housekeeping_name_inside_a_subfolder_is_content(
    tmp_path, monkeypatch, case
):
    rel_path = case["rel_path"]
    source, folder = _draft_artifact(tmp_path=tmp_path)
    content = folder / rel_path
    content.parent.mkdir(parents=True)
    content.write_text("content", encoding="utf-8")

    served = await _serve_draft(
        monkeypatch=monkeypatch, source=source, folder=folder, rel_path=rel_path
    )
    with pinned_dir(folder) as pinned:
        read = read_artifact_file(
            folder=pinned, parts=tuple(rel_path.split("/")), max_bytes=1024
        )

    assert await _body(response=served) == b"content"
    assert read == b"content"


async def test_the_drafts_route_answers_404_for_a_hard_linked_file(
    tmp_path, monkeypatch, outside
):
    source, folder = _draft_artifact(tmp_path=tmp_path)
    _hard_link(target=outside / "secret.txt", link=folder / "leak.txt")

    with pytest.raises(HTTPException) as error:
        await _serve_draft(
            monkeypatch=monkeypatch, source=source, folder=folder, rel_path="leak.txt"
        )
    assert error.value.status_code == 404


async def test_the_drafts_route_serves_a_path_deeper_than_the_export_cap(
    tmp_path, monkeypatch
):
    """The depth cap bounds an export read; the drafts route serves any depth."""
    source, folder = _draft_artifact(tmp_path=tmp_path)
    folders = ("d",) * artifact_files.MAX_PATH_PARTS
    folder.joinpath(*folders).mkdir(parents=True)
    folder.joinpath(*folders, "notes.txt").write_text("deep", encoding="utf-8")
    rel_path = "/".join((*folders, "notes.txt"))

    served = await _serve_draft(
        monkeypatch=monkeypatch, source=source, folder=folder, rel_path=rel_path
    )

    assert await _body(response=served) == b"deep"
    with pinned_dir(folder) as pinned, pytest.raises(ArtifactFileRefused) as refused:
        read_artifact_file_entry(folder=pinned, parts=(*folders, "notes.txt"), max_bytes=1024)
    assert refused.value.reason is OmittedReason.BUDGET


async def test_the_drafts_route_reads_every_entry_past_the_export_scan_cap(
    tmp_path, monkeypatch
):
    """The scan cap bounds an export read; the drafts route serves a file in any folder."""
    source, folder = _draft_artifact(tmp_path=tmp_path)
    (folder / "tiles").mkdir()
    names = [f"{index}.png" for index in range(10)]
    for name in names:
        (folder / "tiles" / name).write_bytes(PNG)
    monkeypatch.setattr(artifact_files, "_MAX_SCANNED_ENTRIES", 2)

    for name in names:
        served = await _serve_draft(
            monkeypatch=monkeypatch, source=source, folder=folder, rel_path=f"tiles/{name}"
        )
        assert await _body(response=served) == PNG
    refusals = []
    with pinned_dir(folder) as pinned:
        for name in names:
            try:
                read_artifact_file_entry(folder=pinned, parts=("tiles", name), max_bytes=1024)
            except ArtifactFileRefused as exc:
                refusals.append(exc.reason)
    assert refusals
    assert set(refusals) == {OmittedReason.BUDGET}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows strips a trailing dot from a path")
async def test_on_windows_the_drafts_route_hides_a_trailing_dot_spelling_of_a_housekeeping_file(
    tmp_path, monkeypatch
):
    source, folder = _draft_artifact(tmp_path=tmp_path)
    (folder / "metadata.json").write_text('{"id": "x"}', encoding="utf-8")
    # Only an extended-length path keeps the trailing dot in the name on disk.
    decoy = "\\\\?\\" + str(folder.resolve()) + "\\metadata.json."
    with open(decoy, "wb") as handle:
        handle.write(b"decoy")
    try:
        with pytest.raises(ArtifactFileRefused) as refused:
            artifact_files.open_pinned_draft_file(source, folder, ("metadata.json.",))
        with pytest.raises(HTTPException) as error:
            await _serve_draft(
                monkeypatch=monkeypatch, source=source, folder=folder, rel_path="metadata.json."
            )
    finally:
        os.unlink(decoy)

    assert refused.value.reason is OmittedReason.TRAVERSAL
    assert error.value.status_code == 404


def test_the_drafts_open_names_its_refusal(tmp_path, outside):
    source, folder = _draft_artifact(tmp_path=tmp_path)
    (folder / "sub").mkdir()
    (folder / "sub" / "metadata.json").write_text("{}", encoding="utf-8")
    _hard_link(target=outside / "secret.txt", link=folder / "leak.txt")

    with pytest.raises(ArtifactFileRefused) as housekeeping:
        artifact_files.open_pinned_draft_file(source, folder, ("sub", "metadata.json"))
    with pytest.raises(ArtifactFileRefused) as hard_link:
        artifact_files.open_pinned_draft_file(source, folder, ("leak.txt",))

    assert housekeeping.value.reason is OmittedReason.HOUSEKEEPING
    assert hard_link.value.reason is OmittedReason.SYMLINK


def _absolute_imports(*, package: tuple[str, ...], node: ast.ImportFrom) -> list[str]:
    """The module a ``from`` import reads, and each name it imports under that module.

    ``from cowork import api`` imports the module ``cowork.api`` as a name, so
    every imported name counts as a possible submodule.
    """
    if node.level:
        base = package[: len(package) - (node.level - 1)]
        module = ".".join((*base, *([node.module] if node.module else [])))
    else:
        module = node.module or ""
    return [module, *(f"{module}.{alias.name}" for alias in node.names)]


def _api_imports(*, source: str, package: tuple[str, ...]) -> list[str]:
    """Each ``line name`` in *source* that imports the API package or below it."""
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            names = _absolute_imports(package=package, node=node)
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        offenders += [
            f"{node.lineno} {name}"
            for name in names
            if name == "cowork.api" or name.startswith("cowork.api.")
        ]
    return offenders


@pytest.mark.parametrize("case", API_IMPORT_FORMS, ids=[case["id"] for case in API_IMPORT_FORMS])
def test_the_api_import_check_reports_every_import_form(case):
    offenders = _api_imports(source=case["source"], package=("cowork", "services"))

    assert bool(offenders) is case["flagged"]


def test_the_services_package_imports_nothing_from_the_api_package():
    services = Path(artifact_files.__file__).parent
    offenders = []
    for module_path in sorted(services.rglob("*.py")):
        package = ("cowork", "services", *module_path.relative_to(services).parent.parts)
        offenders += [
            f"{module_path.name}:{offender}"
            for offender in _api_imports(
                source=module_path.read_text(encoding="utf-8"), package=package
            )
        ]

    assert offenders == []


# The drafts route's helpers moved into the module unchanged, so the route and
# its tests still call them positionally.
_MOVED_FROM_THE_DRAFTS_ROUTE = frozenset(
    {
        "relative_file_parts",
        "existing_draft_entry_name",
        "_names_one_inode",
        "open_pinned_draft_file",
    }
)


def test_every_function_new_to_the_module_takes_keyword_arguments_only():
    positional = [
        name
        for name, function in inspect.getmembers(artifact_files, inspect.isfunction)
        if function.__module__ == artifact_files.__name__
        and name not in _MOVED_FROM_THE_DRAFTS_ROUTE
        and any(
            parameter.kind is not inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(function).parameters.values()
        )
    ]

    assert positional == []
