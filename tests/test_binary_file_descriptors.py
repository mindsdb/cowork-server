"""Regular-file descriptors are opened in binary mode (ENG-2950).

On Windows, ``os.open`` without ``O_BINARY`` returns a CRT text-mode
descriptor: reads turn CRLF into LF and stop at 0x1A, writes turn LF into
CRLF. ``O_BINARY`` does not exist on POSIX, so these tests patch it to a free
bit and check that the bit reaches ``os.open``.
"""
from __future__ import annotations

import os

import pytest

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.api.v1.endpoints import project_files as project_files_ep
from cowork.common import paths
from cowork.services.artifacts import ProjectArtifacts

SENTINEL = 1 << 30
CSV_CRLF = b"id,name\r\n1,alpha\r\n2,beta\r\n"


@pytest.fixture
def recorded_flags(monkeypatch):
    """Record the flags of every ``os.open`` while ``O_BINARY`` is the sentinel."""
    overlapping = [
        name for name, value in vars(os).items()
        if name.startswith("O_") and isinstance(value, int) and value & SENTINEL
    ]
    assert overlapping == [], f"sentinel collides with {overlapping}"
    real_open = os.open
    calls: list[tuple[str, int]] = []

    def spy(path, flags, *args, **kwargs):
        calls.append((os.fspath(path), flags))
        return real_open(path, flags & ~SENTINEL, *args, **kwargs)

    monkeypatch.setattr(paths, "O_BINARY", SENTINEL)
    monkeypatch.setattr(os, "open", spy)
    return calls


def _flags_for(calls: list[tuple[str, int]], name: str) -> int:
    matches = [flags for path, flags in calls if os.path.basename(path) == name]
    assert matches, f"{name} was never opened"
    return matches[-1]


def test_open_fd_adds_o_binary(tmp_path, recorded_flags):
    target = tmp_path / "data.csv"
    target.write_bytes(CSV_CRLF)

    os.close(paths.open_fd(target, os.O_RDONLY))

    assert _flags_for(recorded_flags, "data.csv") == os.O_RDONLY | SENTINEL


def test_dir_open_adds_o_binary(tmp_path, recorded_flags):
    (tmp_path / "data.csv").write_bytes(CSV_CRLF)

    with paths.pinned_dir(tmp_path) as directory:
        os.close(paths.dir_open(directory, "data.csv", os.O_RDONLY | paths.O_NOFOLLOW))

    assert _flags_for(recorded_flags, "data.csv") & SENTINEL


def test_draft_file_open_is_binary(tmp_path, recorded_flags):
    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "report"
    folder.mkdir(parents=True)
    (folder / "data.csv").write_bytes(CSV_CRLF)
    source = ProjectArtifacts(
        base=base,
        project_id=None,
        project_name="project",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )

    resources, _fd, _stat = workspace_ep._open_pinned_draft_file(
        source, folder, ("data.csv",)
    )
    resources.close()

    assert _flags_for(recorded_flags, "data.csv") & SENTINEL


def test_project_file_open_is_binary(tmp_path, recorded_flags):
    base = tmp_path.resolve() / "project"
    base.mkdir()
    target = base / "data.csv"
    target.write_bytes(CSV_CRLF)

    cm, _fd, _stat = project_files_ep._pinned_regular_file(target, base)
    cm.__exit__(None, None, None)

    assert _flags_for(recorded_flags, "data.csv") & SENTINEL
