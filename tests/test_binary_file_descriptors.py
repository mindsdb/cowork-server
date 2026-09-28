"""Regular-file descriptors are opened in binary mode (ENG-2950).

On Windows, ``os.open`` without ``O_BINARY`` returns a CRT text-mode
descriptor: reads turn CRLF into LF and stop at 0x1A, writes turn LF into
CRLF. ``O_BINARY`` does not exist on POSIX, so these tests patch it to a free
bit and check that the bit reaches ``os.open``.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.api.v1.endpoints import project_files as project_files_ep
from cowork.common import paths
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import GlobalMemoryStore
from cowork.services import files as files_service
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


def test_dir_open_without_a_directory_descriptor_adds_o_binary(tmp_path, recorded_flags):
    """The ``d.fd is None`` branch is the one Windows runs (no ``dir_fd``)."""
    (tmp_path / "data.csv").write_bytes(CSV_CRLF)

    os.close(paths.dir_open(paths.PinnedDir(None, tmp_path), "data.csv", os.O_RDONLY))

    assert _flags_for(recorded_flags, "data.csv") == os.O_RDONLY | SENTINEL


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


COWORK_ROOT = Path(paths.__file__).resolve().parents[1]

# Raw ``os.open`` calls that are allowed, by module, with their exact count.
# ``common/paths.py``: ``open_fd`` itself plus four directory opens. The
# revision journal opens a directory to fsync it. Text mode does not apply to
# directories. A changed count means a new raw open: check whether it is a
# regular file and belongs in ``open_fd``.
# Limitation: only the ``os.open(...)`` form is detected, not
# ``from os import open`` or ``posix.open``; the codebase uses neither.
RAW_OS_OPEN_ALLOWED = {
    "common/paths.py": 5,
    "services/artifact_revisions.py": 1,
}


def _raw_os_open_calls() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for source in sorted(COWORK_ROOT.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        if "os.open" not in text:
            continue
        tree = ast.parse(text, filename=str(source))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "open"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                rel = source.relative_to(COWORK_ROOT).as_posix()
                found.setdefault(rel, []).append(node.lineno)
    return found


def test_regular_files_are_opened_through_open_fd():
    offenders = []
    for rel, lines in _raw_os_open_calls().items():
        if rel not in RAW_OS_OPEN_ALLOWED:
            offenders.extend(f"{rel}:{line}" for line in lines)
            continue
        allowed = RAW_OS_OPEN_ALLOWED[rel]
        if len(lines) != allowed:
            offenders.append(f"{rel}: expected {allowed} os.open, found {lines}")
    assert offenders == [], (
        "Use cowork.common.paths.open_fd for regular files (ENG-2950): "
        + ", ".join(offenders)
    )


def test_stage_attachment_replaces_a_text_mode_copy(tmp_path):
    """A copy written through a text-mode descriptor is larger than its source
    (every LF became CRLF), so the size check must not treat it as done."""
    payload = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    src = tmp_path / "upload.png"
    src.write_bytes(payload)
    attachments = tmp_path / "attachments"
    stale = attachments / "file-1" / "upload.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(payload.replace(b"\n", b"\r\n"))
    assert stale.stat().st_size > len(payload)

    with paths.pinned_dir(attachments) as attach:
        files_service._stage_attachment(attach, "file-1", "upload.png", src, len(payload))

    assert stale.read_bytes() == payload


def _memory_slot_with_bytes(root: Path, payload: bytes) -> GlobalMemoryStore:
    store = GlobalMemoryStore(root=root)
    store.write(MemorySlot.RULES, "placeholder")
    (slot_file,) = [entry for entry in root.iterdir() if entry.is_file()]
    slot_file.write_bytes(payload)
    return store


def test_memory_read_undoes_the_legacy_text_mode_double_cr(tmp_path):
    """Before ENG-2950, a Windows slot was written through a text-mode
    descriptor: the text layer wrote CRLF and the CRT added another CR. The
    old text-mode read hid that; a binary read must too, or every line would
    come back followed by a blank one."""
    store = _memory_slot_with_bytes(tmp_path, b"first rule\r\r\nsecond rule\r\r\n")

    assert store.read(MemorySlot.RULES) == "first rule\nsecond rule\n"
    assert store.read_checked(MemorySlot.RULES) == (True, "first rule\r\nsecond rule\r\n")


def test_memory_read_keeps_ordinary_line_endings(tmp_path):
    store = _memory_slot_with_bytes(tmp_path, b"crlf\r\nlf\nlone cr\rend")

    assert store.read(MemorySlot.RULES) == "crlf\nlf\nlone cr\nend"
    assert store.read_checked(MemorySlot.RULES) == (True, "crlf\r\nlf\nlone cr\rend")
