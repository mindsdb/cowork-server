"""Python bytecode inside an artifact folder is not artifact content."""

import os
from pathlib import Path

from cowork.services.artifacts import (
    _user_files_with_mtimes,
    content_mtime_ns,
    is_non_content_path,
)


def _write(path: Path, body: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_user_files_skip_bytecode_at_any_depth(tmp_path: Path):
    for rel in (
        "index.html",
        "routes/x.py",
        "__pycache__/backend.cpython-313.pyc",
        "routes/__pycache__/x.cpython-313.pyc",
        "stray.pyc",
        "static/prd.md",
    ):
        _write(tmp_path / rel)

    names = {p.relative_to(tmp_path).as_posix() for p, _ in _user_files_with_mtimes(tmp_path)}

    assert names == {"index.html", "routes/x.py", "static/prd.md"}


def test_bytecode_does_not_move_the_content_mtime(tmp_path: Path):
    index = _write(tmp_path / "index.html")
    os.utime(index, ns=(1_000_000_000, 1_000_000_000))
    before = content_mtime_ns(tmp_path)

    cache = _write(tmp_path / "routes" / "__pycache__" / "x.cpython-313.pyc")
    os.utime(cache, ns=(9_000_000_000, 9_000_000_000))

    assert content_mtime_ns(tmp_path) == before


def test_bytecode_paths_are_non_content(tmp_path: Path):
    container = tmp_path / "artifacts"
    folder = container / "app-12345678"
    nested = _write(folder / "routes" / "__pycache__" / "x.cpython-313.pyc")
    stray = _write(folder / "x.pyc")
    index = _write(folder / "index.html")

    assert is_non_content_path(nested, container)
    assert is_non_content_path(stray, container)
    assert not is_non_content_path(index, container)
