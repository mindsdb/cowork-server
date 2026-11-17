"""Filesystem observations and image inputs shared by artifact tests."""
from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path


def _snapshot(folder: Path) -> dict[str, object]:
    """Every entry below *folder*: a file's bytes, or a symlink's target."""
    entries: dict[str, object] = {}
    for directory, dirnames, filenames in os.walk(folder):
        for name in [*dirnames, *filenames]:
            path = Path(directory) / name
            key = path.relative_to(folder).as_posix()
            if path.is_symlink():
                entries[key] = ("symlink", os.readlink(path))
            elif path.is_file():
                entries[key] = path.read_bytes()
    return entries


def padded_png(size: int, *, filler: bytes = b"x") -> bytes:
    """A valid 1x1 PNG of *size* bytes, padded with a private ancillary chunk
    that repeats *filler*."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    head = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    tail = chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b"")
    padding = size - len(head) - len(tail) - 12
    filled = (filler * (padding // len(filler) + 1))[:padding]
    return head + chunk(b"teXt", filled) + tail
