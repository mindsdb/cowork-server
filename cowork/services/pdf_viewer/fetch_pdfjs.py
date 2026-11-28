"""Download the pdf.js files of the PDF viewer at build time.

The files come from the pdfjs-dist package on npm and are not committed. The
tarball is checked against SHA512 before anything is unpacked, and only the
files the viewer serves are written, into pdfjs/ next to this file:

    python3 fetch_pdfjs.py [DEST]

A folder that already holds this version is left as it is. PDFJS_TARBALL set
to the path of pdfjs-dist-<VERSION>.tgz unpacks that file instead of
downloading it (offline builds); it is checked the same way.

The shell layer's Makefile and cowork-server's wheel build hook run this file
before anything is installed, so it uses only the standard library.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

VERSION = "6.4.299"
URL = f"https://registry.npmjs.org/pdfjs-dist/-/pdfjs-dist-{VERSION}.tgz"
# The npm "integrity" value of the tarball.
SHA512 = "sha512-AVl138zALtfaAPvADulE0PZThbYzCBS79nL4pOSL/6Sm/4AH5A21BD9VHt97OlCuzJuCpmeZtAtkinisF4Vb1g=="

DEST = Path(__file__).resolve().parent / "pdfjs"
# Prefix of the folder a run unpacks into before it replaces DEST.
STAGING_PREFIX = ".pdfjs-"
# Written last into the folder; its text says which tarball the folder holds.
STAMP = ".pdfjs-dist"
_MAX_TARBALL = 64 * 1024 * 1024
_TIMEOUT_S = 60

# Package path -> served name. The legacy builds also run in older Safari.
_FILES = {
    "package/LICENSE": "LICENSE",
    "package/legacy/build/pdf.min.mjs": "pdf.min.mjs",
    "package/legacy/build/pdf.worker.min.mjs": "pdf.worker.min.mjs",
    "package/legacy/web/pdf_viewer.mjs": "pdf_viewer.mjs",
    "package/legacy/web/pdf_viewer.css": "pdf_viewer.css",
}
# Package folder -> served folder; every file directly inside is taken.
_FOLDERS = {
    "package/legacy/web/images/": "images/",
    "package/cmaps/": "cmaps/",
    "package/standard_fonts/": "standard_fonts/",
    "package/wasm/": "wasm/",
}
# The scripting sandbox is never enabled, so its engine is not shipped.
_SKIPPED = ("package/wasm/quickjs-eval.",)


def _target(name: str) -> str | None:
    if name in _FILES:
        return _FILES[name]
    if name.startswith(_SKIPPED):
        return None
    for folder, served in _FOLDERS.items():
        rest = name[len(folder):]
        if name.startswith(folder) and rest and "/" not in rest and rest not in (".", ".."):
            return served + rest
    return None


def _stamp_text() -> str:
    return f"pdfjs-dist {VERSION} {SHA512}\n"


def is_current(dest: Path = DEST) -> bool:
    """Whether dest already holds the files of this version."""
    try:
        return (dest / STAMP).read_text(encoding="utf-8") == _stamp_text()
    except OSError:
        return False


def _tarball() -> bytes:
    local = os.environ.get("PDFJS_TARBALL")
    if local:
        data = Path(local).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=_TIMEOUT_S) as response:
            data = response.read(_MAX_TARBALL + 1)
    if len(data) > _MAX_TARBALL:
        raise RuntimeError(f"pdfjs-dist {VERSION}: the tarball is larger than {_MAX_TARBALL} bytes")
    digest = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")
    if digest != SHA512:
        raise RuntimeError(f"pdfjs-dist {VERSION}: the tarball does not match the pinned SHA512")
    return data


def _unpack(data: bytes, into: Path) -> None:
    written = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            served = _target(member.name)
            if served is None or not member.isreg():
                continue
            path = into.joinpath(*PurePosixPath(served).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            path.write_bytes(source.read())
            written.add(served)
    missing = set(_FILES.values()) - written
    missing |= {served for served in _FOLDERS.values() if not any(name.startswith(served) for name in written)}
    if missing:
        raise RuntimeError(f"pdfjs-dist {VERSION}: the tarball lacks {sorted(missing)}")


def fetch(dest: Path = DEST) -> bool:
    """Put the pdf.js files into dest; False when they were already there."""
    dest = Path(dest)
    if is_current(dest):
        return False
    data = _tarball()
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=dest.parent))
    try:
        _unpack(data, staging)
        (staging / STAMP).write_text(_stamp_text(), encoding="utf-8")
        shutil.rmtree(dest, ignore_errors=True)
        staging.rename(dest)
    except OSError:
        # Another build on the same tree may have put the same files in place
        # first (a layer build and `make test/unit`, `uv sync` and `uv build`).
        if is_current(dest):
            return False
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return True


def main(argv: list[str]) -> int:
    dest = Path(argv[1]) if len(argv) > 1 else DEST
    fetched = fetch(dest)
    print(f"pdfjs-dist {VERSION} {'unpacked into' if fetched else 'already in'} {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
