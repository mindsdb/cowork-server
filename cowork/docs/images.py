"""Images on this computer, copied beside the doc so the doc can show them.

A doc's image comes from the web, a data: image or a file beside the doc
(model.image_src_ok): exports and shared pages read whatever an image points
at, so a path elsewhere on the machine is never kept. An image the agent may
use (one attached to the conversation, or one in the project) is copied into
the doc's ``images/`` folder and the write points there instead. The person's
pasted or dropped images arrive as bytes and land the same way.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .markdown import _IMAGE
from .model import DocError, image_src_ok
from .store import atomic_write

MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGES_DIR = "images"

_IMAGE_LINE = re.compile(_IMAGE.pattern, re.M)
_LOCAL_PATH_MESSAGE = ("an image's src must be a web address (https://…), a data: image, or a file beside the doc; "
                       "{src} can't be used. Attached images and images in the project can be: give their full path.")


def image_kind(data: bytes) -> str | None:
    """The file extension for image bytes this editor shows, from their first bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def image_name(data: bytes) -> str:
    """Where image bytes live beside a doc (named by their hash), refused unless a real image within the limit."""
    if len(data) > MAX_IMAGE_BYTES:
        raise DocError("too_big", f"an image is at most {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    kind = image_kind(data)
    if kind is None:
        raise DocError("bad_block", "only PNG, JPEG, GIF and WebP images can go in a doc.")
    return f"{IMAGES_DIR}/{hashlib.sha256(data).hexdigest()[:16]}.{kind}"


def write_images(folder: Path, files: dict[str, bytes]) -> None:
    """Write images (``{src: bytes}`` from ``image_name``) beside the doc in ``folder``."""
    for src, data in files.items():
        target = folder / src
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(target, data)


def store_image(folder: Path, data: bytes) -> str:
    """Write image bytes beside the doc and return the src that shows them."""
    src = image_name(data)
    write_images(folder, {src: data})
    return src


def _local_path(src: str) -> Path | None:
    s = (src or "").strip()
    if s.startswith("file://"):
        s = s[len("file://"):]
    if s.startswith("~"):
        s = str(Path(s).expanduser())
    return Path(s) if s.startswith("/") else None


def localize_images(value: Any, allowed: Callable[[Path], bool]) -> tuple[Any, dict[str, bytes]]:
    """``value`` (ops, or a create's markdown or tabs) with each usable local image pointed beside the doc.

    Returns the rewritten value and the images to write there (``write_images``).
    A local image ``allowed`` accepts, that is a real image within the size
    limit, is taken; any other local path is refused here with a message the
    agent can act on: as a Markdown line it would otherwise stay in the doc as
    its literal text.
    """
    files: dict[str, bytes] = {}

    def fix_src(src: str) -> str:
        path = _local_path(src)
        if path is None:
            return src
        try:
            resolved = path.resolve()
            if allowed(resolved) and resolved.is_file() and resolved.stat().st_size <= MAX_IMAGE_BYTES:
                data = resolved.read_bytes()
                if image_kind(data):
                    name = image_name(data)
                    files[name] = data
                    return name
        except OSError:
            pass
        raise DocError("bad_block", _LOCAL_PATH_MESSAGE.format(src=src))

    def fix_markdown(md: str) -> str:
        def line(m: re.Match[str]) -> str:
            src = m.group(2)
            return m.group(0) if image_src_ok(src) else m.group(0).replace(src, fix_src(src), 1)
        return _IMAGE_LINE.sub(line, md)

    def walk(v: Any) -> Any:
        if isinstance(v, list):
            return [walk(x) for x in v]
        if not isinstance(v, dict):
            return v
        out = {}
        for k, x in v.items():
            if k == "markdown" and isinstance(x, str):
                out[k] = fix_markdown(x)
            elif k == "src" and isinstance(x, str) and not image_src_ok(x):
                out[k] = fix_src(x)
            else:
                out[k] = walk(x)
        return out

    return (fix_markdown(value) if isinstance(value, str) else walk(value)), files


def within(*roots: Path) -> Callable[[Path], bool]:
    """A test that a resolved path is inside one of ``roots`` (or is one of them, for single files)."""
    resolved = [r.resolve() for r in roots]

    def check(path: Path) -> bool:
        return any(path == r or r in path.parents for r in resolved)
    return check


__all__ = ["IMAGES_DIR", "MAX_IMAGE_BYTES", "image_kind", "image_name", "localize_images", "store_image",
           "within", "write_images"]
