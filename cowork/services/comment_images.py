"""Images people paste into artifact comments (ENG-4410).

A comment's text is all the comments service keeps (10,000 characters, local
journal and cloud alike), so an image is stored here, in the artifact's own
journal folder, and the comment names it with a marker line the app reads:

    ![screenshot.png](cowork-image:<32 hex>.png)

Whoever may review the artifact may add an image and read one back, as they
may its comments. An image is served only as the raster type its own bytes
say it is (PNG, JPEG, GIF or WebP), never as what the upload claimed, so an
SVG or an HTML page sent as an "image" is refused, and nothing stored here is
ever rendered as a document. The same bounds apply to the quick edit bar's
images, which go to the model with the request and are never stored.
"""
from __future__ import annotations

import base64
import binascii
import re
import uuid
from pathlib import Path

from cowork.services.artifact_journal import (
    JournalFileRefused,
    read_journal_file,
    write_journal_file,
)

#: The journal folder the images live in, `.revisions/comment-images`.
COMMENT_IMAGES_DIRNAME = "comment-images"
#: The largest image, as the model providers take one.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
#: The most images one quick edit sends.
MAX_REQUEST_IMAGES = 4

_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}
_MEDIA_TYPES = {ext: media for media, ext in _EXTENSIONS.items()}

#: An image's id: what the store names it, and what a comment's marker carries.
IMAGE_ID = re.compile(r"^[0-9a-f]{32}\.(png|jpg|gif|webp)$")
#: A comment's image marker, as the app writes it.
IMAGE_MARKER = re.compile(r"!\[[^\]\n]*\]\(cowork-image:([0-9a-f]{32}\.(?:png|jpg|gif|webp))\)")

#: The headers every image answer carries: no sniffing past the declared type,
#: and nothing that could run if the image were opened as a page.
IMAGE_RESPONSE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
    "Content-Disposition": "inline",
    "Cache-Control": "private, max-age=86400",
}


class ImageRefused(ValueError):
    """Bytes that are not an image this store keeps. ``reason`` is
    ``too_large`` or ``unsupported``."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def sniff_image_type(data: bytes) -> str | None:
    """The raster type ``data`` starts as, or None for anything else."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def checked_image(data: bytes) -> str:
    """The media type of ``data``; raises ImageRefused for anything that is
    too large or not one of the raster types."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageRefused("too_large", "Images can be up to 5 MB.")
    media_type = sniff_image_type(data)
    if media_type is None:
        raise ImageRefused("unsupported", "Only PNG, JPEG, GIF and WebP images can be attached.")
    return media_type


def decode_request_image(data: str) -> tuple[str, str]:
    """A base64 image from a request as ``(base64, media type)``, checked as
    a stored one is. Raises ImageRefused."""
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageRefused("unsupported", "The image is not valid base64.") from exc
    media_type = checked_image(raw)
    return base64.standard_b64encode(raw).decode("ascii"), media_type


def save_comment_image(folder: Path, data: bytes) -> dict:
    """Store ``data`` in the artifact at ``folder``; returns its ``id``,
    ``mediaType`` and ``size``. Raises ImageRefused."""
    media_type = checked_image(data)
    image_id = f"{uuid.uuid4().hex}.{_EXTENSIONS[media_type]}"
    write_journal_file(folder, COMMENT_IMAGES_DIRNAME, image_id, content=data)
    return {"id": image_id, "mediaType": media_type, "size": len(data)}


def read_comment_image(folder: Path, image_id: str) -> tuple[bytes, str]:
    """The bytes and media type of a stored image. FileNotFoundError when
    there is none by that id, or what stands there is not the image its id
    names (the artifact's own code can write the folder)."""
    match = IMAGE_ID.match(image_id or "")
    if not match:
        raise FileNotFoundError("Image not found")
    try:
        data = read_journal_file(folder, COMMENT_IMAGES_DIRNAME, image_id, max_bytes=MAX_IMAGE_BYTES)
    except JournalFileRefused as exc:
        raise FileNotFoundError("Image not found") from exc
    if data is None or sniff_image_type(data) != _MEDIA_TYPES[match.group(1)]:
        raise FileNotFoundError("Image not found")
    return data, _MEDIA_TYPES[match.group(1)]


def comment_image_ids(text: str) -> list[str]:
    """The ids of the images a comment's text names, in order, once each."""
    seen: list[str] = []
    for image_id in IMAGE_MARKER.findall(text or ""):
        if image_id not in seen:
            seen.append(image_id)
    return seen
