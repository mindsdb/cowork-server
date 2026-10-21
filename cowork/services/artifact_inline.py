"""Make an artifact's HTML self-contained, with no network and no file outside it.

Every asset the document keeps becomes a ``data:`` URI read from the pinned
artifact folder; every other reference is dropped and listed in ``omitted``.
"""
from cowork.common.paths import PinnedDir
from cowork.schemas.artifact_export import InlinedDocument, InlineLimits, InlineMode


def inline_artifact_html(
    *,
    html: bytes,
    folder: PinnedDir,
    base_parts: tuple[str, ...],
    mode: InlineMode,
    limits: InlineLimits,
) -> InlinedDocument:
    raise NotImplementedError
