"""What a doc does around sharing, reached without importing it.

A doc's published page is rendered from its doc, and while it is shared its
owner can edit it from the link through a live copy in MindsHub. Keeping those
in step is ``cowork.docs``'s job, and it sits above the publish service, so it
registers these hooks at startup (``cowork.docs.cloud.install``). Publishing,
unpublishing and deleting call them whichever way they were asked.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple


class DocShareHooks(NamedTuple):
    # The doc's page is about to be uploaded.
    before_share: Callable[[Path], None]
    # The page went out, or its access changed.
    after_share: Callable[[Path], None]
    # The page was taken down: its publish entry as it stood while shared, and
    # the doc's folder (None when the doc is being deleted).
    after_unshare: Callable[[dict[str, Any], Path | None], None]


_hooks: DocShareHooks | None = None


def register(hooks: DocShareHooks | None) -> None:
    global _hooks
    _hooks = hooks


def registered() -> DocShareHooks | None:
    return _hooks
