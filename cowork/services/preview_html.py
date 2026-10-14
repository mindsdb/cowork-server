"""What an HTML artifact preview gets injected, and in what order.

Kept in services rather than beside the HTTP helpers, because preview_proxy
imports it too and cowork/services/ holds no imports from cowork.api today.
"""
from __future__ import annotations

from cowork.services.comments_layer import inject_layer
from cowork.services.preview_shim import inject_shim


def prepare_preview_html(html: str, *, comments: bool) -> str:
    """The shim first, then the comment layer when the renderer asked for it.

    Order matters in both directions: the shim has to precede the page's own
    scripts, and the layer has to sit at the end of the body where its
    absolutely positioned nodes belong.
    """
    prepared = inject_shim(html)
    return inject_layer(prepared) if comments else prepared
