"""A shared doc's page follows its edits, the way a shared Google Doc does.

Every write re-renders ``doc.html``; when that page is shared, this republishes
it a few seconds after the last write, keeping its URL and access. There is no
"update the shared copy" step for a person to take. Bursts (a person typing, the
agent filling sections) collapse into one republish.

Not in org mode: there, autopublish owns what a shared artifact shows.
"""

from __future__ import annotations

from pathlib import Path

from cowork.db.scoped import TenantScope
from cowork.services.artifacts import load_published_map
from cowork.services.publish import compute_publish_md5, desktop_publish_credential, update_artifact
from cowork.services.published_page_ownership import publication_owner

from .debounce import Debouncer
from .service import refresh_page
from .sharing import is_shared
from .store import DOC_HTML

DELAY_S = 4.0

_republishes = Debouncer(DELAY_S, what="republishing a shared doc")


def schedule(folder: Path, *, scope: TenantScope | None = None) -> bool:
    """Republish ``folder``'s page DELAY_S after the last write to it."""
    if (scope is not None and scope.org_mode) or not is_shared(folder):
        return False
    page = str(folder / DOC_HTML)

    def republish() -> None:
        # Asked again when it runs, under the page's publish lock, which Stop
        # sharing (unpublish_artifact) holds while it takes the page down: an
        # unshare can't land between the check and the upload and see the page
        # put back up. The lock is re-entrant, so the publish inside takes it too.
        _api_key, publish_url = desktop_publish_credential()
        with publication_owner(folder / DOC_HTML, published_dir=folder, publish_url=publish_url, scope=scope):
            if is_shared(folder):
                update_artifact(page, scope=scope)

    _republishes.schedule(page, republish)
    return True


def _went_out_as_it_is(folder: Path) -> bool:
    entry = load_published_map(folder).get(DOC_HTML) or {}
    md5 = compute_publish_md5(folder / DOC_HTML, artifacts_base=folder.parent)
    return md5 is not None and md5 == entry.get("last_md5")


def catch_up(folder: Path, *, scope: TenantScope | None = None) -> None:
    """At startup: re-render a shared doc's page, and republish it if that isn't what went out.

    An edit made just before the app quit had its republish still waiting; a
    page rendered by an older version of the app differs too.
    """
    refresh_page(folder)
    if not _went_out_as_it_is(folder):
        schedule(folder, scope=scope)


def stop() -> None:
    """Drop republishes still waiting (shutdown). ``catch_up`` sends them at the next start."""
    _republishes.cancel_all()


def pending() -> list[str]:
    return _republishes.pending()
