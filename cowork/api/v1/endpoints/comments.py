"""Artifact-comments endpoints: local journal or a proxy to the cloud service.

Mounted at /api/v1/artifact-comments. The renderer calls these without auth; the
proxy attaches the user's MindsHub credential upstream (see comments_proxy).

Publishing is what switches the source. The renderer always addresses an
artifact by its canonical `artifact/<uuid>` key, which - since the stable-key
publish flow - is ALSO the scope the published page uses, so the key cannot
decide the route. The publication record can: a live `restricted` publication
has an access rule in auth and threads in the cloud service, and nothing else
does.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from cowork.services.artifact_identity import resolve_artifact_folder
from cowork.services.artifact_roots import artifacts_sources_for_scan
from cowork.services.comments_proxy import forward_comments_rest, forward_comments_stream
from cowork.services.comments_scope import cloud_comments_scope
from cowork.services.local_artifact_comments import handle_local_comments, local_comments_stream
from cowork.services.publish import published_owner_state

router = APIRouter()

#: First segment of the canonical `artifact/<uuid>` key. Only a key in that
#: shape names a local artifact; anything else is already a cloud scope.
CANONICAL_USER_DIR = "artifact"


def _org_mode() -> bool:
    from cowork.common.settings.app_settings import get_app_settings

    return get_app_settings().tenancy_mode == "org"


def resolve_comments_route(user_dir: str, report_id: str) -> tuple[str, str] | None:
    """Upstream scope for this artifact, or None to use the local journal.

    Org tenancy has no local journal: drafts and publications share one cloud
    scope, so the incoming key is forwarded unchanged.

    A key that is not `artifact/<uuid>` is already a cloud scope - the
    historical `{user_dir}/{report_id}` composite - and is forwarded unchanged
    too. The renderer stopped minting those, but an OTA renderer can lag the
    main process by a release and still hold one from an older card, and it
    addresses a real published artifact.

    Otherwise the answer comes from `.published.json`. Anything unresolvable -
    an unknown id, a folder outside the authorized roots, a duplicated identity
    - falls back to local, where the existing handler raises the user-facing
    404/409. Failing OPEN toward the proxy would send the user's credential
    upstream for an artifact we could not even identify.
    """
    if _org_mode() or user_dir != CANONICAL_USER_DIR:
        return user_dir, report_id
    try:
        _source, folder, _metadata = resolve_artifact_folder(
            artifacts_sources_for_scan(), report_id
        )
    except Exception:
        return None
    scope = cloud_comments_scope(published_owner_state(str(folder)))
    if not scope:
        return None
    upstream_user_dir, _, upstream_report_id = scope.partition("/")
    if not upstream_user_dir or not upstream_report_id:
        return None
    return upstream_user_dir, upstream_report_id


@router.get("/{user_dir}/{report_id}/stream")
async def comments_stream(user_dir: str, report_id: str, request: Request):
    # SSE — registered before the catch-all so it isn't swallowed by {subpath:path}.
    route = resolve_comments_route(user_dir, report_id)
    if route is None:
        return local_comments_stream(report_id)
    return await forward_comments_stream(request, route[0], route[1])


@router.api_route(
    "/{user_dir}/{report_id}/{subpath:path}",
    methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
)
async def comments_rest(user_dir: str, report_id: str, subpath: str, request: Request):
    # threads (list/create/edit/delete), replies (add/edit/delete), status.
    route = resolve_comments_route(user_dir, report_id)
    if route is None:
        return await handle_local_comments(request, report_id, subpath)
    return await forward_comments_rest(request, route[0], route[1], subpath)
