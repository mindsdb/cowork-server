"""Which comments scope a published artifact uses, read from `.published.json`.

Publishing is a source switch: while an artifact is live in `restricted` mode
its threads live in the cloud comments service, and at every other moment they
live in the local journal. Nothing moves between the two — the caller only picks
a scope. The decision is made from the publication RECORD, never from the shape
of the key the client sent: the stable identity `artifact/<uuid>` addresses the
local journal and (since the stable-key publish flow) the cloud service too, so
the string alone cannot say where a thread lives.

Two cloud scopes exist, and which one applies depends on the deployed lambda:

* ``artifact/<uuid>`` — the stable key, echoed back by the upload lambda and
  persisted in its ``_meta.json``. The viewer renders the comments UI against
  it, and auth holds the access rule under it, having deleted the composite
  alias. Survives re-publication under a new report URL.
* ``{user_dir}/{report_id}`` — the historical composite, derived by the viewer
  from the published URL. Still what an environment running the older lambda
  uses.

Restricted is the only mode that qualifies: auth mirrors an access rule for
restricted publications only, and inference authorizes every comment call
against that rule, so a public or password-protected artifact answers 403.
"""
from __future__ import annotations

from urllib.parse import urlparse


def cloud_key_from_url(url: str) -> str:
    """``{user_dir}/{report_id}`` from a static viewer URL, else "".

    Static publications land on ``/view/<user_dir>/<report_id>``. Fullstack ones
    land on ``/a/<report_id>``: the user_dir lives only in the gateway's
    ``_meta.json``, so there is nothing to recover from the URL alone.
    """
    try:
        path = urlparse(url or "").path
    except ValueError:
        return ""
    parts = [segment for segment in path.split("/") if segment]
    if len(parts) == 3 and parts[0] == "view":
        return f"{parts[1]}/{parts[2]}"
    return ""


def published_entry_mode(entry: dict) -> str:
    """Access mode of a `.published.json` entry.

    ``mode`` is authoritative; entries written before that field existed carry
    only the ``requires_password`` flag.
    """
    return entry.get("mode") or ("password" if entry.get("requires_password") else "public")


def cloud_comments_scope(entry: dict | None) -> str:
    """Cloud comments scope for a `.published.json` entry, or "" for local.

    "" covers every case that must keep using the local journal: no record, a
    soft-deleted publication (``published: False``, written by unpublish so a
    later re-publish can reuse the URL), a non-restricted mode, and a record
    from which no scope can be derived at all.

    ``lambda_artifact_key`` is the upload lambda's OWN echo and is trusted
    verbatim. The sibling ``artifact_key`` field is not consulted: it carries
    cowork's canonical fallback when the lambda said nothing, so reading it
    would route a legacy publication to a scope no viewer ever writes to.
    """
    if not isinstance(entry, dict):
        return ""
    if not entry.get("published", True):
        return ""
    if published_entry_mode(entry) != "restricted":
        return ""
    echoed = str(entry.get("lambda_artifact_key") or "").strip()
    if echoed:
        return echoed
    return cloud_key_from_url(str(entry.get("url") or ""))
