"""Which comments scope a `.published.json` entry selects.

"" means the local journal. A non-empty value is the scope the published page
writes to, and the only shape the proxy may forward.
"""
from __future__ import annotations

import pytest

from cowork.services.comments_scope import (
    cloud_comments_scope,
    cloud_key_from_url,
    published_entry_mode,
)

STABLE_KEY = "artifact/e9267de0-3fa4-4c17-b964-dc63216311cf"

# A publication made by the CURRENT deployed lambda: no `artifact_key` echo, so
# the viewer keys threads by the composite scope carried in the URL.
LEGACY = {
    "report_id": "c675003f",
    "url": "https://view.dev.mindshub.ai/view/b9996ebec/c675003f",
    "artifact_key": STABLE_KEY,
    "published": True,
    "mode": "restricted",
}

# A publication made by services#195: the lambda echoed the stable key, stored
# it in _meta.json, and auth's rule now lives under it (the composite alias is
# deleted), so the composite scope is dead for this artifact.
STABLE = {**LEGACY, "lambda_artifact_key": STABLE_KEY}


def test_lambda_echo_wins():
    assert cloud_comments_scope(STABLE) == STABLE_KEY


def test_without_echo_falls_back_to_the_composite_from_url():
    # `artifact_key` in the entry is cowork's own canonical fallback, not an
    # echo — it must never be mistaken for one.
    assert cloud_comments_scope(LEGACY) == "b9996ebec/c675003f"


@pytest.mark.parametrize("mode", ["public", "password"])
def test_non_restricted_publication_stays_local(mode):
    # auth mirrors an access rule for restricted mode only; anything else 403s
    # upstream, so those artifacts keep using the local journal.
    assert cloud_comments_scope({**STABLE, "mode": mode}) == ""


def test_soft_deleted_publication_stays_local():
    # unpublish keeps the record (so a re-publish reuses the URL) and only
    # flips the flag.
    assert cloud_comments_scope({**STABLE, "published": False}) == ""


def test_fullstack_without_echo_stays_local():
    # Fullstack publishes to /a/<report_id>: the user_dir lives only in the
    # gateway's _meta.json, so there is no composite to recover.
    entry = {**LEGACY, "url": "https://view.dev.mindshub.ai/a/c675003f"}
    assert cloud_comments_scope(entry) == ""


def test_fullstack_with_echo_uses_the_stable_key():
    entry = {**STABLE, "url": "https://view.dev.mindshub.ai/a/c675003f"}
    assert cloud_comments_scope(entry) == STABLE_KEY


def test_missing_and_malformed_entries_stay_local():
    assert cloud_comments_scope(None) == ""
    assert cloud_comments_scope({}) == ""
    assert cloud_comments_scope({"published": True, "mode": "restricted"}) == ""


def test_legacy_entry_without_mode_reads_requires_password():
    assert published_entry_mode({"requires_password": True}) == "password"
    assert published_entry_mode({}) == "public"
    assert published_entry_mode({"mode": "restricted"}) == "restricted"


def test_cloud_key_from_url_rejects_anything_but_the_view_shape():
    assert cloud_key_from_url("https://view.dev.mindshub.ai/view/u/r") == "u/r"
    assert cloud_key_from_url("https://view.dev.mindshub.ai/view/u/r/extra") == ""
    assert cloud_key_from_url("https://view.dev.mindshub.ai/view/u") == ""
    assert cloud_key_from_url("") == ""
    assert cloud_key_from_url("not a url") == ""
