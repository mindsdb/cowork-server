"""The on-disk grant that opens one draft to same-org review."""
from __future__ import annotations

import json

from cowork.services.artifact_draft_review import (
    disable_draft_review,
    draft_review_allows,
    draft_review_grant,
    enable_draft_review,
)

ORG = "11111111-1111-1111-1111-111111111111"
OTHER_ORG = "22222222-2222-2222-2222-222222222222"
OWNER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _artifact(tmp_path):
    folder = tmp_path / "sales-report-a1b2c3d4"
    folder.mkdir()
    (folder / "index.html").write_text("<html></html>")
    return folder


def test_a_draft_is_private_until_its_owner_grants_review(tmp_path):
    folder = _artifact(tmp_path)

    assert draft_review_grant(folder) is None
    assert draft_review_allows(folder, ORG) is False

    enable_draft_review(folder, org_id=ORG, enabled_by=OWNER)

    assert draft_review_allows(folder, ORG) is True


def test_the_grant_lives_inside_the_revision_journal(tmp_path):
    """`.revisions/` is already excluded from publish bundles, from `files[]` and
    from the draft preview, so the marker cannot leak to a viewer."""
    folder = _artifact(tmp_path)

    enable_draft_review(folder, org_id=ORG, enabled_by=OWNER)

    assert (folder / ".revisions" / "draft-review.json").is_file()


def test_enabling_twice_keeps_the_first_record(tmp_path):
    """The client calls this on mount, so a repeat must not rewrite the file."""
    folder = _artifact(tmp_path)
    first = enable_draft_review(folder, org_id=ORG, enabled_by=OWNER)

    again = enable_draft_review(folder, org_id=ORG, enabled_by="someone-else")

    assert again == first
    assert again["enabledBy"] == OWNER


def test_another_organization_is_not_covered_by_the_grant(tmp_path):
    """A project row can move between organizations; the grant names the one it
    was made for and does not follow."""
    folder = _artifact(tmp_path)
    enable_draft_review(folder, org_id=ORG, enabled_by=OWNER)

    assert draft_review_allows(folder, OTHER_ORG) is False
    assert draft_review_allows(folder, None) is False


def test_a_damaged_grant_fails_closed(tmp_path):
    folder = _artifact(tmp_path)
    enable_draft_review(folder, org_id=ORG, enabled_by=OWNER)
    (folder / ".revisions" / "draft-review.json").write_text("{not json")

    assert draft_review_grant(folder) is None
    assert draft_review_allows(folder, ORG) is False


def test_a_record_without_the_organization_scope_is_ignored(tmp_path):
    folder = _artifact(tmp_path)
    enable_draft_review(folder, org_id=ORG, enabled_by=OWNER)
    path = folder / ".revisions" / "draft-review.json"
    payload = json.loads(path.read_text())
    payload["scope"] = "everyone"
    path.write_text(json.dumps(payload))

    assert draft_review_allows(folder, ORG) is False


def test_disable_removes_the_grant_and_reports_whether_one_was_there(tmp_path):
    folder = _artifact(tmp_path)
    enable_draft_review(folder, org_id=ORG, enabled_by=OWNER)

    assert disable_draft_review(folder) is True
    assert draft_review_allows(folder, ORG) is False
    assert disable_draft_review(folder) is False
