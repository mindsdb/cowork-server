"""Stop sharing, Update and the artifact status say when a page's old link stays
live, because no owner record ties the page to the artifact. The Stop sharing
cases are in fixtures/published_link_left_live/stop_sharing.yaml."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from cowork.api.v1.endpoints import artifacts as artifact_routes
from cowork.services import publish
from cowork.services.published_page_ownership import link_owner_recorded
from tests.test_local_publish_ownership import ALICE, BOB, publisher  # noqa: F401

STOP_SHARING = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "published_link_left_live" / "stop_sharing.yaml").read_text(encoding="utf-8")
)
OLD_LINK = "https://old-viewer.example/view/legacy-page/"


def _planted(publisher, kind: str):  # noqa: F811
    """A page whose publish record names a page id that is `kind` to Alice, and that id's link."""
    page = publisher.page("artifact")
    if kind == "owned":
        publisher.send(page)
        return page, publisher.entry(page)["url"]
    if kind == "other_artifact":
        original = publisher.page("original")
        publisher.send(original)
        publisher.plant(page, publisher.entry(original))
        return page, publisher.entry(original)["url"]
    publisher.pages["legacy-page"] = {"content": "legacy"}
    publisher.plant(page, {"report_id": "legacy-page", "url": OLD_LINK, "published": True})
    return page, OLD_LINK


@pytest.mark.parametrize("case", STOP_SHARING, ids=lambda case: case["name"])
def test_stop_sharing_says_whether_the_page_came_down(publisher, case):  # noqa: F811
    page, link = _planted(publisher, case["page"])
    result = publish.unpublish_artifact(
        page, artifacts_base=publisher.root, api_key="test-key", publish_url=publisher.endpoint, scope=ALICE,
    )
    assert result == {
        "status": "ok",
        "takenDown": case["taken_down"],
        "liveLink": link if case["live_link"] else None,
        "reason": case["reason"],
    }
    assert bool(publisher.deletes) is case["publisher_deletes"]


def test_update_over_a_page_with_no_owner_record_names_the_link_it_left_live(publisher):  # noqa: F811
    page, _ = _planted(publisher, "no_owner_record")
    result = publisher.send(page)
    assert result["replacedLink"] == OLD_LINK
    assert result["url"] != OLD_LINK
    assert publisher.pages["legacy-page"] == {"content": "legacy"}


def test_updating_the_testers_own_page_replaces_no_link(publisher):  # noqa: F811
    page, link = _planted(publisher, "owned")
    result = publisher.send(page)
    assert result["replacedLink"] is None
    assert result["url"] == link


def test_a_page_another_artifact_owns_is_not_reported_as_left_live_on_update(publisher):  # noqa: F811
    page, _ = _planted(publisher, "other_artifact")
    assert publisher.send(page)["replacedLink"] is None


def test_the_link_is_owned_only_when_an_owner_record_ties_it_to_this_artifact(publisher):  # noqa: F811
    owned, _ = _planted(publisher, "owned")
    assert link_owner_recorded(owned, publisher.entry(owned)["report_id"], publish_url=publisher.endpoint) is True
    legacy = publisher.page("legacy")
    publisher.plant(legacy, {"report_id": "legacy-page", "url": OLD_LINK, "published": True})
    assert link_owner_recorded(legacy, "legacy-page", publish_url=publisher.endpoint) is False
    # Another artifact's page, recorded for the first artifact.
    assert link_owner_recorded(legacy, publisher.entry(owned)["report_id"], publish_url=publisher.endpoint) is False
    # A publisher Cowork doesn't protect keeps no owners to miss.
    assert link_owner_recorded(legacy, "legacy-page", publish_url="https://llm.mdb.ai") is True


@pytest.mark.parametrize("published_url, recorded, expected", [
    ("https://viewer.example/view/a/", False, {"linkOwnerRecorded": False}),
    ("https://viewer.example/view/a/", True, {"linkOwnerRecorded": True}),
    ("", None, {}),
])
async def test_the_status_carries_the_owner_record_only_for_a_live_link(monkeypatch, published_url, recorded, expected):
    base = {"publishedUrl": published_url, "modified": False}
    monkeypatch.setattr(artifact_routes, "_desktop_artifact_status_for_path", lambda path, *, scope: dict(base))
    monkeypatch.setattr(publish, "published_link_owner_recorded", lambda path, *, scope: recorded)
    status = await artifact_routes.artifact_status(scope=ALICE, path="artifact/index.html")
    assert status == {**base, **expected}
