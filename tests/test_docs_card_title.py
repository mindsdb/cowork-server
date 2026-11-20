"""A doc's title is its artifact card's title: metadata.json's ``name`` follows every rename."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings
from cowork.docs import history, store
from cowork.docs.service import apply, create_doc
from cowork.docs.store import find_doc_folder, list_history, load_doc
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture
def made(tmp_path):
    return create_doc(tmp_path / "artifacts", title="Plan", markdown="Lead.", by="agent")


def _meta(folder: Path) -> dict:
    return json.loads((folder / "metadata.json").read_text())


def _rename(folder: Path, title: str, **kw):
    return apply(folder, [{"op": "rename_doc", "title": title}], **kw)


@pytest.mark.parametrize("who", [{"by": "user"}, {"by": "agent", "turn": "reply-2"}])
def test_a_rename_by_the_editor_or_the_agent_renames_the_card(made, who):
    before = _meta(made.folder)
    _rename(made.folder, "Hiring plan", **who)
    after = _meta(made.folder)
    assert after["name"] == "Hiring plan"
    assert {k: v for k, v in after.items() if k != "name"} == {k: v for k, v in before.items() if k != "name"}


def test_a_renamed_doc_is_found_by_its_new_title_not_its_old_one(made):
    _rename(made.folder, "Hiring plan", by="user")
    assert find_doc_folder(made.folder.parent, "hiring plan") == made.folder
    with pytest.raises(store.DocError) as missing:
        find_doc_folder(made.folder.parent, "Plan")
    assert missing.value.code == "no_doc"


def test_restoring_a_version_with_another_title_brings_that_title_to_the_card(made):
    _rename(made.folder, "Hiring plan", by="user")
    _rename(made.folder, "Budget", by="agent", turn="reply-2")  # ends the person's session: theirs is kept
    kept = next(v for v in list_history(made.folder) if v["by"] == "user")
    history.restore(made.folder, kept["rev"])
    assert load_doc(made.folder)["title"] == "Hiring plan"
    assert _meta(made.folder)["name"] == "Hiring plan"


def test_a_doc_renamed_before_the_card_followed_catches_up_on_its_next_write(made):
    meta = _meta(made.folder)
    (made.folder / "metadata.json").write_text(json.dumps({**meta, "name": "Old name"}))
    lead = load_doc(made.folder)["tabs"][0]["blocks"][0]
    apply(made.folder, [{"op": "update", "id": lead["id"], "set": {"text": "Typed."}}], by="user")
    assert _meta(made.folder)["name"] == "Plan"


def test_an_edit_that_keeps_the_title_leaves_metadata_untouched(made):
    path = made.folder / "metadata.json"
    stamp = path.stat().st_mtime_ns
    lead = load_doc(made.folder)["tabs"][0]["blocks"][0]
    apply(made.folder, [{"op": "update", "id": lead["id"], "set": {"text": "Typed."}}], by="user")
    assert path.stat().st_mtime_ns == stamp


def test_a_metadata_file_that_is_a_link_is_not_written_through(made, tmp_path):
    """The agent's code can write the doc's folder: a link planted at metadata.json must
    not let a rename write another artifact's metadata."""
    other = create_doc(tmp_path / "other", title="Other", markdown="x", by="agent")
    target = other.folder / "metadata.json"
    original = target.read_text()
    (made.folder / "metadata.json").unlink()
    (made.folder / "metadata.json").symlink_to(target)
    doc, _ = _rename(made.folder, "Hiring plan", by="user")
    assert doc["title"] == "Hiring plan"  # the edit still lands
    assert target.read_text() == original
    assert (made.folder / "metadata.json").is_symlink()


def test_a_card_that_cannot_be_renamed_is_one_line_in_the_log_not_a_traceback(made, caplog, tmp_path):
    other = create_doc(tmp_path / "other", title="Other", markdown="x", by="agent")
    (made.folder / "metadata.json").unlink()
    (made.folder / "metadata.json").symlink_to(other.folder / "metadata.json")
    _rename(made.folder, "Hiring plan", by="user")
    (record,) = [r for r in caplog.records if "card title" in r.getMessage()]
    assert record.exc_info is None


def test_a_doc_without_metadata_is_still_edited_and_logs_nothing(made, caplog):
    (made.folder / "metadata.json").unlink()
    doc, _ = _rename(made.folder, "Hiring plan", by="user")
    assert doc["title"] == "Hiring plan"
    assert not [r for r in caplog.records if "card title" in r.getMessage()]


@pytest.fixture
def client(monkeypatch, granted_product_permissions):
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield TestClient(create_app())
    get_app_settings.cache_clear()


def test_the_card_the_api_returns_carries_the_new_title(client):
    res = client.post("/api/v1/docs/", json={"projectId": str(GENERAL_PROJECT_ID), "title": "Plan", "markdown": "x"})
    assert res.status_code == 200, res.text
    url = f"/api/v1/docs/{GENERAL_PROJECT_ID}/{res.json()['artifactId']}"
    res = client.post(f"{url}/ops", json={"ops": [{"op": "rename_doc", "title": "Hiring plan"}]})
    assert res.status_code == 200, res.text
    assert client.get(url).json()["card"]["title"] == "Hiring plan"
