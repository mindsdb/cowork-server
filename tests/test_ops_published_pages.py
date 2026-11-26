"""The published-pages operator command: list every page the publisher serves,
take one down, and give a page with no owner to the person who published it.
Its refusals are in fixtures/ops_published_pages/refusals.yaml."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml
from sqlmodel import Session, delete, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.published_page import PublishedPageOwner
from cowork.ops.published_pages import main
from cowork.services.publish import PublishOwnedByOtherUser
from cowork.services.published_page_ownership import PublicationOwner, publisher_key
from cowork.services.shared_resources import _DatabaseResourceLocks, _resource_busy
from tests.test_ops_commands import ALICE, BOB, SUB_A, _arguments, _general, _run, stack  # noqa: F401

REFUSALS = yaml.safe_load((Path(__file__).parent / "fixtures" / "ops_published_pages" / "refusals.yaml").read_text(encoding="utf-8"))
PEOPLE = {"alice": ALICE, "bob": BOB}
TOKEN = "publish-token"


def _session() -> Session:
    return Session(get_engine(get_app_settings().database.uri))


def _owners(key: str) -> dict[str, tuple[str, str]]:
    with _session() as session:
        rows = session.exec(select(PublishedPageOwner).where(PublishedPageOwner.publisher_key == key)).all()
        return {row.report_id: (row.artifact_path, row.owner_user_id) for row in rows}


@pytest.fixture
def pages(stack, monkeypatch, tmp_path):  # noqa: F811
    """A publisher serving four pages, and Alice's project whose artifacts'
    publish records name three of them, as REFUSALS describes."""
    endpoint = f"https://publisher-{uuid4().hex}.example"
    monkeypatch.setenv("ANTON_PUBLISH_URL", endpoint)
    monkeypatch.setenv("COWORK_PUBLISH_TOKEN", TOKEN)
    get_app_settings.cache_clear()
    served = {report_id: f"{endpoint}/view/{report_id}/" for report_id in ("ownerless", "owned", "orphan", "twice")}
    deleted: list[str] = []

    def list_published(*, api_key, publish_url, ssl_verify=True):
        assert api_key == TOKEN
        return [
            {"report_id": report_id, "md5": "hash", "title": f"Page {report_id}", "view_url": url, "last_modified": "2026-10-20T00:00:00"}
            for report_id, url in served.items()
        ]

    def unpublish(report_id, *, api_key, publish_url, ssl_verify=True):
        assert api_key == TOKEN
        deleted.append(report_id)
        served.pop(report_id)
        return {"deleted": True, "report_id": report_id}

    monkeypatch.setattr("anton.publisher.list_published", list_published)
    monkeypatch.setattr("anton.publisher.unpublish", unpublish)

    artifacts = Path(_general(ALICE, tmp_path).path) / ".anton" / "artifacts"

    def artifact(folder: str, report_id: str) -> str:
        path = artifacts / folder
        path.mkdir(parents=True)
        (path / "index.html").write_text("<html><body>page</body></html>")
        entry = {"report_id": report_id, "url": served[report_id], "published": True, "mode": "password"}
        (path / ".published.json").write_text(json.dumps({"index.html": entry}))
        return str((path / "index.html").resolve())

    paths = {name: artifact(name, report_id) for name, report_id in (
        ("ownerless-page", "ownerless"), ("owned-page", "owned"), ("twice-a", "twice"), ("twice-b", "twice"),
    )}
    key = publisher_key(endpoint)
    with _session() as session:
        session.add(PublishedPageOwner(publisher_key=key, report_id="owned", artifact_path=paths["owned-page"], owner_user_id=SUB_A))
        session.commit()
    yield SimpleNamespace(served=served, deleted=deleted, artifacts=artifacts, paths=paths, key=key)
    with _session() as session:
        session.exec(delete(PublishedPageOwner).where(PublishedPageOwner.publisher_key == key))
        session.commit()


def _busy_lock(*_args, **_kwargs):
    """A publish record's lock that a publish still holds when the wait runs out. The tests'
    SQLite takes no lock, so this answers as PostgreSQL's would."""
    raise _resource_busy()


def _records(artifacts: Path) -> dict[str, str]:
    return {str(path): path.read_text() for path in sorted(artifacts.rglob(".published.json"))}


def test_list_names_each_page_its_owner_and_the_artifacts_that_name_it(pages, capsys):
    status, out, err = _run(main, ["list", "--json"], capsys)
    assert status == 0, err
    listed = {page["report_id"]: page for page in map(json.loads, out.splitlines())}
    assert listed["ownerless"]["owner_user_id"] is None
    assert listed["ownerless"]["artifact_paths"] == [pages.paths["ownerless-page"]]
    assert listed["owned"]["owner_user_id"] == SUB_A
    assert listed["orphan"]["artifact_paths"] == []
    assert sorted(listed["twice"]["artifact_paths"]) == sorted([pages.paths["twice-a"], pages.paths["twice-b"]])


def test_the_plain_list_says_which_pages_have_no_owner(pages, capsys):
    status, out, _ = _run(main, ["list"], capsys)
    assert status == 0
    lines = {line.split(" | ")[0]: line for line in out.splitlines()}
    assert "| no owner |" in lines["ownerless"]
    assert f"| owner {SUB_A} |" in lines["owned"]
    assert lines["orphan"].endswith("| no artifact names it")


@pytest.mark.parametrize("action, case", [
    (action, case) for action, cases in REFUSALS.items() for case in cases
], ids=lambda value: value["name"] if isinstance(value, dict) else value)
def test_a_refused_action_changes_nothing(pages, capsys, monkeypatch, action, case):
    argv = [action, *case["argv"], *(_arguments(PEOPLE[case["person"]]) if "person" in case else [])]
    if case.get("busy"):
        monkeypatch.setattr(_DatabaseResourceLocks, "lock", _busy_lock)
    records, owners, served = _records(pages.artifacts), _owners(pages.key), dict(pages.served)
    status, out, err = _run(main, argv, capsys)
    assert status == 1 and out == ""
    assert case["says"] in err
    assert (_records(pages.artifacts), _owners(pages.key), pages.served, pages.deleted) == (records, owners, served, [])


def test_unpublish_takes_a_page_with_no_owner_down_and_drops_the_record_that_named_it(pages, capsys):
    status, out, err = _run(main, ["unpublish", "--report-id", "ownerless"], capsys)
    assert status == 0, err
    assert pages.deleted == ["ownerless"] and "ownerless" not in pages.served
    assert "answers 404" in out
    assert json.loads((pages.artifacts / "ownerless-page" / ".published.json").read_text()) == {}
    # Another artifact's record is left alone.
    assert "index.html" in json.loads((pages.artifacts / "owned-page" / ".published.json").read_text())


def test_unpublish_with_force_takes_an_owned_page_down_and_removes_its_owner(pages, capsys):
    status, out, err = _run(main, ["unpublish", "--report-id", "owned", "--force"], capsys)
    assert status == 0, err
    assert pages.deleted == ["owned"]
    assert "owned" not in _owners(pages.key)
    assert f"its owner record (user {SUB_A})" in out


def test_give_makes_the_person_the_owner_so_cowork_sends_them_the_page_id_again(pages, capsys):
    status, out, err = _run(main, ["give", "--report-id", "ownerless", *_arguments(ALICE)], capsys)
    assert status == 0, err
    assert _owners(pages.key)["ownerless"] == (pages.paths["ownerless-page"], SUB_A)
    assert "Access: password" in out
    with _session() as session:
        # What Cowork asks before an update: Alice keeps the link, Bob is refused.
        assert PublicationOwner(session, pages.key, pages.paths["ownerless-page"], SUB_A).checked_report_id("ownerless") == "ownerless"
        with pytest.raises(PublishOwnedByOtherUser):
            PublicationOwner(session, pages.key, pages.paths["ownerless-page"], BOB.user_id).checked_report_id("ownerless")
    assert pages.deleted == []


def test_every_action_refuses_a_deployment_that_trusts_no_identity(pages, monkeypatch, capsys):
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    get_app_settings.cache_clear()
    for argv in (["list"], ["unpublish", "--report-id", "ownerless"], ["give", "--report-id", "ownerless", *_arguments(ALICE)]):
        status, _, err = _run(main, argv, capsys)
        assert status == 1 and "trusts identity headers" in err
    assert pages.deleted == []
