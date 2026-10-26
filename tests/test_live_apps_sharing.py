"""Sharing a live app means duplicating it: A shares an app of theirs with B's
address or with everyone; B sees it with A's address and duplicates it into a
project of B's own, after confirming the disclaimer. The copy takes a fixed
list of files, the same one anton launches, and never a link, a log, the
publish state or a port."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

# Imported at module scope on purpose: cowork.server builds an app at import
# time (see tests/test_trusted_identity_local.py).
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from tests._live_apps import (
    A,
    B,
    BEARER_TOKEN,
    C,
    EMAIL_A,
    MODULE,
    PARTITION_A,
    PARTITION_B,
    PARTITION_C,
    SUB_A,
    SUB_B,
    SUB_C,
    clear_live_app_rows,
    general_project,
    headers_for,
    person,
    person_scope,
    turn_on_live_apps,
)

SLUG = "orders-dash-1a2b3c4d"
APP_FILES = {
    "backend.py": "from fastapi import FastAPI\napp = FastAPI()\n",
    "requirements.txt": "fastapi\n",
    "state_manifest.json": "{}\n",
    "prd.md": "# Orders\n",
    "spec.md": "# Spec\n",
    "static/index.html": "<html>orders</html>",
    "static/js/app.js": "console.log('orders')",
}
NEVER_COPIED = {
    "backend.log": "DS_POSTGRES_DEMO__PASSWORD=hunter2\n",
    ".published.json": '{"index.html": {"report_id": "r-1"}}',
    "README.md": "# Orders (A's provenance)\n",
    "discovery.json": "{}",
    "openapi.json": "{}",
    ".anton_state.db": "state",
    "notes.txt": "a file outside the list",
}


@pytest.fixture
def stack(personal_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", BEARER_TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    turn_on_live_apps(monkeypatch)
    yield monkeypatch
    clear_live_app_rows()
    get_app_settings.cache_clear()


def _client() -> TestClient:
    get_app_settings.cache_clear()
    return TestClient(cowork_server.create_app())


def _app(project_path: str, slug: str = SLUG, *, type_: str = "fullstack-stateless-app") -> Path:
    folder = Path(project_path) / ".anton" / "artifacts" / slug
    folder.mkdir(parents=True)
    for name, text in {**APP_FILES, **NEVER_COPIED}.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text, encoding="utf-8")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "id": uuid4().hex,
                "slug": slug,
                "name": "Orders dashboard",
                "description": "Open orders by region",
                "type": type_,
                "primary": "static/index.html",
                "port": 51234,
                "provenance": [{"conversation": str(uuid4())}],
                "datasources": [{"engine": "postgres", "name": "demo"}],
            }
        ),
        encoding="utf-8",
    )
    return folder


@pytest.fixture
def people(stack):
    """A owns the full-stack app SLUG in A's general project; B and C have
    general projects of their own."""
    a_project, a_path = general_project(person_scope(PARTITION_A, SUB_A))
    b_project, b_path = general_project(person_scope(PARTITION_B, SUB_B))
    folder = _app(a_path)
    return {"a_project": a_project, "b_project": b_project, "b_path": Path(b_path), "app": folder}


def _share(client: TestClient, project_id, *, headers=A, **body):
    return client.post(
        "/api/v1/artifacts/live-app-share",
        json={"projectId": str(project_id), "slug": body.pop("slug", SLUG), **body},
        headers=headers,
    )


def _duplicate(client: TestClient, share_id: str, project_id, *, headers=B, confirmed=True):
    return client.post(
        "/api/v1/artifacts/live-app-duplicate",
        json={"shareId": share_id, "projectId": str(project_id), "confirmed": confirmed},
        headers=headers,
    )


def _shared_with(client: TestClient, headers):
    return client.get("/api/v1/artifacts/shared-live-apps", headers=headers)


def _shares_of(client: TestClient, project_id, *, slug: str = SLUG, headers=A):
    return client.get(
        "/api/v1/artifacts/live-app-shares", params={"projectId": str(project_id), "slug": slug}, headers=headers
    )


def _withdraw(client: TestClient, share_id: str, *, headers=A):
    return client.delete(f"/api/v1/artifacts/live-app-share/{share_id}", headers=headers)


NO_EMAIL = "Live apps need an email address on your account. Ask the operator to add one."


def _files(folder: Path) -> set[str]:
    return {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}


# ── sharing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("bob+tag@example.com", True),
        ("name!@example.com", True),
        ("name@example..com", True),
        ("name@.example.com", True),
        ("name@example.com.", True),
        ("a@...", True),
        ("a" * 250 + "@b.c", True),
        ("", False),
        ("@a.b", False),
        ("a@b", False),
        ("a@.b", False),
        ("a@b.", False),
        ("a@b.c@d", False),
        ("a b@c.d", False),
        ("a@b.\u00a0c", False),
        ("a@b.c\n", False),
        ("!" * 254, False),
        ("a@" + "." * 250 + "@", False),
        ("a" * 251 + "@b.c", False),
    ],
)
def test_email_validation_preserves_address_rules(value, valid):
    from cowork.services.live_apps import is_email

    assert is_email(value) is valid


def test_a_share_with_one_address_and_again_is_the_same_share(people):
    client = _client()

    first = _share(client, people["a_project"], email="Bob@Example.com ")
    again = _share(client, people["a_project"], email="bob@example.com")

    assert first.status_code == 200
    assert first.json()["email"] == "bob@example.com"
    assert first.json()["everyone"] is False
    assert again.json()["shareId"] == first.json()["shareId"]


def test_a_share_with_everyone_is_one_per_app(people):
    client = _client()

    first = _share(client, people["a_project"], everyone=True)
    again = _share(client, people["a_project"], everyone=True)

    assert first.status_code == 200
    assert (first.json()["email"], first.json()["everyone"]) == (None, True)
    assert again.json()["shareId"] == first.json()["shareId"]


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"email": "b" * 250 + "@example.com"}, "Enter a valid email address."),
        ({"email": "bob@example.com", "everyone": True}, "Choose one address, or everyone on this stack."),
        ({}, "Choose one address, or everyone on this stack."),
        ({"email": "not-an-address"}, "Enter a valid email address."),
        ({"email": "ALICE@example.com"}, "That's your own address."),
    ],
)
def test_a_share_names_one_other_address_or_everyone(people, body, detail):
    response = _share(_client(), people["a_project"], **body)

    assert response.status_code == 400
    assert response.json()["detail"] == detail


def test_only_a_full_stack_app_is_shared(people):
    a_path = people["app"].parents[2]
    _app(str(a_path), "page-9z8y7x6w", type_="html-app")

    response = _share(_client(), people["a_project"], slug="page-9z8y7x6w", everyone=True)

    assert response.status_code == 400
    assert response.json()["detail"] == "Only full-stack apps can be duplicated this way."


def test_a_share_of_a_missing_app_or_someone_elses_project_is_404(people):
    client = _client()

    missing = _share(client, people["a_project"], slug="nothing-here", everyone=True)
    theirs = _share(client, people["b_project"], everyone=True)

    assert (missing.status_code, missing.json()["detail"]) == (404, "No artifact by that name in this project.")
    assert (theirs.status_code, theirs.json()["detail"]) == (404, "No project by that id.")


def test_without_a_verified_email_sharing_says_why_and_lists_are_empty(people):
    client = _client()
    no_email = headers_for(SUB_A, None)
    share = _share(client, people["a_project"], everyone=True).json()

    response = _share(client, people["a_project"], headers=no_email, everyone=True)
    duplicate = _duplicate(client, share["shareId"], people["b_project"], headers=headers_for(SUB_B, None))
    withdraw = client.delete(f"/api/v1/artifacts/live-app-share/{share['shareId']}", headers=no_email)

    for refused in (response, duplicate, withdraw):
        assert (refused.status_code, refused.json()["detail"]) == (403, NO_EMAIL)
    assert _shared_with(client, headers_for(SUB_B, None)).json() == []
    assert _shares_of(client, people["a_project"], headers=no_email).json() == []


@pytest.mark.parametrize("body", [{"everyone": "yes"}, {"everyone": 1}])
def test_everyone_takes_json_true_only(people, body):
    assert _share(_client(), people["a_project"], **body).status_code == 422


# ── "Shared with you" ────────────────────────────────────────────────────────


def test_the_list_names_the_owner_and_shows_each_app_once(people):
    client = _client()
    by_email = _share(client, people["a_project"], email="bob@example.com").json()
    everyone = _share(client, people["a_project"], everyone=True).json()

    (for_b,) = _shared_with(client, B).json()
    (for_c,) = _shared_with(client, C).json()

    assert for_b["shareId"] == by_email["shareId"] and for_b["everyone"] is False
    assert for_c["shareId"] == everyone["shareId"] and for_c["everyone"] is True
    for entry in (for_b, for_c):
        assert entry["ownerEmail"] == EMAIL_A
        assert entry["slug"] == SLUG
        assert entry["name"] == "Orders dashboard"
        assert entry["description"] == "Open orders by region"
    assert _shared_with(client, A).json() == []


def test_a_share_to_another_address_isnt_listed(people):
    client = _client()
    _share(client, people["a_project"], email="bob@example.com")

    assert _shared_with(client, C).json() == []


def test_an_app_its_owner_deleted_drops_out(people):
    client = _client()
    _share(client, people["a_project"], email="bob@example.com")
    for path in sorted(people["app"].rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    people["app"].rmdir()

    assert _shared_with(client, B).json() == []


# ── duplicating ──────────────────────────────────────────────────────────────


def test_a_duplicate_needs_the_disclaimer_confirmed(people):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()

    response = _duplicate(client, share["shareId"], people["b_project"], confirmed=False)

    assert (response.status_code, response.json()["detail"]) == (400, "Confirm the disclaimer first.")
    assert not (people["b_path"] / ".anton" / "artifacts").exists() or not any(
        (people["b_path"] / ".anton" / "artifacts").iterdir()
    )


def test_the_copy_takes_only_the_fixed_list_and_is_the_recipients(people):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()

    response = _duplicate(client, share["shareId"], people["b_project"])

    assert response.status_code == 200
    card = response.json()
    assert card["receivedFrom"] == {"email": EMAIL_A}
    copy = Path(card["folder"])
    assert copy.parent == people["b_path"] / ".anton" / "artifacts"
    assert copy.name != SLUG
    # anton's store writes metadata.json and README.md; every other file is the list.
    assert _files(copy) == set(APP_FILES) | {"metadata.json", "README.md"}
    for name, text in APP_FILES.items():
        assert (copy / name).read_text(encoding="utf-8") == text
    assert "A's provenance" not in (copy / "README.md").read_text(encoding="utf-8")
    metadata = json.loads((copy / "metadata.json").read_text(encoding="utf-8"))
    original = json.loads((people["app"] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["type"] == "fullstack-stateless-app"
    assert metadata["name"] == "Orders dashboard"
    assert metadata["datasources"] == [{"engine": "postgres", "name": "demo"}]
    # anton's store writes every field of its model: the owner's port and
    # provenance never come along, and the copy's are empty.
    assert ("port", metadata["port"]) == ("port", None)
    assert metadata["provenance"] == []
    assert metadata["id"] != original["id"]


def test_the_recipients_list_marks_only_the_copy(people):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()
    copy = _duplicate(client, share["shareId"], people["b_project"]).json()
    _app(str(people["b_path"]), "own-app-5e6f7a8b")

    cards = client.get("/api/v1/artifacts/", params={"project_id": str(people["b_project"])}, headers=B).json()

    marks = {Path(card["folder"]).name: card["receivedFrom"] for card in cards}
    assert marks == {Path(copy["folder"]).name: {"email": EMAIL_A}, "own-app-5e6f7a8b": None}


def test_cards_carry_no_mark_with_live_apps_off(people, monkeypatch):
    monkeypatch.delenv("ANTON_LIVE_APP_SLOTS")
    client = _client()

    cards = client.get("/api/v1/artifacts/", params={"project_id": str(people["a_project"])}, headers=A).json()

    assert cards and all("receivedFrom" not in card for card in cards)


def test_one_copy_per_person_per_share_until_they_delete_it(people):
    """A second Duplicate of one share answers 409 while the first copy's
    folder is there, and the list names that copy; once the person deletes
    it, they duplicate the app again."""
    client = _client()
    share = _share(client, people["a_project"], everyone=True).json()

    first = _duplicate(client, share["shareId"], people["b_project"]).json()
    again = _duplicate(client, share["shareId"], people["b_project"])
    (listed,) = _shared_with(client, B).json()

    assert (again.status_code, again.json()["detail"]) == (
        409, "You already have a copy of this app. Open it, or delete it to duplicate the app again."
    )
    assert listed["yourCopy"] == {"projectId": str(people["b_project"]), "slug": Path(first["folder"]).name}

    shutil.rmtree(first["folder"])
    second = _duplicate(client, share["shareId"], people["b_project"])
    (relisted,) = _shared_with(client, B).json()

    assert second.status_code == 200 and second.json()["folder"] != first["folder"]
    assert relisted["yourCopy"]["slug"] == Path(second.json()["folder"]).name


def test_anyone_on_the_stack_duplicates_an_app_shared_with_everyone(people):
    """C, named in no share, sees and copies an app A shared with everyone.
    The copy declares A's connections by engine and name only, so it runs
    with C's own saved connections of those names; A's log, which held a
    value, stays behind."""
    c_project, c_path = general_project(person_scope(PARTITION_C, SUB_C))
    client = _client()
    everyone = _share(client, people["a_project"], everyone=True).json()

    (listed,) = _shared_with(client, C).json()
    response = _duplicate(client, everyone["shareId"], c_project, headers=C)

    assert listed["shareId"] == everyone["shareId"] and listed["everyone"] is True
    assert response.status_code == 200
    copy = Path(response.json()["folder"])
    assert copy.parent == Path(c_path) / ".anton" / "artifacts"
    metadata = json.loads((copy / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["datasources"] == [{"engine": "postgres", "name": "demo"}]
    assert not (copy / "backend.log").exists()
    assert all("hunter2" not in path.read_text(encoding="utf-8") for path in copy.rglob("*") if path.is_file())
    assert response.json()["receivedFrom"] == {"email": EMAIL_A}


def test_the_owner_never_lists_or_copies_their_own_share(people):
    client = _client()
    everyone = _share(client, people["a_project"], everyone=True).json()

    assert _shared_with(client, A).json() == []
    response = _duplicate(client, everyone["shareId"], people["a_project"], headers=A)
    assert (response.status_code, response.json()["detail"]) == (404, "You can no longer duplicate this app.")


def test_a_share_with_someone_else_or_into_their_project_is_404(people):
    client = _client()
    share = _share(client, people["a_project"], email="carol@example.com").json()
    for_b = _share(client, people["a_project"], email="bob@example.com").json()

    not_b = _duplicate(client, share["shareId"], people["b_project"])
    into_a = _duplicate(client, for_b["shareId"], people["a_project"])

    assert (not_b.status_code, not_b.json()["detail"]) == (404, "You can no longer duplicate this app.")
    assert (into_a.status_code, into_a.json()["detail"]) == (404, "No project by that id.")


def _replace_with_link(app: Path, name: str, target: Path) -> None:
    (app / name).unlink()
    os.symlink(target, app / name)


@pytest.mark.parametrize(
    ("plant", "path"),
    [
        (lambda app, outside: os.symlink(outside, app / "static" / "leak.txt"), "static/leak.txt is a link"),
        # Only the open's no-follow flag guards the top-level names.
        (lambda app, outside: _replace_with_link(app, "backend.py", outside), "backend.py is a link"),
        (lambda app, outside: _replace_with_link(app, "prd.md", outside), "prd.md is a link"),
        (lambda app, outside: os.link(outside, app / "static" / "twin.txt"), "static/twin.txt has a second name"),
        (lambda app, outside: (app / "backend.py").unlink(), "backend.py is missing"),
    ],
    ids=["static-link", "backend-link", "prd-link", "static-hard-link", "no-backend"],
)
def test_the_copy_refuses_a_link_a_hard_link_and_a_missing_backend(people, tmp_path, plant, path):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()
    outside = tmp_path / "outside.txt"
    outside.write_text("COWORK_SECRET=x\n", encoding="utf-8")
    plant(people["app"], outside)

    response = _duplicate(client, share["shareId"], people["b_project"])

    assert response.status_code == 422
    assert path in response.json()["detail"]
    artifacts = people["b_path"] / ".anton" / "artifacts"
    assert not artifacts.exists() or not any(artifacts.iterdir())


@pytest.mark.parametrize(
    ("cap", "value", "detail"),
    [
        ("MAX_LAUNCH_FILES", 4, "the app has more than 4 files"),
        ("MAX_LAUNCH_TOTAL_BYTES", 64, "the app's files are larger than 64 bytes"),
        ("MAX_LAUNCH_FILE_BYTES", 16, "is larger than 16 bytes"),
    ],
)
def test_the_copy_refuses_an_app_over_antons_launch_caps(stack, people, cap, value, detail):
    """The copy reads anton's own launch caps, so it never holds an app anton
    wouldn't launch."""
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()
    stack.setattr(sys.modules[MODULE], cap, value)

    response = _duplicate(client, share["shareId"], people["b_project"])

    assert response.status_code == 422
    assert detail in response.json()["detail"]
    artifacts = people["b_path"] / ".anton" / "artifacts"
    assert not artifacts.exists() or not any(artifacts.iterdir())


def _launch_sizes() -> tuple[int, int, int]:
    """The fixture app's launch files: their count, their total and the
    largest one, in bytes."""
    launched = {name: text.encode() for name, text in APP_FILES.items() if name not in ("prd.md", "spec.md")}
    return len(launched), sum(map(len, launched.values())), max(map(len, launched.values()))


@pytest.mark.parametrize("cap", ["MAX_LAUNCH_FILES", "MAX_LAUNCH_TOTAL_BYTES", "MAX_LAUNCH_FILE_BYTES"])
def test_an_app_exactly_at_antons_launch_caps_is_copied(stack, people, cap):
    files, total, largest = _launch_sizes()
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()
    stack.setattr(sys.modules[MODULE], cap, {"MAX_LAUNCH_FILES": files, "MAX_LAUNCH_TOTAL_BYTES": total, "MAX_LAUNCH_FILE_BYTES": largest}[cap])

    assert _duplicate(client, share["shareId"], people["b_project"]).status_code == 200


def test_the_copy_refuses_an_app_folder_reached_through_a_link(people, tmp_path):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()
    elsewhere = tmp_path / "elsewhere"
    people["app"].rename(elsewhere)
    os.symlink(elsewhere, people["app"])

    response = _duplicate(client, share["shareId"], people["b_project"])

    assert response.status_code == 422
    assert "reached through a link" in response.json()["detail"]


def test_a_share_refuses_a_linked_metadata_file(people, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"type": "fullstack-stateless-app"}), encoding="utf-8")
    _replace_with_link(people["app"], "metadata.json", outside)

    response = _share(_client(), people["a_project"], everyone=True)

    assert response.status_code == 422
    assert "metadata.json is a link" in response.json()["detail"]


def test_the_copy_refuses_an_owner_project_folder_reached_through_a_link(people, tmp_path):
    """A link at the owner's project folder itself, which a no-follow open
    below it can't see."""
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()
    project = people["app"].parents[2]
    elsewhere = tmp_path / "elsewhere-project"
    project.rename(elsewhere)
    os.symlink(elsewhere, project)

    response = _duplicate(client, share["shareId"], people["b_project"])

    assert response.status_code == 422
    assert "reached through a link" in response.json()["detail"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a folder whatever its mode")
def test_an_unreadable_app_folder_drops_out_of_the_list_and_refuses_a_copy(people):
    """One app folder the server can't read never fails anyone's whole list."""
    client = _client()
    a_path = people["app"].parents[2]
    _app(str(a_path), "other-9f8e7d6c")
    unreadable = _share(client, people["a_project"], everyone=True).json()
    readable = _share(client, people["a_project"], slug="other-9f8e7d6c", everyone=True).json()
    people["app"].chmod(0)
    try:
        listed = _shared_with(client, B)
        duplicate = _duplicate(client, unreadable["shareId"], people["b_project"])
    finally:
        people["app"].chmod(0o755)

    assert listed.status_code == 200
    assert [entry["shareId"] for entry in listed.json()] == [readable["shareId"]]
    assert duplicate.status_code == 422
    assert "can't be read" in duplicate.json()["detail"]


def test_confirmed_takes_json_true_only(people):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()

    response = client.post(
        "/api/v1/artifacts/live-app-duplicate",
        json={"shareId": share["shareId"], "projectId": str(people["b_project"]), "confirmed": "yes"},
        headers=B,
    )

    assert response.status_code == 422


def test_a_failed_received_mark_leaves_no_copy(people, monkeypatch):
    """A copy without its received mark would reach the agent as the person's
    own code, so a database failure after the files are written removes it."""
    from cowork.services import live_apps

    client = TestClient(cowork_server.create_app(), raise_server_exceptions=False)
    share = _share(client, people["a_project"], email="bob@example.com").json()

    def fail(*_args, **_kwargs):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(live_apps, "record_copy", fail)

    response = _duplicate(client, share["shareId"], people["b_project"])

    assert response.status_code == 500
    artifacts = people["b_path"] / ".anton" / "artifacts"
    assert not artifacts.exists() or not any(artifacts.iterdir())


# ── stopping a share ─────────────────────────────────────────────────────────


def test_the_owner_lists_their_shares_of_an_app_newest_first(people):
    from sqlmodel import Session

    from cowork.db.session import get_engine
    from cowork.models.live_app import LiveAppShare

    client = _client()
    by_email = _share(client, people["a_project"], email="bob@example.com").json()
    everyone = _share(client, people["a_project"], everyone=True).json()
    # The server's own clock stamps both within one second; set them apart.
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        for share_id, minutes in ((by_email["shareId"], 2), (everyone["shareId"], 1)):
            row = raw.get(LiveAppShare, UUID(share_id))
            row.created_at = datetime(2026, 10, 21, 12, minutes, tzinfo=timezone.utc)
            raw.add(row)
        raw.commit()

    listed = _shares_of(client, people["a_project"]).json()

    assert [entry["shareId"] for entry in listed] == [by_email["shareId"], everyone["shareId"]]
    assert [(entry["email"], entry["everyone"]) for entry in listed] == [("bob@example.com", False), (None, True)]
    # Someone else's view of A's project lists nothing of A's.
    assert _shares_of(client, people["a_project"], headers=B).json()["detail"] == "No project by that id."


def test_the_share_list_refuses_as_sharing_does(people):
    client = _client()
    _app(str(people["app"].parents[2]), "page-9z8y7x6w", type_="html-app")

    missing = _shares_of(client, people["a_project"], slug="nothing-here")
    page = _shares_of(client, people["a_project"], slug="page-9z8y7x6w")

    assert (missing.status_code, missing.json()["detail"]) == (404, "No artifact by that name in this project.")
    assert (page.status_code, page.json()["detail"]) == (400, "Only full-stack apps can be duplicated this way.")


def test_stopping_a_share_ends_it_for_the_recipients_and_keeps_their_copies(people):
    client = _client()
    c_project, _ = general_project(person_scope(PARTITION_C, SUB_C))
    everyone = _share(client, people["a_project"], everyone=True).json()
    by_email = _share(client, people["a_project"], email="bob@example.com").json()
    copy = _duplicate(client, everyone["shareId"], c_project, headers=C).json()

    assert _withdraw(client, everyone["shareId"]).status_code == 204

    assert _shared_with(client, C).json() == []
    # B's own email share of the same app stays.
    assert [entry["shareId"] for entry in _shared_with(client, B).json()] == [by_email["shareId"]]
    again = _duplicate(client, everyone["shareId"], c_project, headers=C)
    assert (again.status_code, again.json()["detail"]) == (404, "You can no longer duplicate this app.")
    cards = client.get("/api/v1/artifacts/", params={"project_id": str(c_project)}, headers=C).json()
    assert {Path(card["folder"]).name: card["receivedFrom"] for card in cards}[Path(copy["folder"]).name] == {
        "email": EMAIL_A
    }
    assert [entry["shareId"] for entry in _shares_of(client, people["a_project"]).json()] == [by_email["shareId"]]


def test_only_the_owner_stops_a_share(people):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()

    for headers in (B, C):
        refused = _withdraw(client, share["shareId"], headers=headers)
        assert (refused.status_code, refused.json()["detail"]) == (404, "That permission no longer exists.")
    assert _withdraw(client, share["shareId"]).status_code == 204
    gone = _withdraw(client, share["shareId"])
    assert (gone.status_code, gone.json()["detail"]) == (404, "That permission no longer exists.")


# ── moving a conversation ────────────────────────────────────────────────────


def test_a_moved_conversation_takes_its_apps_shares_and_launches_along(people, tmp_path, monkeypatch):
    """Moving a conversation to another project moves its app's folder; its
    share, the received mark and the launch records follow, so recipients
    still list it and its card keeps its state. Its running launch stops as
    moved, since anton finds a running app by its folder."""
    from sqlmodel import Session, select

    from cowork.db.scoped import ScopedSession
    from cowork.db.session import get_engine
    from cowork.models.conversation import Conversation
    from cowork.models.live_app import LiveAppCopy, LiveAppLaunchRecord, LiveAppShare
    from cowork.models.project import Project
    from cowork.models.task_object import TaskObject
    from cowork.services import live_apps
    from cowork.services.live_apps import CoworkLiveAppHost
    from cowork.services.task_objects import KIND_ARTIFACT, TaskObjectService

    async def no_slot(*_args, **_kwargs):
        return None

    # No slot answers here: the launch below only needs its record.
    monkeypatch.setattr(live_apps, "_slot_boot_time", no_slot)

    client = _client()
    share = _share(client, people["a_project"], everyone=True).json()
    engine = get_engine(get_app_settings().database.uri)
    dest_path = tmp_path / "shared" / PARTITION_A / "projects" / "moved"
    dest_path.mkdir(parents=True)
    conversation_id = uuid4()
    with Session(engine) as raw:
        dest = Project(name="moved", path=str(dest_path), org_id=PARTITION_A, created_by=SUB_A)
        raw.add(dest)
        raw.add(
            Conversation(
                id=conversation_id, topic="task", project_id=people["a_project"], org_id=PARTITION_A, created_by=SUB_A
            )
        )
        raw.add(TaskObject(conversation_id=conversation_id, project_id=people["a_project"], kind=KIND_ARTIFACT, ref=SLUG))
        # A's app was itself a copy someone shared with A.
        raw.add(LiveAppCopy(partition_id=PARTITION_A, project_id=people["a_project"], slug=SLUG, owner_email="c@example.com", share_id=uuid4()))
        raw.commit()
        dest_id = dest.id
    launch = sys.modules[MODULE].LiveAppLaunch(
        slot=1, slug=SLUG, artifact_dir=str(people["app"]), address="http://app1.view.localhost:8080/", boot_id="1" * 32
    )
    import asyncio

    asyncio.run(
        CoworkLiveAppHost(person(PARTITION_A, SUB_A, EMAIL_A), project_id=people["a_project"], launch_token="t").launched(
            launch
        )
    )

    with Session(engine) as raw:
        session = ScopedSession(raw, person_scope(PARTITION_A, SUB_A))
        TaskObjectService(session).relocate_to_project(
            session.get(Conversation, conversation_id),
            session.get(Project, people["a_project"]),
            session.get(Project, dest_id),
        )

    assert (dest_path / ".anton" / "artifacts" / SLUG / "backend.py").is_file()
    (listed,) = _shared_with(client, C).json()
    assert listed["shareId"] == share["shareId"]
    with Session(engine) as raw:
        assert raw.exec(select(LiveAppShare.project_id)).all() == [dest_id]
        (record,) = raw.exec(select(LiveAppLaunchRecord)).all()
        assert (record.project_id, record.state, record.stop_reason, record.stop_noticed) == (
            dest_id,
            "stopped",
            "moved",
            False,
        )
        assert raw.exec(select(LiveAppCopy.project_id)).all() == [dest_id]


def test_a_failed_owner_record_removes_the_copy_and_its_received_mark(people, monkeypatch):
    from sqlmodel import Session, select

    from cowork.db.session import get_engine
    from cowork.models.live_app import LiveAppCopy
    from cowork.services import artifact_ownership

    client = TestClient(cowork_server.create_app(), raise_server_exceptions=False)
    share = _share(client, people["a_project"], email="bob@example.com").json()

    def fail(*_args, **_kwargs):
        raise RuntimeError("the owner row failed")

    monkeypatch.setattr(artifact_ownership, "record_artifact_owner", fail)

    assert _duplicate(client, share["shareId"], people["b_project"]).status_code == 500
    artifacts = people["b_path"] / ".anton" / "artifacts"
    assert not artifacts.exists() or not any(artifacts.iterdir())
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        assert raw.exec(select(LiveAppCopy)).all() == []


def test_shared_with_you_lists_newest_first(people):
    from sqlmodel import Session

    from cowork.db.session import get_engine
    from cowork.models.live_app import LiveAppShare

    client = _client()
    _app(str(people["app"].parents[2]), "other-9f8e7d6c")
    older = _share(client, people["a_project"], email="bob@example.com").json()
    newer = _share(client, people["a_project"], slug="other-9f8e7d6c", email="bob@example.com").json()
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        for share_id, minutes in ((older["shareId"], 1), (newer["shareId"], 2)):
            row = raw.get(LiveAppShare, UUID(share_id))
            row.created_at = datetime(2026, 10, 21, 12, minutes, tzinfo=timezone.utc)
            raw.add(row)
        raw.commit()

    assert [entry["shareId"] for entry in _shared_with(client, B).json()] == [newer["shareId"], older["shareId"]]
