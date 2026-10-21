"""Sharing a live app means duplicating it: A shares an app of theirs with B's
address or with everyone; B sees it with A's address and duplicates it into a
project of B's own, after confirming the disclaimer. The copy takes a fixed
list of files, the same one anton launches, and never a link, a log, the
publish state or a port."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

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


def _files(folder: Path) -> set[str]:
    return {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}


# ── sharing ──────────────────────────────────────────────────────────────────


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
    assert response.json()["detail"] == "Only full-stack apps can be shared this way."


def test_a_share_of_a_missing_app_or_someone_elses_project_is_404(people):
    client = _client()

    missing = _share(client, people["a_project"], slug="nothing-here", everyone=True)
    theirs = _share(client, people["b_project"], everyone=True)

    assert (missing.status_code, missing.json()["detail"]) == (404, "No artifact by that name in this project.")
    assert (theirs.status_code, theirs.json()["detail"]) == (404, "No project by that id.")


def test_sharing_needs_a_verified_email(people):
    response = _share(_client(), people["a_project"], headers=headers_for(SUB_A, None), everyone=True)

    assert response.status_code == 401


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
    assert metadata.get("port") is None
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


def test_a_later_duplicate_makes_a_new_copy(people):
    client = _client()
    share = _share(client, people["a_project"], everyone=True).json()

    first = _duplicate(client, share["shareId"], people["b_project"]).json()
    second = _duplicate(client, share["shareId"], people["b_project"]).json()

    assert first["folder"] != second["folder"]


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
    assert (response.status_code, response.json()["detail"]) == (404, "This app is no longer shared with you.")


def test_a_share_with_someone_else_or_into_their_project_is_404(people):
    client = _client()
    share = _share(client, people["a_project"], email="carol@example.com").json()
    for_b = _share(client, people["a_project"], email="bob@example.com").json()

    not_b = _duplicate(client, share["shareId"], people["b_project"])
    into_a = _duplicate(client, for_b["shareId"], people["a_project"])

    assert (not_b.status_code, not_b.json()["detail"]) == (404, "This app is no longer shared with you.")
    assert (into_a.status_code, into_a.json()["detail"]) == (404, "No project by that id.")


@pytest.mark.parametrize(
    ("plant", "path"),
    [
        (lambda app: os.symlink("/etc/passwd", app / "static" / "leak.txt"), "static/leak.txt is a link"),
        (lambda app: os.link(app / "requirements.txt", app / "static" / "twin.txt"), "has a second name"),
        (lambda app: (app / "backend.py").unlink(), "backend.py is missing"),
    ],
)
def test_the_copy_refuses_a_link_a_hard_link_and_a_missing_backend(people, plant, path):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()
    plant(people["app"])

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


def test_the_copy_refuses_an_app_folder_reached_through_a_link(people, tmp_path):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()
    elsewhere = tmp_path / "elsewhere"
    people["app"].rename(elsewhere)
    os.symlink(elsewhere, people["app"])

    response = _duplicate(client, share["shareId"], people["b_project"])

    assert response.status_code == 422
    assert "reached through a link" in response.json()["detail"]
