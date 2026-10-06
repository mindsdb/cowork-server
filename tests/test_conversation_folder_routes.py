"""Working folder routes, over HTTP, as the desktop app calls them."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings
from cowork.server import create_app
from cowork.services.conversation_folders import MAX_FOLDERS_PER_CONVERSATION

_LOOPBACK_HOST = {"Host": "localhost"}


def _client(base_url: str = "http://127.0.0.1:26866", peer: str = "127.0.0.1") -> TestClient:
    """`base_url` sets `scope["server"]`, the accepted socket's local address;
    `client` sets the peer `require_local` reads."""
    return TestClient(create_app(), base_url=base_url, client=(peer, 54321))


@pytest.fixture()
def local_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.delenv("COWORK_TURN_BACKEND", raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _folder(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / "user" / name
    folder.mkdir(parents=True)
    (folder / "readme.md").write_text("hello")
    return folder


def _chat(client: TestClient) -> str:
    res = client.post("/api/v1/conversations/", json={"topic": "folders"}, headers=_LOOPBACK_HOST)
    assert res.status_code == 201, res.text
    return res.json()["id"]


def _attach(client: TestClient, chat: str, folder: Path):
    return client.post(
        f"/api/v1/conversations/{chat}/folders", json={"path": str(folder)}, headers=_LOOPBACK_HOST
    )


def test_two_folders_attach_list_show_files_and_detach(local_mode, tmp_path):
    client = _client()
    chat = _chat(client)
    docs = _folder(tmp_path, "docs")
    reports = _folder(tmp_path, "reports")
    (reports / "q3").mkdir()
    (reports / "q3" / "summary.csv").write_text("a,b")

    first = _attach(client, chat, docs)
    second = _attach(client, chat, reports)
    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert first.json()["name"] == "docs"
    assert first.json()["available"] is True

    listed = client.get(f"/api/v1/conversations/{chat}/folders", headers=_LOOPBACK_HOST)
    assert listed.status_code == 200, listed.text
    assert [f["path"] for f in listed.json()["folders"]] == [
        str(docs.resolve()),
        str(reports.resolve()),
    ]

    files = client.get(
        f"/api/v1/conversations/{chat}/folders/{second.json()['id']}/files", headers=_LOOPBACK_HOST
    )
    assert files.status_code == 200, files.text
    assert [f["path"] for f in files.json()["files"]] == ["q3/summary.csv", "readme.md"]
    assert "truncated" not in files.json()

    removed = client.delete(
        f"/api/v1/conversations/{chat}/folders/{first.json()['id']}", headers=_LOOPBACK_HOST
    )
    assert removed.status_code == 204, removed.text
    after = client.get(f"/api/v1/conversations/{chat}/folders", headers=_LOOPBACK_HOST)
    assert [f["name"] for f in after.json()["folders"]] == ["reports"]
    assert docs.is_dir(), "detaching must not touch the folder on disk"


def test_a_folder_of_another_chat_cannot_be_removed_or_listed(local_mode, tmp_path):
    client = _client()
    chat_a = _chat(client)
    chat_b = _chat(client)
    folder_of_b = _attach(client, chat_b, _folder(tmp_path, "b-docs")).json()["id"]

    removed = client.delete(
        f"/api/v1/conversations/{chat_a}/folders/{folder_of_b}", headers=_LOOPBACK_HOST
    )
    files = client.get(
        f"/api/v1/conversations/{chat_a}/folders/{folder_of_b}/files", headers=_LOOPBACK_HOST
    )

    assert removed.status_code == 404, removed.text
    assert files.status_code == 404, files.text
    still = client.get(f"/api/v1/conversations/{chat_b}/folders", headers=_LOOPBACK_HOST)
    assert len(still.json()["folders"]) == 1


def test_a_refused_folder_answers_400_with_the_reason(local_mode, tmp_path):
    client = _client()
    chat = _chat(client)

    res = _attach(client, chat, tmp_path / "missing")

    assert res.status_code == 400, res.text
    assert res.json()["detail"] == "Choose an existing local folder"


def test_the_same_folder_twice_is_409(local_mode, tmp_path):
    client = _client()
    chat = _chat(client)
    docs = _folder(tmp_path, "dup")
    assert _attach(client, chat, docs).status_code == 201

    assert _attach(client, chat, docs).status_code == 409


def test_a_seventeenth_folder_is_422(local_mode, tmp_path):
    client = _client()
    chat = _chat(client)
    for i in range(MAX_FOLDERS_PER_CONVERSATION):
        assert _attach(client, chat, _folder(tmp_path, f"f{i}")).status_code == 201

    res = _attach(client, chat, _folder(tmp_path, "extra"))

    assert res.status_code == 422, res.text


def test_a_request_that_did_not_arrive_over_loopback_is_refused(local_mode, tmp_path):
    """The peer address is forgeable behind `--forwarded-allow-ips "*"`; the
    socket the request landed on is not."""
    chat = _chat(_client())
    folder = _folder(tmp_path, "remote")
    attached = _attach(_client(), chat, folder).json()["id"]
    forging = _client(base_url="http://172.17.0.2:9010", peer="127.0.0.1")

    attach = _attach(forging, chat, _folder(tmp_path, "remote-2"))
    files = forging.get(
        f"/api/v1/conversations/{chat}/folders/{attached}/files", headers=_LOOPBACK_HOST
    )

    assert attach.status_code == 403, attach.text
    assert files.status_code == 403, files.text


def test_a_non_loopback_peer_is_refused(local_mode, tmp_path):
    chat = _chat(_client())

    res = _attach(_client(peer="203.0.113.7"), chat, _folder(tmp_path, "peer"))

    assert res.status_code == 403, res.text


def test_org_mode_refuses_every_folder_route_for_an_authenticated_member(
    local_mode, monkeypatch, tmp_path
):
    """Answered by the desktop-only guard, not by a missing login."""
    org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    member = {"X-User-Id": "11111111-1111-4111-8111-111111111111", "X-Organization-Id": org}
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    get_app_settings.cache_clear()
    org_client = _client()
    chat = "00000000-0000-4000-8000-000000000497"
    folder = "00000000-0000-4000-8000-000000000001"
    headers = {**_LOOPBACK_HOST, **member}

    responses = [
        org_client.get(f"/api/v1/conversations/{chat}/folders", headers=headers),
        org_client.post(
            f"/api/v1/conversations/{chat}/folders",
            json={"path": str(_folder(tmp_path, "org"))},
            headers=headers,
        ),
        org_client.delete(f"/api/v1/conversations/{chat}/folders/{folder}", headers=headers),
        org_client.get(f"/api/v1/conversations/{chat}/folders/{folder}/files", headers=headers),
    ]

    assert [r.status_code for r in responses] == [403] * 4, [r.text for r in responses]
