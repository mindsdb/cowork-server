"""SharePoint connections are created and edited only through their own routes.

A create or edit stores the connection only after the key opens every site
(the Graph calls are answered by a fake Microsoft at the HTTP transport). A
create never replaces an existing connection; an edit updates the named one
in place and may keep its stored secret.
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import cowork.server as cowork_server
from cowork.api.v1.endpoints.connectors import sharepoint as sharepoint_ep
from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.principal import personal_partition_id
from cowork.services.connectors import sharepoint as sharepoint_service
from cowork.services.connectors.identity import VAULT_KEEP_SENTINEL
from cowork.services.connectors.layered_vault import (
    SHARED_CONNECTION_REFUSAL,
    STILL_SHARED_CONNECTION_REFUSAL,
)
from cowork.services.connectors.persist import own_vault_for_scope, persist_connection, vault_for_scope

SECRET = "sp-route-secret-77c2"
NEW_SECRET = "sp-route-secret-rotated-31d9"
TENANT = "contoso.onmicrosoft.com"
CLIENT_ID = "11111111-2222-3333-4444-555555555555"
SITE = "https://contoso.sharepoint.com/sites/team"
SITE_2 = "https://contoso.sharepoint.com/sites/ops"
SITE_ID = "contoso.sharepoint.com,aaaa-1111,bbbb-2222"
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
OTHER_CLIENT = "99999999-0000-0000-0000-000000000000"
BODY = {"name": "corp", "tenant_id": TENANT, "client_id": CLIENT_ID, "client_secret": SECRET, "site_urls": SITE}

AUTH_TOKEN = "sharepoint-routes-test-token"
DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
ALICE = TenantScope(personal=True, org_id=personal_partition_id(DEPLOYMENT, SUB_A), user_id=SUB_A, personal_resources=ALL_PERSONAL_RESOURCES)
BOB = TenantScope(personal=True, org_id=personal_partition_id(DEPLOYMENT, SUB_B), user_id=SUB_B, personal_resources=ALL_PERSONAL_RESOURCES)


class Microsoft:
    """Answers the token call and the site and library lookups the check makes."""

    def __init__(self, *, token_codes: list[int] | None = None, site_status: int = 200) -> None:
        self.token_codes = token_codes
        self.site_status = site_status
        self.secrets_seen: list[str] = []
        self.sites_seen: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            self.secrets_seen.append(parse_qs(request.content.decode())["client_secret"][0])
            if self.token_codes:
                return httpx.Response(400, json={"error_codes": self.token_codes})
            return httpx.Response(200, json={"access_token": "route-token", "expires_in": 3600})
        path = request.url.path
        if path.endswith("/drives"):
            return httpx.Response(200, json={"value": [{"id": "b!lib", "name": "Documents"}]})
        self.sites_seen.append(path)
        if self.site_status != 200:
            return httpx.Response(self.site_status, json={})
        return httpx.Response(200, json={"id": SITE_ID})


@pytest.fixture
def microsoft(monkeypatch, sharepoint_stores):
    ms = Microsoft()
    monkeypatch.setattr(
        sharepoint_ep,
        "SharePointClient",
        lambda key: sharepoint_service.SharePointClient(key, transport=httpx.MockTransport(ms.handler)),
    )
    return ms


def _record(name: str = "corp", scope: TenantScope = LOCAL_SCOPE) -> dict | None:
    return own_vault_for_scope(scope).read_record("sharepoint", name)


def _names(scope: TenantScope = LOCAL_SCOPE) -> list[str]:
    return sorted(c["name"] for c in own_vault_for_scope(scope).list_connections() if c["engine"] == "sharepoint")


def _status(call) -> tuple[int, str]:
    with pytest.raises(HTTPException) as raised:
        call()
    return raised.value.status_code, raised.value.detail


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


def test_create_saves_the_connection_after_the_key_opens_every_site(microsoft):
    response = _client().post(
        "/api/v1/connectors/sharepoint/connections",
        json={**BODY, "site_urls": f"https://Contoso.sharepoint.com/Sites/Team/\n{SITE_2}\n{SITE}", "user_label": "Planning"},
    )

    assert response.status_code == 201, response.text
    assert response.json() == {"ok": True, "name": "corp", "user_label": "Planning"}
    fields = _record()["fields"]
    assert {k: fields[k] for k in ("tenant_id", "client_id", "client_secret", "site_urls")} == {
        "tenant_id": TENANT,
        "client_id": CLIENT_ID,
        "client_secret": SECRET,
        "site_urls": f"{SITE}\n{SITE_2}",
    }
    assert "client_secret" in _record()["secure_keys"]
    assert microsoft.secrets_seen == [SECRET]
    assert len(microsoft.sites_seen) == 2


@pytest.mark.parametrize(
    ("ms", "message"),
    [
        (Microsoft(token_codes=[90002]), "Microsoft doesn't know this tenant"),
        (Microsoft(token_codes=[700016]), "This app isn't registered in the tenant"),
        (Microsoft(token_codes=[7000215]), "Microsoft rejected the client secret"),
        (Microsoft(site_status=403), f"The app can't open {SITE}"),
    ],
)
def test_each_failed_check_answers_its_own_message_and_saves_nothing(monkeypatch, sharepoint_stores, ms, message):
    monkeypatch.setattr(
        sharepoint_ep,
        "SharePointClient",
        lambda key: sharepoint_service.SharePointClient(key, transport=httpx.MockTransport(ms.handler)),
    )

    status, detail = _status(lambda: sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE))

    assert status == 422
    assert detail.startswith(message)
    assert SECRET not in detail
    assert _names() == []


@pytest.mark.parametrize("name", ["corp", "CORP", "Corp"])
def test_create_onto_an_existing_name_is_refused_and_leaves_it_unchanged(microsoft, name):
    sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE)
    before = _record()

    status, _ = _status(lambda: sharepoint_ep.create_connection({**BODY, "name": name, "client_secret": NEW_SECRET}, LOCAL_SCOPE))

    assert status == 409
    assert _record() == before
    assert _names() == ["corp"]


def _create_during_the_check(monkeypatch, name: str, credentials_override: dict) -> None:
    """While the key is checked, another writer saves ``name``; the lock is not held meanwhile."""
    real_check = sharepoint_ep._check

    def check_while_someone_else_creates(key, sites):
        assert not sharepoint_ep._CREATE_LOCK.locked()
        real_check(key, sites)
        credentials = {
            "tenant_id": key.tenant_id,
            "client_id": key.client_id,
            "client_secret": key.client_secret,
            "site_urls": "\n".join(sites),
        }
        persist_connection("sharepoint", None, name, {**credentials, **credentials_override}, scope=LOCAL_SCOPE)

    monkeypatch.setattr(sharepoint_ep, "_check", check_while_someone_else_creates)


def test_a_create_that_loses_a_race_for_its_name_leaves_the_winner_alone(monkeypatch, microsoft):
    _create_during_the_check(monkeypatch, "corp", {"client_id": OTHER_CLIENT})

    status, _ = _status(lambda: sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE))

    assert status == 409
    assert _names() == ["corp"]
    assert _record()["fields"]["client_id"] == OTHER_CLIENT


def test_a_create_racing_the_same_key_for_its_name_is_refused_too(monkeypatch, microsoft):
    _create_during_the_check(monkeypatch, "corp", {"client_secret": NEW_SECRET})

    status, _ = _status(lambda: sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE))

    assert status == 409
    assert _names() == ["corp"]
    assert _record()["fields"]["client_secret"] == NEW_SECRET


def test_a_name_taken_during_the_key_check_leaves_an_existing_suffixed_connection_alone(monkeypatch, microsoft):
    sharepoint_ep.create_connection({**BODY, "name": "corp-2"}, LOCAL_SCOPE)
    before = _record("corp-2")
    _create_during_the_check(monkeypatch, "corp", {"client_id": OTHER_CLIENT})

    status, _ = _status(lambda: sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE))

    assert status == 409
    assert sorted(_names()) == ["corp", "corp-2"]
    assert _record("corp-2") == before


def _create_during_the_save(monkeypatch, name: str, credentials_override: dict) -> None:
    """Another writer saves ``name`` after the locked recheck, just before this create's save."""
    real_persist = sharepoint_ep.persist_connection

    def persist_after_another_writer(engine, method, created, credentials, **kwargs):
        persist_connection("sharepoint", None, name, {**credentials, **credentials_override}, scope=LOCAL_SCOPE)
        return real_persist(engine, method, created, credentials, **kwargs)

    monkeypatch.setattr(sharepoint_ep, "persist_connection", persist_after_another_writer)


def test_a_name_taken_during_the_save_is_refused_and_nothing_is_saved_under_a_suffix(monkeypatch, microsoft):
    _create_during_the_save(monkeypatch, "corp", {"client_id": OTHER_CLIENT})

    status, _ = _status(lambda: sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE))

    assert status == 409
    assert _names() == ["corp"]
    assert _record()["fields"]["client_id"] == OTHER_CLIENT


def test_a_name_taken_during_the_save_never_overwrites_a_same_account_suffixed_connection(monkeypatch, microsoft):
    sharepoint_ep.create_connection({**BODY, "name": "corp-2", "user_label": "Old label"}, LOCAL_SCOPE)
    before = _record("corp-2")
    _create_during_the_save(monkeypatch, "corp", {"client_id": OTHER_CLIENT})

    status, _ = _status(
        lambda: sharepoint_ep.create_connection({**BODY, "client_secret": NEW_SECRET, "user_label": "New label"}, LOCAL_SCOPE)
    )

    assert status == 409
    assert sorted(_names()) == ["corp", "corp-2"]
    assert _record("corp-2") == before


def test_a_name_taken_during_the_save_never_creates_an_env_prefix_twin(monkeypatch, microsoft):
    sharepoint_ep.create_connection({**BODY, "name": "corp_2"}, LOCAL_SCOPE)
    _create_during_the_save(monkeypatch, "corp", {"client_id": OTHER_CLIENT})

    status, _ = _status(lambda: sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE))

    assert status == 409
    assert sorted(_names()) == ["corp", "corp_2"]


def test_a_name_sharing_an_env_prefix_taken_during_the_check_is_refused(monkeypatch, microsoft):
    _create_during_the_check(monkeypatch, "corp_one", {"client_id": OTHER_CLIENT})

    status, _ = _status(lambda: sharepoint_ep.create_connection({**BODY, "name": "corp-one"}, LOCAL_SCOPE))

    assert status == 409
    assert _names() == ["corp_one"]


@pytest.mark.parametrize(
    "name", ["corp one", "corp_", "_corp", "__", "-", "a__b", "a-_b", "a--b", "a" * 65, "corp/x", "naïve"]
)
def test_names_the_vault_would_rewrite_are_refused(microsoft, name):
    status, _ = _status(lambda: sharepoint_ep.create_connection({**BODY, "name": name}, LOCAL_SCOPE))

    assert status == 422
    assert _names() == []


@pytest.mark.parametrize("name", ["corp_one", "CORP-ONE", "Corp_One"])
def test_a_name_with_the_same_env_prefix_as_an_existing_one_is_taken(microsoft, name):
    sharepoint_ep.create_connection({**BODY, "name": "corp-one"}, LOCAL_SCOPE)

    status, _ = _status(lambda: sharepoint_ep.create_connection({**BODY, "name": name}, LOCAL_SCOPE))

    assert status == 409
    assert _names() == ["corp-one"]


def test_a_name_of_64_characters_is_accepted(microsoft):
    name = "a" * 64

    sharepoint_ep.create_connection({**BODY, "name": name}, LOCAL_SCOPE)

    assert _names() == [name]


def test_create_refuses_a_kept_secret(microsoft):
    status, _ = _status(lambda: sharepoint_ep.create_connection({**BODY, "client_secret": VAULT_KEEP_SENTINEL}, LOCAL_SCOPE))

    assert status == 422
    assert microsoft.secrets_seen == []


@pytest.mark.parametrize(
    "body",
    [
        {**BODY, "refresh_token": "x"},
        {**BODY, "_created_by": "someone"},
        {**BODY, "auth_type": "oauth"},
    ],
)
def test_fields_outside_the_connection_are_refused(microsoft, body):
    status, detail = _status(lambda: sharepoint_ep.create_connection(body, LOCAL_SCOPE))

    assert status == 422
    assert detail.startswith("Unknown fields")
    assert _names() == []


@pytest.mark.parametrize(
    "body",
    [
        {k: v for k, v in BODY.items() if k != "name"},
        {**BODY, "client_secret": 12345},
        {**BODY, "name": ["corp"]},
        [BODY],
    ],
)
def test_a_malformed_body_never_echoes_the_secret(microsoft, body):
    secret_body = json.loads(json.dumps(body).replace("12345", f'"{SECRET}"')) if isinstance(body, dict) else body
    response = _client().post("/api/v1/connectors/sharepoint/connections", json=body)
    raw_response = _client().post(
        "/api/v1/connectors/sharepoint/connections",
        content=json.dumps(secret_body)[:-1],
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert raw_response.status_code == 422
    assert SECRET not in response.text
    assert SECRET not in raw_response.text
    assert _names() == []


def test_edit_keeping_the_secret_and_adding_a_site_updates_the_same_record(microsoft):
    sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE)
    microsoft.secrets_seen.clear()

    response = _client().put(
        "/api/v1/connectors/sharepoint/connections/corp",
        json={"tenant_id": TENANT, "client_id": CLIENT_ID, "client_secret": VAULT_KEEP_SENTINEL, "site_urls": f"{SITE}\n{SITE_2}"},
    )

    assert response.status_code == 200, response.text
    assert _names() == ["corp"]
    fields = _record()["fields"]
    assert fields["site_urls"] == f"{SITE}\n{SITE_2}"
    assert fields["client_secret"] == SECRET
    assert microsoft.secrets_seen == [SECRET]


def test_edit_retyping_the_secret_updates_in_place(microsoft):
    sharepoint_ep.create_connection({**BODY, "user_label": "Planning"}, LOCAL_SCOPE)

    sharepoint_ep.edit_connection(
        "corp", {"tenant_id": TENANT, "client_id": CLIENT_ID, "client_secret": NEW_SECRET, "site_urls": SITE}, LOCAL_SCOPE
    )

    assert _names() == ["corp"]
    assert _record()["fields"]["client_secret"] == NEW_SECRET
    assert _record()["fields"]["_user_label"] == "Planning"


def test_a_failed_edit_check_keeps_the_stored_connection(monkeypatch, microsoft):
    sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE)
    before = _record()
    microsoft.token_codes = [7000215]

    status, _ = _status(
        lambda: sharepoint_ep.edit_connection(
            "corp", {"tenant_id": TENANT, "client_id": CLIENT_ID, "client_secret": NEW_SECRET, "site_urls": SITE}, LOCAL_SCOPE
        )
    )

    assert status == 422
    assert _record() == before


@pytest.mark.parametrize("name", ["missing", "corp_", "CORP", "corp one"])
def test_edit_of_a_name_that_is_not_exactly_stored_is_not_found(microsoft, name):
    sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE)
    before = _record()

    status, _ = _status(
        lambda: sharepoint_ep.edit_connection(
            name, {"tenant_id": TENANT, "client_id": CLIENT_ID, "client_secret": NEW_SECRET, "site_urls": SITE}, LOCAL_SCOPE
        )
    )

    assert status == 404
    assert _record() == before
    assert _names() == ["corp"]


@pytest.mark.parametrize("field", ["tenant_id", "client_id", "site_urls"])
def test_edit_keeps_only_the_secret(microsoft, field):
    sharepoint_ep.create_connection(dict(BODY), LOCAL_SCOPE)
    body = {"tenant_id": TENANT, "client_id": CLIENT_ID, "client_secret": SECRET, "site_urls": SITE, field: VAULT_KEEP_SENTINEL}

    status, _ = _status(lambda: sharepoint_ep.edit_connection("corp", body, LOCAL_SCOPE))

    assert status == 422


def test_a_connection_shared_with_the_caller_cannot_be_edited(personal_deployment, managed_dir, microsoft):
    get_app_settings.cache_clear()
    sharepoint_ep.create_connection(dict(BODY), ALICE)
    managed_dir.shared_dir.mkdir(exist_ok=True)
    managed_dir.shared_connections_file.write_text(
        json.dumps([{"user_id": SUB_A, "partition_id": ALICE.org_id, "engine": "sharepoint", "name": "corp"}])
    )
    before = _record(scope=ALICE)

    status, detail = _status(
        lambda: sharepoint_ep.edit_connection(
            "corp", {"tenant_id": TENANT, "client_id": CLIENT_ID, "client_secret": NEW_SECRET, "site_urls": SITE}, BOB
        )
    )

    assert (status, detail) == (403, SHARED_CONNECTION_REFUSAL)
    assert _record(scope=ALICE) == before
    assert vault_for_scope(BOB).load("sharepoint", "corp")["client_secret"] == SECRET


def test_a_name_still_shared_from_a_removed_connection_is_refused_not_a_500(
    monkeypatch, personal_deployment, managed_dir, microsoft
):
    get_app_settings.cache_clear()
    sharepoint_ep.create_connection(dict(BODY), ALICE)
    managed_dir.shared_dir.mkdir(exist_ok=True)
    managed_dir.shared_connections_file.write_text(
        json.dumps([{"user_id": SUB_A, "partition_id": ALICE.org_id, "engine": "sharepoint", "name": "corp"}])
    )
    own_vault_for_scope(ALICE).delete("sharepoint", "corp")
    monkeypatch.setenv("COWORK_AUTH_TOKEN", AUTH_TOKEN)
    get_app_settings.cache_clear()
    client = TestClient(cowork_server.create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))

    response = client.post(
        "/api/v1/connectors/sharepoint/connections",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}", "X-User-Id": SUB_A, "X-Organization-Id": DEPLOYMENT},
        json=BODY,
    )

    assert response.status_code == 403
    assert response.json() == {"detail": STILL_SHARED_CONNECTION_REFUSAL}
    assert _names(ALICE) == []


def test_a_create_records_the_person_who_made_it(personal_deployment, microsoft):
    get_app_settings.cache_clear()

    sharepoint_ep.create_connection(dict(BODY), ALICE)

    assert _record(scope=ALICE)["fields"]["_created_by"] == SUB_A
    assert vault_for_scope(BOB).load("sharepoint", "corp") is None


def test_org_mode_refuses_the_routes(monkeypatch, microsoft):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    client = _client()

    created = client.post("/api/v1/connectors/sharepoint/connections", json=BODY)
    edited = client.put("/api/v1/connectors/sharepoint/connections/corp", json=BODY)

    assert created.status_code == 403
    assert edited.status_code == 403
    assert microsoft.secrets_seen == []
