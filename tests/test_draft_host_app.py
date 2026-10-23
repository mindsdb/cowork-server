"""The draft-preview host's app.

The boot-time route walker does not see this app, so these tests are what
pins its surface: one route on the draft hosts only, identity from the token
only, fixed headers.
"""
from __future__ import annotations

import json
import logging
import time

import pytest
from starlette.testclient import TestClient

from cowork.common.structured_logging import make_formatter
from cowork.db.scoped import TenantScope
from cowork.draft_host import app as draft_app_module
from cowork.draft_host.app import DRAFT_ROUTE, EXPIRED_MESSAGE, create_draft_app, request_host
from cowork.draft_host.token import DRAFT_TTL_SECONDS
from cowork.principal import get_principal
from cowork.services.comments_layer import ACTIVATION_PARAM
from cowork.services.preview_html import prepare_preview_html

from cowork.common.settings.app_settings import get_app_settings

from draft_host_support import (
    ARTIFACT,
    DOC,
    DRAFT_CSP,
    DRAFT_HOST,
    FETCH,
    KEYS,
    ORG,
    PROJECT,
    USER,
    configure_draft_env,
    draft_token,
    fake_review,
    forge_identity,
    sign_claims,
    make_artifact,
)


@pytest.fixture
def draft_env(monkeypatch):
    configure_draft_env(monkeypatch)
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def seen(tmp_path, monkeypatch):
    calls: dict = {}
    source, folder, metadata = make_artifact(tmp_path)
    monkeypatch.setattr(
        draft_app_module, "review_artifact_for_request", fake_review(source, folder, metadata, calls)
    )
    return calls


@pytest.fixture
def client(draft_env):
    return TestClient(create_draft_app(), base_url=f"https://{DRAFT_HOST}")


def _assert_draft_headers(response):
    headers = response.headers
    assert headers.get_list("content-security-policy") == [DRAFT_CSP]
    assert "allow-same-origin" not in headers["content-security-policy"]
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["origin-agent-cluster"] == "?1"
    assert headers["access-control-allow-origin"] == "*"
    assert headers["cache-control"] == "private, no-store"
    assert "set-cookie" not in headers
    assert "access-control-allow-credentials" not in headers


def test_serves_the_draft_document_with_the_shim_and_the_draft_headers(client, seen):
    response = client.get(f"/_t/{draft_token()}/index.html", headers=FETCH)

    assert response.status_code == 200
    assert response.text == prepare_preview_html(DOC, comments=False)
    assert response.headers["content-type"].startswith("text/html")
    _assert_draft_headers(response)


def test_the_comment_layer_follows_the_activation_flag(client, seen):
    response = client.get(f"/_t/{draft_token()}/index.html?{ACTIVATION_PARAM}=1", headers=FETCH)

    assert response.status_code == 200
    assert response.text == prepare_preview_html(DOC, comments=True)


def test_relative_assets_load_with_the_same_token(client, seen):
    response = client.get(f"/_t/{draft_token()}/assets/app.js", headers=FETCH)

    assert response.status_code == 200
    assert response.text == "console.log('draft');"
    _assert_draft_headers(response)


def test_head_answers_with_the_same_headers(client, seen):
    response = client.head(f"/_t/{draft_token()}/index.html", headers=FETCH)

    assert response.status_code == 200
    _assert_draft_headers(response)


def test_the_principal_comes_from_the_token_and_forged_headers_are_ignored(client, seen, monkeypatch):
    principals = []
    real_flag = draft_app_module.wants_comment_layer

    def capture(request):
        principals.append(get_principal(request))
        return real_flag(request)

    monkeypatch.setattr(draft_app_module, "wants_comment_layer", capture)
    forged = {
        "X-User-Id": "99999999-9999-4999-8999-999999999999",
        "X-Organization-Id": "88888888-8888-4888-8888-888888888888",
        "X-User-Email": "attacker@example.com",
        "X-User-Roles": "admin",
        "X-Forwarded-User": "attacker",
        "X-Cowork-Expected-Organization-Id": "88888888-8888-4888-8888-888888888888",
    }

    response = client.get(f"/_t/{draft_token()}/index.html", headers={**FETCH, **forged})

    assert response.status_code == 200
    assert seen["scope"] == TenantScope(org_mode=True, org_id=ORG, user_id=USER)
    assert seen["project_ref"] == PROJECT
    assert seen["artifact_id"] == ARTIFACT
    assert [(p.user_id, p.org_id) for p in principals] == [(USER, ORG)]


def test_a_token_without_fetch_metadata_is_refused(client, seen):
    # A link unfurler or a mail link scanner: no Sec-Fetch-* at all.
    response = client.get(f"/_t/{draft_token()}/index.html")

    assert response.status_code == 401
    assert response.text == "Unauthorized"
    assert seen == {}
    _assert_draft_headers(response)


def test_an_expired_link_shows_the_open_again_page(client, seen):
    stale = draft_token(now=int(time.time()) - DRAFT_TTL_SECONDS - 1)

    response = client.get(f"/_t/{stale}/index.html", headers=FETCH)

    assert response.status_code == 401
    assert EXPIRED_MESSAGE in response.text
    assert EXPIRED_MESSAGE == "This preview link has expired. Open the preview again."
    assert seen == {}
    _assert_draft_headers(response)


def test_a_subresource_with_an_expired_token_gets_a_short_401(client, seen):
    stale = draft_token(now=int(time.time()) - DRAFT_TTL_SECONDS - 1)

    response = client.get(
        f"/_t/{stale}/assets/app.js",
        headers={"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "script"},
    )

    assert response.status_code == 401
    assert response.text == "Unauthorized"


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(lambda: draft_token(env="dev"), id="other-env"),
        pytest.param(lambda: draft_token(env="pr-cowork-server-42"), id="pr-env"),
        pytest.param(lambda: draft_token(keys={"k9": "other"}, kid="k9"), id="other-key"),
        pytest.param(lambda: draft_token(sub="not-a-uuid"), id="bad-identity"),
        pytest.param(lambda: forge_identity(draft_token(), sub="99999999-9999-4999-8999-999999999999"), id="tampered"),
        pytest.param(lambda: "v1.garbage.sig", id="garbage"),
    ],
)
def test_foreign_or_broken_tokens_are_refused(client, seen, token):
    response = client.get(f"/_t/{token()}/index.html", headers=FETCH)

    assert response.status_code == 401
    assert seen == {}


def test_a_token_for_another_purpose_is_refused(client, seen):
    published = sign_claims({
        "p": "published", "env": "staging", "sub": USER, "org": ORG, "proj": PROJECT,
        "art": ARTIFACT, "exp": int(time.time()) + 600, "kid": "k1",
    }, KEYS["k1"])

    response = client.get(f"/_t/{published}/index.html", headers=FETCH)

    assert response.status_code == 401
    assert seen == {}


@pytest.mark.parametrize("query", ["download=1", "download=0", "download"])
def test_download_is_refused_on_this_host(client, seen, query):
    response = client.get(f"/_t/{draft_token()}/index.html?{query}", headers=FETCH)

    assert response.status_code == 400
    assert seen == {}


@pytest.mark.parametrize(
    "path",
    ["/", "/index.html", "/api/v1/health/", "/api/v1/artifacts/", "/_t", "/_t/", "/_t/tok", "/x/_t/tok/index.html"],
)
def test_every_other_path_is_404(client, seen, path):
    response = client.get(path, headers=FETCH)

    assert response.status_code == 404
    _assert_draft_headers(response)


def test_only_get_and_head_are_served(client, seen):
    response = client.post(f"/_t/{draft_token()}/index.html", headers=FETCH)

    assert response.status_code == 405
    assert seen == {}


def test_private_entries_stay_private(client, seen):
    response = client.get(f"/_t/{draft_token()}/metadata.json", headers=FETCH)

    assert response.status_code == 404


async def _asgi_get(app, path: str) -> tuple[int, bytes]:
    """Drive the app with a raw path, as a server would pass it on unnormalized."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", DRAFT_HOST.encode())]
        + [(name.lower().encode(), value.encode()) for name, value in FETCH.items()],
        "client": ("203.0.113.9", 50000),
        "server": (DRAFT_HOST, 443),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], body


@pytest.mark.parametrize(
    "rel, expected",
    [
        ("../secret.txt", 400),
        ("assets/../../secret.txt", 400),
        ("..\\secret.txt", 400),
        ("/etc/passwd", 400),
        # Encoded dots reach the app undecoded here: a name no file has.
        ("%2e%2e/secret.txt", 404),
        ("..%2fsecret.txt", 404),
        ("assets/%2e%2e/%2e%2e/secret.txt", 404),
    ],
)
async def test_paths_outside_the_artifact_are_refused(draft_env, seen, rel, expected):
    # make_artifact puts secret.txt exactly where "../secret.txt" leads.
    status, body = await _asgi_get(create_draft_app(), f"/_t/{draft_token()}/{rel}")

    assert status == expected
    assert b"outside the artifact" not in body


def test_nothing_is_served_while_draft_links_are_off(draft_env, seen, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    client = TestClient(create_draft_app(), base_url=f"https://{DRAFT_HOST}")

    response = client.get(f"/_t/{draft_token()}/index.html", headers=FETCH)

    assert response.status_code == 404
    assert seen == {}


def _refusals(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if getattr(record, "event", None) == "request_refused"]


def test_the_refusal_log_names_the_reason_and_never_the_token(client, seen, caplog):
    token = draft_token(env="dev")

    with caplog.at_level("WARNING", logger="cowork.common.request_logging"):
        response = client.get(f"/_t/{token}/index.html", headers=FETCH)

    [record] = _refusals(caplog)
    assert (record.reason, record.status_code, record.route) == ("draft_token_env", 401, DRAFT_ROUTE)
    line = make_formatter().format(record)
    assert (json.loads(line)["reason"], json.loads(line)["route"]) == ("draft_token_env", DRAFT_ROUTE)
    assert token not in line
    assert token not in " ".join(str(value) for value in vars(record).values())
    assert response.headers["x-request-id"]


@pytest.mark.parametrize(
    "path, headers, reason",
    [
        pytest.param(lambda: f"/_t/{draft_token()}/index.html", {}, "draft_no_fetch_metadata", id="no-fetch-metadata"),
        pytest.param(lambda: "/_t/v1.garbage.sig/index.html", FETCH, "draft_token_malformed", id="malformed"),
        pytest.param(
            lambda: f"/_t/{draft_token(sub='not-a-uuid')}/index.html", FETCH, "draft_token_identity", id="identity"
        ),
    ],
)
def test_each_token_refusal_names_its_reason(client, seen, caplog, path, headers, reason):
    with caplog.at_level("WARNING", logger="cowork.common.request_logging"):
        response = client.get(path(), headers=headers)

    assert response.status_code == 401
    assert [record.reason for record in _refusals(caplog)] == [reason]


def test_a_host_outside_the_draft_hosts_is_logged_as_such(draft_env, seen, caplog):
    client = TestClient(create_draft_app(), base_url="https://evil.example")

    with caplog.at_level("WARNING", logger="cowork.common.request_logging"):
        response = client.get(f"/_t/{draft_token()}/index.html", headers=FETCH)

    assert response.status_code == 404
    assert [record.reason for record in _refusals(caplog)] == ["draft_host_unknown"]


def test_switched_off_draft_links_are_logged_as_such(draft_env, seen, monkeypatch, caplog):
    # What the network check sees before the key is in the pod.
    monkeypatch.delenv("COWORK_DRAFT_LINK_KEYS")
    get_app_settings.cache_clear()
    client = TestClient(create_draft_app(), base_url=f"https://{DRAFT_HOST}")

    with caplog.at_level("WARNING", logger="cowork.common.request_logging"):
        response = client.get(f"/_t/{draft_token()}/index.html", headers=FETCH)

    assert response.status_code == 404
    assert response.text == "Not Found"
    assert [record.reason for record in _refusals(caplog)] == ["draft_links_disabled"]


def test_a_failure_is_logged_by_its_event_and_without_the_token(client, monkeypatch, caplog):
    token = draft_token()

    def broken(*_args):
        raise RuntimeError(f"broken while reading /_t/{token}/index.html")

    monkeypatch.setattr(draft_app_module, "review_artifact_for_request", broken)
    with caplog.at_level("WARNING"):
        response = client.get(f"/_t/{token}/index.html", headers=FETCH)

    assert response.status_code == 500
    assert response.headers["x-request-id"]
    events = {getattr(record, "event", None): record for record in caplog.records}
    assert events["draft_host_failed"].name == "cowork.draft_host.app"
    assert events["request_failed"].reason == "internal_server_error"
    for event in ("draft_host_failed", "request_failed"):
        assert token not in make_formatter().format(events[event])


def test_a_path_outside_the_route_is_logged_without_a_route(client, seen, caplog):
    with caplog.at_level("WARNING", logger="cowork.common.request_logging"):
        client.get("/", headers=FETCH)

    [record] = _refusals(caplog)
    assert (record.status_code, record.reason, record.route) == (404, "not_found", None)


def test_a_missing_file_is_404(client, seen):
    response = client.get(f"/_t/{draft_token()}/assets/missing.js", headers=FETCH)

    assert response.status_code == 404
    _assert_draft_headers(response)


def test_a_fullstack_draft_serves_only_its_public_subtree(draft_env, tmp_path, monkeypatch):
    calls: dict = {}
    source, folder, metadata = make_artifact(
        tmp_path,
        metadata={"type": "fullstack-app", "primary": "static/index.html"},
        files={"static/index.html": DOC, "backend.py": "SECRET = 1"},
    )
    monkeypatch.setattr(
        draft_app_module, "review_artifact_for_request", fake_review(source, folder, metadata, calls)
    )
    client = TestClient(create_draft_app(), base_url=f"https://{DRAFT_HOST}")
    token = draft_token()

    assert client.get(f"/_t/{token}/static/index.html", headers=FETCH).status_code == 200
    backend = client.get(f"/_t/{token}/backend.py", headers=FETCH)
    assert backend.status_code == 404
    assert "SECRET" not in backend.text


@pytest.mark.parametrize(
    "host, expected",
    [
        ("draft-staging.artifacts.test", "draft-staging.artifacts.test"),
        ("Draft-Staging.Artifacts.Test:443", "draft-staging.artifacts.test"),
        ("draft.mindshub.co.", "draft.mindshub.co"),
        ("[::1]:9011", "[::1]"),
        (None, ""),
    ],
)
def test_the_host_is_read_from_the_header_lower_cased_without_port(host, expected):
    headers = [] if host is None else [(b"host", host.encode("latin-1"))]

    assert request_host({"type": "http", "headers": headers, "path": "/"}) == expected


@pytest.mark.parametrize("host", ["cowork.staging.mindshub.ai", "draft-dev.artifacts.test", "evil.example"])
def test_a_host_other_than_the_draft_host_is_404(draft_env, seen, host):
    # The draft Ingress pins Host. Anything else did not come through it, and
    # X-Forwarded-Host is never read in its place.
    client = TestClient(create_draft_app(), base_url=f"https://{host}")

    response = client.get(
        f"/_t/{draft_token()}/index.html", headers={**FETCH, "X-Forwarded-Host": DRAFT_HOST}
    )

    assert response.status_code == 404
    assert seen == {}
    _assert_draft_headers(response)
