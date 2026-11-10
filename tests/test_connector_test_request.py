"""The server runs a custom connector's test request and judges the answer.

Every host is judged before the first request, every request dials only the
addresses that were judged, and the answer must look like data before the
member is shown a masked preview. These run the real runner against an
``httpx.MockTransport`` and a stub resolver.
"""
from __future__ import annotations

import base64
import copy
import json
import re
from urllib.parse import quote, quote_plus

import httpx
import pytest

from cowork.services.connectors import connector_test_run
from cowork.services.connectors.connector_test_run import ConnectorTestError, run_test_request

SECRET = "s3cr3t-Value/with+chars@x"
CLIENT_ID = "planner-client"
ISSUED = "tok_issued_0123456789abcdef"
ROWS = [{"sku": "A-1", "qty": 4, "price": 12.5}, {"sku": "B-2", "qty": 1, "price": 3.0}]

CLIENT_REQUEST = {
    "token": {
        "method": "POST", "url": "https://auth.example.com/oauth2/token",
        "form": {"grant_type": "client_credentials"},
        "basic_auth": ["{{client_id}}", "{{client_secret}}"],
        "token_field": "access_token",
    },
    "data": {
        "method": "POST", "url": "https://api.example.com/integration/V1/query",
        "headers": {"Authorization": "Bearer {{token}}"},
        "json_body": {"query": "select * limit 5"},
    },
}
VALUES = {"client_id": CLIENT_ID, "client_secret": SECRET}
SECRETS = ["client_secret"]
PUBLIC = {"auth.example.com": "93.184.216.10", "api.example.com": "93.184.216.20"}


class Api:
    """A mock API: answers by path and records every request it got."""

    def __init__(self, data: object = None, *, data_status: int = 200, raw: bytes | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.data = {"rows": ROWS} if data is None else data
        self.data_status = data_status
        self.raw = raw

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": ISSUED, "token_type": "bearer", "expires_in": 3600})
        if self.raw is not None:
            return httpx.Response(self.data_status, content=self.raw)
        return httpx.Response(self.data_status, json=self.data)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def resolver(table: dict[str, str | list[str]]):
    """A getaddrinfo stub answering from ``table``; an unknown name fails."""
    asked: list[str] = []

    def resolve(host, _port, type=None):
        asked.append(host)
        if host not in table:
            raise OSError("unknown host")
        answers = table[host] if isinstance(table[host], list) else [table[host]]
        return [(None, None, None, "", (a, 443)) for a in answers]

    resolve.asked = asked
    return resolve


@pytest.fixture(autouse=True)
def no_allowlist(monkeypatch):
    monkeypatch.delenv(connector_test_run.ALLOWLIST_ENV, raising=False)
    connector_test_run.private_allowlist.cache_clear()
    yield
    connector_test_run.private_allowlist.cache_clear()


def allow(monkeypatch, value: str) -> None:
    monkeypatch.setenv(connector_test_run.ALLOWLIST_ENV, value)
    connector_test_run.private_allowlist.cache_clear()


async def run(request=CLIENT_REQUEST, values=VALUES, *, api=None, table=PUBLIC, **kwargs):
    api = api or Api()
    outcome = await run_test_request(
        request, values, secret_names=SECRETS, resolver=resolver(table), transport=api.transport, **kwargs,
    )
    return outcome, api


def with_url(url: str, request=CLIENT_REQUEST) -> dict:
    out = copy.deepcopy(request)
    out["data"]["url"] = url
    return out


async def refused(words: str, *args, **kwargs) -> Api:
    api = kwargs.pop("api", None) or Api()
    with pytest.raises(ConnectorTestError) as exc:
        await run(*args, api=api, **kwargs)
    assert words in str(exc.value)
    for value in (SECRET, ISSUED, CLIENT_ID):
        assert value not in str(exc.value)
    return api


class TestAClientCredentialsApiQueriedByPost:
    @pytest.mark.asyncio
    async def test_it_fetches_a_token_then_rows_and_previews_them(self):
        outcome, api = await run()

        token_call, data_call = api.requests
        assert token_call.headers["authorization"] == "Basic " + base64.b64encode(f"{CLIENT_ID}:{SECRET}".encode()).decode()
        assert data_call.headers["authorization"] == f"Bearer {ISSUED}"
        assert json.loads(data_call.content) == {"query": "select * limit 5"}
        preview = outcome.preview
        assert (preview.status, preview.item_count, preview.keys) == (200, 2, ["rows"])
        assert (preview.token_host, preview.data_host) == ("auth.example.com", "api.example.com")
        assert json.loads(preview.first_item) == ROWS[0]

    @pytest.mark.asyncio
    async def test_each_request_dials_the_vetted_address_under_its_own_name(self):
        _, api = await run()

        assert [r.url.host for r in api.requests] == ["93.184.216.10", "93.184.216.20"]
        assert [r.headers["host"] for r in api.requests] == ["auth.example.com", "api.example.com"]
        assert [r.extensions["sni_hostname"] for r in api.requests] == ["auth.example.com", "api.example.com"]

    @pytest.mark.asyncio
    async def test_the_client_ignores_environment_proxies_and_redirects(self, monkeypatch):
        """httpx skips environment proxies whenever a transport is injected, so
        a request cannot show this; the client's own settings are the check."""
        built = []
        real = connector_test_run.httpx.AsyncClient

        def client(**kwargs):
            built.append(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(connector_test_run.httpx, "AsyncClient", client)
        await run()
        assert [(k["trust_env"], k["follow_redirects"]) for k in built] == [(False, False), (False, False)]

    @pytest.mark.asyncio
    async def test_a_redirect_is_not_followed(self):
        class Redirect(Api):
            def __call__(self, request):
                self.requests.append(request)
                if request.url.path.endswith("/token"):
                    return httpx.Response(200, json={"access_token": ISSUED})
                return httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})

        api = await refused("HTTP 302", api=Redirect())
        assert len(api.requests) == 2


class TestEveryHostIsJudgedBeforeAnyRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("address", [
        "169.254.169.254", "fe80::1", "fd00:ec2::254", "100.100.100.200", "192.0.0.192", "0.0.0.0", "::",
        "198.18.0.1", "240.0.0.1", "255.255.255.255", "224.0.0.251", "fec0::1", "64:ff9b::a9fe:a9fe",
    ])
    async def test_an_address_no_test_may_reach_is_refused_even_when_allowlisted(self, monkeypatch, address):
        allow(monkeypatch, f"api.example.com,{address}/128" if ":" in address else f"api.example.com,{address}/32")
        api = await refused("no test may reach", table={**PUBLIC, "api.example.com": address})
        assert api.requests == []

    @pytest.mark.asyncio
    async def test_one_refused_answer_refuses_the_whole_host(self):
        api = await refused("no test may reach", table={**PUBLIC, "api.example.com": ["93.184.216.20", "169.254.169.254"]})
        assert api.requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("address", ["127.0.0.1", "0.0.0.0", "::ffff:127.0.0.1", "::1"])
    async def test_this_server_is_never_reached(self, monkeypatch, address):
        allow(monkeypatch, "cowork.local")
        port = connector_test_run.get_app_settings().port
        url = f"https://cowork.local:{port}/api/v1/health"
        api = await refused("Cowork itself", with_url(url), table={**PUBLIC, "cowork.local": address})
        assert api.requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("address", ["10.0.0.5", "192.168.1.10", "172.16.0.9", "100.64.0.1", "fd12::5", "64:ff9b::a00:5"])
    async def test_a_private_host_asks_before_anything_is_sent(self, address):
        outcome, api = await run(table={**PUBLIC, "api.example.com": address})

        assert api.requests == []
        assert outcome.preview is None
        host = outcome.private_host
        assert (host.step, host.host, host.method, host.path, host.addresses) == (
            "data", "api.example.com", "POST", "/integration/V1/query", (address,),
        )

    @pytest.mark.asyncio
    async def test_the_question_names_every_address_and_whether_it_is_this_computer(self):
        outcome, _ = await run(table={**PUBLIC, "api.example.com": ["127.0.0.1", "::1"]})
        shown = outcome.private_host.to_dict()
        assert shown["addresses"] == ["127.0.0.1", "::1"] and shown["this_computer"] is True

    @pytest.mark.asyncio
    async def test_an_allowed_host_is_dialled_only_at_its_approved_addresses(self):
        table = {**PUBLIC, "api.example.com": "10.0.0.5"}
        first, _ = await run(table=table)
        approved = {"api.example.com": first.private_host.addresses}

        # The name now answers with another address; the approved one is dialled.
        outcome, api = await run(table={**PUBLIC, "api.example.com": "10.9.9.9"}, approved=approved)
        assert outcome.preview is not None
        assert api.requests[-1].url.host == "10.0.0.5"

    @pytest.mark.asyncio
    async def test_an_operator_listed_host_or_range_needs_no_question(self, monkeypatch):
        for entry in ("api.example.com", "10.0.0.0/8", "API.Example.com."):
            allow(monkeypatch, entry)
            outcome, api = await run(table={**PUBLIC, "api.example.com": "10.0.0.5"})
            assert outcome.preview is not None and api.requests[-1].url.host == "10.0.0.5"

    @pytest.mark.asyncio
    async def test_the_allowlist_comes_only_from_the_process_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))
        (tmp_path / ".env").write_text(f"{connector_test_run.ALLOWLIST_ENV}=10.0.0.0/8\n")
        connector_test_run.private_allowlist.cache_clear()
        outcome, _ = await run(table={**PUBLIC, "api.example.com": "10.0.0.5"})
        assert outcome.private_host is not None

    @pytest.mark.asyncio
    async def test_the_host_vetted_is_the_one_dialled_and_shown(self):
        table = resolver({**PUBLIC, "api.example.com": "93.184.216.20"})
        api = Api()
        outcome = await run_test_request(
            with_url("https://api.example。com/integration/V1/query"), VALUES,
            secret_names=SECRETS, resolver=table, transport=api.transport,
        )
        assert table.asked[-1] == outcome.preview.data_host == api.requests[-1].headers["host"] == "api.example.com"

    @pytest.mark.asyncio
    async def test_http_goes_only_to_this_computer_once_it_is_allowed(self, monkeypatch):
        await refused("use https", with_url("http://api.example.com/q"))
        outcome, _ = await run(with_url("http://api.example.com/q"), table={**PUBLIC, "api.example.com": "127.0.0.1"})
        assert outcome.private_host is not None
        allow(monkeypatch, "127.0.0.1/32")
        outcome, api = await run(with_url("http://api.example.com/q"), table={**PUBLIC, "api.example.com": "127.0.0.1"})
        assert outcome.preview is not None and api.requests[-1].url.scheme == "http"
        allow(monkeypatch, "10.0.0.0/8")
        await refused("use https", with_url("http://api.example.com/q"), table={**PUBLIC, "api.example.com": "10.0.0.5"})


class TestFillingIn:
    @pytest.mark.asyncio
    async def test_an_empty_field_stops_the_test_before_any_request(self):
        api = await refused("needs client_secret", values={"client_id": CLIENT_ID, "client_secret": ""})
        assert api.requests == []

    @pytest.mark.asyncio
    async def test_a_value_cannot_add_a_header_line(self):
        request = copy.deepcopy(CLIENT_REQUEST)
        request["data"]["headers"]["X-Account"] = "{{client_id}}"
        api = await refused("line break", request, values={**VALUES, "client_id": "a\r\nHost: evil"})
        assert api.requests == []

    @pytest.mark.asyncio
    async def test_a_value_is_percent_encoded_in_the_path(self):
        request = with_url("https://api.example.com/v1/{{client_id}}?c={{client_id}}")
        _, api = await run(request, values={**VALUES, "client_id": "../a b/?#"})
        assert api.requests[-1].url.raw_path == b"/v1/..%2Fa%20b%2F%3F%23?c=..%2Fa%20b%2F%3F%23"

    @pytest.mark.asyncio
    async def test_a_token_answer_without_the_named_field_fails(self):
        request = copy.deepcopy(CLIENT_REQUEST)
        request["token"]["token_field"] = "id"
        await refused("has no id", request)


class TestTheAnswerMustLookLikeData:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [
        None, "", [], {}, [{}],
        {"error": "invalid_grant"},
        {"data": None, "errors": [{"message": "unauthorized"}]},
    ], ids=["null", "empty string", "empty list", "empty object", "list of empty", "error", "graphql errors"])
    async def test_an_answer_with_no_data_is_refused(self, body):
        await refused("data step", api=Api(raw=json.dumps(body).encode()))

    @pytest.mark.asyncio
    async def test_whitespace_is_refused(self):
        await refused("empty answer", api=Api(raw=b"  \n"))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [
        {"access_token": "x", "expires_in": 300, "refresh_expires_in": 1800, "not-before-policy": 0},
        {"token_type": "Bearer", "expires_in": 3599, "ext_expires_in": 3599, "expires_on": "1", "resource": "r"},
        {"access_token": "x", "instance_url": "https://acme.my.salesforce.com", "signature": "s"},
        {"data": {"access_token": "x", "token_type": "bearer"}},
    ], ids=["keycloak", "azure v1", "salesforce", "nested"])
    async def test_a_token_is_not_data(self, body):
        await refused("a token, not data", api=Api(data=body))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403, 404, 500])
    async def test_a_failing_status_is_refused(self, status):
        await refused(f"HTTP {status}", api=Api(data_status=status))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [None, False, "", 0])
    async def test_a_falsy_error_beside_data_passes(self, error):
        outcome, _ = await run(api=Api(data={"error": error, "rows": ROWS}))
        assert outcome.preview.item_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", [
        "https://auth.example.com/oauth2/token",
        "https://auth.example.com/oauth2/token/",
        "https://auth.example.com//oauth2/token",
        "https://auth.example.com/oauth2/./token",
        "https://auth.example.com/x/../oauth2/token",
        "https://auth.example.com:443/oauth2/token",
        "https://AUTH.example.com./OAUTH2/TOKEN",
        "https://auth.example.com/oauth2/%74oken",
    ])
    async def test_asking_the_token_url_again_is_not_data(self, url):
        await refused("token URL again", with_url(url))


class TestCredentialsNeverComeBack:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("echo", [
        SECRET,
        json.dumps(SECRET)[1:-1],
        SECRET.replace("/", "\\/").replace("@", "\\u0040"),
        quote(SECRET), quote(SECRET, safe=""), quote_plus(SECRET), re.sub(r"%[0-9A-F]{2}", lambda m: m.group().lower(), quote(SECRET, safe="")),
        SECRET.encode().hex(),
        base64.b64encode(SECRET.encode()).decode(),
        base64.b64encode(b"x" + SECRET.encode()).decode(),
        base64.b64encode(b"xy" + SECRET.encode()).decode(),
        base64.b64encode(f"{CLIENT_ID}:{SECRET}".encode()).decode(),
        ISSUED,
    ], ids=["raw", "json", "escaped", "quote", "quote all", "quote plus", "lower hex", "hex", "b64", "b64+1", "b64+2",
            "basic", "issued token"])
    async def test_an_echoed_secret_refuses_the_answer(self, echo):
        raw = ('{"rows": [{"sku": "A-1", "note": "' + echo + '"}]}').encode()
        await refused("repeats a credential", api=Api(raw=raw))

    @pytest.mark.asyncio
    async def test_a_secret_used_as_a_key_refuses_the_answer(self):
        await refused("repeats a credential", api=Api(data={"rows": [{SECRET: 1}]}))

    @pytest.mark.asyncio
    async def test_a_secret_field_is_guarded_whatever_the_form_calls_it(self):
        request = copy.deepcopy(CLIENT_REQUEST)
        request["data"]["headers"] = {"Authorization": "Bearer {{api_key}}"}
        del request["token"]
        api = Api(data={"rows": [{"api_key": "sk_live_0123456789abcdef", "account": "acme"}]})
        with pytest.raises(ConnectorTestError, match="repeats a credential"):
            await run_test_request(
                request, {"api_key": "sk_live_0123456789abcdef"}, secret_names=[],
                resolver=resolver(PUBLIC), transport=api.transport,
            )

    @pytest.mark.asyncio
    async def test_a_non_secret_value_echoed_in_real_data_is_masked_not_refused(self):
        request = copy.deepcopy(CLIENT_REQUEST)
        request["data"]["headers"]["X-Tenant"] = "{{client_id}}"
        outcome, _ = await run(request, api=Api(data={"rows": [{"tenant": CLIENT_ID, "sku": "A-1"}]}))
        assert json.loads(outcome.preview.first_item) == {"tenant": "***", "sku": "A-1"}
        assert outcome.preview.masked_count == 1

    @pytest.mark.asyncio
    async def test_a_short_value_masks_nothing(self):
        outcome, _ = await run(values={**VALUES, "client_id": "us"}, api=Api(data={"rows": [{"status": "paid", "cur": "usd"}]}))
        assert json.loads(outcome.preview.first_item) == {"status": "paid", "cur": "usd"}

    @pytest.mark.asyncio
    async def test_masking_runs_before_the_cut_and_keeps_keys_distinct(self):
        row = {"pad": "x" * 600, "who": f"by {CLIENT_ID}", f"{CLIENT_ID}-a": 1, f"{CLIENT_ID}-b": 2}
        outcome, _ = await run(api=Api(data=[row]))
        shown = outcome.preview.first_item
        assert CLIENT_ID not in shown and len(shown) == 500
        assert outcome.preview.masked_count == 3

    @pytest.mark.asyncio
    async def test_masked_top_level_keys_stay_distinct(self):
        outcome, _ = await run(api=Api(data={f"{CLIENT_ID}-a": [1], f"{CLIENT_ID}-b": [2], "rows": ROWS}))
        assert outcome.preview.keys == ["***1", "***2", "rows"]


class TestLargeAnswers:
    @pytest.mark.asyncio
    async def test_reading_stops_at_the_cap(self):
        read = []

        async def chunks():
            for _ in range(2000):
                read.append(1)
                yield b"x" * 1024

        class Streaming(Api):
            def __call__(self, request):
                self.requests.append(request)
                if request.url.path.endswith("/token"):
                    return super().__call__(request)
                return httpx.Response(200, content=chunks())

        outcome, _ = await run(api=Streaming())
        assert outcome.preview.over_cap is True
        assert len(read) <= 257

    @pytest.mark.asyncio
    async def test_an_answer_over_the_cap_is_shown_as_too_large(self):
        outcome, _ = await run(api=Api(raw=b"[" + b'{"a": 1},' * 40000 + b'{"a": 1}]'))
        preview = outcome.preview
        assert preview.over_cap is True and preview.size == 256 * 1024
        assert (preview.keys, preview.first_item, preview.item_count) == ([], None, None)

    @pytest.mark.asyncio
    async def test_an_answer_over_the_cap_is_still_checked_for_echoes(self):
        raw = b"[" + (b'{"note": "' + SECRET.encode() + b'"},') * 20000 + b"{}]"
        await refused("repeats a credential", api=Api(raw=raw))
