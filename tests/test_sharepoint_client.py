import logging
import os
import subprocess
import sys
import textwrap
from urllib.parse import parse_qs

import httpx
import pytest

from cowork.services.connectors import sharepoint
from cowork.services.connectors.sharepoint import (
    AppKey,
    DownloadBudget,
    SharePointClient,
    SharePointError,
    SharePointErrorKind,
    Site,
    normalise_site_url,
    parse_site_urls,
    validate_tenant_id,
)

SECRET = "sp-secret-value-5f1c"
TOKEN = "sp-access-token-9a7e"
KEY = AppKey(tenant_id="contoso.onmicrosoft.com", client_id="11111111-2222-3333-4444-555555555555", client_secret=SECRET)
SITE_URL = "https://contoso.sharepoint.com/sites/team"
SITE_ID = "contoso.sharepoint.com,aaaa-1111,bbbb-2222"
DRIVE_ID = "b!AbC_d-1"
TOKEN_URL = "https://login.microsoftonline.com/contoso.onmicrosoft.com/oauth2/v2.0/token"
SITE_LOOKUP = "https://graph.microsoft.com/v1.0/sites/contoso.sharepoint.com:/sites/team"
DRIVES_URL = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drives"
CHILDREN_URL = f"{DRIVES_URL}/{DRIVE_ID}/items/root/children"
NEXT_LINK = f"{CHILDREN_URL}?$top=200&$skiptoken=page-2-token"


@pytest.fixture(autouse=True)
def _empty_token_cache():
    sharepoint._TOKENS.clear()
    yield
    sharepoint._TOKENS.clear()


class FakeMicrosoft:
    """Answers Entra and Graph from a table of (method, URL without query) -> responses, recording every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[tuple[str, str], list[httpx.Response]] = {}
        self.token_count = 0

    def on(self, method: str, url: str, *responses: httpx.Response) -> None:
        self.routes[(method, url)] = list(responses)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url.copy_with(query=None))
        if request.method == "POST" and url == TOKEN_URL and (request.method, url) not in self.routes:
            self.token_count += 1
            return httpx.Response(200, json={"access_token": f"{TOKEN}-{self.token_count}", "expires_in": 3600})
        responses = self.routes.get((request.method, url))
        if not responses:
            return httpx.Response(404, json={"error": {"code": "itemNotFound"}})
        return responses.pop(0) if len(responses) > 1 else responses[0]

    def graph_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == "graph.microsoft.com"]


class FakeTime:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _client(ms: FakeMicrosoft, fake_time: FakeTime | None = None, key: AppKey = KEY) -> SharePointClient:
    fake_time = fake_time or FakeTime()
    return SharePointClient(key, transport=httpx.MockTransport(ms.handler), clock=fake_time.clock, sleep=fake_time.sleep)


def _site_ok(ms: FakeMicrosoft) -> None:
    ms.on("GET", SITE_LOOKUP, httpx.Response(200, json={"id": SITE_ID, "name": "team"}))
    ms.on("GET", DRIVES_URL, httpx.Response(200, json={"value": [{"id": DRIVE_ID, "name": "Documents"}]}))


def test_check_sends_client_credentials_and_resolves_each_site():
    ms = FakeMicrosoft()
    _site_ok(ms)

    with _client(ms) as client:
        sites = client.check([SITE_URL])

    assert sites == [Site(url=SITE_URL, graph_id=SITE_ID)]
    token_request = ms.requests[0]
    assert str(token_request.url) == TOKEN_URL
    assert "authorization" not in token_request.headers
    form = parse_qs(token_request.content.decode())
    assert form == {
        "client_id": [KEY.client_id],
        "client_secret": [SECRET],
        "grant_type": ["client_credentials"],
        "scope": ["https://graph.microsoft.com/.default"],
    }
    lookup, drives = ms.graph_requests()
    assert str(lookup.url) == SITE_LOOKUP
    assert lookup.headers["authorization"] == f"Bearer {TOKEN}-1"
    assert str(drives.url.copy_with(query=None)) == DRIVES_URL
    assert drives.url.params["$select"] == "id,name"


def test_check_caches_the_token_only_after_every_site_passed():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(403, json={}))

    with _client(ms) as client, pytest.raises(SharePointError):
        client.check([SITE_URL])

    assert sharepoint._TOKENS == {}


def test_check_fetches_a_fresh_token_even_when_one_is_cached():
    ms = FakeMicrosoft()
    _site_ok(ms)
    sharepoint._TOKENS[KEY.cache_key()] = ("stale-cached-token", 10**9)

    with _client(ms) as client:
        client.check([SITE_URL])

    assert ms.token_count == 1
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}-1" for r in ms.graph_requests())
    assert sharepoint._TOKENS[KEY.cache_key()][0] == f"{TOKEN}-1"


def test_browse_reuses_a_cached_token_within_its_lifetime():
    ms = FakeMicrosoft()
    _site_ok(ms)
    fake_time = FakeTime()

    with _client(ms, fake_time) as client:
        client.resolve_site(SITE_URL)
    with _client(ms, fake_time) as client:
        client.resolve_site(SITE_URL)

    assert ms.token_count == 1


@pytest.mark.parametrize(
    ("status", "codes", "kind"),
    [
        (400, [90002], SharePointErrorKind.TENANT_NOT_FOUND),
        (400, [700016], SharePointErrorKind.APP_NOT_FOUND),
        (401, [7000215], SharePointErrorKind.KEY_REJECTED),
        (401, [7000222], SharePointErrorKind.KEY_REJECTED),
        (400, [50000], SharePointErrorKind.UNREACHABLE),
        (500, [], SharePointErrorKind.UNREACHABLE),
    ],
)
def test_token_failures_are_told_apart_by_entra_error_code(status, codes, kind):
    ms = FakeMicrosoft()
    ms.on("POST", TOKEN_URL, httpx.Response(status, json={"error": "invalid_request", "error_codes": codes}))

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.check([SITE_URL])

    assert raised.value.kind is kind
    assert ms.graph_requests() == []


def test_distinct_messages_for_each_key_failure():
    messages = {
        SharePointError(kind).args[0]
        for kind in (
            SharePointErrorKind.TENANT_NOT_FOUND,
            SharePointErrorKind.APP_NOT_FOUND,
            SharePointErrorKind.KEY_REJECTED,
        )
    }
    assert len(messages) == 3


@pytest.mark.parametrize("status", [403, 404])
def test_an_ungranted_site_names_the_site_url(status):
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(status, json={}))

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.check([SITE_URL])

    assert raised.value.kind is SharePointErrorKind.SITE_NOT_GRANTED
    assert SITE_URL in str(raised.value)


def test_root_site_uses_the_host_only_form():
    ms = FakeMicrosoft()
    root = "https://graph.microsoft.com/v1.0/sites/contoso.sharepoint.com"
    ms.on("GET", root, httpx.Response(200, json={"id": SITE_ID}))

    with _client(ms) as client:
        site = client.resolve_site("https://Contoso.SharePoint.com/")

    assert site == Site(url="https://contoso.sharepoint.com", graph_id=SITE_ID)
    assert [str(r.url) for r in ms.graph_requests()] == [root]


@pytest.mark.parametrize(
    "url",
    [
        "http://contoso.sharepoint.com/sites/team",
        "https://contoso.sharepoint.com/sites/team:/drives",
        "https://contoso.sharepoint.com/sites/team?x=1",
        "https://contoso.sharepoint.com/sites/team#x",
        "https://contoso.sharepoint.com/sites/..",
        "https://contoso.sharepoint.com/sites/a/b",
        "https://contoso.sharepoint.com/personal/team",
        "https://contoso.sharepoint.com/sites/te%2Fam",
        "https://contoso-my.sharepoint.com/personal/ada",
        "https://user:pw@contoso.sharepoint.com/sites/team",
        "https://contoso.sharepoint.com:8443/sites/team",
        "https://contoso.sharepoint.com.evil.example/sites/team",
        "https://evil.example/sites/team",
    ],
)
def test_site_urls_outside_the_allowed_shapes_are_refused_before_any_request(url):
    ms = FakeMicrosoft()

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.resolve_site(url)

    assert raised.value.kind is SharePointErrorKind.INVALID
    assert ms.requests == []


def test_site_urls_are_normalised_deduplicated_and_capped():
    text = "https://Contoso.sharepoint.com/Sites/Team/\n\nhttps://contoso.sharepoint.com/sites/team\nhttps://contoso.sharepoint.com/teams/ops"
    assert parse_site_urls(text) == [SITE_URL, "https://contoso.sharepoint.com/teams/ops"]
    assert normalise_site_url(" https://contoso.sharepoint.com ") == "https://contoso.sharepoint.com"

    too_many = "\n".join(f"https://contoso.sharepoint.com/sites/s{i}" for i in range(21))
    with pytest.raises(SharePointError):
        parse_site_urls(too_many)
    with pytest.raises(SharePointError):
        parse_site_urls("  \n ")


@pytest.mark.parametrize("tenant", ["contoso.onmicrosoft.com", "11111111-2222-3333-4444-555555555555"])
def test_tenant_ids_accept_a_domain_or_guid(tenant):
    assert validate_tenant_id(f" {tenant} ") == tenant


@pytest.mark.parametrize("tenant", ["contoso", "contoso.com/evil", "a@b.com", "contoso.com?x", "x\\y.com", "%2e.com", ""])
def test_bad_tenant_ids_are_refused_before_any_request(tenant):
    ms = FakeMicrosoft()

    with _client(ms, key=AppKey(tenant, KEY.client_id, SECRET)) as client, pytest.raises(SharePointError) as raised:
        client.check([SITE_URL])

    assert raised.value.kind is SharePointErrorKind.INVALID
    assert ms.requests == []


@pytest.mark.parametrize(
    ("drive_id", "item_id"),
    [("../../sites/other", "root"), (DRIVE_ID, "../../../sites/other/drives/x"), (DRIVE_ID, "a/b"), (DRIVE_ID, ".."), ("b!x%2F", "root")],
)
def test_traversal_ids_are_refused_before_any_request(drive_id, item_id):
    ms = FakeMicrosoft()

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.list_children(SITE_URL, drive_id, item_id)

    assert raised.value.kind is SharePointErrorKind.INVALID
    assert ms.requests == []


def test_a_library_from_another_site_is_refused_before_any_children_call():
    ms = FakeMicrosoft()
    _site_ok(ms)

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.list_children(SITE_URL, "b!OtherSitesDrive")

    assert raised.value.kind is SharePointErrorKind.ITEM_NOT_FOUND
    assert [str(r.url.copy_with(query=None)) for r in ms.graph_requests()] == [SITE_LOOKUP, DRIVES_URL]


def test_children_ask_only_the_listed_fields_and_classify_items():
    ms = FakeMicrosoft()
    _site_ok(ms)
    ms.on(
        "GET",
        CHILDREN_URL,
        httpx.Response(
            200,
            json={
                "value": [
                    {"id": "01ABC", "name": "q3.xlsx", "size": 42, "file": {}, "lastModifiedDateTime": "2026-10-01T00:00:00Z",
                     "@microsoft.graph.downloadUrl": "https://contoso.sharepoint.com/download?tempauth=leak"},
                    {"id": "01DEF", "name": "Reports", "folder": {"childCount": 3}},
                    {"id": "01GHI", "name": "Notebook", "package": {"type": "oneNote"}},
                ]
            },
        ),
    )

    with _client(ms) as client:
        page = client.list_children(SITE_URL, DRIVE_ID)

    children = ms.graph_requests()[-1]
    assert str(children.url.copy_with(query=None)) == CHILDREN_URL
    assert children.url.params["$select"] == "id,name,size,file,folder,lastModifiedDateTime"
    assert children.url.params["$top"] == "200"
    assert [(i.name, i.kind, i.size) for i in page.items] == [("q3.xlsx", "file", 42), ("Reports", "folder", 0), ("Notebook", "other", 0)]
    assert "tempauth" not in repr(page)
    assert page.cursor is None


@pytest.mark.parametrize("key", ["$skiptoken", "%24skiptoken"])
def test_the_next_page_rebuilds_the_same_url_from_the_skiptoken_only(key):
    ms = FakeMicrosoft()
    _site_ok(ms)
    forged_link = f"https://evil.example/other/path?{key}=page-2-token"
    ms.on("GET", CHILDREN_URL, httpx.Response(200, json={"value": [], "@odata.nextLink": forged_link}))

    with _client(ms) as client:
        first = client.list_children(SITE_URL, DRIVE_ID)
        client.list_children(SITE_URL, DRIVE_ID, cursor=first.cursor)

    assert first.cursor == "page-2-token"
    second = ms.graph_requests()[-1]
    assert second.url.host == "graph.microsoft.com"
    assert str(second.url.copy_with(query=None)) == CHILDREN_URL
    assert second.url.params["$skiptoken"] == "page-2-token"
    assert all(r.url.host != "evil.example" for r in ms.requests)


def test_a_forged_cursor_stays_a_query_value():
    ms = FakeMicrosoft()
    _site_ok(ms)
    ms.on("GET", CHILDREN_URL, httpx.Response(200, json={"value": []}))
    forged = "x&$filter=1#/../../sites/other/drives"

    with _client(ms) as client:
        client.list_children(SITE_URL, DRIVE_ID, cursor=forged)

    children = ms.graph_requests()[-1]
    assert str(children.url.copy_with(query=None)) == CHILDREN_URL
    assert children.url.params["$skiptoken"] == forged
    assert "$filter" not in children.url.params


def test_a_next_link_without_a_skiptoken_is_an_error():
    ms = FakeMicrosoft()
    _site_ok(ms)
    ms.on("GET", CHILDREN_URL, httpx.Response(200, json={"value": [], "@odata.nextLink": f"{CHILDREN_URL}?$top=200"}))

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.list_children(SITE_URL, DRIVE_ID)

    assert raised.value.kind is SharePointErrorKind.UNREACHABLE


def test_libraries_are_read_across_pages():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(200, json={"id": SITE_ID}))
    ms.on(
        "GET",
        DRIVES_URL,
        httpx.Response(200, json={"value": [{"id": "b!first", "name": "A"}], "@odata.nextLink": f"{DRIVES_URL}?$skiptoken=d2"}),
        httpx.Response(200, json={"value": [{"id": DRIVE_ID, "name": "B"}]}),
    )

    with _client(ms) as client:
        libraries = client.list_drives(SITE_URL)

    assert [library.id for library in libraries] == ["b!first", DRIVE_ID]
    assert ms.graph_requests()[-1].url.params["$skiptoken"] == "d2"


def test_a_retry_after_longer_than_the_time_left_fails_at_once():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(429, headers={"Retry-After": "120"}))
    fake_time = FakeTime()

    with _client(ms, fake_time) as client, pytest.raises(SharePointError) as raised:
        client.resolve_site(SITE_URL)

    assert raised.value.kind is SharePointErrorKind.THROTTLED
    assert fake_time.slept == []
    assert len(ms.graph_requests()) == 1


@pytest.mark.parametrize("status", [429, 503])
def test_a_short_retry_after_is_waited_out(status):
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(status, headers={"Retry-After": "2"}), httpx.Response(200, json={"id": SITE_ID}))
    fake_time = FakeTime()

    with _client(ms, fake_time) as client:
        site = client.resolve_site(SITE_URL)

    assert site.graph_id == SITE_ID
    assert fake_time.slept == [2.0]


def test_throttling_gives_up_after_three_tries():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(429, headers={"Retry-After": "1"}))
    fake_time = FakeTime()

    with _client(ms, fake_time) as client, pytest.raises(SharePointError) as raised:
        client.resolve_site(SITE_URL)

    assert raised.value.kind is SharePointErrorKind.THROTTLED
    assert len(ms.graph_requests()) == 3
    assert fake_time.slept == [1.0, 1.0]


def test_a_401_from_graph_refetches_the_token_once():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(401, json={}), httpx.Response(200, json={"id": SITE_ID}))
    sharepoint._TOKENS[KEY.cache_key()] = ("revoked-token", 10**9)

    with _client(ms) as client:
        site = client.resolve_site(SITE_URL)

    assert site.graph_id == SITE_ID
    first, second = ms.graph_requests()
    assert first.headers["authorization"] == "Bearer revoked-token"
    assert second.headers["authorization"] == f"Bearer {TOKEN}-1"
    assert sharepoint._TOKENS[KEY.cache_key()][0] == f"{TOKEN}-1"


def test_network_failures_are_unreachable():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = SharePointClient(KEY, transport=httpx.MockTransport(refuse))
    with client, pytest.raises(SharePointError) as raised:
        client.check([SITE_URL])

    assert raised.value.kind is SharePointErrorKind.UNREACHABLE


def test_the_secret_and_token_stay_out_of_errors_and_logs():
    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collect(level=logging.DEBUG)
    loggers = [logging.getLogger(name) for name in ("httpx", "httpcore", sharepoint.__name__)]
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
    try:
        errors = []
        ms = FakeMicrosoft()
        ms.on("POST", TOKEN_URL, httpx.Response(401, json={"error_codes": [7000215], "error_description": f"bad {SECRET}"}))
        with _client(ms) as client, pytest.raises(SharePointError) as raised:
            client.check([SITE_URL])
        errors.append(raised.value)

        ms = FakeMicrosoft()
        ms.on("GET", SITE_LOOKUP, httpx.Response(500, json={"error": {"message": f"token {TOKEN}-1 failed"}}))
        with _client(ms) as client, pytest.raises(SharePointError) as raised:
            client.check([SITE_URL])
        errors.append(raised.value)
    finally:
        for lg, level in saved:
            lg.removeHandler(handler)
            lg.setLevel(level)

    assert records, "the httpx logger was not captured"
    for text in [r.getMessage() for r in records] + [repr(e) for e in errors]:
        assert SECRET not in text
        assert TOKEN not in text
    for error in errors:
        assert "graph.microsoft.com" not in str(error)
        assert "login.microsoftonline.com" not in str(error)


def test_one_browse_call_shares_one_deadline_across_its_requests():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(429, headers={"Retry-After": "25"}), httpx.Response(200, json={"id": SITE_ID}))
    ms.on("GET", DRIVES_URL, httpx.Response(429, headers={"Retry-After": "10"}))
    fake_time = FakeTime()

    with _client(ms, fake_time) as client, pytest.raises(SharePointError) as raised:
        client.list_drives(SITE_URL)

    assert raised.value.kind is SharePointErrorKind.THROTTLED
    assert fake_time.slept == [25.0]


@pytest.mark.parametrize("header", ["nan", "inf", "-inf", "soon", "Wed, 21 Oct 2026 07:28:00 GMT"])
def test_an_unusable_retry_after_waits_the_default(header):
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(429, headers={"Retry-After": header}))
    fake_time = FakeTime()

    with _client(ms, fake_time) as client, pytest.raises(SharePointError) as raised:
        client.resolve_site(SITE_URL)

    assert raised.value.kind is SharePointErrorKind.THROTTLED
    assert fake_time.slept == [1.0, 1.0]


@pytest.mark.parametrize("codes", [7000215, "7000215", {"a": 1}, None])
def test_error_codes_that_are_not_a_list_count_as_unreachable(codes):
    ms = FakeMicrosoft()
    ms.on("POST", TOKEN_URL, httpx.Response(401, json={"error_codes": codes}))

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.check([SITE_URL])

    assert raised.value.kind is SharePointErrorKind.UNREACHABLE


def test_a_modified_time_that_is_not_text_is_dropped():
    ms = FakeMicrosoft()
    _site_ok(ms)
    ms.on("GET", CHILDREN_URL, httpx.Response(200, json={"value": [{"id": "01A", "name": "a", "file": {}, "lastModifiedDateTime": 5}]}))

    with _client(ms) as client:
        page = client.list_children(SITE_URL, DRIVE_ID)

    assert page.items[0].modified is None


def test_a_401_during_the_check_fetches_a_fresh_token_without_touching_the_cache():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(401, json={}), httpx.Response(403, json={}))
    sharepoint._TOKENS[KEY.cache_key()] = ("another-requests-token", 10**9)

    with _client(ms) as client, pytest.raises(SharePointError):
        client.check([SITE_URL])

    first, second = ms.graph_requests()
    assert first.headers["authorization"] == f"Bearer {TOKEN}-1"
    assert second.headers["authorization"] == f"Bearer {TOKEN}-2"
    assert sharepoint._TOKENS[KEY.cache_key()][0] == "another-requests-token"


def test_a_401_during_a_passing_check_caches_the_token_that_worked():
    ms = FakeMicrosoft()
    _site_ok(ms)
    ms.on("GET", SITE_LOOKUP, httpx.Response(401, json={}), httpx.Response(200, json={"id": SITE_ID}))

    with _client(ms) as client:
        client.check([SITE_URL])

    assert sharepoint._TOKENS[KEY.cache_key()][0] == f"{TOKEN}-2"


def test_every_error_kind_has_a_standard_message():
    for kind in SharePointErrorKind:
        assert str(SharePointError(kind))


ITEMS_URL = f"{DRIVES_URL}/{DRIVE_ID}/items"


def _file(item_id: str, name: str, size: int = 10) -> dict:
    return {"id": item_id, "name": name, "size": size, "file": {}}


def _folder(item_id: str, name: str) -> dict:
    return {"id": item_id, "name": name, "folder": {"childCount": 1}}


def _library(ms: FakeMicrosoft, picks: list[dict], children: dict[str, list[dict]] | None = None) -> None:
    """Serve the site, its library, each picked item, and one page of children per folder id."""
    _site_ok(ms)
    for entry in picks:
        ms.on("GET", f"{ITEMS_URL}/{entry['id']}", httpx.Response(200, json=entry))
    for folder_id, entries in (children or {}).items():
        ms.on("GET", f"{ITEMS_URL}/{folder_id}/children", httpx.Response(200, json={"value": entries}))


def _children_calls(ms: FakeMicrosoft, folder_id: str) -> int:
    url = f"{ITEMS_URL}/{folder_id}/children"
    return sum(1 for r in ms.graph_requests() if str(r.url.copy_with(query=None)) == url)


def _paths(plan) -> list[tuple[str, ...]]:
    return [planned.parts for planned in plan.files]


def test_plan_places_a_picked_folders_tree_under_its_name_and_a_picked_file_by_name():
    ms = FakeMicrosoft()
    _library(
        ms,
        [_folder("01A", "Reports"), _file("01T", "top.xlsx", 7)],
        {"01A": [_file("01a", "a.docx", 3), _folder("01B", "Q3")], "01B": [_file("01b", "b.xlsx", 5)]},
    )

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01A", "01T"])

    assert plan.site == Site(url=SITE_URL, graph_id=SITE_ID)
    assert plan.drive_id == DRIVE_ID
    assert [(f.parts, f.item_id, f.size) for f in plan.files] == [
        (("Reports", "a.docx"), "01a", 3),
        (("Reports", "Q3", "b.xlsx"), "01b", 5),
        (("top.xlsx",), "01T", 7),
    ]
    assert plan.skipped == []
    urls = [str(r.url.copy_with(query=None)) for r in ms.graph_requests()]
    assert urls.count(SITE_LOOKUP) == 1
    assert urls.count(DRIVES_URL) == 1
    pick = next(r for r in ms.graph_requests() if str(r.url.copy_with(query=None)) == f"{ITEMS_URL}/01A")
    assert pick.url.params["$select"] == "id,name,size,file,folder"
    walk = next(r for r in ms.graph_requests() if str(r.url.copy_with(query=None)) == f"{ITEMS_URL}/01A/children")
    assert walk.url.params["$select"] == "id,name,size,file,folder,lastModifiedDateTime"
    assert walk.headers["authorization"] == f"Bearer {TOKEN}-1"


def test_plan_reads_a_folder_across_pages_with_one_site_lookup_and_one_library_list():
    ms = FakeMicrosoft()
    _library(ms, [_folder("01A", "Big")])
    first = [_file(f"01f{n}", f"f{n}.txt") for n in range(60)]
    second = [_file(f"01g{n}", f"g{n}.txt") for n in range(40)]
    ms.on(
        "GET",
        f"{ITEMS_URL}/01A/children",
        httpx.Response(200, json={"value": first, "@odata.nextLink": f"{ITEMS_URL}/01A/children?$skiptoken=next-1"}),
        httpx.Response(200, json={"value": second}),
    )

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert len(plan.files) == 100
    pages = [r for r in ms.graph_requests() if str(r.url.copy_with(query=None)) == f"{ITEMS_URL}/01A/children"]
    assert [r.url.params.get("$skiptoken") for r in pages] == [None, "next-1"]
    urls = [str(r.url.copy_with(query=None)) for r in ms.graph_requests()]
    assert urls.count(SITE_LOOKUP) == 1
    assert urls.count(DRIVES_URL) == 1


def test_skipped_items_do_not_count_toward_the_file_limit():
    ms = FakeMicrosoft()
    entries = [_file(f"01f{n}", f"f{n}.txt") for n in range(100)] + [_file("01ds", ".DS_Store")]
    _library(ms, [_folder("01A", "Big")], {"01A": entries})

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert len(plan.files) == 100
    assert [(s.path, s.reason) for s in plan.skipped] == [("Big/.DS_Store", "its name starts with a dot")]


def test_more_than_100_files_is_refused_naming_the_limit():
    ms = FakeMicrosoft()
    _library(ms, [_folder("01A", "Big")], {"01A": [_file(f"01f{n}", f"f{n}.txt") for n in range(101)]})

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert raised.value.kind is SharePointErrorKind.TOO_MANY_FILES
    assert "100-file limit" in str(raised.value)


def test_more_than_1_gb_of_listed_bytes_is_refused_naming_the_limit():
    ms = FakeMicrosoft()
    _library(ms, [_file("01a", "a.bin", 2**29), _file("01b", "b.bin", 2**29 + 1)])

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, DRIVE_ID, ["01a", "01b"])

    assert raised.value.kind is SharePointErrorKind.TOO_LARGE
    assert "1 GB limit" in str(raised.value)
    assert str(raised.value) != str(SharePointError(SharePointErrorKind.TOO_MANY_FILES))


def test_more_than_200_folders_is_refused():
    ms = FakeMicrosoft()
    subfolders = [_folder(f"01s{n}", f"s{n}") for n in range(200)]
    _library(ms, [_folder("01A", "Wide")], {"01A": subfolders, **{f"01s{n}": [] for n in range(200)}})

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert raised.value.kind is SharePointErrorKind.TOO_LARGE
    assert "200-folder limit" in str(raised.value)


def test_more_than_200_pages_is_refused():
    ms = FakeMicrosoft()
    _library(ms, [_folder("01A", "Long")])
    link = f"{ITEMS_URL}/01A/children?$skiptoken=more"
    ms.on("GET", f"{ITEMS_URL}/01A/children", *[httpx.Response(200, json={"value": [], "@odata.nextLink": link})] * 201)

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert raised.value.kind is SharePointErrorKind.TOO_LARGE
    assert _children_calls(ms, "01A") == 200


def test_a_tree_deeper_than_32_levels_is_refused():
    ms = FakeMicrosoft()
    children = {f"01d{n}": [_folder(f"01d{n + 1}", f"d{n + 1}")] for n in range(40)}
    _library(ms, [_folder("01d0", "d0")], children)

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, DRIVE_ID, ["01d0"])

    assert raised.value.kind is SharePointErrorKind.TOO_LARGE
    assert "32 folders deep" in str(raised.value)
    assert _children_calls(ms, "01d32") == 0


@pytest.mark.parametrize("name", ["..", ".", "a/b", "a\\b", "nul\x00byte", "tab\tname", "del\x7f"])
def test_a_hostile_name_fails_the_plan_naming_it(name):
    ms = FakeMicrosoft()
    _library(ms, [_folder("01A", "Docs")], {"01A": [_file("01ok", "ok.txt"), _file("01x", name)]})

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert raised.value.kind is SharePointErrorKind.BAD_NAME
    assert repr(name) in str(raised.value)


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("", "it has no name"),
        (".DS_Store", "its name starts with a dot"),
        ("._report.xlsx", "its name starts with a dot"),
        ("a:b.txt", "its name can't be saved on every computer"),
        ("what?.txt", "its name can't be saved on every computer"),
        (" lead.txt", "its name can't be saved on every computer"),
        ("trail.txt ", "its name can't be saved on every computer"),
        ("dot.", "its name can't be saved on every computer"),
        ("CON", "its name is reserved on Windows"),
        ("con.txt", "its name is reserved on Windows"),
        ("Lpt1.tar.gz", "its name is reserved on Windows"),
        ("x" * 252 + ".txt", "its name is too long"),
    ],
)
def test_a_name_the_desktop_cannot_hold_is_skipped_and_the_rest_planned(name, reason):
    ms = FakeMicrosoft()
    _library(ms, [_folder("01A", "Docs")], {"01A": [_file("01ok", "ok.txt"), _file("01x", name)]})

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert _paths(plan) == [("Docs", "ok.txt")]
    assert [(s.path, s.reason) for s in plan.skipped] == [(f"Docs/{name}", reason)]


def test_a_onenote_item_is_skipped():
    ms = FakeMicrosoft()
    notebook = {"id": "01N", "name": "Notebook", "package": {"type": "oneNote"}}
    _library(ms, [_folder("01A", "Docs")], {"01A": [notebook, _file("01ok", "ok.txt")]})

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert _paths(plan) == [("Docs", "ok.txt")]
    assert [(s.path, s.reason) for s in plan.skipped] == [("Docs/Notebook", "it is not a file or a folder")]


def test_a_skipped_folder_is_not_walked():
    ms = FakeMicrosoft()
    _library(ms, [_folder("01A", "Repo")], {"01A": [_folder("01G", ".git")], "01G": [_file("01h", "HEAD")]})

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert plan.files == []
    assert _children_calls(ms, "01G") == 0


@pytest.mark.parametrize("item_id", ["root", "ROOT"])
def test_the_library_root_cannot_be_picked(item_id):
    ms = FakeMicrosoft()

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, DRIVE_ID, [item_id])

    assert raised.value.kind is SharePointErrorKind.INVALID
    assert ms.requests == []


@pytest.mark.parametrize("order", [["01Q", "01S"], ["01S", "01Q"]])
def test_a_picked_folder_inside_another_picked_folder_lands_under_it_once(order):
    ms = FakeMicrosoft()
    outer, inner = _folder("01Q", "Q&A #1"), _folder("01S", "Sub")
    _library(ms, [outer, inner], {"01Q": [inner, _file("01q", "q.txt")], "01S": [_file("01s", "s.txt")]})

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, order)

    assert sorted(_paths(plan)) == [("Q&A #1", "Sub", "s.txt"), ("Q&A #1", "q.txt")]
    assert _children_calls(ms, "01S") == 1


def test_a_picked_file_inside_a_picked_folder_is_placed_once():
    ms = FakeMicrosoft()
    inner = _file("01r", "r.xlsx")
    _library(ms, [inner, _folder("01A", "Reports")], {"01A": [inner]})

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01r", "01A"])

    assert _paths(plan) == [("Reports", "r.xlsx")]


def test_one_item_reached_by_two_ids_is_placed_once():
    ms = FakeMicrosoft()
    _library(ms, [_file("01X", "x.txt")])
    ms.on("GET", f"{ITEMS_URL}/01X-alias", httpx.Response(200, json=_file("01X", "x.txt")))

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01X", "01X-alias", "01X"])

    assert _paths(plan) == [("x.txt",)]


def test_a_spent_planning_budget_is_reported_as_too_slow():
    ms = FakeMicrosoft()
    _library(ms, [_folder("01A", "Docs")])
    fake_time = FakeTime()

    def slow(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/01A/children"):
            fake_time.now += 61
            raise httpx.ReadTimeout("slow", request=request)
        return ms.handler(request)

    client = SharePointClient(KEY, transport=httpx.MockTransport(slow), clock=fake_time.clock, sleep=fake_time.sleep)
    with client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert raised.value.kind is SharePointErrorKind.PLAN_TOO_SLOW
    assert "longer than a minute" in str(raised.value)


def test_an_unreachable_host_with_time_left_stays_unreachable():
    ms = FakeMicrosoft()
    _library(ms, [_folder("01A", "Docs")])

    def down(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/01A/children"):
            raise httpx.ConnectError("down", request=request)
        return ms.handler(request)

    client = SharePointClient(KEY, transport=httpx.MockTransport(down), clock=FakeTime().clock)
    with client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, DRIVE_ID, ["01A"])

    assert raised.value.kind is SharePointErrorKind.UNREACHABLE


def test_a_library_from_another_site_is_refused_before_any_pick_is_read():
    ms = FakeMicrosoft()
    _site_ok(ms)

    with _client(ms) as client, pytest.raises(SharePointError) as raised:
        client.plan_import(SITE_URL, "b!OtherSitesDrive", ["01A"])

    assert raised.value.kind is SharePointErrorKind.ITEM_NOT_FOUND
    assert [str(r.url.copy_with(query=None)) for r in ms.graph_requests()] == [SITE_LOOKUP, DRIVES_URL]


SITE = Site(url=SITE_URL, graph_id=SITE_ID)
CONTENT_URL = f"{ITEMS_URL}/01f/content"
DOWNLOAD_PAGE = "https://contoso.sharepoint.com/sites/team/_layouts/15/download.aspx"
DOWNLOAD_URL = f"{DOWNLOAD_PAGE}?UniqueId=01f&tempauth=tempauth-secret-7c1d"


class Chunks(httpx.SyncByteStream):
    """A response body read one raw chunk at a time, recording reads and whether it was closed."""

    def __init__(self, chunks: list[bytes], events: list[str] | None = None, before=None) -> None:
        self.chunks = chunks
        self.events = events if events is not None else []
        self.before = before
        self.closed = False

    def __iter__(self):
        for index, chunk in enumerate(self.chunks):
            if self.before is not None:
                self.before(index)
            self.events.append(f"read {index}")
            yield chunk

    def close(self) -> None:
        self.closed = True


def _redirect(location: str | None, status: int = 302) -> httpx.Response:
    headers = {"Location": location} if location is not None else {}
    return httpx.Response(status, headers=headers, stream=Chunks([]))


def _download(ms: FakeMicrosoft, fake_time: FakeTime | None = None, budget: DownloadBudget | None = None, sink=None):
    fake_time = fake_time or FakeTime()
    received: list[bytes] = []
    with _client(ms, fake_time) as client:
        deadline = client.import_deadline()
        written = client.download(SITE, DRIVE_ID, "01f", sink or received.append, budget or DownloadBudget(), deadline)
    return written, received


def _non_token_requests(ms: FakeMicrosoft) -> list[httpx.Request]:
    return [r for r in ms.requests if r.url.host != "login.microsoftonline.com"]


def test_download_follows_one_redirect_to_the_site_host_without_the_token():
    ms = FakeMicrosoft()
    redirect_body = Chunks([])
    body = Chunks([b"hello ", b"world"])
    ms.on("GET", CONTENT_URL, httpx.Response(302, headers={"Location": DOWNLOAD_URL}, stream=redirect_body))
    ms.on("GET", DOWNLOAD_PAGE, httpx.Response(200, stream=body))

    written, received = _download(ms)

    assert written == 11
    assert received == [b"hello ", b"world"]
    content, fetch = _non_token_requests(ms)
    assert str(content.url) == CONTENT_URL
    assert content.headers["authorization"] == f"Bearer {TOKEN}-1"
    assert str(fetch.url) == DOWNLOAD_URL
    assert "authorization" not in fetch.headers
    assert redirect_body.closed and body.closed


def test_a_200_from_graph_is_streamed_directly():
    ms = FakeMicrosoft()
    ms.on("GET", CONTENT_URL, httpx.Response(200, stream=Chunks([b"abc"])))

    written, received = _download(ms)

    assert (written, received) == (3, [b"abc"])
    assert len(_non_token_requests(ms)) == 1


@pytest.mark.parametrize(
    ("status", "location"),
    [
        (302, DOWNLOAD_URL),
        (303, DOWNLOAD_URL),
        (302, DOWNLOAD_URL.replace("contoso.sharepoint.com", "Contoso.SharePoint.COM")),
        (302, DOWNLOAD_URL.replace("contoso.sharepoint.com", "contoso.sharepoint.com:443")),
    ],
)
def test_allowed_redirects(status, location):
    ms = FakeMicrosoft()
    ms.on("GET", CONTENT_URL, _redirect(location, status))
    ms.on("GET", DOWNLOAD_PAGE, httpx.Response(200, stream=Chunks([b"ok"])))

    assert _download(ms) == (2, [b"ok"])


@pytest.mark.parametrize(
    "location",
    [
        None,
        "",
        "https://evil.example/download.aspx?tempauth=tempauth-secret-7c1d",
        "https://other.sharepoint.com/x?tempauth=tempauth-secret-7c1d",
        "http://contoso.sharepoint.com/x?tempauth=tempauth-secret-7c1d",
        "https://contoso.sharepoint.com:8443/x?tempauth=tempauth-secret-7c1d",
        "https://contoso.sharepoint.com:abc/x?tempauth=tempauth-secret-7c1d",
        "https://user:pw@contoso.sharepoint.com/x?tempauth=tempauth-secret-7c1d",
        "https://contoso.sharepoint.com%2F@evil.example/x?tempauth=tempauth-secret-7c1d",
        "/sites/team/_layouts/15/download.aspx?tempauth=tempauth-secret-7c1d",
    ],
)
def test_a_redirect_anywhere_else_is_refused_without_a_request(location):
    ms = FakeMicrosoft()
    ms.on("GET", CONTENT_URL, _redirect(location))

    with pytest.raises(SharePointError) as raised:
        _download(ms)

    assert raised.value.kind is SharePointErrorKind.UNREACHABLE
    assert "tempauth" not in str(raised.value)
    assert [str(r.url) for r in _non_token_requests(ms)] == [CONTENT_URL]


@pytest.mark.parametrize("second", [302, 404, 500])
def test_any_other_answer_from_the_site_host_is_unreachable(second):
    ms = FakeMicrosoft()
    ms.on("GET", CONTENT_URL, _redirect(DOWNLOAD_URL))
    ms.on("GET", DOWNLOAD_PAGE, _redirect(DOWNLOAD_URL) if second == 302 else httpx.Response(second, stream=Chunks([b"x"])))

    with pytest.raises(SharePointError) as raised:
        _download(ms)

    assert raised.value.kind is SharePointErrorKind.UNREACHABLE
    assert len(_non_token_requests(ms)) == 2


def test_a_missing_file_is_item_not_found():
    ms = FakeMicrosoft()
    ms.on("GET", CONTENT_URL, httpx.Response(404, stream=Chunks([b"{}"])))

    with pytest.raises(SharePointError) as raised:
        _download(ms)

    assert raised.value.kind is SharePointErrorKind.ITEM_NOT_FOUND


def test_throttling_is_waited_out_on_both_hops():
    ms = FakeMicrosoft()
    busy = {"Retry-After": "2"}
    ms.on("GET", CONTENT_URL, httpx.Response(429, headers=busy), _redirect(DOWNLOAD_URL))
    ms.on("GET", DOWNLOAD_PAGE, httpx.Response(503, headers=busy), httpx.Response(200, stream=Chunks([b"ok"])))
    fake_time = FakeTime()

    assert _download(ms, fake_time) == (2, [b"ok"])
    assert fake_time.slept == [2.0, 2.0]


def test_a_401_on_the_content_request_refetches_the_token_once():
    ms = FakeMicrosoft()
    ms.on("GET", CONTENT_URL, httpx.Response(401), _redirect(DOWNLOAD_URL))
    ms.on("GET", DOWNLOAD_PAGE, httpx.Response(200, stream=Chunks([b"ok"])))

    assert _download(ms) == (2, [b"ok"])
    first, second, _ = _non_token_requests(ms)
    assert first.headers["authorization"] == f"Bearer {TOKEN}-1"
    assert second.headers["authorization"] == f"Bearer {TOKEN}-2"


def test_two_401s_on_the_content_request_mean_the_key_was_rejected():
    ms = FakeMicrosoft()
    ms.on("GET", CONTENT_URL, httpx.Response(401))

    with pytest.raises(SharePointError) as raised:
        _download(ms)

    assert raised.value.kind is SharePointErrorKind.KEY_REJECTED
    assert ms.token_count == 2


def test_each_raw_read_reaches_the_sink_before_the_next_is_read():
    ms = FakeMicrosoft()
    events: list[str] = []
    ms.on("GET", CONTENT_URL, httpx.Response(200, stream=Chunks([b"a", b"b", b"c"], events)))

    _download(ms, sink=lambda chunk: events.append(f"sink {chunk.decode()}"))

    assert events == ["read 0", "sink a", "read 1", "sink b", "read 2", "sink c"]


def test_the_import_deadline_passing_between_two_reads_stops_the_download():
    ms = FakeMicrosoft()
    fake_time = FakeTime()

    def late(index: int) -> None:
        if index == 1:
            fake_time.now += sharepoint.IMPORT_SECONDS

    body = Chunks([b"first", b"second"], before=late)
    ms.on("GET", CONTENT_URL, httpx.Response(200, stream=body))
    received: list[bytes] = []

    with pytest.raises(SharePointError) as raised:
        _download(ms, fake_time, sink=received.append)

    assert raised.value.kind is SharePointErrorKind.IMPORT_TOO_SLOW
    assert "15 minutes" in str(raised.value)
    assert received == [b"first"]
    assert body.closed


def test_a_read_timeout_at_the_end_of_the_import_budget_is_too_slow():
    ms = FakeMicrosoft()
    fake_time = FakeTime()

    def stall(index: int) -> None:
        if index == 1:
            fake_time.now += sharepoint.IMPORT_SECONDS
            raise httpx.ReadTimeout("stalled")

    ms.on("GET", CONTENT_URL, httpx.Response(200, stream=Chunks([b"first", b"second"], before=stall)))

    with pytest.raises(SharePointError) as raised:
        _download(ms, fake_time)

    assert raised.value.kind is SharePointErrorKind.IMPORT_TOO_SLOW


def test_the_shared_byte_budget_stops_a_file_mid_way():
    ms = FakeMicrosoft()
    body = Chunks([b"abc", b"def"])
    ms.on("GET", CONTENT_URL, httpx.Response(200, stream=body))
    received: list[bytes] = []

    with pytest.raises(SharePointError) as raised:
        _download(ms, budget=DownloadBudget(5), sink=received.append)

    assert raised.value.kind is SharePointErrorKind.TOO_LARGE
    assert received == [b"abc"]
    assert body.closed


def test_a_disk_error_from_the_sink_propagates_and_the_response_is_closed():
    ms = FakeMicrosoft()
    body = Chunks([b"abc"])
    ms.on("GET", CONTENT_URL, _redirect(DOWNLOAD_URL))
    ms.on("GET", DOWNLOAD_PAGE, httpx.Response(200, stream=body))

    def full_disk(_chunk: bytes) -> None:
        raise OSError(28, "No space left on device")

    with pytest.raises(OSError) as raised:
        _download(ms, sink=full_disk)

    assert raised.value.errno == 28
    assert body.closed


def test_the_download_url_stays_out_of_errors_and_the_module_log():
    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collect(level=logging.DEBUG)
    module_logger = logging.getLogger(sharepoint.__name__)
    saved_level = module_logger.level
    module_logger.addHandler(handler)
    module_logger.setLevel(logging.DEBUG)
    errors = []
    try:
        for second in (httpx.Response(500, stream=Chunks([DOWNLOAD_URL.encode()])), _redirect(DOWNLOAD_URL)):
            ms = FakeMicrosoft()
            ms.on("GET", CONTENT_URL, _redirect(DOWNLOAD_URL))
            ms.on("GET", DOWNLOAD_PAGE, second)
            with pytest.raises(SharePointError) as raised:
                _download(ms)
            errors.append(raised.value)
        ms = FakeMicrosoft()
        ms.on("GET", CONTENT_URL, _redirect(DOWNLOAD_URL.replace("contoso", "evil")))
        with pytest.raises(SharePointError) as raised:
            _download(ms)
        errors.append(raised.value)
    finally:
        module_logger.removeHandler(handler)
        module_logger.setLevel(saved_level)

    for text in [r.getMessage() for r in records] + [repr(e) for e in errors] + [repr(e.__cause__) for e in errors]:
        assert "tempauth" not in text
        assert "download.aspx" not in text


_LOGGED_DOWNLOAD = textwrap.dedent(
    """
    import logging

    import httpx

    from cowork.common.logger import setup_logging
    from cowork.services.connectors.sharepoint import AppKey, DownloadBudget, SharePointClient, Site

    setup_logging()
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logging.getLogger().addHandler(Collect(level=logging.DEBUG))

    def answer(request):
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
        if request.url.host == "graph.microsoft.com":
            return httpx.Response(302, headers={"Location": "%(download)s"})
        return httpx.Response(200, content=b"bytes")

    key = AppKey("contoso.onmicrosoft.com", "client", "secret")
    site = Site(url="https://contoso.sharepoint.com/sites/team", graph_id="contoso.sharepoint.com,a,b")
    with SharePointClient(key, transport=httpx.MockTransport(answer)) as client:
        client.download(site, "b!Lib", "01f", lambda chunk: None, DownloadBudget(), client.import_deadline())
    logging.getLogger("cowork.sharepoint.probe").info("collector works")
    print("RECORDS", records)
    """
) % {"download": DOWNLOAD_URL}


def test_cowork_logging_keeps_the_download_url_out_of_every_log():
    env = {**os.environ, "LOG_LEVEL": "DEBUG", "ENABLE_FILE_LOGGING": "false"}
    result = subprocess.run(
        [sys.executable, "-c", _LOGGED_DOWNLOAD], capture_output=True, text=True, timeout=60, env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "collector works" in result.stdout
    assert "tempauth" not in result.stdout + result.stderr


def test_a_skipped_pick_inside_a_picked_folder_is_listed_once_under_its_path():
    ms = FakeMicrosoft()
    hidden = _folder("01G", ".git")
    _library(ms, [hidden, _folder("01A", "Repo")], {"01A": [hidden, _file("01r", "readme.md")]})

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01G", "01A"])

    assert _paths(plan) == [("Repo", "readme.md")]
    assert [(s.path, s.reason) for s in plan.skipped] == [("Repo/.git", "its name starts with a dot")]


def test_a_skipped_item_picked_by_two_ids_is_listed_once():
    ms = FakeMicrosoft()
    _library(ms, [_file("01d", ".DS_Store")])
    ms.on("GET", f"{ITEMS_URL}/01d-alias", httpx.Response(200, json=_file("01d", ".DS_Store")))

    with _client(ms) as client:
        plan = client.plan_import(SITE_URL, DRIVE_ID, ["01d", "01d-alias"])

    assert plan.files == []
    assert [s.path for s in plan.skipped] == [".DS_Store"]
