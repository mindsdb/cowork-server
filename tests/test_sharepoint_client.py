import logging
from urllib.parse import parse_qs

import httpx
import pytest

from cowork.services.connectors import sharepoint
from cowork.services.connectors.sharepoint import (
    AppKey,
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
        client.list_libraries(SITE_URL)
    with _client(ms, fake_time) as client:
        client.list_libraries(SITE_URL)

    assert ms.token_count == 1


@pytest.mark.parametrize(
    ("status", "codes", "kind"),
    [
        (400, [90002], SharePointErrorKind.TENANT_NOT_FOUND),
        (400, [700016], SharePointErrorKind.APP_NOT_FOUND),
        (400, [700038], SharePointErrorKind.APP_NOT_FOUND),
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
    ms.on("GET", DRIVES_URL, httpx.Response(200, json={"value": [{"id": DRIVE_ID, "name": "Documents"}]}))

    with _client(ms) as client:
        libraries = client.list_libraries("https://Contoso.SharePoint.com/")

    assert [library.id for library in libraries] == [DRIVE_ID]
    assert [str(r.url.copy_with(query=None)) for r in ms.graph_requests()] == [root, DRIVES_URL]


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
        client.list_libraries(url)

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
        libraries = client.list_libraries(SITE_URL)

    assert [library.id for library in libraries] == ["b!first", DRIVE_ID]
    assert ms.graph_requests()[-1].url.params["$skiptoken"] == "d2"


def test_a_retry_after_longer_than_the_time_left_fails_at_once():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(429, headers={"Retry-After": "120"}))
    fake_time = FakeTime()

    with _client(ms, fake_time) as client, pytest.raises(SharePointError) as raised:
        client.list_libraries(SITE_URL)

    assert raised.value.kind is SharePointErrorKind.THROTTLED
    assert fake_time.slept == []
    assert len(ms.graph_requests()) == 1


@pytest.mark.parametrize("status", [429, 503])
def test_a_short_retry_after_is_waited_out(status):
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(status, headers={"Retry-After": "2"}), httpx.Response(200, json={"id": SITE_ID}))
    ms.on("GET", DRIVES_URL, httpx.Response(200, json={"value": [{"id": DRIVE_ID, "name": "Documents"}]}))
    fake_time = FakeTime()

    with _client(ms, fake_time) as client:
        libraries = client.list_libraries(SITE_URL)

    assert [library.id for library in libraries] == [DRIVE_ID]
    assert fake_time.slept == [2.0]


def test_throttling_gives_up_after_three_tries():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(429, headers={"Retry-After": "1"}))
    fake_time = FakeTime()

    with _client(ms, fake_time) as client, pytest.raises(SharePointError) as raised:
        client.list_libraries(SITE_URL)

    assert raised.value.kind is SharePointErrorKind.THROTTLED
    assert len(ms.graph_requests()) == 3
    assert fake_time.slept == [1.0, 1.0]


def test_a_401_from_graph_refetches_the_token_once():
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(401, json={}), httpx.Response(200, json={"id": SITE_ID}))
    ms.on("GET", DRIVES_URL, httpx.Response(200, json={"value": [{"id": DRIVE_ID, "name": "Documents"}]}))
    sharepoint._TOKENS[KEY.cache_key()] = ("revoked-token", 10**9)

    with _client(ms) as client:
        libraries = client.list_libraries(SITE_URL)

    assert [library.id for library in libraries] == [DRIVE_ID]
    first, second, _ = ms.graph_requests()
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
        client.list_libraries(SITE_URL)

    assert raised.value.kind is SharePointErrorKind.THROTTLED
    assert fake_time.slept == [25.0]


@pytest.mark.parametrize("header", ["nan", "inf", "-inf", "soon", "Wed, 21 Oct 2026 07:28:00 GMT"])
def test_an_unusable_retry_after_waits_the_default(header):
    ms = FakeMicrosoft()
    ms.on("GET", SITE_LOOKUP, httpx.Response(429, headers={"Retry-After": header}))
    fake_time = FakeTime()

    with _client(ms, fake_time) as client, pytest.raises(SharePointError) as raised:
        client.list_libraries(SITE_URL)

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
