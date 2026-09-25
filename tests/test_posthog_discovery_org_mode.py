"""Where org-mode PostHog project discovery is allowed to dial.

Cloud reaches the server's own PostHog cloud origins only, and only at an
address that was globally routable when the request was made. Desktop keeps
self-hosted hosts, so every test sets the tenancy mode it means.

Nothing here resolves or dials for real: the resolver and the transport are
injected, and the ones a test expects not to be reached fail it if they are.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.connectors import posthog as posthog_service
from cowork.services.connectors.posthog import (
    CLOUD_ORIGINS,
    PostHogDiscoveryError,
    discover_projects,
)

SPECS_DIR = Path(__file__).parent.parent / "cowork" / "services" / "connectors" / "specs"

# A sentinel, not a credential. Distinct so a leak would be attributable.
KEY = "phx_sentinel_discovery_key_4d1c"
PAYLOAD = {"results": [{"id": 12, "name": "Production"}]}
PUBLIC_V4 = "93.184.216.34"
PUBLIC_V6 = "2600:1f18::1"

NOT_PUBLIC = [
    "127.0.0.1",
    "10.0.0.1",
    "172.16.0.1",
    "192.168.1.1",
    "169.254.169.254",
    "0.0.0.0",
    "100.64.0.1",
    "::1",
    "fe80::1",
    "fc00::1",
    "::ffff:127.0.0.1",
    "::ffff:169.254.169.254",
    "::",
]


def _answer(*addresses: str) -> Callable[..., list[tuple]]:
    """A getaddrinfo stand-in answering with one record per address."""

    def resolve(_host, _port, **_kwargs):
        return [(2, 1, 6, "", (address, 443)) for address in addresses]

    return resolve


def _refuse_resolution() -> Callable[..., list[tuple]]:
    def resolve(host, *_args, **_kwargs):
        pytest.fail(f"resolved {host}")

    return resolve


def _refuse_network() -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: pytest.fail(f"dialed {request.url}"))


def _serve(record: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return httpx.Response(200, json=PAYLOAD)

    return httpx.MockTransport(handle)


@pytest.fixture(autouse=True)
def _reset_app_settings():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()


@pytest.mark.asyncio
async def test_an_approved_origin_is_dialed_at_the_vetted_address(org_mode):
    """The address is what gets dialed; the hostname stays on Host and on SNI,
    so the certificate is still verified against it."""
    seen: list[httpx.Request] = []

    projects = await discover_projects(
        personal_api_key=KEY,
        host="https://us.posthog.com",
        transport=_serve(seen),
        resolver=_answer(PUBLIC_V4),
    )

    assert [(project.id, project.name) for project in projects] == [("12", "Production")]
    assert str(seen[0].url) == f"https://{PUBLIC_V4}/api/projects/"
    assert seen[0].headers["Host"] == "us.posthog.com"
    assert seen[0].extensions["sni_hostname"] == "us.posthog.com"
    assert seen[0].headers["Authorization"] == f"Bearer {KEY}"


@pytest.mark.asyncio
async def test_the_second_approved_origin_is_reachable_too(org_mode):
    seen: list[httpx.Request] = []

    await discover_projects(
        personal_api_key=KEY,
        host="https://eu.posthog.com",
        transport=_serve(seen),
        resolver=_answer(PUBLIC_V4),
    )

    assert seen[0].headers["Host"] == "eu.posthog.com"


@pytest.mark.asyncio
async def test_an_uppercase_origin_is_dialed_as_the_canonical_origin(org_mode):
    """DNS is case-insensitive, so the match is too, and what reaches the
    request is the server's constant rather than the caller's string."""
    seen: list[httpx.Request] = []

    await discover_projects(
        personal_api_key=KEY,
        host="https://US.POSTHOG.COM",
        transport=_serve(seen),
        resolver=_answer(PUBLIC_V4),
    )

    assert seen[0].headers["Host"] == "us.posthog.com"


@pytest.mark.asyncio
async def test_a_global_ipv6_answer_is_dialed(org_mode):
    seen: list[httpx.Request] = []

    await discover_projects(
        personal_api_key=KEY,
        host="https://us.posthog.com",
        transport=_serve(seen),
        resolver=_answer(PUBLIC_V6),
    )

    assert str(seen[0].url) == f"https://[{PUBLIC_V6}]/api/projects/"
    assert seen[0].headers["Host"] == "us.posthog.com"


@pytest.mark.asyncio
async def test_a_dual_stack_answer_falls_back_to_the_next_vetted_address(org_mode):
    """A host with no egress for the first family would otherwise lose
    discovery outright, where an unpinned dial walks every address."""
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == PUBLIC_V6:
            raise httpx.ConnectError("network unreachable", request=request)
        return httpx.Response(200, json=PAYLOAD)

    projects = await discover_projects(
        personal_api_key=KEY,
        host="https://us.posthog.com",
        transport=httpx.MockTransport(handle),
        resolver=_answer(PUBLIC_V6, PUBLIC_V4),
    )

    assert [project.id for project in projects] == ["12"]
    assert seen == [PUBLIC_V6, PUBLIC_V4]


def _dual_stack_answer() -> Callable[..., list[tuple]]:
    """Eight records per family, IPv6 first, the shape PostHog's cloud hosts answer with."""
    return _answer(*(f"2600:1f18::{index}" for index in range(1, 9)), *(f"93.184.216.{index}" for index in range(1, 9)))


@pytest.mark.asyncio
async def test_a_family_that_drops_packets_costs_one_short_attempt(org_mode):
    """A route that silently drops packets raises a connect timeout rather than a
    connect error, and waiting it out on each of eight records would take longer
    than the request timeout before IPv4 was tried."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if ":" in request.url.host:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, json=PAYLOAD)

    projects = await discover_projects(
        personal_api_key=KEY,
        host="https://us.posthog.com",
        transport=httpx.MockTransport(handle),
        resolver=_dual_stack_answer(),
    )

    assert [project.id for project in projects] == ["12"]
    assert [request.url.host for request in seen] == ["2600:1f18::1", "93.184.216.1"]
    assert seen[0].extensions["timeout"]["connect"] < 15.0
    assert seen[0].extensions["timeout"]["read"] == 15.0


@pytest.mark.asyncio
async def test_when_every_address_times_out_the_connect_phase_fits_the_request_timeout(org_mode):
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ConnectTimeout("timed out", request=request)

    with pytest.raises(PostHogDiscoveryError, match="Could not reach PostHog") as refused:
        await discover_projects(
            personal_api_key=KEY,
            host="https://us.posthog.com",
            transport=httpx.MockTransport(handle),
            resolver=_dual_stack_answer(),
        )

    assert len(seen) > 1
    assert sum(request.extensions["timeout"]["connect"] for request in seen) <= 15.0
    assert KEY not in str(refused.value)


@pytest.mark.asyncio
async def test_a_redirect_is_not_followed(org_mode):
    """A 302 from an approved origin would otherwise be re-dialed wherever it
    points, past the allowlist and past the address check, and its body would
    come back through the project parser."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    with pytest.raises(PostHogDiscoveryError, match="invalid project list"):
        await discover_projects(
            personal_api_key=KEY,
            host="https://us.posthog.com",
            transport=httpx.MockTransport(handle),
            resolver=_answer(PUBLIC_V4),
        )

    assert [str(request.url) for request in seen] == [f"https://{PUBLIC_V4}/api/projects/"]


@pytest.mark.asyncio
async def test_a_custom_host_is_refused_without_resolving_or_dialing(org_mode):
    with pytest.raises(PostHogDiscoveryError, match="US Cloud or EU Cloud"):
        await discover_projects(
            personal_api_key=KEY,
            host="custom",
            custom_host="https://posthog.internal.example",
            transport=_refuse_network(),
            resolver=_refuse_resolution(),
        )


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("https://us.posthog.com.evil.example", "US Cloud or EU Cloud"),
        ("https://us.posthog.com:8443", "US Cloud or EU Cloud"),
        ("https://us.posthog.com:443", "US Cloud or EU Cloud"),
        ("https://us.posthog.com/api", "US Cloud or EU Cloud"),
        ("https://us.posthog.com.", "US Cloud or EU Cloud"),
        ("https://us.posthog.com%2e.evil.example", "US Cloud or EU Cloud"),
        ("https://us.xn--psthog-8za.com", "US Cloud or EU Cloud"),
        ("https://uſ.posthog.com", "US Cloud or EU Cloud"),
        ("https://user:pw@us.posthog.com", "valid HTTPS PostHog host"),
        ("http://us.posthog.com", "valid HTTPS PostHog host"),
        ("us.posthog.com", "valid HTTPS PostHog host"),
        ("", "valid HTTPS PostHog host"),
    ],
)
@pytest.mark.asyncio
async def test_a_host_outside_the_allowlist_is_refused_before_dialing(org_mode, host, expected):
    with pytest.raises(PostHogDiscoveryError, match=expected):
        await discover_projects(
            personal_api_key=KEY,
            host=host,
            transport=_refuse_network(),
            resolver=_refuse_resolution(),
        )


@pytest.mark.asyncio
async def test_a_unicode_case_mapping_cannot_match_an_origin(org_mode, monkeypatch):
    """U+212A lowercases to "k". Neither origin holds a "k" today, so this
    pins the ASCII-only precondition for the next one that does."""
    monkeypatch.setattr(posthog_service, "CLOUD_ORIGINS", ("https://uk.posthog.com",))

    with pytest.raises(PostHogDiscoveryError, match="US Cloud or EU Cloud"):
        await discover_projects(
            personal_api_key=KEY,
            host="https://uK.posthog.com",
            transport=_refuse_network(),
            resolver=_refuse_resolution(),
        )


@pytest.mark.parametrize("address", NOT_PUBLIC)
@pytest.mark.asyncio
async def test_an_approved_origin_resolving_off_the_public_internet_is_refused(org_mode, address):
    with pytest.raises(PostHogDiscoveryError, match="Could not reach PostHog"):
        await discover_projects(
            personal_api_key=KEY,
            host="https://us.posthog.com",
            transport=_refuse_network(),
            resolver=_answer(address),
        )


@pytest.mark.asyncio
async def test_a_mixed_answer_is_refused_rather_than_partly_dialed(org_mode):
    """One private address refuses the whole host: dialing the rest would
    leave the private one reachable on a later attempt."""
    with pytest.raises(PostHogDiscoveryError, match="Could not reach PostHog"):
        await discover_projects(
            personal_api_key=KEY,
            host="https://us.posthog.com",
            transport=_refuse_network(),
            resolver=_answer(PUBLIC_V4, "10.0.0.1"),
        )


@pytest.mark.asyncio
async def test_a_later_answer_cannot_change_the_address_that_is_dialed(org_mode):
    """The check and the connection use one resolution, so there is no window
    between them for the name to move."""
    calls: list[str] = []

    def resolve(host, _port, **_kwargs):
        calls.append(host)
        address = PUBLIC_V4 if len(calls) == 1 else "169.254.169.254"
        return [(2, 1, 6, "", (address, 443))]

    seen: list[httpx.Request] = []
    await discover_projects(
        personal_api_key=KEY,
        host="https://us.posthog.com",
        transport=_serve(seen),
        resolver=resolve,
    )

    assert calls == ["us.posthog.com"]
    assert str(seen[0].url) == f"https://{PUBLIC_V4}/api/projects/"


@pytest.mark.asyncio
async def test_a_host_that_cannot_be_resolved_answers_the_unreachable_message(org_mode):
    def resolve(_host, _port, **_kwargs):
        raise OSError("temporary failure in name resolution")

    with pytest.raises(PostHogDiscoveryError, match="Could not reach PostHog"):
        await discover_projects(
            personal_api_key=KEY,
            host="https://us.posthog.com",
            transport=_refuse_network(),
            resolver=resolve,
        )


@pytest.mark.asyncio
async def test_a_refusal_names_the_desktop_app_rather_than_a_url_to_fix(org_mode):
    """The spec still offers "Self-hosted (enter URL)" in cloud, so the
    message is the only thing that tells a caller no URL will do."""
    with pytest.raises(PostHogDiscoveryError) as refused:
        await discover_projects(
            personal_api_key=KEY,
            host="custom",
            custom_host="https://posthog.internal.example",
            transport=_refuse_network(),
            resolver=_refuse_resolution(),
        )

    assert "desktop app" in str(refused.value)


@pytest.mark.asyncio
async def test_no_refusal_echoes_the_personal_api_key(org_mode, caplog):
    caplog.set_level(logging.DEBUG)

    for host, custom_host, resolver in [
        ("custom", "https://posthog.internal.example", _refuse_resolution()),
        ("https://us.posthog.com.evil.example", None, _refuse_resolution()),
        ("https://us.posthog.com", None, _answer("169.254.169.254")),
    ]:
        with pytest.raises(PostHogDiscoveryError) as refused:
            await discover_projects(
                personal_api_key=KEY,
                host=host,
                custom_host=custom_host,
                transport=_refuse_network(),
                resolver=resolver,
            )
        assert KEY not in str(refused.value)

    assert KEY not in caplog.text


@pytest.mark.asyncio
async def test_local_mode_keeps_a_self_hosted_host_unvetted():
    """Desktop's own PostHog is routinely on a private network, so the
    restriction is deliberately a cloud-only boundary."""
    seen: list[httpx.Request] = []

    projects = await discover_projects(
        personal_api_key=KEY,
        host="custom",
        custom_host="https://posthog.internal.example",
        transport=_serve(seen),
        resolver=_refuse_resolution(),
    )

    assert [project.id for project in projects] == ["12"]
    assert str(seen[0].url) == "https://posthog.internal.example/api/projects/"
    assert "sni_hostname" not in seen[0].extensions


def test_the_allowlist_matches_the_connector_spec_host_options():
    """A region added to the form is a red build rather than a cloud refusal
    nobody chose."""
    spec = json.loads((SPECS_DIR / "posthog.json").read_text())
    method = next(item for item in spec["form"]["methods"] if item["id"] == "personal-api-key")
    field = next(item for item in method["fields"] if item["name"] == "host")
    offered = [option["value"] for option in field["options"] if option["value"] != "custom"]

    assert list(CLOUD_ORIGINS) == offered
