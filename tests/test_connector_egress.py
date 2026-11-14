"""The order and connect budget for dialing a connector host's vetted addresses.

A cloud host answers with many records per family, so walking them one by
one behind a dead route would spend a timeout on each. These pin that a dead
family costs one attempt and that the connect phase keeps a fixed total.
"""
from __future__ import annotations

import asyncio
import gzip
import ipaddress
import socket
import time

import httpx
import pytest

from cowork.services.connectors import egress
from cowork.services.connectors.egress import (
    EgressHostNotPublic,
    EgressHostUnresolved,
    EgressRefused,
    EgressTimeout,
    PinnedResponse,
    connection_attempts,
    is_public_address,
    pinned_get,
    vetted_public_addresses,
)

V6 = [ipaddress.ip_address(f"2600:1f18::{index}") for index in range(1, 9)]
V4 = [ipaddress.ip_address(f"93.184.216.{index}") for index in range(1, 9)]


def test_a_dual_stack_answer_alternates_families_from_the_first_record_and_is_capped():
    attempts = connection_attempts(V6 + V4, total_seconds=15.0, fallback_seconds=3.0, max_attempts=4)

    assert [address for address, _ in attempts] == [V6[0], V4[0], V6[1], V4[1]]


def test_an_answer_starting_with_ipv4_tries_ipv4_first():
    attempts = connection_attempts(V4[:2] + V6[:2], total_seconds=15.0, fallback_seconds=3.0, max_attempts=4)

    assert [address for address, _ in attempts] == [V4[0], V6[0], V4[1], V6[1]]


def test_one_family_keeps_resolver_order():
    attempts = connection_attempts(V4, total_seconds=15.0, fallback_seconds=3.0, max_attempts=4)

    assert [address for address, _ in attempts] == V4[:4]


def test_every_attempt_but_the_last_is_short_and_the_total_is_fixed():
    attempts = connection_attempts(V6 + V4, total_seconds=15.0, fallback_seconds=3.0, max_attempts=4)

    assert [seconds for _, seconds in attempts] == [3.0, 3.0, 3.0, 6.0]
    assert sum(seconds for _, seconds in attempts) == 15.0


def test_a_single_address_gets_the_whole_budget():
    assert connection_attempts(V4[:1], total_seconds=15.0) == [(V4[0], 15.0)]


def test_no_connect_limit_stays_unlimited_on_the_last_attempt_only():
    attempts = connection_attempts(V6[:1] + V4[:1], total_seconds=None, fallback_seconds=3.0)

    assert attempts == [(V6[0], 3.0), (V4[0], None)]


def test_no_addresses_means_no_attempts():
    assert connection_attempts([], total_seconds=15.0) == []


@pytest.mark.parametrize(
    ("total_seconds", "expected"),
    [
        (9.0, [(V6[0], 3.0), (V4[0], 3.0), (V6[1], 3.0)]),
        (3.0, [(V6[0], 3.0)]),
        (1.0, [(V6[0], 1.0)]),
    ],
)
def test_a_budget_too_small_for_every_short_attempt_tries_fewer_addresses(total_seconds, expected):
    """A request's own timeout sets the budget on the pinned transport, so a
    short one is legal and must still leave the last attempt time to connect."""
    attempts = connection_attempts(V6 + V4, total_seconds=total_seconds, fallback_seconds=3.0, max_attempts=4)

    assert attempts == expected


# ── which addresses count as public ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("8.8.8.8", True),
        ("2606:4700::1111", True),
        # Well-known NAT64 to a public IPv4: an IPv4-only API behind DNS64.
        ("64:ff9b::808:808", True),
        ("10.0.0.1", False),
        ("169.254.169.254", False),
        ("127.0.0.1", False),
        ("::1", False),
        ("fe80::1", False),
        ("fc00::1", False),
        # is_global admits these two, and both reach the metadata address.
        ("64:ff9b::a9fe:a9fe", False),
        ("::a9fe:a9fe", False),
        # Local-use NAT64, 6to4, Teredo and IPv4-mapped forms are never dialed.
        ("64:ff9b:1:a9fe:a9:fe00:808:808", False),
        ("2002:a9fe:a9fe::1", False),
        ("2001:0:4136:e378:8000:63bf:3fff:fdd2", False),
        ("::ffff:10.0.0.1", False),
        ("::ffff:8.8.8.8", False),
    ],
)
def test_is_public_address(address, public):
    assert is_public_address(ipaddress.ip_address(address)) is public


@pytest.mark.parametrize("private", ["64:ff9b::a9fe:a9fe", "::a9fe:a9fe", "2002:a9fe:a9fe::1"])
def test_every_caller_refuses_a_name_that_resolves_to_an_embedded_private_address(private):
    """PostHog, Jev and developer validation vet through this one function."""

    def resolver(host, port, type=None):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (private, 443, 0, 0))]

    with pytest.raises(EgressHostNotPublic):
        vetted_public_addresses("api.example.com", resolver)


# ── pinned_get ──────────────────────────────────────────────────────────────


def _resolve_to(*addresses):
    seen = []

    def resolver(host, port, type=None):
        seen.append(host)
        return [
            (socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 0, "", (a, 443)) for a in addresses
        ]

    resolver.seen = seen
    return resolver


class _Endless(httpx.AsyncByteStream):
    async def __aiter__(self):
        while True:
            yield b"x" * 4096


class _Slow(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"first"
        await asyncio.sleep(5)
        yield b"never"


@pytest.fixture
def pinned(monkeypatch):
    """Route pinned_get through a fake resolver and transport; returns the requests seen."""
    requests = []

    def use(handler, resolver=None):
        def record(request):
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(egress, "_RESOLVER", resolver or _resolve_to("93.184.216.34"))
        monkeypatch.setattr(egress, "_TRANSPORT", httpx.MockTransport(record))
        return requests

    return use


async def _get(path="/v1/me", **kwargs):
    return await pinned_get(
        "https://api.example.com",
        path,
        {"Authorization": "Bearer planted"},
        total_seconds=kwargs.pop("total_seconds", 2.0),
        max_body_bytes=kwargs.pop("max_body_bytes", 64),
    )


async def test_the_vetted_address_is_dialed_with_the_name_kept_for_tls_and_host(pinned):
    resolver = _resolve_to("93.184.216.34")
    requests = pinned(lambda request: httpx.Response(200, stream=httpx.ByteStream(b'{"id": 1}')), resolver)

    response = await _get()

    assert (response.status_code, response.body, response.truncated) == (200, b'{"id": 1}', False)
    [request] = requests
    assert request.url.host == "93.184.216.34"
    assert request.url.path == "/v1/me"
    assert request.headers["Host"] == "api.example.com"
    assert request.headers["Accept-Encoding"] == "identity"
    assert request.headers["Authorization"] == "Bearer planted"
    assert request.extensions["sni_hostname"] == "api.example.com"
    # Resolved as a fully qualified name, so no search domain is tried.
    assert resolver.seen == ["api.example.com."]


async def test_an_endless_body_is_read_only_up_to_the_cap(pinned):
    pinned(lambda request: httpx.Response(200, stream=_Endless()))

    response = await _get(max_body_bytes=100)

    assert len(response.body) == 100
    assert response.truncated is True


async def test_a_compressed_body_is_capped_before_it_is_unpacked(pinned):
    bomb = gzip.compress(b"\0" * 5_000_000)
    pinned(lambda request: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(bomb)))

    response = await _get(max_body_bytes=64)

    assert response.body == bomb[:64]
    assert response.truncated is True


async def test_a_slow_body_runs_into_the_one_deadline(pinned):
    pinned(lambda request: httpx.Response(200, stream=_Slow()))

    with pytest.raises(EgressTimeout):
        await _get(total_seconds=0.3)


async def test_a_slow_resolver_runs_into_the_same_deadline(pinned):
    def slow(host, port, type=None):
        time.sleep(1)
        return []

    pinned(lambda request: httpx.Response(200), slow)

    with pytest.raises(EgressTimeout):
        await _get(total_seconds=0.2)


async def test_a_redirect_is_answered_not_followed(pinned):
    requests = pinned(lambda request: httpx.Response(302, headers={"Location": "http://169.254.169.254/"}, stream=httpx.ByteStream(b"")))

    response = await _get()

    assert response.status_code == 302
    assert len(requests) == 1


@pytest.mark.parametrize("path", ["@evil.example/x", ":8443/v1/me", ".evil.example/v1"])
async def test_a_url_that_leaves_its_origin_is_refused_before_any_lookup(pinned, path):
    resolver = _resolve_to("93.184.216.34")
    requests = pinned(lambda request: httpx.Response(200), resolver)

    with pytest.raises(EgressRefused):
        await _get(path=path)

    assert resolver.seen == []
    assert requests == []


async def test_an_http_origin_is_refused():
    with pytest.raises(EgressRefused):
        await pinned_get("http://api.example.com", "/v1", {}, total_seconds=1, max_body_bytes=1)


async def test_a_name_resolving_to_a_private_address_is_never_dialed(pinned):
    requests = pinned(lambda request: httpx.Response(200), _resolve_to("93.184.216.34", "10.0.0.5"))

    with pytest.raises(EgressHostNotPublic):
        await _get()

    assert requests == []


async def test_an_unresolved_name_is_reported(pinned):
    def fail(host, port, type=None):
        raise socket.gaierror("no such name")

    pinned(lambda request: httpx.Response(200), fail)

    with pytest.raises(EgressHostUnresolved):
        await _get()


async def test_a_failed_connect_moves_to_the_next_address(pinned):
    def handler(request):
        if request.url.host == "93.184.216.34":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(204, stream=httpx.ByteStream(b""))

    requests = pinned(handler, _resolve_to("93.184.216.34", "93.184.216.35"))

    response = await _get()

    assert response.status_code == 204
    assert [r.url.host for r in requests] == ["93.184.216.34", "93.184.216.35"]


async def test_the_last_failed_connect_is_raised(pinned):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    pinned(handler)

    with pytest.raises(httpx.ConnectError):
        await _get()


def test_a_response_keeps_its_body_out_of_its_repr():
    assert "planted" not in repr(PinnedResponse(401, "Unauthorized", b"planted", False))
