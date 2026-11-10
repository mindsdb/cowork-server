"""The order and connect budget for dialing a connector host's vetted addresses.

A cloud host answers with many records per family, so walking them one by
one behind a dead route would spend a timeout on each. These pin that a dead
family costs one attempt and that the connect phase keeps a fixed total.
"""
from __future__ import annotations

import ipaddress

import pytest

from cowork.services.connectors.egress import (
    EgressHostNotPublic,
    connection_attempts,
    embedded_ipv4,
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


def _resolving_to(*addresses: str):
    """A resolver stub that answers every name with ``addresses``."""
    def resolve(_host, _port, type=None):
        return [(None, None, None, "", (address, 443)) for address in addresses]
    return resolve


@pytest.mark.parametrize(
    "address",
    [
        "64:ff9b::a00:5",  # NAT64 of 10.0.0.5
        "64:ff9b::7f00:1",  # NAT64 of 127.0.0.1
        "64:ff9b::a9fe:a9fe",  # NAT64 of the metadata address
        "::a00:5",  # IPv4-compatible 10.0.0.5
        "::ffff:0:a00:5",  # SIIT 10.0.0.5
        "64:ff9b:1::a00:5",  # local-use NAT64 prefix, refused as before
        "64:ff9b:1:a00:5::808:808",  # its IPv4 is not in the last 32 bits
        "fec0::1",  # site-local
        "224.0.0.251",  # multicast
        "ff02::1",
        "ff0e::1",
    ],
)
def test_an_address_that_reaches_a_private_or_multicast_target_is_refused(address):
    with pytest.raises(EgressHostNotPublic):
        vetted_public_addresses(address, _resolving_to(address))
    with pytest.raises(EgressHostNotPublic):
        vetted_public_addresses("api.example.com", _resolving_to("93.184.216.1", address))


@pytest.mark.parametrize("address", ["64:ff9b::808:808", "::ffff:0:808:808", "2600:1f18::1", "93.184.216.1"])
def test_an_address_that_reaches_a_public_target_is_allowed(address):
    assert vetted_public_addresses("api.example.com", _resolving_to(address)) == [ipaddress.ip_address(address)]


def test_embedded_ipv4_reads_only_the_fixed_position_ranges():
    assert embedded_ipv4(ipaddress.ip_address("64:ff9b::a00:5")) == ipaddress.ip_address("10.0.0.5")
    assert embedded_ipv4(ipaddress.ip_address("64:ff9b:1::a00:5")) is None
    assert embedded_ipv4(ipaddress.ip_address("10.0.0.5")) is None
