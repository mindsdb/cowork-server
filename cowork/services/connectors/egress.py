"""Address vetting for connector hosts that a caller selected or entered.

Returning the addresses instead of a yes/no is the point: the caller dials
the address that was vetted, so a host cannot pass the check and then
resolve to a private address before the connection is made.

Provider-neutral on purpose. Each caller raises its own user-facing error
from these, because the host it is talking about is the caller's concept.
"""
from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable

#: Connect timeout for an attempt that has another address after it.
FALLBACK_CONNECT_SECONDS = 3.0
#: Addresses tried per request. A cloud host can answer with a dozen or more.
MAX_CONNECTION_ATTEMPTS = 4


class EgressHostUnresolved(RuntimeError):
    """The hostname could not be resolved."""


class EgressHostNotPublic(ValueError):
    """The hostname resolved to an address that is not globally routable."""


def vetted_public_addresses(
    hostname: str,
    resolver: Callable[..., list[tuple]],
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a hostname to the addresses a caller may dial.

    Accepts an address literal as itself. Raises ``EgressHostUnresolved``
    when resolution fails and ``EgressHostNotPublic`` when the answer holds
    any address that is not globally routable: one private address in a
    mixed answer refuses the whole host, because dialing the rest would
    still leave the private one reachable on a later attempt.

    A record whose address will not parse, such as a link-local IPv6 that
    ``getaddrinfo`` returns with its zone attached, is dropped rather than
    refused, and an answer left with nothing refuses.
    """

    try:
        literal = ipaddress.ip_address(hostname)
        addresses = [literal]
    except ValueError:
        try:
            records = resolver(hostname, 443, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise EgressHostUnresolved(hostname) from exc
        addresses = []
        for record in records:
            try:
                addresses.append(ipaddress.ip_address(record[4][0]))
            except (IndexError, ValueError):
                continue
    if not addresses or any(not address.is_global for address in addresses):
        raise EgressHostNotPublic(hostname)
    return addresses


def connection_attempts(
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address],
    *,
    total_seconds: float,
    fallback_seconds: float = FALLBACK_CONNECT_SECONDS,
    max_attempts: int = MAX_CONNECTION_ATTEMPTS,
) -> list[tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, float]]:
    """Return the vetted addresses to try, in order, each with its connect timeout.

    Families alternate, starting with the resolver's first answer, so a family
    with no route costs one short attempt rather than one per record. Every
    attempt but the last gets ``fallback_seconds`` and the last gets what is
    left of ``total_seconds``, so trying more addresses never lengthens the
    connect phase. Takes the output of ``vetted_public_addresses`` and never
    adds an address to it.
    """
    if not addresses:
        return []
    first_family = [address for address in addresses if address.version == addresses[0].version]
    other_family = [address for address in addresses if address.version != addresses[0].version]
    ordered = []
    for index in range(max(len(first_family), len(other_family))):
        ordered.extend(family[index] for family in (first_family, other_family) if index < len(family))
    ordered = ordered[:max_attempts]

    last_seconds = total_seconds - fallback_seconds * (len(ordered) - 1)
    if last_seconds <= 0:
        raise ValueError("total_seconds must leave the last attempt a positive connect timeout")
    return [(address, fallback_seconds) for address in ordered[:-1]] + [(ordered[-1], last_seconds)]
