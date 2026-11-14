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
#: IPv6 ranges that carry an IPv4 address in their last 32 bits: well-known
#: NAT64, IPv4-compatible and SIIT. A translator dials that IPv4 address.
_EMBEDDED_IPV4_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("::ffff:0:0:0/96"),
)
_NOT_EMBEDDED = frozenset({ipaddress.ip_address("::"), ipaddress.ip_address("::1")})


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
    any address that is not public (``is_public_address``): one private address in a
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
    if not addresses or any(not is_public_address(address) for address in addresses):
        raise EgressHostNotPublic(hostname)
    return addresses


def embedded_ipv4(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | None:
    """Return the IPv4 address a NAT64, IPv4-compatible or SIIT address
    carries in its last 32 bits, or None for any other address."""
    if address.version != 6 or not any(address in net for net in _EMBEDDED_IPV4_NETWORKS):
        return None
    # "::" and "::1" sit inside ::/96 but are IPv6's own unspecified and loopback.
    if address in _NOT_EMBEDDED:
        return None
    return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)


def is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether ``address`` may be dialed as a public host.

    It must be globally routable, not multicast and not IPv6 site-local, and
    so must the IPv4 address it embeds (``embedded_ipv4``), since that is the
    one a translator connects to.
    """
    for candidate in (address, embedded_ipv4(address)):
        if candidate is None:
            continue
        if not candidate.is_global or candidate.is_multicast:
            return False
        if candidate.version == 6 and candidate.is_site_local:
            return False
    return True


def connection_attempts(
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address],
    *,
    total_seconds: float | None,
    fallback_seconds: float = FALLBACK_CONNECT_SECONDS,
    max_attempts: int = MAX_CONNECTION_ATTEMPTS,
) -> list[tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, float | None]]:
    """Return the vetted addresses to try, in order, each with its connect timeout.

    Families alternate, starting with the resolver's first answer, so a family
    with no route costs one short attempt rather than one per record. Every
    attempt but the last gets ``fallback_seconds`` and the last gets what is
    left of ``total_seconds``, so the connect timeouts of all attempts add up
    to the one a single attempt would have had. A budget too small for every
    short attempt tries fewer addresses. ``None`` is httpx's "no limit" and
    stays that on the last attempt. Takes the output of
    ``vetted_public_addresses`` and never adds an address to it.
    """
    if not addresses:
        return []
    first_family = [address for address in addresses if address.version == addresses[0].version]
    other_family = [address for address in addresses if address.version != addresses[0].version]
    ordered = []
    for index in range(max(len(first_family), len(other_family))):
        ordered.extend(family[index] for family in (first_family, other_family) if index < len(family))
    ordered = ordered[:max_attempts]

    last_seconds = None
    if total_seconds is not None:
        while len(ordered) > 1 and total_seconds - fallback_seconds * (len(ordered) - 1) <= 0:
            ordered.pop()
        last_seconds = total_seconds - fallback_seconds * (len(ordered) - 1)
    return [(address, fallback_seconds) for address in ordered[:-1]] + [(ordered[-1], last_seconds)]
