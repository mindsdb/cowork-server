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
