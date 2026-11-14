"""Address vetting for connector hosts that a caller selected or entered.

Returning the addresses instead of a yes/no is the point: the caller dials
the address that was vetted, so a host cannot pass the check and then
resolve to a private address before the connection is made.

Provider-neutral on purpose. Each caller raises its own user-facing error
from these, because the host it is talking about is the caller's concept.
"""
from __future__ import annotations

import asyncio
import contextvars
import ipaddress
import logging
import socket
import ssl
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote

import httpx

#: Connect timeout for an attempt that has another address after it.
FALLBACK_CONNECT_SECONDS = 3.0
#: Addresses tried per request. A cloud host can answer with a dozen or more.
MAX_CONNECTION_ATTEMPTS = 4


class EgressHostUnresolved(RuntimeError):
    """The hostname could not be resolved."""


class EgressHostNotPublic(ValueError):
    """The hostname resolved to an address that is not globally routable."""


class EgressRefused(ValueError):
    """The request would leave the origin it was built for."""


class EgressTimeout(TimeoutError):
    """Resolution and every connection attempt together outlasted their deadline."""


#: Global unicast IPv6. Everything outside it, IPv4-embedding forms included,
#: is refused unless it is the well-known NAT64 prefix below.
_GLOBAL_UNICAST_V6 = ipaddress.ip_network("2000::/3")
#: Well-known NAT64 (RFC 6052): the IPv4 it reaches is in the low 32 bits.
_WELL_KNOWN_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether a dial to ``address`` can only reach the public internet.

    IPv4 is ``is_global``. IPv6 is global unicast that ``is_global`` accepts,
    or well-known NAT64 whose embedded IPv4 is public, so an IPv4-only API
    behind DNS64 still works while NAT64 to a private or metadata address
    does not. ``is_global`` alone admits ``64:ff9b::a9fe:a9fe`` and
    ``::a9fe:a9fe``, which reach 169.254.169.254; it already refuses 6to4
    (``2002::/16``) and Teredo (``2001::/32``), which the tests pin.
    """
    if address.version == 4:
        return address.is_global
    if address in _WELL_KNOWN_NAT64:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF).is_global
    return address in _GLOBAL_UNICAST_V6 and address.is_global


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
    if not addresses or any(not is_public_address(address) for address in addresses):
        raise EgressHostNotPublic(hostname)
    return addresses


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


#: Resolution runs here, not in the default pool: a lookup that outlives its
#: deadline keeps its thread until it ends, and this pool bounds how many do.
_RESOLVER_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="egress-resolve")
# Replaced in tests.
_RESOLVER: Callable[..., list[tuple]] = socket.getaddrinfo
#: A fake transport for tests; None builds a fresh real one for each send.
_TRANSPORT: httpx.AsyncBaseTransport | None = None
#: Certificate verification for the real transport; a test points it at its own CA.
_VERIFY: ssl.SSLContext | bool = True

PINNED_USER_AGENT = "MindsHub-Connector/1"
#: Sent unless the caller sets the same header, in any case.
_DEFAULT_HEADERS = {"Accept": "*/*", "User-Agent": PINNED_USER_AGENT}
#: Always ours: a caller header of either name, in any case, is dropped.
_FIXED_HEADER_NAMES = frozenset({"host", "accept-encoding"})

#: Every logger httpx and httpcore write to; a test checks this against their source.
LIBRARY_LOGGERS = (
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)
_PINNED_SEND = contextvars.ContextVar("egress_pinned_send", default=False)


class _QuietDuringPinnedSend(logging.Filter):
    """Drop library records while a pinned send runs in this context.

    Those records hold the URL and response headers, which carry a key sent
    in a query or echoed back. A filter on ``httpcore`` would not see its
    children's records, so each logger gets this filter itself.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep the record unless a pinned send is running in the current context."""
        return not _PINNED_SEND.get()


_QUIET_FILTER = _QuietDuringPinnedSend()
for _name in LIBRARY_LOGGERS:
    logging.getLogger(_name).addFilter(_QUIET_FILTER)


@dataclass(frozen=True, repr=False)
class PinnedResponse:
    """The answer to a pinned GET: its status, reason and at most the first bytes of its body.

    ``repr=False`` keeps the body, which may echo a credential, out of any traceback.
    """

    status_code: int
    reason_phrase: str
    body: bytes
    truncated: bool


def _with_params(path: str, params: Sequence[tuple[str, str]]) -> str:
    """Append ``params`` to the path's own query, which is kept byte for byte."""
    if not params:
        return path
    pairs = "&".join(f"{quote(name, safe='')}={quote(value, safe='')}" for name, value in params)
    if "?" not in path:
        return f"{path}?{pairs}"
    return f"{path}{'' if path.endswith(('?', '&')) else '&'}{pairs}"


def _request_headers(headers: dict[str, str], netloc: str) -> list[tuple[str, str]]:
    """The exact headers sent: the caller's, our defaults it did not set, and ours that it may not set."""
    given = {name.lower() for name in headers}
    sent = [(name, value) for name, value in headers.items() if name.lower() not in _FIXED_HEADER_NAMES]
    sent += [(name, value) for name, value in _DEFAULT_HEADERS.items() if name.lower() not in given]
    return [*sent, ("Host", netloc), ("Accept-Encoding", "identity")]


async def _read_capped(response: httpx.Response, max_body_bytes: int) -> PinnedResponse:
    """Read at most ``max_body_bytes`` of the raw body."""
    body = bytearray()
    truncated = False
    async for chunk in response.aiter_raw():
        body.extend(chunk[: max_body_bytes - len(body)])
        if len(body) >= max_body_bytes:
            truncated = True
            break
    return PinnedResponse(response.status_code, response.reason_phrase, bytes(body), truncated)


async def pinned_get(
    origin: str,
    path: str,
    headers: dict[str, str],
    *,
    total_seconds: float,
    max_body_bytes: int,
    params: Sequence[tuple[str, str]] = (),
) -> PinnedResponse:
    """GET ``origin`` + ``path`` at a vetted public address, within one deadline.

    ``origin`` is ``https://host[:port]``. ``params`` are appended to the
    path's query, which is otherwise sent as written. The URL is refused
    unless it keeps the origin's scheme, host and port. The name is resolved
    with a trailing dot, so a search domain is never tried, and every address
    must be public (``is_public_address``). The vetted address is dialed with
    the hostname on SNI and ``Host``, through a transport of its own, so no
    redirect is followed, no proxy used and no header added beyond
    ``_request_headers``. The body is asked for without compression and read
    raw, at most ``max_body_bytes``. One ``total_seconds`` covers resolution
    and every attempt. No httpx or httpcore record is written meanwhile.

    Raises ``EgressRefused``, ``EgressHostUnresolved``, ``EgressHostNotPublic``,
    ``EgressTimeout`` or the last attempt's ``httpx`` connect error.
    """
    token = _PINNED_SEND.set(True)
    try:
        return await _pinned_get(origin, _with_params(path, params), headers, total_seconds, max_body_bytes)
    finally:
        _PINNED_SEND.reset(token)


async def _pinned_get(
    origin: str, path: str, headers: dict[str, str], total_seconds: float, max_body_bytes: int
) -> PinnedResponse:
    """The send ``pinned_get`` describes, run while its records are dropped."""
    base = httpx.URL(origin)
    url = httpx.URL(origin + path)
    if (
        base.scheme != "https"
        or not base.host
        or (url.scheme, url.host, url.port) != (base.scheme, base.host, base.port)
    ):
        raise EgressRefused("the request would leave its origin")
    request_headers = _request_headers(headers, url.netloc.decode("ascii"))
    loop = asyncio.get_running_loop()
    transport = _TRANSPORT or httpx.AsyncHTTPTransport(verify=_VERIFY, trust_env=False)
    try:
        async with asyncio.timeout(total_seconds):
            addresses = await loop.run_in_executor(_RESOLVER_POOL, vetted_public_addresses, url.host + ".", _RESOLVER)
            attempts = connection_attempts(addresses, total_seconds=total_seconds, fallback_seconds=1.0)
            for index, (address, connect_seconds) in enumerate(attempts):
                request = httpx.Request(
                    "GET",
                    url.copy_with(host=str(address)),
                    headers=request_headers,
                    extensions={
                        "sni_hostname": url.host,
                        "timeout": httpx.Timeout(total_seconds, connect=connect_seconds).as_dict(),
                    },
                )
                try:
                    response = await transport.handle_async_request(request)
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    # Only a failed connect moves on: the headers are sent after it.
                    if index == len(attempts) - 1:
                        raise
                    continue
                try:
                    return await _read_capped(response, max_body_bytes)
                finally:
                    await response.aclose()
    except TimeoutError as exc:
        raise EgressTimeout("the request outlasted its deadline") from exc
    finally:
        if transport is not _TRANSPORT:
            await transport.aclose()
    raise EgressHostNotPublic(url.host)
