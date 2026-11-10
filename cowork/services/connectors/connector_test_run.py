"""Run a custom connector's test request and judge what came back.

The request is data the connector carries (``ConnectorTestRequest``); the
member's values are filled in here and never reach a prompt, a log or an
error. Every host is resolved and judged before the first request is sent, and
each request dials only the addresses that were judged, so a name cannot pass
the check and then rebind. A private address goes out only when the operator
listed it or the member allowed it for this test.

What comes back is judged by floor checks that refuse what is plainly not data
(an error, a token, an echo of a secret). What passes them is not saved: the
caller shows the member a masked preview and saves only on their confirmation.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import html
import ipaddress
import json
import os
import posixpath
import re
import socket
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, quote_plus, unquote, urlsplit

import httpx

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.connectors.egress import (
    connection_attempts,
    embedded_ipv4,
    is_public_address,
)

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

ALLOWLIST_ENV = "COWORK_CONNECTOR_TEST_PRIVATE_ALLOWLIST"
_REQUEST_SECONDS = 10.0
_RESOLVE_SECONDS = 5.0
_BODY_CAP = 256 * 1024
_PREVIEW_ITEM_CHARS = 500
#: Values shorter than this are not masked: "us" or "1" would blank real data.
_MASK_MIN = 4
#: Secrets shorter than this are not searched for: they match too much data.
_ECHO_MIN = 6
_TOKEN_KEYS = frozenset({"access_token", "id_token", "token_type"})
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_RESOLVER: Callable[..., list[tuple]] = socket.getaddrinfo

#: Ranges a member may allow for one test. Everything else that is not public
#: is refused outright and never offered.
_OFFERABLE = tuple(ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "127.0.0.0/8", "::1/128", "fc00::/7",
))
#: Cloud metadata addresses inside otherwise offerable ranges.
_METADATA = frozenset(ipaddress.ip_address(a) for a in ("fd00:ec2::254", "100.100.100.200"))


class ConnectorTestError(Exception):
    """A test that cannot be judged as passing. The message is safe to show:
    it names a step, a status code or a field, never a value or a URL."""


@dataclass(frozen=True)
class PrivateHost:
    """A host the member must allow before the test may connect to it."""

    step: str
    method: str
    scheme: str
    host: str
    port: int
    path: str
    addresses: tuple[str, ...]

    @property
    def key(self) -> str:
        """What one approval covers: this step, its method, scheme, host and port."""
        return approval_key(self.step, self.method, self.scheme, self.host, self.port)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "method": self.method, "scheme": self.scheme, "host": self.host,
            "host_ascii": _ascii_host(self.host), "port": self.port, "path": self.path,
            "addresses": list(self.addresses), "this_computer": all(_is_loopback(a) for a in self.addresses),
        }


@dataclass(frozen=True)
class ConnectorTestPreview:
    """What the member sees before confirming, with every value masked."""

    status: int
    token_host: str | None
    data_host: str
    content_type: str
    size: int
    over_cap: bool
    item_count: int | None
    keys: list[str]
    first_item: str | None
    masked_count: int

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class ConnectorTestOutcome:
    """The runner's answer: a preview to confirm, or a host to allow first."""

    preview: ConnectorTestPreview | None = None
    private_host: PrivateHost | None = None


@functools.cache
def private_allowlist() -> tuple[frozenset[str], tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]]:
    """The operator's private hosts and ranges, from the process environment.

    Read once and only from ``os.environ``: never from ``<COWORK_HOME>/.env``
    or settings, which code the agent runs can write.
    """
    hosts: set[str] = set()
    networks = []
    for entry in (os.environ.get(ALLOWLIST_ENV) or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            hosts.add(_normal_host(entry))
    return frozenset(hosts), tuple(networks)


def approval_key(step: str, method: str, scheme: str, host: str, port: int) -> str:
    """The key an approval is stored under; a different step, method, scheme,
    host or port is a different question."""
    return f"{step} {method} {scheme}://{_normal_host(host)}:{port}"


def _normal_host(host: str) -> str:
    return host.strip().rstrip(".").lower()


def _ascii_host(host: str) -> str:
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def _judged(address: IPAddress) -> IPAddress:
    """The address a connection really reaches: the IPv4 one a translator dials."""
    return embedded_ipv4(address) or address


def _is_loopback(address: str) -> bool:
    return _judged(ipaddress.ip_address(address)).is_loopback


def _offerable(address: IPAddress) -> bool:
    judged = _judged(address)
    return judged not in _METADATA and address not in _METADATA and any(judged in net for net in _OFFERABLE)


def _reaches_this_server(address: IPAddress, port: int) -> bool:
    """Whether a connection to ``address:port`` lands on this server's listener.

    The unspecified address reaches a listener bound to loopback, so it counts.
    """
    judged = _judged(address)
    return port == get_app_settings().port and (judged.is_loopback or judged.is_unspecified)


async def _resolve(host: str, resolver: Callable[..., list[tuple]]) -> tuple[IPAddress, ...]:
    try:
        return (ipaddress.ip_address(host),)
    except ValueError:
        pass
    try:
        records = await asyncio.wait_for(
            asyncio.to_thread(resolver, host, 443, type=socket.SOCK_STREAM), timeout=_RESOLVE_SECONDS,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        raise ConnectorTestError(f"Cowork could not resolve {host}.") from exc
    addresses = []
    for record in records:
        try:
            addresses.append(ipaddress.ip_address(record[4][0]))
        except (IndexError, ValueError):
            continue
    if not addresses:
        raise ConnectorTestError(f"Cowork could not resolve {host}.")
    return tuple(dict.fromkeys(addresses))


def _fill(text: str, values: Mapping[str, str], *, encode: Callable[[str], str] | None = None) -> str:
    def sub(match: re.Match) -> str:
        value = values[match.group(1)]
        return encode(value) if encode else value
    return _PLACEHOLDER.sub(sub, text)


def _fill_json(value: Any, values: Mapping[str, str]) -> Any:
    """Fill placeholders in a parsed JSON body's leaf strings, never in its text."""
    if isinstance(value, str):
        return _fill(value, values)
    if isinstance(value, dict):
        return {k: _fill_json(v, values) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_json(v, values) for v in value]
    return value


def _fill_url(url: str, values: Mapping[str, str]) -> httpx.URL:
    """The step's URL with path and query values percent-encoded, parsed once."""
    return httpx.URL(_fill(url, values, encode=lambda v: quote(v, safe="")))


def placeholder_names(step: Mapping[str, Any]) -> set[str]:
    """The ``{{name}}`` placeholders one test step uses."""
    texts = [step.get("url") or "", *(step.get("headers") or {}).values(), *(step.get("basic_auth") or [])]
    texts += list((step.get("form") or {}).values())
    texts += _json_leaves(step.get("json_body"))
    return {name for text in texts for name in _PLACEHOLDER.findall(str(text))}


def request_hosts(test_request: Mapping[str, Any]) -> set[str]:
    """The hosts a test request's steps are sent to, as written in the form."""
    hosts = set()
    for name in ("token", "data"):
        step = test_request.get(name)
        if step:
            host = urlsplit(_PLACEHOLDER.sub("x", str(step.get("url") or ""))).hostname
            if host:
                hosts.add(_normal_host(host))
    return hosts


def _json_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _json_leaves(v)]
    if isinstance(value, list):
        return [s for v in value for s in _json_leaves(v)]
    return []


def _endpoint(url: httpx.URL) -> tuple[str, str, int, str]:
    """A URL reduced to what decides where it lands, for comparing two steps."""
    path = re.sub(r"/+", "/", "/" + unquote(url.path or "/"))
    path = posixpath.normpath(path).rstrip("/").lower() or "/"
    port = url.port or (443 if url.scheme == "https" else 80)
    return url.scheme, _normal_host(url.host), port, path


@dataclass
class _Step:
    name: str
    method: str
    url: httpx.URL
    headers: dict[str, str]
    auth: tuple[str, str] | None
    form: dict[str, str] | None
    json_body: Any


def _build_step(name: str, step: Mapping[str, Any], values: Mapping[str, str]) -> _Step:
    headers = {k: _fill(v, values) for k, v in (step.get("headers") or {}).items()}
    for key, value in headers.items():
        if "\r" in value or "\n" in value:
            raise ConnectorTestError(f"The {name} step's {key} header holds a line break once filled in.")
    auth = None
    if step.get("basic_auth"):
        user, password = (_fill(item, values) for item in step["basic_auth"])
        auth = (user, password)
    form = {k: _fill(v, values) for k, v in step["form"].items()} if step.get("form") is not None else None
    body = _fill_json(step.get("json_body"), values) if step.get("json_body") is not None else None
    return _Step(name, step["method"], _fill_url(step["url"], values), headers, auth, form, body)


def _encodings(value: str) -> set[str]:
    """``value`` as it may appear echoed in a response body."""
    out = {value, quote(value), quote(value, safe=""), quote_plus(value), html.escape(value), html.escape(value, quote=False)}
    out |= {quote(quote(value, safe=""), safe="")}
    out |= {re.sub(r"%[0-9A-F]{2}", lambda m: m.group().lower(), v) for v in list(out)}
    raw = value.encode()
    out |= {raw.hex(), raw.hex().upper()}
    for k in range(3):
        encoded = base64.b64encode(b"\0" * k + raw).decode()
        bits = 8 * len(raw)
        first = next(i for i in range(len(encoded)) if 6 * i >= 8 * k)
        last = max((i for i in range(len(encoded)) if 6 * i + 6 <= 8 * k + bits), default=first - 1) + 1
        stable = encoded[first:last]
        if len(stable) >= _ECHO_MIN:
            out |= {stable, stable.replace("+", "-").replace("/", "_")}
    return {v for v in out if v}


class _Guard:
    """The values a response may not echo, and the wider set the preview masks."""

    def __init__(self, refused: Iterable[str], masked: Iterable[str]) -> None:
        self.refused = [e for v in refused if len(v) >= _ECHO_MIN for e in _encodings(v)]
        self.masked = sorted(
            {e for v in [*masked, *refused] if len(v) >= _MASK_MIN for e in _encodings(v)}, key=len, reverse=True,
        )

    def echoes(self, text: str) -> bool:
        return any(e in text for e in self.refused)

    def hides(self, text: str) -> bool:
        return any(e in text for e in self.masked)


def _walk(value: Any) -> Iterable[str]:
    """Every key and leaf string of a parsed JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def _has_token_object(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(_TOKEN_KEYS & set(value)) or any(_has_token_object(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_token_object(v) for v in value)
    return False


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {}) or value == [{}]


def _mask(value: Any, guard: _Guard, counter: list[int]) -> Any:
    if isinstance(value, str):
        if guard.hides(value):
            counter[0] += 1
            return "***"
        return value
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            name = str(key)
            if guard.hides(name):
                counter[0] += 1
                name = f"***{counter[0]}"
            out[name] = _mask(item, guard, counter)
        return out
    if isinstance(value, list):
        return [_mask(v, guard, counter) for v in value]
    return value


def _item_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        lists = [len(v) for v in value.values() if isinstance(v, list)]
        return max(lists) if lists else None
    return None


def _first_item(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, dict):
        lists = [v for v in value.values() if isinstance(v, list) and v]
        return max(lists, key=len)[0] if lists else value
    return value


def judge(
    *, status: int, body: bytes, over_cap: bool, content_type: str, guard: _Guard,
    token_host: str | None, data_host: str,
) -> ConnectorTestPreview:
    """Apply the floor checks to the data step's answer and build its preview.

    Raises:
        ConnectorTestError: when the answer is plainly not data.
    """
    if not 200 <= status < 300:
        raise ConnectorTestError(f"The data step got HTTP {status}.")
    text = body.decode("utf-8", errors="replace")
    if not text.strip():
        raise ConnectorTestError("The data step got an empty answer.")
    if over_cap:
        if guard.echoes(text):
            raise ConnectorTestError("The data step's answer repeats a credential, so it was not shown.")
        return ConnectorTestPreview(status, token_host, data_host, content_type, len(body), True, None, [], None, 0)
    try:
        parsed: Any = json.loads(text)
        is_json = True
    except ValueError:
        parsed, is_json = text, False
    if is_json:
        if _is_empty(parsed):
            raise ConnectorTestError("The data step's answer holds no data.")
        if isinstance(parsed, dict) and (parsed.get("error") or parsed.get("errors")):
            raise ConnectorTestError("The data step's answer is an error.")
        if _has_token_object(parsed):
            raise ConnectorTestError("The data step's answer is a token, not data.")
        if any(guard.echoes(s) for s in _walk(parsed)):
            raise ConnectorTestError("The data step's answer repeats a credential, so it was not shown.")
    elif guard.echoes(text):
        raise ConnectorTestError("The data step's answer repeats a credential, so it was not shown.")
    counter = [0]
    masked = _mask(parsed, guard, counter)
    keys = list(masked) if isinstance(masked, dict) else []
    item = _first_item(masked)
    shown = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str)
    return ConnectorTestPreview(
        status, token_host, data_host, content_type, len(body), False, _item_count(masked), keys,
        None if item is None else shown[:_PREVIEW_ITEM_CHARS], counter[0],
    )


async def _judge_host(
    step: _Step, approved: Mapping[str, tuple[str, ...]], resolver: Callable[..., list[tuple]],
) -> tuple[tuple[IPAddress, ...], PrivateHost | None]:
    """The addresses to dial for ``step``, or the host the member must allow.

    Raises:
        ConnectorTestError: when any address is one no test may reach.
    """
    host = _normal_host(step.url.host)
    port = step.url.port or (443 if step.url.scheme == "https" else 80)
    key = approval_key(step.name, step.method, step.url.scheme, host, port)
    if key in approved:
        addresses = tuple(ipaddress.ip_address(a) for a in approved[key])
        allowed = True
    else:
        addresses = await _resolve(host, resolver)
        hosts, networks = private_allowlist()
        allowed = host in hosts or all(any(_judged(a) in n or a in n for n in networks) for a in addresses)
    for address in addresses:
        if _reaches_this_server(address, port):
            raise ConnectorTestError(f"The {step.name} step points at Cowork itself.")
        if not is_public_address(address) and not _offerable(address):
            raise ConnectorTestError(f"The {step.name} step points at an address no test may reach ({address}).")
    private = [a for a in addresses if not is_public_address(a)]
    if private and not allowed:
        return addresses, PrivateHost(
            step.name, step.method, step.url.scheme, host, port, step.url.path or "/",
            tuple(str(a) for a in addresses),
        )
    # http only to this computer, once it is allowed; anywhere else it would
    # carry the member's credential in the clear.
    if step.url.scheme == "http" and not (private and all(_judged(a).is_loopback for a in addresses)):
        raise ConnectorTestError(f"The {step.name} step uses http; use https.")
    return addresses, None


async def _send(
    step: _Step, addresses: tuple[IPAddress, ...], transport: httpx.AsyncBaseTransport | None,
) -> tuple[int, bytes, bool, str]:
    """Send ``step`` to its vetted addresses: status, body (capped), over-cap, content type."""
    host = step.url.host
    # Identity only: the cap counts what is read, and a compressed answer
    # could expand far past it in one decoded chunk.
    headers = {**step.headers, "Host": step.url.netloc.decode("ascii"), "Accept-Encoding": "identity"}
    kwargs: dict[str, Any] = {"headers": headers}
    if step.form is not None:
        kwargs["data"] = step.form
    elif step.json_body is not None:
        kwargs["json"] = step.json_body
    # trust_env stays off: an environment proxy would carry the request to a
    # host nobody vetted. No redirects, for the same reason.
    async with httpx.AsyncClient(follow_redirects=False, trust_env=False, transport=transport) as client:
        attempts = connection_attempts(list(addresses), total_seconds=_REQUEST_SECONDS)
        for index, (address, connect_seconds) in enumerate(attempts):
            request = client.build_request(
                step.method, step.url.copy_with(host=str(address)),
                extensions={"sni_hostname": host},
                timeout=httpx.Timeout(_REQUEST_SECONDS, connect=connect_seconds),
                **kwargs,
            )
            try:
                response = await client.send(request, stream=True, auth=step.auth or httpx.USE_CLIENT_DEFAULT)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if index < len(attempts) - 1:
                    continue
                raise ConnectorTestError(f"The {step.name} step could not connect.") from exc
            except httpx.HTTPError as exc:
                raise ConnectorTestError(f"The {step.name} step failed before an answer came back.") from exc
            if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                await response.aclose()
                raise ConnectorTestError(f"The {step.name} step's answer came back compressed; Cowork reads plain answers only.")
            try:
                chunks, size = [], 0
                # Identity was checked above, so these bytes are the bytes read.
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > _BODY_CAP:
                        break
            except httpx.HTTPError as exc:
                raise ConnectorTestError(f"The {step.name} step's answer was cut off.") from exc
            finally:
                await response.aclose()
            body = b"".join(chunks)
            return response.status_code, body[:_BODY_CAP], size > _BODY_CAP, response.headers.get("content-type", "")
    raise ConnectorTestError(f"The {step.name} step could not connect.")


def _token_from(status: int, body: bytes, token_field: str) -> str:
    if not 200 <= status < 300:
        raise ConnectorTestError(f"The token step got HTTP {status}.")
    try:
        token = json.loads(body).get(token_field)
    except (ValueError, AttributeError) as exc:
        raise ConnectorTestError("The token step's answer is not a JSON object.") from exc
    if not isinstance(token, str) or not token:
        raise ConnectorTestError(f"The token step's answer has no {token_field}.")
    return token


async def run_test_request(
    test_request: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    secret_names: Iterable[str],
    approved: Mapping[str, tuple[str, ...]] | None = None,
    resolver: Callable[..., list[tuple]] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ConnectorTestOutcome:
    """Run a connector's test request with the member's ``values``.

    Args:
        test_request: The ``ConnectorTestRequest`` as a dict.
        values: The member's submitted values by field name.
        secret_names: Fields whose values a response may never echo: the
            form's secret fields and, on a follow-up step, every saved field.
        approved: The member's approvals (``PrivateHost.key``), each with the
            exact addresses to dial.
        resolver, transport: Seams for tests.

    Returns:
        A preview to confirm, or the first private host to allow.

    Raises:
        ConnectorTestError: when the test fails or its answer is not data.
    """
    resolver = resolver or _RESOLVER
    approved = dict(approved or {})
    filled = {k: str(v) for k, v in values.items() if v is not None and str(v) != ""}
    token_spec = test_request.get("token")
    data_spec = test_request["data"]
    for name, spec in (("token", token_spec), ("data", data_spec)):
        if not spec:
            continue
        for placeholder in placeholder_names(spec) - ({"token"} if name == "data" and token_spec else set()):
            if placeholder not in filled:
                raise ConnectorTestError(f"The {name} step needs {placeholder}, which was left empty.")

    pending = dict(filled)
    if token_spec:
        pending["token"] = "{{token}}"
    steps = [_build_step("token", token_spec, filled)] if token_spec else []
    data_probe = _build_step("data", data_spec, pending)
    outcome = ConnectorTestOutcome()
    dial: dict[str, tuple[IPAddress, ...]] = {}
    for step in [*steps, data_probe]:
        addresses, private = await _judge_host(step, approved, resolver)
        if private is not None:
            outcome.private_host = private
            return outcome
        dial[step.name] = addresses

    refused = {filled[n] for n in secret_names if n in filled}
    # The password half of basic_auth only: the user is often an account name that data repeats.
    for spec in (token_spec or {}, data_spec):
        for item in (spec.get("basic_auth") or [])[1:]:
            refused |= {filled[n] for n in _PLACEHOLDER.findall(item) if n in filled}
    for key, value in (data_spec.get("headers") or {}).items():
        if key.lower() == "authorization":
            refused |= {filled[n] for n in _PLACEHOLDER.findall(value) if n in filled}
    for value in ((token_spec or {}).get("form") or {}).values():
        refused |= {filled[n] for n in _PLACEHOLDER.findall(value) if n in filled}
    for value in _json_leaves((token_spec or {}).get("json_body")):
        refused |= {filled[n] for n in _PLACEHOLDER.findall(value) if n in filled}

    token_host = None
    if token_spec:
        token_step = steps[0]
        status, body, _, _ = await _send(token_step, dial["token"], transport)
        token = _token_from(status, body, token_spec["token_field"])
        refused.add(token)
        filled["token"] = token
        token_host = token_step.url.host
    data_step = _build_step("data", data_spec, filled)
    if token_spec and _endpoint(data_step.url) == _endpoint(steps[0].url):
        raise ConnectorTestError("The data step asks the token URL again; it must fetch data.")
    status, body, over_cap, content_type = await _send(data_step, dial["data"], transport)
    guard = _Guard(refused, filled.values())
    outcome.preview = judge(
        status=status, body=body, over_cap=over_cap, content_type=content_type, guard=guard,
        token_host=token_host, data_host=data_step.url.host,
    )
    return outcome
