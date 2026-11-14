"""Request guards shared by endpoints that return unmasked secrets."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from fastapi import HTTPException, Request, status

_TRUSTED_LOOPBACK_HOSTS = frozenset({"localhost"})


def _is_loopback_address(value: str | None) -> bool:
    """Whether an address literal, or a host name trusted as local, is loopback.

    Unwraps an IPv4-mapped IPv6 address first. A dual-stack socket reports an
    IPv4 loopback connection as ``::ffff:127.0.0.1``, and ``is_loopback`` does
    not answer True for that form on every Python this server runs on.
    """
    name = (value or "").strip().strip("[]").casefold()
    if name in _TRUSTED_LOOPBACK_HOSTS:
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def _trusted_loopback_host(value: str) -> bool:
    return _is_loopback_address(urlparse(f"//{value}").hostname)


def require_local(request: Request) -> None:
    """Reject the request unless it comes from this machine.

    For routes only the desktop sidecar, its renderer and the Electron main
    process call, all over 127.0.0.1: the ones that return unmasked secrets
    (settings reveal-key and /raw, OAuth client credentials) or write
    credentials. A network-exposed deployment, such as a self-host compose
    that binds 0.0.0.0, must not answer them to a remote peer. Hosted-web
    builds call ``/settings/raw`` on boot and treat the 403 as expected.
    """
    client = request.client.host if request.client else ""
    if client not in {"127.0.0.1", "::1"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="local requests only")

    # The peer address alone is forgeable wherever uvicorn trusts a proxy: the
    # image passes --forwarded-allow-ips '*', so an X-Forwarded-For from any
    # peer rewrites request.client. scope["server"] is the local address of
    # the accepted socket. The proxy middleware never rewrites it and no caller
    # chooses it, so a request that landed on a loopback socket really came
    # from this machine.
    server = request.scope.get("server") or ()
    if not _is_loopback_address(server[0] if server else None):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="local requests only")

    # A loopback TCP peer is not sufficient in a browser: DNS rebinding can
    # make an attacker-controlled hostname resolve to 127.0.0.1. Browsers set
    # Host themselves, so requiring an actual loopback host closes that path.
    headers = getattr(request, "headers", {})
    host = str(headers.get("host") or "")
    if host and not _trusted_loopback_host(host):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="local host required")

    # CORS protects response visibility; validating Origin here also protects
    # state-changing local endpoints from cross-site request forgery.
    origin = str(headers.get("origin") or "").rstrip("/")
    if origin:
        from cowork.common.settings.app_settings import get_app_settings

        allowed = {item.rstrip("/") for item in get_app_settings().allowed_origins}
        if origin not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="trusted local origin required")


def require_local_in_desktop_mode(request: Request) -> None:
    """``require_local``, but only where org mode is not already the boundary.

    For a route that is destructive on desktop and a deliberate no-op in org
    mode (``/settings/logout``). Applying ``require_local`` outright would 403
    the org-mode call that is documented to answer 200 and do nothing, and
    applying nothing leaves the desktop half reachable by any network peer of
    a self-host deployment bound to 0.0.0.0 — and, being a simple POST, by any
    page the user happens to have open.

    Declared rather than called in the handler so the route walker can see it:
    a guard invoked from inside a handler body is exactly what the walker
    cannot tell apart from no guard at all.
    """
    from cowork.common.settings.app_settings import get_app_settings

    if get_app_settings().tenancy_mode == "org":
        return
    require_local(request)


def require_local_tenancy() -> None:
    """Reject the request in org mode.

    For desktop-only surfaces that read or write deployment-global state (the
    ``.env`` dotenv sync): they carry no tenant scope and their writes land in
    global rows every org falls back to, so loopback alone isn't enough of a
    boundary once the deployment serves many orgs.

    The refusal answers 403, not 501. Turning a caller away at a tenant
    boundary is an authorization decision, and the artifact routes that
    decline to serve an org deployment already answer 403. A 501 puts a
    deliberate refusal in the 5xx class, where nginx books it as a server
    fault and the ingress alerts page on it.
    """
    from cowork.common.settings.app_settings import get_app_settings

    if get_app_settings().tenancy_mode == "org":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not available in org deployments",
        )


def refuse_code_mode_in_sandbox() -> None:
    """Reject a Code mode request when agent code runs in the sandbox.

    Code mode runs git, project commands and Codex with its terminals in this
    server's own container, outside the Landlock launcher, so under
    ``COWORK_CODE_SANDBOX=landlock`` every Code mode route answers 403. Declared
    on the coding routers rather than called in a handler, so a route added
    there inherits it.
    """
    from cowork.code_sandbox import NO_CODE_MODE_DETAIL, code_sandbox_on

    if code_sandbox_on():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=NO_CODE_MODE_DETAIL)
