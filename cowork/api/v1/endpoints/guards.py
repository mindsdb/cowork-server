"""Request guards shared by endpoints that return unmasked secrets."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from fastapi import HTTPException, Request, status


def _trusted_loopback_host(value: str) -> bool:
    parsed = urlparse(f"//{value}")
    hostname = (parsed.hostname or "").casefold()
    if hostname in {"localhost", "testserver"}:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def require_local(request: Request) -> None:
    """Reject the request unless it originates from loopback.

    Endpoints that return unmasked secrets (settings reveal-key and /raw,
    OAuth client credentials) are restricted to loopback so a network-exposed
    deployment (e.g. the self-host compose that binds 0.0.0.0) can't hand
    secrets to a remote peer (ENG-457, ENG-868). The desktop sidecar, UI, and
    Electron main process all talk over 127.0.0.1, so the legitimate flows are
    unaffected; hosted-web builds call `/settings/raw` on boot and treat the 403
    as expected (ENG-817).
    """
    client = request.client.host if request.client else ""
    if client not in {"127.0.0.1", "::1"}:
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
