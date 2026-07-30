"""GET /connectors/oauth/{engine}/credentials must refuse non-loopback callers (ENG-868).

The endpoint returns the raw OAuth client_secret for builtin engines — the
same class of secret as settings reveal-key and /raw, so it takes the same
loopback guard (ENG-457). Only the Electron main process calls it, always
over 127.0.0.1; hosted-web builds never do (ENG-817).
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _request(host):
    """Minimal stand-in for a Starlette Request — only `.client.host` matters."""
    client = SimpleNamespace(host=host) if host is not None else None
    return SimpleNamespace(client=client)


@pytest.mark.parametrize("host", ["10.0.0.5", "0.0.0.0", "192.168.1.10", "203.0.113.7", "", None])
def test_credentials_rejects_non_loopback(host):
    from cowork.api.v1.endpoints.connectors.oauth import get_oauth_credentials

    # The guard is the first statement, so a non-loopback caller is rejected
    # before the engine lookup or any settings (secrets) are read.
    with pytest.raises(HTTPException) as exc:
        get_oauth_credentials("gmail", request=_request(host))
    assert exc.value.status_code == 403


def test_credentials_admits_loopback():
    from cowork.api.v1.endpoints.connectors.oauth import get_oauth_credentials

    # An unknown engine 404s only past the guard — proves loopback callers get
    # through without needing configured OAuth credentials in the test env.
    for host in ("127.0.0.1", "::1"):
        with pytest.raises(HTTPException) as exc:
            get_oauth_credentials("not-an-engine", request=_request(host))
        assert exc.value.status_code == 404
