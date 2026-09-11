"""GET /connectors/oauth/{engine}/credentials must refuse non-loopback callers (ENG-868).

The endpoint returns the raw OAuth client_secret for builtin engines — the
same class of secret as settings reveal-key and /raw, so it takes the same
loopback guard (ENG-457). Only the Electron main process calls it, always
over 127.0.0.1; hosted-web builds never do (ENG-817).
"""


def _oauth_client(*, local: bool):
    # require_local is a declared dependency now (not called in the handler
    # body), so this goes through the real router — a bare handler call would
    # no longer prove anything is enforced.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cowork.api.v1.router import api_router

    app = FastAPI()
    app.include_router(api_router)
    if local:
        return TestClient(app, client=("127.0.0.1", 50000))
    return TestClient(app)


def test_credentials_rejects_non_loopback():
    # The guard runs before the engine lookup or any settings (secrets) are
    # read, so a non-loopback caller never reaches either.
    resp = _oauth_client(local=False).get("/api/v1/connectors/oauth/gmail/credentials")
    assert resp.status_code == 403


def test_credentials_admits_loopback():
    # An unknown engine 404s only past the guard — proves loopback callers get
    # through without needing configured OAuth credentials in the test env.
    resp = _oauth_client(local=True).get("/api/v1/connectors/oauth/not-an-engine/credentials")
    assert resp.status_code == 404
