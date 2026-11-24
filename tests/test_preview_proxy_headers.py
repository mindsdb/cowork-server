"""What the fullstack preview proxy (cowork.services.preview_proxy) forwards
to an artifact's backend.

On the desktop, desktop main puts the window's bearer token on the preview
frame's requests to this server. The backend runs the agent's code, so the
proxy forwards none of the app's own credentials to it: not the bearer token,
not the window proof and not the app secret.

Every secret here is a fixed fake value.
"""
from __future__ import annotations

import httpx
import pytest
from starlette.requests import Request

from cowork import app_secret
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services import artifacts, preview_proxy

TOKEN = "preview-proxy-headers-token"


@pytest.fixture
def backend(tmp_path, monkeypatch):
    """A stub backend behind a mounted preview. Returns the requests it got."""
    received: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(preview_proxy, "_org_mode", lambda: False)
    monkeypatch.setattr(
        preview_proxy, "get_proxy_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(answer))
    )
    root = tmp_path / "artifact_root"
    root.mkdir()
    (root / "metadata.json").write_text('{"port": 4242}', encoding="utf-8")
    monkeypatch.setitem(artifacts._PREVIEW_MOUNTS, TOKEN, root)
    # The mount sits outside every project here; the scope check has its own
    # tests (tests/test_personal_partition_artifacts.py).
    monkeypatch.setattr(
        preview_proxy, "preview_mount_for", lambda token, *, scope: artifacts._PREVIEW_MOUNTS.get(token)
    )
    return received


def _request(*, headers: dict[str, str]) -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": f"/api/v1/artifacts/proxy/{TOKEN}/api/data",
            "headers": [(name.lower().encode(), value.encode()) for name, value in headers.items()],
            "query_string": b"",
        },
        receive,
    )


async def test_the_backend_gets_none_of_the_apps_credentials(backend):
    request = _request(headers={
        "Authorization": "Bearer " + "cd" * 32,
        app_secret.WINDOW_PROOF_HEADER: "ef" * 32,
        app_secret.HEADER: "ab" * 32,
        "X-Artifact-Header": "kept",
        "Accept": "application/json",
    })

    response = await preview_proxy.proxy_artifact_request(TOKEN, "api/data", request, scope=LOCAL_SCOPE)

    assert response.status_code == 200
    [forwarded] = backend
    assert str(forwarded.url) == "http://127.0.0.1:4242/api/data"
    assert "authorization" not in forwarded.headers
    assert app_secret.WINDOW_PROOF_HEADER not in forwarded.headers
    assert app_secret.HEADER not in forwarded.headers
    # An ordinary header still reaches the backend, as does the rewritten host.
    assert forwarded.headers["x-artifact-header"] == "kept"
    assert forwarded.headers["accept"] == "application/json"
    assert forwarded.headers["host"] == "127.0.0.1:4242"
