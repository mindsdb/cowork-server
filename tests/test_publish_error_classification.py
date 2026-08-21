"""publish_artifact's failure-message classification (ENG-1547 review follow-up).

Two things the ad-hoc string-matching used to get wrong:

1. The `/publish` endpoints mapped a RuntimeError to 503 whenever its message
   happened to contain the substring "unavailable" — meant for the two local
   "X is unavailable" import guards, but upstream error text can contain that
   word too (an HTTP 503 reason phrase, or the timeout advice string), which
   silently flipped an ordinary upstream failure from 502 to 503.
2. `publish_artifact` wrapped anton.publisher.publish()'s entire call
   (including local file/zip/vault work that happens before any network
   request) in a handler that always framed the failure as "Connection
   failed", misattributing local errors to the network.
"""
from __future__ import annotations

import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import publish as publish_ep
from cowork.services import publish
from cowork.services.publish import PublisherUnavailable


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")
    return TestClient(app)


def _patch_context():
    """The endpoint resolves the artifact + credential before publishing, so the
    exception-mapping tests below must get past that step to reach `_publish`."""
    return patch.object(
        publish_ep, "_desktop_context",
        lambda raw: (Path(raw), Path(raw).parent, "key", "https://4nton.ai"),
    )


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("Publishing failed: connection reset"),
        RuntimeError(
            "Publishing failed. Connection failed because the request timed out. "
            "Common reasons: the server is slow or unavailable, the URL is wrong, "
            "or there is a network path issue."
        ),
        RuntimeError("Publishing failed. Connection failed (HTTP 503: Service Unavailable). "
                     "The server returned an error. Common reasons: server-side failure or a temporary outage."),
    ],
)
def test_upstream_runtime_error_always_maps_to_502(exc):
    """An upstream/transport failure is a 502 regardless of what words end up
    in its message — including the two cases whose text happens to contain
    "unavailable" (a timeout's advice string, and a real HTTP 503 reason
    phrase) that used to flip this to 503 via substring matching."""
    with _patch_context(), patch.object(publish_ep, "_publish", side_effect=exc):
        res = _client().post("/api/v1/publish/", json={"path": "/some/art"})
    assert res.status_code == 502
    assert res.json()["detail"] == str(exc)


def test_publisher_unavailable_maps_to_503():
    """The local "a publish dependency didn't import" case is still a 503,
    now via the exception type instead of message sniffing."""
    exc = PublisherUnavailable("Anton publisher is unavailable")
    with _patch_context(), patch.object(publish_ep, "_publish", side_effect=exc):
        res = _client().post("/api/v1/publish/", json={"path": "/some/art"})
    assert res.status_code == 503
    assert res.json()["detail"] == "Anton publisher is unavailable"


def _wire_publish(monkeypatch, tmp_path, target: Path, key: str, *, publish_side_effect):
    monkeypatch.setenv("ANTON_COWORK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(
        publish, "get_user_settings",
        lambda: SimpleNamespace(
            minds_api_key="key",
            minds_url="https://api.mindshub.ai/v1",
            openai_base_url="",
            openai_api_key=None,
            publish_url="",
        ),
    )
    monkeypatch.setattr(
        publish, "get_app_settings",
        lambda: SimpleNamespace(connector=SimpleNamespace(vault_dir=str(tmp_path / "vault"))),
    )
    monkeypatch.setattr(publish, "resolve_artifact_path", lambda raw, allow_dir=True: target)
    monkeypatch.setattr(
        publish, "_resolve_publish_target",
        lambda a, container_dirs=None: (target, target.parent, key, False),
    )

    def fake_publish(src, **kw):
        raise publish_side_effect

    monkeypatch.setattr("anton.publisher.publish", fake_publish)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", lambda *a, **k: object())


def test_local_failure_before_any_request_is_not_framed_as_connection_failure(tmp_path, monkeypatch):
    """anton.publisher.publish() does local file/zip/vault work before it ever
    opens a socket. A failure from that stage (simulated here — the real
    OSError/FileNotFoundError would come from _zip_html/_zip_fullstack) must
    not be told to the user as a network problem."""
    art = tmp_path / "art"
    art.mkdir()
    page = art / "dashboard.html"
    page.write_text("<h1>dash</h1>")

    local_exc = FileNotFoundError(str(tmp_path / "art" / "some-referenced-asset.js"))
    _wire_publish(monkeypatch, tmp_path, page, "dashboard.html", publish_side_effect=local_exc)

    with pytest.raises(RuntimeError) as exc_info:
        publish.publish_artifact(page, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai")

    msg = str(exc_info.value)
    assert "Connection failed" not in msg
    assert str(tmp_path / "art" / "some-referenced-asset.js") in msg


def test_real_gateway_timeout_is_still_framed_as_connection_failure(tmp_path, monkeypatch):
    """The actual transport failure (ENG-1580: API Gateway's 504 while a
    fullstack artifact's deps are still installing) keeps its
    connection-oriented message — only local, pre-request failures lose it."""
    art = tmp_path / "art"
    art.mkdir()
    page = art / "dashboard.html"
    page.write_text("<h1>dash</h1>")

    upstream_exc = urllib.error.HTTPError("https://api.mindshub.ai/v1/upload", 504, "Gateway Timeout", {}, None)
    _wire_publish(monkeypatch, tmp_path, page, "dashboard.html", publish_side_effect=upstream_exc)

    with pytest.raises(RuntimeError) as exc_info:
        publish.publish_artifact(page, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai")

    msg = str(exc_info.value)
    assert "Connection failed" in msg
    assert "504" in msg
