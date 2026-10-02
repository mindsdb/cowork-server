"""A full-stack artifact is refused before upload unless MindsHub hosts it.

`anton.publisher.publish` sends a full-stack artifact's datasource credentials,
in plaintext, in the upload body. A publishing service built from the
specification (the one `ANTON_PUBLISH_URL` points at on a self-hosted
deployment) stores static bundles only and answers 400, so the credentials
would reach it for nothing. These tests pin that the refusal happens in
cowork-server before the publisher is called or the vault is read, on every
entry point that publishes: the service function, the HTTP route, and the
agent's publish tool.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import publish as publish_ep
from cowork.harnesses.anton_harness import tools as htools
from cowork.services import publish
from cowork.services.providers import is_mindshub_publish_url

SELF_HOSTED_PUBLISH_URL = "http://publisher-api:8081"


def _artifact(base: Path, slug: str, *, artifact_type: str, files: dict[str, str]) -> Path:
    folder = base / slug
    for rel, body in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (folder / "metadata.json").write_text(
        json.dumps({
            "slug": slug,
            "name": slug,
            "type": artifact_type,
            "datasources": [{"engine": "postgres", "name": "scm"}],
        })
    )
    return folder


@pytest.fixture
def artifacts_base(tmp_path: Path) -> Path:
    base = tmp_path / "proj" / ".anton" / "artifacts"
    base.mkdir(parents=True)
    return base


@pytest.fixture
def fullstack(artifacts_base: Path) -> Path:
    return _artifact(
        artifacts_base,
        "planner-app",
        artifact_type="fullstack-stateless-app",
        files={
            "backend.py": "from fastapi import FastAPI\napp = FastAPI()\n",
            "requirements.txt": "fastapi\nmangum\nuvicorn\n",
            "static/index.html": "<html>app</html>",
        },
    )


@pytest.fixture
def static_page(artifacts_base: Path) -> Path:
    return _artifact(
        artifacts_base,
        "report",
        artifact_type="html-app",
        files={"report.html": "<html>report</html>"},
    )


@pytest.fixture
def publisher(monkeypatch):
    """Stand-ins for the two calls a full-stack upload makes on the way out.

    `vault_for_scope` is where the credentials come from, and
    `anton.publisher.publish` is what sends them. The refusal must come before
    both, so both are recorded rather than only the second.
    """
    upload = mock.Mock(
        return_value={"view_url": "https://view.example/r/1", "report_id": "rid-1", "md5": "d"}
    )
    vault = mock.Mock(return_value=object())
    monkeypatch.setattr("anton.publisher.publish", upload)
    monkeypatch.setattr(publish, "vault_for_scope", vault)
    return mock.Mock(upload=upload, vault=vault)


@pytest.mark.parametrize(
    "url",
    [
        "https://4nton.ai",
        "https://4nton.ai/",
        "https://api.staging.mindshub.ai",
        "https://api-ns1.dev.mindshub.ai",
        "https://api.mindshub.ai",
    ],
)
def test_mindshub_publish_hosts_are_recognised(url):
    assert is_mindshub_publish_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        SELF_HOSTED_PUBLISH_URL,
        "https://publish.customer.example",
        # A lookalike that only starts with our host must not pass.
        "https://4nton.ai.customer.example",
        "https://mindshub.ai.customer.example",
        "",
        None,
        # `.hostname` raises on an unbalanced bracket.
        "http://[publisher",
    ],
)
def test_other_publish_urls_are_not_mindshub(url):
    assert is_mindshub_publish_url(url) is False


def test_fullstack_to_a_self_hosted_publisher_is_refused_before_upload(
    fullstack, artifacts_base, publisher
):
    with pytest.raises(ValueError) as exc:
        publish.publish_artifact(
            fullstack,
            artifacts_base=artifacts_base,
            api_key="mdb_publish_secret",
            publish_url=SELF_HOSTED_PUBLISH_URL,
        )

    assert str(exc.value) == publish.FULLSTACK_PUBLISH_UNSUPPORTED
    publisher.upload.assert_not_called()
    publisher.vault.assert_not_called()
    assert not (fullstack / ".published.json").exists()


@pytest.mark.parametrize("url", ["https://4nton.ai", "https://api.staging.mindshub.ai"])
def test_fullstack_to_mindshub_still_publishes(fullstack, artifacts_base, publisher, url):
    out = publish.publish_artifact(
        fullstack, artifacts_base=artifacts_base, api_key="k", publish_url=url
    )

    assert out["url"] == "https://view.example/r/1"
    args, kwargs = publisher.upload.call_args
    assert args[0] == fullstack
    assert kwargs["publish_url"] == url


def test_static_artifact_to_a_self_hosted_publisher_still_publishes(
    static_page, artifacts_base, publisher
):
    out = publish.publish_artifact(
        static_page,
        artifacts_base=artifacts_base,
        api_key="mdb_publish_secret",
        publish_url=SELF_HOSTED_PUBLISH_URL,
    )

    assert out["url"] == "https://view.example/r/1"
    args, kwargs = publisher.upload.call_args
    assert args[0] == static_page / "report.html"
    assert kwargs["publish_url"] == SELF_HOSTED_PUBLISH_URL


def test_share_route_answers_400_with_the_reason(fullstack, artifacts_base, publisher):
    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")
    context = (fullstack, artifacts_base, "mdb_publish_secret", SELF_HOSTED_PUBLISH_URL)

    with mock.patch.object(publish_ep, "_desktop_context", lambda raw: context):
        res = TestClient(app).post("/api/v1/publish/", json={"path": str(fullstack)})

    assert res.status_code == 400
    assert res.json()["detail"] == publish.FULLSTACK_PUBLISH_UNSUPPORTED
    publisher.upload.assert_not_called()


@pytest.mark.asyncio
async def test_agent_publish_tool_reports_the_refusal_without_a_key_stop(
    fullstack, artifacts_base, publisher, monkeypatch
):
    """The tool turns a missing-key ValueError into a STOP directive by matching
    "api key" in the message. This refusal must read as a plain failure, or the
    agent would send the tester to configure a key they already have."""
    context = (fullstack, artifacts_base, "mdb_publish_secret", SELF_HOSTED_PUBLISH_URL)
    monkeypatch.setattr(publish, "desktop_publish_context", lambda raw: context)
    session = mock.Mock()
    session._workspace = mock.Mock(base=str(artifacts_base.parent.parent))

    out = await htools._cowork_publish_or_preview(
        session,
        {"file_path": str(fullstack / "static" / "index.html"), "action": "publish"},
    )

    text = getattr(out, "content", out)
    assert f"PUBLISH FAILED: {publish.FULLSTACK_PUBLISH_UNSUPPORTED}" in text
    assert "STOP" not in text
    publisher.upload.assert_not_called()
