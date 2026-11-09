"""Shared fixtures for the draft-host tests."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging

import pytest

from cowork.api.v1.artifact_scope import ResolvedArtifact, ReviewedRead
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.units import run_efs
from cowork.draft_host.token import issue_draft_token
from cowork.services.artifacts import ProjectArtifacts

USER = "11111111-1111-4111-8111-111111111111"
ORG = "22222222-2222-4222-8222-222222222222"
PROJECT = "33333333-3333-4333-8333-333333333333"
ARTIFACT = "44444444444444444444444444444444"
DRAFT_HOST = "draft-staging.artifacts.test"
WEB_ORIGIN = "https://cowork.staging.mindshub.ai"
# At least 32 characters: draft_link_keyring() drops anything shorter.
KEYS = {"k1": "draft-secret-0123456789-abcdefghijklmnop"}
DOC = "<html><head></head><body>draft body</body></html>"
# What a browser frame sends. Sec-Fetch-Site is the one the host requires.
FETCH = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "iframe"}
DRAFT_CSP = (
    "sandbox allow-scripts allow-popups allow-forms allow-modals; "
    f"connect-src 'self'; frame-ancestors {WEB_ORIGIN}"
)


def configure_draft_env(monkeypatch) -> None:
    """Org mode on staging with the six draft-link values set.

    Nothing listens on COWORK_DRAFT_PORT here: only start_draft_listener()
    binds it, and the tests that call it bind a free port instead.
    """
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("ENV", "staging")
    monkeypatch.delenv("POD_NAMESPACE", raising=False)
    monkeypatch.delenv("NAMESPACE", raising=False)
    monkeypatch.setenv("COWORK_DRAFT_HOSTS", DRAFT_HOST)
    monkeypatch.setenv("COWORK_DRAFT_ORIGIN", f"https://{DRAFT_HOST}")
    monkeypatch.setenv("COWORK_DRAFT_LINK_KEYS", json.dumps(KEYS))
    monkeypatch.setenv("COWORK_DRAFT_LINK_KID", "k1")
    monkeypatch.setenv("COWORK_WEB_ORIGIN", WEB_ORIGIN)
    monkeypatch.setenv("COWORK_DRAFT_PORT", "9011")
    get_app_settings.cache_clear()


@pytest.fixture
def draft_env(monkeypatch):
    configure_draft_env(monkeypatch)
    yield
    get_app_settings.cache_clear()


def events(caplog, name: str) -> list[logging.LogRecord]:
    """The captured records whose structured ``event`` is ``name``."""
    return [record for record in caplog.records if getattr(record, "event", None) == name]


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def draft_token(**overrides) -> str:
    fields = dict(env="staging", sub=USER, org=ORG, proj=PROJECT, art=ARTIFACT, keys=KEYS, kid="k1")
    fields.update(overrides)
    return issue_draft_token(**fields)


def sign_claims(claims: dict, secret: str = "draft-secret") -> str:
    """The token wire format written out independently of the module under test.

    The default is the short key the token tests sign with. A test that goes
    through the keyring passes ``KEYS["k1"]``.
    """

    payload = b64url(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"v1.{payload}"
    sig = b64url(hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest())
    return f"{signing_input}.{sig}"


def forge_identity(token: str, **claims) -> str:
    """``token``'s own signature over a payload with ``claims`` changed."""
    _, payload, sig = token.split(".")
    decoded = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    return sign_claims({**decoded, **claims}).rsplit(".", 1)[0] + "." + sig


def make_artifact(tmp_path, *, metadata=None, files=None):
    """A real artifact folder, in the layout the pinned-open code walks."""
    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "report"
    folder.mkdir(parents=True)
    for rel, text in (files or {"index.html": DOC, "assets/app.js": "console.log('draft');"}).items():
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (folder / "metadata.json").write_text(json.dumps(metadata or {"type": "html-app"}))
    # Where "../secret.txt" from the artifact folder leads, so the path tests
    # aim at a real file outside it.
    (base / "secret.txt").write_text("outside the artifact")
    source = ProjectArtifacts(
        base=base,
        project_id=None,
        project_name="project",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )
    return source, folder, metadata or {"type": "html-app"}


def fake_review(source, folder, metadata, seen: dict):
    """Stand-in for review_artifact_for_request that records who asked."""

    def review(scope, project_ref, artifact_id):
        seen.update(scope=scope, project_ref=project_ref, artifact_id=artifact_id)
        return source, folder, metadata, True

    return review


def reviewed_and_read(review):
    """``review``, a stand-in for review_artifact_for_request, as one for
    review_artifact_and_read: the route's read runs on what ``review``
    returned, in a file store thread as the real lookup runs it, and never
    when ``review`` raised."""

    async def review_and_read(scope, project_ref, artifact_id, *, read):
        artifact = ResolvedArtifact(*review(scope, project_ref, artifact_id))
        return await run_efs(lambda: ReviewedRead(artifact, read(artifact)), scope=scope)

    return review_and_read
