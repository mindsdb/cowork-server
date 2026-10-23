"""Harness publish tool forwards access + preserves previous (ENG-322 follow-up)."""

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi import HTTPException

from cowork.harnesses.anton_harness import tools as htools
from cowork.services.publish import PublishOwnedByOtherUser, UploadServiceUnavailable


def _artifact(tmp_path: Path) -> Path:
    art = tmp_path / "app"
    art.mkdir(parents=True)
    (art / "index.html").write_text("<html></html>")
    return art / "index.html"


def _session(base: Path):
    s = mock.Mock()
    ws = mock.Mock()
    ws.base = str(base)
    ws.artifacts_dir = Path(base) / ".anton" / "artifacts"
    s._workspace = ws
    return s


@pytest.mark.asyncio
async def test_harness_forwards_explicit_password(tmp_path):
    f = _artifact(tmp_path)
    fake = mock.Mock(return_value={"url": "https://v/r/1"})
    with mock.patch.object(htools, "_publish_artifact", fake):
        out = await htools._cowork_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish",
             "access_mode": "password", "password": "hunter2"},
        )
    _, kwargs = fake.call_args
    assert kwargs["access"] == {"mode": "password", "password": "hunter2"}
    assert "Published" in getattr(out, "content", out)


@pytest.mark.asyncio
async def test_harness_preserves_previous(tmp_path):
    f = _artifact(tmp_path)
    fake = mock.Mock(return_value={"url": "https://v/r/1"})
    prev_state = {"published": True, "url": "u", "report_id": "r",
                  "mode": "password", "requires_password": True, "access_password": "old"}
    with mock.patch.object(htools, "_publish_artifact", fake), \
         mock.patch.object(htools, "_published_owner_state", return_value=prev_state):
        await htools._cowork_publish_or_preview(
            _session(tmp_path), {"file_path": str(f), "action": "publish"},
        )
    _, kwargs = fake.call_args
    assert kwargs["access"] == {"mode": "password", "password": "old"}  # NOT public


@pytest.mark.asyncio
async def test_harness_forwards_owner_only(tmp_path):
    f = _artifact(tmp_path)
    fake = mock.Mock(return_value={"url": "https://v/r/1"})
    with mock.patch.object(htools, "_publish_artifact", fake):
        await htools._cowork_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish",
             "access_mode": "restricted", "emails": [], "owner_only": True},
        )
    _, kwargs = fake.call_args
    assert kwargs["access"] == {
        "mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True,
    }


@pytest.mark.asyncio
async def test_harness_restricted_without_owner_only_omits_flag(tmp_path):
    f = _artifact(tmp_path)
    fake = mock.Mock(return_value={"url": "https://v/r/1"})
    with mock.patch.object(htools, "_publish_artifact", fake):
        await htools._cowork_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish",
             "access_mode": "restricted", "emails": ["a@x.com"]},
        )
    _, kwargs = fake.call_args
    assert kwargs["access"]["owner_only"] is False
    assert kwargs["access"]["emails"] == ["a@x.com"]


@pytest.mark.asyncio
async def test_harness_restricted_normalizes_emails(tmp_path):
    """Parity with anton's tool path, which already runs parse_emails."""
    f = _artifact(tmp_path)
    fake = mock.Mock(return_value={"url": "https://v/r/1"})
    with mock.patch.object(htools, "_publish_artifact", fake):
        await htools._cowork_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish",
             "access_mode": "restricted", "emails": [" A@X.com ", "a@x.com"]},
        )
    _, kwargs = fake.call_args
    assert kwargs["access"]["emails"] == ["a@x.com"]


@pytest.mark.asyncio
async def test_harness_restricted_invalid_email_returns_error(tmp_path):
    f = _artifact(tmp_path)
    fake = mock.Mock(return_value={"url": "https://v/r/1"})
    with mock.patch.object(htools, "_publish_artifact", fake):
        out = await htools._cowork_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish",
             "access_mode": "restricted", "emails": ["colleague@corp"]},
        )
    text = getattr(out, "content", out)
    assert "colleague@corp" in text
    assert "INVALID" in text
    fake.assert_not_called()


@pytest.mark.asyncio
async def test_harness_tells_the_agent_not_to_work_around_another_owner(tmp_path):
    f = _artifact(tmp_path)
    fake = mock.Mock(side_effect=PublishOwnedByOtherUser())
    with mock.patch.object(htools, "_publish_artifact", fake):
        out = await htools._cowork_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish", "access_mode": "public"},
        )

    fake.assert_called_once()
    assert out.ok is False
    assert out.reason == "owned_by_other_user"
    message = json.loads(out.content)["message"]
    assert message.startswith("PUBLISH FAILED: This artifact was published by another owner.")
    assert "do not publish this artifact under a new report on your own" in message
    # Not a ban for the rest of the conversation: the user may fix the account.
    assert "publish again only if they ask after resolving it" in message


SLUG = "path-check-1a2b3c4d"


def _text(out) -> str:
    return getattr(out, "content", out)


@pytest.fixture
def project(tmp_path):
    """A project folder laid out like desktop Cowork's: one artifact, a
    `.anton/memory` folder and a symlink in the artifacts folder pointing at it.

    Returns (base, artifact, memory), resolved: on macOS `tmp_path` sits under
    the `/var` -> `/private/var` symlink and the tool replies with resolved paths.
    """
    base = tmp_path / "general"
    artifacts = base / ".anton" / "artifacts"
    artifact = artifacts / SLUG
    artifact.mkdir(parents=True)
    (artifact / "index.html").write_text("<h1>hello</h1>")
    (artifact / "metadata.json").write_text("{}")
    memory = base / ".anton" / "memory"
    memory.mkdir()
    (memory / "notes.md").write_text("private")
    (artifacts / "escape").symlink_to(memory)
    return base.resolve(), artifact.resolve(), memory.resolve()


async def _ask(base: Path, raw: str) -> str:
    with mock.patch.object(htools, "_published_state", return_value={}):
        out = await htools._cowork_publish_or_preview(
            _session(base), {"file_path": raw, "action": "ask"},
        )
    return _text(out)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "form",
    [
        "{slug}",
        "{slug}/index.html",
        "artifacts/{slug}",
        "artifacts/{slug}/index.html",
        "artifacts//{slug}",
        "./{slug}/./index.html",
        ".anton/artifacts/{slug}",
        "{absolute}",
    ],
)
async def test_ask_finds_the_artifact_by_any_supported_path(project, form):
    base, artifact, _ = project
    text = await _ask(base, form.format(slug=SLUG, absolute=artifact))
    assert str(artifact) in text
    assert "NOT been published" in text
    assert "not found" not in text.lower()


@pytest.mark.asyncio
async def test_artifacts_folder_wins_over_a_same_named_project_folder(project):
    base, artifact, _ = project
    (base / SLUG).mkdir()
    (base / SLUG / "index.html").write_text("<h1>user file</h1>")
    assert f"is at {artifact} " in await _ask(base, SLUG)


@pytest.mark.asyncio
async def test_publish_by_short_path_hands_the_service_the_absolute_artifact_path(project):
    base, artifact, _ = project
    fake = mock.Mock(return_value={"url": "https://v/r/1"})
    with mock.patch.object(htools, "_publish_artifact", fake):
        out = await htools._cowork_publish_or_preview(
            _session(base),
            {"file_path": f"artifacts/{SLUG}", "action": "publish", "access_mode": "public"},
        )
    args, _ = fake.call_args
    assert args[0] == str(artifact)
    assert "https://v/r/1" in _text(out)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["ask", "publish"])
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("artifacts/../memory", id="dotdot"),
        pytest.param("artifacts/../.anton/memory", id="dotdot-into-anton"),
        pytest.param("artifacts/no-such-slug", id="missing"),
        pytest.param("artifacts/escape", id="symlink-escape"),
        pytest.param("artifacts/escape/notes.md", id="symlink-escape-file"),
        pytest.param("x\x00y", id="nul"),
        pytest.param("a" * 300, id="too-long"),
        pytest.param("~no-such-user-7f3a/report", id="unknown-user"),
    ],
)
async def test_unmatched_path_gets_a_not_found_reply_and_publishes_nothing(project, raw, action):
    base, _, memory = project
    fake = mock.Mock(return_value={"url": "https://v/r/1"})
    with mock.patch.object(htools, "_publish_artifact", fake), \
         mock.patch.object(htools, "_published_state", return_value={}):
        out = await htools._cowork_publish_or_preview(
            _session(base),
            {"file_path": raw, "action": action, "access_mode": "public"},
        )
    text = _text(out)
    assert "not found" in text
    assert "Pass the artifact's slug" in text
    assert "create_artifact" in text
    assert str(memory) not in text
    # The reply echoes the input, so `.anton/memory` may appear only when the
    # agent wrote it.
    if ".anton/memory" not in raw:
        assert ".anton/memory" not in text
    fake.assert_not_called()


@pytest.mark.asyncio
async def test_not_found_reply_shortens_a_long_path(project):
    base, _, _ = project
    text = await _ask(base, "a" * 300)
    assert "…" in text
    assert "a" * 201 not in text
    assert repr("a" * 200 + "…") in text


def test_publish_description_says_what_to_pass_not_where_artifacts_live():
    description = htools.build_cowork_publish_tool().description
    for stale in ("artifacts/<artifact-id>", "e.g. artifacts/<slug>", "<workspace>/artifacts/"):
        assert stale not in description
    assert "the artifact's slug" in description
    assert "`create_artifact`" in description
    assert "`generate_artifact`" in description
    assert "open_artifact" not in description


# The publish tool's description and prompt at staging 8b1ecf4, before
# deployment identity support. Hosts without COWORK_DEPLOYMENT_KIND keep them.
DEFAULT_DESCRIPTION_SHA256 = "5bae1e84c433d3e09c5b572e6b1e60221143a30086a8e71b4898e2239149a6b1"
DEFAULT_PROMPT_SHA256 = "5e040dfeaca047ee2307d953989e2282106f6972c8d7aaf4655301e50ec1e73a"


@pytest.mark.asyncio
async def test_default_hosts_keep_the_publish_tool_text(project):
    base, artifact, _ = project
    publish = htools.build_cowork_publish_tool()
    assert hashlib.sha256(publish.description.encode()).hexdigest() == DEFAULT_DESCRIPTION_SHA256
    assert hashlib.sha256(publish.prompt.encode()).hexdigest() == DEFAULT_PROMPT_SHA256
    assert await _ask(base, SLUG) == (
        f"Dashboard is at {artifact} and visible in the Live Artifacts panel. It has NOT been "
        "published to the public web yet. If the user asks to publish / share / make it public, "
        "call this tool again with action='publish' — MindsHub publishing works directly from "
        "chat in the desktop app, no slash command needed."
    )


@pytest.mark.asyncio
async def test_harness_tells_the_agent_not_to_retry_a_too_large_artifact(tmp_path):
    from cowork.services.publish import PublishTooLarge

    f = _artifact(tmp_path)
    fake = mock.Mock(side_effect=PublishTooLarge())
    with mock.patch.object(htools, "_publish_artifact", fake):
        out = await htools._cowork_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish", "access_mode": "public"},
        )

    assert out.ok is False
    assert out.reason == "too_large"
    message = json.loads(out.content)["message"]
    assert message.startswith("PUBLISH FAILED: This artifact is too large to publish.")
    assert "Do not retry with the same files." in message


@pytest.mark.asyncio
@pytest.mark.parametrize("exc, reason, text", [
    (UploadServiceUnavailable(), "publish_unavailable",
     "The publishing service is temporarily unavailable. Try again in a moment."),
    (HTTPException(503, detail="Publishing is busy right now."), "http_error", "Publishing is busy right now."),
    (HTTPException(400, detail={"code": "bad"}), "bad", "{'code': 'bad'}"),
])
async def test_harness_passes_a_typed_refusal_on_without_a_traceback(tmp_path, caplog, exc, reason, text):
    import logging

    f = _artifact(tmp_path)
    fake = mock.Mock(side_effect=exc)
    with caplog.at_level(logging.INFO, logger="cowork.harnesses.anton_harness.tools"):
        with mock.patch.object(htools, "_publish_artifact", fake):
            out = await htools._cowork_publish_or_preview(
                _session(tmp_path),
                {"file_path": str(f), "action": "publish", "access_mode": "public"},
            )

    assert out.ok is False
    assert out.reason == reason
    assert json.loads(out.content)["message"] == f"PUBLISH FAILED: {text}"
    assert "Traceback" not in caplog.text


@pytest.mark.asyncio
async def test_harness_reports_a_rejected_publish_without_a_second_traceback(tmp_path, caplog):
    import logging

    from cowork.services.publish import PublishRejected

    f = _artifact(tmp_path)
    fake = mock.Mock(side_effect=PublishRejected("Publishing failed: This organization is archived."))
    with caplog.at_level(logging.INFO, logger="cowork.harnesses.anton_harness.tools"):
        with mock.patch.object(htools, "_publish_artifact", fake):
            out = await htools._cowork_publish_or_preview(
                _session(tmp_path),
                {"file_path": str(f), "action": "publish", "access_mode": "public"},
            )

    assert out.ok is False
    assert out.reason == "PublishRejected"
    assert json.loads(out.content)["message"] == "PUBLISH FAILED: Publishing failed: This organization is archived."
    assert "Traceback" not in caplog.text
