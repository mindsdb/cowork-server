"""Harness publish tool forwards access + preserves previous (ENG-322 follow-up)."""

from pathlib import Path
from unittest import mock

import pytest

from cowork.harnesses.anton_harness import tools as htools


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
