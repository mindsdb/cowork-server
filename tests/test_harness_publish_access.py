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


def _session(tmp_path):
    s = mock.Mock()
    ws = mock.Mock()
    ws.base = str(tmp_path)
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
    assert "Published" in out


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
