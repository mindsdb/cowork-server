"""A remote turn's end diff reads the pod's list of the artifacts it touched.

anton names the artifacts a cloud turn's tools touched on ``turn_completed``.
With that list the diff claims and checks only those folders: a sibling turn's
artifact is never this turn's, and the content of every other artifact in the
project is left unread. A pod that sends no list gets today's diff, where
provenance decides what the turn created.
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from cowork.services import task_objects
from cowork.services.task_objects import (
    snapshot_artifact_state,
    tracked_artifacts_from_reply,
    turn_artifact_changes,
)
from test_responses_remote_backend import (
    _artifact,
    _conversation_at,
    _FakeBuffer,
    _remote_handler,
    responses_mod,
)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"artifacts_touched": ["a", "b"]}, {"a", "b"}),
        ({"artifacts_touched": []}, set()),
        ({}, None),
        ({"artifacts_touched": "a"}, None),
        ({"artifacts_touched": ["a", 7]}, None),
        ({"artifacts_touched": None}, None),
        ({"artifacts_touched": [f"a{i}" for i in range(1001)]}, None),
    ],
    ids=["list", "empty", "older-pod", "string", "non-string", "null", "too-many"],
)
def test_the_pods_list_is_read_only_when_it_is_a_list_of_names(data, expected):
    assert tracked_artifacts_from_reply(data) == expected


class _ContentReads:
    """Records the files statted inside artifact folders, other than the
    metadata the folder listing checks."""

    def __init__(self, monkeypatch, base: Path):
        self.paths: list[Path] = []
        self.counting = False
        base_text = str(base)
        for name in ("stat", "lstat", "scandir"):
            real = getattr(os, name)

            def recorded(path, *args, _real=real, **kwargs):
                if self.counting and isinstance(path, (str, os.PathLike)):
                    text = os.fspath(path)
                    if text.startswith(base_text) and not text.endswith("metadata.json"):
                        self.paths.append(Path(text))
                return _real(path, *args, **kwargs)

            monkeypatch.setattr(os, name, recorded)

    def folders(self, base: Path) -> set[str]:
        """The artifact folders whose own files were read."""
        return {path.relative_to(base).parts[0] for path in self.paths if len(path.relative_to(base).parts) > 1}


def test_with_the_pods_list_only_its_folders_content_is_read(tmp_path, monkeypatch):
    conversation = uuid4()
    base = tmp_path / ".anton" / "artifacts"
    for index in range(200):
        _artifact(base, f"old-{index}")
    before, before_mtimes = snapshot_artifact_state(base)
    edited = _artifact(base, "old-7", body="<html>edited</html>")
    os.utime(edited / "report.html", ns=(10**19, 10**19))
    _artifact(base, "made", conversation_id=conversation)
    _artifact(base, "siblings", conversation_id=conversation)  # provenance alone would claim it

    reads = _ContentReads(monkeypatch, base)
    reads.counting = True
    changes = turn_artifact_changes(
        conversation_id=conversation, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        tracked_new={"made", "old-7"}, tracked_edits={"made", "old-7"}, attribute_by_provenance=True,
    )
    reads.counting = False

    assert changes.created == ["made"]
    assert changes.touched == {"made", "old-7"}
    # Only the two named folders' files are read. The other 199 and the
    # sibling's get the listing's metadata check and the queued-repair check
    # under `.revisions`, nothing else.
    untouched_reads = [
        path for path in reads.paths
        if len(path.relative_to(base).parts) > 1
        and path.relative_to(base).parts[0] not in {"made", "old-7"}
        and path.relative_to(base).parts[1] != ".revisions"
    ]
    assert untouched_reads == []
    assert {"made", "old-7"} <= reads.folders(base)


def test_without_the_pods_list_every_folder_is_checked_for_edits(tmp_path, monkeypatch):
    """An older pod: today's diff, which reads every folder's content."""
    conversation = uuid4()
    base = tmp_path / ".anton" / "artifacts"
    for index in range(20):
        _artifact(base, f"old-{index}")
    before, before_mtimes = snapshot_artifact_state(base)

    reads = _ContentReads(monkeypatch, base)
    reads.counting = True
    turn_artifact_changes(
        conversation_id=conversation, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        attribute_by_provenance=True,
    )
    reads.counting = False

    assert {f"old-{index}" for index in range(20)} <= reads.folders(base)


async def _remote_turn(monkeypatch, tmp_path, completed: dict) -> list[str]:
    """Run a remote turn whose pod writes `mine` and a sibling's `theirs`,
    both naming this conversation in their provenance, and return the slugs
    it indexed."""
    saved: dict = {}
    handler = _remote_handler(monkeypatch, saved)
    project_dir = tmp_path / "proj"
    artifacts_base = project_dir / ".anton" / "artifacts"
    artifacts_base.mkdir(parents=True)
    conversation = _conversation_at(project_dir)
    monkeypatch.setattr(
        responses_mod.ResponsesHandler, "_remote_artifacts_context",
        staticmethod(lambda session, conv_id: (
            conversation, artifacts_base, str(conversation.project_id), conversation.project.name,
        )),
    )
    indexed: dict = {}

    def spy_index(session, **kwargs):
        indexed["slugs"] = kwargs["slugs"]

    monkeypatch.setattr(task_objects, "record_new_artifacts", spy_index)

    async def fake_autopublish(base, scope, *, touched, **kwargs):
        return set(touched)

    monkeypatch.setattr("cowork.services.artifact_autopublish.autopublish_project_artifacts", fake_autopublish)
    conv_id = uuid4()

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        _artifact(artifacts_base, "mine", conversation_id=conv_id)
        _artifact(artifacts_base, "theirs", conversation_id=conv_id)
        yield "turn_completed", completed

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)
    await handler._produce_remote(
        conv_id=conv_id, input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=_FakeBuffer(),
    )
    return indexed.get("slugs", [])


async def test_a_remote_turn_claims_only_what_the_pod_named(monkeypatch, tmp_path):
    """A sibling turn's folder can carry this conversation's provenance (it
    copied the folder, or anton crashed between writes); the pod's list is
    what this turn's own tools did."""
    assert await _remote_turn(monkeypatch, tmp_path, {"artifacts_touched": ["mine"]}) == ["mine"]


async def test_a_remote_turn_from_an_older_pod_keeps_todays_diff(monkeypatch, tmp_path):
    assert await _remote_turn(monkeypatch, tmp_path, {}) == ["mine", "theirs"]
