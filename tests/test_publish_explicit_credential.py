"""Publish takes its credential and its root from the caller.

In org mode there is no stored provider key (a turn key is minted per turn) and
the module-level FS scan finds no artifact roots, so neither the key nor the root
can be discovered inside these functions.
"""
from __future__ import annotations

import inspect
import json

import pytest

from cowork.services import publish as p

# What the autopublish reconciler will publish with. `org_allowed` is not
# optional decoration: `resolve_access` degrades a `restricted` request with
# neither emails nor org to `public` (anton/publish_access.py), so without it
# every auto-published artifact would be world-readable. It also gives the
# author two independent grants — owner-by-FK and org membership — instead of
# depending on the owner check alone.
AUTOPUBLISH_ACCESS = {"mode": "restricted", "emails": [], "org_allowed": True}


def test_installed_publisher_accepts_stable_artifact_key():
    """Pin the cross-repository contract that publish_artifact relies on.

    Most publish tests replace Anton with a ``**kwargs`` fake, which cannot
    detect a stale lockfile resolving a publisher that rejects this argument.
    """
    from anton.publisher import publish

    assert "artifact_key" in inspect.signature(publish).parameters


def _make_artifact(base, slug, *, files: dict[str, str], meta: dict):
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


@pytest.fixture
def artifact(tmp_path):
    base = tmp_path / "org-1" / "proj" / ".anton" / "artifacts"
    folder = _make_artifact(base, "rep", files={"report.html": "<html>v1</html>"},
                            meta={"slug": "rep", "name": "Rep", "type": "html-app"})
    return folder, base


def test_publish_passes_caller_key_and_url_to_publisher(artifact, monkeypatch):
    folder, base = artifact
    seen = {}

    def fake_publish(file_path, **kwargs):
        seen.update(kwargs)
        seen["file_path"] = file_path
        return {"view_url": "https://view.example/r/1", "report_id": "rid-1",
                "md5": "deadbeef", "artifact_key": "u/rid-1"}

    monkeypatch.setattr("anton.publisher.publish", fake_publish)

    out = p.publish_artifact(
        folder, artifacts_base=base,
        api_key="turnkey-abc", publish_url="https://api.staging.mindshub.ai",
        access=dict(AUTOPUBLISH_ACCESS),
    )

    assert out["url"] == "https://view.example/r/1"
    assert seen["api_key"] == "turnkey-abc"
    assert seen["publish_url"] == "https://api.staging.mindshub.ai"
    # Asserted on what the PUBLISHER receives, not on what we passed in:
    # `resolve_access` rewrites the request on the way through, and a mode that
    # silently became `public` there is exactly the failure this pins down.
    assert seen["access"] == {"mode": "restricted", "emails": [], "org_allowed": True}


def test_publish_writes_published_json_with_report_id(artifact, monkeypatch):
    folder, base = artifact
    monkeypatch.setattr(
        "anton.publisher.publish",
        lambda file_path, **kw: {"view_url": "https://view.example/r/1",
                                 "report_id": "rid-1", "md5": "deadbeef",
                                 "artifact_key": "u/rid-1"},
    )

    p.publish_artifact(folder, artifacts_base=base, api_key="k",
                       publish_url="https://api.staging.mindshub.ai",
                       access=dict(AUTOPUBLISH_ACCESS))

    entry = json.loads((folder / ".published.json").read_text())["report.html"]
    assert entry["report_id"] == "rid-1"
    assert entry["published"] is True
    assert entry["mode"] == "restricted"
    assert entry["org_allowed"] is True


def test_republish_reuses_stored_report_id(artifact, monkeypatch):
    folder, base = artifact
    calls = []

    def fake_publish(file_path, **kwargs):
        calls.append(kwargs.get("report_id"))
        return {"view_url": "https://view.example/r/1", "report_id": "rid-1",
                "md5": "beef", "artifact_key": "u/rid-1"}

    monkeypatch.setattr("anton.publisher.publish", fake_publish)
    kw = dict(artifacts_base=base, api_key="k",
              publish_url="https://api.staging.mindshub.ai",
              access={"mode": "restricted", "emails": []})

    p.publish_artifact(folder, **kw)
    p.publish_artifact(folder, **kw)

    assert calls == [None, "rid-1"]


def test_published_json_write_is_atomic_no_tmp_left_behind(artifact, monkeypatch):
    folder, base = artifact
    monkeypatch.setattr(
        "anton.publisher.publish",
        lambda file_path, **kw: {"view_url": "u", "report_id": "rid-1",
                                 "md5": "m", "artifact_key": "k"},
    )

    p.publish_artifact(folder, artifacts_base=base, api_key="k",
                       publish_url="https://api.staging.mindshub.ai",
                       access=dict(AUTOPUBLISH_ACCESS))

    leftovers = [x.name for x in folder.iterdir() if x.name.startswith(".published.json.")]
    assert leftovers == []


def test_restricted_without_org_or_emails_degrades_to_public(artifact, monkeypatch):
    """Why AUTOPUBLISH_ACCESS carries org_allowed.

    `resolve_access` treats "restricted with nothing selected" as a caller
    mistake and falls back to public — a safety net for programmatic callers.
    Auto-publishing with `{"mode": "restricted", "emails": []}` would therefore
    make every artifact world-readable. Pinned here so the flag cannot be
    dropped as redundant.
    """
    folder, base = artifact
    seen = {}
    monkeypatch.setattr(
        "anton.publisher.publish",
        lambda file_path, **kw: seen.update(kw) or {
            "view_url": "u", "report_id": "rid-1", "md5": "m", "artifact_key": "k",
        },
    )

    p.publish_artifact(folder, artifacts_base=base, api_key="k",
                       publish_url="https://api.staging.mindshub.ai",
                       access={"mode": "restricted", "emails": []})

    assert seen["access"] == {"mode": "public"}


def test_publish_without_api_key_raises(artifact):
    folder, base = artifact

    with pytest.raises(ValueError):
        p.publish_artifact(folder, artifacts_base=base, api_key="",
                           publish_url="https://api.staging.mindshub.ai")


def test_unpublish_uses_caller_key_and_soft_deletes(artifact, monkeypatch):
    folder, base = artifact
    monkeypatch.setattr(
        "anton.publisher.publish",
        lambda file_path, **kw: {"view_url": "u", "report_id": "rid-1",
                                 "md5": "m", "artifact_key": "k"},
    )
    p.publish_artifact(folder, artifacts_base=base, api_key="k",
                       publish_url="https://api.staging.mindshub.ai",
                       access=dict(AUTOPUBLISH_ACCESS))

    seen = {}
    monkeypatch.setattr("anton.publisher.unpublish",
                        lambda ident, **kw: seen.update({"ident": ident, **kw}))

    p.unpublish_artifact(folder, artifacts_base=base, api_key="turnkey-xyz",
                         publish_url="https://api.staging.mindshub.ai")

    assert seen["ident"] == "rid-1"
    assert seen["api_key"] == "turnkey-xyz"
    entry = json.loads((folder / ".published.json").read_text())["report.html"]
    assert entry["published"] is False
    assert entry["report_id"] == "rid-1"  # kept so a re-publish reuses the URL


def test_unpublish_404_is_treated_as_already_gone_and_logs_orphan(artifact, monkeypatch, caplog):
    folder, base = artifact
    monkeypatch.setattr(
        "anton.publisher.publish",
        lambda file_path, **kw: {"view_url": "https://view.example/r/1",
                                 "report_id": "rid-1", "md5": "m", "artifact_key": "k"},
    )
    p.publish_artifact(folder, artifacts_base=base, api_key="k",
                       publish_url="https://api.staging.mindshub.ai",
                       access=dict(AUTOPUBLISH_ACCESS))

    def boom(ident, **kw):
        raise RuntimeError("HTTP Error 404: Not Found")

    monkeypatch.setattr("anton.publisher.unpublish", boom)

    with caplog.at_level("WARNING"):
        out = p.unpublish_artifact(folder, artifacts_base=base, api_key="k",
                                   publish_url="https://api.staging.mindshub.ai")

    assert out["status"] == "ok"
    # The remote object may still exist under a DIFFERENT owner prefix; the log
    # line is the only trace left once the local record is cleared.
    assert "orphaned_publish" in caplog.text
    assert "rid-1" in caplog.text


def test_unpublish_other_error_propagates(artifact, monkeypatch):
    folder, base = artifact
    monkeypatch.setattr(
        "anton.publisher.publish",
        lambda file_path, **kw: {"view_url": "u", "report_id": "rid-1",
                                 "md5": "m", "artifact_key": "k"},
    )
    p.publish_artifact(folder, artifacts_base=base, api_key="k",
                       publish_url="https://api.staging.mindshub.ai",
                       access=dict(AUTOPUBLISH_ACCESS))

    def boom(ident, **kw):
        raise RuntimeError("HTTP Error 500: Server Error")

    monkeypatch.setattr("anton.publisher.unpublish", boom)

    with pytest.raises(RuntimeError):
        p.unpublish_artifact(folder, artifacts_base=base, api_key="k",
                             publish_url="https://api.staging.mindshub.ai")


def test_agent_publish_tool_wrapper_still_works(artifact, monkeypatch):
    """The agent's publish_or_preview tool goes through this wrapper.

    Existing harness tests patch the wrapper itself, so a signature change here
    would break the tool in production while the suite stayed green. This test
    calls the real wrapper with only the HTTP layer mocked.
    """
    folder, base = artifact
    from cowork.harnesses.anton_harness import tools as htools

    monkeypatch.setattr(
        "anton.publisher.publish",
        lambda file_path, **kw: {"view_url": "u", "report_id": "rid-1",
                                 "md5": "m", "artifact_key": "k"},
    )
    monkeypatch.setattr(
        p, "desktop_publish_context",
        lambda raw: (folder, base, "desktop-key", "https://api.staging.mindshub.ai"),
    )

    out = htools._publish_artifact(str(folder), access={"mode": "public"})

    assert out["status"] == "ok"
