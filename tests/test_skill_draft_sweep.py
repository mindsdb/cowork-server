"""DELETE /projects/{name}/skill_drafts/{slug} — sweep a staged draft on Save."""
from pathlib import Path

from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import project_files
from cowork.common.settings.app_settings import get_app_settings
from cowork.server import app

client = TestClient(app)


def _drafts_dir() -> Path:
    return Path(get_app_settings().project.root_dir) / "general" / ".anton" / "skill_drafts"


def test_delete_skill_draft_removes_folder():
    folder = _drafts_dir() / "sweep-me"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")

    r = client.delete("/api/v1/projects/general/skill_drafts/sweep-me")
    assert r.status_code == 204
    assert not folder.exists()


def test_delete_skill_draft_missing_is_noop():
    r = client.delete("/api/v1/projects/general/skill_drafts/never-existed")
    assert r.status_code == 204


def test_delete_skill_draft_rejects_traversal():
    r = client.delete("/api/v1/projects/general/skill_drafts/..%2f..%2fsecret")
    assert r.status_code in (400, 404)


def test_delete_skill_draft_refuses_a_symlinked_drafts_directory(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    anton = project / ".anton"
    outside = tmp_path / "outside"
    victim = outside / "keep-me"
    anton.mkdir(parents=True)
    victim.mkdir(parents=True)
    (victim / "SKILL.md").write_text("keep", encoding="utf-8")
    (anton / "skill_drafts").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        project_files,
        "_opened_selected_project_directory",
        lambda *_args: project_files.pinned_dir(project),
    )

    r = client.delete("/api/v1/projects/general/skill_drafts/keep-me")

    assert r.status_code == 204
    assert (victim / "SKILL.md").read_text(encoding="utf-8") == "keep"


def test_delete_skill_draft_unknown_project_404():
    r = client.delete("/api/v1/projects/no-such-project/skill_drafts/x")
    assert r.status_code == 404
