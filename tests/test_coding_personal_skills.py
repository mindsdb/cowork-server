from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from coding_service_fakes import CREDS, FakeEngine, repository, service_with, wait_for_status
from cowork.api.v1.endpoints import coding
from cowork.api.v1.endpoints.coding_personal_skills import MAX_SKILL_BYTES, PersonalSkillStore
from cowork.coding.contracts import SessionCreateRequest, SessionStatus
from cowork.common.settings.app_settings import get_app_settings
from cowork.services.skills import CodeSkillService, SkillService

BASE = "/api/v1/coding/skills/personal"
BODY = {"name": "Review TypeScript", "description": "Use when reviewing TypeScript changes.", "instructions": "Read the changed files.\nCheck error handling.", "enabled": True}
MARKDOWN = "---\nname: human-writing\ndescription: Use when writing user-facing copy.\n---\nKeep it concise and specific.\n"
BUILTIN = "thermo-nuclear-code-quality-review"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "home" / "skills"))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    engine = FakeEngine()
    service = service_with(tmp_path / "coding", engine)
    monkeypatch.setattr(coding, "_service", lambda: service)
    app = FastAPI()
    app.state.engine = engine
    app.include_router(coding.router, prefix="/api/v1/coding")
    yield app
    get_app_settings.cache_clear()


@pytest.fixture
def client(app):
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        yield client


def test_create_catalogue_read_edit_disable_and_delete(client):
    created = client.post(BASE, json=BODY)
    assert created.status_code == 201, created.text
    skill = created.json()
    assert skill == {**BODY, "id": "review-typescript", "projects": []}
    assert client.get(f"{BASE}/{skill['id']}").json() == skill
    catalogue = client.get("/api/v1/coding/skills/library").json()
    item = next(item for item in catalogue["items"] if item["path"] == skill["id"])
    assert item["origin"] == "personal" and item["enabled"] is True
    # A new service instance reads the persisted file; this is not a UI cache.
    assert CodeSkillService().get_skill(skill["id"]).instructions == BODY["instructions"]
    assert not SkillService().list_skills()

    updated = client.put(f"{BASE}/{skill['id']}", json={**BODY, "name": "My review", "instructions": "Version two.", "enabled": False})
    assert updated.status_code == 200, updated.text
    assert updated.json()["id"] == skill["id"]
    assert CodeSkillService().get_skill(skill["id"]).enabled is False
    assert client.delete(f"{BASE}/{skill['id']}").status_code == 204
    assert client.delete(f"{BASE}/{skill['id']}").status_code == 404
    assert not (CodeSkillService().root / skill["id"]).exists()


def test_import_keeps_metadata_and_duplicate_never_overwrites(client):
    source = MARKDOWN.replace("---\nKeep", "metadata:\n  custom: original\n  projects: чужой\n---\nKeep")
    result = client.post(f"{BASE}/import", json={"content": source})
    assert result.status_code == 201, result.text
    skill = CodeSkillService().get_skill("human-writing")
    assert skill.metadata["custom"] == "original"
    assert skill.projects == []
    original = (CodeSkillService().root / skill.name / "SKILL.md").read_bytes()
    duplicate = client.post(f"{BASE}/import", json={"content": source.replace("specific", "different")})
    assert duplicate.status_code == 409
    assert (CodeSkillService().root / skill.name / "SKILL.md").read_bytes() == original


def test_edit_preserves_supporting_files_metadata_and_project_restrictions(client):
    client.post(f"{BASE}/import", json={"content": MARKDOWN})
    store = CodeSkillService()
    store.update_skill("human-writing", projects=["project-one"])
    reference = store.root / "human-writing" / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("Keep this reference.")
    result = client.put(f"{BASE}/human-writing", json=BODY)
    assert result.status_code == 200
    assert result.json()["projects"] == ["project-one"]
    assert reference.read_text() == "Keep this reference."


@pytest.mark.parametrize("field,value", [("name", " "), ("name", "!!!"), ("description", " "), ("instructions", "\n\t"), ("instructions", "a" * (MAX_SKILL_BYTES + 1)), ("instructions", "🙂" * 40_000), ("name", None), ("projects", ["x"])])
def test_invalid_create_does_not_write(client, field, value):
    result = client.post(BASE, json={**BODY, field: value})
    assert result.status_code in {400, 422}
    assert not CodeSkillService().list_skills()


@pytest.mark.parametrize("content", ["", "Just ordinary text", "---\nname: [broken\n---\n", "a" * (MAX_SKILL_BYTES + 1), "🙂" * 40_000])
def test_invalid_import_is_non_destructive(client, content):
    result = client.post(f"{BASE}/import", json={"content": content})
    assert result.status_code in {400, 422}
    assert not CodeSkillService().list_skills()


def test_import_uses_the_canonical_parser_to_normalize_the_directory_name(client):
    result = client.post(f"{BASE}/import", json={"content": MARKDOWN.replace("human-writing", "../escape")})
    assert result.status_code == 201
    assert result.json()["id"] == "escape"
    assert (CodeSkillService().root / "escape" / "SKILL.md").is_file()


@pytest.mark.parametrize("operation", ["create", "import", "update", "delete", "alias-update", "alias-delete"])
def test_builtins_cannot_be_changed_through_personal_routes(client, operation):
    store = CodeSkillService()
    store.ensure_builtin_skills()
    original = (store.root / BUILTIN / "SKILL.md").read_bytes()
    alias = store.root / "alias"
    alias.symlink_to(store.root / BUILTIN, target_is_directory=True)
    actions = {
        "create": lambda: client.post(BASE, json={**BODY, "name": BUILTIN}),
        "import": lambda: client.post(f"{BASE}/import", json={"content": MARKDOWN.replace("human-writing", BUILTIN)}),
        "update": lambda: client.put(f"{BASE}/{BUILTIN}", json=BODY),
        "delete": lambda: client.delete(f"{BASE}/{BUILTIN}"),
        "alias-update": lambda: client.put(f"{BASE}/alias", json=BODY),
        "alias-delete": lambda: client.delete(f"{BASE}/alias"),
    }
    assert actions[operation]().status_code == 403
    assert (store.root / BUILTIN / "SKILL.md").read_bytes() == original


def test_symlink_cannot_read_write_or_delete_outside_the_code_store(client, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(MARKDOWN)
    store = CodeSkillService()
    store.root.mkdir(parents=True)
    (store.root / "escape").symlink_to(outside, target_is_directory=True)
    assert client.get(f"{BASE}/escape").status_code == 400
    assert client.put(f"{BASE}/escape", json=BODY).status_code == 400
    assert client.delete(f"{BASE}/escape").status_code == 400
    assert (outside / "SKILL.md").read_text() == MARKDOWN


@pytest.mark.parametrize("method,path,body", [("POST", "", BODY), ("POST", "/import", {"content": MARKDOWN}), ("GET", "/missing", None), ("PUT", "/missing", BODY), ("DELETE", "/missing", None)])
def test_cloud_and_non_loopback_callers_are_refused(app, monkeypatch, method, path, body):
    with TestClient(app, client=("203.0.113.1", 12345)) as remote:
        assert remote.request(method, BASE + path, json=body).status_code == 403
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    with TestClient(app, client=("127.0.0.1", 12345)) as cloud:
        assert cloud.request(method, BASE + path, json=body).status_code == 403


def test_concurrent_imports_have_one_winner(client):
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.post(f"{BASE}/import", json={"content": MARKDOWN}), range(2)))
    assert sorted(item.status_code for item in responses) == [201, 409]
    assert CodeSkillService().get_skill("human-writing").instructions.strip() == "Keep it concise and specific."


def test_untrusted_browser_origin_cannot_write(client):
    result = client.post(BASE, json=BODY, headers={"Origin": "https://untrusted.example"})
    assert result.status_code == 403
    assert not CodeSkillService().list_skills()


def test_new_task_receives_the_skill_created_over_http(client, app, tmp_path):
    assert client.post(BASE, json=BODY).status_code == 201
    service = coding._service()
    created = service.create_session(
        SessionCreateRequest(path=str(repository(tmp_path)), prompt="Use my review skill."),
        CREDS, "fake", "fake-model", code_skills=CodeSkillService(),
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    task = service.get_session(created.id)
    assert any(item.id == "personal:review-typescript" for item in task.resolved_skills)
    assert app.state.engine.configs[0].skill_roots == tuple(task.skill_roots)
    assert BODY["instructions"] in (Path(task.skill_roots[0]) / "review-typescript" / "SKILL.md").read_text()


@pytest.mark.parametrize("failure", [OSError("private path and internal details"), PermissionError(13, "private path and internal details")])
def test_failed_atomic_save_keeps_original(client, monkeypatch, failure):
    client.post(BASE, json=BODY)
    path = CodeSkillService().root / "review-typescript" / "SKILL.md"
    before = path.read_bytes()
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(PersonalSkillStore, "_replace_direct_child", fail)
    result = client.put(f"{BASE}/review-typescript", json={**BODY, "instructions": "Changed"})
    assert result.status_code == 500
    assert "internal details" not in result.text
    assert path.read_bytes() == before


def test_task_snapshots_survive_edits_disabling_and_deletion(client):
    service = coding._service()
    client.post(BASE, json=BODY)
    from cowork.coding.skill_runtime import SkillRuntimeResolver
    resolver = SkillRuntimeResolver(service.skill_library)
    def snapshot(task):
        result = resolver.resolve(task, None, CodeSkillService())
        return result, service.skill_library.root / "snapshots" / task / "review-typescript" / "SKILL.md"
    first, first_file = snapshot("first")
    original = first_file.read_bytes()
    assert any(item.name == BODY["name"] and item.origin == "personal" for item in first.items)
    client.put(f"{BASE}/review-typescript", json={**BODY, "instructions": "Version two"})
    _, next_file = snapshot("next")
    assert "Version two" in next_file.read_text()
    assert first_file.read_bytes() == original
    client.put(f"{BASE}/review-typescript", json={**BODY, "enabled": False})
    disabled, _ = snapshot("disabled")
    assert not any(item.id == "personal:review-typescript" for item in disabled.items)
    client.put(f"{BASE}/review-typescript", json=BODY)
    enabled, _ = snapshot("enabled")
    assert any(item.id == "personal:review-typescript" for item in enabled.items)
    client.delete(f"{BASE}/review-typescript")
    deleted, _ = snapshot("deleted")
    assert not any(item.id == "personal:review-typescript" for item in deleted.items)
    assert first_file.read_bytes() == original
