from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from coding_service_fakes import CREDS, FakeEngine, repository, service_with, wait_for_status
from cowork.api.v1.endpoints import coding
from cowork.api.v1.endpoints.coding_personal_skills import MAX_SKILL_BYTES
from cowork.api.v1.router import api_router
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
    app.include_router(api_router)
    yield app
    get_app_settings.cache_clear()


@pytest.fixture
def client(app):
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        yield client


def test_personal_routes_are_mounted_once_in_the_canonical_router(client):
    from cowork.api.v1.route_walker import route_key

    routes = [
        route_key(route)
        for route in api_router.routes
        if route.path.startswith(BASE)
    ]
    assert len(routes) == 5
    assert len(set(routes)) == 5
    schema = client.get("/openapi.json").json()
    assert "PersonalSkillWrite" in schema["components"]["schemas"]
    assert "PersonalSkillImport" in schema["components"]["schemas"]


def test_other_code_routes_keep_standard_validation_errors(client):
    result = client.post("/api/v1/coding/skills/sources", json={})
    assert result.status_code == 422
    assert isinstance(result.json()["detail"], list)


@pytest.mark.parametrize("body", [None, [], {"name": "Incomplete"}])
def test_malformed_skill_body_has_string_validation_detail(client, body):
    result = client.post(BASE, json=body)
    assert result.status_code == 400
    assert isinstance(result.json()["detail"], str)
    assert not CodeSkillService().list_skills()


def test_malformed_skill_json_has_string_validation_detail(client):
    result = client.post(BASE, content="{", headers={"Content-Type": "application/json"})
    assert result.status_code == 400
    assert isinstance(result.json()["detail"], str)


@pytest.mark.parametrize("text", ["a" * MAX_SKILL_BYTES, "🙂" * (MAX_SKILL_BYTES // 4)], ids=["ascii", "unicode"])
def test_instructions_at_the_byte_limit_are_accepted(client, text):
    result = client.post(BASE, json={**BODY, "instructions": text})
    assert result.status_code == 201
    assert CodeSkillService().get_skill("review-typescript").instructions == text


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


def test_duplicate_create_returns_conflict_without_overwriting(client):
    assert client.post(BASE, json=BODY).status_code == 201
    path = CodeSkillService().root / "review-typescript" / "SKILL.md"
    original = path.read_bytes()
    duplicate = client.post(BASE, json={**BODY, "name": "review-typescript", "instructions": "Do not overwrite."})
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "A skill with that name already exists. Edit it or use a different name."
    assert path.read_bytes() == original


@pytest.fixture
def concurrent_writes(monkeypatch):
    barrier = Barrier(2)
    write = CodeSkillService._write

    def concurrent_write(self, skill, **kwargs):
        # Both requests pass the existence check before either claims the slug.
        barrier.wait(timeout=5)
        return write(self, skill, **kwargs)

    monkeypatch.setattr(CodeSkillService, "_write", concurrent_write)


def test_concurrent_creates_return_conflict_at_the_atomic_write(client, concurrent_writes):
    bodies = [{**BODY, "instructions": instructions} for instructions in ["First", "Second"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda body: client.post(BASE, json=body), bodies))
    assert sorted(result.status_code for result in responses) == [201, 409]
    winner = next(result.json() for result in responses if result.status_code == 201)
    assert CodeSkillService().get_skill(winner["id"]).instructions == winner["instructions"]


@pytest.mark.parametrize("deleted", [False, True], ids=["missing", "deleted"])
@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE"])
def test_missing_skill_returns_not_found(client, method, deleted):
    path = f"{BASE}/review-typescript"
    if deleted:
        assert client.post(BASE, json=BODY).status_code == 201
        assert client.delete(path).status_code == 204
    result = client.request(method, path, json=BODY if method == "PUT" else None)
    assert result.status_code == 404
    assert result.json()["detail"] == "This personal skill no longer exists."
    assert not (CodeSkillService().root / "review-typescript").exists()


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


@pytest.mark.parametrize("field,value", [
    pytest.param("name", " ", id="blank-name"),
    pytest.param("name", "!!!", id="invalid-slug"),
    pytest.param("description", " ", id="blank-description"),
    pytest.param("instructions", "\n\t", id="blank-instructions"),
    pytest.param("instructions", "a" * (MAX_SKILL_BYTES + 1), id="large-ascii"),
    pytest.param("instructions", "🙂" * 40_000, id="large-unicode"),
    pytest.param("name", None, id="null-name"),
    pytest.param("projects", ["x"], id="unknown-field"),
])
def test_invalid_create_does_not_write(client, field, value):
    result = client.post(BASE, json={**BODY, field: value})
    assert result.status_code == 400
    assert isinstance(result.json()["detail"], str)
    assert not CodeSkillService().list_skills()


@pytest.mark.parametrize("content", [
    pytest.param("", id="empty"),
    pytest.param("Just ordinary text", id="no-frontmatter"),
    pytest.param("---\nname: [broken\n---\n", id="invalid-frontmatter"),
    pytest.param("a" * (MAX_SKILL_BYTES + 1), id="large-ascii"),
    pytest.param("🙂" * 40_000, id="large-unicode"),
])
def test_invalid_import_is_non_destructive(client, content):
    result = client.post(f"{BASE}/import", json={"content": content})
    assert result.status_code == 400
    assert isinstance(result.json()["detail"], str)
    assert not CodeSkillService().list_skills()


@pytest.mark.parametrize("text", ["a" * (MAX_SKILL_BYTES + 1), "🙂" * 40_000], ids=["ascii", "unicode"])
@pytest.mark.parametrize("operation", ["create", "update", "import"])
def test_oversized_text_has_a_readable_error(client, text, operation):
    assert client.post(BASE, json=BODY).status_code == 201
    path = CodeSkillService().root / "review-typescript" / "SKILL.md"
    before = path.read_bytes()
    if operation == "import":
        result = client.post(f"{BASE}/import", json={"content": text})
    else:
        method = "PUT" if operation == "update" else "POST"
        url = f"{BASE}/review-typescript" if operation == "update" else BASE
        result = client.request(method, url, json={**BODY, "instructions": text})
    assert result.status_code == 400
    assert "Keep the skill under 120 KB." in result.json()["detail"]
    assert path.read_bytes() == before


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


def test_concurrent_imports_have_one_winner(client, concurrent_writes):
    sources = [MARKDOWN, MARKDOWN.replace("specific", "friendly")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(
            lambda content: client.post(f"{BASE}/import", json={"content": content}),
            sources,
        ))
    assert sorted(item.status_code for item in responses) == [201, 409]
    winner = next(item.json() for item in responses if item.status_code == 201)
    assert CodeSkillService().get_skill("human-writing").instructions == winner["instructions"]


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


@pytest.mark.parametrize("failure", [
    pytest.param(OSError("private path and internal details"), id="storage"),
    pytest.param(PermissionError(13, "private path and internal details"), id="permission-errno"),
    pytest.param(PermissionError("private path and internal details"), id="permission-no-errno"),
])
def test_failed_atomic_save_keeps_original(client, monkeypatch, failure):
    client.post(BASE, json=BODY)
    path = CodeSkillService().root / "review-typescript" / "SKILL.md"
    before = path.read_bytes()
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(CodeSkillService, "_replace_direct_child", fail)
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
