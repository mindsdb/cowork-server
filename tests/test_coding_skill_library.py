from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from coding_service_fakes import CREDS, FakeEngine, service_with, wait_for_status

from cowork.coding.contracts import SessionCreateRequest, SessionStatus
from cowork.coding.control_errors import StateConflict
from cowork.coding.guidance_items import discover_guidance_items
from cowork.coding.project_models import ProjectCreateRequest, ProjectFolder
from cowork.coding.skill_models import ProjectSkillSource, SkillProjectAssignment, TeamSkillSource
from cowork.coding.workspace import GitRunner, WorkspaceError
from cowork.common.settings.app_settings import get_app_settings
from cowork.services.skills import CodeSkillService, SkillService


class GitSpy(GitRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(self, cwd: Path, *args: str, **kwargs):
        self.calls.append(args)
        return super().run(cwd, *args, **kwargs)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "cowork@example.invalid")
    git(repo, "config", "user.name", "Cowork Test")
    (repo / "README.md").write_text(f"{name}\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    return repo


def add_skill(repo: Path, body: str, name: str = "review") -> str:
    skill = repo / "skills" / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Apply the team's review standard.\n---\n{body}\n",
        encoding="utf-8",
    )
    git(repo, "add", ".")
    git(repo, "commit", "-m", f"skill: {body}")
    return f"skills/{name}/SKILL.md"


def test_code_catalogue_never_inherits_cowork_skills(tmp_path: Path, monkeypatch) -> None:
    storage = tmp_path / "storage"
    monkeypatch.setenv("COWORK_HOME", str(storage))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(storage / "skills"))
    get_app_settings.cache_clear()
    try:
        cowork_skills = SkillService()
        cowork_skills.create_skill(
            label="prepare-documents",
            description="Prepare a polished document.",
            instructions="Use the document workflow.",
        )
        code_skills = CodeSkillService()
        service = service_with(tmp_path / "coding", FakeEngine())

        names = {item.path for item in service.skill_library.catalog(code_skills).items}

        assert code_skills.root != cowork_skills.root
        assert "thermo-nuclear-code-quality-review" in names
        assert "prepare-documents" not in names
    finally:
        get_app_settings.cache_clear()


def test_team_source_is_discoverable_versioned_and_project_scoped(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    skill_path = add_skill(skills_repo, "Review version one.")
    project_repo = repository(tmp_path, "product")
    other_repo = repository(tmp_path, "other")
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(
        ProjectCreateRequest(
            name="Product",
            folders=[ProjectFolder(id="product", name="Product", path=str(project_repo))],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    other = service.projects.create(
        ProjectCreateRequest(
            name="Other",
            folders=[ProjectFolder(id="other", name="Other", path=str(other_repo))],
        )
    )

    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering standards",
    )
    global_item = service.skill_library.list().items[0]
    assert global_item.path == skill_path
    assert global_item.enabled is False
    assert global_item.enabled_project_ids == []

    selected = service.skill_library.set_project_items(project.id, source.id, [skill_path])
    assert selected.items[0].enabled is True
    assert selected.items[0].enabled_project_ids == [project.id]
    assert service.skill_library.list(other.id).items[0].enabled is False

    add_skill(skills_repo, "Review version two.")
    refreshed = service.skill_library.refresh(source.id)
    assert refreshed.update_available is True
    assert refreshed.current_revision != refreshed.available_revision
    applied = service.skill_library.apply_update(source.id)
    assert applied.update_available is False
    assert applied.current_revision == applied.available_revision

    with pytest.raises(WorkspaceError, match="Remove this source from Product"):
        service.skill_library.remove(source.id)


def test_project_creation_saves_validated_skill_configuration_once(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    skill_path = add_skill(skills_repo, "Review standard.")
    project_repo = repository(tmp_path, "product")
    service = service_with(tmp_path, FakeEngine())
    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering standards",
    )

    project = service.projects.create(
        ProjectCreateRequest(
            name="Product",
            folders=[ProjectFolder(id="product", name="Product", path=str(project_repo))],
            skill_sources=[ProjectSkillSource(source_id=source.id, enabled_paths=[skill_path])],
        )
    )

    assert project.skill_sources == [
        ProjectSkillSource(source_id=source.id, enabled_paths=[skill_path])
    ]
    assert service.skill_library.list(project.id).items[0].enabled is True

    with pytest.raises(KeyError, match="Skill source not found"):
        service.projects.create(
            ProjectCreateRequest(
                name="Invalid",
                folders=[ProjectFolder(id="invalid", name="Invalid", path=str(project_repo))],
                skill_sources=[ProjectSkillSource(source_id="missing", enabled_paths=[skill_path])],
            )
        )
    assert [item.name for item in service.projects.list().items] == ["Product"]


def test_project_skill_assignments_are_atomic_across_projects(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    skill_path = add_skill(skills_repo, "Review standard.")
    service = service_with(tmp_path, FakeEngine())
    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering standards",
    )
    projects = [
        service.projects.create(
            ProjectCreateRequest(
                name=name,
                folders=[ProjectFolder(id=name.lower(), name=name, path=str(repository(tmp_path, name.lower())))],
            )
        )
        for name in ("Product", "Inference")
    ]

    service.skill_library.set_project_assignments(
        source.id,
        [
            SkillProjectAssignment(project_id=project.id, enabled_paths=[skill_path])
            for project in projects
        ],
    )
    assert all(service.projects.get(project.id).skill_sources for project in projects)

    with pytest.raises(KeyError, match="Code Project not found"):
        service.skill_library.set_project_assignments(
            source.id,
            [
                SkillProjectAssignment(project_id=projects[0].id, enabled_paths=[]),
                SkillProjectAssignment(project_id="missing", enabled_paths=[skill_path]),
            ],
        )
    assert service.projects.get(projects[0].id).skill_sources


def test_source_catalogue_enforces_repository_branch_uniqueness(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    add_skill(skills_repo, "Review standard.")
    service = service_with(tmp_path, FakeEngine())
    branch = git(skills_repo, "branch", "--show-current")
    source = service.skill_library.add(str(skills_repo), branch, "Engineering standards")

    duplicate = TeamSkillSource(
        id="duplicate",
        name="Duplicate",
        repository=f"{skills_repo}/",
        branch=branch.upper(),
        applied_revision=source.current_revision,
        available_revision=source.available_revision,
        cache_path=str(tmp_path / "unused"),
    )
    with pytest.raises(StateConflict, match="already in the Skills Library"):
        service.skill_library.store.create(duplicate)


def test_task_receives_an_immutable_agent_neutral_skill_snapshot(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    skill_path = add_skill(skills_repo, "Review version one.")
    project_repo = repository(tmp_path, "product")
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    project = service.projects.create(
        ProjectCreateRequest(
            name="Product",
            folders=[ProjectFolder(id="product", name="Product", path=str(project_repo))],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering standards",
    )
    service.skill_library.set_project_items(project.id, source.id, [skill_path])

    first = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Review this project"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, first.id, SessionStatus.completed)
    first = service.get_session(first.id)
    first_root = Path(first.skill_roots[0])
    assert first.resolved_skills[0].version == source.current_revision
    assert first.resolved_skills[0].content_hash
    assert "Review version one." in (first_root / "review" / "SKILL.md").read_text(encoding="utf-8")
    assert engine.configs[0].skill_roots == (str(first_root),)

    add_skill(skills_repo, "Review version two.")
    service.skill_library.refresh(source.id)
    service.skill_library.apply_update(source.id)
    assert "Review version one." in (first_root / "review" / "SKILL.md").read_text(encoding="utf-8")

    second = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Review this project again"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, second.id, SessionStatus.completed)
    second = service.get_session(second.id)
    second_root = Path(second.skill_roots[0])
    assert second_root != first_root
    assert "Review version two." in (second_root / "review" / "SKILL.md").read_text(encoding="utf-8")
    assert second.resolved_skills[0].version != first.resolved_skills[0].version

    service.delete_session(first.id)
    assert not first_root.exists()


def test_skill_document_exposes_contained_text_files_for_read_only_preview(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    add_skill(skills_repo, "Review version one.")
    reference = skills_repo / "skills" / "review" / "references" / "checklist.md"
    reference.parent.mkdir()
    reference.write_text("# Review checklist\n\nCheck the boundaries.\n", encoding="utf-8")
    binary = skills_repo / "skills" / "review" / "fixture.bin"
    binary.write_bytes(b"\x00\x01\x02")
    git(skills_repo, "add", ".")
    git(skills_repo, "commit", "-m", "add review reference")
    service = service_with(tmp_path, FakeEngine())
    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering standards",
    )
    item = service.skill_library.list().items[0]

    main = service.skill_library.document(CodeSkillService(), item.id)
    reference_document = service.skill_library.document(
        CodeSkillService(), item.id, "references/checklist.md"
    )

    assert main.item.source_id == source.id
    assert main.selected_path == "SKILL.md"
    assert main.files == ["SKILL.md", "references/checklist.md"]
    assert "Review version one." in main.content
    assert "Check the boundaries." in reference_document.content

    with pytest.raises(WorkspaceError, match="not part of this skill"):
        service.skill_library.document(CodeSkillService(), item.id, "../../README.md")


def test_skill_document_resolves_by_id_without_running_git(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    add_skill(skills_repo, "Review version one.")
    service = service_with(tmp_path, FakeEngine())
    spy = GitSpy()
    service.skill_library.git = spy
    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering standards",
    )
    item = service.skill_library.list().items[0]
    add_skill(skills_repo, "Review version two.")
    assert service.skill_library.refresh(source.id).update_available is True
    assert spy.calls
    spy.calls.clear()

    document = service.skill_library.document(CodeSkillService(), item.id)
    builtin = service.skill_library.document(CodeSkillService(), "personal:thermo-nuclear-code-quality-review")

    assert spy.calls == []
    assert document.item == item
    assert "Review version one." in document.content
    assert builtin.item.origin == "built_in"
    assert builtin.selected_path == "SKILL.md"
    with pytest.raises(KeyError, match="Skill library item not found"):
        service.skill_library.document(CodeSkillService(), f"{source.id}:skills/missing/SKILL.md")
    with pytest.raises(KeyError, match="Skill library item not found"):
        service.skill_library.document(CodeSkillService(), "personal:../escape")


def test_catalogue_folds_a_personal_skill_under_the_enabled_team_skill_of_the_same_name(
    tmp_path: Path, monkeypatch
) -> None:
    storage = tmp_path / "storage"
    monkeypatch.setenv("COWORK_HOME", str(storage))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(storage / "skills"))
    get_app_settings.cache_clear()
    try:
        code_skills = CodeSkillService()
        code_skills.create_skill(label="Review", description="My own review.", instructions="Review my way.")
        skills_repo = repository(tmp_path, "engineering-skills")
        skill_path = add_skill(skills_repo, "Review the team's way.")
        project_repo = repository(tmp_path, "product")
        service = service_with(tmp_path / "coding", FakeEngine())
        project = service.projects.create(
            ProjectCreateRequest(
                name="Product",
                folders=[ProjectFolder(id="product", name="Product", path=str(project_repo))],
            )
        )
        source = service.skill_library.add(
            str(skills_repo),
            git(skills_repo, "branch", "--show-current"),
            "Engineering standards",
        )

        unassigned = service.skill_library.catalog(code_skills, project.id)
        service.skill_library.set_project_items(project.id, source.id, [skill_path])
        assigned = service.skill_library.catalog(code_skills, project.id)
        unscoped = service.skill_library.catalog(code_skills)

        assert {item.id for item in unassigned.items} >= {f"{source.id}:{skill_path}", "personal:review"}
        reviews = [item for item in assigned.items if item.name.casefold() == "review"]
        assert [(item.id, item.origin, item.source_name) for item in reviews] == [
            (f"{source.id}:{skill_path}", "team", "Engineering standards")
        ]
        assert [(item.id, item.source_name) for item in reviews[0].supersedes] == [("personal:review", "Yours")]
        assert {item.id for item in unscoped.items} >= {f"{source.id}:{skill_path}", "personal:review"}
        assert all(item.supersedes == [] for item in unscoped.items)
    finally:
        get_app_settings.cache_clear()


def test_team_instructions_flow_through_the_agent_neutral_runtime_contract(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-guidance")
    (skills_repo / "AGENTS.md").write_text("Always run the focused checks.\n", encoding="utf-8")
    git(skills_repo, "add", "AGENTS.md")
    git(skills_repo, "commit", "-m", "add team instructions")
    project_repo = repository(tmp_path, "product")
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    project = service.projects.create(
        ProjectCreateRequest(
            name="Product",
            folders=[ProjectFolder(id="product", name="Product", path=str(project_repo))],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering guidance",
    )
    service.skill_library.set_project_items(project.id, source.id, ["AGENTS.md"])

    created = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Apply the team guidance"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    session = service.get_session(created.id)
    assert session.skill_roots == []
    assert session.resolved_skills[0].kind == "instructions"
    assert session.resolved_skills[0].version == source.current_revision
    assert "Always run the focused checks." in session.skill_instructions
    assert "Always run the focused checks." in engine.configs[0].developer_instructions


def test_guidance_discovery_and_snapshots_ignore_symlinks(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    add_skill(skills_repo, "Safe instructions.")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not include", encoding="utf-8")
    link = skills_repo / "skills" / "review" / "secret.txt"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this platform: {exc}")
    git(skills_repo, "add", ".")
    git(skills_repo, "commit", "-m", "add symlink")

    service = service_with(tmp_path, FakeEngine())
    project_repo = repository(tmp_path, "product")
    project = service.projects.create(
        ProjectCreateRequest(
            name="Product",
            folders=[ProjectFolder(id="product", name="Product", path=str(project_repo))],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering standards",
    )
    skill_path = service.skill_library.list().items[0].path
    service.skill_library.set_project_items(project.id, source.id, [skill_path])
    created = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Use the standard"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    snapshot = Path(service.get_session(created.id).skill_roots[0])
    assert not (snapshot / "review" / "secret.txt").exists()
    assert "do not include" not in "".join(
        path.read_text(encoding="utf-8") for path in snapshot.rglob("*") if path.is_file()
    )


def test_project_rejects_ambiguous_team_skill_names(tmp_path: Path) -> None:
    first_repo = repository(tmp_path, "first-skills")
    first_path = add_skill(first_repo, "First review standard.")
    second_repo = repository(tmp_path, "second-skills")
    second_path = add_skill(second_repo, "Second review standard.")
    project_repo = repository(tmp_path, "product")
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(
        ProjectCreateRequest(
            name="Product",
            folders=[ProjectFolder(id="product", name="Product", path=str(project_repo))],
        )
    )
    branch = git(first_repo, "branch", "--show-current")
    first = service.skill_library.add(str(first_repo), branch, "First standards")
    second = service.skill_library.add(str(second_repo), branch, "Second standards")
    service.skill_library.set_project_items(project.id, first.id, [first_path])

    with pytest.raises(WorkspaceError, match="already includes a team skill named 'review'"):
        service.skill_library.set_project_items(project.id, second.id, [second_path])


def test_source_update_cannot_remove_an_item_used_by_a_project(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    skill_path = add_skill(skills_repo, "Review standard.")
    project_repo = repository(tmp_path, "product")
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(
        ProjectCreateRequest(
            name="Product",
            folders=[ProjectFolder(id="product", name="Product", path=str(project_repo))],
        )
    )
    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering standards",
    )
    service.skill_library.set_project_items(project.id, source.id, [skill_path])
    (skills_repo / skill_path).unlink()
    (skills_repo / "AGENTS.md").write_text("Replacement guidance.\n", encoding="utf-8")
    git(skills_repo, "add", "-A")
    git(skills_repo, "commit", "-m", "remove review skill")
    service.skill_library.refresh(source.id)

    with pytest.raises(WorkspaceError, match="removes an item used by Product"):
        service.skill_library.apply_update(source.id)

    current = service.skill_library.list(project.id)
    assert current.sources[0].update_available is True
    assert current.items[0].path == skill_path


def test_missing_source_cache_does_not_hide_the_rest_of_the_library(tmp_path: Path) -> None:
    skills_repo = repository(tmp_path, "engineering-skills")
    add_skill(skills_repo, "Review standard.")
    service = service_with(tmp_path, FakeEngine())
    source = service.skill_library.add(
        str(skills_repo),
        git(skills_repo, "branch", "--show-current"),
        "Engineering standards",
    )
    cache = Path(service.skill_library.store.get(source.id).cache_path)
    cache.rename(cache.with_name(f"{cache.name}-missing"))

    page = service.skill_library.list()

    assert page.items == []
    assert page.sources[0].error == "The managed skill source cache is unavailable; reconnect the source"


def test_discovery_skips_vendor_trees_and_only_indexes_github_workflows(tmp_path: Path) -> None:
    (tmp_path / "skills" / "release").mkdir(parents=True)
    (tmp_path / "skills" / "release" / "SKILL.md").write_text("Release safely.\n", encoding="utf-8")
    (tmp_path / "node_modules" / "dependency").mkdir(parents=True)
    (tmp_path / "node_modules" / "dependency" / "SKILL.md").write_text("Ignore me.\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "verify.yml").write_text("name: verify\n", encoding="utf-8")
    (tmp_path / ".github" / "dependabot.yml").write_text("version: 2\n", encoding="utf-8")

    items = discover_guidance_items(tmp_path)

    assert [(item.kind, item.path) for item in items] == [
        ("workflow", ".github/workflows/verify.yml"),
        ("skill", "skills/release/SKILL.md"),
    ]
