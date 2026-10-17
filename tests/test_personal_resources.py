"""The per-resource switch: every partitioned table and store has a resource,
the dependency rule holds, and COWORK_PERSONAL_RESOURCES parses as documented.

The completeness checks read the code itself, so a new table with an org_id or
created_by column, or a new store name passed to a scoped storage root, fails
here until cowork/db/personal_resources.py maps it. Unmapped, ScopedSession and
the storage roots partition it wherever anything is partitioned; mapped, the
operator decides whether it is personal.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlmodel import SQLModel

import cowork
from cowork.common.settings.app_settings import AppSettings, get_app_settings
from cowork.db.personal_resources import (
    ALL_PERSONAL_RESOURCES,
    MODEL_RESOURCE,
    STORE_RESOURCE,
    PersonalResource,
    PersonalResourceDependencyError,
    check_personal_resource_dependencies,
)

COWORK_ROOT = Path(cowork.__file__).parent
_ROOT_FUNCTIONS = {"scoped_storage_root", "scoped_user_storage_root"}


def _cowork_tables() -> dict[str, set[str]]:
    """Table name to column names, for every table a cowork model declares.

    Filtered by the model's module so the throwaway tables other test modules
    register on the same metadata don't count.
    """
    import cowork.models as models_pkg

    for _, name, _ in pkgutil.iter_modules(models_pkg.__path__):
        importlib.import_module(f"cowork.models.{name}")
    tables: dict[str, set[str]] = {}
    for mapper in SQLModel._sa_registry.mappers:
        model = mapper.class_
        if not model.__module__.startswith("cowork."):
            continue
        table = getattr(model, "__table__", None)
        if table is not None:
            tables[table.name] = set(table.columns.keys())
    return tables


def _store_names_passed_to_roots() -> tuple[set[str], list[str]]:
    """Every literal ``store=`` given to a scoped storage root, every
    ``store_name`` class attribute (SkillService passes it through), and the
    call sites whose store is neither, which the test refuses."""
    literals: set[str] = set()
    dynamic: list[str] = []
    for path in sorted(COWORK_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "store_name" for target in node.targets
            ):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    literals.add(node.value.value)
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in _ROOT_FUNCTIONS:
                continue
            for keyword in node.keywords:
                if keyword.arg != "store":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    literals.add(value.value)
                elif ast.unparse(value) == "MEMORY_STORE":
                    literals.add("memory")  # the shared memory-store-name constant
                elif ast.unparse(value) in {"store", "self.store_name"}:
                    continue  # scoped_user_storage_root's and SkillService's pass-through
                else:
                    dynamic.append(f"{path.relative_to(COWORK_ROOT)}:{node.lineno} store={ast.unparse(value)}")
    return literals, dynamic


def test_every_table_with_org_id_or_created_by_has_a_resource():
    tables = _cowork_tables()
    partitioned = {name for name, columns in tables.items() if columns & {"org_id", "created_by"}}
    assert "conversations" in partitioned and "messages" in partitioned  # the scan sees the models
    assert sorted(partitioned - MODEL_RESOURCE.keys()) == []


def test_resource_map_names_only_real_tables():
    assert sorted(MODEL_RESOURCE.keys() - _cowork_tables().keys()) == []


def test_every_store_passed_to_a_scoped_root_has_a_resource():
    literals, dynamic = _store_names_passed_to_roots()
    assert dynamic == [], "a store name the scan can't read; pass a literal so it can be mapped"
    assert {"projects", "files", "skills", "data-vault", "memory"} <= literals  # the scan sees the calls
    assert sorted(literals - STORE_RESOURCE.keys()) == []


# Files that read a store root straight from settings without scoped_storage_root
# or scoped_user_storage_root, each for a reason that makes the unscoped read
# correct. A NEW store read not on this list fails the test below, so a
# per-person store built straight from settings can't ship unpartitioned
# unnoticed. Keyed by file for stability against line shifts.
_ALLOWED_UNSCOPED_ROOT_READS: dict[str, str] = {
    "common/settings/app_settings.py": "defines the root_dir/vault_dir settings fields and their cowork_home() defaults",
    "harnesses/anton_harness/settings.py": "anton harness settings defaults under cowork_home()",
    "harnesses/anton_harness/harness.py": "cowork_home() for the turn scratch root, not a partitioned store",
    "harnesses/memory/layout.py": "the memory layout is scoped by GlobalMemoryStore, not here",
    "harnesses/memory/migration.py": "one-time desktop .env->DB migration over the unkeyed root",
    "harnesses/memory/store.py": "base memory root; scoped_user_storage_root wraps it in MemoryStore",
    "db/alembic/versions/93375a6617f4_init.py": "initial migration over the unkeyed root",
    "dev_setup.py": "desktop boot over the unkeyed project root (LOCAL_SCOPE)",
    "migrations.py": ".env->DB migration over the unkeyed root",
    "managed.py": "the operator's managed vault is deployment-wide, read unscoped by design",
    "ops/connections.py": "operator command over the managed vault",
    "server.py": "startup reads of the managed vault and cowork_home, deployment-wide",
    "services/connectors/persist.py": "the managed vault is deployment-wide (own_vault_root scopes the person's own)",
    "services/connectors/probe.py": "cowork_home() for a desktop probe, not a partitioned store",
    "code_sandbox.py": "the code sandbox's trusted and forbidden roots and cowork-home's own entries; "
    "its sandbox store goes through scoped_storage_root (_sandbox_root)",
    "services/projects.py": "projects_root passes this into scoped_storage_root at its one call site",
    "services/skill_links.py": "the unpartitioned fan-out only runs where skills and projects stay shared (links_apply_to)",
    "services/skills.py": "remove_project_memory_tiers walks the shared root with an O_NOFOLLOW guard",
    "api/v1/endpoints/settings.py": "cowork_home() for the .env a desktop setting writes",
    "coding/engines/codex.py": "the coding product's skill root; the proxy refuses coding on the shipped stack",
    "coding/service.py": "the coding store is owner-checked and the proxy refuses coding on the shipped stack",
    "streaming/backend.py": "turn stream buffers are owner-checked, not a per-person filesystem store",
}


def _files_with_unscoped_root_reads() -> set[str]:
    """Every file that reads a ``*.root_dir``/``*.vault_dir``, ``StreamSettings().dir``
    or ``cowork_home() / ...`` store root without passing it through
    scoped_storage_root or scoped_user_storage_root."""
    flagged: set[str] = set()
    for path in sorted(COWORK_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parent: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[id(child)] = node
        scoped_first_args: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name in _ROOT_FUNCTIONS and node.args:
                    scoped_first_args.add(id(node.args[0]))

        def _inside_scoped(node: ast.AST) -> bool:
            cur: ast.AST | None = node
            for _ in range(8):
                if cur is None or id(cur) in scoped_first_args:
                    return cur is not None
                cur = parent.get(id(cur))
            return False

        for node in ast.walk(tree):
            target = None
            if isinstance(node, ast.Attribute) and node.attr in {"root_dir", "vault_dir"}:
                target = node
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == "dir"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "StreamSettings"
            ):
                target = node
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "cowork_home"
                and isinstance(parent.get(id(node)), ast.BinOp)
                and isinstance(parent[id(node)].op, ast.Div)
            ):
                target = parent[id(node)]
            if target is not None and not _inside_scoped(target):
                flagged.add(path.relative_to(COWORK_ROOT).as_posix())
    return flagged


def test_no_store_root_is_read_unscoped_outside_the_allowlist():
    flagged = _files_with_unscoped_root_reads()
    # The scan sees the known intentional readers (so it isn't silently empty).
    assert "coding/service.py" in flagged and "streaming/backend.py" in flagged
    unexplained = sorted(flagged - _ALLOWED_UNSCOPED_ROOT_READS.keys())
    assert unexplained == [], (
        "a store root is read without scoped_storage_root/scoped_user_storage_root; "
        "scope it, or add it to _ALLOWED_UNSCOPED_ROOT_READS with a reason: "
        + ", ".join(unexplained)
    )


def test_store_map_names_only_real_stores():
    literals, _ = _store_names_passed_to_roots()
    assert sorted(STORE_RESOURCE.keys() - literals) == []


# ── dependencies ─────────────────────────────────────────────────────────────

def test_every_resource_together_is_valid():
    check_personal_resource_dependencies(ALL_PERSONAL_RESOURCES)


def test_no_resource_is_valid():
    check_personal_resource_dependencies(frozenset())


@pytest.mark.parametrize(
    "resource",
    [PersonalResource.SKILLS, PersonalResource.CONNECTIONS, PersonalResource.MEMORY, PersonalResource.SETTINGS],
)
def test_independent_resources_stand_alone(resource):
    check_personal_resource_dependencies(frozenset({resource}))


def test_projects_requires_conversations():
    resources = ALL_PERSONAL_RESOURCES - {PersonalResource.CONVERSATIONS}
    with pytest.raises(PersonalResourceDependencyError, match="projects without conversations"):
        check_personal_resource_dependencies(resources)


@pytest.mark.parametrize("missing", ["files", "pins", "schedules"])
def test_conversations_requires_files_pins_and_schedules(missing):
    resources = ALL_PERSONAL_RESOURCES - {PersonalResource.PROJECTS, PersonalResource(missing)}
    with pytest.raises(PersonalResourceDependencyError, match=f"conversations without {missing}"):
        check_personal_resource_dependencies(resources)


def test_the_message_names_every_missing_resource():
    with pytest.raises(PersonalResourceDependencyError, match="without files, pins, schedules"):
        check_personal_resource_dependencies(frozenset({PersonalResource.CONVERSATIONS}))


# ── COWORK_PERSONAL_RESOURCES ────────────────────────────────────────────────

@pytest.fixture
def _settings_env(monkeypatch):
    get_app_settings.cache_clear()
    yield monkeypatch
    get_app_settings.cache_clear()


def test_default_is_every_resource(_settings_env):
    _settings_env.delenv("COWORK_PERSONAL_RESOURCES", raising=False)
    assert AppSettings().personal_resources == ALL_PERSONAL_RESOURCES


def test_blank_means_none(_settings_env):
    _settings_env.setenv("COWORK_PERSONAL_RESOURCES", "")
    assert AppSettings().personal_resources == frozenset()


def test_comma_list_with_spaces(_settings_env):
    _settings_env.setenv("COWORK_PERSONAL_RESOURCES", " memory , skills,,settings ")
    assert AppSettings().personal_resources == {
        PersonalResource.MEMORY, PersonalResource.SKILLS, PersonalResource.SETTINGS,
    }


def test_unknown_name_fails_settings_load(_settings_env):
    _settings_env.setenv("COWORK_PERSONAL_RESOURCES", "memory,chats")
    with pytest.raises(ValidationError, match="chats"):
        AppSettings()


def test_trust_identity_headers_defaults_off(_settings_env):
    _settings_env.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    assert AppSettings().trust_identity_headers is False


# ── memory off and memory shared (cowork.harnesses.memory.policy) ───────────

SHARES_MEMORY = "conversations,projects,files,skills,connections,schedules,pins,settings"


def test_memory_left_off_the_list_is_shared_and_warns_that_turns_learn_into_one_store(
    personal_deployment, _settings_env
):
    from cowork.harnesses.memory.policy import resolve_memory_policy

    _settings_env.setenv("COWORK_PERSONAL_RESOURCES", SHARES_MEMORY)
    _settings_env.delenv("MEMORY_ENABLED", raising=False)
    get_app_settings.cache_clear()

    policy = resolve_memory_policy()

    assert (policy.shared, policy.off) == (True, False)
    warning = policy.start_warning()
    assert "every person's turns learn into one store" in warning
    assert "profile included" in warning
    assert "no turn learns" not in warning


def test_shared_memory_turned_off_warns_that_the_store_is_kept(personal_deployment, _settings_env):
    from cowork.harnesses.memory.policy import resolve_memory_policy

    _settings_env.setenv("COWORK_PERSONAL_RESOURCES", SHARES_MEMORY)
    _settings_env.setenv("MEMORY_ENABLED", "false")
    get_app_settings.cache_clear()

    policy = resolve_memory_policy()

    assert (policy.shared, policy.off) == (True, True)
    warning = policy.start_warning()
    assert "MEMORY_ENABLED is off: no turn learns" in warning
    assert "the shared store is kept, unread, until memory goes back on" in warning
    assert "learn into one store" not in warning


@pytest.mark.parametrize("enabled, off", [("true", False), ("false", True)])
def test_memory_kept_per_person_warns_of_nothing(personal_deployment, _settings_env, enabled, off):
    from cowork.harnesses.memory.policy import resolve_memory_policy

    _settings_env.setenv("MEMORY_ENABLED", enabled)
    get_app_settings.cache_clear()

    policy = resolve_memory_policy()

    assert (policy.shared, policy.off) == (False, off)
    assert policy.start_warning() is None


@pytest.mark.parametrize("stored, environment, off", [("false", "true", True), ("true", "false", False)])
def test_a_stored_memory_setting_decides_over_the_environment(
    personal_deployment, _settings_env, stored_memory_enabled, stored, environment, off
):
    """memory_enabled is operator-pinned: the deployment's stored row wins over
    MEMORY_ENABLED, for the operator view and a person alike, as a turn's
    settings resolve it."""
    from cowork.db.scoped import LOCAL_SCOPE, scope_for_identity
    from cowork.harnesses.memory.policy import resolve_memory_policy
    from cowork.principal import personal_partition_id

    _settings_env.setenv("MEMORY_ENABLED", environment)
    get_app_settings.cache_clear()
    stored_memory_enabled(stored)
    deployment = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
    person = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
    person_scope = scope_for_identity(org_id=personal_partition_id(deployment, person), user_id=person)

    assert resolve_memory_policy(scope=LOCAL_SCOPE).off is off
    assert resolve_memory_policy(scope=person_scope).off is off


def test_memory_settings_change_nothing_without_trusted_identity(_settings_env):
    """The desktop: MEMORY_ENABLED only sets anton's memory mode, and memory
    left off the list means nothing, so neither part of the policy applies."""
    from cowork.harnesses.memory.policy import memory_policy, resolve_memory_policy

    _settings_env.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    _settings_env.setenv("COWORK_PERSONAL_RESOURCES", "")
    _settings_env.setenv("MEMORY_ENABLED", "false")
    get_app_settings.cache_clear()

    for policy in (resolve_memory_policy(), memory_policy(memory_enabled=False)):
        assert (policy.shared, policy.off) == (False, False)
        assert policy.start_warning() is None


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_memory_kept_per_person_reads_no_stored_setting_for_the_start_warning(
    personal_deployment, _settings_env, monkeypatch, enabled
):
    """Only shared memory has a start warning, so the stored settings are read
    for it only where memory is shared."""
    from cowork.harnesses.memory import policy

    _settings_env.setenv("MEMORY_ENABLED", enabled)
    get_app_settings.cache_clear()

    def unreadable(*_args, **_kwargs):
        raise AssertionError("the start warning read the stored settings")

    monkeypatch.setattr(policy, "get_user_settings", unreadable)

    assert policy.memory_shared() is False
    assert policy.memory_start_warning() is None


def test_shared_memory_reads_the_stored_setting_for_the_start_warning(
    personal_deployment, _settings_env, stored_memory_enabled
):
    from cowork.harnesses.memory.policy import memory_shared, memory_start_warning

    _settings_env.setenv("COWORK_PERSONAL_RESOURCES", SHARES_MEMORY)
    _settings_env.setenv("MEMORY_ENABLED", "true")
    get_app_settings.cache_clear()
    stored_memory_enabled("false")

    assert memory_shared() is True
    assert "MEMORY_ENABLED is off: no turn learns" in memory_start_warning()
