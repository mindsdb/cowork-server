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
