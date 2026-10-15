"""COWORK_MANAGED_DIR at startup.

create_app refuses a managed folder in org tenancy, one that is missing or
lacks memory/, skills/ or data-vault/, naming each missing folder, one whose
shared-connection registry is malformed, naming the file, and one an anton
without the read-only memory layer would silently drop. A folder with no
shared/ starts: an operator who shares nothing needs none. A complete folder
starts and says what it holds. With the setting unset or blank, the desktop
and hosted servers start as before.
"""
from __future__ import annotations

import json
import logging
import shutil

import pytest

# Imported at module scope on purpose: cowork.server builds an app at import
# time, so importing it inside a test would raise before pytest.raises saw it.
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.managed import ManagedFolder, cortex_takes_managed_layer, managed_folder

SHARED_ENTRY = {
    "user_id": "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d",
    "partition_id": "44e92829-e701-599b-88f9-69f5bf1165eb",
    "engine": "postgres",
    "name": "erp",
}


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


def _set_managed_dir(monkeypatch, value: str) -> None:
    monkeypatch.setenv("COWORK_MANAGED_DIR", value)
    get_app_settings.cache_clear()


def test_blank_reads_as_no_managed_folder(monkeypatch, home):
    _set_managed_dir(monkeypatch, "  ")
    assert get_app_settings().managed_dir is None
    assert managed_folder() is None
    cowork_server.create_app()


def test_unset_starts_the_desktop_as_before(monkeypatch, home):
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    assert managed_folder() is None
    cowork_server.create_app()


def test_unset_starts_hosted_org_tenancy_as_before(monkeypatch, home, org_deployment):
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    cowork_server.create_app()


def test_a_missing_folder_is_refused_naming_every_missing_one(monkeypatch, home):
    root = home / "nowhere"
    _set_managed_dir(monkeypatch, str(root))
    with pytest.raises(RuntimeError) as refused:
        cowork_server.create_app()
    message = str(refused.value)
    for missing in (root, root / "memory", root / "skills", root / "data-vault"):
        assert str(missing) in message


@pytest.mark.parametrize("folder", ["memory", "skills", "data-vault"])
def test_a_missing_inner_folder_is_refused_by_name(managed_dir, folder):
    shutil.rmtree(managed_dir.root / folder)
    with pytest.raises(RuntimeError) as refused:
        cowork_server.create_app()
    message = str(refused.value)
    assert str(managed_dir.root / folder) in message
    others = {"memory", "skills", "data-vault"} - {folder}
    assert not any(str(managed_dir.root / other) in message for other in others)


def test_refused_in_org_tenancy(managed_dir, org_deployment):
    with pytest.raises(RuntimeError, match="COWORK_MANAGED_DIR is not supported in org tenancy"):
        cowork_server.create_app()


def test_refused_on_an_anton_without_the_managed_memory_layer(managed_dir, monkeypatch):
    from anton.core.memory import cortex as anton_cortex

    class CortexWithoutManagedLayer:
        def __init__(self, global_hc, project_hc, mode="autopilot", llm_client=None, episodic=None):
            pass

    monkeypatch.setattr(anton_cortex, "Cortex", CortexWithoutManagedLayer)
    assert cortex_takes_managed_layer() is False
    with pytest.raises(RuntimeError, match="Cortex takes managed_hc"):
        cowork_server.create_app()


def test_the_installed_anton_takes_the_managed_memory_layer():
    """The anton the lockfile installs, with no fixture: a stack that sets
    COWORK_MANAGED_DIR refuses to start on an anton without the layer, so a
    pin that loses it must fail here first."""
    assert cortex_takes_managed_layer(), (
        "the pinned anton-agent's Cortex takes no managed_hc: move the anton pin "
        "(the pyproject.toml floor, its [tool.uv.sources] rev and uv.lock) to a "
        "release that has it"
    )


def test_a_complete_folder_starts_and_says_what_it_holds(managed_dir, caplog, owned_logger):
    (managed_dir.skills_dir / "team-report").mkdir()
    (managed_dir.skills_dir / "team-report" / "SKILL.md").write_text(
        "---\nname: team-report\ndescription: Write the team report.\n---\nWrite it.\n"
    )
    from anton.core.datasources.data_vault import LocalDataVault

    LocalDataVault(managed_dir.vault_dir).save("postgres", "warehouse", {"host": "db"})
    managed_dir.shared_dir.mkdir()
    managed_dir.shared_connections_file.write_text(json.dumps([SHARED_ENTRY]))

    owned_logger("cowork.common.logger", level=logging.INFO)
    cowork_server.create_app()
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == "cowork.common.logger" and record.levelno == logging.INFO
    ]
    assert (
        f"managed folder: {managed_dir.root} (1 skills, 1 connections, 1 shared connections)" in lines
    )


def test_a_folder_with_no_shared_folder_starts_sharing_nothing(managed_dir):
    assert not managed_dir.shared_dir.exists()
    cowork_server.create_app()
    assert managed_dir.shared_connections().root == ()


@pytest.mark.parametrize(
    "content",
    ['{"user_id": "x"}', "[{\"engine\": \"postgres\"}]", "not json"],
    ids=["not-a-list", "missing-fields", "not-json"],
)
def test_a_malformed_registry_is_refused_naming_the_file(managed_dir, content):
    managed_dir.shared_dir.mkdir()
    managed_dir.shared_connections_file.write_text(content)
    with pytest.raises(RuntimeError, match="shared-connection registry is malformed") as refused:
        cowork_server.create_app()
    assert str(managed_dir.shared_connections_file) in str(refused.value)


def test_the_folder_model_names_its_folders(tmp_path):
    folder = ManagedFolder.at(tmp_path)
    assert (folder.memory_dir, folder.skills_dir, folder.vault_dir, folder.shared_dir) == (
        tmp_path / "memory",
        tmp_path / "skills",
        tmp_path / "data-vault",
        tmp_path / "shared",
    )
    assert folder.shared_connections_file == tmp_path / "shared" / "connections.json"
    assert folder.missing_dirs() == [tmp_path / "memory", tmp_path / "skills", tmp_path / "data-vault"]
