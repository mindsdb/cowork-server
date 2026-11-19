"""COWORK_MANAGED_DIR at startup.

create_app refuses a managed folder in org tenancy, one that is missing or
lacks memory/, skills/ or data-vault/, naming each missing folder, one whose
shared-connection registry is malformed, naming the file, one whose memory/
the server can't read and search, one beside remote turns, and one an anton
without the read-only memory layer would silently drop. A folder with no
shared/ starts: an operator who shares nothing needs none. A complete folder
starts and says what it holds. With the setting unset or blank, the desktop
and hosted servers start as before.

The skills people share are listed outside the managed folder, in the shared
root (cowork.shared_skills). A deployment that keeps skills personal starts
with a malformed list and warns naming it, and says how many it shares. One
that trusts no identity header reads no list.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import pytest

# Imported at module scope on purpose: cowork.server builds an app at import
# time, so importing it inside a test would raise before pytest.raises saw it.
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.managed import ManagedFolder, cortex_takes_managed_layer, managed_folder
from cowork.shared_skills import read_shared_skills, shared_skills_file

SHARED_ENTRY = {
    "user_id": "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d",
    "partition_id": "44e92829-e701-599b-88f9-69f5bf1165eb",
    "engine": "postgres",
    "name": "erp",
}
SHARED_SKILL_ENTRY = {
    "user_id": "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d",
    "partition_id": "44e92829-e701-599b-88f9-69f5bf1165eb",
    "skill": "team-skill",
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


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX modes, enforced for non-root")
def test_a_memory_folder_the_server_cannot_search_is_refused(managed_dir):
    """anton reads memory/ on every turn, so a folder without the search bit
    would fail each turn; the start stops instead, naming the folder."""
    managed_dir.memory_dir.chmod(0o644)
    try:
        with pytest.raises(RuntimeError, match="memory folder .* can't be read") as refused:
            cowork_server.create_app()
    finally:
        managed_dir.memory_dir.chmod(0o755)
    assert str(managed_dir.memory_dir) in str(refused.value)


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX modes, enforced for non-root")
def test_a_managed_skill_file_the_server_cannot_read_is_refused(managed_dir):
    """A managed skill whose SKILL.md the server can't read would crash every
    person's Skills page and every turn, and loop the restart; the start stops
    instead, naming the file and the fix."""
    skill = managed_dir.skills_dir / "forecast-review"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: forecast-review\n---\nbody\n")
    (skill / "SKILL.md").chmod(0o000)
    try:
        with pytest.raises(RuntimeError, match="skill file .* can't be read") as refused:
            cowork_server.create_app()
    finally:
        (skill / "SKILL.md").chmod(0o644)
    assert str(skill / "SKILL.md") in str(refused.value)
    assert "chmod" in str(refused.value)


def test_an_unreadable_managed_skill_stays_reserved_but_not_readable(managed_dir, caplog, owned_logger, monkeypatch):
    """One managed skill the server can't stat must not crash the listing: it
    drops out of readable but stays reserved, so its name is still refused for a
    create. Driven by a monkeypatch, so it holds whether or not CI runs as root."""
    from cowork.managed import readonly_skill_dirs

    for slug in ("good", "bad"):
        folder = managed_dir.skills_dir / slug
        folder.mkdir()
        (folder / "SKILL.md").write_text(f"---\nname: {slug}\n---\nx\n")
    bad_skill_file = managed_dir.skills_dir / "bad" / "SKILL.md"
    real_is_file = Path.is_file

    def flaky_is_file(self):
        if self == bad_skill_file:
            raise PermissionError(13, "Permission denied")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", flaky_is_file)
    owned_logger("cowork.managed", level=logging.ERROR)

    result = readonly_skill_dirs(managed_dir.skills_dir)

    assert set(result.readable) == {"good"}
    assert result.reserved == frozenset({"good", "bad"})
    assert any(
        "bad" in record.getMessage() and "chmod" in record.getMessage()
        for record in caplog.records
        if record.name == "cowork.managed" and record.levelno == logging.ERROR
    )


def test_a_skill_root_that_is_a_file_reads_as_empty_without_an_error(tmp_path, caplog, owned_logger):
    """A read-only skill root that is missing or a file holds no skills, and
    logs nothing: only a root the server's user can't list is an error."""
    from cowork.managed import ManagedSkills, readonly_skill_dirs

    root = tmp_path / "not-a-folder"
    root.write_text("")
    owned_logger("cowork.managed", level=logging.ERROR)

    assert readonly_skill_dirs(root) == ManagedSkills({}, frozenset())
    assert readonly_skill_dirs(tmp_path / "missing") == ManagedSkills({}, frozenset())
    assert not [record for record in caplog.records if record.name == "cowork.managed"]


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX modes, enforced for non-root")
def test_a_managed_connection_file_the_server_cannot_read_is_refused(managed_dir):
    """A managed connection file the server can't read vanishes for everyone and
    frees its name, so a tester could save their own under it; the start stops
    instead, naming the file and the chown."""
    from anton.core.datasources.data_vault import LocalDataVault

    LocalDataVault(managed_dir.vault_dir).save("postgres", "warehouse", {"host": "w.example"})
    vault_file = next(path for path in managed_dir.vault_dir.iterdir() if path.is_file())
    vault_file.chmod(0o000)
    try:
        with pytest.raises(RuntimeError, match="connection file .* can't be read") as refused:
            cowork_server.create_app()
    finally:
        vault_file.chmod(0o644)
    assert str(vault_file) in str(refused.value)
    assert "chown" in str(refused.value)


def test_refused_with_remote_turns(managed_dir, monkeypatch):
    """A remote turn runs on a worker that receives none of the managed folder."""
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    with pytest.raises(RuntimeError, match="COWORK_TURN_BACKEND=remote is not supported"):
        cowork_server.create_app()


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
        "(the pyproject.toml floor and uv.lock) to a "
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
    assert f"managed folder: {managed_dir.root} (1 skills, 1 connections, 1 shared connections)" in lines


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
    with pytest.raises(RuntimeError, match="shared registry is malformed") as refused:
        cowork_server.create_app()
    assert str(managed_dir.shared_connections_file) in str(refused.value)


@pytest.fixture
def personal_skills(personal_deployment, home, monkeypatch):
    """A deployment that keeps skills personal, its shared root under the
    test's folder."""
    monkeypatch.setenv("COWORK_SHARED_DIR", str(home / "shared-root"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", "managed-folder-test-token")
    get_app_settings.cache_clear()
    return home


def test_a_personal_skills_deployment_says_how_many_skills_are_shared(personal_skills, caplog, owned_logger):
    shared_skills_file().parent.mkdir(parents=True)
    shared_skills_file().write_text(json.dumps([SHARED_SKILL_ENTRY]))

    owned_logger("cowork.common.logger", level=logging.INFO)
    cowork_server.create_app()

    lines = [record.getMessage() for record in caplog.records if record.name == "cowork.common.logger"]
    assert f"shared skills: 1 in {shared_skills_file()}" in lines


def test_a_deployment_that_trusts_no_identity_reads_no_shared_skill_list(
    home, monkeypatch, caplog, owned_logger
):
    """COWORK_PERSONAL_RESOURCES lists skills by default, so the desktop
    would read the list on every start if the check skipped the trust
    setting."""
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    monkeypatch.delenv("COWORK_PERSONAL_RESOURCES", raising=False)
    get_app_settings.cache_clear()

    def read_no_list():
        raise AssertionError("read the shared-skill list")

    monkeypatch.setattr(cowork_server, "read_shared_skills", read_no_list)
    owned_logger("cowork.common.logger", level=logging.INFO)
    cowork_server.create_app()

    lines = [record.getMessage() for record in caplog.records if record.name == "cowork.common.logger"]
    assert not any(line.startswith("shared skills:") for line in lines)


@pytest.mark.parametrize(
    "content",
    ['{"user_id": "x"}', "not json", json.dumps([{**SHARED_SKILL_ENTRY, "skill": "../escape"}])],
    ids=["not-a-list", "not-json", "no-valid-slug"],
)
def test_a_malformed_shared_skill_list_starts_and_warns_naming_it(
    personal_skills, managed_dir, caplog, owned_logger, content
):
    """With a managed folder too: its own registry check no longer reads the
    skill list, so a malformed one never stops the start."""
    shared_skills_file().parent.mkdir(parents=True)
    shared_skills_file().write_text(content)
    owned_logger("cowork.shared_skills", level=logging.WARNING)

    cowork_server.create_app()

    warnings = [record.getMessage() for record in caplog.records if record.name == "cowork.shared_skills"]
    assert warnings and str(shared_skills_file()) in warnings[0]
    assert read_shared_skills().root == ()


def test_the_folder_model_names_its_folders(tmp_path):
    folder = ManagedFolder.at(tmp_path)
    assert (folder.memory_dir, folder.skills_dir, folder.vault_dir, folder.shared_dir) == (
        tmp_path / "memory",
        tmp_path / "skills",
        tmp_path / "data-vault",
        tmp_path / "shared",
    )
    assert folder.shared_connections_file == tmp_path / "shared" / "connections.json"
    # The skills people share are listed in the shared root, not here.
    assert not hasattr(folder, "shared_skills_file")
    assert folder.missing_dirs() == [tmp_path / "memory", tmp_path / "skills", tmp_path / "data-vault"]


def test_a_registry_reads_where_os_has_no_nonblocking_flag(monkeypatch, tmp_path):
    """Windows Python has no os.O_NONBLOCK; reading a registry there must not
    raise AttributeError, even for a missing file."""
    from cowork.managed import SharedConnections

    monkeypatch.delattr(os, "O_NONBLOCK")

    assert SharedConnections.read(tmp_path / "missing.json") == SharedConnections(())
