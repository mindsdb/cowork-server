"""COWORK_HOME data-root isolation.

Every cowork path must derive from a single root so preview/stable desktop
builds can be fully isolated from a user's production ~/.cowork (ENG-324) by
setting one env var. These tests pin that contract.
"""
import ast
from collections import Counter
from pathlib import Path

from cowork.common.paths import cowork_home, pod_local_only
from cowork.common.settings.app_settings import (
    AppSettings,
    CodingSettings,
    OAuthSettings,
    StreamSettings,
    _env_file_chain,
    get_app_settings,
)
from cowork.harnesses.anton_harness.settings import AntonHarnessSettings


def test_cowork_home_defaults_to_dot_cowork(monkeypatch):
    monkeypatch.delenv("COWORK_HOME", raising=False)
    assert cowork_home() == Path.home() / ".cowork"


def test_cowork_home_honors_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "cowork-preview"))
    assert cowork_home() == tmp_path / "cowork-preview"


def test_cowork_home_expands_user(monkeypatch):
    monkeypatch.setenv("COWORK_HOME", "~/.cowork-preview")
    assert cowork_home() == Path.home() / ".cowork-preview"


def test_isolated_build_does_not_inherit_legacy_anton_env(monkeypatch, tmp_path):
    # An isolated build (COWORK_HOME set) must NOT read ~/.anton/.env — a path
    # var there (DATABASE_URI, …) would resolve every build onto the same DB
    # and defeat the isolation. Only <COWORK_HOME>/.env and local .env apply.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "cowork-preview"))
    legacy = str(Path.home() / ".anton" / ".env")
    assert legacy not in _env_file_chain()


def test_prod_build_still_reads_legacy_anton_env(monkeypatch):
    # The default (prod) home keeps the legacy fallback for un-migrated
    # installs, ordered BEFORE <COWORK_HOME>/.env so the migrated file wins.
    monkeypatch.delenv("COWORK_HOME", raising=False)
    chain = _env_file_chain()
    legacy = str(Path.home() / ".anton" / ".env")
    assert legacy in chain
    assert chain.index(legacy) < chain.index(str(cowork_home() / ".env"))


# Per-resource env vars that, when set, intentionally win over the
# COWORK_HOME-derived default. The test harness (conftest) injects some of
# these, so clear them all to observe the pure derivation.
_PER_RESOURCE_OVERRIDES = [
    "DATABASE_URI",
    "MASTER_KEY_PATH",
    "STATE_PATH",
    "COWORK_PROJECTS_DIR",
    "PROJECTS_ROOT_DIR",
    "COWORK_FILES_DIR",
    "FILES_ROOT_DIR",
    "COWORK_SKILLS_DIR",
    "SKILLS_ROOT_DIR",
    "COWORK_VAULT_DIR",
    "CONNECTOR_VAULT_DIR",
    "COWORK_STREAMS_DIR",
    "COWORK_MEMORY_DIR",
    "MEMORY_ROOT_DIR",
    "COWORK_CODING_DIR",
    "ANTON_SKILLS_ROOT_DIR",
]


def test_all_settings_paths_derive_from_cowork_home(monkeypatch, tmp_path):
    home = tmp_path / "cowork-preview"
    monkeypatch.setenv("COWORK_HOME", str(home))
    for var in _PER_RESOURCE_OVERRIDES:
        monkeypatch.delenv(var, raising=False)
    get_app_settings.cache_clear()

    s = AppSettings(_env_file=None)
    assert s.database.uri == f"sqlite:///{home / 'cowork.db'}"
    assert Path(s.project.root_dir) == home / "projects"
    assert Path(s.file.root_dir) == home / "files"
    assert Path(s.skill.root_dir) == home / "skills"
    assert Path(s.connector.vault_dir) == home / "data-vault"
    assert Path(s.memory.root_dir) == home / "memory"
    assert Path(s.coding.root_dir) == home / "coding"
    assert Path(s.master_key_path) == home / ".master_key"
    assert Path(StreamSettings(_env_file=None).dir) == home / "streams"
    assert Path(OAuthSettings(_env_file=None).state_path) == home / "oauth_state.json"
    assert Path(AntonHarnessSettings(_env_file=None).skills_root_dir) == home / "anton" / "skills"

    get_app_settings.cache_clear()


def test_explicit_database_uri_still_overrides_cowork_home(monkeypatch, tmp_path):
    # Per-resource env vars keep their precedence over the derived default.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DATABASE_URI", "sqlite:////tmp/explicit.db")
    get_app_settings.cache_clear()

    assert AppSettings(_env_file=None).database.uri == "sqlite:////tmp/explicit.db"

    get_app_settings.cache_clear()


def test_coding_root_derives_from_cowork_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.delenv("COWORK_CODING_DIR", raising=False)
    get_app_settings.cache_clear()

    assert Path(CodingSettings(_env_file=None).root_dir) == home / "coding"

    get_app_settings.cache_clear()


def test_explicit_coding_dir_still_overrides_cowork_home(monkeypatch, tmp_path):
    # The desktop app points this at a per-organization subtree so one
    # organization's code tasks, workspaces and code projects stay its own.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_CODING_DIR", "/explicit/coding")
    get_app_settings.cache_clear()

    assert CodingSettings(_env_file=None).root_dir == "/explicit/coding"

    get_app_settings.cache_clear()


def test_coding_service_honors_the_coding_dir_override(monkeypatch, tmp_path):
    """The behavioural half: the service must READ the setting, not recompute
    the path. Without that, the desktop's per-organization override is ignored
    and one organization's code tasks stay visible in the next."""
    from cowork.coding.service import get_coding_service

    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_CODING_DIR", str(tmp_path / "orgs" / "org-b" / "coding"))
    get_app_settings.cache_clear()
    get_coding_service.cache_clear()
    try:
        assert get_coding_service().root == tmp_path / "orgs" / "org-b" / "coding"
    finally:
        get_coding_service.cache_clear()
        get_app_settings.cache_clear()


def test_explicit_state_path_still_overrides_cowork_home(monkeypatch, tmp_path):
    # OAuthSettings.state_path has no validation_alias, so pydantic-settings
    # falls back to the bare uppercased field name: STATE_PATH, not a
    # COWORK_-prefixed name (same pattern as MASTER_KEY_PATH). The cowork-server
    # Helm values file relies on this exact name to keep OAuth state off the
    # shared EFS tree; this pins it so a future validation_alias addition
    # can't silently change the env var cloud config depends on.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATE_PATH", "/home/app/oauth_state.json")

    assert OAuthSettings(_env_file=None).state_path == "/home/app/oauth_state.json"


# pod_local_only, the mechanism that keeps scratch/deployment-local state
# (connector-probe credential files, publish's state.json, the anton
# harness's temp data-vault dir) off the shared COWORK_HOME tree in org mode,
# since none of the three carry an org_id segment for scoped_storage_root to
# key on.
def test_pod_local_only_is_a_noop_in_local_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()

    local_path = tmp_path / "cowork" / "tmp"
    assert pod_local_only(local_path, "tmp") == local_path

    get_app_settings.cache_clear()


def test_pod_local_only_relocates_off_cowork_home_in_org_mode(monkeypatch, tmp_path):
    home = tmp_path / "cowork-shared"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_POD_SCRATCH_DIR", raising=False)
    get_app_settings.cache_clear()

    resolved = pod_local_only(home / "tmp", "tmp")

    assert home not in resolved.parents
    assert resolved != home / "tmp"

    get_app_settings.cache_clear()


def test_pod_local_only_org_mode_defaults_under_system_temp_dir(monkeypatch, tmp_path):
    import tempfile

    home = tmp_path / "cowork-shared"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_POD_SCRATCH_DIR", raising=False)
    get_app_settings.cache_clear()

    resolved = pod_local_only(home / "tmp", "tmp")

    assert resolved == Path(tempfile.gettempdir()) / "cowork" / "tmp"

    get_app_settings.cache_clear()


def test_pod_local_only_honors_explicit_scratch_dir_override(monkeypatch, tmp_path):
    home = tmp_path / "cowork-shared"
    scratch = tmp_path / "pod-scratch"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_POD_SCRATCH_DIR", str(scratch))
    get_app_settings.cache_clear()

    assert pod_local_only(home / "tmp", "tmp") == scratch / "tmp"

    get_app_settings.cache_clear()


def test_bearer_auth_token_env_derives_from_cowork_home(monkeypatch, tmp_path):
    # With COWORK_REQUIRE_AUTH=true the effective token is mirrored to
    # <cowork_home()>/.env so the desktop app can read it. A hardcoded
    # ~/.cowork/.env would leave an isolated build (COWORK_HOME set) writing
    # token state into another install's data home (ENG-868).
    from cowork.server import create_app

    home = tmp_path / "cowork-preview"
    # Redirect the OS home too, so a regression writes into tmp_path instead
    # of the developer's real ~/.cowork/.env.
    fake_os_home = tmp_path / "os-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_os_home))
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_REQUIRE_AUTH", "true")
    monkeypatch.setenv("COWORK_AUTH_TOKEN", "tok-eng-868")
    get_app_settings.cache_clear()
    try:
        create_app()
        env_file = home / ".env"
        assert env_file.exists(), "auth token state must live under cowork_home()"
        assert "COWORK_AUTH_TOKEN=tok-eng-868" in env_file.read_text(encoding="utf-8")
    finally:
        get_app_settings.cache_clear()


# The desktop app gives each organization its own stores by overriding the
# paths below, one per cowork_home()-derived store. A store added here without
# an override there is shared across organizations silently, which is the whole
# failure the overrides exist to prevent — so this fails until someone decides
# which side a new one belongs on.
#
# Mirrors `orgStoreEnv` in cowork's src/main/account-data.ts.
_PER_ORGANIZATION_OVERRIDES = {
    "DATABASE_URI",
    "COWORK_PROJECTS_DIR",
    "COWORK_FILES_DIR",
    "COWORK_SKILLS_DIR",
    "COWORK_VAULT_DIR",
    "COWORK_MEMORY_DIR",
    "COWORK_STREAMS_DIR",
    "COWORK_CODING_DIR",
    "ANTON_SKILLS_ROOT_DIR",
    "ANTON_COWORK_STATE_DIR",
}


def _expression_identity(expression: ast.expr) -> str:
    return ast.dump(expression, include_attributes=False)


def _callsite(path: str, owner: str, expression: str) -> str:
    parsed = ast.parse(expression, mode="eval")
    return f"{path}:{owner}[{_expression_identity(parsed.body)}]"


_APP_SETTINGS_PATH = "cowork/common/settings/app_settings.py"

# Deliberately NOT per organization, each with the reason it stays shared.
_ACCOUNT_LEVEL_COWORK_HOME_CALLS = Counter(
    _callsite(*spec)
    for spec in (
        # The dotenv and provider config must survive organization switches.
        ("cowork/server.py", "create_app.env_path", "cowork_home() / '.env'"),
        ("cowork/migrations.py", "_ENV_PATH", "cowork_home() / '.env'"),
        ("cowork/api/v1/endpoints/settings.py", "_ENV_PATH", "cowork_home() / '.env'"),
        (_APP_SETTINGS_PATH, "_env_file_chain.files", "str(cowork_home() / '.env')"),
        (_APP_SETTINGS_PATH, "OAuthSettings.state_path", "str(cowork_home() / 'oauth_state.json')"),
        (_APP_SETTINGS_PATH, "StorageSettings.shared_root", "str(cowork_home())"),
        (_APP_SETTINGS_PATH, "AppSettings.master_key_path", "str(cowork_home() / '.master_key')"),
        # Per-turn scratch holds nothing that outlives a turn.
        (
            "cowork/harnesses/anton_harness/harness.py",
            "_vault_scratch_dir",
            "pod_local_only(cowork_home() / 'tmp', 'tmp')",
        ),
        (
            "cowork/services/connectors/probe.py",
            "_probe_tmp_dir",
            "pod_local_only(cowork_home() / 'tmp', 'tmp')",
        ),
        # These are read-only sources for a one-time legacy import.
        *(
            ("cowork/harnesses/memory/migration.py", "_MIGRATION_SOURCES", f"cowork_home() / '{path}'")
            for path in (
                "anton/memory/rules.md",
                "anton/memory/lessons.md",
                "anton/memory/profile.md",
                "hermes/memories/USER.md",
                "hermes/memories/MEMORY.md",
            )
        ),
    )
)

_PER_ORGANIZATION_COWORK_HOME_CALLS = {
    _callsite(path, owner, expression): override
    for path, owner, expression, override in (
        (_APP_SETTINGS_PATH, "DatabaseSettings.uri", "f\"sqlite:///{cowork_home() / 'cowork.db'}\"", "DATABASE_URI"),
        (_APP_SETTINGS_PATH, "ProjectSettings.root_dir", "str(cowork_home() / 'projects')", "COWORK_PROJECTS_DIR"),
        (_APP_SETTINGS_PATH, "FileSettings.root_dir", "str(cowork_home() / 'files')", "COWORK_FILES_DIR"),
        (_APP_SETTINGS_PATH, "CodingSettings.root_dir", "str(cowork_home() / 'coding')", "COWORK_CODING_DIR"),
        (_APP_SETTINGS_PATH, "SkillSettings.root_dir", "str(cowork_home() / 'skills')", "COWORK_SKILLS_DIR"),
        (_APP_SETTINGS_PATH, "ConnectorSettings.vault_dir", "str(cowork_home() / 'data-vault')", "COWORK_VAULT_DIR"),
        (_APP_SETTINGS_PATH, "MemorySettings.root_dir", "str(cowork_home() / 'memory')", "COWORK_MEMORY_DIR"),
        (_APP_SETTINGS_PATH, "StreamSettings.dir", "str(cowork_home() / 'streams')", "COWORK_STREAMS_DIR"),
        (
            "cowork/harnesses/anton_harness/settings.py",
            "AntonHarnessSettings.skills_root_dir",
            "str(cowork_home() / 'anton' / 'skills')",
            "ANTON_SKILLS_ROOT_DIR",
        ),
        (
            "cowork/services/publish.py",
            "_cowork_state_dir.path",
            "pod_local_only(cowork_home(), 'publish')",
            "ANTON_COWORK_STATE_DIR",
        ),
    )
}


def _attribute_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _assigned_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _call_expression(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> str:
    """The smallest stable expression that distinguishes this path read."""
    current: ast.expr = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.Lambda):
            current = parent.body
            break
        if not isinstance(parent, ast.expr) or isinstance(
            parent, (ast.Dict, ast.List, ast.Set, ast.Tuple)
        ):
            break
        current = parent
    return _expression_identity(current)


def _cowork_home_calls(path: str, source: str) -> Counter[str]:
    """Every real cowork_home call, keyed by its stable owning symbol.

    Parsed rather than grepped: the name appears in docstrings across the
    codebase. Import resolution covers direct aliases and qualified calls so a
    different import style cannot bypass the organization-store guard.
    """
    tree = ast.parse(source)
    parents = {
        child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
    }
    direct_names: set[str] = set()
    module_names = {"cowork.common.paths"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "cowork.common.paths":
            direct_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "cowork_home"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "cowork.common":
            module_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "paths"
            )
        elif isinstance(node, ast.Import):
            module_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "cowork.common.paths"
            )

    found: Counter[str] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _attribute_name(node.func)
        is_direct = isinstance(node.func, ast.Name) and node.func.id in direct_names
        if not is_direct and not any(
            name == f"{module}.cowork_home" for module in module_names
        ):
            continue

        scopes: list[str] = []
        binding: str | None = None
        current: ast.AST = node
        while current in parents:
            current = parents[current]
            binding = binding or _assigned_name(current)
            if isinstance(current, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scopes.append(current.name)
        owner = ".".join([*reversed(scopes), *([binding] if binding else [])]) or "<module>"
        expression = _call_expression(node, parents)
        found[f"{path}:{owner}[{expression}]"] += 1
    return found


def _all_cowork_home_calls() -> Counter[str]:
    root = Path(__file__).resolve().parent.parent
    found: Counter[str] = Counter()
    for path in (root / "cowork").rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        found.update(_cowork_home_calls(relative, path.read_text(encoding="utf-8")))
    return found


def _unclassified_cowork_home_calls(calls: Counter[str]) -> Counter[str]:
    known = Counter(_ACCOUNT_LEVEL_COWORK_HOME_CALLS)
    known.update(_PER_ORGANIZATION_COWORK_HOME_CALLS.keys())
    return calls - known


def test_every_cowork_home_reader_is_classified():
    calls = _all_cowork_home_calls()
    known = Counter(_ACCOUNT_LEVEL_COWORK_HOME_CALLS)
    known.update(_PER_ORGANIZATION_COWORK_HOME_CALLS.keys())
    unclassified = calls - known
    stale = known - calls
    assert not unclassified, (
        "These call sites derive a path from cowork_home() and are not classified as "
        "per-organization or account-level. Decide which, and if per-organization "
        "add the override to orgStoreEnv in cowork's src/main/account-data.ts: "
        f"{sorted(unclassified.elements())}"
    )
    assert not stale, (
        f"Remove or update stale cowork_home classifications: {sorted(stale.elements())}"
    )
    overrides = list(_PER_ORGANIZATION_COWORK_HOME_CALLS.values())
    assert len(overrides) == len(set(overrides)), "Each store needs its own desktop override"
    assert set(overrides) == _PER_ORGANIZATION_OVERRIDES


def test_cowork_home_scan_catches_new_settings_and_qualified_calls():
    source = """
from cowork.common import paths as store_paths
from cowork.common.paths import cowork_home as data_home

class FutureSettings:
    root_dir = Field(default_factory=lambda: data_home() / "future")

def another_store():
    return store_paths.cowork_home() / "another"
"""
    calls = _cowork_home_calls("cowork/future.py", source)

    assert calls == Counter(
        {
            _callsite(
                "cowork/future.py", "FutureSettings.root_dir", "data_home() / 'future'"
            ): 1,
            _callsite(
                "cowork/future.py",
                "another_store",
                "store_paths.cowork_home() / 'another'",
            ): 1,
        }
    )
    assert _unclassified_cowork_home_calls(calls) == calls
    replaced = _cowork_home_calls("cowork/future.py", source.replace('"future"', '"replacement"'))
    assert replaced != calls
    assert _unclassified_cowork_home_calls(replaced) == replaced


def test_publish_state_dir_is_overridable_per_organization(monkeypatch, tmp_path):
    """publish's state.json holds publish_history, which carries no org_id, so
    left on cowork_home() every organization reads every other's. It is read
    with os.environ.get rather than a settings field, so it needs its own
    assertion."""
    from cowork.services import publish

    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTON_COWORK_STATE_DIR", str(tmp_path / "orgs" / "org-b"))

    assert publish._cowork_state_dir() == tmp_path / "orgs" / "org-b"


def test_every_per_organization_override_is_a_real_knob(monkeypatch, tmp_path):
    """Each name the desktop sets must actually move something. A typo there is
    silent: the store stays on the shared root and the organization reads the
    previous one's data."""
    from cowork.services import publish

    readers = {
        "DatabaseSettings.uri": lambda: AppSettings(_env_file=None).database.uri,
        "ProjectSettings.root_dir": lambda: AppSettings(_env_file=None).project.root_dir,
        "FileSettings.root_dir": lambda: AppSettings(_env_file=None).file.root_dir,
        "CodingSettings.root_dir": lambda: AppSettings(_env_file=None).coding.root_dir,
        "SkillSettings.root_dir": lambda: AppSettings(_env_file=None).skill.root_dir,
        "ConnectorSettings.vault_dir": lambda: AppSettings(_env_file=None).connector.vault_dir,
        "MemorySettings.root_dir": lambda: AppSettings(_env_file=None).memory.root_dir,
        "StreamSettings.dir": lambda: StreamSettings(_env_file=None).dir,
        "AntonHarnessSettings.skills_root_dir": lambda: AntonHarnessSettings(
            _env_file=None
        ).skills_root_dir,
        "_cowork_state_dir.path": publish._cowork_state_dir,
    }
    owners = {
        callsite.split(":", 1)[1].split("[", 1)[0]
        for callsite in _PER_ORGANIZATION_COWORK_HOME_CALLS
    }
    assert set(readers) == owners

    home = tmp_path / "home"
    monkeypatch.setenv("COWORK_HOME", str(home))
    for callsite, env_name in _PER_ORGANIZATION_COWORK_HOME_CALLS.items():
        for name in _PER_ORGANIZATION_OVERRIDES:
            monkeypatch.delenv(name, raising=False)
        owner = callsite.split(":", 1)[1].split("[", 1)[0]
        target = tmp_path / "orgs" / "org-b" / owner.replace(".", "-")
        value = f"sqlite:///{target}" if env_name == "DATABASE_URI" else str(target)
        monkeypatch.setenv(env_name, value)
        get_app_settings.cache_clear()

        resolved = readers[owner]()

        assert str(target) in str(resolved), (
            f"{callsite} did not follow its declared override {env_name}: {resolved}"
        )
        assert str(home) not in str(resolved)

    get_app_settings.cache_clear()
