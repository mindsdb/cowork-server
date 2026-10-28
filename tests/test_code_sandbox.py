"""COWORK_CODE_SANDBOX: what cowork-server hands anton so each person's agent
code runs locked to their own space.

The startup refusals and the shared-project warning, through create_app. The
token's copy in cowork-home's .env. The policy build_cell_sandbox builds, with
the real anton model checking it. What a turn, the credential probe and the
backend pool pass, with the sandbox on and off: off passes exactly what it
passed before. The refusals of app backends (the preview mount, its proxy
and the launch), Code mode, the old .env fallback, and a session or pad built
without the policy. Links agent code left before the sandbox: no grant follows
one, and under the sandbox neither do the server's own folders in a project,
nor its reads and writes of a person's vault, memory and publish store, nor
cowork-home's own entries, while off keeps the desktop's behavior. The kernel
itself is stubbed (conftest's stub_landlock_kernel): anton's own tests run the
launcher, and the one real-kernel test here runs it where Landlock is.
"""
from __future__ import annotations

import asyncio
import getpass
import json
import logging
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# Imported at module scope on purpose: cowork.server builds an app at import
# time (see tests/test_trusted_identity_local.py).
import cowork.server as cowork_server
from cowork import code_sandbox
from cowork.auth_middleware import remove_auth_token
from cowork.code_sandbox import NO_CODE_MODE_DETAIL, PlantedLinkError, build_cell_sandbox
from cowork.common.paths import makedirs_below_nofollow
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import current_settings_scope
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, MissingTenantScopeError, TenantScope
from cowork.harnesses.anton_harness import harness
from cowork.principal import personal_partition_id
from tests._turns import built_turn

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
SUB_C = "33333333-3333-4333-8333-333333333333"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
PARTITION_C = personal_partition_id(DEPLOYMENT, SUB_C)
TOKEN = "code-sandbox-test-token"
SHARED_PROJECTS = "conversations,files,skills,connections,schedules,pins,memory,settings"
SHARED_MEMORY = "conversations,projects,files,skills,connections,schedules,pins,settings"
BACKEND_SENTENCE = (
    "For a service that must keep running, build a full-stack artifact and start it with `launch_backend`."
)


def _person(partition: str, sub: str, resources=ALL_PERSONAL_RESOURCES) -> TenantScope:
    return TenantScope(personal=True, org_id=partition, user_id=sub, personal_resources=resources)


ALICE = _person(PARTITION_A, SUB_A)
BOB = _person(PARTITION_B, SUB_B)


@pytest.fixture
def stack(personal_deployment, monkeypatch, tmp_path):
    """A self-hosted stack: identity headers, the sandbox on, and cowork-home
    at the test's own folder, which is also the shared root."""
    home = tmp_path / "home"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(home))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    yield monkeypatch
    get_app_settings.cache_clear()


@pytest.fixture
def landlock_alone(landlock_kernel, monkeypatch, tmp_path):
    """The sandbox on without identity headers: one person on a Linux host."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_CODE_SANDBOX", "landlock")
    get_app_settings.cache_clear()
    yield monkeypatch
    get_app_settings.cache_clear()


def _refusal() -> str:
    get_app_settings.cache_clear()
    with pytest.raises(RuntimeError) as refused:
        cowork_server.create_app()
    return str(refused.value)


def _never(*_args, **_kwargs):
    raise AssertionError("the sandbox's kernel checks must not run here")


def _hide_the_launcher(monkeypatch) -> None:
    """The installed anton as one without anton.core.backends.sandbox."""
    import anton.core.backends as backends

    monkeypatch.setitem(sys.modules, "anton.core.backends.sandbox", None)
    monkeypatch.delattr(backends, "sandbox", raising=False)


# ── startup ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["off", None])
def test_identity_headers_with_the_sandbox_off_are_refused(stack, value):
    if value is None:
        stack.delenv("COWORK_CODE_SANDBOX")
    else:
        stack.setenv("COWORK_CODE_SANDBOX", value)
    message = _refusal()
    assert message.startswith("COWORK_TRUST_IDENTITY_HEADERS needs COWORK_CODE_SANDBOX=landlock.")
    assert "kernel 6.2 or later" in message


def test_the_desktop_checks_no_kernel(monkeypatch):
    """Off by default, and off runs none of the sandbox's checks."""
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    monkeypatch.setattr(code_sandbox, "_probe_abi", _never)
    monkeypatch.setattr(code_sandbox, "_run_self_test", _never)
    get_app_settings.cache_clear()
    assert get_app_settings().code_sandbox == "off"
    cowork_server.create_app()


@pytest.mark.parametrize("abi", [0, 2])
def test_a_kernel_below_landlock_abi_3_is_refused(stack, abi):
    stack.setattr(code_sandbox, "_probe_abi", lambda: abi)
    stack.setattr(code_sandbox, "_run_self_test", _never)
    message = _refusal()
    assert f"needs Landlock ABI 3 or later, and this kernel reports ABI {abi}" in message
    assert "Linux 6.2 or later" in message
    assert "bookworm-backports" in message


def _landlock_required() -> bool:
    """Whether a missing Landlock fails the real-kernel test instead of
    skipping it: on a Linux runner in GitHub Actions, or wherever
    LANDLOCK_REQUIRED=1 asks for it."""
    on_ci_linux = sys.platform == "linux" and os.environ.get("GITHUB_ACTIONS") == "true"
    return on_ci_linux or os.environ.get("LANDLOCK_REQUIRED") == "1"


def test_the_real_self_test_passes_where_the_kernel_has_landlock(monkeypatch, tmp_path):
    """The seam unstubbed: anton's launcher, in a child process, on this
    kernel. Runs on a Linux runner with Landlock ABI 3 or later, and fails
    there rather than skipping when the kernel has none (_landlock_required)."""
    from anton.core.backends.sandbox import MIN_LANDLOCK_ABI

    abi = code_sandbox._probe_abi()
    if abi < MIN_LANDLOCK_ABI:
        if _landlock_required():
            pytest.fail(
                f"this runner must have Landlock ABI {MIN_LANDLOCK_ABI} or later, and it reports ABI {abi}"
            )
        pytest.skip("this kernel has no usable Landlock")
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    code_sandbox._run_self_test()
    assert list((tmp_path / "home" / "tmp").iterdir()) == []


def test_landlock_abi_3_starts(stack):
    stack.setattr(code_sandbox, "_probe_abi", lambda: 3)
    get_app_settings.cache_clear()
    cowork_server.create_app()


def test_a_failing_self_test_is_refused(stack):
    from anton.core.backends.sandbox import SandboxUnavailable

    def fails() -> None:
        raise SandboxUnavailable("the child read the file beside its grant")

    stack.setattr(code_sandbox, "_run_self_test", fails)
    message = _refusal()
    assert "COWORK_CODE_SANDBOX=landlock: the startup self-test couldn't confirm" in message
    assert "the child read the file beside its grant" in message


def test_an_anton_without_the_launcher_is_refused(stack):
    _hide_the_launcher(stack)
    stack.setattr(code_sandbox, "_probe_abi", _never)
    message = _refusal()
    assert "The installed anton has no anton.core.backends.sandbox" in message
    assert "agent code would run unlocked" in message


def test_an_anton_with_another_launcher_api_is_refused(stack):
    from anton.core.backends import sandbox

    stack.setattr(sandbox, "LAUNCHER_API_VERSION", 2)
    message = _refusal()
    assert "has launcher API version 2" in message
    assert message.endswith("Install the anton this cowork-server pins (launcher API version 1).")


def test_an_anton_without_a_launcher_api_version_is_refused(stack):
    from anton.core.backends import sandbox

    stack.delattr(sandbox, "LAUNCHER_API_VERSION")
    message = _refusal()
    assert "has no LAUNCHER_API_VERSION in anton.core.backends.sandbox" in message
    assert message.endswith("Install the anton this cowork-server pins (launcher API version 1).")


@pytest.mark.parametrize(
    ("module", "name"),
    [("anton.core.session", "ChatSessionConfig"), ("anton.core.backends.local", "LocalScratchpadRuntime")],
)
def test_an_anton_whose_spawning_classes_take_no_policy_is_refused(stack, module, name):
    import importlib

    @dataclass
    class Older:
        llm_client: object = None

    Older.__name__ = name
    stack.setattr(importlib.import_module(module), name, Older)
    assert f"has a {name} that takes no cell_sandbox" in _refusal()


def test_an_anton_whose_policy_model_refuses_the_policy_is_refused(stack):
    """A throwaway policy, built as a session builds one, at start: a model
    without one of the fields would otherwise fail every session."""
    from anton.core.backends import sandbox
    from pydantic import BaseModel, ConfigDict

    class OlderCellSandbox(BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid")

        project_dir: Path
        user_name: str

    stack.setattr(sandbox, "CellSandbox", OlderCellSandbox)
    message = _refusal()
    assert "has a CellSandbox that refuses the policy this server builds" in message
    assert "home_dir" in message


def test_an_anton_whose_workspace_follows_links_is_refused(stack):
    """The harness passes refuse_links directly under the sandbox, so an anton
    without it is refused at start rather than failing every turn."""
    from anton.workspace import Workspace

    def initialize(self, *, create_anton_md: bool = True) -> list[str]:
        return []

    stack.setattr(Workspace, "initialize", initialize)
    assert "has a Workspace.initialize that takes no refuse_links" in _refusal()


def test_landlock_in_org_tenancy_is_refused(landlock_alone):
    landlock_alone.setenv("COWORK_TENANCY_MODE", "org")
    assert _refusal().startswith("COWORK_CODE_SANDBOX=landlock is not supported in org tenancy mode")


def test_landlock_with_remote_turns_is_refused(landlock_alone):
    landlock_alone.setenv("COWORK_TURN_BACKEND", "remote")
    assert _refusal().startswith(
        "COWORK_CODE_SANDBOX=landlock is not supported with COWORK_TURN_BACKEND=remote"
    )


def _sandbox_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "cowork.code_sandbox" and record.levelno == logging.WARNING
    ]


def test_shared_projects_start_with_a_warning(stack, owned_logger, caplog):
    stack.setenv("COWORK_PERSONAL_RESOURCES", SHARED_PROJECTS.replace("connections,", ""))
    owned_logger("cowork.code_sandbox", level=logging.WARNING)
    get_app_settings.cache_clear()
    cowork_server.create_app()
    (warning,) = _sandbox_warnings(caplog)
    assert "projects are left off COWORK_PERSONAL_RESOURCES" in warning
    assert "venvs and saved cell state" in warning


def test_private_connections_with_shared_project_execution_refuse_before_kernel_probe(stack):
    stack.setenv("COWORK_PERSONAL_RESOURCES", SHARED_PROJECTS)
    stack.setattr(code_sandbox, "_probe_abi", _never)
    message = _refusal()
    assert "private connections require personal projects" in message
    assert "COWORK_PERSONAL_RESOURCES" in message
    assert "venvs and saved cell state" in message


def test_personal_projects_start_without_the_warning(stack, owned_logger, caplog):
    owned_logger("cowork.code_sandbox", level=logging.WARNING)
    cowork_server.create_app()
    assert _sandbox_warnings(caplog) == []


@pytest.mark.parametrize(
    "resources, enabled",
    [(SHARED_MEMORY, "true"), (SHARED_MEMORY, "false"), (None, "false")],
    ids=["memory shared", "memory shared and off", "memory personal and off"],
)
def test_memory_shared_or_off_starts_under_the_sandbox(stack, resources, enabled):
    """Memory left off COWORK_PERSONAL_RESOURCES, or MEMORY_ENABLED false,
    starts with the sandbox on: neither changes whose sandbox store a cell
    gets."""
    if resources is not None:
        stack.setenv("COWORK_PERSONAL_RESOURCES", resources)
    stack.setenv("MEMORY_ENABLED", enabled)
    get_app_settings.cache_clear()
    cowork_server.create_app()


def test_the_shared_folder_is_made_at_start(stack, tmp_path):
    cowork_server.create_app()
    assert (tmp_path / "home" / "shared-files").is_dir()


@pytest.mark.parametrize("name", [".env", ".master_key", "streams", "tmp", "shared"])
def test_a_link_at_one_of_cowork_homes_own_entries_is_refused(stack, tmp_path, name):
    """Agent code before the sandbox could swap streams/ for a link into the
    shared folder, and every answer stream would land where every cell reads."""
    home = tmp_path / "home"
    where_it_leads = _project(tmp_path, "shared-files-before", name)
    _plant_link(home / name, where_it_leads)
    message = _refusal()
    assert message.startswith(f"COWORK_CODE_SANDBOX=landlock: {home / name} is a link")
    assert "remove the link and put the real file or folder back" in message
    assert list(where_it_leads.iterdir()) == []


def test_cowork_homes_own_entries_start_as_real_files_and_folders(stack, tmp_path):
    home = tmp_path / "home"
    for name in ("streams", "tmp", "shared"):
        _project(home, name)
    for name in (".env", ".master_key"):
        (home / name).write_text("", encoding="utf-8")
    cowork_server.create_app()


# ── the token's copy in cowork-home's .env ───────────────────────────────────


def test_identity_headers_remove_the_token_an_earlier_start_copied(stack, tmp_path):
    env = tmp_path / "home" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text("FIRST=1\nCOWORK_AUTH_TOKEN=copied-by-an-earlier-start\nLAST=2\n", encoding="utf-8")
    cowork_server.create_app()
    assert env.read_text(encoding="utf-8") == "FIRST=1\nLAST=2\n"


def test_identity_headers_never_copy_the_token(stack, tmp_path):
    cowork_server.create_app()
    env = tmp_path / "home" / ".env"
    assert not env.exists() or TOKEN not in env.read_text(encoding="utf-8")


def test_identity_headers_need_the_token_in_the_environment(stack):
    stack.setenv("COWORK_AUTH_TOKEN", "")
    assert _refusal().startswith(
        "COWORK_TRUST_IDENTITY_HEADERS needs COWORK_AUTH_TOKEN in the server's environment"
    )


def test_identity_headers_never_take_the_token_from_cowork_homes_env(stack, tmp_path):
    """AppSettings also fills auth_token from cowork-home's .env, which every
    cell could read before the sandbox. A token only there is the copy the
    start removes, not one the server may serve."""
    from cowork.common.settings.app_settings import AppSettings

    stack.delenv("COWORK_AUTH_TOKEN")
    env = tmp_path / "home" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text("COWORK_AUTH_TOKEN=only-in-the-file\n", encoding="utf-8")
    # The chain is fixed when the module loads, at the session's cowork-home.
    stack.setitem(AppSettings.model_config, "env_file", [str(env)])
    get_app_settings.cache_clear()
    assert get_app_settings().auth_token == "only-in-the-file"
    assert _refusal().startswith(
        "COWORK_TRUST_IDENTITY_HEADERS needs COWORK_AUTH_TOKEN in the server's environment"
    )


def test_remove_auth_token_keeps_every_other_line(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "A=1\n  COWORK_AUTH_TOKEN = one\nexport COWORK_AUTH_TOKEN=two\nCOWORK_AUTH_TOKEN_NOTE=kept\nB=2",
        encoding="utf-8",
    )
    assert remove_auth_token(env) is True
    assert env.read_text(encoding="utf-8") == "A=1\nCOWORK_AUTH_TOKEN_NOTE=kept\nB=2"


def test_remove_auth_token_replaces_the_file_and_keeps_its_mode(tmp_path):
    """The kept lines are written beside .env and moved over it, never into
    it, so a second name for the old file keeps what it held."""
    env = tmp_path / ".env"
    env.write_text("A=1\nCOWORK_AUTH_TOKEN=one\nB=2\n", encoding="utf-8")
    env.chmod(0o640)
    second_name = tmp_path / "second-name"
    os.link(env, second_name)

    assert remove_auth_token(env) is True

    assert env.read_text(encoding="utf-8") == "A=1\nB=2\n"
    assert env.stat().st_mode & 0o777 == 0o640
    assert second_name.read_text(encoding="utf-8") == "A=1\nCOWORK_AUTH_TOKEN=one\nB=2\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [".env", "second-name"]


def test_remove_auth_token_replaces_a_linked_env_instead_of_writing_through_it(tmp_path):
    elsewhere = _project(tmp_path, "elsewhere") / "settings"
    elsewhere.write_text("A=1\nCOWORK_AUTH_TOKEN=one\n", encoding="utf-8")
    env = tmp_path / ".env"
    env.symlink_to(elsewhere)

    assert remove_auth_token(env) is True

    assert not env.is_symlink()
    assert env.read_text(encoding="utf-8") == "A=1\n"
    assert elsewhere.read_text(encoding="utf-8") == "A=1\nCOWORK_AUTH_TOKEN=one\n"


def test_remove_auth_token_leaves_a_file_without_one(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\n", encoding="utf-8")
    assert remove_auth_token(env) is False
    assert env.read_text(encoding="utf-8") == "A=1\n"
    assert remove_auth_token(tmp_path / "missing.env") is False


# ── the policy ───────────────────────────────────────────────────────────────


def _project(home: Path, *parts: str) -> Path:
    folder = home.joinpath(*parts)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def test_off_builds_no_policy_and_never_imports_the_launcher(monkeypatch, tmp_path):
    _hide_the_launcher(monkeypatch)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    assert build_cell_sandbox(scope=LOCAL_SCOPE, project_dir=tmp_path) is None
    assert not (tmp_path / "home" / "sandbox").exists()


def test_a_persons_policy(stack, tmp_path):
    home = tmp_path / "home"
    project = _project(home, PARTITION_A, "projects", "general")
    _project(home, PARTITION_B)
    attachment_folder = _project(home, PARTITION_A, "files", "f-1")

    policy = build_cell_sandbox(
        scope=ALICE, project_dir=project, read_only_dirs=[attachment_folder, attachment_folder]
    )

    real = home.resolve()
    store = real / PARTITION_A / "sandbox"
    assert policy.project_dir == project.resolve()
    assert policy.home_dir == store / "home" and policy.home_dir.is_dir()
    assert policy.tmp_dir == store / "tmp" and policy.tmp_dir.is_dir()
    assert policy.shared_dir == real / "shared-files"
    assert policy.read_only_dirs == (attachment_folder.resolve(),)
    assert set(policy.forbidden_roots) == {real, real / "shared", real / PARTITION_A, real / PARTITION_B}
    assert policy.user_name == getpass.getuser()
    # anton's own environment builder hands the cell these folders.
    env = policy.child_env({"COWORK_AUTH_TOKEN": "x", "PATH": "/usr/bin"})
    assert (env["HOME"], env["TMPDIR"]) == (str(store / "home"), str(store / "tmp"))
    assert "COWORK_AUTH_TOKEN" not in env


def test_a_build_resolves_each_configured_root_once(stack, tmp_path, monkeypatch):
    """Every grant checks its descendants, but shares this build's root resolution."""
    from collections import Counter

    home = tmp_path / "home"
    project = _project(home, PARTITION_A, "projects", "general")
    attachments = [_project(home, PARTITION_A, "files", str(i)) for i in range(6)]
    roots = set(code_sandbox._trusted_roots())
    real = code_sandbox._real
    calls = Counter()

    def counted(path):
        calls[path] += 1
        return real(path)

    monkeypatch.setattr(code_sandbox, "_real", counted)
    for _ in range(2):
        calls.clear()
        policy = build_cell_sandbox(scope=ALICE, project_dir=project, read_only_dirs=attachments)
        assert policy.read_only_dirs == tuple(path.resolve() for path in attachments)
        assert {root: calls[root] for root in roots} == dict.fromkeys(roots, 1)


def test_a_later_build_rechecks_a_previously_granted_child(stack, tmp_path):
    """A cached root must not make a prior successful descendant check reusable."""
    home = tmp_path / "home"
    project = _project(home, PARTITION_A, "projects", "general")
    attachment = _project(home, PARTITION_A, "files", "attachment")
    first = build_cell_sandbox(scope=ALICE, project_dir=project, read_only_dirs=[attachment])
    assert first.read_only_dirs == (attachment.resolve(),)
    attachment.rename(attachment.with_name("original"))
    attachment.symlink_to(_project(home, PARTITION_B, "files", "private"), target_is_directory=True)
    second = build_cell_sandbox(scope=ALICE, project_dir=project, read_only_dirs=[attachment])
    assert second.read_only_dirs == ()


@pytest.mark.parametrize(
    ("suffix", "names"),
    [
        ("root", ()),
        ("root/child", ("child",)),
        ("root/../root/child", ("child",)),
        ("rooted/child", None),
        ("ROOT/child", ("child",) if os.name == "nt" else None),
    ],
)
def test_anchoring_compares_whole_normalized_path_components(monkeypatch, tmp_path, suffix, names):
    root = tmp_path / "root"
    monkeypatch.setattr(code_sandbox, "_trusted_roots", lambda: [root])
    found = code_sandbox._anchored(tmp_path / suffix)
    if names is None:
        assert found is None
    else:
        assert found == code_sandbox._Anchored(root.resolve(), names)


def test_a_cached_outer_root_still_checks_a_linked_inner_root(monkeypatch, tmp_path):
    outer = _project(tmp_path, "outer")
    inner = outer / "projects"
    inner.symlink_to(_project(tmp_path, "outside"), target_is_directory=True)
    monkeypatch.setattr(code_sandbox, "_trusted_roots", lambda: [inner, outer])
    roots = code_sandbox._TrustedRoots()
    path = inner / "project"
    assert code_sandbox._anchored(path, roots=roots) == code_sandbox._Anchored(
        outer.resolve(), ("projects", "project"),
    )
    with pytest.raises(OSError) as refused:
        code_sandbox._required_dir(path, roots=roots)
    assert refused.value.errno in code_sandbox._LINK_ERRNOS


def test_a_persons_home_stays_theirs_with_everything_else_shared(stack, tmp_path):
    """The sandbox store is the person's own whatever the list leaves shared,
    memory included."""
    alice = _person(PARTITION_A, SUB_A, frozenset())
    project = _project(tmp_path / "home", "projects", "general")
    policy = build_cell_sandbox(scope=alice, project_dir=project)
    assert policy.home_dir == (tmp_path / "home").resolve() / PARTITION_A / "sandbox" / "home"


@pytest.mark.parametrize("memory", ["personal", "shared"])
def test_two_people_never_share_a_sandbox_store(stack, tmp_path, memory):
    """HOME, TMPDIR and the probes' folder are each person's own, with memory
    personal and with memory shared: one person's pip cache must never reach
    another's installs."""
    resources = (
        ALL_PERSONAL_RESOURCES
        if memory == "personal"
        else ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY}
    )
    home = tmp_path / "home"
    real = home.resolve()
    stores = {}
    for partition, sub in ((PARTITION_A, SUB_A), (PARTITION_B, SUB_B)):
        scope = _person(partition, sub, resources)
        policy = build_cell_sandbox(scope=scope, project_dir=_project(home, partition, "projects", "p"))
        probes = code_sandbox.probe_workspaces_dir(scope)
        stores[partition] = (policy.home_dir, policy.tmp_dir, probes)
        assert (policy.home_dir, policy.tmp_dir, probes) == tuple(
            real / partition / "sandbox" / name for name in ("home", "tmp", "probes")
        )
    assert set(stores[PARTITION_A]).isdisjoint(stores[PARTITION_B])


def test_a_persons_policy_names_every_other_partition(stack, tmp_path):
    """The policy model refuses a writable grant inside another person's
    partition, so the builder names every partition but the asker's; the
    operator view has none of its own."""
    home = tmp_path / "home"
    real = home.resolve()
    _project(home, PARTITION_B)
    alice = build_cell_sandbox(scope=ALICE, project_dir=_project(home, PARTITION_A, "projects", "p"))
    assert alice.other_partitions == (real / PARTITION_B,)
    operator = build_cell_sandbox(scope=LOCAL_SCOPE, project_dir=_project(home, "projects", "general"))
    assert set(operator.other_partitions) == {real / PARTITION_A, real / PARTITION_B}


def _realpath(path: Path) -> Path:
    return Path(os.path.realpath(path))


def _shared_root(tmp_path: Path, *, reached: str) -> Path:
    """The test's shared root, ``tmp_path/home``, as configured or through a
    link to it. pytest resolves tmp_path, so without the link no platform
    reaches the shared root through one."""
    home = _project(tmp_path, "home")
    if reached == "as configured":
        return home
    linked = tmp_path / "linked-home"
    linked.symlink_to(home, target_is_directory=True)
    return linked


@pytest.mark.parametrize("reached", ["as configured", "through a link"])
def test_each_partition_resolves_where_realpath_resolves_it(tmp_path, reached):
    """The partitions are listed with each one resolved once, to what
    os.path.realpath gives its entry: Alice's folder, and Bob's, which is a
    link to a folder elsewhere. The shared root is resolved once for all of
    them, through its link too. A name that isn't a partition id in its
    canonical form is no partition."""
    shared_root = _shared_root(tmp_path, reached=reached)
    _project(shared_root, PARTITION_A)
    bobs = _project(tmp_path, "elsewhere", "bobs")
    (shared_root / PARTITION_B).symlink_to(bobs, target_is_directory=True)
    _project(shared_root, "shared")
    _project(shared_root, PARTITION_C.upper())

    partitions = code_sandbox._partition_roots(shared_root)

    assert sorted(partitions) == sorted(
        code_sandbox._Partition(name, _realpath(shared_root / name))
        for name in (PARTITION_A, PARTITION_B)
    )
    assert {partition.name: partition.real for partition in partitions}[PARTITION_B] == bobs.resolve()


@pytest.mark.parametrize("asker", ["a person", "the operator view"])
@pytest.mark.parametrize("reached", ["as configured", "through a link"])
def test_the_policy_names_each_partition_as_realpath_resolves_it(
    stack, tmp_path, managed_dir, reached, asker
):
    """forbidden_roots and other_partitions, in order and each once, as
    resolving every root and partition entry with os.path.realpath names
    them. Bob's partition is a link to a folder that bears Alice's id, so
    his resolves to her name: Alice's own partition is still the one her id
    names on disk, in her forbidden_roots and not her other_partitions."""
    shared_root = _shared_root(tmp_path, reached=reached)
    stack.setenv("COWORK_HOME", str(shared_root))
    stack.setenv("COWORK_SHARED_DIR", str(shared_root))
    get_app_settings.cache_clear()
    bobs = _project(tmp_path, "elsewhere", PARTITION_A)
    (shared_root / PARTITION_B).symlink_to(bobs, target_is_directory=True)
    _project(shared_root, "shared")
    if asker == "a person":
        scope, own = ALICE, PARTITION_A
        project = _project(shared_root, PARTITION_A, "projects", "p")
    else:
        scope, own = LOCAL_SCOPE, None
        _project(shared_root, PARTITION_A)
        project = _project(shared_root, "projects", "general")
    entries = [
        shared_root / name
        for name in os.listdir(shared_root)
        if name in (PARTITION_A, PARTITION_B)
    ]

    policy = build_cell_sandbox(scope=scope, project_dir=project)

    roots = [shared_root, shared_root, shared_root / "shared", *entries, managed_dir.root]
    assert policy.forbidden_roots == tuple(dict.fromkeys(_realpath(root) for root in roots))
    assert policy.other_partitions == tuple(
        dict.fromkeys(_realpath(entry) for entry in entries if entry.name != own)
    )
    alices = _realpath(shared_root / PARTITION_A)
    assert alices in policy.forbidden_roots
    assert bobs.resolve() in policy.other_partitions
    assert (alices in policy.other_partitions) is (asker == "the operator view")


def test_a_partition_made_after_a_build_is_denied_on_the_next(stack, tmp_path):
    """Nothing keeps the partitions from one build to the next: a person
    whose partition appears after Alice's turn is in her next policy's
    forbidden_roots and other_partitions."""
    home = tmp_path / "home"
    real = home.resolve()
    _project(home, PARTITION_B)
    project = _project(home, PARTITION_A, "projects", "p")
    first = build_cell_sandbox(scope=ALICE, project_dir=project)
    assert real / PARTITION_C not in first.forbidden_roots

    _project(home, PARTITION_C)
    second = build_cell_sandbox(scope=ALICE, project_dir=project)

    assert real / PARTITION_C in second.forbidden_roots
    assert real / PARTITION_C in second.other_partitions


def test_two_people_never_share_a_home(stack, tmp_path):
    home = tmp_path / "home"
    alice = build_cell_sandbox(scope=ALICE, project_dir=_project(home, PARTITION_A, "projects", "p"))
    bob = build_cell_sandbox(scope=BOB, project_dir=_project(home, PARTITION_B, "projects", "p"))
    for mine, theirs in ((alice, bob), (bob, alice)):
        for folder in (mine.home_dir, mine.tmp_dir):
            assert theirs.home_dir not in (folder, *folder.parents)
            assert theirs.tmp_dir not in (folder, *folder.parents)


def test_the_operator_view_gets_the_base_store(stack, tmp_path):
    home = tmp_path / "home"
    policy = build_cell_sandbox(scope=LOCAL_SCOPE, project_dir=_project(home, "projects", "general"))
    assert policy.home_dir == home.resolve() / "sandbox" / "home"
    assert policy.tmp_dir == home.resolve() / "sandbox" / "tmp"


def test_no_scope_on_a_partitioned_deployment_is_refused(stack, tmp_path):
    with pytest.raises(MissingTenantScopeError):
        build_cell_sandbox(scope=None, project_dir=_project(tmp_path / "home", "projects", "p"))


@pytest.mark.parametrize("grant", ["cowork-home", "shared", "another person", "managed folder"])
def test_a_grant_over_a_forbidden_root_is_refused(stack, tmp_path, managed_dir, grant):
    home = tmp_path / "home"
    project = _project(home, PARTITION_A, "projects", "p")
    folder = {
        "cowork-home": home,
        "shared": _project(home, "shared"),
        "another person": _project(home, PARTITION_B),
        "managed folder": managed_dir.root,
    }[grant]
    with pytest.raises(ValueError, match="equals or contains"):
        build_cell_sandbox(scope=ALICE, project_dir=project, read_only_dirs=[folder])


def test_the_managed_skills_folder_can_be_granted(stack, tmp_path, managed_dir):
    project = _project(tmp_path / "home", PARTITION_A, "projects", "p")
    policy = build_cell_sandbox(scope=ALICE, project_dir=project, read_only_dirs=[managed_dir.skills_dir])
    assert policy.read_only_dirs == (managed_dir.skills_dir.resolve(),)
    assert managed_dir.root.resolve() in policy.forbidden_roots


def test_a_linked_grant_is_checked_where_it_points(stack, tmp_path):
    home = tmp_path / "home"
    project = _project(home, PARTITION_A, "projects", "p")
    link = tmp_path / "looks-harmless"
    link.symlink_to(home, target_is_directory=True)
    with pytest.raises(ValueError, match="equals or contains"):
        build_cell_sandbox(scope=ALICE, project_dir=project, read_only_dirs=[link])


def test_a_cowork_home_written_with_dots_still_builds(stack, tmp_path):
    """The policy model takes absolute, normalized paths only."""
    home = tmp_path / "home"
    stack.setenv("COWORK_HOME", str(tmp_path / "elsewhere" / ".." / "home"))
    get_app_settings.cache_clear()
    policy = build_cell_sandbox(scope=ALICE, project_dir=_project(home, PARTITION_A, "projects", "p"))
    assert home.resolve() in policy.forbidden_roots
    assert policy.shared_dir == home.resolve() / "shared-files"


def test_a_linked_home_is_refused(stack, tmp_path):
    """A cell before the sandbox could plant a link where HOME goes; the
    launcher would grant what it points at."""
    home = tmp_path / "home"
    store = _project(home, PARTITION_A, "sandbox")
    bobs = _project(home, PARTITION_B, "memory")
    (store / "home").symlink_to(bobs, target_is_directory=True)
    with pytest.raises(OSError):
        build_cell_sandbox(scope=ALICE, project_dir=_project(home, PARTITION_A, "projects", "p"))


# ── links left by agent code from before the sandbox ─────────────────────────
#
# Before the sandbox, any person's cell could write anywhere on the volume, so
# a link can sit anywhere below cowork-home. The launcher grants what a path
# resolves to, and a grant inside another person's partition, rather than over
# it, passes the policy model's forbidden roots.


def test_a_linked_shared_folder_is_refused(stack, tmp_path):
    home = tmp_path / "home"
    alices_projects = _project(home, PARTITION_A, "projects")
    (home / "shared-files").symlink_to(alices_projects, target_is_directory=True)
    with pytest.raises(OSError, match="grants no folder through a link"):
        build_cell_sandbox(scope=BOB, project_dir=_project(home, PARTITION_B, "projects", "general"))
    message = _refusal()
    assert message.startswith("COWORK_CODE_SANDBOX=landlock can't make the shared folder")
    assert "grants no folder through a link" in message


def test_a_linked_project_is_refused(stack, tmp_path):
    home = tmp_path / "home"
    alices = _project(home, PARTITION_A, "projects", "general")
    bobs = _project(home, PARTITION_B, "projects") / "general"
    bobs.symlink_to(alices, target_is_directory=True)
    with pytest.raises(OSError, match="grants no folder through a link"):
        build_cell_sandbox(scope=BOB, project_dir=bobs)


def test_a_linked_partition_is_refused(stack, tmp_path):
    """HOME and TMPDIR are opened from the shared root, so a link at the
    person's partition folder itself is refused too."""
    home = tmp_path / "home"
    alices = _project(home, PARTITION_A)
    (home / PARTITION_B).symlink_to(alices, target_is_directory=True)
    with pytest.raises(OSError, match="grants no folder through a link"):
        build_cell_sandbox(scope=BOB, project_dir=_project(tmp_path, "adopted"))
    assert list(alices.iterdir()) == []


@pytest.mark.parametrize("linked", ["shared skill", "attachment folder"])
async def test_a_turn_leaves_a_linked_folder_out_of_its_grants(
    stack, monkeypatch, tmp_path, owned_logger, caplog, linked
):
    """A skill Alice shares with Bob, or Bob's own attachment folder, swapped
    for a link into Alice's memory: Bob's turn grants neither."""
    home = tmp_path / "home"
    alices_memory = _project(home, PARTITION_A, "memory")
    (alices_memory / "sales.csv").write_text("hers\n", encoding="utf-8")
    if linked == "shared skill":
        link = _project(home, PARTITION_A, "skills") / "shared-with-bob"
        rows = []
    else:
        link = _project(home, PARTITION_B, "files") / "f-1"
        rows = [harness._AttachedFile(path=str(link / "sales.csv"), filename="sales.csv")]
    link.symlink_to(alices_memory, target_is_directory=True)

    async def attachments(conversation, *, scope):
        return rows

    monkeypatch.setattr(harness, "_read_attachments", attachments)
    monkeypatch.setattr(
        harness, "link_turn_skills", lambda **_: (link,) if linked == "shared skill" else (),
    )
    owned_logger("cowork.code_sandbox", level=logging.WARNING)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=BOB) as turn:
        policy = turn.config.cell_sandbox

    granted = set(policy.read_only_dirs)
    assert alices_memory.resolve() not in granted
    assert not any(alices_memory.resolve() in (grant, *grant.parents) for grant in granted)
    (warning,) = _sandbox_warnings(caplog)
    assert warning.startswith(f"code sandbox: not granting {link}")


# ── links left in a person's stores ──────────────────────────────────────────
#
# The server isn't locked, and it reads and writes a person's data vault, the
# memory tiers a turn hands anton and the publish store by path. A link, or a
# second name for another person's file, that agent code left in one before
# the sandbox fails that read or write (refuse_planted_links). Neither the
# server nor anton makes a link in these stores.

PLANTED = "Agent code that ran before the sandbox may have left it there"


def _tree(folder: Path) -> dict[str, bytes | None]:
    """Every file and folder below ``folder``, with each file's bytes."""
    return {
        str(path.relative_to(folder)): None if path.is_dir() else path.read_bytes()
        for path in sorted(folder.rglob("*"))
    }


def _plant_in_store(planted: str, *, store: Path, name: str, bobs: Path) -> None:
    """Agent code before the sandbox: ``store`` swapped for a link to Bob's
    folder, or ``store/name`` made a second name for Bob's file ``name``."""
    if planted == "link":
        _plant_link(store, bobs)
        return
    (bobs / name).write_text("bob's\n", encoding="utf-8")
    store.mkdir(parents=True, exist_ok=True)
    os.link(bobs / name, store / name)


@pytest.mark.parametrize("planted", ["link", "hard link"])
def test_a_planted_link_in_a_persons_vault_is_refused(stack, tmp_path, planted):
    from cowork.services.connectors.persist import own_vault_root, persist_connection, vault_for_scope

    bobs = _project(tmp_path / "home", PARTITION_B, "projects", "general")
    _plant_in_store(planted, store=own_vault_root(ALICE), name="postgres-prod", bobs=bobs)
    before = _tree(bobs)

    with pytest.raises(PlantedLinkError, match=PLANTED):
        persist_connection("postgres", None, "prod", {"host": "db", "password": "alice-s3cret"}, scope=ALICE)
    with pytest.raises(PlantedLinkError, match=PLANTED):
        vault_for_scope(ALICE).list_connections()

    assert _tree(bobs) == before


@pytest.mark.parametrize("planted", ["link", "hard link"])
async def test_a_planted_link_in_a_persons_memory_fails_their_turn(
    stack, monkeypatch, tmp_path, planted
):
    """anton's Hippocampus appends to a tier's files by path, so Alice's
    lessons would land in Bob's project, and his in her recall."""
    from cowork.harnesses.memory.store import GlobalMemoryStore

    bobs = _project(tmp_path / "home", PARTITION_B, "projects", "general")
    _plant_in_store(planted, store=GlobalMemoryStore(scope=ALICE).root, name="lessons.md", bobs=bobs)
    before = _tree(bobs)

    with pytest.raises(PlantedLinkError, match=PLANTED):
        async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=ALICE):
            pass

    assert _tree(bobs) == before


@pytest.mark.parametrize("planted", ["link", "hard link"])
async def test_a_planted_link_in_the_shared_memory_fails_a_turn(
    stack, monkeypatch, tmp_path, planted
):
    """With memory left off COWORK_PERSONAL_RESOURCES every person's turn
    appends to the deployment's one store by path, so a link there fails the
    turn as it does in a person's own tier."""
    from cowork.harnesses.memory.store import GlobalMemoryStore, deployment_memory_root

    stack.setenv("COWORK_PERSONAL_RESOURCES", SHARED_MEMORY)
    get_app_settings.cache_clear()
    alice = _person(PARTITION_A, SUB_A, ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY})
    store = GlobalMemoryStore(scope=alice).root
    assert store == deployment_memory_root()
    bobs = _project(tmp_path / "home", PARTITION_B, "projects", "general")
    _plant_in_store(planted, store=store, name="lessons.md", bobs=bobs)
    before = _tree(bobs)

    with pytest.raises(PlantedLinkError, match=PLANTED):
        async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=alice):
            pass

    assert _tree(bobs) == before


@pytest.mark.parametrize("memory", ["personal", "shared"])
async def test_memory_off_uses_no_stored_tier_and_keeps_the_sandbox_store(
    stack, monkeypatch, tmp_path, memory
):
    """With MEMORY_ENABLED false a turn gets two empty tiers in its scratch
    folder and uses no stored one, so a link left in a stored tier neither
    fails the turn nor is followed. The cell's HOME is still the person's own."""
    from cowork.harnesses.memory.store import GlobalMemoryStore

    resources = ALL_PERSONAL_RESOURCES
    if memory == "shared":
        resources = ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY}
        stack.setenv("COWORK_PERSONAL_RESOURCES", SHARED_MEMORY)
    stack.setenv("MEMORY_ENABLED", "false")
    get_app_settings.cache_clear()
    alice = _person(PARTITION_A, SUB_A, resources)
    bobs = _project(tmp_path / "home", PARTITION_B, "projects", "general")
    _plant_in_store("link", store=GlobalMemoryStore(scope=alice).root, name="lessons.md", bobs=bobs)
    before = _tree(bobs)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=alice) as turn:
        assert turn.config.cortex.mode == "off"
        assert turn.config.background_memory is False
        assert turn.config.cell_sandbox.home_dir == (
            (tmp_path / "home").resolve() / PARTITION_A / "sandbox" / "home"
        )

    assert _tree(bobs) == before


@pytest.mark.parametrize("planted", ["link", "hard link"])
def test_a_planted_link_in_a_persons_publish_store_is_refused(stack, tmp_path, planted):
    from cowork.services import publish

    bobs = _project(tmp_path / "home", PARTITION_B, "projects", "general")
    _plant_in_store(planted, store=tmp_path / "home" / PARTITION_A / "publish", name="state.json", bobs=bobs)
    before = _tree(bobs)

    with pytest.raises(PlantedLinkError, match=PLANTED):
        publish._save_state({"publish_history": [{"artifact": "alice's"}]}, scope=ALICE)
    with pytest.raises(PlantedLinkError, match=PLANTED):
        publish._load_state(scope=ALICE)

    assert _tree(bobs) == before


async def test_a_persons_stores_without_planted_links_work(stack, monkeypatch, tmp_path):
    from cowork.harnesses.memory.store import GlobalMemoryStore
    from cowork.services import publish
    from cowork.services.connectors.persist import persist_connection, vault_for_scope

    slug = persist_connection("postgres", None, "prod", {"host": "db", "password": "pw"}, scope=ALICE)
    assert vault_for_scope(ALICE).load("postgres", slug)["host"] == "db"
    publish._save_state({"publish_history": [{"artifact": "a"}]}, scope=ALICE)
    assert publish._load_state(scope=ALICE) == {"publish_history": [{"artifact": "a"}]}
    tier = GlobalMemoryStore(scope=ALICE).root
    tier.mkdir(parents=True, exist_ok=True)
    (tier / "lessons.md").write_text("- check the fiscal year\n", encoding="utf-8")
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=ALICE) as turn:
        assert turn.config.cortex is not None


def test_a_link_on_the_way_to_a_store_is_refused(stack, tmp_path):
    """The folders from the trusted root down to the store are opened without
    following links too: here Alice's partition itself is Bob's."""
    home = tmp_path / "home"
    bobs = _project(home, PARTITION_B)
    _plant_link(home / PARTITION_A, bobs)
    with pytest.raises(PlantedLinkError, match=f"{home.resolve() / PARTITION_A} is a link"):
        code_sandbox.refuse_planted_links(home / PARTITION_A / "data-vault")
    assert list(bobs.iterdir()) == []


def test_a_link_deep_in_a_store_is_refused_and_a_missing_store_passes(stack, tmp_path):
    store = _project(tmp_path / "home", PARTITION_A, "memory", "users", SUB_A, "projects", "p")
    _plant_link(store / "deeper" / "lessons", _project(tmp_path, "elsewhere"))
    with pytest.raises(PlantedLinkError, match="lessons is a link"):
        code_sandbox.refuse_planted_links(tmp_path / "home" / PARTITION_A / "memory")
    code_sandbox.refuse_planted_links(tmp_path / "home" / PARTITION_B / "data-vault")


def test_a_store_outside_the_servers_roots_is_trusted_as_configured(stack, tmp_path):
    """Such as ANTON_COWORK_STATE_DIR set to a folder of the operator's: the
    configured path is resolved, and what is there is still checked."""
    elsewhere = _project(tmp_path, "operators-state")
    (elsewhere / "state.json").write_text("{}", encoding="utf-8")
    linked = tmp_path / "operators-state-link"
    linked.symlink_to(elsewhere, target_is_directory=True)
    code_sandbox.refuse_planted_links(linked / "state.json")
    os.link(elsewhere / "state.json", _project(tmp_path, "bobs") / "state.json")
    with pytest.raises(PlantedLinkError, match="a file with 2 names"):
        code_sandbox.refuse_planted_links(linked / "state.json")


def test_a_link_at_the_state_file_of_a_publish_store_outside_the_roots_is_refused(stack, tmp_path):
    """ANTON_COWORK_STATE_DIR set to a folder of the operator's outside the
    server's roots is trusted as configured, through a link too, and only the
    store's own files in it are checked: a link at state.json is refused, not
    followed."""
    from cowork.services import publish

    elsewhere = _project(tmp_path, "operators-state")
    (elsewhere / "unrelated").symlink_to(tmp_path)
    linked = tmp_path / "operators-state-link"
    linked.symlink_to(elsewhere, target_is_directory=True)
    stack.setenv("ANTON_COWORK_STATE_DIR", str(linked))
    publish._save_state({"publish_history": [{"artifact": "a"}]}, scope=LOCAL_SCOPE)
    assert publish._load_state(scope=LOCAL_SCOPE) == {"publish_history": [{"artifact": "a"}]}

    bobs = _project(tmp_path / "home", PARTITION_B, "projects", "general")
    (bobs / "notes.json").write_text('{"bob": "private"}', encoding="utf-8")
    (elsewhere / "state.json").unlink()
    (elsewhere / "state.json").symlink_to(bobs / "notes.json")
    before = _tree(bobs)

    with pytest.raises(PlantedLinkError, match="state.json is a link"):
        publish._load_state(scope=LOCAL_SCOPE)
    with pytest.raises(PlantedLinkError, match="state.json is a link"):
        publish._save_state({"publish_history": []}, scope=LOCAL_SCOPE)
    assert _tree(bobs) == before


def test_off_never_checks_a_store_for_links(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    store = tmp_path / "store"
    _plant_link(store, _project(tmp_path, "linked-on-purpose"))
    code_sandbox.refuse_planted_links(store)


# ── the turn ─────────────────────────────────────────────────────────────────


def _record_configs(monkeypatch) -> list[dict]:
    """Every ChatSessionConfig's keyword arguments, as the real class gets them."""
    import anton.core.session as session_module

    calls: list[dict] = []
    real = session_module.ChatSessionConfig

    class RecordingChatSessionConfig(real):
        def __init__(self, **kwargs):
            calls.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(session_module, "ChatSessionConfig", RecordingChatSessionConfig)
    return calls


async def test_a_turn_off_passes_what_it_passed_before(monkeypatch, tmp_path):
    """Off adds no keyword and keeps the backend sentence and the rules on
    other files; on adds only the policy, and tells the agent about the
    shared folder every person's agent reads and changes."""
    monkeypatch.setattr(code_sandbox, "_probe_abi", _never)
    calls = _record_configs(monkeypatch)
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None) as off:
        off_suffix = off.config.system_prompt_context.suffix
    off_keys = set(calls[-1])

    monkeypatch.setenv("COWORK_CODE_SANDBOX", "landlock")
    get_app_settings.cache_clear()
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None) as on:
        on_suffix = on.config.system_prompt_context.suffix
        policy = on.config.cell_sandbox
    get_app_settings.cache_clear()

    assert "cell_sandbox" not in off_keys
    assert set(calls[-1]) - off_keys == {"cell_sandbox"}
    assert off_keys - set(calls[-1]) == set()
    assert BACKEND_SENTENCE in off_suffix
    assert "datasets, fullstack apps, etc." in off_suffix
    assert "runtime-local address" not in off_suffix
    assert BACKEND_SENTENCE not in on_suffix
    assert harness._SANDBOXED_BACKENDS_RULE in on_suffix
    assert "fullstack apps" not in on_suffix
    assert policy.project_dir == Path(on.project.path).resolve()
    # The shared folder: absent off, named and allowed on.
    shared = str(code_sandbox.shared_files_dir())
    assert harness._OTHER_FILES_RULE in off_suffix
    assert harness._ATTACHMENT_AFFORDANCE in off_suffix
    assert "shared folder" not in off_suffix and shared not in off_suffix
    assert f"and the files in the shared folder at `{shared}`. " in on_suffix
    assert "Every person's agent on this deployment can read and change the shared folder" in on_suffix
    assert f"or they are in the shared folder at `{shared}`" in on_suffix
    assert harness._OTHER_FILES_RULE not in on_suffix
    assert harness._ATTACHMENT_AFFORDANCE not in on_suffix
    assert "strictly forbidden" in on_suffix and "exactly two ways" not in on_suffix


@dataclass(frozen=True)
class _PolicyBuild:
    """One build_cell_sandbox call a turn made, and where it ran."""

    on_an_event_loop: bool
    on_the_main_thread: bool
    settings_scope: TenantScope | None
    policy: object


async def test_a_turn_builds_its_policy_off_the_event_loop(stack, monkeypatch, tmp_path):
    """The policy lists every person's partition and opens the session's
    folders, so the turn builds it in a worker thread rather than on the
    event loop every other request shares. The thread carries the turn's
    settings scope, and the policy it builds is the one the session gets."""
    builds: list[_PolicyBuild] = []
    build = harness.build_cell_sandbox

    def recording(**kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_an_event_loop = False
        else:
            on_an_event_loop = True
        policy = build(**kwargs)
        builds.append(
            _PolicyBuild(
                on_an_event_loop=on_an_event_loop,
                on_the_main_thread=threading.current_thread() is threading.main_thread(),
                settings_scope=current_settings_scope(),
                policy=policy,
            )
        )
        return policy

    monkeypatch.setattr(harness, "build_cell_sandbox", recording)
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=ALICE) as turn:
        policy = turn.config.cell_sandbox

    (made,) = builds
    assert made.on_an_event_loop is False
    assert made.on_the_main_thread is False
    assert made.settings_scope == ALICE
    assert policy is not None
    assert made.policy is policy


async def test_a_persons_turn_grants_its_skills_and_attachments(personal_deployment, monkeypatch, tmp_path):
    attachment = tmp_path / "files" / "f-1" / "sales.csv"
    attachment.parent.mkdir(parents=True)
    attachment.write_text("a,b\n", encoding="utf-8")

    async def attachments(conversation, *, scope):
        return [harness._AttachedFile(path=str(attachment), filename="sales.csv")]

    own_skill = _project(tmp_path, "alice-skills", "own")
    shared_skill = _project(tmp_path, "bob-skills", "shared-with-alice")
    linked: list[dict] = []

    def link(**kwargs):
        linked.append(kwargs)
        return (own_skill, shared_skill)

    monkeypatch.setattr(harness, "_read_attachments", attachments)
    monkeypatch.setattr(harness, "link_turn_skills", link)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=ALICE) as turn:
        policy = turn.config.cell_sandbox
        suffix = turn.config.system_prompt_context.suffix
        shared_root = turn.scratch / "shared-skills"

    granted = set(policy.read_only_dirs)
    assert policy.project_dir == Path(turn.project.path).resolve()
    assert own_skill.resolve() in granted and shared_skill.resolve() in granted
    assert attachment.parent.resolve() in granted
    assert harness.HOST_SKILLS_DIR.resolve() in granted
    # The folder of links to the shared skills is listable; the links resolve
    # into the grants above.
    assert linked[0]["shared_skills_dir"] == shared_root
    assert shared_root.resolve() in granted
    assert str(attachment) in suffix
    assert harness._SANDBOXED_BACKENDS_RULE in suffix


async def test_a_turn_grants_the_managed_skills_its_links_point_at(
    personal_deployment, monkeypatch, tmp_path, managed_dir
):
    """The turn links each managed skill into its scratch folder, and Landlock
    checks where a link resolves, so the managed skill folders are granted
    themselves, read-only."""
    skill = managed_dir.skills_dir / "quarterly-report"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: quarterly-report\n---\n", encoding="utf-8")
    monkeypatch.setattr(harness, "link_turn_skills", lambda **_: ())

    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=ALICE) as turn:
        policy = turn.config.cell_sandbox
        links = turn.scratch / "managed-skills"

    granted = set(policy.read_only_dirs)
    assert skill.resolve() in granted
    assert links.resolve() in granted
    assert managed_dir.root.resolve() not in granted


def _plant_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("linked", ["skills", ".anton", ".anton/artifacts"])
async def test_a_sandboxed_turn_refuses_a_linked_project_folder(landlock_alone, monkeypatch, tmp_path, linked):
    """Under the sandbox, the server's own folders in a project, and anton's
    .anton (Workspace.initialize's refuse_links), never follow a planted link."""
    outside = _project(tmp_path, "someone-elses")

    with pytest.raises(OSError):
        async with built_turn(
            monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None,
            before_build=lambda project: _plant_link(project / linked, outside),
        ):
            pass
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("linked", ["skills", ".anton", ".anton/artifacts"])
async def test_a_turn_off_follows_a_linked_project_folder(monkeypatch, tmp_path, linked):
    """With the sandbox off, a project that links these folders on purpose
    builds as it does on the desktop."""
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    elsewhere = _project(tmp_path, "linked-on-purpose")

    async with built_turn(
        monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None,
        before_build=lambda project: _plant_link(project / linked, elsewhere),
    ) as turn:
        folder = Path(turn.project.path) / linked

    assert folder.is_symlink() and folder.resolve() == elsewhere.resolve()
    if linked == ".anton":
        assert (elsewhere / "memory").is_dir() and (elsewhere / ".env").is_file()


async def test_a_sandboxed_turn_refuses_a_project_reached_through_a_link(landlock_alone, monkeypatch, tmp_path):
    """The server isn't locked, so it refuses the project before its own
    writes (the skills/ and .anton folders, the turn's skill links) follow
    the link into another person's project."""
    landlock_alone.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    someone_elses = _project(tmp_path, "someone-elses-project")

    def plant(project: Path) -> None:
        project.rmdir()
        project.symlink_to(someone_elses, target_is_directory=True)

    with pytest.raises(OSError, match="grants no folder through a link"):
        async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None, before_build=plant):
            pass
    assert list(someone_elses.iterdir()) == []


async def test_a_sandboxed_turn_makes_a_missing_project_folder_again(landlock_alone, monkeypatch, tmp_path):
    """As a turn with the sandbox off does, without following links."""
    landlock_alone.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    async with built_turn(
        monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None, before_build=lambda project: project.rmdir(),
    ) as turn:
        project = Path(turn.project.path)

    assert project.is_dir() and (project / "skills").is_dir()
    assert turn.config.cell_sandbox.project_dir == project.resolve()


async def test_a_sandboxed_turn_refuses_a_context_folder_outside_the_project(
    landlock_alone, monkeypatch, tmp_path
):
    """The server makes the context folder unlocked, so it makes none outside
    the project, where it couldn't refuse a link on the way."""
    elsewhere = tmp_path / "elsewhere" / "context"
    landlock_alone.setenv("ANTON_CONTEXT_DIR", str(elsewhere))

    with pytest.raises(RuntimeError, match="Unset ANTON_CONTEXT_DIR, or set it to a folder inside the project"):
        async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None):
            pass

    assert not elsewhere.parent.exists()


def test_the_env_fallback_is_refused_under_the_sandbox(landlock_alone):
    workspace = SimpleNamespace(applied=False)
    workspace.apply_env_to_process = lambda: setattr(workspace, "applied", True)
    with pytest.raises(RuntimeError, match="refused under COWORK_CODE_SANDBOX=landlock"):
        harness._apply_overlay_fallback(workspace, {"KEY": "value"}, {})
    assert workspace.applied is False


def test_the_env_fallback_still_applies_with_the_sandbox_off(monkeypatch):
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    workspace = SimpleNamespace(applied=False)
    workspace.apply_env_to_process = lambda: setattr(workspace, "applied", True)
    assert harness._apply_overlay_fallback(workspace, {"KEY": "value"}, {}) is True
    assert workspace.applied is True


# ── the credential probe ─────────────────────────────────────────────────────


class _PromptRecordingSession:
    """A probe session whose turn records the prompt it is given and ends."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def turn_stream(self, prompt: str, *, trace_tags: list[str]):
        self.prompts.append(prompt)
        return
        yield


async def _run_probe(
    monkeypatch, *, cell_sandbox, credentials, session: _PromptRecordingSession | None = None,
) -> tuple[object, dict, list]:
    """Run the probe. With no ``session`` it stops where its session would be
    built; with one, it drives that session's turn."""
    from cowork.services.connectors import probe as probe_module

    calls = _record_configs(monkeypatch)

    def build(config):
        if session is None:
            raise RuntimeError("stop before the session starts")
        return session

    monkeypatch.setattr(probe_module, "build_chat_session", build)
    monkeypatch.setattr(probe_module, "close_session_scratchpads", lambda *_args, **_kwargs: None)
    probe = probe_module.CredentialProbe(
        engine="postgres", credentials=credentials, llm_client=object(), workspace=None,
        cell_sandbox=cell_sandbox,
    )
    events = [event async for event in probe.run()]
    return probe, calls[-1], events


async def test_a_sandboxed_probe_hands_the_credentials_to_the_pad(stack, monkeypatch, tmp_path):
    """The probe's prompt names the flat DS_<KEY> variables, and anton's
    connection listing in its system prompt the namespaced ones: the pad
    holds every name either one gives, and neither holds a value."""
    import re

    from anton.core.backends.manager import ScratchpadManager
    from anton.utils.datasources import build_datasource_context

    from cowork.services.connectors.probe import (
        CandidateCredentialsVault,
        CredentialProbe,
        _probe_tmp_dir,
    )

    written: list[str] = []
    real_write = CredentialProbe._write_credentials_env

    def recording_write(self):
        path, names = real_write(self)
        written.append(path)
        return path, names

    monkeypatch.setattr(CredentialProbe, "_write_credentials_env", recording_write)
    policy = build_cell_sandbox(
        scope=ALICE, project_dir=_project(tmp_path / "home", PARTITION_A, "sandbox", "probes", "p")
    )
    session = _PromptRecordingSession()
    _, kwargs, events = await _run_probe(
        monkeypatch, cell_sandbox=policy, credentials={"host": "db", "password": "hunter2"},
        session=session,
    )

    # No plaintext credentials file, and a prompt that names none.
    assert written == []
    (prompt,) = session.prompts
    assert "Path: `" not in prompt and "dotenv_values" not in prompt
    assert str(_probe_tmp_dir()) not in prompt
    assert "Variable names: DS_HOST, DS_PASSWORD\n" in prompt
    assert "read them from `os.environ` by the variable names above." in prompt
    assert "hunter2" not in prompt
    # The pad gets them as its DS_*: anton's own manager derives them.
    assert kwargs["cell_sandbox"] is policy
    vault = kwargs["data_vault"]
    assert isinstance(vault, CandidateCredentialsVault)
    manager = ScratchpadManager(
        runtime_factory=lambda **_: None, coding_provider="", coding_model="",
        coding_api_key="", coding_base_url="", data_vault=vault,
    )
    pad_env = manager._derive_ds_env()
    assert pad_env == {
        "DS_HOST": "db", "DS_PASSWORD": "hunter2",
        "DS_POSTGRES_PROBE__HOST": "db", "DS_POSTGRES_PROBE__PASSWORD": "hunter2",
    }
    listing = build_datasource_context(vault)
    listed = set(re.findall(r"^ - (DS_\w+)$", listing, re.MULTILINE))
    assert listed == {"DS_POSTGRES_PROBE__HOST", "DS_POSTGRES_PROBE__PASSWORD"}
    assert listed <= pad_env.keys() and {"DS_HOST", "DS_PASSWORD"} <= pad_env.keys()
    assert "hunter2" not in listing and "Host: db" not in listing
    assert vault.load("postgres", "probe") is None
    tmp = _probe_tmp_dir()
    assert not tmp.exists() or list(tmp.iterdir()) == []
    assert events[-1][0] == "verdict"


async def test_a_probe_off_writes_its_file_and_passes_no_policy(monkeypatch, tmp_path):
    from anton.core.datasources.data_vault import LocalDataVault

    from cowork.services.connectors.probe import _probe_tmp_dir

    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    written: list[str] = []
    from cowork.services.connectors.probe import CredentialProbe

    real_write = CredentialProbe._write_credentials_env

    def recording_write(self):
        path, names = real_write(self)
        written.append(Path(path).read_text(encoding="utf-8"))
        return path, names

    monkeypatch.setattr(CredentialProbe, "_write_credentials_env", recording_write)
    _, kwargs, _ = await _run_probe(monkeypatch, cell_sandbox=None, credentials={"password": "hunter2"})

    assert "cell_sandbox" not in kwargs
    assert isinstance(kwargs["data_vault"], LocalDataVault)
    assert kwargs["data_vault"]._dir == _probe_tmp_dir() / "empty-vault"
    assert written == ['DS_PASSWORD="hunter2"\n']
    assert list(_probe_tmp_dir().glob("cowork-vault-*.env")) == []


def test_a_sandboxed_probe_prompt_names_the_environment_not_a_file():
    from cowork.services.connectors.probe import CredentialProbe

    probe = CredentialProbe(engine="postgres", credentials={}, llm_client=None, workspace=None)
    sandboxed = probe._build_prompt(None, ["DS_HOST"])
    assert "Path: `" not in sandboxed and "dotenv_values" not in sandboxed
    assert "The user-submitted credentials are in the scratchpad's environment:\n  Variable names: DS_HOST\n" in sandboxed
    assert "read them from `os.environ` by the variable names above." in sandboxed
    off = probe._build_prompt("/x/c.env", ["DS_HOST"])
    assert "The user-submitted credentials are in a temporary `.env` file:\n  Path: `/x/c.env`\n" in off
    assert "2. In the scratchpad, parse the .env file (e.g. `dotenv_values('/x/c.env')`). NEVER print" in off


def _run_probe_handler(monkeypatch, *, scope: TenantScope) -> list[dict]:
    """The probe handler for a submission with no conversation, up to the
    probe it builds, whose keyword arguments it returns."""
    import asyncio

    from cowork.handlers import probe as probe_handler
    from cowork.services.connectors.probe import ProbeOutcome
    from cowork.services.connectors.submissions import SubmissionStore

    created: list[dict] = []

    class RecordingProbe:
        def __init__(self, **kwargs):
            created.append(kwargs)

        async def run(self):
            yield "verdict", ProbeOutcome(status="failure", error="stop here")

    monkeypatch.setattr(probe_handler, "CredentialProbe", RecordingProbe)
    monkeypatch.setattr(probe_handler.ProbeHandler, "_build_llm_client", staticmethod(lambda settings=None: object()))
    monkeypatch.setattr(
        probe_handler, "_read_probe_settings", lambda _session: SimpleNamespace(resolved_planning_provider=None),
    )
    monkeypatch.setattr("cowork.services.providers.web_tool_kwargs_for", lambda _provider: {})
    store = SubmissionStore()
    monkeypatch.setattr(probe_handler, "store", store)
    submission_id = store.stage(
        form_id="postgres-connector", connector_id="postgres", conversation_id=None,
        values={"host": "db", "password": "pw"},
    )

    async def drain():
        return [
            chunk async for chunk in probe_handler.ProbeHandler(scope=scope).run(
                submission_id, "postgres", None, "", None
            )
        ]

    asyncio.run(drain())
    return created


def test_a_sandboxed_probe_without_a_conversation_runs_in_the_askers_store(stack, tmp_path):
    (created,) = _run_probe_handler(stack, scope=ALICE)
    probes = (tmp_path / "home").resolve() / PARTITION_A / "sandbox" / "probes"
    workspace = created["workspace"].base.resolve()
    assert workspace.parent == probes
    assert created["cell_sandbox"].project_dir == workspace
    # Removed when the probe ends, as the /tmp folder was.
    assert not workspace.exists()


def test_a_probe_off_without_a_conversation_runs_in_the_system_tmp(monkeypatch):
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    (created,) = _run_probe_handler(monkeypatch, scope=LOCAL_SCOPE)
    assert created["cell_sandbox"] is None
    assert created["workspace"].base.parent == Path(tempfile.gettempdir())


# ── a construction without the policy ───────────────────────────────────────
#
# Behind tests/test_no_subprocess_static.py, which reads each construction:
# under the sandbox, a session or a backend pool's pad built without the
# policy raises rather than running the agent's code unlocked.


def test_a_chat_session_without_the_policy_is_refused_under_the_sandbox(landlock_alone):
    from cowork.common.chat_session import build_chat_session

    with pytest.raises(RuntimeError, match="a chat session was built without a code-sandbox policy"):
        build_chat_session(SimpleNamespace(cell_sandbox=None))


def test_a_backend_pad_without_the_policy_is_refused_under_the_sandbox(landlock_alone, monkeypatch, tmp_path):
    from cowork.services import scratchpad_runtime

    monkeypatch.setattr(_RecordingRuntime, "calls", [])
    monkeypatch.setattr("anton.core.backends.local.LocalScratchpadRuntime", _RecordingRuntime)
    monkeypatch.setattr(scratchpad_runtime, "_pads", {})
    monkeypatch.setattr(scratchpad_runtime, "_resolve_coding", lambda **coding: tuple(coding.values()))
    refusal = "a backend pool's scratchpad was built without a code-sandbox policy"

    with pytest.raises(RuntimeError, match=refusal):
        scratchpad_runtime._make_runtime(
            "app", workspace=tmp_path, coding_provider="openai", coding_model="m",
            coding_api_key="k", coding_base_url="u",
        )
    with pytest.raises(RuntimeError, match=refusal):
        scratchpad_runtime.get_or_create("app", workspace_path=str(tmp_path))

    assert _RecordingRuntime.calls == []
    assert scratchpad_runtime._pads == {}


# ── app backends ─────────────────────────────────────────────────────────────


async def test_a_backend_is_refused_before_the_pool_is_touched(landlock_alone, monkeypatch, tmp_path):
    from anton.core.backends import sandbox
    from cowork.services import artifacts, scratchpad_runtime

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    monkeypatch.setattr(scratchpad_runtime, "WorkspaceScopedPool", _never)
    monkeypatch.setattr(artifacts, "_probe_port", _never)
    monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})
    artifact_dir = _project(tmp_path, "projects", "proj", ".anton", "artifacts", "app")

    status = await artifacts._ensure_backend_running(artifact_dir, 8000, scope=LOCAL_SCOPE)

    assert status == (False, sandbox.BACKENDS_OFF_MESSAGE, 0)
    assert artifacts._LAUNCHED_BACKENDS == {}


async def test_the_backend_refusal_says_what_antons_own_refusal_says(landlock_alone, monkeypatch, tmp_path):
    """The preview path answers with anton's text, read when it refuses, so
    the person and the agent's launch_backend never give two reasons."""
    from anton.core.backends import sandbox
    from cowork.services import artifacts

    monkeypatch.setattr(sandbox, "BACKENDS_OFF_MESSAGE", "anton's own words")
    monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})
    artifact_dir = _project(tmp_path, "projects", "proj", ".anton", "artifacts", "app")

    status = await artifacts._ensure_backend_running(artifact_dir, 8000, scope=LOCAL_SCOPE)

    assert status == (False, "anton's own words", 0)


def _backend_artifact(tmp_path: Path, *, port: int) -> Path:
    """An artifact folder whose agent-written metadata.json names ``port``."""
    folder = _project(tmp_path, "projects", "proj", ".anton", "artifacts", "app")
    (folder / "metadata.json").write_text(json.dumps({"port": port}), encoding="utf-8")
    (folder / "index.html").write_text("<html></html>", encoding="utf-8")
    return folder


async def test_a_sandboxed_backend_preview_mints_no_token(landlock_alone, monkeypatch, tmp_path):
    from anton.core.backends import sandbox
    from cowork.services import artifacts

    folder = _backend_artifact(tmp_path, port=26866)
    monkeypatch.setattr(artifacts, "_artifact_root_for", lambda path, *, scope: folder)
    monkeypatch.setattr(artifacts, "_PREVIEW_MOUNTS", {})
    monkeypatch.setattr(artifacts, "_ensure_backend_running", _never)

    with pytest.raises(artifacts.BackendPreviewRefused) as refused:
        await artifacts.mount_preview(folder / "index.html", scope=LOCAL_SCOPE)

    assert str(refused.value) == sandbox.BACKENDS_OFF_MESSAGE
    assert artifacts._PREVIEW_MOUNTS == {}


@pytest.mark.parametrize("method", ["GET", "OPTIONS"])
async def test_a_sandboxed_preview_proxy_forwards_nothing(landlock_alone, monkeypatch, tmp_path, method):
    """A static preview's token still names a folder whose metadata.json the
    agent writes, so a `port` there must not reach the server's own port."""
    from anton.core.backends import sandbox
    from starlette.requests import Request

    from cowork.services import preview_proxy

    folder = _backend_artifact(tmp_path, port=26866)
    monkeypatch.setattr(preview_proxy, "preview_mount_for", lambda token, *, scope: folder)
    monkeypatch.setattr(preview_proxy, "get_proxy_client", _never)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {
            "type": "http", "method": method, "path": "/x", "query_string": b"",
            "headers": [(b"authorization", b"Bearer stack-token")],
        },
        receive=receive,
    )
    response = await preview_proxy.proxy_artifact_request(
        "token", "api/v1/memory/", request, scope=LOCAL_SCOPE,
    )

    assert response.status_code == 403
    assert response.body.decode() == sandbox.BACKENDS_OFF_MESSAGE


@pytest.mark.parametrize("sandbox", ["off", "landlock"])
async def test_the_backend_pool_carries_the_policy(landlock_kernel, monkeypatch, tmp_path, sandbox):
    """Behind the refusal, the pool still carries the policy, which anton's
    launch_artifact_backend refuses on its own."""
    from cowork.services import artifacts

    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_CODE_SANDBOX", sandbox)
    get_app_settings.cache_clear()
    pools: list = []

    async def fake_launch(*, scratchpad_pool, **_kwargs):
        pools.append(scratchpad_pool)
        return "Error: stop here"

    monkeypatch.setattr("anton.core.artifacts.backend_launcher.launch_artifact_backend", fake_launch)
    monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})
    project = _project(tmp_path, "projects", "proj")
    artifact_dir = _project(project, ".anton", "artifacts", "app")

    await artifacts._launch_backend_locked(artifact_dir, project_root=project, scope=LOCAL_SCOPE)
    get_app_settings.cache_clear()

    (pool,) = pools
    if sandbox == "off":
        assert pool.cell_sandbox is None
    else:
        assert pool.cell_sandbox.project_dir == project.resolve()


class _RecordingRuntime:
    calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).calls.append((args, kwargs))


def test_the_pool_runtime_off_gets_todays_arguments(monkeypatch, tmp_path):
    from cowork.services import scratchpad_runtime

    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    monkeypatch.setattr(_RecordingRuntime, "calls", [])
    monkeypatch.setattr("anton.core.backends.local.LocalScratchpadRuntime", _RecordingRuntime)
    scratchpad_runtime._make_runtime(
        "app", workspace=tmp_path, coding_provider="openai", coding_model="m",
        coding_api_key="k", coding_base_url="u",
    )
    assert _RecordingRuntime.calls == [(
        ("app",),
        {
            "coding_provider": "openai", "coding_model": "m", "coding_api_key": "k",
            "coding_base_url": "u", "workspace_path": tmp_path,
        },
    )]


async def test_the_pool_hands_its_policy_to_each_pad(monkeypatch, tmp_path):
    from cowork.services import scratchpad_runtime

    policy = object()
    monkeypatch.setattr(_RecordingRuntime, "calls", [])
    monkeypatch.setattr("anton.core.backends.local.LocalScratchpadRuntime", _RecordingRuntime)
    monkeypatch.setattr(scratchpad_runtime, "_pads", {})
    monkeypatch.setattr(scratchpad_runtime, "_resolve_coding", lambda **coding: tuple(coding.values()))
    pool = scratchpad_runtime.WorkspaceScopedPool(str(tmp_path), cell_sandbox=policy)

    await pool.get_or_create("app")

    (_args, kwargs), = _RecordingRuntime.calls
    assert kwargs["cell_sandbox"] is policy


def test_a_pad_is_never_handed_to_a_caller_under_another_policy(monkeypatch, tmp_path):
    """The registry keys pads by workspace and name, and a pad's policy binds
    when it starts, so another policy's caller is refused, not handed it."""
    from cowork.services import scratchpad_runtime

    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    alices, bobs = object(), object()
    monkeypatch.setattr(_RecordingRuntime, "calls", [])
    monkeypatch.setattr("anton.core.backends.local.LocalScratchpadRuntime", _RecordingRuntime)
    monkeypatch.setattr(scratchpad_runtime, "_pads", {})
    monkeypatch.setattr(scratchpad_runtime, "_resolve_coding", lambda **coding: tuple(coding.values()))
    workspace = str(tmp_path)

    pad = scratchpad_runtime.get_or_create("app", workspace_path=workspace, cell_sandbox=alices)

    assert scratchpad_runtime.get_or_create("app", workspace_path=workspace, cell_sandbox=alices) is pad
    for other in (bobs, None):
        with pytest.raises(RuntimeError, match="runs under another code-sandbox policy"):
            scratchpad_runtime.get_or_create("app", workspace_path=workspace, cell_sandbox=other)
    assert scratchpad_runtime.get("app", workspace_path=workspace) is pad
    assert len(_RecordingRuntime.calls) == 1


# ── Code mode ────────────────────────────────────────────────────────────────


def _loopback(app) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/v1/coding/terminal-shells"),
        ("get", "/api/v1/coding/skills/personal/some-skill"),
        ("post", "/api/v1/coding/runtime/register"),
    ],
)
def test_code_mode_answers_403_under_the_sandbox(stack, method, path):
    client = _loopback(cowork_server.create_app())
    response = getattr(client, method)(path, headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 403
    assert response.json() == {"detail": NO_CODE_MODE_DETAIL}


def test_code_mode_answers_with_the_sandbox_off(monkeypatch):
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    get_app_settings.cache_clear()
    response = _loopback(cowork_server.create_app()).get("/api/v1/coding/terminal-shells")
    assert response.status_code == 200


def test_codex_refuses_to_start_under_the_sandbox(landlock_alone):
    from cowork.coding.engines.codex import CodexEngineSession

    with pytest.raises(RuntimeError) as refused:
        CodexEngineSession(
            cowork_root=Path("/unused"), workspace=Path("/unused"), config=None,
            credentials=None, existing_session_id=None, approval_handler=None,
        )
    assert str(refused.value) == NO_CODE_MODE_DETAIL


# ── the server's own writes in a project ─────────────────────────────────────


def test_makedirs_below_nofollow_makes_the_chain(tmp_path):
    base = _project(tmp_path, "project")
    makedirs_below_nofollow(base, target=base / ".anton" / "artifacts")
    assert (base / ".anton" / "artifacts").is_dir()


def test_makedirs_below_nofollow_refuses_a_planted_link(tmp_path):
    base = _project(tmp_path, "project")
    outside = _project(tmp_path, "outside")
    (base / ".anton").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        makedirs_below_nofollow(base, target=base / ".anton" / "artifacts")
    assert list(outside.iterdir()) == []


def test_makedirs_outside_the_base_is_plain(tmp_path):
    base = _project(tmp_path, "project")
    makedirs_below_nofollow(base, target=tmp_path / "elsewhere" / "context")
    makedirs_below_nofollow(base, target=base / ".." / "beside")
    assert (tmp_path / "elsewhere" / "context").is_dir()
    assert (tmp_path / "beside").is_dir()


async def _launch_on_port(monkeypatch, *, project: Path, folder: Path, port: int):
    from cowork.services import artifacts

    async def fake_launch(**_kwargs):
        return {"port": port, "pid": 1}

    monkeypatch.setattr("anton.core.artifacts.backend_launcher.launch_artifact_backend", fake_launch)
    monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})
    return await artifacts._launch_backend_locked(folder, project_root=project.resolve(), scope=LOCAL_SCOPE)


async def test_a_launch_writes_its_port_into_metadata(monkeypatch, tmp_path):
    project = _project(tmp_path, "project")
    folder = _project(project, ".anton", "artifacts", "app")
    (folder / "metadata.json").write_text(json.dumps({"slug": "app", "port": 1}), encoding="utf-8")
    status = await _launch_on_port(monkeypatch, project=project, folder=folder, port=4242)
    assert status == (True, "launched", 4242)
    assert json.loads((folder / "metadata.json").read_text(encoding="utf-8")) == {"slug": "app", "port": 4242}


async def test_a_launch_writes_its_port_as_the_desktop_always_has(monkeypatch, tmp_path):
    """Backends launch only with the sandbox off, so the port is written as on
    the desktop: through a metadata.json the project links, in place, keeping
    the file's mode."""
    project = _project(tmp_path, "project")
    folder = _project(project, ".anton", "artifacts", "app")
    target = _project(tmp_path, "elsewhere") / "metadata.json"
    target.write_text(json.dumps({"slug": "app", "port": 1}), encoding="utf-8")
    target.chmod(0o644)
    (folder / "metadata.json").symlink_to(target)

    await _launch_on_port(monkeypatch, project=project, folder=folder, port=4242)

    assert json.loads(target.read_text(encoding="utf-8")) == {"slug": "app", "port": 4242}
    assert (folder / "metadata.json").is_symlink()
    assert target.stat().st_mode & 0o777 == 0o644
