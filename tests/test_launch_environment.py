"""A local launch moves the server's own COWORK_* and MINDSHUB_* variables out
of its environment (cowork.common.launch_environment), so git, project
commands, terminals, the Codex app-server and anton's cells inherit none of
them, while the server keeps reading the values it started with.

Each runtime reader of those variables is pinned below against a scrubbed
environment: the live variable is gone and the captured value is all there is.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import pytest

from cowork.common import launch_environment
from cowork.common.launch_environment import launch_value, scrub_launch_environment


@pytest.fixture
def scrubbed(monkeypatch):
    """Stand in for a launch that captured ``values`` and removed them from
    os.environ. Returns a setter."""

    def scrub(**values: str) -> None:
        for name in values:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(launch_environment, "_captured", MappingProxyType(dict(values)))

    monkeypatch.setattr(launch_environment, "_captured", MappingProxyType({}))
    return scrub


class TestLaunchValue:
    def test_the_live_environment_wins(self, scrubbed, monkeypatch):
        scrubbed(COWORK_PROBE="captured")
        monkeypatch.setenv("COWORK_PROBE", "live")
        assert launch_value("COWORK_PROBE") == "live"

    def test_the_captured_value_answers_once_the_variable_is_gone(self, scrubbed):
        scrubbed(COWORK_PROBE="captured")
        assert launch_value("COWORK_PROBE") == "captured"

    def test_nothing_answers_none(self, scrubbed):
        scrubbed()
        assert launch_value("COWORK_PROBE") is None


def test_the_scrub_moves_only_its_prefixes(monkeypatch):
    fake_environment = {
        "COWORK_HOME": "/srv/cowork",
        "MINDSHUB_CODEX_API_KEY": "fake0000",
        "ANTON_MINDS_API_KEY": "fake1111",
        "ANTHROPIC_API_KEY": "fake2222",
        "PATH": "/usr/bin",
        "XCOWORK_HOME": "kept",
    }
    monkeypatch.setattr(os, "environ", fake_environment)
    monkeypatch.setattr(launch_environment, "_captured", MappingProxyType({}))

    scrub_launch_environment()

    assert fake_environment == {
        "ANTON_MINDS_API_KEY": "fake1111",
        "ANTHROPIC_API_KEY": "fake2222",
        "PATH": "/usr/bin",
        "XCOWORK_HOME": "kept",
    }
    assert dict(launch_environment.captured_environment()) == {
        "COWORK_HOME": "/srv/cowork",
        "MINDSHUB_CODEX_API_KEY": "fake0000",
    }


class TestRuntimeReaders:
    """Each reader that runs after the scrub, given only the captured value."""

    def test_cowork_home(self, scrubbed, tmp_path):
        from cowork.common.paths import cowork_home

        scrubbed(COWORK_HOME=str(tmp_path / "account"))
        assert cowork_home() == tmp_path / "account"

    def test_settings_built_after_the_scrub(self, scrubbed, tmp_path):
        from cowork.common.settings.app_settings import (
            CodingSettings,
            ConnectorSettings,
            OAuthSettings,
            TurnQueueSettings,
        )

        scrubbed(
            COWORK_CODING_DIR=str(tmp_path / "org" / "coding"),
            COWORK_VAULT_DIR=str(tmp_path / "org" / "data-vault"),
            COWORK_SERVER_ORIGIN="http://127.0.0.1:41234",
            COWORK_TURN_JEV_ROUTE_MODEL="jev-probe",
        )
        assert CodingSettings().root_dir == str(tmp_path / "org" / "coding")
        assert ConnectorSettings().vault_dir == str(tmp_path / "org" / "data-vault")
        assert OAuthSettings().server_origin == "http://127.0.0.1:41234"
        # An env_prefix class reads the captured values under its prefix.
        assert TurnQueueSettings().jev_route_model == "jev-probe"

    def test_app_settings_built_after_the_scrub(self, scrubbed):
        """AppSettings lists its own sources, for the credential files, and
        still ranks the captured values under the live environment."""
        from cowork.common.settings.app_settings import AppSettings

        scrubbed(COWORK_SERVER_PORT="41235")
        assert AppSettings().port == 41235

    def test_the_credential_files(self, scrubbed, monkeypatch, tmp_path):
        from cowork.common.settings.secret_files import (
            SecretFileError,
            process_secret,
            read_secret_file,
            require_process_secrets,
        )

        token_file = tmp_path / "auth-token"
        token_file.write_text("file-token\n", encoding="utf-8")
        publish_file = tmp_path / "publish-token"
        publish_file.write_text("publish-token", encoding="utf-8")
        monkeypatch.setenv("DATABASE_URI", "postgresql://db.example.test/cowork")
        monkeypatch.setenv("MINDS_API_KEY", "minds-key")
        scrubbed(COWORK_AUTH_TOKEN_FILE=str(token_file), COWORK_PUBLISH_TOKEN_FILE=str(publish_file))

        assert process_secret(name="COWORK_AUTH_TOKEN") == "file-token"
        assert read_secret_file(name="COWORK_PUBLISH_TOKEN") == "publish-token"
        require_process_secrets()

        scrubbed(COWORK_AUTH_TOKEN="env-token", COWORK_AUTH_TOKEN_FILE=str(token_file))
        with pytest.raises(SecretFileError, match="both set"):
            read_secret_file(name="COWORK_AUTH_TOKEN")

    def test_the_key_file(self, scrubbed, tmp_path):
        from cowork.common.encryption import key_file_path

        scrubbed(COWORK_KEY_FILE=str(tmp_path / "keys"))
        assert key_file_path() == tmp_path / "keys"

    def test_pydantic_settings_still_calls_the_launch_environment_hook(self, scrubbed):
        """LaunchEnvironmentSource overrides _load_env_vars, a private
        pydantic-settings method, and the dependency has only a floor. A class
        that reads only through that source, and needs a value only the
        captured environment holds, loads only while pydantic-settings still
        calls the hook."""
        from pydantic_settings import BaseSettings

        from cowork.common.settings.app_settings import LaunchEnvironmentSource

        class HookOnly(BaseSettings):
            cowork_probe_hook: str

            @classmethod
            def settings_customise_sources(
                cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
            ):
                return (LaunchEnvironmentSource(settings_cls),)

        scrubbed(COWORK_PROBE_HOOK="captured")
        assert HookOnly().cowork_probe_hook == "captured"

    def test_settings_rank_the_live_environment_first(self, scrubbed, monkeypatch, tmp_path):
        from cowork.common.settings.app_settings import CodingSettings

        scrubbed(COWORK_CODING_DIR=str(tmp_path / "captured"))
        monkeypatch.setenv("COWORK_CODING_DIR", str(tmp_path / "live"))
        assert CodingSettings().root_dir == str(tmp_path / "live")

    def test_the_turn_idle_bound(self, scrubbed):
        from cowork.streaming.registry import _idle_bound_seconds

        scrubbed(COWORK_MAX_TURN_IDLE_SECONDS="1234")
        assert _idle_bound_seconds() == 1234

    def test_the_redis_url(self, scrubbed):
        from cowork.turnqueue import redis_client

        scrubbed(COWORK_TURN_REDIS_URL="redis://redis.example.test:6380/2")
        assert redis_client._url() == "redis://redis.example.test:6380/2"

    def test_the_codex_compaction_limit(self, scrubbed):
        from cowork.coding.engines import codex_config

        scrubbed(MINDSHUB_CODE_AUTO_COMPACT_TOKEN_LIMIT="48000")
        assert codex_config.auto_compact_token_limit() == 48000

    def test_the_connector_test_allowlist(self, scrubbed):
        from cowork.services.connectors import connector_test_run

        scrubbed(COWORK_CONNECTOR_TEST_PRIVATE_ALLOWLIST="db.internal,10.1.0.0/16")
        connector_test_run.private_allowlist.cache_clear()
        try:
            hosts, networks = connector_test_run.private_allowlist()
        finally:
            connector_test_run.private_allowlist.cache_clear()
        assert "db.internal" in hosts
        assert [str(network) for network in networks] == ["10.1.0.0/16"]

    def test_the_pre_token_desktop_check(self, scrubbed, monkeypatch):
        from cowork.common.settings import app_settings

        monkeypatch.delenv("COWORK_SERVER_OWNER", raising=False)
        scrubbed(COWORK_SERVER_HOST="127.0.0.1")
        assert app_settings._launched_by_pre_token_desktop() is True
        scrubbed(COWORK_SERVER_HOST="127.0.0.1", COWORK_SERVER_OWNER="owner-token")
        assert app_settings._launched_by_pre_token_desktop() is False

    def test_the_env_file_chain(self, scrubbed, monkeypatch, tmp_path):
        from cowork.common.settings import app_settings

        monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
        scrubbed(COWORK_HOME=str(tmp_path / "account"))
        # An isolated home reads no legacy ~/.anton/.env.
        assert app_settings._env_file_chain() == [str(tmp_path / "account" / ".env"), ".env"]
        scrubbed(COWORK_HOME=str(tmp_path / "account"), COWORK_TENANCY_MODE="org")
        assert app_settings._env_file_chain() == [".env"]

    def test_the_developer_oauth_file_pointer(self, scrubbed, monkeypatch, tmp_path):
        from pathlib import Path

        from cowork.common.settings import app_settings

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        scrubbed(COWORK_DEV_OAUTH_ENV_FILE=str(tmp_path / ".cowork-dev" / ".env"))
        assert app_settings._dev_oauth_env_path() == tmp_path / ".cowork-dev" / ".env"


_ROOT = Path(__file__).resolve().parents[1]

#: How the scan names ``os.environ`` used whole (passed to a call, bound to a
#: name or returned): it cannot see which variables are read from it after that.
_WHOLE_ENVIRONMENT = "os.environ"


class _RawRead(NamedTuple):
    """One raw read of the process environment: the file, the function or
    class that makes it (a dotted name, or ``<module>``), and the variable."""

    path: str
    scope: str
    name: str


#: The raw reads of a COWORK_* or MINDSHUB_* variable that the scan below
#: finds in cowork/, each with why it stays right. Everything else reads
#: through launch_value or a settings class, since the scrub has emptied
#: os.environ of these names by the time it runs. An entry allows its variable
#: in its own function or class only. The scan sees ``os.environ.get``,
#: ``os.environ[...]``, ``in os.environ``, ``os.getenv`` and ``os.environ``
#: used whole. It does not follow a copy such as ``os.environ.copy()``, or a
#: variable name built at run time.
_RAW_READS_ALLOWED: Mapping[_RawRead, str] = {
    _RawRead("cowork/cli.py", "main", "COWORK_APP_SECRET_STDIN"): (
        "runs before the scrub, and before .env joins the environment"
    ),
    _RawRead("cowork/cli.py", "main", "COWORK_EXPORT_PDF_ENGINE"): (
        "runs before the scrub, to keep the PDF setting the server started with"
    ),
    _RawRead("cowork/cli.py", "main", "COWORK_EXPORT_RENDER_URL"): (
        "runs before the scrub, to keep the PDF setting the server started with"
    ),
    _RawRead("cowork/common/settings/app_settings.py", "AppSettings", "COWORK_SERVER_OWNER"): (
        "a field default, evaluated at import, before the scrub"
    ),
    _RawRead("cowork/common/settings/app_settings.py", "AppSettings", "COWORK_SERVER_HEALTH_PROOF_KEY"): (
        "a field default, evaluated at import, before the scrub"
    ),
    _RawRead("cowork/common/settings/app_settings.py", "get_export_pdf_settings", _WHOLE_ENVIRONMENT): (
        "reads the two PDF variables once: lru_cache holds the result, and cli.main calls it before the scrub"
    ),
    _RawRead("cowork/coding/runtime_worker.py", "main", "COWORK_RUNTIME_SERVER"): (
        "the cowork-code-runtime command's own process, which never scrubs"
    ),
    _RawRead("cowork/coding/runtime_worker.py", "main", "COWORK_RUNTIME_REGISTRATION_TOKEN"): (
        "the cowork-code-runtime command's own process, which never scrubs"
    ),
    _RawRead("cowork/coding/runtime_worker.py", "main", "COWORK_RUNTIME_NAME"): (
        "the cowork-code-runtime command's own process, which never scrubs"
    ),
    _RawRead("cowork/coding/runtime_worker.py", "main", "COWORK_RUNTIME_ROOT"): (
        "the cowork-code-runtime command's own process, which never scrubs"
    ),
    _RawRead("cowork/ops/reseal.py", "_legacy_key_left", "COWORK_MASTER_KEY"): (
        "the cowork.ops re-seal command's own process, which never scrubs"
    ),
}


def _string_constants(*, tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """Each name a source binds to a string, or to a tuple, list or set of
    strings, with those strings."""
    constants: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            strings: tuple[str, ...] = (value.value,)
        elif isinstance(value, (ast.Tuple, ast.List, ast.Set)) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts
        ):
            strings = tuple(item.value for item in value.elts)
        else:
            continue
        constants[target.id] = constants.get(target.id, ()) + strings
    return constants


def _raw_environment_reads(*, path: str, tree: ast.AST, shared: Mapping[str, tuple[str, ...]]) -> set[_RawRead]:
    """The raw reads of the process environment in one parsed source:
    ``os.environ.get(...)``, ``os.environ[...]``, ``... in os.environ`` and
    ``os.getenv(...)``, under any import alias, and ``os.environ`` used whole
    (passed to a call, bound to a name or returned) as ``_WHOLE_ENVIRONMENT``.

    A variable name resolves when it is a string literal, the leading literal
    of an f-string (as ``PREFIX*``), a constant this source or ``shared``
    binds, or the variable of a loop over such strings. A name built any other
    way is not seen, and neither is a read from a copy such as
    ``os.environ.copy()``.
    """
    os_names, environ_names, getenv_names = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names |= {alias.asname or alias.name for alias in node.names if alias.name == "os"}
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            environ_names |= {alias.asname or alias.name for alias in node.names if alias.name == "environ"}
            getenv_names |= {alias.asname or alias.name for alias in node.names if alias.name == "getenv"}
    if not (os_names or environ_names or getenv_names):
        return set()
    local = _string_constants(tree=tree)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def is_environ(*, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in environ_names
        return (
            isinstance(node, ast.Attribute) and node.attr == "environ"
            and isinstance(node.value, ast.Name) and node.value.id in os_names
        )

    def is_getenv(*, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in getenv_names
        return (
            isinstance(node, ast.Attribute) and node.attr == "getenv"
            and isinstance(node.value, ast.Name) and node.value.id in os_names
        )

    def constant(*, name: str) -> tuple[str, ...]:
        return local.get(name) or shared.get(name, ())

    def strings(*, node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return (node.value,)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return tuple(value for item in node.elts for value in strings(node=item))
        if isinstance(node, ast.Attribute):
            return constant(name=node.attr)
        if isinstance(node, ast.Name):
            return constant(name=node.id)
        return ()

    def loop_values(*, name: ast.Name) -> tuple[str, ...]:
        node: ast.AST = name
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.For, ast.AsyncFor)):
                loops = [node]
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                loops = node.generators
            else:
                continue
            for loop in loops:
                if isinstance(loop.target, ast.Name) and loop.target.id == name.id:
                    return strings(node=loop.iter)
        return ()

    def names(*, key: ast.AST) -> tuple[str, ...]:
        if isinstance(key, ast.JoinedStr):
            first = key.values[0] if key.values else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return (first.value + "*",)
            return ()
        if isinstance(key, ast.Name):
            return loop_values(name=key) or constant(name=key.id)
        return strings(node=key)

    def read_key(*, node: ast.AST) -> ast.AST | None:
        """The key ``node`` reads the environment by, or None if it reads none."""
        if isinstance(node, ast.Call) and node.args:
            function = node.func
            if is_getenv(node=function) or (
                isinstance(function, ast.Attribute) and function.attr == "get" and is_environ(node=function.value)
            ):
                return node.args[0]
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and is_environ(node=node.value):
            return node.slice
        elif (
            isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and is_environ(node=node.comparators[0])
        ):
            return node.left
        return None

    def used_whole(*, node: ast.AST) -> bool:
        """Whether ``node`` is ``os.environ`` itself, used other than read by
        key, written by key, or reached through a method such as ``.copy()``."""
        if not is_environ(node=node):
            return False
        parent = parents.get(node)
        return not (
            isinstance(parent, ast.Attribute)
            or (isinstance(parent, ast.Subscript) and parent.value is node)
            or (isinstance(parent, ast.Compare) and read_key(node=parent) is not None)
        )

    def scope(*, node: ast.AST) -> str:
        owners: list[str] = []
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                owners.append(node.name)
        return ".".join(reversed(owners)) or "<module>"

    reads: set[_RawRead] = set()
    for node in ast.walk(tree):
        key = read_key(node=node)
        if key is not None:
            reads |= {_RawRead(path, scope(node=node), name) for name in names(key=key)}
        elif used_whole(node=node):
            reads.add(_RawRead(path, scope(node=node), _WHOLE_ENVIRONMENT))
    return reads


def test_the_scan_sees_each_form_of_a_raw_read():
    source = """
import os
import os as operating_system
from os import environ as env, getenv

NAME = "COWORK_CONSTANT"
NAMES = ("COWORK_IN_A_TUPLE", "PATH")

os.environ.get("COWORK_GET")
os.environ["MINDSHUB_SUBSCRIPT"]
os.getenv("COWORK_GETENV")
getenv("COWORK_IMPORTED_GETENV")
operating_system.environ.get("COWORK_ALIASED")
"COWORK_MEMBERSHIP" in os.environ
"COWORK_NOT_IN" not in env
env.get(NAME)
os.environ.get(other_module.SHARED)
started = {name: os.environ.get(name) for name in NAMES}
for each in ("COWORK_LOOP",):
    os.environ.get(each)
os.environ.get(f"COWORK_{suffix}")
os.environ["COWORK_WRITTEN"] = "1"
del os.environ["COWORK_REMOVED"]
os.environ.get("ANTON_KEPT")
copied = os.environ.copy()


class Holder:
    field = os.environ.get("COWORK_FIELD_DEFAULT")

    def method(self):
        return os.getenv("COWORK_IN_A_METHOD")


def hand_on():
    return load(environ=os.environ)


def bind():
    alias = os.environ
    return alias.get("COWORK_THROUGH_AN_ALIAS")


def give():
    return env
"""
    at_module = (
        "COWORK_GET",
        "MINDSHUB_SUBSCRIPT",
        "COWORK_GETENV",
        "COWORK_IMPORTED_GETENV",
        "COWORK_ALIASED",
        "COWORK_MEMBERSHIP",
        "COWORK_NOT_IN",
        "COWORK_CONSTANT",
        "COWORK_SHARED",
        "COWORK_IN_A_TUPLE",
        "PATH",
        "COWORK_LOOP",
        "COWORK_*",
        "ANTON_KEPT",
    )
    reads = _raw_environment_reads(path="probe.py", tree=ast.parse(source), shared={"SHARED": ("COWORK_SHARED",)})
    assert reads == {
        *(_RawRead("probe.py", "<module>", name) for name in at_module),
        _RawRead("probe.py", "Holder", "COWORK_FIELD_DEFAULT"),
        _RawRead("probe.py", "Holder.method", "COWORK_IN_A_METHOD"),
        _RawRead("probe.py", "hand_on", _WHOLE_ENVIRONMENT),
        _RawRead("probe.py", "bind", _WHOLE_ENVIRONMENT),
        _RawRead("probe.py", "give", _WHOLE_ENVIRONMENT),
    }


def test_no_new_raw_read_of_a_scrubbed_variable():
    """A raw read of a COWORK_* or MINDSHUB_* variable after the scrub answers
    None. Read it through launch_value (or a settings class), or add the site
    to _RAW_READS_ALLOWED with why the raw read stays right. ``os.environ``
    used whole needs an entry too, since the scan cannot see what reads it."""
    trees = {
        path.relative_to(_ROOT).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted((_ROOT / "cowork").rglob("*.py"))
    }
    shared: dict[str, tuple[str, ...]] = {}
    for tree in trees.values():
        for name, values in _string_constants(tree=tree).items():
            shared[name] = shared.get(name, ()) + values
    found = {
        read
        for path, tree in trees.items()
        for read in _raw_environment_reads(path=path, tree=tree, shared=shared)
        if read.name == _WHOLE_ENVIRONMENT or read.name.startswith(launch_environment.SCRUBBED_PREFIXES)
    }

    assert sorted(found - _RAW_READS_ALLOWED.keys()) == []
    # An entry whose read is gone comes off the list.
    assert sorted(_RAW_READS_ALLOWED.keys() - found) == []


_LAUNCH = """
import json, os, subprocess, sys
import uvicorn
uvicorn.run = lambda *a, **k: None
import cowork
from cowork import cli
cli.main()
from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import CodingSettings
child = subprocess.run(
    [sys.executable, "-c", "import json, os; print(json.dumps(sorted(os.environ)))"],
    capture_output=True, text=True, check=True,
)
child_names = json.loads(child.stdout)
print("RESULT" + json.dumps({
    "own": sorted(name for name in os.environ if name.startswith(("COWORK_", "MINDSHUB_"))),
    "child": [name for name in child_names if name.startswith(("COWORK_", "MINDSHUB_"))],
    "anton_own": os.environ.get("ANTON_PROBE_MARKER"),
    "anton_child": "ANTON_PROBE_MARKER" in child_names,
    "home": str(cowork_home()),
    "coding_dir": CodingSettings().root_dir,
    "package": cowork.__file__,
}))
"""


def test_a_local_launch_leaves_no_cowork_or_mindshub_variable_to_its_children(tmp_path):
    """cowork.cli.main with uvicorn stubbed out: the server's own environment,
    and a process it starts afterwards, hold none of its variables, anton's
    stay, and the server still reads the values it started with."""
    env = dict(os.environ)
    env.update({
        # The child runs in tmp_path, so without this it imports whatever
        # cowork the interpreter finds, which in a shared virtualenv is
        # another checkout.
        "PYTHONPATH": str(_ROOT),
        "COWORK_HOME": str(tmp_path / "home"),
        "COWORK_CODING_DIR": str(tmp_path / "org" / "coding"),
        "MINDSHUB_PROBE_MARKER": "fake-mindshub",
        "ANTON_PROBE_MARKER": "fake-anton",
    })
    env.pop("COWORK_APP_SECRET_STDIN", None)
    result = subprocess.run(
        [sys.executable, "-c", _LAUNCH],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT"))
    seen = json.loads(line[len("RESULT"):])

    assert Path(seen["package"]).resolve().is_relative_to(_ROOT)
    assert seen["own"] == []
    assert seen["child"] == []
    assert seen["anton_own"] == "fake-anton"
    assert seen["anton_child"] is True
    assert seen["home"] == str(tmp_path / "home")
    assert seen["coding_dir"] == str(tmp_path / "org" / "coding")
