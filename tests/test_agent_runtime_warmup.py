"""Startup imports the agent runtime so the first turn does not pay for it."""
import importlib

from cowork import server


def test_every_listed_module_exists():
    for name in server.AGENT_RUNTIME_MODULES:
        assert importlib.util.find_spec(name) is not None, name


def test_warm_up_imports_each_listed_module(monkeypatch):
    imported = []
    monkeypatch.setattr(importlib, "import_module", imported.append)
    server._warm_agent_runtime()
    assert imported == list(server.AGENT_RUNTIME_MODULES)


def test_a_failing_import_does_not_stop_startup(monkeypatch):
    imported = []

    def flaky(name):
        if name == server.AGENT_RUNTIME_MODULES[0]:
            raise ImportError("broken optional dependency")
        imported.append(name)

    monkeypatch.setattr(importlib, "import_module", flaky)
    server._warm_agent_runtime()  # must not raise
    assert imported == list(server.AGENT_RUNTIME_MODULES[1:])


def test_every_turn_path_module_exists():
    for name in server.TURN_PATH_MODULES:
        assert importlib.util.find_spec(name) is not None, name


def test_turn_path_warm_up_imports_its_modules_and_caches_the_platform(monkeypatch):
    import platform

    imported, asked = [], []
    monkeypatch.setattr(importlib, "import_module", imported.append)
    monkeypatch.setattr(platform, "platform", lambda: asked.append("platform") or "x")
    monkeypatch.setattr(platform, "machine", lambda: asked.append("machine") or "x")
    server._warm_turn_path()
    assert imported == list(server.TURN_PATH_MODULES)
    assert asked == ["platform", "machine"]


def test_turn_path_warm_up_never_raises(monkeypatch):
    import platform

    def broken(*_a, **_k):
        raise RuntimeError("no uname")

    monkeypatch.setattr(importlib, "import_module", broken)
    monkeypatch.setattr(platform, "platform", broken)
    server._warm_turn_path()  # must not raise
