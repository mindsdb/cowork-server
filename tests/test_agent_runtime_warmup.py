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
