"""The MCP client must be optional at the anton version boundary.

Every anton this server can start with has the `anton.core.mcp` package (the
MCP client, ENG-1816): it predates the `anton.core.artifacts.internal_files`
constants the server imports at startup. So `_anton_mcp_wiring`'s `None`
branch serves only an anton too old to start this server. It returns `None`
there rather than raising, the same tolerance convention as
`services/providers.py`'s `router_provider` gate.
"""
from __future__ import annotations

import sys

from cowork.harnesses.anton_harness.harness import _anton_mcp_wiring


def test_returns_none_when_the_installed_anton_has_no_mcp_client(monkeypatch):
    # A `None` entry in sys.modules makes the import raise ImportError, as on
    # an older anton. Stub the package, not just `wiring`: once another test
    # has imported the real module, `from anton.core.mcp import wiring` finds
    # it as a package attribute and never consults the submodule entry.
    # Runs on every build, so the skew path stays tested on a newer anton too.
    monkeypatch.setitem(sys.modules, "anton.core.mcp", None)

    assert _anton_mcp_wiring() is None


def test_resolves_the_wiring_module_when_anton_provides_it():
    wiring = _anton_mcp_wiring()

    assert wiring is not None
    # The two entry points the harness actually calls. Named explicitly so a
    # rename in anton fails here rather than at turn time.
    assert callable(wiring.discover_mcp_tools_async)
    assert callable(wiring.close_mcp_sessions)
