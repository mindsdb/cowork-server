"""The MCP client must be optional at the anton version boundary.

cowork-server pins anton from a git branch, but its *published* floor
(`anton-agent>=2.26.8.23.1`, pyproject.toml) still resolves builds with no
`anton.core.mcp` package: promoting the MCP client to anton's `main` is a
separate release step (ENG-1816), so there is a real window where a perfectly
valid dependency lacks it.

Before the gate, `_build_chat_session` imported `anton.core.mcp.wiring`
unconditionally and passed `mcp_sessions=` to `ChatSessionConfig` — so on that
skew EVERY turn died (ImportError, then TypeError), not just the MCP ones.
Same tolerance convention as `services/providers.py`'s `router_provider` gate
and `handlers/turn_errors.py`'s refusal to import a type this repo can be
deployed ahead of.
"""
from __future__ import annotations

import sys

from cowork.harnesses.anton_harness.harness import _anton_mcp_wiring


def test_returns_none_when_the_installed_anton_has_no_mcp_client(monkeypatch):
    # A `None` entry in sys.modules makes `import anton.core.mcp.wiring` raise
    # ImportError — exactly what an older anton does. Runs on every build,
    # including ones that do ship the module, so the skew path is never left
    # untested just because CI happens to be on a newer anton.
    monkeypatch.setitem(sys.modules, "anton.core.mcp.wiring", None)

    assert _anton_mcp_wiring() is None


def test_resolves_the_wiring_module_when_anton_provides_it():
    wiring = _anton_mcp_wiring()

    if wiring is None:
        # Nothing to assert on an anton without the client; the companion test
        # above is what covers this build.
        return
    # The two entry points the harness actually calls. Named explicitly so a
    # rename in anton fails here rather than at turn time.
    assert callable(wiring.discover_mcp_tools_async)
    assert callable(wiring.close_mcp_sessions)
