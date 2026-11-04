"""Stand-ins for the Chimera anton, for tests on the uv.lock anton.

The pinned anton has no `anton.core.chimera`, no sidekick role on LLMClient and
no `chimera` / `sidekick_state` on ChatSessionConfig. ``install`` puts all three
in place for one test, shaped as the Chimera contract declares them, so the
server's guarded wiring can be driven end to end without the real package.
"""
from __future__ import annotations

import dataclasses
import sys
from types import ModuleType

from anton.core.llm.client import LLMClient as _RealLLMClient
from anton.core.session import ChatSessionConfig as _RealChatSessionConfig


@dataclasses.dataclass
class ChimeraConfig:
    enabled: bool
    max_rounds: int = 20
    max_rounds_cap: int = 40
    jev_every: int = 4
    jev_threshold: float = 0.8
    upgrade_ladder: tuple[str, ...] = ()


class ChimeraLLMClient(_RealLLMClient):
    """The real client, plus the contract's sidekick role and cache lifetime."""

    def __init__(self, *, sidekick_provider=None, sidekick_model=None, main_cache_ttl=None, **kwargs):
        super().__init__(**kwargs)
        self.sidekick_provider = sidekick_provider
        self.sidekick_model = sidekick_model
        self.main_cache_ttl = main_cache_ttl
        self.has_sidekick = bool(sidekick_model)


def _chimera_session_config():
    """ChatSessionConfig with the contract's two Chimera inputs added."""
    fields = []
    for f in dataclasses.fields(_RealChatSessionConfig):
        if f.default is not dataclasses.MISSING:
            spec = dataclasses.field(default=f.default)
        elif f.default_factory is not dataclasses.MISSING:
            spec = dataclasses.field(default_factory=f.default_factory)
        else:
            spec = dataclasses.field()
        fields.append((f.name, f.type, spec))
    fields.append(("chimera", "object", dataclasses.field(default=None)))
    fields.append(("sidekick_state", "object", dataclasses.field(default=None)))
    return dataclasses.make_dataclass("ChatSessionConfig", fields)


ChimeraChatSessionConfig = _chimera_session_config()


def install(monkeypatch, *, config=True, client=True, session=True) -> None:
    """Make the installed anton look like the Chimera one, piece by piece."""
    if config:
        package = ModuleType("anton.core.chimera")
        module = ModuleType("anton.core.chimera.config")
        module.ChimeraConfig = ChimeraConfig
        package.config = module
        monkeypatch.setitem(sys.modules, "anton.core.chimera", package)
        monkeypatch.setitem(sys.modules, "anton.core.chimera.config", module)
    if client:
        monkeypatch.setattr("anton.core.llm.client.LLMClient", ChimeraLLMClient)
    if session:
        monkeypatch.setattr("anton.core.session.ChatSessionConfig", ChimeraChatSessionConfig)
