"""The Chimera anton, present or absent, whichever anton is installed.

The uv.lock anton has no `anton.core.chimera`, no sidekick role on LLMClient
and no `chimera` / `sidekick_state` on ChatSessionConfig; anton's Chimera
branch has all three. ``install`` makes each piece present for one test (the
installed anton's own when it has it, else a stand-in shaped as the Chimera
contract declares it), and ``install(..., piece=False)`` or ``hide`` makes it
absent, so the server's guarded wiring is driven the same way on either anton.
"""
from __future__ import annotations

import dataclasses
import inspect
import sys
from types import ModuleType

from anton.core.llm.client import LLMClient as _RealLLMClient
from anton.core.session import ChatSessionConfig as _RealChatSessionConfig

try:
    from anton.core.chimera.config import ChimeraConfig as _RealChimeraConfig
except ImportError:
    _RealChimeraConfig = None

_CHIMERA_SESSION_FIELDS = ("chimera", "sidekick_state")
_CHIMERA_CLIENT_PARAMS = ("sidekick_provider", "sidekick_model", "main_cache_ttl")

_REAL_SESSION_FIELDS = {f.name for f in dataclasses.fields(_RealChatSessionConfig)}
_REAL_CLIENT_PARAMS = set(inspect.signature(_RealLLMClient.__init__).parameters)


if _RealChimeraConfig is not None:
    ChimeraConfig = _RealChimeraConfig
else:
    @dataclasses.dataclass(frozen=True)
    class ChimeraConfig:
        enabled: bool
        max_rounds: int = 20
        max_rounds_cap: int = 40
        jev_every: int = 4
        jev_threshold: float = 0.8
        upgrade_ladder: tuple[str, ...] = ()


if set(_CHIMERA_CLIENT_PARAMS) <= _REAL_CLIENT_PARAMS:
    ChimeraLLMClient = _RealLLMClient
else:
    class ChimeraLLMClient(_RealLLMClient):
        """The real client, plus the contract's sidekick role and cache lifetime."""

        def __init__(self, *, sidekick_provider=None, sidekick_model=None, main_cache_ttl=None, **kwargs):
            super().__init__(**kwargs)
            self.sidekick_provider = sidekick_provider
            self.sidekick_model = sidekick_model
            self.main_cache_ttl = main_cache_ttl
            self.has_sidekick = bool(sidekick_model)


class _ClientWithoutChimera(_RealLLMClient):
    """The installed client, declaring no Chimera kwarg, as the uv.lock anton's does."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


_ClientWithoutChimera.__init__.__signature__ = inspect.signature(_RealLLMClient.__init__).replace(
    parameters=[
        p for name, p in inspect.signature(_RealLLMClient.__init__).parameters.items()
        if name not in _CHIMERA_CLIENT_PARAMS
    ]
)


def _session_config(*, chimera: bool):
    """ChatSessionConfig with the contract's two Chimera inputs, or without them."""
    fields = []
    for f in dataclasses.fields(_RealChatSessionConfig):
        if f.name in _CHIMERA_SESSION_FIELDS:
            continue
        if f.default is not dataclasses.MISSING:
            spec = dataclasses.field(default=f.default)
        elif f.default_factory is not dataclasses.MISSING:
            spec = dataclasses.field(default_factory=f.default_factory)
        else:
            spec = dataclasses.field()
        fields.append((f.name, f.type, spec))
    if chimera:
        fields.extend((name, "object", dataclasses.field(default=None)) for name in _CHIMERA_SESSION_FIELDS)
    return dataclasses.make_dataclass("ChatSessionConfig", fields)


if set(_CHIMERA_SESSION_FIELDS) <= _REAL_SESSION_FIELDS:
    ChimeraChatSessionConfig = _RealChatSessionConfig
else:
    ChimeraChatSessionConfig = _session_config(chimera=True)


def installed_anton_has_chimera() -> bool:
    """Whether the anton in this venv ships all three pieces itself."""
    return (
        _RealChimeraConfig is not None
        and set(_CHIMERA_CLIENT_PARAMS) <= _REAL_CLIENT_PARAMS
        and set(_CHIMERA_SESSION_FIELDS) <= _REAL_SESSION_FIELDS
    )


def install(monkeypatch, *, config=True, client=True, session=True) -> None:
    """Make the installed anton look like the Chimera one, piece by piece: a
    piece passed True is present, one passed False is absent."""
    if not config:
        # An import of the module now fails, as on an anton without it.
        monkeypatch.setitem(sys.modules, "anton.core.chimera.config", None)
    elif _RealChimeraConfig is None:
        package = ModuleType("anton.core.chimera")
        module = ModuleType("anton.core.chimera.config")
        module.ChimeraConfig = ChimeraConfig
        package.config = module
        monkeypatch.setitem(sys.modules, "anton.core.chimera", package)
        monkeypatch.setitem(sys.modules, "anton.core.chimera.config", module)
    if not client:
        monkeypatch.setattr("anton.core.llm.client.LLMClient", _ClientWithoutChimera)
    elif ChimeraLLMClient is not _RealLLMClient:
        monkeypatch.setattr("anton.core.llm.client.LLMClient", ChimeraLLMClient)
    if not session:
        monkeypatch.setattr("anton.core.session.ChatSessionConfig", _session_config(chimera=False))
    elif ChimeraChatSessionConfig is not _RealChatSessionConfig:
        monkeypatch.setattr("anton.core.session.ChatSessionConfig", ChimeraChatSessionConfig)


def hide(monkeypatch) -> None:
    """Make the installed anton look like one without Chimera, as uv.lock pins."""
    install(monkeypatch, config=False, client=False, session=False)
