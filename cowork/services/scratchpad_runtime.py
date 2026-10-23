"""Scratchpad runtime pool — adapter for anton's `launch_artifact_backend`.

Ported from `cowork/server/anton_api/scratchpad_runtime.py`. We only
need the `WorkspaceScopedPool` adapter and the underlying `_pads`
registry — full /v1/scratchpad/* endpoints are not exposed by
cowork-server. The pool gives `launch_artifact_backend` access to a
slug-keyed Python venv at `<workspace>/.anton/scratchpad-venvs/<slug>/`
so artifact backends share dependencies provisioned by the agent
during artifact creation.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional

from cowork.code_sandbox import refuse_a_missing_policy

if TYPE_CHECKING:
    from anton.core.backends.sandbox import CellSandbox


MAX_PADS = int(os.environ.get("ANTON_SERVER_MAX_PADS", "5"))


# Keyed by resolved workspace, then by pad name. An artifact's slug names its
# pad, and a slug is unique only inside one project: two people's projects can
# each hold `sales-dashboard`. A registry keyed by name alone handed the second
# project the first one's runtime, so its requirements installed into, and its
# backend ran from, a venv inside the other project's folder. MAX_PADS caps
# each workspace on its own for the same reason.
_pads: dict[Path, dict[str, _CachedPad]] = {}  # workspace -> name -> pad
last_activity: float = time.time()


class _CachedPad(NamedTuple):
    """A registered pad and the code-sandbox policy it was built with
    (cowork.code_sandbox.build_cell_sandbox), None with the sandbox off. The
    policy binds when the pad starts, so a later caller asking for the same
    pad under another policy is refused rather than handed this one."""

    runtime: object  # LocalScratchpadRuntime
    cell_sandbox: CellSandbox | None


def _touch_activity() -> None:
    global last_activity
    last_activity = time.time()


def _resolve_workspace(workspace_path: Optional[str]) -> Path:
    return (
        Path(workspace_path).expanduser().resolve()
        if workspace_path
        else Path.cwd().resolve()
    )


def _resolve_coding(
    *,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
) -> tuple[str, str, str, str]:
    """Fill in any blank coding fields from cowork's UserSettings.

    UserSettings is the authoritative store and owns the dedicated gemini /
    openai-compatible key slots, so the key/base resolve through the same single-
    source helpers (provider_api_key_str / provider_base_url) the agent path
    uses. AntonSettings carries only the three legacy slots (anthropic / openai /
    minds), so reading keys from it would hand a dedicated-only gemini /
    openai-compatible user an empty key — the gap SailingSF flagged on #111.
    """
    from cowork.common.settings.user_settings import (
        UI_TYPE_TO_PROVIDER,
        get_user_settings,
        provider_api_key_str,
    )
    from cowork.services.providers import provider_base_url

    us = get_user_settings()
    # Normalize to the dash form up front so every return value (and the lookups
    # below) is consistent — anton's scratchpad_boot only understands the dash
    # spellings; the snake form would silently fall through to AnthropicProvider.
    # Resolved, not raw: the raw coding_model can be a wallet-locked pin the
    # resolution layer deliberately swaps for an affordable model (ENG-1632) —
    # reading it here would hand the scratchpad the exact model the agent path
    # refuses to run.
    provider = (coding_provider or us.resolved_coding_provider.ui_value).replace("_", "-")
    model = coding_model or us.resolved_coding_model or ""
    enum_provider = UI_TYPE_TO_PROVIDER.get(provider)

    # Resolve the API key from the correct dedicated slot (gemini/openai-
    # compatible fall back to the shared openai key). An explicit coding_api_key
    # always wins.
    if coding_api_key:
        api_key = coding_api_key
    elif enum_provider is not None:
        api_key = provider_api_key_str(us, enum_provider)
    else:
        api_key = ""

    # Derive the base URL deterministically per provider (see
    # providers.provider_base_url): openai/gemini never inherit the shared
    # openai_base_url slot, so a stale value left by another provider can't
    # misroute this key. An explicit coding_base_url always wins. Empty string
    # means "let anton's OpenAIProvider use its SDK default host".
    base_url = coding_base_url or provider_base_url(
        provider, openai_base_url=us.openai_base_url or "", minds_url=us.minds_url
    ) or ""

    # anton's scratchpad (scratchpad_boot.py) only understands "openai" /
    # "openai-compatible" → OpenAIProvider; every other string falls through to
    # AnthropicProvider. minds-cloud and gemini are OpenAI-compatible gateways,
    # so present them as "openai-compatible" (with their correct base above) —
    # otherwise the scratchpad would silently hit Anthropic with the wrong key.
    if provider in ("minds-cloud", "gemini"):
        provider = "openai-compatible"

    return provider, model, api_key, base_url


def _make_runtime(
    name: str,
    *,
    workspace: Path,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
    cell_sandbox: CellSandbox | None = None,
):
    refuse_a_missing_policy(cell_sandbox, site="a backend pool's scratchpad")
    from anton.core.backends.local import LocalScratchpadRuntime

    return LocalScratchpadRuntime(
        name,
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
        workspace_path=workspace,
        # Passed only when set, so with the sandbox off the call is today's.
        **({} if cell_sandbox is None else {"cell_sandbox": cell_sandbox}),
    )


def get(name: str, *, workspace_path: Optional[str] = None):
    _touch_activity()
    cached = _pads.get(_resolve_workspace(workspace_path), {}).get(name)
    return None if cached is None else cached.runtime


def get_or_create(
    name: str,
    *,
    workspace_path: Optional[str] = None,
    coding_provider: str = "",
    coding_model: str = "",
    coding_api_key: str = "",
    coding_base_url: str = "",
    cell_sandbox: CellSandbox | None = None,
):
    refuse_a_missing_policy(cell_sandbox, site="a backend pool's scratchpad")
    _touch_activity()
    workspace = _resolve_workspace(workspace_path)
    pads = _pads.setdefault(workspace, {})
    cached = pads.get(name)
    if cached is not None:
        if cached.cell_sandbox != cell_sandbox:
            raise RuntimeError(
                f"Scratchpad {name!r} in {workspace} runs under another code-sandbox "
                "policy than this caller's; close it before starting it under this one."
            )
        return cached.runtime
    if len(pads) >= MAX_PADS:
        raise RuntimeError(
            f"Maximum concurrent scratchpads ({MAX_PADS}) reached. "
            f"Close an existing pad first."
        )
    provider, model, api_key, base_url = _resolve_coding(
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
    )
    pad = _make_runtime(
        name,
        workspace=workspace,
        coding_provider=provider,
        coding_model=model,
        coding_api_key=api_key,
        coding_base_url=base_url,
        cell_sandbox=cell_sandbox,
    )
    pads[name] = _CachedPad(pad, cell_sandbox)
    return pad


def remove(name: str, *, workspace_path: Optional[str] = None) -> None:
    _pads.get(_resolve_workspace(workspace_path), {}).pop(name, None)


def list_pads(*, workspace_path: Optional[str] = None) -> list[str]:
    return list(_pads.get(_resolve_workspace(workspace_path), {}))


async def close_all() -> None:
    for pads in list(_pads.values()):
        for cached in list(pads.values()):
            try:
                await cached.runtime.close()
            except Exception:
                pass
    _pads.clear()


class WorkspaceScopedPool:
    """ScratchpadPoolLike adapter satisfying anton's launch_artifact_backend.

    Pins the workspace_path at construction so the helper sees a no-arg
    `venv_python(name)` / `get_or_create(name)` API. `venv_python`
    provisions the venv on demand via `LocalScratchpadRuntime._ensure_venv`
    — fast no-op when the deterministic disk path
    (`<workspace>/.anton/scratchpad-venvs/<name>/`) already exists,
    full `uv venv` / `python -m venv` creation when it doesn't. The
    runtime is registered in the module-level pool under this workspace,
    so subsequent `get_or_create` calls (e.g. for `install_packages`)
    reuse the same instance, and a same-named pad of another workspace
    is never handed out here.

    Note: anton's `_ensure_venv` is a private sync method that returns
    None and populates `_venv_python` on success. We touch both
    private members because anton doesn't expose a public equivalent;
    using `start()` would be heavier (spawns the scratchpad subprocess)
    and unnecessary — `launch_artifact_backend` only needs the path
    to the venv's python binary.

    `cell_sandbox` is the policy (cowork.code_sandbox.build_cell_sandbox)
    every pad of this pool runs under, None with the sandbox off. anton's
    `launch_artifact_backend` reads it off the pool and refuses a backend
    when it is set.
    """

    def __init__(self, workspace_path: str, *, cell_sandbox: CellSandbox | None = None):
        self._workspace_path = workspace_path
        self.cell_sandbox = cell_sandbox

    async def venv_python(self, name: str) -> Optional[str]:
        pad = self._pad(name)
        # Off-thread because `_ensure_venv` may shell out to `uv venv` /
        # `python -m venv` on first call (cold-start cost is seconds).
        await asyncio.to_thread(pad._ensure_venv)
        return pad._venv_python

    async def get_or_create(self, name: str):
        return self._pad(name)

    def _pad(self, name: str):
        return get_or_create(
            name, workspace_path=self._workspace_path, cell_sandbox=self.cell_sandbox
        )
