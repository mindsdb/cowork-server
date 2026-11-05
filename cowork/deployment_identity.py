"""Read the deployment's product identity before the server accepts turns.

The operator mounts the selected branding pack separately from writable user
storage. create_app validates it once; every turn uses that immutable snapshot.
No member setting or request value participates in this identity.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import os
import stat
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from cowork.common.paths import O_NOFOLLOW, O_NONBLOCK, dir_open, opened_subdir_nofollow, read_bounded
from cowork.common.settings.app_settings import AppSettings, get_app_settings

if TYPE_CHECKING:
    from anton.core.llm.identity import DeploymentIdentity
    from anton.core.memory.base import LearnedMemoryScope


class _BrandPack(BaseModel):
    """Only the agent's name is read; the frontend owns the rest of the pack."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    name: StrictStr


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Match the branding pack loader's refusal of duplicate JSON keys."""
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate branding pack key")
        document[key] = value
    return document


def _identity_type() -> type[DeploymentIdentity]:
    """Refuse an explicit opt-in that the installed Anton would ignore."""
    try:
        from anton.core.llm.identity import DeploymentIdentity
        from anton.core.memory.cortex import Cortex
        from anton.core.memory.hippocampus import Hippocampus
        from anton.core.session import ChatSessionConfig
    except ImportError:
        raise RuntimeError(
            "COWORK_DEPLOYMENT_KIND=self-hosted needs an Anton with DeploymentIdentity "
            "and learned-memory scope support. Upgrade the pinned Anton dependency."
        ) from None
    fields = {field.name for field in dataclasses.fields(ChatSessionConfig)}
    if not {"deployment_identity", "learned_memory_scope"}.issubset(fields):
        raise RuntimeError(
            "COWORK_DEPLOYMENT_KIND=self-hosted needs an Anton ChatSessionConfig with "
            "deployment_identity and learned_memory_scope. Upgrade the pinned Anton dependency."
        )
    if (
        any("learned_memory_scope" not in inspect.signature(method).parameters for method in (
            Cortex.build_memory_context, Cortex.get_scratchpad_context,
        ))
        or not callable(getattr(Hippocampus, "memory_presence", None))
    ):
        raise RuntimeError(
            "COWORK_DEPLOYMENT_KIND=self-hosted needs Anton's scoped memory context "
            "and stored-memory presence support. Upgrade the pinned Anton dependency."
        )
    return DeploymentIdentity


def initialize_deployment_identity(settings: AppSettings, *, remote_turns: bool) -> None:
    """Validate the opt-in and freeze its pack name for this settings instance."""
    if settings.deployment_kind is None:
        return
    if settings.tenancy_mode == "org" or remote_turns:
        raise RuntimeError(
            "COWORK_DEPLOYMENT_KIND=self-hosted requires local, in-process turns: "
            "remote workers do not receive deployment identity."
        )
    if settings._deployment_identity_initialized:
        return
    identity_type = _identity_type()
    path = settings.deployment_brand_file
    if path is None or not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(
            "COWORK_DEPLOYMENT_BRAND_FILE must name an absolute path to the "
            "read-only branding pack.json when COWORK_DEPLOYMENT_KIND=self-hosted."
        )
    try:
        # Anchor at the filesystem root and refuse links in every component.
        # O_NONBLOCK prevents a mistaken FIFO mount from hanging startup.
        with opened_subdir_nofollow(path.anchor, *path.parent.parts[1:]) as parent:
            fd = dir_open(parent, path.name, os.O_RDONLY | O_NONBLOCK | O_NOFOLLOW)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("not a regular file")
                # Bound this snapshot to the opened file's size, even if the
                # operator appends to it while startup reads it.
                raw = b"".join(read_bounded(fd, info.st_size))
            finally:
                os.close(fd)
    except (OSError, ValueError):
        raise RuntimeError(
            f"COWORK_DEPLOYMENT_BRAND_FILE {path} must be a readable regular file, "
            "with no symlinks in its path."
        ) from None
    try:
        # A self-hosted deployment's pack loader also uses json.loads on
        # bytes, preserving BOM and UTF-16/32 pack support.
        pack = _BrandPack.model_validate(json.loads(raw, object_pairs_hook=_unique_object))
        identity = identity_type(kind="self-hosted", product_name=pack.name)
    except (ValidationError, ValueError, TypeError):
        # Do not include a malformed file's contents in a startup traceback.
        raise RuntimeError(
            f"COWORK_DEPLOYMENT_BRAND_FILE {path} must be a JSON object with a valid "
            "name: 1 to 64 characters, without outer whitespace, control or format characters."
        ) from None
    settings._deployment_identity = identity
    settings._deployment_identity_initialized = True


def identity_for_turn() -> DeploymentIdentity | None:
    """Return the startup snapshot, never reread a file written during a turn."""
    settings = get_app_settings()
    if settings.deployment_kind is None:
        return None
    if not settings._deployment_identity_initialized:
        raise RuntimeError("Self-hosted deployment identity is not initialized; create the server app before running turns.")
    return settings._deployment_identity


def learned_memory_scope(*, off: bool, learns: bool, shared: bool) -> LearnedMemoryScope:
    """Name the existing memory policy without inspecting disabled stores."""
    if off:
        return "off"
    if not learns:
        return "unavailable"
    return "shared" if shared else "personal"
