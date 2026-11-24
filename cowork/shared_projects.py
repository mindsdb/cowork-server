"""Operator-owned grants for read-only project files, outside agent-writable trees."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal, Self
from uuid import UUID

from pydantic import field_validator, model_validator

from cowork.common.paths import O_NOFOLLOW, open_fd
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import TenantScope
from cowork.managed import MAX_REGISTRY_BYTES, SharedEntry, SharedRegistry, reads_shared_registry
from cowork.principal import personal_partition_id

logger = logging.getLogger(__name__)


class SharedProject(SharedEntry):
    organization_id: UUID
    project_id: UUID
    audience: Literal["everyone"] | tuple[UUID, ...]

    @model_validator(mode="after")
    def _owner_partition(self) -> Self:
        if self.partition_id != personal_partition_id(str(self.organization_id), self.user_id):
            raise ValueError("owner partition does not match organization and user")
        return self

    @field_validator("audience")
    @classmethod
    def _audience(cls, value: Literal["everyone"] | tuple[UUID, ...]) -> Literal["everyone"] | tuple[UUID, ...]:
        if value != "everyone" and (not value or len(set(value)) != len(value)):
            raise ValueError("audience must name at least one unique user, or everyone")
        return value

    def readable_by(self, scope: TenantScope) -> bool:
        if not reads_shared_registry(scope, resource=PersonalResource.PROJECTS) or not scope.user_id:
            return False
        # A subject in another deployment is not part of 'everyone' here.
        try:
            partition = personal_partition_id(str(self.organization_id), scope.user_id)
        except ValueError:
            return False
        return (
            scope.org_id == partition
            and scope.org_id != self.partition_id
            and (self.audience == "everyone" or UUID(scope.user_id) in self.audience)
        )


class SharedProjects(SharedRegistry[SharedProject]):
    entry_noun: ClassVar[str] = "project"
    entry_fields: ClassVar[str] = "organization_id, user_id, partition_id, project_id, audience"

    @model_validator(mode="after")
    def _unique_projects(self) -> Self:
        if len({entry.project_id for entry in self.root}) != len(self.root):
            raise ValueError("a project may have only one grant")
        return self


def shared_projects_file() -> Path:
    return Path(get_app_settings().storage.shared_root) / "shared" / "projects.json"


def read_shared_projects() -> SharedProjects:
    return SharedProjects.read_live(shared_projects_file(), logger=logger)


def change_shared_projects(change: Callable[[SharedProjects], SharedProjects]) -> SharedProjects:
    """Serialize operator updates; never overwrite a malformed or unreadable registry."""
    import fcntl  # Only trusted-identity deployments write grants, never Windows desktop.

    path = shared_projects_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = open_fd(path.with_suffix(".json.lock"), os.O_RDWR | os.O_CREAT | O_NOFOLLOW, 0o644)
    try:
        deadline = time.monotonic() + 5.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Project sharing is busy; try again.") from None
                time.sleep(0.05)
        current = SharedProjects.read(path)
        updated = change(current)
        if updated != current:
            if len(updated.dump_json().encode("utf-8")) > MAX_REGISTRY_BYTES:
                raise ValueError("Project registry would exceed its size limit")
            updated.write(path)
        return updated
    finally:
        os.close(fd)
