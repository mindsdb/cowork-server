"""Where a turn's agent works with files, so that a path its code names can be
found in the conversation's project on this server's disk."""

from __future__ import annotations

import ntpath
import os
import posixpath
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath
from uuid import UUID

from cowork.common.paths import safe_join
from cowork.services.artifact_roots import CONVERSATIONS_DIRNAME

# The directory a cloud pod's scratchpad runs in: anton's default workspace
# mount (anton.cloud_turn.session.DEFAULT_CLOUD_WORKSPACE_PATH), which holds
# the conversation's folder in the project, `conversations/<id>/` (see
# TurnJob.project_id). Attachments are staged under it at
# `attachments/<file_id>/<name>` (FileService.stage_conversation_attachments).
CLOUD_WORKING_DIR = PurePosixPath("/workspace")


@dataclass(frozen=True)
class AgentFiles:
    """The project on this server's disk and the agent's view of it.

    ``working_dir`` is the directory the agent's code runs in, named as the
    agent names it, and ``working_dir_in_project`` is where that directory
    sits inside the project (empty for the project root).
    """

    project_root: Path
    working_dir: PurePath
    working_dir_in_project: PurePosixPath = field(default_factory=PurePosixPath)

    @classmethod
    def on_this_server(cls, project_root: str | Path) -> AgentFiles:
        """A turn this server runs: its scratchpad works in the project root
        (AntonHarness._build_chat_session tells the agent so)."""
        root = Path(project_root)
        return cls(project_root=root, working_dir=root)

    @classmethod
    def in_cloud_pod(cls, project_root: str | Path, conversation_id: UUID | str) -> AgentFiles:
        """A turn a cloud pod runs, in the conversation's folder."""
        return cls(
            project_root=Path(project_root),
            working_dir=CLOUD_WORKING_DIR,
            working_dir_in_project=PurePosixPath(CONVERSATIONS_DIRNAME, str(conversation_id)),
        )

    def project_path(self, literal: str) -> str | None:
        """The project-relative POSIX path of the file ``literal`` names, or
        None when it names a place outside the agent's working directory, a
        link that leads out of it, or no file now."""
        parts = self._project_parts(literal)
        if parts is None:
            return None
        try:
            # Resolves links and refuses a result outside the working directory:
            # a pod sees only its conversation's folder, so a link into another
            # member's folder names a file the pod could not read.
            working_dir = safe_join(self.project_root, *self.working_dir_in_project.parts)
            target = safe_join(working_dir, *parts)
            if not target.is_file():
                return None
        except (OSError, ValueError):
            return None
        return target.relative_to(os.path.realpath(self.project_root)).as_posix()

    def _project_parts(self, literal: str) -> tuple[str, ...] | None:
        """The literal's place in the working directory as path parts, read the
        way the agent reads it, with ``..`` folded away before anything is
        mapped. An absolute server path counts only for a turn this server
        runs: a pod can't see the server's disk."""
        if not literal:
            return None
        working_dir = _normalized(self.working_dir)
        # Joining an absolute literal replaces the working directory.
        seen = _normalized(working_dir / literal)
        if seen.is_relative_to(working_dir):
            return seen.relative_to(working_dir).parts
        on_server = Path(literal)
        if not self.working_dir_in_project.parts and on_server.is_absolute():
            normalized = Path(os.path.normpath(on_server))
            if normalized.is_relative_to(self.project_root):
                return normalized.relative_to(self.project_root).parts
        return None


def _normalized(path: PurePath) -> PurePath:
    """``path`` with ``.`` and ``..`` folded away under its own flavor's rules:
    POSIX for a pod, and the server's own for a turn it runs."""
    rules = posixpath if isinstance(path, PurePosixPath) else ntpath
    return type(path)(rules.normpath(path))
