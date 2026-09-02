from __future__ import annotations

import re
import shutil
from pathlib import Path

from cowork.coding.contracts import utc_now
from cowork.coding.project_models import (
    CodeProject,
    PlaybookItem,
    PlaybookReference,
    PlaybookStatus,
)
from cowork.coding.project_store import CodeProjectStore
from cowork.coding.workspace import GitRunner, WorkspaceError

_INSTRUCTION_FILES = {"agents.md", "claude.md", "instructions.md"}
_WORKFLOW_SUFFIXES = {".yml", ".yaml"}
_FRONTMATTER_DESCRIPTION = re.compile(r"^description:\s*['\"]?(.*?)['\"]?\s*$", re.MULTILINE)


class PlaybookService:
    """Cache, inspect, and normalize a project's Git-backed team playbook."""

    def __init__(self, root: Path, projects: CodeProjectStore, git: GitRunner | None = None) -> None:
        self.root = root / "playbooks"
        self.root.mkdir(parents=True, exist_ok=True)
        self.projects = projects
        self.git = git or GitRunner()

    def configure(self, project_id: str, repository: str, branch: str = "main") -> PlaybookStatus:
        branch = self._branch(branch)
        project = self.projects.get(project_id)
        cache = self._cache(project.id)
        staging = self.root / f".{project.id}.next"
        backup = self.root / f".{project.id}.previous"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        installed = False
        try:
            self.git.run(
                self.root,
                "clone",
                "--single-branch",
                "--branch",
                branch,
                "--",
                repository,
                str(staging),
            )
            revision = self._revision(staging, "HEAD")
            if cache.exists():
                cache.rename(backup)
            staging.rename(cache)
            installed = True
            reference = PlaybookReference(
                repository=repository,
                branch=branch,
                applied_revision=revision,
                available_revision=revision,
                cache_path=str(cache),
                last_checked_at=utc_now(),
            )
            self.projects.update(project.id, lambda current: setattr(current, "playbook", reference))
        except Exception:
            if installed:
                shutil.rmtree(cache, ignore_errors=True)
            if backup.exists():
                backup.rename(cache)
            shutil.rmtree(staging, ignore_errors=True)
            raise
        else:
            shutil.rmtree(backup, ignore_errors=True)
        return self.status(project.id)

    def refresh(self, project_id: str) -> PlaybookStatus:
        project, reference, cache = self._configured(project_id)
        result = self.git.run(cache, "fetch", "--quiet", "origin", reference.branch, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Playbook update check failed").strip()
            return self._status(project, error=detail[:2_000])
        available = self._revision(cache, "FETCH_HEAD")

        def record(current: CodeProject) -> None:
            if current.playbook is None:
                raise WorkspaceError("The team playbook was removed while it was refreshing")
            current.playbook.available_revision = available
            current.playbook.last_checked_at = utc_now()

        self.projects.update(project.id, record)
        return self.status(project.id)

    def apply_update(self, project_id: str) -> PlaybookStatus:
        project, reference, cache = self._configured(project_id)
        available = reference.available_revision
        if not available:
            return self.refresh(project_id)
        self.git.run(cache, "switch", "--detach", available)

        def record(current: CodeProject) -> None:
            if current.playbook is None:
                raise WorkspaceError("The team playbook was removed while it was updating")
            current.playbook.applied_revision = available
            current.playbook.available_revision = available
            current.playbook.last_checked_at = utc_now()

        try:
            self.projects.update(project.id, record)
        except Exception as exc:
            if reference.applied_revision:
                rollback = self.git.run(cache, "switch", "--detach", reference.applied_revision, check=False)
                if rollback.returncode != 0:
                    raise WorkspaceError(
                        "The playbook metadata could not be saved and its cache could not be restored; reconnect the playbook"
                    ) from exc
            raise
        return self.status(project.id)

    def status(self, project_id: str) -> PlaybookStatus:
        project = self.projects.get(project_id)
        if project.playbook is None:
            return PlaybookStatus(configured=False)
        return self._status(project)

    def remove(self, project_id: str) -> None:
        """Detach a playbook without touching its source repository."""
        project = self.projects.get(project_id)
        self.projects.update(project.id, lambda current: setattr(current, "playbook", None))
        self.cleanup(project.id)

    def cleanup(self, project_id: str) -> None:
        shutil.rmtree(self._cache(project_id), ignore_errors=True)

    def guidance(self, project_id: str) -> tuple[str, str]:
        """Return adapter-neutral guidance plus a concise UI summary."""
        project, _, cache = self._configured(project_id)
        items = self._items(project, cache)
        sections = [
            "# MindsHub Code Project guidance",
            "Instruction precedence (highest first): task, folder/repository, project playbook, application defaults.",
            "Apply guidance only within its stated scope. Preserve user instructions when sources disagree.",
        ]
        for item in items:
            if not item.enabled or item.kind not in {"instructions", "skill"}:
                continue
            path = cache / item.path
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            remaining = 120_000 - sum(len(section) for section in sections)
            if remaining <= 0:
                break
            sections.append(f"\n## {item.kind.title()}: {item.name}\nSource: {item.path}\n{content[:remaining]}")
        active_count = sum(item.enabled for item in items)
        summary = f"{project.name} playbook · {active_count} active item{'s' if active_count != 1 else ''}"
        return "\n".join(sections), summary

    def set_enabled(self, project_id: str, enabled_paths: list[str]) -> PlaybookStatus:
        project, _, cache = self._configured(project_id)
        detected = {item.path for item in self._discover(cache)}
        enabled = set(enabled_paths)
        unknown = enabled - detected
        if unknown:
            raise WorkspaceError("One or more selected playbook items are no longer available")

        def update(current: CodeProject) -> None:
            if current.playbook is None:
                raise WorkspaceError("The team playbook was removed while it was updating")
            current.playbook.disabled_items = sorted(detected - enabled)

        self.projects.update(project.id, update)
        return self.status(project.id)

    def _status(self, project: CodeProject, error: str | None = None) -> PlaybookStatus:
        reference = project.playbook
        if reference is None:
            return PlaybookStatus(configured=False)
        cache = Path(reference.cache_path or self._cache(project.id))
        current = reference.applied_revision
        available = reference.available_revision
        diff = ""
        if current and available and current != available and cache.is_dir():
            diff = self.git.run(
                cache,
                "diff",
                "--stat",
                "--patch",
                "--no-ext-diff",
                current,
                available,
                check=False,
            ).stdout[:128_000]
        return PlaybookStatus(
            configured=True,
            current_revision=current,
            available_revision=available,
            update_available=bool(current and available and current != available),
            items=self._items(project, cache) if cache.is_dir() else [],
            diff=diff,
            error=error,
        )

    def _configured(self, project_id: str) -> tuple[CodeProject, PlaybookReference, Path]:
        project = self.projects.get(project_id)
        if project.playbook is None:
            raise WorkspaceError("This Code Project does not have a team playbook")
        cache = Path(project.playbook.cache_path or self._cache(project.id)).resolve()
        if cache != self._cache(project.id).resolve() or not cache.is_dir():
            raise WorkspaceError("The managed playbook cache is unavailable; reconnect the playbook")
        return project, project.playbook, cache

    def _discover(self, cache: Path) -> list[PlaybookItem]:
        items: list[PlaybookItem] = []
        for path in sorted(cache.rglob("*")):
            # Playbook repositories are shared input. Do not follow a
            # checked-in symlink and accidentally pull a local file outside
            # the managed cache into the agent's instructions.
            if path.is_symlink() or not path.is_file() or ".git" in path.parts:
                continue
            relative = path.relative_to(cache).as_posix()
            lower_name = path.name.casefold()
            if lower_name == "skill.md":
                description = ""
                try:
                    match = _FRONTMATTER_DESCRIPTION.search(path.read_text(encoding="utf-8")[:8_000])
                    description = match.group(1).strip() if match else ""
                except (OSError, UnicodeError):
                    pass
                items.append(PlaybookItem(kind="skill", name=path.parent.name, path=relative, description=description))
            elif lower_name in _INSTRUCTION_FILES:
                items.append(PlaybookItem(kind="instructions", name=path.name, path=relative))
            elif ".github" in path.parts and path.suffix.casefold() in _WORKFLOW_SUFFIXES:
                items.append(PlaybookItem(kind="workflow", name=path.stem, path=relative))
        return items[:250]

    def _items(self, project: CodeProject, cache: Path) -> list[PlaybookItem]:
        disabled = set(project.playbook.disabled_items if project.playbook else [])
        return [item.model_copy(update={"enabled": item.path not in disabled}) for item in self._discover(cache)]

    def _cache(self, project_id: str) -> Path:
        return self.root / project_id

    def _revision(self, cache: Path, ref: str) -> str:
        return self.git.run(cache, "rev-parse", ref).stdout.strip()

    def _branch(self, value: str) -> str:
        branch = value.strip()
        if self.git.run(self.root, "check-ref-format", "--branch", branch, check=False).returncode != 0:
            raise WorkspaceError("Enter a valid playbook branch name")
        return branch
