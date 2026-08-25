from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from anton.core.tools.skill_format import normalize_name

from cowork.coding.contracts import utc_now
from cowork.coding.guidance_items import GuidanceItemSpec, discover_guidance_items
from cowork.coding.project_models import CodeProject
from cowork.coding.project_store import CodeProjectStore
from cowork.coding.skill_models import (
    ProjectSkillSource,
    SkillLibraryDocument,
    SkillLibraryItem,
    SkillLibraryPage,
    SkillLibrarySource,
    TeamSkillSource,
)
from cowork.coding.skill_source_store import SkillSourceStore
from cowork.coding.workspace import GitRunner, WorkspaceError
from cowork.services.skills import BUILTIN_SKILLS_DIR, SkillService

_MAX_DOCUMENT_BYTES = 512_000
_MAX_DOCUMENT_FILES = 100


class SkillLibraryService:
    """Organisation catalogue of versioned, Git-backed agent guidance."""

    def __init__(
        self,
        root: Path,
        projects: CodeProjectStore,
        git: GitRunner | None = None,
        store: SkillSourceStore | None = None,
    ) -> None:
        self.root = root / "skill-library"
        self.caches = self.root / "sources"
        self.caches.mkdir(parents=True, exist_ok=True)
        self.projects = projects
        self.git = git or GitRunner()
        self.store = store or SkillSourceStore(root)

    def list(self, project_id: str | None = None) -> SkillLibraryPage:
        projects = self.projects.list()
        selected = self.projects.get(project_id) if project_id else None
        enabled_by_source = self._enabled_by_source(projects)
        sources: list[SkillLibrarySource] = []
        items: list[SkillLibraryItem] = []
        for source in self.store.list():
            enabled_projects = enabled_by_source.get(source.id, {})
            try:
                specs = discover_guidance_items(self._validated_cache(source))
            except WorkspaceError as exc:
                sources.append(
                    self._source_status(source, 0, len(enabled_projects), str(exc))
                )
                continue
            source_items = [
                SkillLibraryItem(
                    id=f"{source.id}:{spec.path}",
                    kind=spec.kind,
                    name=spec.name,
                    description=spec.description,
                    origin="team",
                    source_id=source.id,
                    source_name=source.name,
                    path=spec.path,
                    version=source.applied_revision,
                    enabled=(
                        selected is not None
                        and spec.path in enabled_projects.get(selected.id, set())
                    ),
                    enabled_project_ids=sorted(
                        project_id for project_id, paths in enabled_projects.items() if spec.path in paths
                    ),
                )
                for spec in specs
            ]
            items.extend(source_items)
            sources.append(self._source_status(source, len(source_items), len(enabled_projects)))
        return SkillLibraryPage(sources=sources, items=items)

    def catalog(self, personal: SkillService, project_id: str | None = None) -> SkillLibraryPage:
        page = self.list(project_id)
        selected = self.projects.get(project_id) if project_id else None
        selected_keys = {selected.id, selected.name} if selected else set()
        personal.ensure_builtin_skills()
        builtin_names = {
            path.name for path in BUILTIN_SKILLS_DIR.iterdir() if path.is_dir()
        } if BUILTIN_SKILLS_DIR.is_dir() else set()
        for skill in personal.list_skills():
            origin = "built_in" if skill.name in builtin_names else "personal"
            page.items.append(
                SkillLibraryItem(
                    id=f"personal:{skill.name}",
                    kind="skill",
                    name=skill.display_name,
                    description=skill.description,
                    origin=origin,
                    source_name="MindsHub" if origin == "built_in" else "Yours",
                    path=skill.name,
                    version=skill.updated_at.isoformat() if skill.updated_at else None,
                    enabled=skill.enabled and (not skill.projects or bool(selected_keys.intersection(skill.projects))),
                    enabled_project_ids=skill.projects,
                )
            )
        return page

    def document(
        self,
        personal: SkillService,
        item_id: str,
        selected_path: str | None = None,
    ) -> SkillLibraryDocument:
        """Return one library item's readable source without exposing arbitrary files."""
        page = self.catalog(personal)
        item = next((candidate for candidate in page.items if candidate.id == item_id), None)
        if item is None:
            raise KeyError("Skill library item not found")

        if item.origin == "team":
            if not item.source_id:
                raise WorkspaceError("This team item has no source")
            _, cache = self.cache_for(item.source_id)
            source_file = self._contained_file(cache, item.path)
        else:
            source_file = self._contained_file(personal.root, f"{item.path}/SKILL.md")

        if item.kind == "skill":
            root = source_file.parent
            files = self._readable_files(root)
            default_path = "SKILL.md"
        else:
            root = source_file.parent
            files = [source_file.name]
            default_path = source_file.name
        if not files:
            raise WorkspaceError("This skill has no readable files")

        path = selected_path or default_path
        if path not in files:
            raise WorkspaceError("That file is not part of this skill")
        content = self._read_document(self._contained_file(root, path))
        return SkillLibraryDocument(
            item=item,
            files=files,
            selected_path=path,
            content=content,
        )

    def add(self, repository: str, branch: str = "main", name: str | None = None) -> SkillLibrarySource:
        repository = repository.strip()
        if not repository:
            raise ValueError("Choose a Git repository or local Git folder")
        normalized_branch = self._branch(branch)
        repository_key = repository.rstrip("/\\").casefold()
        if any(
            item.repository.rstrip("/\\").casefold() == repository_key
            and item.branch.casefold() == normalized_branch.casefold()
            for item in self.store.list()
        ):
            raise WorkspaceError("That repository branch is already in the Skills Library")
        source_id = str(uuid.uuid4())
        cache = self._cache(source_id)
        staging = self.caches / f".{source_id}.next"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            self.git.run(
                self.caches,
                "clone",
                "--single-branch",
                "--branch",
                normalized_branch,
                "--",
                repository,
                str(staging),
            )
            revision = self._revision(staging, "HEAD")
            specs = discover_guidance_items(staging)
            if not specs:
                raise WorkspaceError(
                    "No skills, instructions, or workflows were found in that repository"
                )
            staging.rename(cache)
            source = TeamSkillSource(
                id=source_id,
                name=self._name(name or self._repository_name(repository)),
                repository=repository,
                branch=normalized_branch,
                applied_revision=revision,
                available_revision=revision,
                cache_path=str(cache),
            )
            try:
                self.store.create(source)
            except Exception:
                shutil.rmtree(cache, ignore_errors=True)
                raise
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return self._source_status(source, len(specs), 0)

    def refresh(self, source_id: str) -> SkillLibrarySource:
        source = self.store.get(source_id)
        cache = self._validated_cache(source)
        result = self.git.run(cache, "fetch", "--quiet", "origin", source.branch, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Skill source update check failed").strip()
            return self._source_status(source, len(discover_guidance_items(cache)), self._project_count(source.id), detail[:2_000])
        available = self._revision(cache, "FETCH_HEAD")
        source = self.store.update(source.id, lambda current: self._record_refresh(current, available))
        return self._source_status(source, len(discover_guidance_items(cache)), self._project_count(source.id))

    def apply_update(self, source_id: str) -> SkillLibrarySource:
        source = self.store.get(source_id)
        cache = self._validated_cache(source)
        available = source.available_revision
        if not available:
            return self.refresh(source.id)
        previous = source.applied_revision
        self.git.run(cache, "switch", "--detach", available)
        try:
            specs = discover_guidance_items(cache)
            if not specs:
                raise WorkspaceError("The update contains no skills, instructions, or workflows")
            for project in self.projects.list():
                if self._binding(project, source.id):
                    self._validate_project_bindings(
                        project,
                        project.skill_sources,
                        candidate_source_id=source.id,
                        candidate_specs=specs,
                    )
            source = self.store.update(source.id, lambda current: self._record_apply(current, available))
        except Exception as exc:
            rollback = self.git.run(cache, "switch", "--detach", previous, check=False)
            if rollback.returncode != 0:
                raise WorkspaceError(
                    "The skill source metadata could not be saved and its cache could not be restored; reconnect the source"
                ) from exc
            raise
        return self._source_status(
            source,
            len(discover_guidance_items(cache)),
            self._project_count(source.id),
        )

    def remove(self, source_id: str) -> None:
        source = self.store.get(source_id)
        projects = [project.name for project in self.projects.list() if self._binding(project, source_id)]
        if projects:
            raise WorkspaceError(
                f"Remove this source from {', '.join(projects[:3])} before deleting it from the library"
            )
        self.store.delete(source_id)
        shutil.rmtree(self._cache(source.id), ignore_errors=True)

    def set_project_items(self, project_id: str, source_id: str, enabled_paths: list[str]) -> SkillLibraryPage:
        project = self.projects.get(project_id)
        source = self.store.get(source_id)
        available = {item.path for item in discover_guidance_items(self._validated_cache(source))}
        selected = set(enabled_paths)
        unknown = selected - available
        if unknown:
            raise WorkspaceError("One or more selected library items are no longer available")

        def update(current: CodeProject) -> None:
            retained = [binding for binding in current.skill_sources if binding.source_id != source_id]
            if selected:
                retained.append(ProjectSkillSource(source_id=source_id, enabled_paths=sorted(selected)))
            self._validate_project_bindings(current, retained)
            current.skill_sources = retained

        self.projects.update(project.id, update)
        return self.list(project.id)

    def cache_for(self, source_id: str) -> tuple[TeamSkillSource, Path]:
        source = self.store.get(source_id)
        return source, self._validated_cache(source)

    def _source_status(
        self,
        source: TeamSkillSource,
        item_count: int,
        project_count: int,
        error: str | None = None,
    ) -> SkillLibrarySource:
        diff = ""
        if not error and source.applied_revision != source.available_revision:
            cache = self._validated_cache(source)
            diff = self.git.run(
                cache,
                "diff",
                "--stat",
                "--no-ext-diff",
                source.applied_revision,
                source.available_revision,
                check=False,
            ).stdout[:32_000]
        return SkillLibrarySource(
            id=source.id,
            name=source.name,
            repository=source.repository,
            branch=source.branch,
            current_revision=source.applied_revision,
            available_revision=source.available_revision,
            update_available=source.applied_revision != source.available_revision,
            last_checked_at=source.last_checked_at,
            item_count=item_count,
            enabled_project_count=project_count,
            diff=diff,
            error=error,
        )

    @staticmethod
    def _record_refresh(source: TeamSkillSource, revision: str) -> None:
        source.available_revision = revision
        source.last_checked_at = utc_now()

    @staticmethod
    def _record_apply(source: TeamSkillSource, revision: str) -> None:
        source.applied_revision = revision
        source.available_revision = revision
        source.last_checked_at = utc_now()

    @staticmethod
    def _binding(project: CodeProject, source_id: str) -> ProjectSkillSource | None:
        return next((item for item in project.skill_sources if item.source_id == source_id), None)

    @staticmethod
    def _enabled_by_source(projects: list[CodeProject]) -> dict[str, dict[str, set[str]]]:
        enabled: dict[str, dict[str, set[str]]] = {}
        for project in projects:
            for binding in project.skill_sources:
                enabled.setdefault(binding.source_id, {})[project.id] = set(binding.enabled_paths)
        return enabled

    def _project_count(self, source_id: str) -> int:
        return sum(bool(self._binding(project, source_id)) for project in self.projects.list())

    def _validate_project_bindings(
        self,
        project: CodeProject,
        bindings: list[ProjectSkillSource],
        *,
        candidate_source_id: str | None = None,
        candidate_specs: list[GuidanceItemSpec] | None = None,
    ) -> None:
        """Keep project skill selections resolvable across sources and updates."""
        names: dict[str, str] = {}
        for binding in bindings:
            source = self.store.get(binding.source_id)
            specs = (
                candidate_specs
                if binding.source_id == candidate_source_id and candidate_specs is not None
                else discover_guidance_items(self._validated_cache(source))
            )
            by_path = {spec.path: spec for spec in specs}
            missing = [path for path in binding.enabled_paths if path not in by_path]
            if missing:
                raise WorkspaceError(
                    f"The update removes an item used by {project.name}; remove it from the project first"
                )
            for path in binding.enabled_paths:
                spec = by_path[path]
                if spec.kind != "skill":
                    continue
                slug = normalize_name(spec.name)
                existing = names.get(slug)
                if existing:
                    raise WorkspaceError(
                        f"{project.name} already includes a team skill named {spec.name!r} from {existing}"
                    )
                names[slug] = source.name

    def _validated_cache(self, source: TeamSkillSource) -> Path:
        expected = self._cache(source.id).resolve()
        cache = Path(source.cache_path).resolve()
        if cache != expected or not cache.is_dir():
            raise WorkspaceError("The managed skill source cache is unavailable; reconnect the source")
        return cache

    @staticmethod
    def _contained_file(root: Path, relative: str) -> Path:
        boundary = root.resolve()
        candidate = boundary / relative
        if candidate.is_symlink():
            raise WorkspaceError("The requested skill file is unavailable")
        path = candidate.resolve()
        if not path.is_relative_to(boundary) or not path.is_file():
            raise WorkspaceError("The requested skill file is unavailable")
        return path

    @classmethod
    def _readable_files(cls, root: Path) -> list[str]:
        boundary = root.resolve()
        files: list[str] = []
        for path in sorted(root.rglob("*")):
            if len(files) >= _MAX_DOCUMENT_FILES:
                break
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part.startswith(".") for part in relative.parts) or relative.name == "stats.json":
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(boundary) or path.stat().st_size > _MAX_DOCUMENT_BYTES:
                continue
            try:
                sample = path.read_bytes()
                if b"\x00" in sample:
                    continue
                sample.decode("utf-8")
            except (OSError, UnicodeError):
                continue
            files.append(relative.as_posix())
        if "SKILL.md" in files:
            files.remove("SKILL.md")
            files.insert(0, "SKILL.md")
        return files

    @staticmethod
    def _read_document(path: Path) -> str:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise WorkspaceError("The requested skill file could not be read") from exc
        if len(data) > _MAX_DOCUMENT_BYTES:
            raise WorkspaceError("That skill file is too large to preview")
        if b"\x00" in data:
            raise WorkspaceError("That skill file is not text")
        try:
            return data.decode("utf-8")
        except UnicodeError as exc:
            raise WorkspaceError("That skill file is not UTF-8 text") from exc

    def _cache(self, source_id: str) -> Path:
        return self.caches / source_id

    def _revision(self, cache: Path, ref: str) -> str:
        return self.git.run(cache, "rev-parse", ref).stdout.strip()

    def _branch(self, value: str) -> str:
        branch = value.strip()
        if self.git.run(self.caches, "check-ref-format", "--branch", branch, check=False).returncode != 0:
            raise WorkspaceError("Enter a valid branch name")
        return branch

    @staticmethod
    def _name(value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Name this skill source")
        return normalized[:120]

    @staticmethod
    def _repository_name(repository: str) -> str:
        normalized = repository.rstrip("/\\").removesuffix(".git")
        return normalized.replace("\\", "/").rsplit("/", 1)[-1] or "Team skills"
