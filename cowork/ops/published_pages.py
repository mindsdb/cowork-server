"""Operator-only published pages: list every page the deployment's publisher
serves, take one down, or give a page with no owner to the person who
published it.

A page published before Cowork recorded owners, on a deployment whose projects
were shared, has no owner: the one-time import refuses shared history
(published_page_owner_backfill). Cowork then never sends its id to the
publisher again, so nobody can update or take it down from Cowork. These
actions are how the operator finds such a page, removes it, or hands it back
to the person the recording proxy's audit lines name as its publisher.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session, select

from cowork.common.minds_tls import minds_ssl_verify
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import get_user_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.models.published_page import PublishedPageOwner
from cowork.ops import (
    OperatorCommandError, Person, add_person_arguments, person_from_arguments,
    personal_scope, run_operator_command,
)
from cowork.services.projects import ProjectService
from cowork.services.publish import _resolve_publish_endpoint, _write_published_map
from cowork.services.published_page_owner_backfill import _published_pages
from cowork.services.published_page_ownership import (
    find_owner, protects_publisher, publish_record_lock_key, publisher_key,
)
from cowork.services.shared_resources import _DatabaseResourceLocks


class _ListedPage(BaseModel):
    """One entry of the publisher's GET /list, by the fields this command reads."""

    model_config = ConfigDict(extra="ignore")

    report_id: str = Field(min_length=1, max_length=255)
    view_url: str = Field(min_length=1)
    title: str = ""
    last_modified: str = ""


class PublishedPage(BaseModel):
    """A page the publisher serves, its owner on record, and the artifact
    folders whose publish record still names it."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    view_url: str
    title: str
    last_modified: str
    owner_user_id: str | None
    artifact_paths: tuple[str, ...]


class _Publisher(BaseModel):
    """The deployment's own publisher: where it is, its key in the owner
    table, and the token every call sends."""

    model_config = ConfigDict(frozen=True)

    url: str
    key: str
    api_key: str


def _publisher() -> _Publisher:
    settings = get_app_settings()
    if settings.tenancy_mode == "org" or not settings.trust_identity_headers:
        raise OperatorCommandError(
            "Published page owners apply only to a local deployment that trusts identity headers."
        )
    try:
        url, api_key = _resolve_publish_endpoint(get_user_settings())
    except Exception:
        raise OperatorCommandError("The deployment names no publisher.") from None
    if not protects_publisher(url):
        raise OperatorCommandError("This deployment publishes through MindsHub, which keeps its own owners.")
    if not api_key:
        raise OperatorCommandError("The deployment has no publish token, so the publisher can't be asked.")
    return _Publisher(url=url, key=publisher_key(url), api_key=api_key)


def _listed(publisher: _Publisher) -> list[_ListedPage]:
    from anton.publisher import list_published

    try:
        values = list_published(api_key=publisher.api_key, publish_url=publisher.url, ssl_verify=minds_ssl_verify())
        return [_ListedPage.model_validate(value) for value in values]
    except Exception:
        # An HTTP error can hold the token or the response body: name only the failure.
        raise OperatorCommandError("The publisher's page list could not be read.") from None


def _folder_entries(session: Session) -> dict[str, list[str]]:
    """Each page id that a project's publish records name, with the artifact
    each record names it for."""
    entries: dict[str, list[str]] = {}
    for project in session.exec(select(Project)).all():
        for report_id, artifact_path in _published_pages(Path(project.path)):
            entries.setdefault(report_id, []).append(artifact_path)
    return entries


def _pages(session: Session, publisher: _Publisher) -> list[PublishedPage]:
    entries = _folder_entries(session)
    pages = []
    for listed in _listed(publisher):
        owner = find_owner(session, publisher.key, listed.report_id)
        pages.append(PublishedPage(
            report_id=listed.report_id, view_url=listed.view_url, title=listed.title,
            last_modified=listed.last_modified,
            owner_user_id=owner.owner_user_id if owner is not None else None,
            artifact_paths=tuple(sorted(entries.get(listed.report_id, ()))),
        ))
    return pages


def _session() -> Session:
    return Session(get_engine(get_app_settings().database.uri))


def list_pages(*, as_json: bool) -> str:
    publisher = _publisher()
    with _session() as session:
        pages = _pages(session, publisher)
    if as_json:
        return "\n".join(page.model_dump_json() for page in pages)
    if not pages:
        return "The publisher serves no pages."
    lines = []
    for page in pages:
        owner = f"owner {page.owner_user_id}" if page.owner_user_id else "no owner"
        folders = ", ".join(page.artifact_paths) or "no artifact names it"
        lines.append(f"{page.report_id} | {owner} | {page.view_url} | {page.title or '(untitled)'} | {folders}")
    return "\n".join(lines)


def _page(pages: list[PublishedPage], report_id: str) -> PublishedPage:
    page = next((candidate for candidate in pages if candidate.report_id == report_id), None)
    if page is None:
        raise OperatorCommandError(f"The publisher serves no page {report_id}; nothing changed.")
    return page


def _record_entry(artifact_path: str) -> tuple[Path, str]:
    """The folder whose .published.json names this artifact, and its key there."""
    artifact = Path(artifact_path)
    return artifact.parent, artifact.name


def _read_record(published_dir: Path) -> dict[str, Any]:
    try:
        record = json.loads((published_dir / ".published.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


@contextmanager
def _record_locks(session: Session, *, publisher: _Publisher, artifact_paths: tuple[str, ...]) -> Iterator[None]:
    """Hold the lock Cowork's publish holds on each folder whose record names
    the page (publication_owner), in one order, so the record and the owner
    row can't change under the command."""
    locks = _DatabaseResourceLocks(ScopedSession(session, LOCAL_SCOPE))
    keys = sorted({publish_record_lock_key(_record_entry(artifact_path)[0]) for artifact_path in artifact_paths})
    with ExitStack() as held:
        try:
            for key in keys:
                held.enter_context(locks.lock(publisher.key, "published_page", key))
        except HTTPException:
            # A lock still held when its wait runs out answers 503.
            raise OperatorCommandError(
                "A publish from a folder that names this page is still running; nothing changed. Run the command again."
            ) from None
        yield


def _drop_entry(artifact_path: str, report_id: str) -> bool:
    """Removes the artifact's entry when it still names report_id."""
    published_dir, name = _record_entry(artifact_path)
    record = _read_record(published_dir)
    entry = record.get(name)
    if not isinstance(entry, dict) or entry.get("report_id") != report_id:
        return False
    record.pop(name)
    _write_published_map(published_dir / ".published.json", record)
    return True


def unpublish_page(*, report_id: str, force: bool) -> str:
    from anton.publisher import unpublish

    publisher = _publisher()
    with _session() as session:
        page = _page(_pages(session, publisher), report_id)
        with _record_locks(session, publisher=publisher, artifact_paths=page.artifact_paths):
            owner = find_owner(session, publisher.key, report_id)
            owner_user_id = owner.owner_user_id if owner is not None else None
            if owner_user_id is not None and not force:
                raise OperatorCommandError(
                    f"Page {report_id} belongs to user {owner_user_id}, who can stop sharing it in Cowork. "
                    "Run with --force to take it down anyway. Nothing changed."
                )
            try:
                unpublish(report_id, api_key=publisher.api_key, publish_url=publisher.url, ssl_verify=minds_ssl_verify())
            except Exception:
                raise OperatorCommandError(f"The publisher could not take down page {report_id}; nothing changed.") from None
            dropped = sum(_drop_entry(artifact_path, report_id) for artifact_path in page.artifact_paths)
            if owner is not None:
                session.delete(owner)
            session.commit()
    removed = [f"{dropped} artifact publish record entr{'y' if dropped == 1 else 'ies'} that named it"]
    if owner_user_id:
        removed.insert(0, f"its owner record (user {owner_user_id})")
    return (
        f"Took down page {report_id} ({page.view_url}): its link answers 404 from now on. "
        f"Removed {' and '.join(removed)}."
    )


def _visible_project(person: Person, artifact_path: str) -> Project | None:
    """The project holding the artifact, when the person's own requests can open it."""
    with _session() as raw:
        for project in ProjectService(ScopedSession(raw, personal_scope(person))).list_projects():
            root = Path(project.path).resolve()
            if Path(artifact_path).is_relative_to(root):
                return project
    return None


def give_page(*, report_id: str, person: Person) -> str:
    publisher = _publisher()
    with _session() as session:
        page = _page(_pages(session, publisher), report_id)
        if page.owner_user_id is not None:
            raise OperatorCommandError(f"Page {report_id} already belongs to user {page.owner_user_id}; nothing changed.")
        if not page.artifact_paths:
            raise OperatorCommandError(
                f"No artifact's publish record names page {report_id}, so nobody could update it from Cowork. "
                "Take it down with unpublish instead. Nothing changed."
            )
        if len(page.artifact_paths) > 1:
            raise OperatorCommandError(
                f"{len(page.artifact_paths)} artifacts' publish records name page {report_id}: "
                f"{', '.join(page.artifact_paths)}. Nothing changed."
            )
        (artifact_path,) = page.artifact_paths
        project = _visible_project(person, artifact_path)
        if project is None:
            raise OperatorCommandError(
                f"User {person.user_id} can't open the project that holds {artifact_path}; nothing changed."
            )
        published_dir, name = _record_entry(artifact_path)
        with _record_locks(session, publisher=publisher, artifact_paths=(artifact_path,)):
            entry = _read_record(published_dir).get(name)
            if not isinstance(entry, dict) or entry.get("report_id") != report_id:
                raise OperatorCommandError(f"{artifact_path}'s publish record no longer names page {report_id}; nothing changed.")
            if find_owner(session, publisher.key, report_id) is not None:
                raise OperatorCommandError(f"Page {report_id} got an owner meanwhile; nothing changed.")
            session.add(PublishedPageOwner(
                publisher_key=publisher.key, report_id=report_id,
                artifact_path=artifact_path, owner_user_id=person.user_id,
            ))
            session.commit()
    access = entry.get("mode") or "public"
    return (
        f"Gave page {report_id} ({page.view_url}, {page.title or 'untitled'}) to user {person.user_id}. "
        f"Artifact: {artifact_path}, in project {project.display_name or project.name}. "
        f"Access: {access}. They can now update it or stop sharing it in Cowork, and keep its link."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    listing = commands.add_parser("list", help="every page the publisher serves, its owner and the artifacts naming it")
    listing.add_argument("--json", action="store_true", help="one JSON object per page")
    unpublishing = commands.add_parser("unpublish", help="take one page down")
    unpublishing.add_argument("--report-id", required=True)
    unpublishing.add_argument("--force", action="store_true", help="also take down a page that has an owner")
    giving = commands.add_parser("give", help="make a person the owner of a page with no owner")
    giving.add_argument("--report-id", required=True)
    add_person_arguments(giving)
    arguments = parser.parse_args(argv)
    if arguments.action == "list":
        return run_operator_command(lambda: list_pages(as_json=arguments.json))
    if arguments.action == "unpublish":
        return run_operator_command(lambda: unpublish_page(report_id=arguments.report_id, force=arguments.force))
    return run_operator_command(lambda: give_page(report_id=arguments.report_id, person=person_from_arguments(arguments)))


if __name__ == "__main__":
    sys.exit(main())
