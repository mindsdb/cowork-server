"""Working folders attached to a desktop chat."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.conversation_folder import ConversationFolder
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.services.conversation_folders import (
    MAX_FOLDERS_PER_CONVERSATION,
    ConversationFolderService,
    FolderAlreadyAttached,
    FolderLimitReached,
    FolderNotFound,
    folder_refusal,
    resolve_folder,
)
from cowork.services.conversations import ConversationService
from cowork.services.projects import ProjectService


def _fk_enforcing_engine():
    """Foreign keys on, as on Postgres, so a folder row left behind refuses the
    conversation delete instead of silently orphaning."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    return engine


def _folders(session: Session, conversation_id) -> list[ConversationFolder]:
    return list(
        session.exec(
            select(ConversationFolder).where(ConversationFolder.conversation_id == conversation_id)
        ).all()
    )


def _chat_with_folder(session: Session, tmp_path: Path, name: str):
    project = Project(name=name, path=str(tmp_path / name))
    session.add(project)
    session.commit()
    scoped = ScopedSession(session, LOCAL_SCOPE)
    conversation = ConversationService(scoped).create_conversation("topic", project_id=project.id)
    scoped.add(ConversationFolder(conversation_id=conversation.id, path=str(tmp_path / "docs")))
    session.commit()
    return project, conversation


def test_deleting_a_chat_removes_its_folders(tmp_path):
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        _project, conversation = _chat_with_folder(session, tmp_path, "folders-chat")
        assert _folders(session, conversation.id)

        scoped = ScopedSession(session, LOCAL_SCOPE)
        assert ConversationService(scoped).delete_conversation(conversation.id) is True

        assert _folders(session, conversation.id) == []


def test_deleting_a_project_removes_its_chats_folders(tmp_path):
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        project, conversation = _chat_with_folder(session, tmp_path, "folders-project")

        assert ProjectService(ScopedSession(session, LOCAL_SCOPE)).delete_project(project.id) is True

        assert _folders(session, conversation.id) == []


def test_clearing_a_chats_history_keeps_its_folders(tmp_path):
    """Folders are the user's setting for the chat, not something a turn made."""
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        _project, conversation = _chat_with_folder(session, tmp_path, "folders-history")
        scoped = ScopedSession(session, LOCAL_SCOPE)
        service = ConversationService(scoped)
        session.add(Message(conversation_id=conversation.id, role="user", content='"hi"'))
        session.commit()
        service.save_assistant_turn(conversation.id, "hello", events=[])
        first_answer = next(
            m.id
            for m in session.exec(
                select(Message).where(
                    Message.conversation_id == conversation.id, Message.role == "assistant"
                )
            ).all()
        )

        service.delete_turn(conversation.id, first_answer)

        assert len(_folders(session, conversation.id)) == 1


_STORE_ENV = {
    "COWORK_HOME": "home",
    "COWORK_PROJECTS_DIR": "projects",
    "COWORK_FILES_DIR": "files",
    "COWORK_SKILLS_DIR": "skills",
    "COWORK_VAULT_DIR": "vault",
    "COWORK_MEMORY_DIR": "memory",
    "COWORK_CODING_DIR": "coding",
}


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    """Every Cowork store in its own directory, apart from the user's folders."""
    roots = {}
    for env, name in _STORE_ENV.items():
        root = tmp_path / "stores" / name
        root.mkdir(parents=True)
        monkeypatch.setenv(env, str(root))
        roots[env] = root
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.delenv("COWORK_TURN_BACKEND", raising=False)
    get_app_settings.cache_clear()
    yield roots
    get_app_settings.cache_clear()


@pytest.fixture()
def user_folders(tmp_path):
    docs = tmp_path / "user" / "docs"
    reports = tmp_path / "user" / "reports"
    docs.mkdir(parents=True)
    reports.mkdir(parents=True)
    return docs, reports


def _service_chat(session: Session, tmp_path: Path, name: str):
    project_dir = tmp_path / "user" / name
    project_dir.mkdir(parents=True, exist_ok=True)
    project = Project(name=name, path=str(project_dir))
    session.add(project)
    session.commit()
    scoped = ScopedSession(session, LOCAL_SCOPE)
    conversation = ConversationService(scoped).create_conversation("topic", project_id=project.id)
    return ConversationFolderService(scoped), conversation, project_dir


def test_an_ordinary_folder_is_accepted(stores, user_folders):
    docs, _reports = user_folders
    assert folder_refusal(str(docs), None) is None
    assert resolve_folder(str(docs), None) == docs.resolve()


@pytest.mark.parametrize("raw", ["relative/docs", "/no/such/folder", "/tmp/x\x00y"])
def test_a_path_that_is_not_an_existing_absolute_folder_is_refused(stores, raw):
    assert folder_refusal(raw, None) == "Choose an existing local folder"


def test_a_file_is_refused(stores, tmp_path):
    file = tmp_path / "user" / "notes.txt"
    file.parent.mkdir(parents=True)
    file.write_text("x")
    assert folder_refusal(str(file), None) == "Choose an existing local folder"


@pytest.mark.parametrize("env", sorted(_STORE_ENV))
def test_every_store_root_is_refused_inside_equal_and_around(stores, env):
    root = stores[env]
    inside = root / "inside"
    inside.mkdir()
    reason = "Choose a folder that does not hold Cowork's own data"
    assert folder_refusal(str(root), None) == reason
    assert folder_refusal(str(inside), None) == reason
    # A folder holding the store is refused too: the agent may write in it.
    assert folder_refusal(str(root.parent), None) == reason


def test_the_chats_own_project_folder_is_refused(stores, tmp_path):
    project = tmp_path / "user" / "proj"
    (project / "sub").mkdir(parents=True)
    reason = "This folder is already part of the chat's project"
    assert folder_refusal(str(project), str(project)) == reason
    assert folder_refusal(str(project / "sub"), str(project)) == reason


@pytest.mark.skipif(sys.platform != "darwin", reason="case-insensitive volume")
def test_a_case_variant_of_a_store_root_is_refused_on_macos(stores):
    root = stores["COWORK_VAULT_DIR"]
    variant = root.parent / root.name.upper()
    assert folder_refusal(str(variant), None) == "Choose a folder that does not hold Cowork's own data"


@pytest.mark.parametrize(
    "env, value", [("COWORK_TENANCY_MODE", "org"), ("COWORK_TURN_BACKEND", "remote")]
)
def test_org_mode_and_a_remote_backend_refuse_before_touching_disk(stores, monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    get_app_settings.cache_clear()
    # A missing path would read "Choose an existing local folder" if the
    # filesystem were consulted first.
    assert folder_refusal("/no/such/folder", None) == "Working folders are only available in the desktop app"


def test_a_folder_swapped_for_a_link_into_a_store_is_refused_later(stores, tmp_path):
    folder = tmp_path / "user" / "swapped"
    folder.mkdir(parents=True)
    assert folder_refusal(str(folder), None) is None

    folder.rmdir()
    folder.symlink_to(stores["COWORK_VAULT_DIR"], target_is_directory=True)

    assert folder_refusal(str(folder), None) == "Choose a folder that does not hold Cowork's own data"


def test_two_folders_attach_and_list_in_order(stores, user_folders, tmp_path):
    docs, reports = user_folders
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        service, conversation, _ = _service_chat(session, tmp_path, "svc-two")

        first = service.add_folder(conversation.id, str(docs))
        second = service.add_folder(conversation.id, str(reports))

        _conv, rows = service.list_folders(conversation.id)
        assert [r.id for r in rows] == [first.id, second.id]
        assert [r.path for r in rows] == [str(docs.resolve()), str(reports.resolve())]


def test_the_same_folder_twice_is_a_duplicate(stores, user_folders, tmp_path):
    docs, _ = user_folders
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        service, conversation, _ = _service_chat(session, tmp_path, "svc-dup")
        service.add_folder(conversation.id, str(docs))

        with pytest.raises(FolderAlreadyAttached):
            service.add_folder(conversation.id, str(docs / ".." / "docs"))


@pytest.mark.skipif(sys.platform != "darwin", reason="case-insensitive volume")
def test_a_case_variant_of_an_attached_folder_is_a_duplicate_on_macos(stores, user_folders, tmp_path):
    docs, _ = user_folders
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        service, conversation, _ = _service_chat(session, tmp_path, "svc-case")
        service.add_folder(conversation.id, str(docs))

        with pytest.raises(FolderAlreadyAttached):
            service.add_folder(conversation.id, str(docs.parent / docs.name.upper()))


def test_a_seventeenth_folder_is_refused(stores, tmp_path):
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        service, conversation, _ = _service_chat(session, tmp_path, "svc-cap")
        for i in range(MAX_FOLDERS_PER_CONVERSATION):
            folder = tmp_path / "user" / f"f{i}"
            folder.mkdir(parents=True)
            service.add_folder(conversation.id, str(folder))
        extra = tmp_path / "user" / "extra"
        extra.mkdir()

        with pytest.raises(FolderLimitReached):
            service.add_folder(conversation.id, str(extra))


def test_a_folder_of_another_chat_is_not_found(stores, user_folders, tmp_path):
    docs, _ = user_folders
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        service, chat_a, _ = _service_chat(session, tmp_path, "svc-a")
        _, chat_b, _ = _service_chat(session, tmp_path, "svc-b")
        folder_of_b = service.add_folder(chat_b.id, str(docs))

        with pytest.raises(FolderNotFound):
            service.remove_folder(chat_a.id, folder_of_b.id)
        with pytest.raises(FolderNotFound):
            service.get_folder(chat_a.id, folder_of_b.id)
        assert len(service.list_folders(chat_b.id)[1]) == 1


def test_another_members_chat_is_not_found_in_org_scope(stores, tmp_path):
    engine = _fk_enforcing_engine()
    with Session(engine, expire_on_commit=False) as session:
        project = Project(name="org-proj", path=str(tmp_path / "org-proj"), org_id="org-1")
        session.add(project)
        session.commit()
        owner = ScopedSession(session, TenantScope(org_mode=True, org_id="org-1", user_id="u-1"))
        conversation = ConversationService(owner).create_conversation("t", project_id=project.id)

    with Session(engine) as session:
        other = ScopedSession(session, TenantScope(org_mode=True, org_id="org-1", user_id="u-2"))
        with pytest.raises(FolderNotFound):
            ConversationFolderService(other).list_folders(conversation.id)


def test_removing_a_folder_detaches_it(stores, user_folders, tmp_path):
    docs, reports = user_folders
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        service, conversation, _ = _service_chat(session, tmp_path, "svc-remove")
        kept = service.add_folder(conversation.id, str(docs))
        gone = service.add_folder(conversation.id, str(reports))

        service.remove_folder(conversation.id, gone.id)

        assert [r.id for r in service.list_folders(conversation.id)[1]] == [kept.id]


def test_the_legacy_anton_home_is_refused(stores, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".anton").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    reason = "Choose a folder that does not hold Cowork's own data"

    assert folder_refusal(str(home / ".anton"), None) == reason
    assert folder_refusal(str(home), None) == reason


@pytest.mark.parametrize(
    "env, value_for",
    [
        ("MASTER_KEY_PATH", lambda f: str(f / ".master_key")),
        ("DATABASE_URI", lambda f: f"sqlite:///{f / 'cowork.db'}"),
    ],
)
def test_a_folder_holding_the_master_key_or_the_database_is_refused(
    stores, tmp_path, monkeypatch, env, value_for
):
    holder = tmp_path / "user" / "secrets"
    holder.mkdir(parents=True)
    monkeypatch.setenv(env, value_for(holder))
    get_app_settings.cache_clear()

    assert folder_refusal(str(holder), None) == "Choose a folder that does not hold Cowork's own data"


def test_a_store_root_that_cannot_resolve_still_refuses(stores, monkeypatch):
    """Dropping a root that fails to resolve would accept folders inside it."""
    vault = stores["COWORK_VAULT_DIR"]
    inside = vault / "inside"
    inside.mkdir()
    real_resolve = Path.resolve

    def _resolve(self, strict=False):
        if self == vault:
            raise RuntimeError("symlink loop")
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _resolve)

    assert folder_refusal(str(inside), None) == "Choose a folder that does not hold Cowork's own data"
