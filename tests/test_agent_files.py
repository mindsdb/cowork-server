from __future__ import annotations

import errno
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

import pytest
from anton.core.backends.base import Cell
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness import agent_files as agent_files_mod
from cowork.harnesses.anton_harness.agent_files import (
    AgentFiles,
    Attachment,
    NamedFile,
    _posix_path_in,
    conversation_attachments,
    read_conversation_attachments,
)
from cowork.harnesses.anton_harness.figure_sources import SourceKind, TurnStep, find_figure_sources
from cowork.services.files import FileService, attachment_purpose


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "projects" / "general"
    (root / "data").mkdir(parents=True)
    (root / "data" / "stock.csv").write_text("part,qty\nP100,250\n")
    (tmp_path / "outside.csv").write_text("secret\n")
    return root


@pytest.fixture
def stored(tmp_path: Path) -> Attachment:
    """A file attached in the chat composer, where the files store keeps it."""
    file_id = str(uuid4())
    path = tmp_path / "files" / file_id / "stock.csv"
    path.parent.mkdir(parents=True)
    path.write_text("part,qty\nP100,999\n")
    return Attachment(file_id=file_id, stored_path=str(path))


@pytest.fixture
def artifacts(project: Path) -> Path:
    """The project's artifacts folder, which a cloud pod sees at
    /project-artifacts, holding report/data.csv."""
    folder = project / ".anton" / "artifacts"
    (folder / "report").mkdir(parents=True)
    (folder / "report" / "data.csv").write_text("qty\n250\n")
    return folder


def _in_pod(project: Path, conversation_id, **kwargs) -> AgentFiles:
    return AgentFiles.in_cloud_pod(
        project, conversation_id=conversation_id, workspace_rel_path="projects/general", **kwargs,
    )


def _flaky_stat(monkeypatch, *, name: str, error: int) -> None:
    """os.stat fails with ``error`` for any path ending in ``name``."""
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if os.fspath(path).endswith(name):
            raise OSError(error, os.strerror(error), os.fspath(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)


class TestATurnThisServerRuns:
    def test_a_relative_path_is_read_from_the_project_root(self, project: Path):
        files = AgentFiles.on_this_server(project)

        assert files.named_file("data/stock.csv") == NamedFile(path="data/stock.csv")
        assert files.named_file("./data/../data/stock.csv") == NamedFile(path="data/stock.csv")

    def test_an_absolute_path_inside_the_project_maps_into_it(self, project: Path):
        files = AgentFiles.on_this_server(project)

        assert files.named_file(str(project / "data" / "stock.csv")) == NamedFile(path="data/stock.csv")

    def test_an_absolute_path_through_a_link_to_the_project_maps_into_it(self, project: Path, tmp_path: Path):
        # A folder picked through ~/Dropbox is stored resolved, while the
        # agent may name it by the link the user pasted.
        alias = tmp_path / "alias"
        alias.symlink_to(project)

        files = AgentFiles.on_this_server(project)

        assert files.named_file(str(alias / "data" / "stock.csv")) == NamedFile(path="data/stock.csv")

    def test_a_path_that_climbs_out_of_the_project_names_nothing(self, project: Path):
        files = AgentFiles.on_this_server(project)

        assert files.named_file("../../outside.csv") is None
        assert files.named_file(str(project / ".." / ".." / "outside.csv")) is None

    def test_a_link_that_leads_out_of_the_project_names_nothing(self, project: Path):
        (project / "data" / "link.csv").symlink_to(project.parent.parent / "outside.csv")

        assert AgentFiles.on_this_server(project).named_file("data/link.csv") is None

    def test_a_dotdot_after_a_link_climbs_from_where_the_link_leads(self, project: Path):
        # The OS reads dl/../stock.csv as sub/inner/../stock.csv, which is
        # sub/stock.csv, not the project's own stock.csv.
        (project / "stock.csv").write_text("qty\n1\n")
        (project / "sub" / "inner").mkdir(parents=True)
        (project / "sub" / "stock.csv").write_text("qty\n250\n")
        (project / "dl").symlink_to(project / "sub" / "inner")

        assert AgentFiles.on_this_server(project).named_file("dl/../stock.csv") == NamedFile(path="sub/stock.csv")

    def test_a_dotdot_after_a_link_out_of_the_project_names_nothing(self, project: Path, tmp_path: Path):
        (project / "stock.csv").write_text("qty\n1\n")
        (tmp_path / "elsewhere" / "inner").mkdir(parents=True)
        (tmp_path / "elsewhere" / "stock.csv").write_text("qty\n250\n")
        (project / "out").symlink_to(tmp_path / "elsewhere" / "inner")

        assert AgentFiles.on_this_server(project).named_file("out/../stock.csv") is None

    def test_a_file_that_is_not_there_names_nothing(self, project: Path):
        files = AgentFiles.on_this_server(project)

        assert files.named_file("data/missing.csv") is None
        assert files.named_file("data") is None
        assert files.named_file("") is None

    def test_a_missing_file_logs_nothing(self, project: Path, caplog):
        with caplog.at_level(logging.WARNING, logger=agent_files_mod.__name__):
            assert AgentFiles.on_this_server(project).named_file("data/missing.csv") is None

        assert caplog.records == []

    @pytest.mark.parametrize("error", [errno.EIO, errno.ESTALE])
    def test_a_lookup_the_disk_fails_is_logged_and_names_nothing(self, project: Path, monkeypatch, caplog, error):
        _flaky_stat(monkeypatch, name="stock.csv", error=error)

        with caplog.at_level(logging.WARNING, logger=agent_files_mod.__name__):
            assert AgentFiles.on_this_server(project).named_file("data/stock.csv") is None

        [record] = caplog.records
        assert record.levelno == logging.WARNING
        assert record.exc_info[1].errno == error

    def test_a_file_outside_the_project_is_not_looked_at(self, project: Path, monkeypatch, caplog):
        _flaky_stat(monkeypatch, name="outside.csv", error=errno.EIO)

        with caplog.at_level(logging.WARNING, logger=agent_files_mod.__name__):
            assert AgentFiles.on_this_server(project).named_file(str(project.parent.parent / "outside.csv")) is None

        assert caplog.records == []

    def test_a_nested_project_path_uses_forward_slashes(self, project: Path):
        (project / "data" / "2026").mkdir()
        (project / "data" / "2026" / "q3.csv").write_text("qty\n1\n")

        assert AgentFiles.on_this_server(project).named_file("data/2026/q3.csv") == NamedFile(path="data/2026/q3.csv")

    def test_the_turn_carries_its_projects_id(self, project: Path):
        project_id = uuid4()

        assert AgentFiles.on_this_server(project, project_id=project_id).project_id == str(project_id)
        assert AgentFiles.on_this_server(project).project_id is None


class TestAFileAttachedToAConversationThisServerRuns:
    def test_it_is_named_by_its_files_store_id(self, project: Path, stored: Attachment):
        files = AgentFiles.on_this_server(project, attachments=[stored])

        assert files.named_file(stored.stored_path) == NamedFile(attachment_id=stored.file_id)

    def test_a_relative_path_to_it_is_named_by_its_files_store_id(self, project: Path, stored: Attachment):
        files = AgentFiles.on_this_server(project, attachments=[stored])

        assert files.named_file(os.path.relpath(stored.stored_path, project)) == NamedFile(
            attachment_id=stored.file_id,
        )

    def test_a_stored_file_this_conversation_did_not_attach_names_nothing(self, project: Path, stored: Attachment):
        files = AgentFiles.on_this_server(project, attachments=[])

        assert files.named_file(stored.stored_path) is None

    def test_an_attachment_gone_from_the_store_names_nothing(self, project: Path, stored: Attachment):
        Path(stored.stored_path).unlink()

        assert AgentFiles.on_this_server(project, attachments=[stored]).named_file(stored.stored_path) is None

    def test_a_dotdot_after_a_link_that_folds_onto_it_names_nothing(
        self, project: Path, stored: Attachment, tmp_path: Path,
    ):
        # <id>/hop/../stock.csv folds to the attachment as text, but the OS
        # climbs from where hop leads and reads another file.
        (tmp_path / "elsewhere" / "inner").mkdir(parents=True)
        (tmp_path / "elsewhere" / "stock.csv").write_text("qty\n1\n")
        (Path(stored.stored_path).parent / "hop").symlink_to(tmp_path / "elsewhere" / "inner")
        literal = str(Path(stored.stored_path).parent / "hop" / ".." / "stock.csv")

        assert AgentFiles.on_this_server(project, attachments=[stored]).named_file(literal) is None


class TestATurnACloudPodRuns:
    @pytest.mark.parametrize(
        "working_dir",
        [
            # The controller's conversation folder on the shared mount (persistent workspace).
            "/mnt/cowork-shared/projects/general/conversations/{conversation_id}",
            # The controller's copy of that folder (ephemeral workspace).
            "/workspace-ephemeral",
            # anton's own default when the controller names none.
            "/workspace",
        ],
    )
    def test_the_pod_names_the_conversations_folder_by_its_working_directory(self, project: Path, working_dir: str):
        conversation_id = uuid4()
        folder = project / "conversations" / str(conversation_id)
        folder.mkdir(parents=True)
        (folder / "stock.csv").write_text("qty\n250\n")
        literal = f"{working_dir.format(conversation_id=conversation_id)}/stock.csv"

        assert _in_pod(project, conversation_id).named_file(literal) == NamedFile(
            path=f"conversations/{conversation_id}/stock.csv",
        )
        assert _in_pod(project, conversation_id).named_file("stock.csv") == NamedFile(
            path=f"conversations/{conversation_id}/stock.csv",
        )

    def test_the_shared_mount_root_is_configurable(self, project: Path, monkeypatch):
        monkeypatch.setenv("COWORK_TURN_POD_SHARED_MOUNT_ROOT", "/efs")
        conversation_id = uuid4()
        folder = project / "conversations" / str(conversation_id)
        folder.mkdir(parents=True)
        (folder / "stock.csv").write_text("qty\n250\n")
        files = _in_pod(project, conversation_id)

        assert files.named_file(f"/efs/projects/general/conversations/{conversation_id}/stock.csv") == NamedFile(
            path=f"conversations/{conversation_id}/stock.csv",
        )
        assert files.named_file(f"/mnt/cowork-shared/projects/general/conversations/{conversation_id}/stock.csv") is None

    def test_another_folder_on_the_mount_names_nothing(self, project: Path):
        mine, theirs = uuid4(), uuid4()
        for conversation_id in (mine, theirs):
            folder = project / "conversations" / str(conversation_id)
            folder.mkdir(parents=True)
            (folder / "stock.csv").write_text("qty\n250\n")
        files = _in_pod(project, mine)

        assert files.named_file(f"/mnt/cowork-shared/projects/general/conversations/{theirs}/stock.csv") is None
        assert files.named_file(f"/mnt/cowork-shared/projects/other/conversations/{mine}/stock.csv") is None

    def test_without_the_projects_place_on_the_mount_only_the_fixed_working_directories_count(self, project: Path):
        conversation_id = uuid4()
        folder = project / "conversations" / str(conversation_id)
        folder.mkdir(parents=True)
        (folder / "stock.csv").write_text("qty\n250\n")
        files = AgentFiles.in_cloud_pod(project, conversation_id=conversation_id, workspace_rel_path=None)

        assert files.named_file("/workspace-ephemeral/stock.csv") == NamedFile(
            path=f"conversations/{conversation_id}/stock.csv",
        )
        assert files.named_file(f"/mnt/cowork-shared/projects/general/conversations/{conversation_id}/stock.csv") is None

    def test_antons_default_workspace_is_a_working_directory(self, project: Path):
        from anton.cloud_turn.session import DEFAULT_CLOUD_WORKSPACE_PATH

        assert PurePosixPath(DEFAULT_CLOUD_WORKSPACE_PATH) in _in_pod(project, uuid4()).pod_working_dirs

    def test_a_path_the_pod_reads_outside_its_working_directory_names_nothing(self, project: Path):
        # Inside the pod both read /data/stock.csv, which is not the project's
        # file even though the project holds a file at data/stock.csv.
        files = _in_pod(project, uuid4())

        assert files.named_file("/workspace/../../data/stock.csv") is None
        assert files.named_file("../../data/stock.csv") is None

    def test_an_absolute_path_outside_the_working_directories_names_nothing_though_the_folder_holds_its_name(
        self, project: Path,
    ):
        # The pod reads /stock.csv from its own disk, not the conversation's
        # stock.csv.
        conversation_id = uuid4()
        folder = project / "conversations" / str(conversation_id)
        folder.mkdir(parents=True)
        (folder / "stock.csv").write_text("qty\n250\n")

        assert _in_pod(project, conversation_id).named_file("/stock.csv") is None

    def test_a_dotdot_that_climbs_above_the_working_directory_and_back_names_nothing(self, project: Path):
        # The pod sees nothing above its working directory, so ../<id>/x does
        # not lead back into it there.
        conversation_id = uuid4()
        folder = project / "conversations" / str(conversation_id)
        folder.mkdir(parents=True)
        (folder / "stock.csv").write_text("qty\n250\n")

        assert _in_pod(project, conversation_id).named_file(f"../{conversation_id}/stock.csv") is None

    def test_a_dotdot_after_a_link_climbs_from_where_the_link_leads(self, project: Path):
        conversation_id = uuid4()
        folder = project / "conversations" / str(conversation_id)
        (folder / "sub" / "inner").mkdir(parents=True)
        (folder / "stock.csv").write_text("qty\n1\n")
        (folder / "sub" / "stock.csv").write_text("qty\n250\n")
        (folder / "dl").symlink_to(folder / "sub" / "inner")

        assert _in_pod(project, conversation_id).named_file("/workspace/dl/../stock.csv") == NamedFile(
            path=f"conversations/{conversation_id}/sub/stock.csv",
        )

    def test_a_link_into_another_conversations_folder_names_nothing(self, project: Path):
        mine, theirs = uuid4(), uuid4()
        (project / "conversations" / str(mine)).mkdir(parents=True)
        (project / "conversations" / str(theirs)).mkdir(parents=True)
        (project / "conversations" / str(theirs) / "notes.csv").write_text("a,b\n1,2\n")
        (project / "conversations" / str(mine) / "peek.csv").symlink_to(f"../{theirs}/notes.csv")

        assert _in_pod(project, mine).named_file("peek.csv") is None

    def test_a_server_path_names_nothing_even_into_the_pods_own_folder(self, project: Path):
        # The pod can't see this server's disk, so it never read this path,
        # though the file it names is in the pod's own folder here.
        mine = uuid4()
        own = project / "conversations" / str(mine) / "notes.csv"
        own.parent.mkdir(parents=True)
        own.write_text("a,b\n1,2\n")

        assert _in_pod(project, mine).named_file(str(own)) is None

    def test_a_file_the_pod_reads_in_the_projects_artifacts_is_named_by_its_project_path(
        self, project: Path, artifacts: Path,
    ):
        conversation_id = uuid4()
        (project / "conversations" / str(conversation_id)).mkdir(parents=True)

        assert _in_pod(project, conversation_id).named_file("/project-artifacts/report/data.csv") == NamedFile(
            path=".anton/artifacts/report/data.csv",
        )

    def test_it_names_the_path_a_turn_this_server_runs_names_for_the_same_artifact(
        self, project: Path, artifacts: Path,
    ):
        conversation_id = uuid4()
        (project / "conversations" / str(conversation_id)).mkdir(parents=True)

        in_pod = _in_pod(project, conversation_id).named_file("/project-artifacts/report/data.csv")
        here = AgentFiles.on_this_server(project).named_file(".anton/artifacts/report/data.csv")

        assert in_pod == here

    def test_without_the_projects_place_on_the_mount_the_artifacts_mount_names_nothing(
        self, project: Path, artifacts: Path,
    ):
        # The turn's job then runs as the default project, so the pod's
        # /project-artifacts holds that project's artifacts, not these.
        conversation_id = uuid4()
        (project / "conversations" / str(conversation_id)).mkdir(parents=True)
        files = AgentFiles.in_cloud_pod(project, conversation_id=conversation_id, workspace_rel_path=None)

        assert files.named_file("/project-artifacts/report/data.csv") is None

    @pytest.mark.parametrize("literal", [
        "/project-artifacts/../data/stock.csv",
        "/project-artifacts/report/../../data/stock.csv",
        "/project-artifacts/../../../data/stock.csv",
    ])
    def test_a_dotdot_above_the_artifacts_mount_names_nothing(self, project: Path, artifacts: Path, literal: str):
        # The pod sees nothing of the project above its artifacts folder.
        conversation_id = uuid4()
        (project / "conversations" / str(conversation_id)).mkdir(parents=True)

        assert _in_pod(project, conversation_id).named_file(literal) is None

    def test_a_link_from_the_artifacts_into_a_conversations_folder_names_nothing(
        self, project: Path, artifacts: Path,
    ):
        mine, theirs = uuid4(), uuid4()
        (project / "conversations" / str(mine)).mkdir(parents=True)
        (project / "conversations" / str(theirs)).mkdir(parents=True)
        (project / "conversations" / str(theirs) / "notes.csv").write_text("a,b\n1,2\n")
        (artifacts / "peek.csv").symlink_to(f"../../conversations/{theirs}/notes.csv")

        assert _in_pod(project, mine).named_file("/project-artifacts/peek.csv") is None

    def test_a_link_from_the_artifacts_to_the_pods_staged_attachment_names_nothing(
        self, project: Path, artifacts: Path, stored: Attachment,
    ):
        # The pod's artifacts mount holds the link, not the folder it leads
        # into, so the pod can't read through it.
        conversation_id = uuid4()
        staged = project / "conversations" / str(conversation_id) / "attachments" / stored.file_id / "stock.csv"
        staged.parent.mkdir(parents=True)
        staged.write_text(Path(stored.stored_path).read_text())
        (artifacts / "att.csv").symlink_to(
            f"../../conversations/{conversation_id}/attachments/{stored.file_id}/stock.csv",
        )
        files = _in_pod(project, conversation_id, attachments=[stored])

        assert files.named_file("/project-artifacts/att.csv") is None
        assert files.named_file(f"attachments/{stored.file_id}/stock.csv") == NamedFile(attachment_id=stored.file_id)

    def test_an_artifact_at_an_attachments_staged_place_is_named_by_its_project_path(
        self, project: Path, artifacts: Path, stored: Attachment,
    ):
        # The server stages attachments in the conversation's folder only, so
        # a file the artifacts folder holds at the same place is an artifact.
        conversation_id = uuid4()
        (project / "conversations" / str(conversation_id)).mkdir(parents=True)
        lookalike = artifacts / "attachments" / stored.file_id / "stock.csv"
        lookalike.parent.mkdir(parents=True)
        lookalike.write_text("part,qty\nP100,7\n")
        files = _in_pod(project, conversation_id, attachments=[stored])

        assert files.named_file(f"/project-artifacts/attachments/{stored.file_id}/stock.csv") == NamedFile(
            path=f".anton/artifacts/attachments/{stored.file_id}/stock.csv",
        )

    @pytest.mark.parametrize("literal", [
        "../../data/stock.csv",
        "/mnt/cowork-shared/projects/general/data/stock.csv",
        "/project-artifacts/../data/stock.csv",
    ])
    def test_a_project_files_upload_the_pod_cannot_see_names_nothing(
        self, project: Path, artifacts: Path, literal: str,
    ):
        # A Project files upload sits in the project, outside the
        # conversation's folder and the artifacts folder the pod mounts.
        conversation_id = uuid4()
        (project / "conversations" / str(conversation_id)).mkdir(parents=True)

        assert _in_pod(project, conversation_id).named_file(literal) is None

    def test_the_turn_carries_its_projects_id(self, project: Path):
        project_id = uuid4()

        assert _in_pod(project, uuid4(), project_id=project_id).project_id == str(project_id)

    def test_the_conversation_is_named_by_keyword(self, project: Path):
        with pytest.raises(TypeError):
            AgentFiles.in_cloud_pod(project, uuid4(), "projects/general")


class TestAFileAttachedToAConversationACloudPodRuns:
    @pytest.fixture
    def staged(self, project: Path, stored: Attachment):
        """``stored`` as stage_conversation_attachments copies it into the
        conversation's folder, and the conversation's id."""
        conversation_id = uuid4()
        copy = project / "conversations" / str(conversation_id) / "attachments" / stored.file_id / "stock.csv"
        copy.parent.mkdir(parents=True)
        copy.write_text(Path(stored.stored_path).read_text())
        return conversation_id

    @pytest.mark.parametrize(
        "working_dir",
        [
            "/mnt/cowork-shared/projects/general/conversations/{conversation_id}",
            "/workspace-ephemeral",
            "/workspace",
            ".",
        ],
    )
    def test_its_staged_copy_is_named_by_its_files_store_id(
        self, project: Path, stored: Attachment, staged, working_dir: str,
    ):
        files = _in_pod(project, staged, attachments=[stored])
        literal = f"{working_dir.format(conversation_id=staged)}/attachments/{stored.file_id}/stock.csv"

        assert files.named_file(literal) == NamedFile(attachment_id=stored.file_id)

    def test_a_staged_folder_this_conversation_did_not_attach_is_only_a_path(
        self, project: Path, stored: Attachment, staged,
    ):
        files = _in_pod(project, staged, attachments=[])

        assert files.named_file(f"attachments/{stored.file_id}/stock.csv") == NamedFile(
            path=f"conversations/{staged}/attachments/{stored.file_id}/stock.csv",
        )

    def test_a_link_planted_at_its_place_names_where_it_leads(self, project: Path, stored: Attachment, staged):
        folder = project / "conversations" / str(staged)
        (folder / "notes.csv").write_text("qty\n1\n")
        copy = folder / "attachments" / stored.file_id / "stock.csv"
        copy.unlink()
        copy.symlink_to(folder / "notes.csv")

        assert _in_pod(project, staged, attachments=[stored]).named_file(
            f"attachments/{stored.file_id}/stock.csv",
        ) == NamedFile(path=f"conversations/{staged}/notes.csv")


class TestAFigureACloudPodReadsFromTheProjectsArtifacts:
    def test_its_step_names_the_artifact_by_its_project_path(self, project: Path):
        conversation_id = uuid4()
        (project / "conversations" / str(conversation_id)).mkdir(parents=True)
        (project / ".anton" / "artifacts" / "stock").mkdir(parents=True)
        (project / ".anton" / "artifacts" / "stock" / "data.csv").write_text("part,site,qty\nP100,S1,250\n")
        code = "import pandas as pd\nprint(pd.read_csv('/project-artifacts/stock/data.csv'))"
        cell = TurnStep(
            tool_use_id="toolu_read",
            tool_name="scratchpad",
            action="exec",
            tool_input=json.dumps({"action": "exec", "name": "main", "code": code}),
            content=json.dumps(asdict(Cell(
                code=code, stdout="part site qty\nP100 S1 250\n", stderr="", error=None, description="",
            ))),
        )

        [step] = find_figure_sources(
            answer="S1 holds 250 units.", question="", steps=[cell], files=_in_pod(project, conversation_id),
        ).steps

        assert (step.kind, step.path) == (SourceKind.file, ".anton/artifacts/stock/data.csv")


class TestAProjectPathOnWindows:
    """CI runs on Linux, so a Windows desktop's paths are checked as pure paths."""

    root = PureWindowsPath(r"C:\Users\me\projects\general")

    def test_a_file_inside_the_project_is_given_with_forward_slashes(self):
        # The web UI reads the path as POSIX and drops one with a backslash.
        assert _posix_path_in(self.root / "data" / "stock.csv", root=self.root) == "data/stock.csv"

    def test_a_differently_cased_spelling_of_the_project_is_inside_it(self):
        target = PureWindowsPath(r"c:\users\ME\Projects\general\data\stock.csv")

        assert _posix_path_in(target, root=self.root) == "data/stock.csv"

    def test_another_drive_or_a_sibling_folder_is_outside_the_project(self):
        assert _posix_path_in(PureWindowsPath(r"D:\Users\me\projects\general\stock.csv"), root=self.root) is None
        assert _posix_path_in(PureWindowsPath(r"C:\Users\me\projects\general-old\stock.csv"), root=self.root) is None


class TestTheConversationsAttachments:
    @pytest.fixture
    def scoped(self):
        with Session(get_engine(get_app_settings().database.uri)) as session:
            yield ScopedSession(session, LOCAL_SCOPE)

    def _attach(self, scoped: ScopedSession, *, purpose: str):
        return FileService(scoped).create_file_from_bytes(
            filename="stock.csv", content_type="text/csv", data=b"qty\n250\n", purpose=purpose,
        )

    def test_they_are_the_files_attached_to_it(self, scoped: ScopedSession):
        conversation_id = uuid4()
        mine = self._attach(scoped, purpose=attachment_purpose(str(conversation_id)))
        self._attach(scoped, purpose=attachment_purpose(str(uuid4())))
        self._attach(scoped, purpose="assistants")

        assert conversation_attachments(scoped, conversation_id=conversation_id) == (
            Attachment(file_id=str(mine.id), stored_path=mine.path),
        )

    @pytest.mark.asyncio
    async def test_a_turn_reads_them_in_a_unit_of_its_own(self, scoped: ScopedSession):
        conversation_id = uuid4()
        mine = self._attach(scoped, purpose=attachment_purpose(str(conversation_id)))

        assert await read_conversation_attachments(conversation_id=conversation_id, scope=LOCAL_SCOPE) == (
            Attachment(file_id=str(mine.id), stored_path=mine.path),
        )

    @pytest.mark.asyncio
    async def test_a_failed_read_is_logged_and_lists_none(self, monkeypatch, caplog):
        def fail(session, *, conversation_id):
            raise RuntimeError("database gone")

        monkeypatch.setattr(agent_files_mod, "conversation_attachments", fail)

        with caplog.at_level(logging.WARNING, logger=agent_files_mod.__name__):
            assert await read_conversation_attachments(conversation_id=uuid4(), scope=LOCAL_SCOPE) == ()

        [record] = caplog.records
        assert record.levelno == logging.WARNING
