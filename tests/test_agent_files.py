from __future__ import annotations

from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from cowork.harnesses.anton_harness.agent_files import CLOUD_WORKING_DIR, AgentFiles


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "projects" / "general"
    (root / "data").mkdir(parents=True)
    (root / "data" / "stock.csv").write_text("part,qty\nP100,250\n")
    (tmp_path / "outside.csv").write_text("secret\n")
    return root


class TestATurnThisServerRuns:
    def test_a_relative_path_is_read_from_the_project_root(self, project: Path):
        files = AgentFiles.on_this_server(project)

        assert files.project_path("data/stock.csv") == "data/stock.csv"
        assert files.project_path("./data/../data/stock.csv") == "data/stock.csv"

    def test_an_absolute_path_inside_the_project_maps_into_it(self, project: Path):
        files = AgentFiles.on_this_server(project)

        assert files.project_path(str(project / "data" / "stock.csv")) == "data/stock.csv"

    def test_a_path_that_climbs_out_of_the_project_names_nothing(self, project: Path):
        files = AgentFiles.on_this_server(project)

        assert files.project_path("../../outside.csv") is None
        assert files.project_path(str(project / ".." / ".." / "outside.csv")) is None

    def test_a_link_that_leads_out_of_the_project_names_nothing(self, project: Path):
        (project / "data" / "link.csv").symlink_to(project.parent.parent / "outside.csv")

        assert AgentFiles.on_this_server(project).project_path("data/link.csv") is None

    def test_a_file_that_is_not_there_names_nothing(self, project: Path):
        files = AgentFiles.on_this_server(project)

        assert files.project_path("data/missing.csv") is None
        assert files.project_path("data") is None


class TestATurnACloudPodRuns:
    def test_the_pods_working_directory_is_the_conversations_folder(self, project: Path):
        conversation_id = uuid4()
        attachment = project / "conversations" / str(conversation_id) / "attachments" / "f1" / "rates.pdf"
        attachment.parent.mkdir(parents=True)
        attachment.write_bytes(b"%PDF")
        files = AgentFiles.in_cloud_pod(project, conversation_id)
        expected = f"conversations/{conversation_id}/attachments/f1/rates.pdf"

        assert files.project_path("/workspace/attachments/f1/rates.pdf") == expected
        assert files.project_path("attachments/f1/rates.pdf") == expected

    def test_a_path_the_pod_reads_outside_its_working_directory_names_nothing(self, project: Path):
        # Inside the pod both read /data/stock.csv, which is not the project's
        # file even though the project holds a file at data/stock.csv.
        files = AgentFiles.in_cloud_pod(project, uuid4())

        assert files.project_path("/workspace/../../data/stock.csv") is None
        assert files.project_path("../../data/stock.csv") is None

    def test_a_link_into_another_conversations_folder_names_nothing(self, project: Path):
        mine, theirs = uuid4(), uuid4()
        (project / "conversations" / str(mine)).mkdir(parents=True)
        (project / "conversations" / str(theirs)).mkdir(parents=True)
        (project / "conversations" / str(theirs) / "notes.csv").write_text("a,b\n1,2\n")
        (project / "conversations" / str(mine) / "peek.csv").symlink_to(f"../{theirs}/notes.csv")

        assert AgentFiles.in_cloud_pod(project, mine).project_path("peek.csv") is None

    def test_a_server_path_into_another_conversations_folder_names_nothing(self, project: Path):
        theirs = uuid4()
        other = project / "conversations" / str(theirs) / "notes.csv"
        other.parent.mkdir(parents=True)
        other.write_text("a,b\n1,2\n")

        assert AgentFiles.in_cloud_pod(project, uuid4()).project_path(str(other)) is None

    def test_the_working_directory_is_antons_cloud_workspace(self):
        from anton.cloud_turn.session import DEFAULT_CLOUD_WORKSPACE_PATH

        assert CLOUD_WORKING_DIR == PurePosixPath(DEFAULT_CLOUD_WORKSPACE_PATH)
