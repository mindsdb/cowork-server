"""Images a remote turn's doc names by its pod's path: placed in the project by
the pod's mounts, and read one component at a time without following links,
since this server sees every organization's files."""
from __future__ import annotations

import os
from uuid import uuid4

import pytest

from cowork.docs.images import MAX_IMAGE_BYTES, localize_images
from cowork.docs.model import DocError
from cowork.docs.remote import pod_image_reader
from cowork.harnesses.anton_harness.agent_files import AgentFiles
from cowork.services.artifact_roots import project_artifacts_base

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d4944415478da63f8cfc0f01f0005000201a0d5e1b70000000049454e44ae426082")
POD_PATH = "/workspace/attachments/f1/banner.png"


def _pod(tmp_path):
    project = tmp_path / "projects" / "proj"
    conv = uuid4()
    files = AgentFiles.in_cloud_pod(project, conversation_id=conv, workspace_rel_path="projects/proj")
    return project, conv, files


def test_a_pod_path_lands_in_the_conversations_folder_or_the_projects_artifacts(tmp_path):
    project, conv, files = _pod(tmp_path)
    assert files.pod_project_parts(POD_PATH) == ("conversations", str(conv), "attachments", "f1", "banner.png")
    assert files.pod_project_parts("/project-artifacts/report/chart.png") == (
        ".anton", "artifacts", "report", "chart.png")
    for outside in ("/etc/passwd", "/workspace/../other/x.png", "/workspace/a/../../x.png",
                    "/workspace/a/../b.png", "/project-artifacts"):
        assert files.pod_project_parts(outside) is None, outside
    assert AgentFiles.on_this_server(project).pod_project_parts("/workspace/a.png") is None


def test_the_org_layout_keeps_doc_images_and_attachments_in_the_project(tmp_path):
    project, _conv, files = _pod(tmp_path)
    # The desktop's may_use_image root: the folder that holds .anton/artifacts.
    assert project_artifacts_base(project).parent.parent == project
    assert files.project_root == project
    assert files.pod_project_parts(POD_PATH)[0] == "conversations"


def test_an_attached_image_is_read_by_its_pod_path(tmp_path):
    project, conv, files = _pod(tmp_path)
    upload = project / "conversations" / str(conv) / "attachments" / "f1" / "banner.png"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(PNG)
    content, images = localize_images(f"![x]({POD_PATH})", lambda p: False, reader=pod_image_reader(files))
    ((src, data),) = images.items()
    assert data == PNG and content == f"![x]({src})"


@pytest.mark.parametrize("link_at", ["folder", "file"])
def test_a_link_anywhere_in_the_path_is_refused_and_the_error_names_the_pod_path(tmp_path, link_at):
    project, conv, files = _pod(tmp_path)
    elsewhere = tmp_path / "other-org"
    elsewhere.mkdir()
    (elsewhere / "banner.png").write_bytes(PNG)
    attachments = project / "conversations" / str(conv) / "attachments"
    attachments.mkdir(parents=True)
    if link_at == "folder":
        os.symlink(elsewhere, attachments / "f1")
    else:
        (attachments / "f1").mkdir()
        os.symlink(elsewhere / "banner.png", attachments / "f1" / "banner.png")
    with pytest.raises(DocError) as err:
        localize_images(f"![x]({POD_PATH})", lambda p: True, reader=pod_image_reader(files))
    assert POD_PATH in err.value.message
    assert str(tmp_path) not in err.value.message


def test_without_the_pods_files_every_local_image_is_refused():
    with pytest.raises(DocError):
        localize_images("![x](/workspace/a.png)", lambda p: True, reader=pod_image_reader(None))


@pytest.mark.parametrize("size, ok", [(MAX_IMAGE_BYTES, True), (MAX_IMAGE_BYTES + 1, False)])
def test_an_image_over_the_size_cap_is_refused(tmp_path, size, ok):
    project, conv, files = _pod(tmp_path)
    upload = project / "conversations" / str(conv) / "attachments" / "f1" / "banner.png"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(PNG + b"\0" * (size - len(PNG)))
    read = pod_image_reader(files)
    if ok:
        assert len(read(POD_PATH)) == size
    else:
        with pytest.raises(OSError):
            read(POD_PATH)

