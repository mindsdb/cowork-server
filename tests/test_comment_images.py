"""Images in artifact comments (ENG-4410): stored in the artifact's journal,
read back by whoever may review it, and only ever served as an image."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Imported at module scope on purpose: cowork.server builds an app at import
# time (see tests/test_trusted_identity_local.py).
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.services import comment_images
from cowork.services.artifact_identity import ensure_full_id
from cowork.services.comment_images import (
    ImageRefused,
    comment_image_ids,
    decode_request_image,
    read_comment_image,
    save_comment_image,
)
from tests._live_apps import (
    BEARER_TOKEN,
    PARTITION_A,
    SUB_A,
    A,
    B,
    clear_live_app_rows,
    general_project,
    person_scope,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'


# ── the store ─────────────────────────────────────────────────────────────


def test_an_image_round_trips_through_the_artifact_journal(tmp_path):
    stored = save_comment_image(tmp_path, PNG)

    assert comment_images.IMAGE_ID.match(stored["id"])
    assert stored["id"].endswith(".png")
    assert (stored["mediaType"], stored["size"]) == ("image/png", len(PNG))
    assert (tmp_path / ".revisions" / "comment-images" / stored["id"]).read_bytes() == PNG
    assert read_comment_image(tmp_path, stored["id"]) == (PNG, "image/png")


@pytest.mark.parametrize("data", [SVG, b"<!doctype html><script>1</script>", b"GIF8", b""])
def test_anything_but_a_raster_image_is_refused(tmp_path, data):
    with pytest.raises(ImageRefused) as refused:
        save_comment_image(tmp_path, data)

    assert refused.value.reason == "unsupported"
    assert not (tmp_path / ".revisions" / "comment-images").exists()


def test_an_image_past_the_cap_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(comment_images, "MAX_IMAGE_BYTES", len(PNG) - 1)

    with pytest.raises(ImageRefused) as refused:
        save_comment_image(tmp_path, PNG)

    assert refused.value.reason == "too_large"


def test_a_file_planted_under_an_image_id_is_not_served(tmp_path):
    """The artifact's own code can write the folder: what stands at an id must
    be the image type the id names."""
    image_id = "a" * 32 + ".png"
    folder = tmp_path / ".revisions" / "comment-images"
    folder.mkdir(parents=True)
    (folder / image_id).write_bytes(SVG)

    with pytest.raises(FileNotFoundError):
        read_comment_image(tmp_path, image_id)
    with pytest.raises(FileNotFoundError):
        read_comment_image(tmp_path, "../metadata.json")


def test_a_request_image_is_checked_like_a_stored_one():
    data = base64.b64encode(JPEG).decode()

    assert decode_request_image(data) == (data, "image/jpeg")
    with pytest.raises(ImageRefused):
        decode_request_image(base64.b64encode(SVG).decode())
    with pytest.raises(ImageRefused):
        decode_request_image("not base64!")


def test_a_comments_image_ids_are_read_from_its_markers():
    first, second = "1" * 32 + ".png", "2" * 32 + ".jpg"
    text = f"Look\n\n![a.png](cowork-image:{first})\n![b](cowork-image:{second})\n![a.png](cowork-image:{first})"

    assert comment_image_ids(text) == [first, second]
    assert comment_image_ids("![x](https://example.com/a.png)") == []


# ── the routes ────────────────────────────────────────────────────────────


@pytest.fixture
def deployment(personal_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", BEARER_TOKEN)
    get_app_settings.cache_clear()
    yield
    clear_live_app_rows()
    get_app_settings.cache_clear()


@pytest.fixture
def client(deployment) -> TestClient:
    get_app_settings.cache_clear()
    return TestClient(cowork_server.create_app())


@pytest.fixture
def draft(deployment) -> tuple[str, str, Path]:
    """Tester A's draft artifact in A's own `general` project."""
    project_id, project_path = general_project(person_scope(PARTITION_A, SUB_A))
    folder = Path(project_path) / ".anton" / "artifacts" / "image-test"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html><body><h1>Image test</h1></body></html>")
    (folder / "metadata.json").write_text(
        json.dumps({"slug": "image-test", "name": "image test", "type": "html-app"})
    )
    artifact_id, _metadata = ensure_full_id(folder)
    return str(project_id), artifact_id, folder


def _images(project_id: str, artifact_id: str) -> str:
    return f"/api/v1/artifacts/workspace/{project_id}/{artifact_id}/comment-images"


def test_a_pasted_image_is_stored_and_served_only_as_an_image(client, draft):
    project_id, artifact_id, folder = draft

    # The upload's own type is not believed: the bytes say PNG.
    uploaded = client.post(
        _images(project_id, artifact_id), headers=A, files={"file": ("shot.html", PNG, "text/html")},
    )
    assert uploaded.status_code == 200, uploaded.text
    image_id = uploaded.json()["id"]
    assert uploaded.json()["mediaType"] == "image/png"

    served = client.get(f"{_images(project_id, artifact_id)}/{image_id}", headers=A)
    assert served.status_code == 200
    assert served.content == PNG
    assert served.headers["content-type"] == "image/png"
    assert served.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in served.headers["content-security-policy"]


def test_an_svg_or_an_oversized_image_is_refused(client, draft, monkeypatch):
    project_id, artifact_id, folder = draft

    svg = client.post(
        _images(project_id, artifact_id), headers=A, files={"file": ("x.svg", SVG, "image/svg+xml")},
    )
    from cowork.api.v1.endpoints import artifact_comment_images

    monkeypatch.setattr(comment_images, "MAX_IMAGE_BYTES", 16)
    monkeypatch.setattr(artifact_comment_images, "MAX_IMAGE_BYTES", 16)
    large = client.post(
        _images(project_id, artifact_id), headers=A, files={"file": ("big.png", PNG, "image/png")},
    )

    assert (svg.status_code, large.status_code) == (415, 413)
    assert not (folder / ".revisions" / "comment-images").exists()


def test_another_person_can_neither_add_nor_read_an_image(client, draft):
    project_id, artifact_id, folder = draft
    image_id = client.post(
        _images(project_id, artifact_id), headers=A, files={"file": ("a.png", PNG, "image/png")},
    ).json()["id"]

    answers = [
        client.post(_images(project_id, artifact_id), headers=B, files={"file": ("b.png", PNG, "image/png")}),
        client.get(f"{_images(project_id, artifact_id)}/{image_id}", headers=B),
    ]

    assert [answer.status_code for answer in answers] == [404, 404]
    assert [p.name for p in (folder / ".revisions" / "comment-images").iterdir()] == [image_id]


def test_an_unknown_image_is_404(client, draft):
    project_id, artifact_id, _folder = draft

    missing = client.get(f"{_images(project_id, artifact_id)}/{'f' * 32}.png", headers=A)
    malformed = client.get(f"{_images(project_id, artifact_id)}/metadata.json", headers=A)

    assert (missing.status_code, malformed.status_code) == (404, 404)
