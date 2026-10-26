"""Images on this computer, copied beside the doc (cowork.docs.images)."""
from __future__ import annotations

import pytest

from cowork.docs.images import image_kind, localize_images, store_image, within, write_images
from cowork.docs.model import DocError

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d4944415478da63f8cfc0f01f0005000201a0d5e1b70000000049454e44ae426082")


def test_only_real_images_are_taken():
    assert image_kind(PNG) == "png"
    assert image_kind(b"\xff\xd8\xff\xe0rest") == "jpg"
    assert image_kind(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert image_kind(b"<svg onload=alert(1)>") is None
    assert image_kind(b"%PDF-1.7") is None


def test_an_attached_image_is_copied_beside_the_doc_and_markdown_points_at_it(tmp_path):
    """Found in QA: the agent wrote ![banner](/abs/path/upload.png) and the doc
    kept the line as its literal text, while the reply said the image was added."""
    upload = tmp_path / "files" / "u1" / "banner.png"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(PNG)
    md, files = localize_images(f'![Support team]({upload} "Support team")\n\nText.', within(upload))
    (src,) = files
    assert src.startswith("images/") and src.endswith(".png")
    assert md == f'![Support team]({src} "Support team")\n\nText.'
    folder = tmp_path / "doc"
    write_images(folder, files)
    assert (folder / src).read_bytes() == PNG


def test_a_local_image_it_may_not_use_is_refused_with_a_way_forward(tmp_path):
    secret = tmp_path / "home" / "id.png"
    secret.parent.mkdir()
    secret.write_bytes(PNG)
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(DocError, match="can't be used. Attached images and images in the project can be"):
        localize_images([{"op": "insert", "markdown": f"![x]({secret})"}], within(project))
    with pytest.raises(DocError):
        localize_images([{"op": "update", "id": "b_1", "set": {"src": str(secret)}}], within(project))
    not_an_image = project / "notes.png"
    not_an_image.write_bytes(b"plain text")
    with pytest.raises(DocError):
        localize_images("![n](" + str(not_an_image) + ")", within(project))


def test_web_inline_and_beside_the_doc_images_pass_untouched(tmp_path):
    ops = [{"op": "insert", "markdown": "![a](https://example.com/a.png)\n\n![b](images/b.png)"},
           {"op": "update", "id": "b_1", "set": {"src": "data:image/png;base64,AAAA"}}]
    out, files = localize_images(ops, within(tmp_path))
    assert out == ops and files == {}


def test_a_stored_image_is_named_by_its_content(tmp_path):
    a = store_image(tmp_path, PNG)
    assert store_image(tmp_path, PNG) == a
    with pytest.raises(DocError, match="only PNG, JPEG, GIF and WebP"):
        store_image(tmp_path, b"GIF8 nope")
