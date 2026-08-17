"""compute_publish_md5 over an explicitly given artifacts root.

The org deployment keeps projects under `<root>/<org_id>/<project>`, which the
module-level FS scan never finds. Every resolver on the publish path therefore
takes its container root as an argument instead of discovering it.
"""
from __future__ import annotations

import json

from cowork.services.publish import compute_publish_md5


def _make_artifact(base, slug, *, files: dict[str, str], meta: dict):
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


def test_md5_resolves_from_given_root_outside_any_registered_project(tmp_path):
    # Deliberately NOT under the projects root the module scan knows about —
    # this is what an org-mode `<root>/<org_id>/<project>` path looks like to
    # the scanner.
    base = tmp_path / "some-org-id" / "proj" / ".anton" / "artifacts"
    folder = _make_artifact(
        base, "rep",
        files={"report.html": "<html>v1</html>"},
        meta={"slug": "rep", "name": "Rep", "type": "html-app"},
    )

    digest = compute_publish_md5(folder, artifacts_base=base)

    assert isinstance(digest, str) and len(digest) == 32


def test_md5_is_stable_for_identical_content(tmp_path):
    base = tmp_path / "a" / ".anton" / "artifacts"
    folder = _make_artifact(base, "rep", files={"report.html": "<html>v1</html>"},
                            meta={"slug": "rep", "type": "html-app"})

    first = compute_publish_md5(folder, artifacts_base=base)
    second = compute_publish_md5(folder, artifacts_base=base)

    assert first == second


def test_md5_changes_when_content_changes(tmp_path):
    base = tmp_path / "a" / ".anton" / "artifacts"
    folder = _make_artifact(base, "rep", files={"report.html": "<html>v1</html>"},
                            meta={"slug": "rep", "type": "html-app"})
    before = compute_publish_md5(folder, artifacts_base=base)

    (folder / "report.html").write_text("<html>v2</html>")
    after = compute_publish_md5(folder, artifacts_base=base)

    assert before != after


def test_md5_none_for_unpublishable_primary(tmp_path):
    base = tmp_path / "a" / ".anton" / "artifacts"
    folder = _make_artifact(base, "data", files={"rows.csv": "a,b\n1,2"},
                            meta={"slug": "data", "type": "dataset"})

    assert compute_publish_md5(folder, artifacts_base=base) is None
