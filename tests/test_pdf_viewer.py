"""The PDF viewer folder copied from mindshub_services (cowork/services/pdf_viewer).

manifest.txt lists every file with its sha256; MANIFEST_SHA256 pins
manifest.txt itself and equals the constant in mindshub_services's
tests/test_pdf_viewer.py. Refresh the copy with that repository's
scripts/vendor_pdf_viewer.sh, then copy the folder here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from cowork.services import pdf_viewer

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "cowork" / "services" / "pdf_viewer"
MANIFEST_SHA256 = "b40e62a7979eca43a81db943608b59096658cdd43bec20acb1752b5e50bfe275"


def test_manifest_lists_every_file_with_its_sha256():
    found = {}
    for path in FOLDER.rglob("*"):
        rel = path.relative_to(FOLDER)
        if path.is_file() and "__pycache__" not in rel.parts and rel.as_posix() != pdf_viewer.MANIFEST_NAME:
            found[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert found == pdf_viewer.FILES


def test_manifest_is_the_one_pinned_in_both_repositories():
    assert pdf_viewer.manifest_digest() == MANIFEST_SHA256


def test_the_folder_is_kept_byte_for_byte_by_git():
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "cowork/services/pdf_viewer/** -text" in lines


def test_the_folder_is_not_ignored():
    # Both repositories' .gitignore ignore a file named MANIFEST; ours is manifest.txt.
    assert pdf_viewer.MANIFEST_NAME == "manifest.txt"
