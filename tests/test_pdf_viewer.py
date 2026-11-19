"""The PDF viewer folder copied from mindshub_services (cowork/services/pdf_viewer).

SOURCE_SHA256 pins source_digest() and equals the constant in
mindshub_services's tests/test_pdf_viewer.py; copy the folder from there after
a change. That file also tests the folder's own behaviour. pdf.js is not
committed: hatch_build.py fetches it for every wheel, `uv sync` included.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from cowork.services import pdf_viewer
from cowork.services.pdf_viewer import fetch_pdfjs

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "cowork" / "services" / "pdf_viewer"
SOURCE_SHA256 = "b5f9b95add1d45fff240fcf078dab4fd87dc54b7ccaa589140c6d82ed58d9015"


def test_the_source_is_the_one_pinned_in_both_repositories():
    assert pdf_viewer.source_digest() == SOURCE_SHA256


def test_pdfjs_is_fetched():
    assert fetch_pdfjs.is_current(pdf_viewer.PDFJS_DIR) and pdf_viewer.PDFJS_FILES, (
        "run python cowork/services/pdf_viewer/fetch_pdfjs.py"
    )
    assert pdf_viewer.asset("pdf.min.mjs")[1] == "text/javascript"


def test_pdfjs_is_not_committed_but_goes_into_the_wheel():
    assert "/pdfjs/" in (FOLDER / ".gitignore").read_text(encoding="utf-8").splitlines()
    hatch = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["hatch"]
    wheel = hatch["build"]["targets"]["wheel"]
    assert "cowork/services/pdf_viewer/pdfjs/" in wheel["artifacts"]
    assert "custom" in wheel["hooks"]
    assert (ROOT / "hatch_build.py").is_file()


def test_the_folder_is_kept_byte_for_byte_by_git():
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "cowork/services/pdf_viewer/** -text" in lines
