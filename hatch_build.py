"""Fetch the pdf.js files of the PDF viewer before a wheel is built.

pdf.js is not committed (cowork/services/pdf_viewer/fetch_pdfjs.py). This hook
runs it for every wheel, editable ones included, so `uv sync`, `uv build` and
installs from git all get the files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PdfjsBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        path = Path(self.root) / "cowork" / "services" / "pdf_viewer" / "fetch_pdfjs.py"
        # Loaded by path: importing the cowork package needs its dependencies.
        spec = importlib.util.spec_from_file_location("_cowork_fetch_pdfjs", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.fetch()
