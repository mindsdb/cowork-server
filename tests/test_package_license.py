"""The package metadata declares the license the LICENSE file grants."""

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_declares_the_license_the_license_file_grants():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    license_text = (ROOT / "LICENSE").read_text()

    assert license_text.lstrip().startswith("GNU AFFERO GENERAL PUBLIC LICENSE")
    assert "Version 3, 19 November 2007" in license_text
    assert project["license"] == "AGPL-3.0-only"
