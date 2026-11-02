"""Links a pre-sandbox cell could leave in a project's storage, for the tests
that check no route serves through them."""

import shutil
from pathlib import Path

# A project's saved secret, which no artifact route may serve.
SECRET = "SECRET_TOKEN=do-not-serve-7f3a"


def replace_with_link(path: Path, target: Path) -> None:
    """Replace a folder with a symlink, as a pre-sandbox cell could."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)
