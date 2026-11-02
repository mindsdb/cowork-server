"""Links a cell could leave in a project's storage, and the serve route's
warning for a linked folder, for the tests that check no route serves
through them."""

import json
import logging
import shutil
from pathlib import Path

from pydantic import BaseModel

# A project's saved secret, which no artifact route may serve.
SECRET = "SECRET_TOKEN=do-not-serve-7f3a"
REFUSED_LINKED_FOLDER = "Refused a linked artifacts folder"


class LoggedRefusal(BaseModel):
    """The fields of a refused linked folder's warning that the deployment's
    JSON log keeps."""

    event: str | None = None
    reason: str | None = None
    project_id: str | None = None


def replace_with_link(path: Path, target: Path) -> None:
    """Replace a folder with a symlink, as a pre-sandbox cell could."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


def linked_folder_refusals(records: list[logging.LogRecord]) -> list[LoggedRefusal]:
    """Each refused linked folder's warning in ``records``, as the deployment's
    JSON log prints it, so a field that log drops is missing here too."""
    from cowork.common.structured_logging import make_formatter

    formatter = make_formatter()
    return [
        LoggedRefusal.model_validate(json.loads(formatter.format(record)))
        for record in records
        if record.getMessage() == REFUSED_LINKED_FOLDER
    ]
