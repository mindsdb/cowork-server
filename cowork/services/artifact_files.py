"""Open a file inside an artifact folder without following anything out of it."""
from cowork.common.paths import PinnedDir


def read_artifact_file(*, folder: PinnedDir, parts: tuple[str, ...], max_bytes: int) -> bytes:
    raise NotImplementedError
