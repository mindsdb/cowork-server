import os
from pathlib import Path


def is_relative_to(base: Path, path: Path) -> bool:
    """True if `path` is `base` or lives inside it.

    Both paths must already be resolved (absolute, symlinks followed) by the
    caller — this only does the containment comparison, using the
    `os.path.normpath` + `str.startswith` idiom that static analyzers
    recognize as a path-traversal sanitizer.
    """
    base_normalized = os.path.normpath(str(base))
    path_normalized = os.path.normpath(str(path))
    return path_normalized == base_normalized or path_normalized.startswith(base_normalized + os.sep)
