"""Headers for responses a person saves as a file."""

from __future__ import annotations

from urllib.parse import quote


def attachment_disposition(filename: str) -> str:
    """``Content-Disposition`` for saving ``filename``, safe to put on the wire.

    The name is a request-derived path component. ``relative_file_parts``
    rejects NUL, separators and ``..``, but a quote, CR or LF still pass — and
    any of them interpolated raw into a header is a header-injection vector
    (``project_files.download_project_file`` has exactly that shape; ENG-2044).

    Two spellings so every client gets a usable name: an ASCII ``filename=``
    with quotes and backslashes escaped and non-printables dropped, and an
    RFC 5987 ``filename*=UTF-8''…`` carrying the exact name percent-encoded,
    which is what current browsers read first.
    """
    cleaned = "".join(ch for ch in filename if ch.isprintable())
    ascii_name = cleaned.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace("\\", "\\\\").replace('"', '\\"').strip() or "download"
    # A fully non-ASCII stem leaves only the extension(s) behind — "报告.xlsx"
    # -> ".xlsx", "报告.tar.gz" -> ".tar.gz" — a HIDDEN dotfile on macOS/Linux.
    # Prefix rather than rpartition: splitting on the last dot rescued only
    # single-extension names (review pass 2 on #413). Distinct artifacts can
    # still degrade to the same "download.xlsx" — acceptable, since the ASCII
    # spelling is a fallback and `filename*` below carries the exact name.
    if ascii_name.startswith("."):
        ascii_name = "download" + ascii_name
    return (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(cleaned or 'download', safe='')}"
    )
