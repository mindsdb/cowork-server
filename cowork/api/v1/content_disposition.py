"""``Content-Disposition`` for routes that send a file to be saved."""
from __future__ import annotations

import unicodedata
from urllib.parse import quote

_BIDI_CONTROLS = frozenset({"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"})


def _keep(ch: str) -> bool:
    # Cc: CR/LF would inject a header line. Cs: lone surrogates stand for
    # undecodable bytes of an on-disk name and make `quote()` raise. Bidi
    # controls can make a save dialog show a fake extension. Everything else,
    # including non-ASCII spaces and ZWJ/ZWNJ, is safe once percent-encoded.
    return (
        unicodedata.category(ch) not in ("Cc", "Cs")
        and unicodedata.bidirectional(ch) not in _BIDI_CONTROLS
    )


def attachment_disposition(filename: str) -> str:
    """``Content-Disposition`` for saving ``filename``, safe to put on the wire.

    The name comes from a request path or from the file system. Path
    validation rejects NUL, separators and ``..``, but a quote, CR or LF still
    pass, and any of them interpolated raw into a header is a header-injection
    vector. A name outside Latin-1 cannot be put in a header at all: Starlette
    encodes header values as Latin-1.

    Two spellings so every client gets a usable name: an ASCII ``filename=``
    with quotes and backslashes escaped, and an RFC 5987 ``filename*=UTF-8''…``
    carrying the name percent-encoded, which is what current browsers read
    first.
    """
    cleaned = "".join(ch for ch in filename if _keep(ch))
    ascii_name = cleaned.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace("\\", "\\\\").replace('"', '\\"').strip() or "download"
    # A fully non-ASCII stem leaves only the extension(s) behind ("报告.xlsx"
    # -> ".xlsx", "报告.tar.gz" -> ".tar.gz"), a hidden dotfile on macOS/Linux.
    # Prefixing rather than splitting on the last dot keeps multi-extension
    # names whole. A name that was a dotfile to begin with stays one. Distinct
    # names can still degrade to the same "download.xlsx"; the ASCII spelling
    # is a fallback and `filename*` carries the exact name.
    # Check the original name: "\udcff.pdf" cleans to ".pdf" but was not a dotfile.
    if ascii_name.startswith(".") and not filename.startswith("."):
        ascii_name = "download" + ascii_name
    return (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(cleaned or 'download', safe='')}"
    )
