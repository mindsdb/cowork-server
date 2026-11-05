"""``Content-Disposition`` for routes that send a file to be saved."""
from __future__ import annotations

import unicodedata
from urllib.parse import quote

_BIDI_CONTROLS = frozenset({"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"})
# ALM, ZWSP, LRM, RLM, word joiner, BOM.
_INVISIBLE = frozenset("\u061c\u200b\u200e\u200f\u2060\ufeff")


def _keep(ch: str) -> bool:
    # Lone surrogates (undecodable on-disk bytes) make `quote()` raise; bidi
    # controls and invisible marks can disguise a name. Everything else,
    # including ZWJ/ZWNJ, is safe once percent-encoded.
    return (
        unicodedata.category(ch) not in ("Cc", "Cs", "Zl", "Zp")
        and ch not in _INVISIBLE
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
    # NFKD keeps "cafe" from "café", but also turns a fullwidth solidus into "/".
    ascii_name = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace("/", "_").replace("\\", "\\\\").replace('"', '\\"').strip()
    # A non-ASCII stem leaves a hidden ".xlsx"; a real dotfile such as ".env" stays one.
    if ascii_name.startswith(".") and not filename.startswith("."):
        ascii_name = "download" + ascii_name
    # Empty, or only dots and spaces, which Windows trims down to a directory name.
    if not ascii_name.strip(". "):
        ascii_name = "download"
    return (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(cleaned or 'download', safe='')}"
    )
