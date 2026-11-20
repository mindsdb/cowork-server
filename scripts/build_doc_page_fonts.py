"""Build the Inter subsets a doc's page (cowork/docs/render.py) embeds.

The editor draws a doc in the app's own Inter (cowork, ``src/renderer/public/fonts``);
the page ships the same font so a paragraph wraps at the same words for a
reader without Inter installed. The page carries the font inline, so it is cut
to what a doc's text needs: Latin-1, Latin Extended-A, basic Cyrillic and
common punctuation, weights 400-700, with the optical-size axis kept (the
app sets ``font-optical-sizing: auto``). A character outside the subset falls
back to the next font in the page's stack.

Run from the repository root, with the app's fonts as input:

    uv run --no-project --with fonttools --with brotli \\
        python scripts/build_doc_page_fonts.py <cowork>/src/renderer/public/fonts
"""
from __future__ import annotations

import sys
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

OUT = Path(__file__).resolve().parent.parent / "cowork" / "docs" / "page_fonts"
UNICODES = (
    "U+0000-00FF,U+0100-017F,U+0131,U+0152-0153,U+0218-021B,U+02C6,U+02DA,U+02DC,"
    "U+0400-045F,U+0490-0491,U+2000-206F,U+20AC,U+20BD,U+2122,U+2190-2193,U+2212,U+FFFD"
)
LAYOUT_FEATURES = ["kern", "liga", "calt", "ccmp", "locl", "mark", "mkmk", "tnum", "case"]
FILES = {"InterVariable.woff2": "inter.woff2", "InterVariable-Italic.woff2": "inter-italic.woff2"}


def build(source: Path, target: Path) -> None:
    font = TTFont(source)
    options = subset.Options()
    options.layout_features = LAYOUT_FEATURES
    options.name_IDs = ["*"]  # keeps the copyright and the OFL notice
    options.notdef_outline = True
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=subset.parse_unicodes(UNICODES))
    subsetter.subset(font)
    # Subset first: instancing the full font trips over glyphs the subset drops.
    font = instancer.instantiateVariableFont(font, {"wght": (400, 700), "opsz": (14, 32)})
    font.flavor = "woff2"
    font.save(target)
    print(f"{target.name}: {target.stat().st_size} bytes")


def main() -> None:
    fonts = Path(sys.argv[1])
    for name, out in FILES.items():
        build(fonts / name, OUT / out)


if __name__ == "__main__":
    main()
