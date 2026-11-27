"""Syntax highlighting for a doc's code blocks on its page and in its PDF.

Pygments splits the code; each token gets the highlight.js class the editor
(cowork, components/markdown/hljs.js) gives it, so the page takes the editor's
palette. The two tokenizers differ in smaller tokens; the main ones are pinned
by tests/fixtures/docs_code_highlight/contract.json (``anchors``).
"""
from __future__ import annotations

import html
import re
from functools import lru_cache

from pygments.lexers import get_lexer_by_name
from pygments.token import Comment, Generic, Keyword, Literal, Name, Number, Operator, String, _TokenType

# Over this many code points a block stays plain text, in the editor too.
MAX_CHARS = 10_000

# Each spelling the editor highlights: its canonical name and Pygments lexer.
LANGUAGES: dict[str, tuple[str, str]] = {
    "bash": ("bash", "bash"), "sh": ("bash", "bash"),
    "shell": ("shell", "console"), "zsh": ("shell", "console"),
    "css": ("css", "css"),
    "diff": ("diff", "diff"), "patch": ("diff", "diff"),
    "dockerfile": ("dockerfile", "docker"), "docker": ("dockerfile", "docker"),
    "go": ("go", "go"), "golang": ("go", "go"),
    "javascript": ("javascript", "javascript"), "js": ("javascript", "javascript"), "jsx": ("javascript", "jsx"),
    "json": ("json", "json"),
    "markdown": ("markdown", "markdown"), "md": ("markdown", "markdown"),
    "python": ("python", "python"), "py": ("python", "python"),
    "rust": ("rust", "rust"), "rs": ("rust", "rust"),
    "sql": ("sql", "sql"),
    "typescript": ("typescript", "typescript"), "ts": ("typescript", "typescript"), "tsx": ("typescript", "tsx"),
    "xml": ("xml", "xml"), "html": ("html", "html"),
    "yaml": ("yaml", "yaml"), "yml": ("yaml", "yaml"),
}


def _palette(groups: list[tuple[str, str]]) -> dict[str, str]:
    return {cls: value for names, value in groups for cls in names.split()}


# The editor's palette (cowork styles/globals.css); `*-bg` is a background.
COLORS: dict[str, dict[str, str]] = {
    "dark": _palette([
        ("keyword selector-tag literal doctag", "rgb(196, 144, 255)"),
        ("string regexp attr symbol template-tag template-variable", "rgb(125, 226, 184)"),
        ("number built_in builtin-name", "rgb(246, 199, 121)"),
        ("comment quote", "var(--ink-4)"), ("title section", "var(--ink)"),
        ("name tag meta variable bullet", "rgb(123, 178, 255)"), ("property params", "var(--ink-2)"),
        ("deletion", "rgb(255, 122, 145)"), ("deletion-bg", "rgba(255, 122, 145, 0.12)"),
        ("addition", "rgb(125, 226, 184)"), ("addition-bg", "rgba(125, 226, 184, 0.12)"),
    ]),
    "light": _palette([
        ("keyword selector-tag literal doctag", "rgb(124, 58, 175)"),
        ("string regexp attr symbol template-tag template-variable", "rgb(20, 121, 79)"),
        ("number built_in builtin-name", "rgb(170, 100, 0)"),
        ("comment quote", "var(--ink-4)"), ("title section", "rgb(17, 85, 98)"),
        ("name tag meta variable bullet", "rgb(28, 76, 169)"), ("property params", "var(--ink-2)"),
        ("deletion", "rgb(180, 35, 60)"), ("deletion-bg", "rgba(180, 35, 60, 0.12)"),
        ("addition", "rgb(20, 121, 79)"), ("addition-bg", "rgba(20, 121, 79, 0.12)"),
    ]),
}

# Token type -> hljs class, the closest listed ancestor winning; "" is no class.
_BASE: list[tuple[_TokenType, str]] = [
    (Keyword.Constant, "literal"), (Keyword.Type, ""), (Keyword, "keyword"),
    (Operator.Word, "keyword"), (Generic.Prompt, "meta"),
    (Name.Builtin, "built_in"), (Name.Decorator, "meta"), (Name.Function, "title"), (Name.Class, "title"),
    (Name.Tag, "name"), (Name.Attribute, "attr"), (Name.Variable, "variable"),
    (String, "string"), (Number, "number"),
    (Comment.Preproc, "meta"), (Comment, "comment"),
    (Generic.Heading, "section"), (Generic.Subheading, "section"),
    (Generic.Inserted, "addition"), (Generic.Deleted, "deletion"),
]
# Per lexer, ahead of the base: where hljs marks a token otherwise.
_BY_LEXER: dict[str, list[tuple[_TokenType, str]]] = {
    "json": [(Name.Tag, "attr"), (Keyword.Constant, "keyword")],
    "yaml": [(Name.Tag, "attr"), (Name.Constant, "string")],  # Name.Constant: a block scalar's body
    "diff": [(Generic.Heading, "comment"), (Generic.Subheading, "meta")],
    "css": [(Keyword.Constant, ""), (Keyword, "attribute"), (Name.Tag, "selector-tag"), (Name.Class, "selector-class")],
    "typescript": [(Keyword.Type, "built_in")],
    "tsx": [(Keyword.Type, "built_in")],
    "rust": [(String.Doc, "comment")],
    # Inline code and fenced blocks: hljs leaves them uncoloured as `code`.
    "markdown": [(String, "code")],
}
# Lexer options. Markdown would hand a fenced block to the lexer its fence
# names, any of Pygments' hundreds (some take seconds on a 10,000-character
# block); the editor doesn't colour fenced code either.
_OPTIONS: dict[str, dict[str, bool]] = {"markdown": {"handlecodeblocks": False}}
_YAML_NUMBER = re.compile(r"[-+]?\d[\d_]*(\.\d+)?([eE][-+]?\d+)?")


def _class(lexer: str, ttype: _TokenType, value: str, prev: tuple[_TokenType, str] | None) -> str:
    if lexer == "yaml" and ttype in Literal.Scalar.Plain:
        if _YAML_NUMBER.fullmatch(value):
            return "number"
        return "literal" if value in ("true", "false", "null") else "string"
    if lexer == "diff" and value.startswith(("---", "+++")):
        return "comment"
    if lexer == "css" and prev is not None:
        if ttype in Keyword and prev[1] == "@":
            return "keyword"  # an at-rule's name
        if ttype in Keyword.Type and prev[0] in Number:
            return "number"  # a unit right after its number
    for parent, cls in _BY_LEXER.get(lexer, []) + _BASE:
        if ttype in parent:
            return cls
    return ""


@lru_cache(maxsize=128)
def _render(lexer: str, code: str) -> str | None:
    """The page re-renders on every change to a doc, so a block is split once per text."""
    runs: list[list[str]] = []  # [class, text], neighbours of one class merged
    prev = None
    # A session lexer (console) drops a last line that has no newline.
    added = not code.endswith("\n")
    source = code + "\n" if added else code
    split = get_lexer_by_name(lexer, stripnl=False, ensurenl=False, **_OPTIONS.get(lexer, {}))
    for ttype, value in split.get_tokens(source):
        cls = _class(lexer, ttype, value, prev)
        if runs and runs[-1][0] == cls:
            runs[-1][1] += value
        else:
            runs.append([cls, value])
        if value.strip():
            prev = (ttype, value)
    if added and runs and runs[-1][1].endswith("\n"):
        runs[-1][1] = runs[-1][1][:-1]
        if not runs[-1][1]:
            runs.pop()
    if "".join(text for _, text in runs) != code:
        return None  # Pygments rewrote it (\r\n, a BOM): plain text, as the editor shows it
    return "".join(f'<span class="hljs-{cls}">{html.escape(text, quote=False)}</span>' if cls
                   else html.escape(text, quote=False) for cls, text in runs)


def highlight(code: str, language: str | None) -> tuple[str, str] | None:
    """``(name, markup)`` for ``code`` in ``language``, or None to show it as plain text."""
    entry = LANGUAGES.get((language or "").strip().lower())
    if entry is None or len(code) > MAX_CHARS:
        return None
    markup = _render(entry[1], code)
    return None if markup is None else (entry[0], markup)


def code_css(theme: str) -> str:
    """The palette of ``theme`` as CSS rules for the classes ``highlight`` writes."""
    return "".join(f".hljs-{cls.removesuffix('-bg')}{{{'background' if cls.endswith('-bg') else 'color'}:{value}}}"
                   for cls, value in COLORS[theme].items())
