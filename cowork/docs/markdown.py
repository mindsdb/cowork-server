"""Markdown in and out of doc blocks.

The agent authors docs in the Markdown it already writes well; this module
turns that into typed blocks and back. Beyond CommonMark-ish basics it reads:

- ``- [ ]`` / ``- [x]`` task items
- ``> [!NOTE]`` / ``[!TIP]`` / ``[!WARNING]`` / ``[!CAUTION]`` / ``[!IMPORTANT]`` callouts
- pipe tables (column kinds inferred: number, date, status)
- fenced ``chart`` and ``metrics`` blocks holding JSON
- ``{{pending: what comes here}}`` alone on a line: a placeholder the agent
  fills later, shown to the reader as work in progress
- ``![alt](src "caption")`` alone on a line: an image

Inline text stays Markdown (``**bold**``, ``*italic*``, `` `code` ``,
``~~strike~~``, ``[text](url)``) plus chips ``{{date:YYYY-MM-DD}}`` and
``{{status:Label}}``; the renderer draws those.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .model import (
    LIST_TYPES,
    Block,
    DocError,
    image_src_ok,
    is_number,
    new_id,
    normalize_block,
    tone_for,
)

# A closing run of #s is stripped only after a space: "## Learning C#" keeps its #.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_LIST = re.compile(r"^(\s*)([-*+]|\d{1,3}[.)])\s+(?:\[([ xX])\]\s+)?(.*)$")
_FENCE = re.compile(r"^\s*(```+|~~~+)\s*([\w+-]*)\s*(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_DIVIDER = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_CALLOUT = re.compile(r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION|INFO|SUCCESS|DANGER)\]\s*(.*)$", re.I)
_PENDING = re.compile(r"^\s*\{\{\s*pending\s*:\s*(.*?)\s*\}\}\s*$", re.I)
_IMAGE = re.compile(r'^\s*!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+"([^"]*)")?\s*\)\s*$')
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# A callout's `[!MARKER]` → its tone, and back (TIP is written for success).
_TONE_OF_MARKER = {
    "note": "note", "info": "info", "tip": "success", "success": "success",
    "important": "info", "warning": "warning", "caution": "danger", "danger": "danger",
}
_TONE_MARKER = {"note": "NOTE", "info": "INFO", "success": "TIP", "warning": "WARNING", "danger": "CAUTION"}

# A column with one of these names and a few short values is a status column.
STATUS_HEADERS = {
    "status", "state", "stage", "priority", "risk", "decision", "health", "result",
    "outcome", "verdict", "severity", "confidence", "impact", "effort", "phase", "rag",
}


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, buf, i = [], [], 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            buf.append("|")
            i += 2
            continue
        if ch == "|":
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    cells.append("".join(buf).strip())
    return cells


_MAX_STATUS_OPTIONS = 8


def _distinct(values: list[str], *, limit: int) -> list[str]:
    """The values without case-insensitive repeats, first spelling kept; stops past ``limit``."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
            if len(out) > limit:
                break
    return out


def _infer_columns(header: list[str], rows: list[list[str]]) -> list[dict[str, Any]]:
    cols = []
    for ci, name in enumerate(header):
        values = [r[ci].strip() for r in rows if ci < len(r) and r[ci].strip()]
        kind = "text"
        options: list[dict[str, str]] = []
        plain = re.sub(r"[*_`]", "", name).strip().lower()
        if values and all(is_number(re.sub(r"[*_]", "", v)) for v in values):
            kind = "number"
        elif values and all(_DATE.match(v) for v in values):
            kind = "date"
        else:
            distinct = _distinct(values, limit=_MAX_STATUS_OPTIONS)
            short = all(len(v) <= 24 and "](" not in v and "**" not in v for v in distinct)
            if distinct and short and len(distinct) <= _MAX_STATUS_OPTIONS and (
                plain in STATUS_HEADERS
                or (len(values) >= 3 and len(distinct) < len(values)
                    and all(tone_for(v) != "gray" or v.lower() in ("not started", "todo", "planned", "backlog")
                            for v in distinct))
            ):
                kind = "status"
                options = [{"name": v, "tone": tone_for(v)} for v in distinct]
        col: dict[str, Any] = {"id": new_id("c_"), "name": name, "kind": kind}
        if kind == "status":
            col["options"] = options
        cols.append(col)
    return cols


def _parse_json_fence(lang: str, body: str) -> dict[str, Any]:
    try:
        spec = json.loads(body or "{}")
    except json.JSONDecodeError as exc:
        raise DocError("bad_markdown", f"the ```{lang} block is not valid JSON ({exc.msg} at line {exc.lineno}).") from exc
    if lang == "metrics":
        if isinstance(spec, list):
            spec = {"items": spec}
        if not isinstance(spec, dict):
            raise DocError("bad_markdown", "a ```metrics block holds a JSON list of {label, value} tiles.")
        return {"type": "metrics", **spec}
    if not isinstance(spec, dict):
        raise DocError("bad_markdown", "a ```chart block holds one JSON object.")
    if "type" in spec and "chart" not in spec:
        spec["chart"] = spec.pop("type")
    spec.pop("type", None)
    return {"type": "chart", **spec}


# Each reader takes the lines, the index of a non-blank line, and whether a
# paragraph is open there; it returns the blocks it read and the next index, or
# None when the line doesn't start its construct.
_Read = tuple[list[dict[str, Any]], int] | None


def _read_fence(lines: list[str], i: int, in_para: bool) -> _Read:
    fence = _FENCE.match(lines[i])
    if not fence:
        return None
    marker, lang = fence.group(1), (fence.group(2) or "").lower()
    # Closed only by the same character, at least as many, alone on the line,
    # so a block opened with ```` can hold a ``` line.
    closing = re.compile(rf"^\s*{re.escape(marker[0])}{{{len(marker)},}}\s*$")
    body: list[str] = []
    i += 1
    while i < len(lines) and not closing.match(lines[i]):
        body.append(lines[i])
        i += 1
    text = "\n".join(body)
    block = _parse_json_fence(lang, text) if lang in ("chart", "metrics") else {
        "type": "code", "language": lang, "code": text}
    return [block], i + 1  # past the closing fence (or EOF)


def _read_pending(lines: list[str], i: int, in_para: bool) -> _Read:
    pending = _PENDING.match(lines[i])
    return ([{"type": "pending", "intent": pending.group(1)}], i + 1) if pending else None


def _read_heading(lines: list[str], i: int, in_para: bool) -> _Read:
    heading = _HEADING.match(lines[i])
    if not heading:
        return None
    return [{"type": "heading", "level": min(3, len(heading.group(1))), "text": heading.group(2)}], i + 1


def _read_divider(lines: list[str], i: int, in_para: bool) -> _Read:
    # Under a paragraph, `---` is that paragraph's text (a setext heading elsewhere).
    return ([{"type": "divider"}], i + 1) if not in_para and _DIVIDER.match(lines[i]) else None


def _read_image(lines: list[str], i: int, in_para: bool) -> _Read:
    image = _IMAGE.match(lines[i])
    if not image or not image_src_ok(image.group(2)):
        return None
    img: dict[str, Any] = {"type": "image", "alt": image.group(1), "src": image.group(2)}
    if image.group(3):
        img["caption"] = image.group(3)
    return [img], i + 1


def _read_table(lines: list[str], i: int, in_para: bool) -> _Read:
    if not (lines[i].strip().startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1])):
        return None
    header = _split_row(lines[i])
    i += 2
    rows: list[list[str]] = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append((_split_row(lines[i]) + [""] * len(header))[: len(header)])
        i += 1
    cols = _infer_columns(header, rows)
    return [{
        "type": "table",
        "columns": cols,
        "rows": [{"id": new_id("r_"), "cells": {c["id"]: r[ci] for ci, c in enumerate(cols)}} for r in rows],
    }], i


def _read_quote(lines: list[str], i: int, in_para: bool) -> _Read:
    if not lines[i].strip().startswith(">"):
        return None
    quoted: list[str] = []
    while i < len(lines) and lines[i].strip().startswith(">"):
        quoted.append(re.sub(r"^\s*>\s?", "", lines[i]))
        i += 1
    callout = _CALLOUT.match(quoted[0].strip())
    if not callout:
        return [{"type": "quote", "text": " ".join(q.strip() for q in quoted if q.strip())}], i
    body_text = " ".join(q.strip() for q in quoted[1:] if q.strip())
    block: dict[str, Any] = {"type": "callout", "tone": _TONE_OF_MARKER[callout.group(1).lower()], "text": body_text}
    title = callout.group(2).strip()
    if title and body_text:
        block["title"] = title
    elif title:
        block["text"] = title
    return [block], i


def _read_list_item(lines: list[str], i: int, in_para: bool) -> _Read:
    item = _LIST.match(lines[i])
    # A numbered line inside a paragraph ("in 2026. Then…") is that paragraph's text.
    if not item or (in_para and item.group(2)[0].isdigit()):
        return None
    indent_spaces, marker, check, text = item.groups()
    i += 1
    # continuation lines: indented further, not a new item or block
    while i < len(lines) and lines[i].strip() and not _LIST.match(lines[i]) \
            and len(lines[i]) - len(lines[i].lstrip()) > len(indent_spaces) \
            and not _FENCE.match(lines[i]):
        text += " " + lines[i].strip()
        i += 1
    block: dict[str, Any] = {"text": text.strip(), "indent": len(indent_spaces) // 2}
    if check is not None:
        block.update(type="task", done=check.lower() == "x")
    else:
        block["type"] = "numbered" if marker[0].isdigit() else "bullet"
    return [block], i


_READERS = (_read_fence, _read_pending, _read_heading, _read_divider, _read_image,
            _read_table, _read_quote, _read_list_item)


def parse_markdown(md: str, rev: int = 0, by: str | None = None) -> list[Block]:
    """Markdown → normalized blocks. Raises DocError naming the bad construct."""
    lines = (md or "").replace("\r\n", "\n").replace("\t", "    ").split("\n")
    blocks: list[dict[str, Any]] = []
    para: list[str] = []

    def flush_para() -> None:
        text = " ".join(p.strip() for p in para).strip()
        if text:
            blocks.append({"type": "paragraph", "text": text})
        para.clear()

    i = 0
    while i < len(lines):
        if not lines[i].strip():
            flush_para()
            i += 1
            continue
        for reader in _READERS:
            read = reader(lines, i, bool(para))
            if read is not None:
                flush_para()
                found, i = read
                blocks.extend(found)
                break
        else:
            para.append(lines[i])
            i += 1
    flush_para()
    _normalize_list_indents(blocks)
    for b in blocks:
        if by:
            b["by"] = by
    return [normalize_block(b, rev) for b in blocks]


def _normalize_list_indents(blocks: list[dict[str, Any]]) -> None:
    """Indent steps of 4 spaces (common) and 2 spaces both nest one level."""
    run: list[dict[str, Any]] = []

    def close() -> None:
        if not run:
            return
        levels = sorted({b.get("indent", 0) for b in run})
        remap = {lvl: n for n, lvl in enumerate(levels)}
        prev = 0
        for b in run:
            lvl = remap[b.get("indent", 0)]
            lvl = min(lvl, prev + 1)  # never jump two levels at once
            b["indent"] = lvl
            prev = lvl
        run.clear()

    for b in blocks:
        if b["type"] in LIST_TYPES:
            run.append(b)
        else:
            close()
    close()


# ---------------------------------------------------------------- out


def _escape_cell(v: str) -> str:
    return (v or "").replace("|", "\\|").replace("\n", " ")


def table_to_markdown(b: dict[str, Any]) -> str:
    cols = b["columns"]
    head = "| " + " | ".join(_escape_cell(c["name"]) for c in cols) + " |"
    sep = "| " + " | ".join("---:" if c["kind"] == "number" else "---" for c in cols) + " |"
    rows = ["| " + " | ".join(_escape_cell(r["cells"].get(c["id"], "")) for c in cols) + " |" for r in b["rows"]]
    return "\n".join([head, sep, *rows])


def _code_fence(code: str) -> str:
    """A fence longer than any run of backticks in ``code``, so no line of it closes the block."""
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    return "`" * max(3, longest + 1)


def block_to_markdown(b: dict[str, Any]) -> str:
    t = b["type"]
    if t == "heading":
        # Text ending in " #" would read back as a closing sequence; one written after it keeps it.
        closing = " " + "#" * b["level"] if re.search(r"\s#+\s*$", b["text"]) else ""
        return "#" * b["level"] + " " + b["text"] + closing
    if t == "paragraph":
        return b["text"]
    if t in LIST_TYPES:
        pad = "  " * b.get("indent", 0)
        if t == "task":
            return f"{pad}- [{'x' if b.get('done') else ' '}] {b['text']}"
        return f"{pad}{'1.' if t == 'numbered' else '-'} {b['text']}"
    if t == "quote":
        return "> " + b["text"]
    if t == "callout":
        marker = _TONE_MARKER.get(b.get("tone", "info"), "NOTE")
        head = f"> [!{marker}]" + (f" {b['title']}" if b.get("title") else "")
        return head + ("\n> " + b["text"] if b.get("text") else "")
    if t == "code":
        code = b.get("code", "")
        fence = _code_fence(code)
        return f"{fence}{b.get('language', '')}\n{code}\n{fence}"
    if t == "divider":
        return "---"
    if t == "table":
        return table_to_markdown(b)
    if t == "chart":
        # Written back the way the agent writes it: `type` names the chart kind.
        spec = {"type": b.get("chart", "bar")}
        spec.update({k: v for k, v in b.items() if k not in ("id", "type", "v", "by", "chart")})
        return "```chart\n" + json.dumps(spec, ensure_ascii=False) + "\n```"
    if t == "metrics":
        return "```metrics\n" + json.dumps(b["items"], ensure_ascii=False) + "\n```"
    if t == "image":
        cap = f' "{b["caption"]}"' if b.get("caption") else ""
        return f"![{b.get('alt', '')}]({b['src']}{cap})"
    if t == "pending":
        return "{{pending: " + b.get("intent", "") + "}}"
    return ""


def blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    prev_type = None
    for b in blocks:
        md = block_to_markdown(b)
        # consecutive list items stay one list (single newline), whatever their kinds
        if parts and b["type"] in LIST_TYPES and prev_type in LIST_TYPES:
            parts[-1] += "\n" + md
        else:
            parts.append(md)
        prev_type = b["type"]
    return "\n\n".join(parts) + ("\n" if parts else "")


def doc_to_markdown(doc: dict[str, Any]) -> str:
    if len(doc["tabs"]) == 1:
        return blocks_to_markdown(doc["tabs"][0]["blocks"])
    out = []
    for tab in doc["tabs"]:
        body = blocks_to_markdown(tab["blocks"])
        out.append(body if body.lstrip().startswith("# ") else f"# {tab['title']}\n\n{body}")
    return "\n\n".join(out)


__all__ = [
    "parse_markdown",
    "block_to_markdown",
    "blocks_to_markdown",
    "doc_to_markdown",
    "table_to_markdown",
]
