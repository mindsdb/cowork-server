"""The doc tools' work: create_doc, read_doc, edit_doc and list_docs.

One executor for both places the agent runs: a turn this server runs wraps it
in anton ToolDefs (cowork.harnesses.anton_harness.doc_tools), and a remote
turn's pod calls it as host tools (cowork.docs.remote). The doc goes through
cowork.docs.service, the same write path the editor uses. Every read, write
and render runs in a worker thread (``DocToolContext.run_in_thread``): a write
fsyncs and waits for the doc's lock, which the person's typing may hold, and on
the event loop it would hold every request and stream with it. The desktop
uses ``asyncio.to_thread``; a remote turn uses the file store's threads
(``run_efs``). Both copy the context, so the turn's settings scope still
applies there.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from cowork.common.settings.user_settings import current_settings_scope
from cowork.db.scoped import LOCAL_SCOPE
from cowork.docs import render
from cowork.docs.images import localize_images, within, write_images
from cowork.docs.model import TEXT_TYPES, Doc, DocError, block_text_for_search, find_tab
from cowork.docs.prompt import (
    CREATE_DOC_DESCRIPTION,
    CREATE_DOC_SCHEMA,
    DOC_PROMPT,
    EDIT_DOC_DESCRIPTION,
    EDIT_DOC_SCHEMA,
    LIST_DOCS_DESCRIPTION,
    LIST_DOCS_SCHEMA,
    READ_DOC_DESCRIPTION,
    READ_DOC_SCHEMA,
    REMOTE_RETRY_RULE,
)
from cowork.docs.service import announce, apply, create_doc
from cowork.docs.store import NewDoc, find_doc_folder, list_doc_folders, load_doc
from cowork.docs.sync import after_write
from cowork.services.artifacts import card_for_folder

logger = logging.getLogger(__name__)

DOC_TOOL_SPECS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("create_doc", CREATE_DOC_DESCRIPTION, CREATE_DOC_SCHEMA),
    ("read_doc", READ_DOC_DESCRIPTION, READ_DOC_SCHEMA),
    ("edit_doc", EDIT_DOC_DESCRIPTION, EDIT_DOC_SCHEMA),
    ("list_docs", LIST_DOCS_DESCRIPTION, LIST_DOCS_SCHEMA),
)
DOC_TOOL_NAMES = tuple(name for name, _description, _schema in DOC_TOOL_SPECS)
DOC_WRITE_TOOL_NAMES = ("create_doc", "edit_doc")
DOC_TOOL_TIMEOUT_S = 60
TOO_BIG_TEXT = (
    "Refused (too_big): this answer would be over 1 MiB. Read the doc in parts: pass `tab` for one tab, "
    "or `ids` for some blocks; view \"outline\" lists the tabs and the block ids."
)

T = TypeVar("T")
EmitStep = Callable[[str], Awaitable[None]]
OnWritten = Callable[[str, str], None]
RunInThread = Callable[[Callable[[], T]], Awaitable[T]]


@dataclass(frozen=True)
class DocToolResult:
    """What one call answers the agent; ``touched_slug`` names the doc it wrote."""

    text: str
    ok: bool = True
    touched_slug: str | None = None


class DocToolContext:
    """What one turn's doc tools share; built once per turn, so ``turn`` makes
    everything one reply writes one version of the doc."""

    def __init__(
        self,
        *,
        artifacts_base: Path,
        conversation_id: str | None = None,
        project_id: str | None = None,
        project_name: str = "",
        project_label: str = "",
        owner_check: Callable[[Path], None] | None = None,
        image_reader: Callable[[str], bytes] | None = None,
        max_result_bytes: int | None = None,
        run_in_thread: RunInThread | None = None,
    ) -> None:
        self.artifacts_base = Path(artifacts_base)
        # Where every blocking step of a call runs, owner check and on_written included.
        self.run_in_thread: RunInThread = run_in_thread or asyncio.to_thread
        self.conversation_id = conversation_id
        self.project_id = project_id
        self.project_name = project_name
        self.project_label = project_label
        # Raises DocError("not_owner") before an edit_doc writes; None allows every edit.
        self.owner_check = owner_check
        # Reads a local image path the server must not resolve itself (a pod's).
        self.image_reader = image_reader
        # Caps a read_doc answer before it renders; None leaves it uncapped.
        self.max_result_bytes = max_result_bytes
        self.turn = uuid.uuid4().hex
        # Files attached to the conversation, set once the harness has read them:
        # with the project's own images, the local images a doc may take.
        self.attached_files: frozenset[Path] = frozenset()

    def may_use_image(self, path: Path) -> bool:
        # The project is the folder that holds `.anton/artifacts`.
        return within(self.artifacts_base.parent.parent, *self.attached_files)(path)

    def artifacts_dir(self, session: Any) -> Path:
        # anton exposes no public accessor; its own tools read the workspace the same way.
        workspace = getattr(session, "_workspace", None)
        path = getattr(workspace, "artifacts_dir", None)
        return Path(path) if path else self.artifacts_base

    def card(self, folder: Path) -> dict[str, Any] | None:
        try:
            card = card_for_folder(
                folder,
                scope=current_settings_scope(),
                project_id=self.project_id,
                project_name=self.project_name,
                project_label=self.project_label,
                artifacts_base=folder.parent,
            )
            # The card rides on a JSON event (cowork.docs.service.announce).
            return card.to_json() if card is not None else None
        except Exception:  # noqa: BLE001 - the doc is written; failing here would have the agent make it again
            # At warning, with the stack: a bug here once sent every doc.created out without its card, unseen.
            logger.warning("doc card unavailable for %s", folder, exc_info=True)
            return None


def _refusal(err: DocError, doc: Doc | None = None) -> DocToolResult:
    lines = [f"Refused ({err.code}): {err.message}"]
    if err.code == "stale" and doc is not None:
        # Show the block as it stands, so the next call can build on it.
        lines.append("It now reads:\n" + render.changed_lines(doc, {t["id"]: err.detail.get("ids", []) for t in doc["tabs"]}))
    for key in ("tabs", "columns", "ids", "rows"):
        if key in err.detail:
            lines.append(f"{key}: {err.detail[key]}")
    return DocToolResult("\n".join(lines), ok=False)


def _tab_text_bytes(doc: Doc) -> int:
    """A floor on the size of the first tab rendered in full: its blocks' text."""
    return sum(len(block_text_for_search(b).encode("utf-8")) for b in find_tab(doc, None)["blocks"])


def _pending_note(doc: dict[str, Any]) -> str:
    pending = [
        (tab, b) for tab in doc["tabs"] for b in tab["blocks"] if b["type"] == "pending"
    ]
    if not pending:
        return ""
    items = "; ".join(f"{b['id']} ({b.get('intent', '')})" for _tab, b in pending[:12])
    return (
        f"\nPending sections still to write: {items}. Fill each with edit_doc "
        "{op:\"replace\", target:\"<that id>\", markdown:\"## Heading\\n\\n…\"}, one section per call."
    )


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKS = re.compile(r"[*_`~]")


def _plain(text: str) -> str:
    return " ".join(_MARKS.sub("", text).lower().split())


def _sentences(b: dict[str, Any]) -> list[str]:
    parts = [b.get("title", ""), b.get("text", "")] if b["type"] == "callout" else [b.get("text", "")]
    return [s.strip() for part in parts for s in _SENTENCE_BREAK.split(part or "") if s.strip()]


def _taken_out(before: dict[str, Any], after: dict[str, Any], result: dict[str, Any], ops: list[Any],
               pending_before: set[str]) -> list[str]:
    """Sentences a replace took out of a block that are nowhere in the doc now.

    The agent adds to a block by rewriting it whole, and now and then the
    rewrite leaves out what the block already said. Naming those words in the
    reply lets it put them back in the same turn when the person only asked for
    an addition.
    """
    old = {b["id"]: b for t in before["tabs"] for b in t["blocks"]}
    now = _plain("\n".join(block_text_for_search(b) for t in after["tabs"] for b in t["blocks"]))
    out: list[str] = []
    for r in result["results"]:
        if ops[r["op"]].get("op") != "replace":
            continue
        for block_id in r.get("replaced", []):
            b = old.get(block_id)
            if b is None or block_id in pending_before or b["type"] not in TEXT_TYPES:
                continue
            for s in _sentences(b):
                if len(s.split()) >= 3 and _plain(s).rstrip(".!?") not in now and s not in out:
                    out.append(s)
    return out[:5]


def _repeated(doc: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Text this edit left twice in a tab: a block it wrote reads the same as another there.

    Asked to turn a sentence into a list, an agent once added the list, then
    rewrote the paragraph with the list in it again, and reported success.
    """
    out: list[str] = []
    for tab in doc["tabs"]:
        written = set(result["touched"].get(tab["id"], []))
        if not written:
            continue
        seen: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for b in tab["blocks"]:
            if b["type"] in TEXT_TYPES and len(_plain(b.get("text", "")).split()) >= 3:
                seen.setdefault((b["type"], _plain(b["text"])), []).append(b)
        for same in seen.values():
            if len(same) > 1 and any(b["id"] in written for b in same):
                out.append(same[0]["text"])
    return out[:5]


def edit_label(ops: Any, title: str) -> str:
    """What one edit_doc call is doing, in the words the chat shows."""
    if not isinstance(ops, list) or not ops:
        return f"Editing “{render.clip(title)}”"
    headings = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        md = op.get("markdown") if isinstance(op.get("markdown"), str) else ""
        first = md.lstrip().split("\n", 1)[0]
        if op.get("op") in ("replace", "insert") and first.startswith("#"):
            headings.append(first.lstrip("#").strip())
        elif op.get("op") == "add_tab":
            headings.append(str(op.get("title") or "new tab"))
    if headings:
        return "Writing " + ", ".join(f"“{render.clip(h, 40)}”" for h in headings[:3]) + (
            f" +{len(headings) - 3}" if len(headings) > 3 else "")
    n = len(ops)
    return f"Editing “{render.clip(title)}” · {n} change{'s' if n != 1 else ''}"


def _if_rev(value: Any) -> int | None:
    """The rev the agent read, for the stale guard. Docs start at rev 1, so a
    filled-in 0 (or blank) means the agent named none."""
    if value in (None, "", 0, "0"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise DocError("bad_op", "if_rev is the rev number your last read_doc returned.") from None


def _edit_report(before: Doc, doc: Doc, result: dict[str, Any], ops: list[Any]) -> str:
    """What one edit_doc call did, for the agent: the blocks it wrote, sections filled, words taken out."""
    pending_before = {b["id"] for t in before["tabs"] for b in t["blocks"] if b["type"] == "pending"}
    written = render.changed_lines(doc, result["touched"])
    filled = [i for r in result["results"] for i in r.get("replaced", []) if i in pending_before]
    removed = sum(len(v) for v in result["removed"].values()) - len(filled)
    new_tabs = [r["tab"] for r in result["results"] if ops[r["op"]].get("op") == "add_tab"]
    lines = [f"Applied {len(result['results'])} op(s) to “{doc['title']}” · now rev {doc['rev']}."]
    if new_tabs:
        lines.append("New tab id(s): " + ", ".join(new_tabs))
    if written:
        lines.append("Written:\n" + written)
    if filled:
        lines.append("Filled pending section(s): " + ", ".join(filled) + ".")
    if removed > 0:
        lines.append(f"Removed {removed} block(s).")
    taken_out = _taken_out(before, doc, result, ops, pending_before)
    if taken_out:
        lines.append("Words this edit took out: " + "; ".join(f"“{render.clip(s, 160)}”" for s in taken_out)
                     + ". Put them back unless the person asked to cut or change them.")
    repeated = _repeated(doc, result)
    if repeated:
        lines.append("Now in the doc twice: " + "; ".join(f"“{render.clip(s, 120)}”" for s in repeated)
                     + ". Unless the person asked for a copy, delete the extra one.")
    return "\n".join(lines) + _pending_note(doc)


def _open_doc(artifacts_dir: Path, ref: str) -> tuple[Path, Doc]:
    folder = find_doc_folder(artifacts_dir, ref)
    return folder, load_doc(folder)



async def _create(ctx: DocToolContext, tc_input: dict, emit_step: EmitStep, *, artifacts_dir: Path,
                  on_written: OnWritten | None) -> DocToolResult:
    await emit_step(f"Creating doc “{render.clip(tc_input.get('title') or 'Untitled')}”")

    def create() -> tuple[NewDoc, dict[str, Any] | None, str]:
        content, images = localize_images({"markdown": tc_input.get("markdown"), "tabs": tc_input.get("tabs")},
                                          ctx.may_use_image, reader=ctx.image_reader)
        made = create_doc(
            artifacts_dir,
            title=str(tc_input.get("title") or ""),
            markdown=content["markdown"],
            tabs=content["tabs"],
            description=str(tc_input.get("description") or ""),
            by="agent",
            turn=ctx.turn,
        )
        # First, while nothing else can fail: the doc exists from here on.
        if on_written is not None:
            on_written(made.slug, "create_doc")
        write_images(made.folder, images)
        text = (
            f"Created doc “{made.doc['title']}” — it is open beside the chat and the user can see it.\n"
            + render.outline(made.doc, all_tabs=True)
            + _pending_note(made.doc)
        )
        return made, ctx.card(made.folder), text

    try:
        made, card, text = await ctx.run_in_thread(create)
    except DocError as err:
        return _refusal(err)
    announce(made.artifact_id, made.doc, None, folder=made.folder, by="agent",
             conversation_id=ctx.conversation_id, created=True, card=card)
    return DocToolResult(text, touched_slug=made.slug)


async def _read(ctx: DocToolContext, tc_input: dict, emit_step: EmitStep, *, artifacts_dir: Path,
                on_written: OnWritten | None) -> DocToolResult:
    try:
        _folder, doc = await ctx.run_in_thread(partial(_open_doc, artifacts_dir, str(tc_input.get("doc") or "")))
    except DocError as err:
        await emit_step("Reading a doc")
        return _refusal(err)
    await emit_step(f"Reading “{render.clip(doc['title'])}”")
    view = tc_input.get("view") or "full"
    tab, ids = tc_input.get("tab"), tc_input.get("ids")

    def read() -> DocToolResult:
        if view == "outline":
            return DocToolResult(render.outline(doc, all_tabs=True))
        if ctx.max_result_bytes is not None and not tab and not ids and _tab_text_bytes(doc) > ctx.max_result_bytes:
            return DocToolResult(TOO_BIG_TEXT, ok=False)
        return DocToolResult(render.full(doc, tab, ids=ids))

    try:
        return await ctx.run_in_thread(read)
    except DocError as err:
        return _refusal(err)


async def _edit(ctx: DocToolContext, tc_input: dict, emit_step: EmitStep, *, artifacts_dir: Path,
                on_written: OnWritten | None) -> DocToolResult:
    try:
        if_rev = _if_rev(tc_input.get("if_rev"))
        folder, before = await ctx.run_in_thread(partial(_open_doc, artifacts_dir, str(tc_input.get("doc") or "")))
    except DocError as err:
        await emit_step("Editing a doc")
        return _refusal(err)
    ops = tc_input.get("ops")
    await emit_step(edit_label(ops, before["title"]))

    def edit() -> tuple[Doc, dict[str, Any], str]:
        if ctx.owner_check is not None:
            ctx.owner_check(folder)
        local_ops, images = localize_images(ops, ctx.may_use_image, reader=ctx.image_reader)
        write_images(folder, images)
        doc, result = apply(folder, local_ops, tab=tc_input.get("tab"), by="agent", if_rev=if_rev, turn=ctx.turn)
        if on_written is not None:
            on_written(folder.name, "edit_doc")
        return doc, result, _edit_report(before, doc, result, ops)

    try:
        doc, result, text = await ctx.run_in_thread(edit)
    except DocError as err:
        current = None
        if err.code == "stale":
            try:
                current = await ctx.run_in_thread(partial(load_doc, folder))
            except Exception:  # noqa: BLE001 - nothing was written; the refusal stands without the block's text
                logger.debug("could not reread %s for a stale refusal", folder, exc_info=True)
        return _refusal(err, current)
    after_write(folder, doc, result, by="agent", scope=current_settings_scope() or LOCAL_SCOPE,
                conversation_id=ctx.conversation_id)
    return DocToolResult(text, touched_slug=folder.name)


async def _list(ctx: DocToolContext, tc_input: dict, emit_step: EmitStep, *, artifacts_dir: Path,
                on_written: OnWritten | None) -> DocToolResult:
    await emit_step("Looking for docs")
    docs = await ctx.run_in_thread(partial(list_doc_folders, artifacts_dir))
    if not docs:
        return DocToolResult("No docs in this project yet. create_doc makes one.")
    lines = []
    for d in docs[:50]:
        tabs = ", ".join(f"“{t['title']}” {t['id']}" for t in d["tabs"])
        lines.append(f"“{d['title']}” · id {d['id']} · rev {d['rev']} · updated {d['updated_at']} · tabs: {tabs}")
    return DocToolResult("\n".join(lines))


_HANDLERS = {"create_doc": _create, "read_doc": _read, "edit_doc": _edit, "list_docs": _list}


async def run_doc_tool(
    ctx: DocToolContext,
    name: str,
    tc_input: dict,
    emit_step: EmitStep,
    *,
    artifacts_dir: Path | None = None,
    on_written: OnWritten | None = None,
) -> DocToolResult:
    """Run one doc tool call. ``emit_step`` gets the label the chat shows for it,
    before the work; ``on_written`` runs in the worker thread right after a
    write, so whatever must follow the write happens even when the caller's
    task is cancelled (the thread runs on). ``artifacts_dir`` defaults to
    ``ctx.artifacts_base``. A refused call answers ``Refused (<code>): …``."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return DocToolResult(f"No doc tool named {name!r}; the doc tools are {', '.join(DOC_TOOL_NAMES)}.", ok=False)
    return await handler(
        ctx,
        tc_input if isinstance(tc_input, dict) else {},
        emit_step,
        artifacts_dir=Path(artifacts_dir) if artifacts_dir is not None else ctx.artifacts_base,
        on_written=on_written,
    )


def remote_doc_tool_specs(*, timeout_s: int = DOC_TOOL_TIMEOUT_S) -> list[dict[str, Any]]:
    """The doc tools as a remote turn's ``host_tools`` block (see the shared wire
    contract). ``create_doc`` carries the prompt, as its ToolDef does on the
    desktop, plus the rule for a call the host never answered."""
    specs = []
    for name, description, schema in DOC_TOOL_SPECS:
        spec: dict[str, Any] = {"name": name, "description": description, "input_schema": schema,
                                "timeout_s": timeout_s}
        if name == "create_doc":
            spec["prompt"] = DOC_PROMPT + REMOTE_RETRY_RULE
        specs.append(spec)
    return specs


__all__ = [
    "DOC_TOOL_NAMES", "DOC_TOOL_SPECS", "DOC_WRITE_TOOL_NAMES", "DOC_TOOL_TIMEOUT_S", "TOO_BIG_TEXT", "DocToolContext", "DocToolResult",
    "EmitStep", "OnWritten", "RunInThread", "edit_label", "remote_doc_tool_specs", "run_doc_tool",
]
