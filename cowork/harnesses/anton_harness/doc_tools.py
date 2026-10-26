"""The agent's doc tools: create_doc, read_doc, edit_doc, list_docs.

Built for each turn, so each handler knows the conversation and project it
writes for: that is how a doc the agent creates opens in the right chat the
moment it exists (``cowork.docs.hub``), rather than only as an end-of-turn
card. It also names the turn, so everything one reply writes is one version
of the doc. The doc itself goes through ``cowork.docs.service``, the same
write path the editor uses.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from anton.core.artifacts import ArtifactStore
from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef
from anton.core.tools.tool_handlers import _track_artifact

from cowork.common.settings.user_settings import current_settings_scope
from cowork.db.scoped import LOCAL_SCOPE
from cowork.docs import render
from cowork.docs.model import TEXT_TYPES, Doc, DocError, block_text_for_search
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
)
from cowork.docs.images import localize_images, within, write_images
from cowork.docs.service import announce, apply, create_doc
from cowork.docs.store import find_doc_folder, list_doc_folders, load_doc
from cowork.docs.sync import after_write
from cowork.services.artifacts import card_for_folder

logger = logging.getLogger(__name__)

DOC_TOOL_NAMES = ("create_doc", "read_doc", "edit_doc", "list_docs")


class DocToolContext:
    def __init__(
        self,
        *,
        artifacts_base: Path,
        conversation_id: str | None = None,
        project_id: str | None = None,
        project_name: str = "",
        project_label: str = "",
    ) -> None:
        self.artifacts_base = Path(artifacts_base)
        self.conversation_id = conversation_id
        self.project_id = project_id
        self.project_name = project_name
        self.project_label = project_label
        # The harness builds the tools for each turn: this names the reply.
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
            return card_for_folder(
                folder,
                scope=current_settings_scope(),
                project_id=self.project_id,
                project_name=self.project_name,
                project_label=self.project_label,
                artifacts_base=folder.parent,
            )
        except Exception:  # noqa: BLE001 - the doc is written; failing here would have the agent make it again
            # At warning, with the stack: a bug here once sent every doc.created out without its card, unseen.
            logger.warning("doc card unavailable for %s", folder, exc_info=True)
            return None


def _outcome(text: str, ok: bool = True) -> ToolOutcome:
    return ToolOutcome(content=text, ok=ok)


def _refusal(err: DocError, doc: Doc | None = None) -> ToolOutcome:
    lines = [f"Refused ({err.code}): {err.message}"]
    if err.code == "stale" and doc is not None:
        # Show the block as it stands, so the next call can build on it.
        lines.append("It now reads:\n" + render.changed_lines(doc, {t["id"]: err.detail.get("ids", []) for t in doc["tabs"]}))
    for key in ("tabs", "columns", "ids", "rows"):
        if key in err.detail:
            lines.append(f"{key}: {err.detail[key]}")
    return _outcome("\n".join(lines), ok=False)


def _track(session: Any, artifacts_dir: Path, slug: str, summary: str) -> None:
    """Mark the doc as touched this turn, so its card lands on this turn's answer.

    The same call anton's own artifact tools make; anton has no public one.
    """
    try:
        _track_artifact(session, ArtifactStore(artifacts_dir), slug, summary=summary)
    except Exception:
        logger.debug("could not record doc %s as touched", slug, exc_info=True)


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


def _step(text: str) -> ToolProgress:
    """A progress line: the chat shows each doc call as a step with this label."""
    return ToolProgress(text=text)


def _quote(text: str, n: int = 60) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def edit_label(ops: Any, title: str) -> str:
    """What one edit_doc call is doing, in the words the chat shows."""
    if not isinstance(ops, list) or not ops:
        return f"Editing “{_quote(title)}”"
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
        return "Writing " + ", ".join(f"“{_quote(h, 40)}”" for h in headings[:3]) + (
            f" +{len(headings) - 3}" if len(headings) > 3 else "")
    n = len(ops)
    return f"Editing “{_quote(title)}” · {n} change{'s' if n != 1 else ''}"


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
        lines.append("Words this edit took out: " + "; ".join(f"“{_quote(s, 160)}”" for s in taken_out)
                     + ". Put them back unless the person asked to cut or change them.")
    repeated = _repeated(doc, result)
    if repeated:
        lines.append("Now in the doc twice: " + "; ".join(f"“{_quote(s, 120)}”" for s in repeated)
                     + ". Unless the person asked for a copy, delete the extra one.")
    return "\n".join(lines) + _pending_note(doc)


def _open_doc(artifacts_dir: Path, ref: str) -> tuple[Path, Doc]:
    folder = find_doc_folder(artifacts_dir, ref)
    return folder, load_doc(folder)


# Every read and write below runs in a worker thread (to_thread copies the
# context, so the settings scope still applies there). A write fsyncs and waits
# for the doc's lock, which the person's typing may be holding on an HTTP
# thread: on the event loop, that would hold every request and stream with it.
def build_doc_tools(ctx: DocToolContext) -> list[ToolDef]:
    async def handle_create(session: Any, tc_input: dict) -> Any:
        artifacts_dir = ctx.artifacts_dir(session)
        yield _step(f"Creating doc “{_quote(tc_input.get('title') or 'Untitled')}”")
        def create() -> Any:
            content, images = localize_images({"markdown": tc_input.get("markdown"), "tabs": tc_input.get("tabs")},
                                              ctx.may_use_image)
            made = create_doc(
                artifacts_dir,
                title=str(tc_input.get("title") or ""),
                markdown=content["markdown"],
                tabs=content["tabs"],
                description=str(tc_input.get("description") or ""),
                by="agent",
                turn=ctx.turn,
            )
            write_images(made.folder, images)
            return made

        try:
            made = await asyncio.to_thread(create)
        except DocError as err:
            yield _refusal(err)
            return
        _track(session, artifacts_dir, made.slug, "Created doc")
        doc = made.doc
        announce(made.artifact_id, doc, None, by="agent", conversation_id=ctx.conversation_id,
                 created=True, card=ctx.card(made.folder))
        yield _outcome(
            f"Created doc “{doc['title']}” — it is open beside the chat and the user can see it.\n"
            + render.outline(doc, all_tabs=True)
            + _pending_note(doc)
        )

    async def handle_read(session: Any, tc_input: dict) -> Any:
        try:
            _folder, doc = await asyncio.to_thread(_open_doc, ctx.artifacts_dir(session), str(tc_input.get("doc") or ""))
        except DocError as err:
            yield _step("Reading a doc")
            yield _refusal(err)
            return
        yield _step(f"Reading “{_quote(doc['title'])}”")
        try:
            view = tc_input.get("view") or "full"
            if view == "outline":
                yield _outcome(render.outline(doc, all_tabs=True))
            else:
                yield _outcome(render.full(doc, tc_input.get("tab"), ids=tc_input.get("ids")))
        except DocError as err:
            yield _refusal(err)

    async def handle_edit(session: Any, tc_input: dict) -> Any:
        artifacts_dir = ctx.artifacts_dir(session)
        try:
            if_rev = _if_rev(tc_input.get("if_rev"))
            folder, before = await asyncio.to_thread(_open_doc, artifacts_dir, str(tc_input.get("doc") or ""))
        except DocError as err:
            yield _step("Editing a doc")
            yield _refusal(err)
            return
        ops = tc_input.get("ops")
        label = edit_label(ops, before["title"])
        yield _step(label)
        def edit() -> tuple[Doc, dict[str, Any]]:
            local_ops, images = localize_images(ops, ctx.may_use_image)
            write_images(folder, images)
            return apply(folder, local_ops, tab=tc_input.get("tab"), by="agent", if_rev=if_rev, turn=ctx.turn)

        try:
            doc, result = await asyncio.to_thread(edit)
        except DocError as err:
            yield _refusal(err, await asyncio.to_thread(load_doc, folder) if err.code == "stale" else None)
            return
        _track(session, artifacts_dir, folder.name, "Edited doc")
        after_write(folder, doc, result, by="agent", scope=current_settings_scope() or LOCAL_SCOPE,
                    conversation_id=ctx.conversation_id)
        yield _outcome(_edit_report(before, doc, result, ops))

    async def handle_list(session: Any, tc_input: dict) -> Any:
        yield _step("Looking for docs")
        docs = await asyncio.to_thread(list_doc_folders, ctx.artifacts_dir(session))
        if not docs:
            yield _outcome("No docs in this project yet. create_doc makes one.")
            return
        lines = []
        for d in docs[:50]:
            tabs = ", ".join(f"“{t['title']}” {t['id']}" for t in d["tabs"])
            lines.append(f"“{d['title']}” · id {d['id']} · rev {d['rev']} · updated {d['updated_at']} · tabs: {tabs}")
        yield _outcome("\n".join(lines))

    return [
        ToolDef(name="create_doc", description=CREATE_DOC_DESCRIPTION, input_schema=CREATE_DOC_SCHEMA,
                handler=handle_create, prompt=DOC_PROMPT),
        ToolDef(name="read_doc", description=READ_DOC_DESCRIPTION, input_schema=READ_DOC_SCHEMA,
                handler=handle_read),
        ToolDef(name="edit_doc", description=EDIT_DOC_DESCRIPTION, input_schema=EDIT_DOC_SCHEMA,
                handler=handle_edit),
        ToolDef(name="list_docs", description=LIST_DOCS_DESCRIPTION, input_schema=LIST_DOCS_SCHEMA,
                handler=handle_list),
    ]
