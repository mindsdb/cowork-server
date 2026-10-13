"""Redact credentials already stored in conversation history.

Scratchpad output is scrubbed before it is stored, but rows and turn files
written before that may still hold a printed OAuth token or API key. This
runs anton's `scrub_output` over what Cowork has already saved:

- `message_events.event_data`, including the cell JSON a scratchpad result
  event carries inside its `content` string;
- assistant messages and tool_result rows in `messages`;
- `conversations.history_summary`;
- file-backed turn buffers under the streams directory.

Left as written: a cell's `code`, a tool_use block's `input`, the code in a
tool call event, and messages the user typed. A tool call event cut off at
its size cap is no longer valid JSON and is scrubbed as plain text, code
included. Turn buffers kept in Redis are not covered.

It covers every org in the database: it is an operator tool for one
deployment's own data. A dry run is the default and prints counts only,
never a value; `--apply` writes the changes. Stop the server before
`--apply`: it rewrites turn files the server may be appending to.

Usage, from a cowork-server checkout with the same DATABASE_URI and
COWORK_HOME the server uses:

    uv run python -m scripts.redact_stored_credentials
    uv run python -m scripts.redact_stored_credentials --apply

On an installed deployment, where `scripts/` is not part of the package,
run this file with the server's Python and environment:

    python redact_stored_credentials.py [--apply]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anton.utils.datasources import scrub_output
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine

# Every model module must be imported before any mapper is used; see
# seed_long_conversation.py for why.
import cowork.models as _models_pkg

for _, _name, _ in pkgutil.iter_modules(_models_pkg.__path__):
    importlib.import_module(f"cowork.models.{_name}")

from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.schemas.responses import Role
from cowork.streaming.backend import get_streams_dir

BATCH_SIZE = 500


@dataclass
class RedactionReport:
    """Counts of what changed, or would change in a dry run, per store."""

    message_events: int = 0
    messages: int = 0
    history_summaries: int = 0
    turn_files: int = 0
    skipped_turn_files: list[str] = field(default_factory=list)


def _scrub_string(text: str) -> str:
    """Scrub one string, walking any JSON object or list it holds.

    A scratchpad result event stores the cell as JSON inside a string, where
    escaping can hide a token and a whole-string scrub would also reach the
    cell's code, so nested JSON is decoded, scrubbed field by field and
    encoded again with its surrounding whitespace kept.
    """
    body = text.strip()
    if body[:1] in ("{", "["):
        try:
            decoded = json.loads(body)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)):
            scrubbed = scrub_value(decoded)
            if scrubbed == decoded:
                return text
            start = text.index(body)
            return text[:start] + json.dumps(scrubbed) + text[start + len(body):]
    return scrub_output(text)


def _is_token_value(key: str, value: str) -> bool:
    """True when ``value`` is a token by its key, judged as the printed pair the filter knows."""
    return scrub_output(json.dumps({key: value})) == json.dumps({key: "[REDACTED_TOKEN]"})


def _model_written_key(value: dict) -> str | None:
    """The key holding code the model wrote, for a cell, a tool call or a tool_use block."""
    if value.get("type") == "tool_use":
        return "input"
    if "code" in value and ("stdout" in value or "action" in value):
        return "code"
    return None


def scrub_value(value: Any) -> Any:
    """Return ``value`` with every string scrubbed, except code the model wrote.

    Walks dicts and lists. A decoded value has lost the key text that marks
    it as a token, so each string field is also judged together with its key.
    """
    if isinstance(value, str):
        return _scrub_string(value)
    if isinstance(value, list):
        return [scrub_value(item) for item in value]
    if isinstance(value, dict):
        keep = _model_written_key(value)
        scrubbed = {}
        for key, item in value.items():
            if key == keep:
                scrubbed[key] = item
            elif isinstance(item, str) and _is_token_value(key, item):
                scrubbed[key] = "[REDACTED_TOKEN]"
            else:
                scrubbed[key] = scrub_value(item)
        return scrubbed
    return value


def scrub_message_content(role: str, content: Any) -> Any:
    """Scrub one stored message's content.

    Assistant content is scrubbed throughout, except tool_use input. In a
    user message only tool_result blocks are scrubbed: the rest is what the
    user typed and stays as it was.
    """
    if role == Role.assistant.value:
        return scrub_value(content)
    if isinstance(content, list):
        return [
            scrub_value(block) if isinstance(block, dict) and block.get("type") == "tool_result" else block
            for block in content
        ]
    return content


def _batches(session: Session, model: type, *where: Any) -> Iterator[list[Any]]:
    """Yield ``model`` rows in id order, ``BATCH_SIZE`` at a time, without holding them all."""
    last_id = None
    while True:
        query = select(model).where(*where).order_by(model.id).limit(BATCH_SIZE)
        if last_id is not None:
            query = query.where(model.id > last_id)
        rows = list(session.exec(query))
        if not rows:
            return
        # Read before yielding: the caller commits and expunges the batch.
        last_id = rows[-1].id
        yield rows


def _scrub_rows(session: Session, model: type, column: str, scrub: Any, *where: Any, apply: bool) -> int:
    """Scrub one JSON or text column of ``model`` batch by batch; return how many rows change.

    Each batch is committed and released on its own, so memory stays flat
    and an interrupted run keeps what it wrote; a rerun finds only the rest.
    """
    changed = 0
    for rows in _batches(session, model, *where):
        for row in rows:
            scrubbed = scrub(row)
            if scrubbed != getattr(row, column):
                changed += 1
                if apply:
                    setattr(row, column, scrubbed)
                    session.add(row)
        if apply:
            session.commit()
        session.expunge_all()
    return changed


def _message_role(message: Message) -> str:
    """The message's role as its stored string."""
    return message.role.value if isinstance(message.role, Role) else str(message.role)


def _turn_files(streams_dir: Path) -> Iterator[Path]:
    """Every file-backed turn buffer under ``streams_dir``."""
    if streams_dir.is_dir():
        yield from sorted(streams_dir.glob("*/turn_*.jsonl"))


def _scrub_turn_file(path: Path) -> str | None:
    """Return the scrubbed file text, or None when nothing in it changes.

    Lines are rewritten the way the buffer writes them; a line that is not
    JSON, such as a half-written last line, is kept as it is.
    """
    original = path.read_text(encoding="utf-8")
    lines = []
    for line in original.splitlines(keepends=True):
        body = line.rstrip("\n")
        try:
            record = json.loads(body)
        except ValueError:
            lines.append(line)
            continue
        scrubbed = scrub_value(record)
        if scrubbed == record:
            lines.append(line)
        else:
            lines.append(json.dumps(scrubbed, ensure_ascii=False) + line[len(body):])
    updated = "".join(lines)
    return None if updated == original else updated


def _replace_file(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temp file in the same directory, so a crash leaves the old file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def redact_stored_credentials(session: Session, streams_dir: Path, *, apply: bool) -> RedactionReport:
    """Scrub stored history in ``session``'s database and ``streams_dir``.

    Counts every row and file that holds something to redact; with
    ``apply`` it also writes them back. Covers every org in the database.
    """
    report = RedactionReport()
    report.message_events = _scrub_rows(
        session, MessageEvent, "event_data", lambda e: scrub_value(e.event_data), apply=apply,
    )
    report.messages = _scrub_rows(
        session, Message, "content", lambda m: scrub_message_content(_message_role(m), m.content), apply=apply,
    )
    report.history_summaries = _scrub_rows(
        session, Conversation, "history_summary", lambda c: scrub_output(c.history_summary),
        Conversation.history_summary.is_not(None), apply=apply,
    )

    for path in _turn_files(streams_dir):
        try:
            updated = _scrub_turn_file(path)
            if updated is None:
                continue
            report.turn_files += 1
            if apply:
                _replace_file(path, updated)
        except (OSError, UnicodeDecodeError) as exc:
            report.skipped_turn_files.append(f"{path.name}: {type(exc).__name__}")

    return report


def main() -> None:
    """Run against the configured database and streams directory, print counts."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = parser.parse_args()

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        report = redact_stored_credentials(session, get_streams_dir(), apply=args.apply)

    verb = "Redacted" if args.apply else "Would redact"
    print(f"{verb}: {report.message_events} message events, {report.messages} messages, "
          f"{report.history_summaries} conversation summaries, {report.turn_files} turn files.")
    for name in report.skipped_turn_files:
        print(f"Skipped turn file {name}")
    if not args.apply:
        print("Dry run: nothing was changed. Run again with --apply to write.")


if __name__ == "__main__":
    main()
