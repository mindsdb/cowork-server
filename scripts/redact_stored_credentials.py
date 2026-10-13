"""Redact credentials already stored in conversation history.

Scratchpad output is scrubbed before it is stored, but rows and turn files
written before that may still hold a printed OAuth token or API key. This
runs anton's `scrub_output` over what Cowork has already saved:

- `message_events.event_data`, including the cell JSON a scratchpad result
  event carries inside its `content` string;
- assistant messages and tool_result rows in `messages`;
- `conversations.history_summary`;
- file-backed turn buffers under the streams directory.

Code the model wrote (`code` keys and tool_use `input`) and user-typed
messages are left as they are. A dry run is the default and prints counts
only, never a value; `--apply` writes the changes.

Usage (run from cowork-server/, same DATABASE_URI the server itself uses):

    uv run python -m scripts.redact_stored_credentials
    uv run python -m scripts.redact_stored_credentials --apply
"""
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
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

_MODEL_WRITTEN_KEYS = frozenset({"code", "input"})


@dataclass
class RedactionReport:
    """Counts of what changed, or would change in a dry run, per store."""

    message_events: int = 0
    messages: int = 0
    history_summaries: int = 0
    turn_files: int = 0
    unreadable_turn_files: list[str] = field(default_factory=list)


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


def scrub_value(value: Any) -> Any:
    """Return ``value`` with every string scrubbed, except code the model wrote.

    Walks dicts and lists; keys in `_MODEL_WRITTEN_KEYS` are left as written.
    A decoded value has lost the key text that marks it as a token, so each
    string field is also judged together with its key.
    """
    if isinstance(value, str):
        return _scrub_string(value)
    if isinstance(value, list):
        return [scrub_value(item) for item in value]
    if isinstance(value, dict):
        scrubbed = {}
        for key, item in value.items():
            if key in _MODEL_WRITTEN_KEYS:
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


def redact_stored_credentials(session: Session, streams_dir: Path, *, apply: bool) -> RedactionReport:
    """Scrub stored history in ``session``'s database and ``streams_dir``.

    Counts every row and file that holds something to redact; with
    ``apply`` it also writes them back and commits. Runs across every org:
    it is an operator tool for one deployment's own data.
    """
    report = RedactionReport()

    for event in session.exec(select(MessageEvent)):
        scrubbed = scrub_value(event.event_data)
        if scrubbed != event.event_data:
            report.message_events += 1
            if apply:
                event.event_data = scrubbed
                session.add(event)

    for message in session.exec(select(Message)):
        role = message.role.value if isinstance(message.role, Role) else str(message.role)
        scrubbed = scrub_message_content(role, message.content)
        if scrubbed != message.content:
            report.messages += 1
            if apply:
                message.content = scrubbed
                session.add(message)

    for conversation in session.exec(select(Conversation).where(Conversation.history_summary.is_not(None))):
        scrubbed = scrub_output(conversation.history_summary)
        if scrubbed != conversation.history_summary:
            report.history_summaries += 1
            if apply:
                conversation.history_summary = scrubbed
                session.add(conversation)

    if apply:
        session.commit()

    for path in _turn_files(streams_dir):
        try:
            updated = _scrub_turn_file(path)
        except (OSError, UnicodeDecodeError) as exc:
            report.unreadable_turn_files.append(f"{path.name}: {type(exc).__name__}")
            continue
        if updated is not None:
            report.turn_files += 1
            if apply:
                path.write_text(updated, encoding="utf-8")

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
    for name in report.unreadable_turn_files:
        print(f"Skipped unreadable turn file {name}")
    if not args.apply:
        print("Dry run: nothing was changed. Run again with --apply to write.")


if __name__ == "__main__":
    main()
