"""Seed a conversation with a long message history, for manually verifying
ENG-2768 (loading feedback + cursor pagination) at realistic sizes.

Bulk-inserts `Message` rows directly rather than going through
`ConversationService.save_assistant_turn` — that path does one extra
`_next_seq` query per turn (see its own docstring), which is the right
tradeoff for a real turn but makes seeding 1,000 messages needlessly slow.
`seq` is assigned the same way `_next_seq` would (a monotonic counter from
0), so the seeded rows are indistinguishable from real ones to
`_MESSAGE_ORDER`/pagination.

Every 6th assistant turn also gets a hidden tool_use/tool_result pair ahead
of its visible answer, so a seeded conversation exercises the pagination
service's visible-row-skipping instead of only ever seeing plain rows.

Usage (run from cowork-server/, same DATABASE_URI the server itself uses):

    uv run python -m scripts.seed_long_conversation --count 300
    uv run python -m scripts.seed_long_conversation --count 1000 --project-id <uuid>

Prints the conversation id to open in the UI or hit
`GET /api/v1/conversations/<id>/items` directly. Delete it afterward via
`DELETE /api/v1/conversations/<id>` — these are throwaway fixtures, not
meant to be reused across runs.
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine

# Every model module must be imported before any mapper is used — SQLModel
# relationships resolve their target class by name against the whole
# registry, not just what this script itself imports (same reason
# tests/conftest.py's db_schema fixture does the same full-package import).
import cowork.models as _models_pkg

for _, _name, _ in pkgutil.iter_modules(_models_pkg.__path__):
    importlib.import_module(f"cowork.models.{_name}")

from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.schemas.responses import Role
from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID

_TOOL_TURN_EVERY = 6  # every Nth turn also carries a hidden tool_use/tool_result pair


def _tool_rows(conversation_id: UUID, seq: int, created_at: datetime) -> list[Message]:
    call_id = f"seed-tool-{seq}"
    return [
        Message(
            conversation_id=conversation_id,
            role=Role.assistant,
            content=[{"type": "tool_use", "id": call_id, "name": "recall_skill", "input": {"name": "seed"}}],
            seq=seq,
            created_at=created_at,
        ),
        Message(
            conversation_id=conversation_id,
            role=Role.user,
            content=[{"type": "tool_result", "tool_use_id": call_id, "content": "seeded tool output"}],
            seq=seq + 1,
            created_at=created_at,
        ),
    ]


def _ensure_project(session: Session, project_id: UUID | None) -> UUID:
    if project_id is not None:
        project = session.get(Project, project_id)
        if project is None:
            raise SystemExit(f"no project with id {project_id}")
        return project.id
    project = session.get(Project, GENERAL_PROJECT_ID)
    if project is not None:
        return project.id
    project = Project(id=GENERAL_PROJECT_ID, name=GENERAL_PROJECT, path="/tmp/cowork-seed-general")
    session.add(project)
    session.commit()
    return project.id


def seed(session: Session, *, count: int, project_id: UUID | None) -> UUID:
    resolved_project_id = _ensure_project(session, project_id)
    conversation = Conversation(project_id=resolved_project_id, topic=f"ENG-2768 seed ({count} messages)")
    session.add(conversation)
    session.commit()
    session.refresh(conversation)

    base_time = datetime.now(timezone.utc) - timedelta(minutes=count)
    rows: list[Message] = []
    seq = 0
    turn = 0
    while len(rows) < count:
        turn += 1
        minute = timedelta(minutes=turn)
        rows.append(Message(
            conversation_id=conversation.id, role=Role.user,
            content=f"seed message {turn}: what's the status of item {turn}?",
            seq=seq, created_at=base_time + minute,
        ))
        seq += 1
        if turn % _TOOL_TURN_EVERY == 0:
            tool_rows = _tool_rows(conversation.id, seq, base_time + minute)
            rows.extend(tool_rows)
            seq += len(tool_rows)
        rows.append(Message(
            conversation_id=conversation.id, role=Role.assistant,
            content=f"seed answer {turn}: item {turn} is on track, seeded for pagination testing.",
            seq=seq, created_at=base_time + minute,
        ))
        seq += 1

    del rows[count:]
    session.add_all(rows)
    session.commit()
    return conversation.id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300, help="total message rows to seed (default: 300)")
    parser.add_argument("--project-id", type=UUID, default=None, help="existing project id (default: the general project)")
    args = parser.parse_args()

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        conversation_id = seed(session, count=args.count, project_id=args.project_id)

    print(conversation_id)


if __name__ == "__main__":
    main()
