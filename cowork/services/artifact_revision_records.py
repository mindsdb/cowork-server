"""The records an artifact's revision journal holds, each checked by one model.

The journal (artifact_journal) sits in a folder the agent's code can write, so
every record is checked here before the server uses a field of it: a
revision's content hash names a blob file, a repair's id names its record
file, and a path names the source that recovery or a rejection writes. A
record that fails its model is one the server never wrote, and reads as
damaged.

The keys on disk are the ones the server has written since revisions began. A
record keeps any key it doesn't declare, and writes back only the keys it was
read or made with plus those the code set, in the order the server writes
them, so a journal reads and writes the same across builds.
"""
from __future__ import annotations

import re
import uuid
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from pydantic.alias_generators import to_camel

# A blob's name is the sha256 of its bytes. Matched with fullmatch: `$` would
# also accept a trailing newline.
_CONTENT_HASH = re.compile(r"[0-9a-f]{64}")


def is_content_hash(value: object) -> bool:
    """Whether ``value`` can name a blob: 64 lowercase hex characters."""
    return isinstance(value, str) and _CONTENT_HASH.fullmatch(value) is not None


def is_repair_id(value: object) -> bool:
    """Whether ``value`` is a repair id the server could have minted: a UUID in
    its canonical spelling, which is also safe as a file name. A repair record
    is a file the agent's code can write, so its id is checked before it names
    a file."""
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _content_hash(value: str) -> str:
    if not is_content_hash(value):
        raise ValueError("A content hash is 64 lowercase hex characters")
    return value


def _repair_id(value: str) -> str:
    if not is_repair_id(value):
        raise ValueError("A repair id is a UUID in its canonical spelling")
    return value


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("A path is never blank")
    return value


ContentHash = Annotated[str, AfterValidator(_content_hash)]
RepairId = Annotated[str, AfterValidator(_repair_id)]
NonBlankPath = Annotated[str, AfterValidator(_non_blank)]

RepairStatus = Literal[
    "queued",
    "ready",
    "conflict",
    "no_change",
    "cancelled",
    "discarded",
    "accepted",
    "rejected",
]


class JournalRecord(BaseModel):
    """A record as the journal holds it: camelCase keys on disk, any key the
    model doesn't declare kept, and a field set after the read checked too."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="allow",
        validate_assignment=True,
    )

    def to_json(self) -> dict[str, Any]:
        """The record as the journal and the API hold it: the keys it was read
        or made with, and those the code set since."""
        return self.model_dump(mode="json", by_alias=True, exclude_unset=True)


class RevisionActor(JournalRecord):
    """Who made a revision: ``system``, ``manual`` or ``agent``, and the
    person's id for a manual one."""

    kind: str
    id: str | None = None


class RevisionEntry(JournalRecord):
    """One revision in the manifest: its saved bytes are ``blobs/<contentHash>``.

    Fields the server always writes but never reads back stay optional, so a
    record written without them, as older test fixtures and repairs are, still
    reads.
    """

    id: str
    number: int = 0
    artifact_id: str | None = None
    path: NonBlankPath
    created_at: str | None = None
    actor: RevisionActor | None = None
    summary: str = ""
    base_revision_id: str | None = None
    content_hash: ContentHash
    size: int | None = Field(default=None, alias="bytes")
    conversation_id: str | None = None
    comment_thread_ids: list[str] = []


def _entry_or_none(raw: object) -> RevisionEntry | None:
    try:
        return RevisionEntry.model_validate(raw)
    except ValidationError:
        return None


class Manifest(JournalRecord):
    """``manifest.json``: the artifact's revisions, oldest first.

    An entry that fails RevisionEntry is left out, so it answers as a missing
    revision does, and the next write of the manifest drops it. The manifest
    itself must be an object with a ``revisions`` list.
    """

    version: int = 1
    revisions: list[RevisionEntry]
    #: How many entries the read left out. Never saved.
    skipped: int = Field(default=0, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _leave_out_damaged_entries(cls, data: object) -> object:
        if not isinstance(data, dict) or not isinstance(data.get("revisions"), list):
            # The field types refuse it.
            return data
        entries = [_entry_or_none(raw) for raw in data["revisions"]]
        kept = [entry for entry in entries if entry is not None]
        return {**data, "revisions": kept, "skipped": len(entries) - len(kept)}

    def head(self, path: str) -> RevisionEntry | None:
        """The newest revision of the source at ``path``."""
        return next((entry for entry in reversed(self.revisions) if entry.path == path), None)

    def revision(self, revision_id: str | None) -> RevisionEntry | None:
        return next((entry for entry in self.revisions if entry.id == revision_id), None)


class PendingSourceWrite(JournalRecord):
    """``pending-source-write.json``: a source save the server began, which
    recovery finishes or abandons on the next read under the lock."""

    version: Literal[1]
    path: NonBlankPath
    before_content_hash: ContentHash
    revision: RevisionEntry

    @model_validator(mode="after")
    def _names_one_path(self) -> PendingSourceWrite:
        if self.revision.path != self.path:
            raise ValueError("A pending write and its revision name one path")
        return self


class AgentRepair(JournalRecord):
    """``repairs/<id>.json``: one agent repair handoff and where it stands.

    Fields the server always writes but never reads back stay optional, as on
    RevisionEntry.
    """

    id: RepairId
    artifact_id: str | None = None
    path: NonBlankPath
    base_revision_id: str | None = None
    base_content_hash: str | None = None
    comment_thread_id: str | None = None
    selector: str | None = None
    thread: list[Any] | None = None
    conversation_id: str | None = None
    status: RepairStatus
    revision_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
