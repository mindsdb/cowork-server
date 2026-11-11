"""The metadata.json document of one artifact folder, read through one model.

anton writes this file through its `Artifact` model
(anton.core.artifacts.models.Artifact), and the keys here are the ones it
writes, except two the server keeps for itself: `title`, which only the share
email reads, and `stableId`, the retired second id field that id widening
removes.

The folder is agent-writable, so nothing guarantees anton's model checked the
file. A declared field of the wrong type reads as missing, so the artifact
lists and opens with the fallback a missing field gets. The identity fields
pass through unchanged for `_resolved_id` to judge.

Writes never come from the fields: the model keeps the document it was
validated from, and a write changes one key of that document. Keys the
server doesn't declare, values of the wrong type and the key order survive.
The model is frozen and refuses `model_copy(update=...)`, so it never drifts
from its document. Don't hash it: an identity field may hold a list.
"""
from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PrivateAttr,
    ValidatorFunctionWrapHandler,
    model_validator,
)


def lenient_text(value: Any) -> str | None:
    """``value`` when it is text a JSON response can carry, else None."""
    # A string that can't be encoded (a lone surrogate) would break every
    # JSON response that carries it.
    if not isinstance(value, str):
        return None
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return value


def _port(value: Any) -> int | None:
    # `type(...) is int` keeps out bool; strings and floats never coerce.
    if type(value) is not int:
        return None
    return value if 0 < value < 65536 else None


LenientStr = Annotated[str | None, BeforeValidator(lenient_text)]
LenientPort = Annotated[int | None, BeforeValidator(_port)]


class DatasourceRef(BaseModel):
    """A connection a fullstack artifact's backend reads (``datasources[]``)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    engine: LenientStr = None
    name: LenientStr = None


class ProvenanceEntry(BaseModel):
    """One conversation that touched the artifact (``provenance[]``)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    conversation: LenientStr = None


def _datasource_items(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _provenance_items(value: Any) -> list[Any]:
    # An entry that isn't an object stays in place as None: the first entry
    # names the creating conversation, so entries must not shift.
    if not isinstance(value, list):
        return []
    return [item if isinstance(item, dict) else None for item in value]


class MetadataNotAnObject(ValueError):
    """A metadata.json that parses but is not a JSON object. Such a folder
    holds no artifact id."""


class ArtifactMetadata(BaseModel):
    """An artifact's metadata.json as the server reads it (see the module
    docstring)."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    # frozen=True would generate a hash that fails on the list fields.
    __hash__ = None  # type: ignore[assignment]

    id: Any = None
    stableId: Any = None
    createdAt: Any = None
    slug: LenientStr = None
    name: LenientStr = None
    title: LenientStr = None
    description: LenientStr = None
    type: LenientStr = None
    primary: LenientStr = None
    port: LenientPort = None
    datasources: Annotated[list[DatasourceRef], BeforeValidator(_datasource_items)] = []
    provenance: Annotated[list[ProvenanceEntry | None], BeforeValidator(_provenance_items)] = []

    _document: dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def _keep_document(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> ArtifactMetadata:
        if isinstance(data, cls):
            return handler(data)
        if not isinstance(data, dict):
            raise MetadataNotAnObject("Artifact metadata is not an object")
        model = handler(data)
        model._document = dict(data)
        return model

    @classmethod
    def parse(cls, text: str) -> ArtifactMetadata:
        """The model of a metadata.json text. Raises ValueError for text that
        is not JSON, a document that is not an object, and one nested too
        deeply to parse."""
        try:
            document = json.loads(text)
        except RecursionError as exc:
            raise ValueError("Artifact metadata nests too deeply") from exc
        if not isinstance(document, dict):
            raise MetadataNotAnObject("Artifact metadata is not an object")
        return cls.model_validate(document)

    @classmethod
    def empty(cls) -> ArtifactMetadata:
        return cls.model_validate({})

    def with_value(self, key: str, value: Any) -> ArtifactMetadata:
        """The document with ``key`` set to ``value``: an existing key keeps
        its place, a new one goes last."""
        return type(self).model_validate({**self._document, key: value})

    def with_id(self, artifact_id: str) -> ArtifactMetadata:
        """The document with its id widened: ``id`` set and the retired
        ``stableId`` dropped, in one validation."""
        document = {k: v for k, v in self._document.items() if k != "stableId"}
        document["id"] = artifact_id
        return type(self).model_validate(document)

    def to_document(self) -> dict[str, Any]:
        """A copy of the document's top level. Nested values are shared with
        the model, so treat them as read-only."""
        return dict(self._document)

    def encoded(self) -> bytes:
        """The document as metadata.json holds it. Raises ValueError for one
        that can't be written back, such as one holding a lone surrogate."""
        try:
            text = json.dumps(self._document, indent=2, ensure_ascii=False) + "\n"
            return text.encode("utf-8")
        except (UnicodeEncodeError, RecursionError) as exc:
            raise ValueError("Artifact metadata can't be written back") from exc

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> ArtifactMetadata:
        if update:
            raise TypeError("ArtifactMetadata changes only through with_value or with_id")
        return super().model_copy(deep=deep)
