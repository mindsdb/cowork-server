"""The messages cowork-server and an export's conversion process exchange.

The server starts one conversion process per export
(``cowork.services.export_child``) and talks to it over its standard input and
output, one frame per message: the message as JSON, then the raw bytes it
carries, its payload. The process converts; the server reads every file the
conversion needs and writes the result. Nothing here is pickled, so a process
that a hostile document took over can send only these shapes, which the
server validates before it uses a field.
"""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from cowork.schemas.artifact_export import ExportFormat, InlineLimits, OmittedReason


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ConvertRequest(_Message):
    """Convert one source. The payload is the source file's bytes, as read."""

    type: Literal["convert"] = "convert"
    export_format: ExportFormat
    # The source's suffix, lowercased, such as ".md", and its name without it.
    suffix: str
    title: str
    # The folder of the source inside the artifact folder. Every file the
    # conversion asks for sits below it.
    base_parts: tuple[str, ...]
    limits: InlineLimits


class AssetRequest(_Message):
    """The conversion asks for one file of the artifact, by its path inside the folder."""

    type: Literal["read"] = "read"
    parts: tuple[str, ...]


class AssetReply(_Message):
    """The server's answer to an ``AssetRequest``. The payload is the file's bytes.

    A refused read carries its reason and no bytes. ``device`` and ``inode``
    name the file read, so the conversion can find a stylesheet that imports
    itself.
    """

    type: Literal["asset"] = "asset"
    refused: OmittedReason | None = None
    device: int = 0
    inode: int = 0


class Converted(_Message):
    """The finished conversion. The payload is the output.

    The output is the Word or HTML file, or, for a PDF, the inlined HTML the
    render service gets. ``denied_events`` counts the network and process
    starts the conversion process refused; any is a failure.
    """

    type: Literal["converted"] = "converted"
    omitted_count: int
    scripts_dropped: bool
    denied_events: int


class ConversionFailed(_Message):
    """The conversion raised. The server logs it and answers 500."""

    type: Literal["failed"] = "failed"
    error_type: str
    detail: str


ServerMessage = Annotated[ConvertRequest | AssetReply, Field(discriminator="type")]
ChildMessage = Annotated[AssetRequest | Converted | ConversionFailed, Field(discriminator="type")]


class ChildLimits(BaseModel, frozen=True):
    """What one conversion process may spend before the server kills it.

    ``memory_bytes`` is the address space the process may map on Linux, and
    the memory the server lets it hold everywhere.
    """

    wall_seconds: float
    memory_bytes: int
