"""The figures in an answer and the steps they came from, as the
``response.figure_sources`` event and a non-streaming response carry them.
``cowork.harnesses.anton_harness.figure_sources`` finds them."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class FigureOrigin(StrEnum):
    step = "step"
    question = "question"
    none = "none"


class SourceKind(StrEnum):
    query = "query"
    file = "file"
    web = "web"
    memory = "memory"
    step = "step"


class FigureStep(BaseModel):
    """A step at least one figure in the answer names."""

    tool_use_id: str
    kind: SourceKind
    # A short label: the file's name, the web page's host and path, or the
    # step's description. Empty when the step has none, and the client words
    # that case by the kind alone.
    label: str
    # The path, relative to FigureSources.project_id and in POSIX form, of the
    # project file the step read, when it is in the conversation's project
    # and exists when the turn ends.
    path: str | None = None
    # The files-store id of the file attached to the conversation that the
    # step read. At most one of path and attachment_id is set.
    attachment_id: str | None = None
    # The http(s) URL of the web page the step reached, without its user,
    # password, query string or fragment. None when the URL itself holds a
    # credential, such as a webhook's.
    url: str | None = None
    # The SQL statements and "METHOD url-template" requests the step's code
    # shows, in code order, redacted.
    queries: list[str] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)


class FigureSource(BaseModel):
    """A step whose output holds a figure, or that read the data a step whose
    output holds it was computed from, named by its entry in
    ``FigureSources.steps``."""

    tool_use_id: str
    page: int | None = None


class Figure(BaseModel):
    """A figure in the answer and where it came from.

    ``start`` and ``end`` count UTF-16 code units into the answer text, the
    unit the renderer's JavaScript strings index by, so ``text`` is
    ``content.slice(start, end)`` there.
    """

    start: int
    end: int
    text: str
    origin: FigureOrigin
    sources: list[FigureSource] = Field(default_factory=list)


class FigureSources(BaseModel):
    """The figures in an answer and, once each, the steps they name.
    ``project_id`` is the project the turn ran in, which each step's ``path``
    is relative to, or None when it is unknown."""

    project_id: str | None = None
    steps: list[FigureStep] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
