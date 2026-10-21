"""Write an inlined document as a Word file, from bytes only."""
from cowork.schemas.artifact_export import InlinedDocument


def write_docx(*, document: InlinedDocument, title: str) -> bytes:
    raise NotImplementedError
