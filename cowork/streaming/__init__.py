"""Detached, resumable turn streaming.

A turn runs as a detached producer (registry) that writes ordered events
to a StreamBuffer (file now, Redis WIP for cloud). Readers tail from a
`from_seq` offset and live-tail, so the frontend can disconnect and
resume from where it left off while the server keeps working.
"""
from cowork.streaming.backend import get_backend, get_streams_dir, new_buffer
from cowork.streaming.buffer import StreamBuffer
from cowork.streaming.records import TerminalReason, TurnRecord
from cowork.streaming.registry import RunHandle, RunRegistry, registry

__all__ = [
    "StreamBuffer",
    "TurnRecord",
    "TerminalReason",
    "RunRegistry",
    "RunHandle",
    "registry",
    "new_buffer",
    "get_backend",
    "get_streams_dir",
]
