"""Coroutine stand-ins built from plain functions."""
from __future__ import annotations


def awaitable(fn):
    """``fn`` as a coroutine function, called when awaited, for a stub of an
    async function such as the artifact resolver or the owner gate."""

    async def stub(*args, **kwargs):
        return fn(*args, **kwargs)

    return stub
