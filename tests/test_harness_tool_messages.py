from __future__ import annotations

import inspect
from dataclasses import dataclass

from cowork.harnesses.anton_harness import harness
from cowork.harnesses.anton_harness.harness import _tool_message_kwargs
from cowork.harnesses.base import ChannelContext


@dataclass
class _Config:
    tool_messages: bool = False


@dataclass
class _OldConfig:
    other: int = 0


def test_a_ui_turn_renders_tool_messages():
    assert _tool_message_kwargs(_Config, None) == {"tool_messages": True}


def test_a_channel_turn_does_not():
    ctx = ChannelContext(channel_type="slack")
    assert _tool_message_kwargs(_Config, ctx) == {"tool_messages": False}


def test_an_anton_without_the_field_gets_nothing():
    assert _tool_message_kwargs(_OldConfig, None) == {}


def test_the_session_builder_passes_it():
    source = inspect.getsource(harness.AntonHarness._build_chat_session)
    assert "**_tool_message_kwargs(ChatSessionConfig, channel_context)" in source
