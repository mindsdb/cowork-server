"""Tests for `cowork.common.history_scrub`, the shared scrub every path that
replays persisted history into an LLM (gate, remote/delegated turns, anton
and hermes harness replay) is supposed to use instead of rolling its own.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from anton.utils.datasources import _DS_KNOWN_VARS, _DS_SECRET_VARS
from cowork.common.history_scrub import scrub_message_dict, scrubbed_openai_dump


@pytest.fixture(autouse=True)
def clean_ds_state():
    def _clean():
        _DS_SECRET_VARS.clear()
        _DS_KNOWN_VARS.clear()
        for k in list(os.environ):
            if k.startswith("DS_"):
                del os.environ[k]

    _clean()
    yield
    _clean()


def test_scrubs_a_shape_matching_key_from_string_content():
    om = {"role": "user", "content": "my key is sk-" + "a" * 30}
    assert "sk-" not in scrub_message_dict(om)["content"]


def test_scrubs_a_vaulted_ds_secret_from_string_content(monkeypatch):
    _DS_SECRET_VARS.add("DS_POSTGRES_ABC12__PASSWORD")
    monkeypatch.setenv("DS_POSTGRES_ABC12__PASSWORD", "SuperSecret123!")
    om = {"role": "user", "content": "connect postgres, password: SuperSecret123!"}

    result = scrub_message_dict(om)

    assert "SuperSecret123!" not in result["content"]
    assert "[DS_POSTGRES_ABC12__PASSWORD]" in result["content"]


def test_tool_blocks_pass_through_untouched():
    om = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {"code": "select 1"}},
    ]}
    assert scrub_message_dict(om)["content"] == om["content"]


def test_scrubs_a_secret_in_a_text_block_sibling_to_a_tool_call():
    """The gap the PR review caught: a text block sitting next to a tool_use
    block in the same message reached the gate unscrubbed, because the old
    guard skipped scrubbing entirely for any list-shaped content."""
    leaked = "sk-" + "b" * 30
    om = {"role": "assistant", "content": [
        {"type": "text", "text": f"using key {leaked}"},
        {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {}},
    ]}

    result = scrub_message_dict(om)

    assert leaked not in result["content"][0]["text"]
    assert result["content"][1] == om["content"][1]


def test_scrubbed_openai_dump_forwards_dump_kwargs():
    leaked = "sk-" + "c" * 30
    m = SimpleNamespace(
        to_openai_message=lambda: SimpleNamespace(
            model_dump=lambda **kw: {"role": "user", "content": f"key {leaked}", "mode_seen": kw}
        )
    )

    result = scrubbed_openai_dump(m, mode="json")

    assert leaked not in result["content"]
    assert result["mode_seen"] == {"mode": "json"}
