"""The in-process agent prompt names only tools the locked anton or cowork
registers, as far as this guard reads it: a snake_case tool name of two or
more words in backticks, alone or called, in the project_context,
output_context and skill_output_context sections _build_chat_session
assembles. A one-word name, a name outside backticks, and the other prompt
pieces (the turn style, the project-file rule, the attachment hint and the
integration guidance) are not read.

An anton release can remove a tool (mindsdb/anton#545 removed
open_artifact). A prompt that still tells the agent to call it sends every
in-process turn after a tool that does not exist.
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

from anton.core.tools import tool_defs
from anton.core.tools.tool_defs import ToolDef

from cowork.harnesses.anton_harness import harness, tools

# The prompt sections _build_chat_session assembles into the system prompt.
PROMPT_SECTIONS = frozenset({"project_context", "output_context", "skill_output_context"})
# Names the prompt puts in backticks that are fields of a tool's result,
# not tools.
RESULT_FIELDS = frozenset({"skill_file"})
# A tool as the prompt names one: a snake_case name, alone or called.
_TOOL_NAME = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?:\([^`]*\))?`")


def _prompt_text() -> str:
    """The literal text of the prompt sections, in source order, without the
    values formatted into them."""
    function = ast.parse(textwrap.dedent(inspect.getsource(harness.AntonHarness._build_chat_session))).body[0]
    sections = [
        node.value for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in PROMPT_SECTIONS for target in node.targets)
    ]
    pieces = sorted(
        (
            constant for section in sections for constant in ast.walk(section)
            if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
        ),
        key=lambda constant: (constant.lineno, constant.col_offset),
    )
    return "".join(constant.value for constant in pieces)


def _registered_tools() -> set[str]:
    """The tools the locked anton defines and the ones cowork adds."""
    antons = {value.name for value in vars(tool_defs).values() if isinstance(value, ToolDef)}
    cowork = {
        build().name for build in (
            tools.build_cowork_publish_tool, tools.build_cowork_lookup_connector_tool,
            tools.build_cowork_request_credentials_tool, tools.build_cowork_label_connection_tool,
            tools.build_cowork_create_skill_draft_tool,
        )
    }
    return antons | cowork


def test_the_prompt_names_only_registered_tools():
    named = set(_TOOL_NAME.findall(_prompt_text())) - RESULT_FIELDS

    assert named, "the prompt names no tool; the section names may have changed"
    assert named <= _registered_tools(), f"unregistered: {sorted(named - _registered_tools())}"


def test_the_prompt_points_at_read_text_file_to_read_an_artifact():
    text = _prompt_text()

    assert "`read_text_file`" in text
    assert "open_artifact" not in text
