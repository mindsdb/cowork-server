"""The project-file permission rule follows the "Act first, ask later" setting.

With the setting on, a blanket ask-first rule made the agent ask before
creating a file the user had just asked it to create.
"""
import inspect

import pytest

from cowork.harnesses.anton_harness import harness
from cowork.harnesses.anton_harness.harness import _project_file_actions_rule

READ_RULE = "You can read, search, extract from and analyse any of these project files without asking. "

ACT_FIRST_RULE = READ_RULE + (
    "Create the files the user asks for and change the files they ask you to change "
    "without asking for permission first. "
    "Ask the user before deleting files or overwriting files they did not ask you to change. "
)

ASK_FIRST_RULE = READ_RULE + (
    "Ask the user for permission before you create, change, move or delete any of these files. "
)


def test_act_first_rule_text():
    assert _project_file_actions_rule(True) == ACT_FIRST_RULE


def test_ask_first_rule_text():
    assert _project_file_actions_rule(False) == ASK_FIRST_RULE


@pytest.mark.parametrize("act_first", [True, False])
def test_reading_needs_no_permission(act_first):
    assert _project_file_actions_rule(act_first).startswith(READ_RULE)


def test_project_context_uses_the_session_setting():
    source = inspect.getsource(harness.AntonHarness._build_chat_session)
    assert "_project_file_actions_rule(anton_settings.act_first)" in source
    assert "ask the user for permission first" not in source
    assert "You can freely read" not in source
