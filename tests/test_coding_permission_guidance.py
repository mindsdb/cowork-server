from __future__ import annotations

from pathlib import Path

import pytest

from cowork.coding.contracts import PermissionMode
from cowork.coding.engines.base import EngineSessionConfig
from cowork.coding.engines.codex_config import prepare_launch
from cowork.coding.engines.permission_guidance import READ_ONLY_MESSAGE, permission_guidance


def guidance(mode=PermissionMode.workspace, **kwargs):
    return permission_guidance(EngineSessionConfig(model="gpt", permission_mode=mode, **kwargs))


def test_read_only_uses_the_requested_message_for_patches_and_commands():
    message = guidance(PermissionMode.read_only)
    assert READ_ONLY_MESSAGE == (
        "The agent is running in read-only mode. To edit files, switch to Ask first "
        "or Workspace Auto or Full Access"
    )
    assert f"File writes (including writes made by commands): {READ_ONLY_MESSAGE}" in message
    assert "To edit it, add it under" not in message


@pytest.mark.parametrize("mode", list(PermissionMode))
def test_plan_mode_recovery_does_not_suggest_that_full_access_bypasses_planning(mode):
    message = guidance(mode, task_mode="plan")
    assert "To edit files, turn Plan mode off." in message
    assert ("Then switch to Ask first or Workspace Auto or Full Access." in message) == (mode == PermissionMode.read_only)
    assert READ_ONLY_MESSAGE not in message
    assert "Full Access has no runtime filesystem" not in message


def test_workspace_distinguishes_outside_writes_protected_paths_and_allowed_reads():
    message = guidance()
    assert "Task controls → Additional folders, or switch to Ask first or Full Access." in message
    assert "This file is protected by the task’s permissions." in message
    assert "It does not forbid reading outside" in message
    assert "It cannot request approval for an exception." in message


def test_ask_first_requests_approval_instead_of_recommending_a_wider_mode():
    message = guidance(PermissionMode.supervised)
    assert "Ask first allows normal workspace edits." in message
    assert "use the runtime's approval request" in message
    assert "or switch to Ask first or Full Access" not in message


@pytest.mark.parametrize("mode", list(PermissionMode))
@pytest.mark.parametrize("task_mode", ["build", "plan"])
@pytest.mark.parametrize("network_access", [False, True])
def test_network_remedy_matches_the_effective_runtime_policy(mode, task_mode, network_access):
    config = EngineSessionConfig(model="gpt", permission_mode=mode, task_mode=task_mode, network_access=network_access)
    launch = prepare_launch(config, Path("/workspace"), "http://127.0.0.1:1234")
    allowed = launch.sandbox_policy["type"] == "dangerFullAccess" or launch.sandbox_policy["networkAccess"]
    assert ("Network access is off for this task." in permission_guidance(config)) == (not allowed)


@pytest.mark.parametrize("enabled", [False, True])
def test_web_search_is_separate_from_network_access_even_in_full_access(enabled):
    message = guidance(PermissionMode.full_access, web_search=enabled, network_access=True)
    assert ("Web search is off for this task." in message) == (not enabled)
    assert "Network access is off for this task." not in message


@pytest.mark.parametrize("mode", list(PermissionMode))
def test_permission_errors_have_specific_recovery_without_encouraging_bypasses(mode):
    message = guidance(mode)
    assert "This action was not approved." in message
    assert "ask me to try again and approve the request" in message
    assert "Task controls → Windows sandbox → Set up" in message
    assert "Your computer denied access" in message
    assert "Reconnect it in Connectors" in message
    assert "Ask its owner or your administrator for access" in message
    assert "Do not recommend Full Access for operating-system, sign-in, account or organisation restrictions." in message
    assert "Never change permissions yourself or retry through another tool to bypass" in message
    assert "If the cause is uncertain, say so" in message
    assert "distinguish completed work from blocked work" in message
    assert "Task controls opens with /permissions" in message


@pytest.mark.parametrize("project_instructions", ["", "Use the team playbook."])
def test_launch_preserves_project_instructions_and_refreshes_guidance_for_changed_permissions(project_instructions):
    config = EngineSessionConfig(model="gpt", permission_mode=PermissionMode.read_only, developer_instructions=project_instructions)
    first = prepare_launch(config, Path("/workspace"), "http://127.0.0.1:1234")
    assert first.thread_params["developerInstructions"] == "\n\n".join(
        item for item in (project_instructions, permission_guidance(config)) if item
    )
    new_config = EngineSessionConfig(model="gpt", permission_mode=PermissionMode.full_access, developer_instructions=project_instructions)
    resumed = prepare_launch(new_config, Path("/workspace"), "http://127.0.0.1:1234")
    assert READ_ONLY_MESSAGE not in resumed.thread_params["developerInstructions"]
    assert "Full Access has no runtime filesystem" in resumed.thread_params["developerInstructions"]
    assert config.developer_instructions == project_instructions
