from __future__ import annotations

from cowork.coding.contracts import PermissionMode
from cowork.coding.engines.base import EngineSessionConfig


READ_ONLY_MESSAGE = (
    "The agent is running in read-only mode. To edit files, switch to Ask first "
    "or Workspace Auto or Full Access"
)


def permission_guidance(config: EngineSessionConfig) -> str:
    """Explain refusals in the conversation, including tool failures with no UI event.

    This is response guidance, not enforcement. The runtime sandbox and approval
    policy remain authoritative. Rebuild it when starting or resuming a runtime
    so a permission change cannot leave instructions for the previous mode.
    """
    guidance = [
        "## MindsHub Code permission refusals",
        "These instructions describe this task's current settings and replace any earlier "
        "MindsHub Code permission-refusal guidance. The runtime's effective policy remains authoritative.",
        "When a requested action is blocked by permissions, tell the user what did not happen "
        "and how to unblock it. Use the applicable message below verbatim, then add only necessary "
        "detail (such as the affected file or service). Do not claim an action succeeded when it "
        "was blocked. For partial success, distinguish completed work from blocked work.",
        "Use the effective policy and actual tool results, not a generic 'permission denied' "
        "string, to identify the cause. Do not mistake a timeout, missing file, network outage "
        "or HTTP error for a sandbox restriction. If the cause is uncertain, say so and use safe "
        "read-only checks; do not guess a remedy or recommend broader access.",
        "Never change permissions yourself or retry through another tool to bypass a restriction "
        "or a denied approval. After an approval is declined or expires, wait for the user to "
        "explicitly request another approval attempt. Task controls opens with /permissions; changes apply "
        "to the next turn, after the user clicks Apply.",
    ]
    if config.task_mode == "plan":
        message = "The agent is running in Plan mode. To edit files, turn Plan mode off."
        if config.permission_mode == PermissionMode.read_only:
            message += " Then switch to Ask first or Workspace Auto or Full Access."
        guidance.append(f"File writes are blocked by Plan mode, regardless of the selected permissions: {message}")
    elif config.permission_mode == PermissionMode.read_only:
        guidance.append(f"File writes (including writes made by commands): {READ_ONLY_MESSAGE}")
    elif config.permission_mode == PermissionMode.workspace:
        guidance.extend([
            "Workspace Auto allows writes in the workspace and additional allowed folders, "
            "subject to the runtime's protected-path rules. It does not forbid reading outside "
            "those folders. It cannot request approval for an exception.",
            "Write blocked because its destination is outside the allowed folders: "
            "This folder is outside the task’s allowed folders. To edit it, add it under "
            "Task controls → Additional folders, or switch to Ask first or Full Access.",
            "Write blocked by a protected-path rule, even inside an allowed folder: "
            "This file is protected by the task’s permissions. To change it, switch to "
            "Ask first to approve the change, or Full Access.",
        ])
    elif config.permission_mode == PermissionMode.supervised:
        guidance.append(
            "Ask first allows normal workspace edits. When an action needs an exception, "
            "use the runtime's approval request rather than telling the user to disable protections."
        )
    else:
        guidance.append(
            "Full Access has no runtime filesystem or network restriction. It does not bypass "
            "operating-system permissions or a service's account permissions."
        )
    if not config.network_access and (config.permission_mode != PermissionMode.full_access or config.task_mode == "plan"):
        guidance.append(
            "Command blocked because network access is disabled: Network access is off for this task. "
            "To connect to external services, turn on Network access in Task controls."
        )
    if not config.web_search:
        guidance.append(
            "The web search tool is disabled: Web search is off for this task. "
            "To use it, turn on Web search in Task controls. "
            "This setting is separate from network access for commands."
        )
    guidance.extend([
        "Approval declined or expired: This action was not approved. "
        "If you want to allow it, ask me to try again and approve the request.",
        "Confirmed Windows sandbox setup failure: The Windows sandbox needs attention before "
        "this action can run. Open Task controls → Windows sandbox → Set up, then try again.",
        "Confirmed operating-system access restriction: Your computer denied access to this "
        "file or folder. Check its access permissions or choose a location you can access, then try again.",
        "Confirmed expired or missing connection credentials: The connection to this service "
        "needs attention. Reconnect it in Connectors, then try again.",
        "Confirmed account or organisation access restriction: Your account does not have "
        "permission to access this resource. Ask its owner or your administrator for access, then try again.",
        "Name the affected service/resource for those last two cases. Do not recommend Full Access "
        "for operating-system, sign-in, account or organisation restrictions.",
    ])
    return "\n\n".join(guidance)
