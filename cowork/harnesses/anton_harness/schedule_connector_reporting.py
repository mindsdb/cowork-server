"""A scheduled run can explicitly report that an excluded connector stopped it."""

import json

REPORT_BLOCKED_CONNECTOR = "report_blocked_connector"


def reported_connector(excluded: list[dict], arguments: object) -> str | None:
    """The excluded connector a report call names, as the run's record shows
    it. None when the call names anything the task does not exclude."""
    if not isinstance(arguments, dict):
        return None
    engine, name = arguments.get("engine"), arguments.get("name")
    if not isinstance(engine, str) or not isinstance(name, str):
        return None
    if (engine, name) not in {(entry["engine"], entry["name"]) for entry in excluded}:
        return None
    return f"{name} ({engine})"


def build_blocked_connector_tool(excluded: list[dict]):
    from anton.core.tools.tool_defs import ToolDef

    async def report(_session, arguments: dict) -> str:
        label = reported_connector(excluded, arguments)
        if label is None:
            return "Choose a connector that this task excludes."
        return json.dumps({"blocked_connector": label})

    return ToolDef(
        name=REPORT_BLOCKED_CONNECTOR,
        description=(
            "Report that this scheduled task cannot finish because it requires a connector "
            "excluded in its settings. Call this only when you cannot finish with the "
            "allowed connectors. Then end the run and explain which connector it needed."
        ),
        input_schema={
            "type": "object",
            "properties": {"engine": {"type": "string"}, "name": {"type": "string"}},
            "required": ["engine", "name"],
            "additionalProperties": False,
        },
        handler=report,
    )
