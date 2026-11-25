"""A scheduled run can explicitly report that an excluded connector stopped it."""

import json

REPORT_BLOCKED_CONNECTOR = "report_blocked_connector"


def build_blocked_connector_tool(excluded: list[dict]):
    from anton.core.tools.tool_defs import ToolDef

    connections = {(entry["engine"], entry["name"]) for entry in excluded}

    async def report(_session, arguments: dict) -> str:
        engine, name = arguments.get("engine"), arguments.get("name")
        if not isinstance(engine, str) or not isinstance(name, str) or (engine, name) not in connections:
            return "Choose a connector that this task excludes."
        return json.dumps({"blocked_connector": f"{name} ({engine})"})

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
