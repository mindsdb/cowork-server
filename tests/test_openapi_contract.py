from __future__ import annotations

from typing import Any

from cowork.server import create_app


def _response(
    schema: dict[str, Any],
    path: str,
    method: str,
    status: int,
) -> dict[str, Any]:
    return schema["paths"][path][method]["responses"][str(status)]


def _content_schema(
    schema: dict[str, Any],
    path: str,
    method: str,
    status: int,
    media_type: str = "application/json",
) -> dict[str, Any]:
    return _response(schema, path, method, status)["content"][media_type]["schema"]


def _assert_ref(value: dict[str, Any], component: str) -> None:
    assert value == {"$ref": f"#/components/schemas/{component}"}


def test_task_aliases_advertise_their_json_responses():
    schema = create_app().openapi()

    for prefix in ("conversations", "tasks"):
        base = f"/api/v1/{prefix}"
        expected = (
            (f"{base}/", "get", 200, "ConversationListResponse"),
            (f"{base}/", "post", 201, "ConversationListItem"),
            (f"{base}/{{conversation_id}}", "get", 200, "ConversationListItem"),
            (f"{base}/{{conversation_id}}", "patch", 200, "ConversationListItem"),
            (
                f"{base}/{{conversation_id}}/move",
                "post",
                200,
                "ConversationListItem",
            ),
            (
                f"{base}/{{conversation_id}}",
                "delete",
                200,
                "ConversationDeleteResponse",
            ),
            (
                f"{base}/{{conversation_id}}/turns/{{turn_index}}",
                "delete",
                200,
                "ConversationTurnDeleteResponse",
            ),
        )
        for path, method, status, component in expected:
            _assert_ref(
                _content_schema(schema, path, method, status),
                component,
            )

        message_list = _content_schema(
            schema,
            f"{base}/{{conversation_id}}/items",
            "get",
            200,
        )
        assert message_list["type"] == "array"
        _assert_ref(message_list["items"], "ConversationMessageResponse")

    item = schema["components"]["schemas"]["ConversationListItem"]
    assert {"projectId", "projectPath", "reasoningEffort"} <= set(
        item["properties"]
    )
    assert {"project_id", "project_path", "reasoning_effort"}.isdisjoint(
        item["properties"]
    )
    message = schema["components"]["schemas"]["ConversationMessageResponse"]
    assert "created_at" in message["properties"]


def test_responses_api_advertises_json_and_event_stream_contracts():
    schema = create_app().openapi()
    base = "/api/v1/responses"

    create_content = _response(schema, f"{base}/", "post", 200)["content"]
    _assert_ref(create_content["application/json"]["schema"], "Response")
    assert create_content["text/event-stream"]["schema"] == {"type": "string"}

    expected = (
        ("in-flight-list", "InFlightListResponse"),
        ("in-flight", "InFlightStatusResponse"),
        ("cancel", "CancelResponse"),
    )
    for suffix, component in expected:
        method = "post" if suffix == "cancel" else "get"
        _assert_ref(
            _content_schema(schema, f"{base}/{suffix}", method, 200),
            component,
        )

    tail_content = _response(schema, f"{base}/tail", "get", 200)["content"]
    assert tail_content == {"text/event-stream": {"schema": {"type": "string"}}}


def test_schedule_api_advertises_statuses_and_camel_case_responses():
    schema = create_app().openapi()
    base = "/api/v1/schedules"
    expected = (
        (f"{base}/", "get", 200, "ScheduleListResponse"),
        (f"{base}/", "post", 201, "ScheduleResponse"),
        (f"{base}/{{schedule_id}}", "get", 200, "ScheduleResponse"),
        (f"{base}/{{schedule_id}}", "put", 200, "ScheduleResponse"),
        (f"{base}/{{schedule_id}}", "patch", 200, "ScheduleResponse"),
        (f"{base}/{{schedule_id}}/pause", "post", 200, "ScheduleResponse"),
        (f"{base}/{{schedule_id}}/resume", "post", 200, "ScheduleResponse"),
        (
            f"{base}/{{schedule_id}}/run-now",
            "post",
            202,
            "ScheduleTriggerResponse",
        ),
        (
            f"{base}/{{schedule_id}}/runs",
            "get",
            200,
            "ScheduleRunListResponse",
        ),
    )
    for path, method, status, component in expected:
        _assert_ref(_content_schema(schema, path, method, status), component)

    assert "content" not in _response(
        schema,
        f"{base}/{{schedule_id}}",
        "delete",
        204,
    )
    schedule = schema["components"]["schemas"]["ScheduleResponse"]
    assert {"nextRunAt", "lastRunAt", "projectId"} <= set(schedule["properties"])
    assert {"next_run_at", "last_run_at", "project_id"}.isdisjoint(
        schedule["properties"]
    )


def test_artifact_and_search_operations_advertise_response_shapes():
    schema = create_app().openapi()

    artifacts = _content_schema(schema, "/api/v1/artifacts/", "get", 200)
    assert artifacts["type"] == "array"
    _assert_ref(artifacts["items"], "ArtifactCardResponse")
    _assert_ref(
        _content_schema(schema, "/api/v1/artifacts/preview", "get", 200),
        "ArtifactPreviewResponse",
    )
    _assert_ref(
        _content_schema(schema, "/api/v1/artifacts/open", "post", 200),
        "ArtifactOpenResponse",
    )
    served = _response(
        schema,
        "/api/v1/artifacts/serve/{project_name}/{file_path}",
        "get",
        200,
    )["content"]
    assert served == {
        "application/octet-stream": {
            "schema": {"type": "string", "format": "binary"}
        }
    }

    card = schema["components"]["schemas"]["ArtifactCardResponse"]
    assert {"fileCount", "projectId", "draftUrl", "capabilities"} <= set(
        card["properties"]
    )
    assert {"file_count", "project_id", "draft_url"}.isdisjoint(
        card["properties"]
    )

    _assert_ref(
        _content_schema(schema, "/api/v1/search", "get", 200),
        "SearchResponse",
    )
    search_result = schema["components"]["schemas"]["SearchResultResponse"]
    assert set(search_result["properties"]) == {
        "type",
        "id",
        "title",
        "subtitle",
        "route",
        "score",
    }
