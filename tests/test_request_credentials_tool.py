"""request_credentials: secret-value scrubbing + tool-result guidance."""

import json

import pytest

from cowork.harnesses.anton_harness.tools import (
    _cowork_request_credentials,
    _scrub_secret_values,
)


class TestScrubSecretValues:
    def test_password_value_stripped_top_level(self):
        spec = {
            "fields": [
                {"name": "host", "label": "Host", "type": "text", "value": "db.internal"},
                {"name": "password", "label": "Password", "type": "password", "value": "hunter2"},
            ]
        }
        out = _scrub_secret_values(spec)
        assert out["fields"][0]["value"] == "db.internal"  # non-secret pre-fill kept
        assert "value" not in out["fields"][1]

    def test_password_value_stripped_in_methods(self):
        spec = {
            "methods": [
                {
                    "id": "app-password",
                    "label": "App Password",
                    "fields": [
                        {"name": "email", "label": "Email", "type": "text", "value": "a@b.c"},
                        {"name": "app_password", "label": "App password", "type": "password", "value": "s3cret"},
                    ],
                }
            ]
        }
        out = _scrub_secret_values(spec)
        fields = out["methods"][0]["fields"]
        assert fields[0]["value"] == "a@b.c"
        assert "value" not in fields[1]

    def test_input_not_mutated(self):
        spec = {"fields": [{"name": "p", "label": "P", "type": "password", "value": "x"}]}
        _scrub_secret_values(spec)
        assert spec["fields"][0]["value"] == "x"

    def test_malformed_entries_pass_through(self):
        spec = {"fields": ["not-a-dict", {"name": "x", "label": "X", "type": "text"}], "methods": ["nope"]}
        out = _scrub_secret_values(spec)
        assert out["fields"][0] == "not-a-dict"
        assert out["methods"] == ["nope"]


class TestHandlerResult:
    @pytest.mark.asyncio
    async def test_no_password_value_in_emitted_block(self):
        result = await _cowork_request_credentials(
            session=None,
            tc_input={
                "engine": "postgres",
                "title": "Connect to Postgres",
                "fields": [
                    {"name": "password", "label": "Password", "type": "password", "value": "hunter2"},
                ],
            },
        )
        assert "hunter2" not in result
        block = result.split("```data-vault-form\n", 1)[1].rsplit("\n```", 1)[0]
        spec = json.loads(block)
        assert "value" not in spec["fields"][0]

    @pytest.mark.asyncio
    async def test_result_does_not_reference_unregistered_tools(self):
        result = await _cowork_request_credentials(
            session=None,
            tc_input={"engine": "postgres", "title": "Connect"},
        )
        assert "fetch_submission" not in result
        assert "update_form" not in result
        # replacement guidance aligned with _REQUEST_CREDENTIALS_PROMPT step 4
        assert "your job is done" in result.lower()
