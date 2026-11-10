"""A handcrafted form carries the request Cowork tests it with, as data.

On a single-user install every submittable method needs exactly one valid
``test_request``; these pin the shape rules that keep a placeholder from
moving the host or slipping a credential into literal text.
"""
from __future__ import annotations

import copy
import time

import pytest

from cowork.services.connectors import custom_connectors
from cowork.services.connectors.custom_connectors import UnstorableFormError, check_handcrafted_form, stored_form
from tests._connector_test_request import bearer_test_request

TOKEN_FIELD = {"name": "token", "label": "Token", "type": "password", "secret": True}
FORM = {
    "form_id": "fm_aaaaaaaaaa",
    "title": "Connect Acme",
    "fields": [TOKEN_FIELD],
    "test_request": bearer_test_request(),
}
CLIENT_FORM = {
    "form_id": "fm_bbbbbbbbbb",
    "title": "Connect Planner",
    "fields": [
        {"name": "client_id", "label": "Client id", "type": "text"},
        {"name": "client_secret", "label": "Client secret", "type": "password", "secret": True},
    ],
    "test_request": {
        "token": {
            "method": "POST", "url": "https://auth.example.com/oauth2/token",
            "form": {"grant_type": "client_credentials"},
            "basic_auth": ["{{client_id}}", "{{client_secret}}"],
            "token_field": "access_token",
        },
        "data": {
            "method": "POST", "url": "https://api.example.com/integration/V1/query",
            "headers": {"Authorization": "Bearer {{token}}"},
            "json_body": {"query": "select 1", "limit": 5},
        },
    },
}
OAUTH_METHOD = {
    "id": "oauth", "label": "Sign in", "submit_action": "oauth_launch",
    "oauth": {"auth_url": "https://example.com/auth", "token_url": "https://example.com/token"},
}


def _with(form: dict, path: tuple, value) -> dict:
    """A deep copy of ``form`` with ``value`` set at ``path``."""
    out = copy.deepcopy(form)
    target = out
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return out


def _refused(form: dict, words: str) -> None:
    with pytest.raises(UnstorableFormError) as exc:
        check_handcrafted_form(form)
    assert words in str(exc.value)


class TestEveryMethodNeedsOne:
    def test_a_form_with_a_bearer_request_passes(self):
        check_handcrafted_form(FORM)

    def test_a_client_credentials_api_queried_by_post_passes(self):
        check_handcrafted_form(CLIENT_FORM)

    def test_a_form_without_one_is_refused(self):
        _refused({k: v for k, v in FORM.items() if k != "test_request"}, "needs a test_request")

    def test_a_multi_method_form_missing_one_on_any_method_is_refused(self):
        form = {
            "form_id": "fm_cccccccccc", "title": "Connect Acme",
            "methods": [
                {"id": "key", "label": "Key", "fields": [TOKEN_FIELD], "test_request": bearer_test_request()},
                {"id": "other", "label": "Other", "fields": [TOKEN_FIELD]},
            ],
        }
        _refused(form, "Method 'other' has none")

    def test_a_multi_method_form_with_a_top_level_one_too_is_refused(self):
        form = {
            "form_id": "fm_cccccccccc", "title": "Connect Acme", "test_request": bearer_test_request(),
            "methods": [{"id": "key", "label": "Key", "fields": [TOKEN_FIELD], "test_request": bearer_test_request()}],
        }
        _refused(form, "not at the top level")

    def test_a_hidden_method_is_checked_too(self):
        good = {"id": "key", "label": "Key", "fields": [TOKEN_FIELD], "test_request": bearer_test_request()}
        hidden = {**good, "id": "legacy", "hidden": True}
        _refused({"form_id": "fm_cccccccccc", "title": "Acme", "methods": [good, {k: v for k, v in hidden.items() if k != "test_request"}]}, "has none")
        bad = _with(hidden, ("test_request", "data", "url"), "https://{{token}}.example.com/x")
        _refused({"form_id": "fm_cccccccccc", "title": "Acme", "methods": [good, bad]}, "only in its path and query")

    def test_an_oauth_grant_method_without_one_is_refused(self):
        _refused({"form_id": "fm_dddddddddd", "title": "Acme", "methods": [OAUTH_METHOD]}, "has none")

    def test_an_oauth_grant_method_may_send_the_granted_access_token(self):
        method = {**OAUTH_METHOD, "test_request": bearer_test_request("access_token")}
        check_handcrafted_form({"form_id": "fm_dddddddddd", "title": "Acme", "methods": [method]})

    def test_self_hosted_and_org_mode_keep_todays_rule(self, monkeypatch):
        monkeypatch.setattr(custom_connectors, "tenant_scope_required", lambda: True)
        check_handcrafted_form({k: v for k, v in FORM.items() if k != "test_request"})

    def test_a_stored_definition_keeps_its_request(self):
        stored = stored_form("acme", CLIENT_FORM)
        assert stored["test_request"]["token"]["basic_auth"] == ["{{client_id}}", "{{client_secret}}"]
        assert stored["test_request"]["data"]["json_body"] == {"query": "select 1", "limit": 5}

    def test_an_unknown_key_is_refused(self):
        _refused(_with(FORM, ("test_request", "data", "json"), {"q": 1}), "not valid")


class TestPlaceholdersCannotMoveTheHost:
    @pytest.mark.parametrize("url", [
        "https://{{tenant}}.example.com/v1",
        "https://api.example.com{{path}}",
        "https://api.example.com:{{port}}/v1",
        "{{scheme}}://api.example.com/v1",
        "https://{{token}}@api.example.com/v1",
    ])
    def test_a_placeholder_before_the_path_is_refused(self, url):
        _refused(_with(FORM, ("test_request", "data", "url"), url), "only in its path and query")

    def test_user_info_in_the_url_is_refused(self):
        _refused(_with(FORM, ("test_request", "data", "url"), "https://me:pw@api.example.com/v1"), "user info")

    @pytest.mark.parametrize("url", ["ftp://api.example.com/x", "https:///x", "api.example.com/x"])
    def test_a_url_without_an_http_scheme_and_host_is_refused(self, url):
        _refused(_with(FORM, ("test_request", "data", "url"), url), "http or https URL")

    @pytest.mark.parametrize("url", [
        "https://p.example.com:443:80/d", "https://api.exa\tmple.com/x", "https://api.example.com\n/x",
        "https://api.example.com\u200b/x", "https://api.ex\xadample.com/x",
    ], ids=["two ports", "tab", "newline", "zero width space", "soft hyphen"])
    def test_a_url_the_runner_cannot_parse_is_refused_when_the_form_is_built(self, url):
        with pytest.raises(UnstorableFormError):
            check_handcrafted_form(_with(FORM, ("test_request", "data", "url"), url))

    @pytest.mark.parametrize("url", ["https://api.example.com:99999/rows", "https://api.example.com:0/rows"])
    def test_a_port_outside_the_range_is_refused_when_the_form_is_built(self, url):
        _refused(_with(FORM, ("test_request", "data", "url"), url), "port outside 1 to 65535")

    def test_a_very_long_url_is_checked_quickly(self):
        start = time.monotonic()
        with pytest.raises(UnstorableFormError):
            check_handcrafted_form(_with(FORM, ("test_request", "data", "url"), '"' * 200_000))
        assert time.monotonic() - start < 2

    def test_http_is_allowed_when_the_form_is_built(self):
        check_handcrafted_form(_with(FORM, ("test_request", "data", "url"), "http://127.0.0.1:8765/items"))

    def test_placeholders_in_path_and_query_pass(self):
        check_handcrafted_form(_with(FORM, ("test_request", "data", "url"), "https://api.example.com/v1/{{token}}?k={{token}}"))


class TestWhatTheRequestMayName:
    def test_a_name_that_is_not_a_field_is_refused(self):
        _refused(_with(FORM, ("test_request", "data", "headers"), {"Authorization": "Bearer {{password}}"}), "not form fields")

    def test_token_is_the_issued_token_only_after_a_token_step(self):
        form = _with(CLIENT_FORM, ("test_request",), {"data": CLIENT_FORM["test_request"]["data"]})
        _refused(form, "not form fields")

    def test_a_follow_up_step_may_name_saved_fields(self):
        form = {**_with(FORM, ("test_request", "data", "headers"), {"Authorization": "Bearer {{saved_key}}"}), "_extends_connection": "acme-1"}
        check_handcrafted_form(form)

    def test_the_data_step_must_send_a_credential(self):
        form = {
            "form_id": "fm_eeeeeeeeee", "title": "Acme",
            "fields": [{"name": "account", "label": "Account", "type": "text"}],
            "test_request": {"data": {"method": "GET", "url": "https://api.example.com/{{account}}"}},
        }
        _refused(form, "must send a credential")

    def test_basic_auth_counts_as_a_credential(self):
        form = _with(FORM, ("test_request", "data"), {
            "method": "GET", "url": "https://api.example.com/me", "basic_auth": ["{{token}}", "{{token}}"],
        })
        check_handcrafted_form(form)

    def test_basic_auth_takes_placeholders_only(self):
        form = _with(FORM, ("test_request", "data", "basic_auth"), ["admin", "{{token}}"])
        _refused(form, "basic_auth takes two")

    @pytest.mark.parametrize("text", ["{{ token", "{{token}", "{{1x}}"])
    def test_a_malformed_placeholder_is_refused(self, text):
        _refused(_with(FORM, ("test_request", "data", "headers"), {"Authorization": f"Bearer {text}"}), "malformed")


class TestHeadersAndBodies:
    @pytest.mark.parametrize("name", ["Host", "Connection", "Transfer-Encoding", "Content-Length", "TE", "Upgrade",
                                      "Proxy-Authorization", "X-Forwarded-For"])
    def test_a_header_that_steers_the_connection_is_refused(self, name):
        _refused(_with(FORM, ("test_request", "data", "headers", name), "x"), f"may not set the {name} header")

    @pytest.mark.parametrize("header", [{"X-A": "b\r\nHost: evil"}, {"X-A\nB": "c"}])
    def test_a_line_break_in_a_header_is_refused(self, header):
        _refused(_with(FORM, ("test_request", "data", "headers"), {**header, "Authorization": "Bearer {{token}}"}), "line break")

    def test_a_placeholder_in_a_header_name_is_refused(self):
        _refused(_with(FORM, ("test_request", "data", "headers", "X-{{token}}"), "1"), "header name")

    def test_a_placeholder_in_a_json_key_is_refused(self):
        _refused(_with(CLIENT_FORM, ("test_request", "data", "json_body"), {"{{token}}": 1}), "JSON key")

    def test_a_placeholder_in_a_form_key_is_refused(self):
        _refused(_with(CLIENT_FORM, ("test_request", "token", "form"), {"{{client_id}}": "x"}), "form key")

    def test_a_form_and_a_json_body_together_are_refused(self):
        _refused(_with(CLIENT_FORM, ("test_request", "data", "form"), {"a": "b"}), "not both")

    def test_a_credential_written_into_the_request_is_refused(self):
        literal = "Bearer sk-proj-abcdefghijklmnopqrstuvwx1234"
        _refused(_with(FORM, ("test_request", "data", "headers"), {"Authorization": literal}), "holds a credential value")
