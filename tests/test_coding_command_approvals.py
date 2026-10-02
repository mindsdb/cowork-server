from __future__ import annotations

import shlex

import pytest

from cowork.coding.command_approvals import command_rule

METHOD = "item/commandExecution/requestApproval"


def request(script="npm test", prefix=None, **updates):
    return {
        "command": shlex.join(["/bin/zsh", "-lc", script]),
        "cwd": "/repo with spaces", "environmentId": "local",
        "proposedExecpolicyAmendment": ["npm", "test"] if prefix is None else prefix,
        **updates,
    }


@pytest.mark.parametrize("command", [
    "npm test", "npm test -- --reporter=verbose", 'npm test "file with spaces.js"',
    "npm test 'file with spaces.js'", "npm test -- --testNamePattern=approval",
])
def test_literal_arguments_match_a_previously_approved_prefix(command):
    original = command_rule(METHOD, request())
    candidate = command_rule(METHOD, request(command))
    assert original and candidate and candidate.matches([original.grant])
    assert len(original.grant) == 64
    assert "npm" not in original.grant


@pytest.mark.parametrize("command", [
    "npm test; echo other", "npm test && echo other", "npm test || echo other",
    "npm test | echo other", "npm test &", "npm test > result", "npm test < input",
    "npm test\necho other", "npm test\recho other", "npm test $(echo other)",
    "npm test `echo other`", 'npm test "$TOKEN"', "npm test *.js", "npm test ?.js",
    "npm test ~/file", "npm test {a,b}", "npm test [ab]", "npm test \\; echo other",
    "npm test # comment", "npm test 'unterminated", "npm test <<EOF", "npm test (echo other)",
    "TOKEN=value npm test", "npm test\x00", "npm test\tfile", "npm test \\\nnext",
])
def test_shell_expressions_do_not_offer_or_reuse_a_grant(command):
    assert command_rule(METHOD, request(command)) is None


@pytest.mark.parametrize("updates", [
    {"command": "npm test"}, {"command": ["npm", "test"]}, {"command": None},
    {"command": "powershell.exe -Command 'npm test'"}, {"command": "cmd.exe /c npm test"},
    {"command": "/bin/sh -lc 'npm test'"}, {"command": "/bin/zsh -lic 'npm test'"},
    {"command": "/bin/zsh -lc 'npm test' extra"}, {"command": "'broken"},
    {"command": "a" * 8193}, {"cwd": None}, {"cwd": ""}, {"environmentId": "remote"},
    {"additionalPermissions": {"network": True}}, {"networkApprovalContext": {"host": "example.com"}},
    {"proposedNetworkPolicyAmendments": [{"host": "example.com"}]}, {"approvalId": "subcommand"},
    {"proposedExecpolicyAmendment": [""]}, {"proposedExecpolicyAmendment": [1]},
    {"proposedExecpolicyAmendment": "npm"}, {"proposedExecpolicyAmendment": {"prefix": ["npm"]}},
    {"proposedExecpolicyAmendment": []}, {"proposedExecpolicyAmendment": ["echo"]},
])
def test_unsupported_or_malformed_requests_fail_closed(updates):
    assert command_rule(METHOD, request(**updates)) is None


def test_matching_is_token_based_and_bound_to_context():
    original = command_rule(METHOD, request())
    assert original
    for candidate in [
        request("npm testing", prefix=["npm", "testing"]),
        request("npm install", prefix=["npm", "install"]),
        request(cwd="/another-repo"),
        request(command="/bin/bash -lc 'npm test'"),
        request(command="/bin/zsh -c 'npm test'"),
    ]:
        rule = command_rule(METHOD, candidate)
        assert rule and not rule.matches([original.grant])
    assert command_rule("item/fileChange/requestApproval", request()) is None


def test_legacy_camel_case_array_is_supported_but_not_a_fake_prefix_object():
    params = request()
    params["proposedExecPolicyAmendment"] = params.pop("proposedExecpolicyAmendment")
    assert command_rule(METHOD, params)
