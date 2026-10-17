"""The datasource capability policy: which connector methods cloud may run.

A method is a candidate only when its spec declares a `cloud` block, so the
policy can never offer a form the hosted path has no fields for, and it is
enableable only when that block says the adapters can execute it. It becomes
available only when a deployment manifest of the version this code
understands also names it. Everything about the configuration fails closed:
unparseable, a version from the future, or a pair naming something the
registry does not declare leaves the whole policy unavailable or ignores the
pair, and says so in a log that never carries the value.
"""

from __future__ import annotations

import logging

import pytest

from cowork.common.settings.app_settings import AppSettings
from cowork.services.connectors.datasource_capabilities import (
    MANIFEST_VERSION,
    load_datasource_capabilities,
)

ENABLED_ONE = '{"manifest_version": 1, "enabled": ["postgres:host-port"]}'


def _caps(raw: str):
    return load_datasource_capabilities(AppSettings(COWORK_DATASOURCE_CAPABILITIES=raw))


def test_the_default_offers_nothing_while_still_knowing_the_candidates():
    caps = _caps("")

    assert caps.is_available("postgres", "host-port") is False
    assert caps.is_available("mysql", "host-password") is False
    assert caps.available_connector_ids() == set()
    # The candidates are the spec's own cloud methods, so the capability
    # response can say "known but off" rather than staying silent about them.
    assert caps.methods["postgres"] == {"host-port": False}
    assert caps.methods["mysql"] == {"host-password": False}


def test_a_method_without_a_cloud_block_is_never_a_candidate():
    caps = _caps('{"manifest_version": 1, "enabled": ["postgres:connection-string"]}')

    assert "connection-string" not in caps.methods["postgres"]
    assert caps.is_available("postgres", "connection-string") is False


def test_enabling_one_pair_marks_only_that_pair(adapter_verified_datasources):
    caps = _caps(ENABLED_ONE)

    assert caps.is_available("postgres", "host-port") is True
    assert caps.is_available("mysql", "host-password") is False
    assert caps.available_connector_ids() == {"postgres"}


def test_configuration_cannot_enable_a_method_the_spec_calls_unverified(
    adapter_unverified_datasources, caplog
):
    with caplog.at_level(logging.WARNING):
        caps = _caps(ENABLED_ONE)

    assert caps.is_available("postgres", "host-port") is False
    assert caps.available_connector_ids() == set()
    assert "adapter-verified" in caplog.text
    assert "postgres:host-port" not in caplog.text


def test_a_manifest_from_another_version_turns_everything_off(adapter_verified_datasources, caplog):
    with caplog.at_level(logging.WARNING):
        caps = _caps('{"manifest_version": 2, "enabled": ["postgres:host-port"]}')

    assert caps.is_available("postgres", "host-port") is False
    assert caps.available_connector_ids() == set()
    assert "manifest_version" in caplog.text


def test_malformed_configuration_fails_closed_without_echoing_it(caplog):
    with caplog.at_level(logging.WARNING):
        caps = _caps('{"manifest_version": 1, "enabled": ["postgres:host-port"')

    assert caps.available_connector_ids() == set()
    assert "postgres:host-port" not in caplog.text


def test_an_unknown_pair_is_ignored_and_named_by_shape_only(adapter_verified_datasources, caplog):
    with caplog.at_level(logging.WARNING):
        caps = _caps('{"manifest_version": 1, "enabled": ["postgres:host-port", "nope:whatever"]}')

    assert caps.is_available("postgres", "host-port") is True
    assert caps.is_available("nope", "whatever") is False
    assert "1 enabled capability pair(s)" in caplog.text
    assert "nope:whatever" not in caplog.text


def test_an_unexpected_manifest_field_fails_closed():
    caps = _caps('{"manifest_version": 1, "enabled": [], "surprise": true}')

    assert caps.available_connector_ids() == set()


def test_the_version_it_reports_is_the_one_it_understands(adapter_verified_datasources):
    assert _caps(ENABLED_ONE).manifest_version == MANIFEST_VERSION


def test_cloud_fields_come_from_the_spec_and_only_for_a_candidate():
    caps = _caps(ENABLED_ONE)

    names = [f.name for f in caps.cloud_fields("postgres", "host-port")]
    assert names == ["host", "port", "database", "schema", "username", "password", "tls_verify"]
    assert caps.cloud_fields("postgres", "connection-string") == []


@pytest.mark.parametrize("raw", ["", "   ", "null", "[]"])
def test_anything_that_is_not_a_manifest_object_is_off(raw):
    caps = load_datasource_capabilities(AppSettings(COWORK_DATASOURCE_CAPABILITIES=raw))

    assert caps.available_connector_ids() == set()
