"""Which connector methods this deployment may run as a cloud datasource.

Two gates, and both have to pass. A method is a *candidate* only when its
spec declares a `cloud` block, so the policy can never offer a form the
hosted path has no fields for; candidacy comes from the registry, not from
configuration. A candidate is *available* only when the deployment's manifest
names it and that manifest is the version this code understands.

Everything about the configuration fails closed. Unparseable JSON, a version
this code does not know, or an unexpected field leaves every method
unavailable; a pair naming something the registry does not declare is ignored.
The logs say what was wrong and how many pairs were dropped, never the value:
the manifest is operator configuration, but this module has no way to know a
deployment has not put something private in it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from cowork.common.settings.app_settings import AppSettings, get_app_settings
from cowork.schemas.connectors import ConnectorField
from cowork.services.connectors.specs._registry import registry as default_registry

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


class DatasourceManifest(BaseModel):
    """The deployment's own list, as `COWORK_DATASOURCE_CAPABILITIES` carries it."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: int
    enabled: list[str] = []


@dataclass(frozen=True)
class DatasourceCapabilities:
    """The one policy object. ENG-2807's producer and the relay both read it."""

    manifest_version: int
    # connector id -> method id -> available. Every candidate appears, so a
    # caller can tell "known but off" from "not a thing".
    methods: dict[str, dict[str, bool]] = field(default_factory=dict)
    _fields: dict[tuple[str, str], list[ConnectorField]] = field(default_factory=dict)

    def is_available(self, connector_id: str, method: str) -> bool:
        return bool(self.methods.get(connector_id, {}).get(method, False))

    def available_connector_ids(self) -> set[str]:
        return {
            connector_id
            for connector_id, methods in self.methods.items()
            if any(methods.values())
        }

    def cloud_fields(self, connector_id: str, method: str) -> list[ConnectorField]:
        return list(self._fields.get((connector_id, method), []))


def _candidates(registry) -> tuple[dict[str, dict[str, bool]], dict[tuple[str, str], list[ConnectorField]]]:
    """Every spec method that declares a cloud block, all switched off."""
    methods: dict[str, dict[str, bool]] = {}
    fields: dict[tuple[str, str], list[ConnectorField]] = {}
    for metadata in registry.list_connectors():
        spec = registry.get_connector(metadata.id)
        for method in (spec.form.methods or []) if spec and spec.form else []:
            if method.cloud is None:
                continue
            methods.setdefault(metadata.id, {})[method.id] = False
            fields[(metadata.id, method.id)] = list(method.cloud.fields)
    return methods, fields


def _parse(raw: str) -> DatasourceManifest | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        payload: Any = json.loads(text)
    except ValueError:
        logger.warning("[datasources] capability configuration is not valid JSON; every method stays unavailable")
        return None
    if not isinstance(payload, dict):
        logger.warning("[datasources] capability configuration is not an object; every method stays unavailable")
        return None
    try:
        return DatasourceManifest.model_validate(payload)
    except ValidationError:
        logger.warning("[datasources] capability configuration does not match the manifest; every method stays unavailable")
        return None


def load_datasource_capabilities(settings: AppSettings | None = None, registry=default_registry) -> DatasourceCapabilities:
    methods, fields = _candidates(registry)
    manifest = _parse((settings or get_app_settings()).datasource_capabilities)

    if manifest is None:
        return DatasourceCapabilities(MANIFEST_VERSION, methods, fields)
    if manifest.manifest_version != MANIFEST_VERSION:
        logger.warning(
            "[datasources] capability manifest_version %s is not %s; every method stays unavailable",
            manifest.manifest_version,
            MANIFEST_VERSION,
        )
        return DatasourceCapabilities(MANIFEST_VERSION, methods, fields)

    unknown = 0
    for pair in manifest.enabled:
        connector_id, _, method = str(pair).partition(":")
        if method and method in methods.get(connector_id, {}):
            methods[connector_id][method] = True
        else:
            unknown += 1
    if unknown:
        logger.warning(
            "[datasources] %s enabled capability pair(s) name no connector method that declares cloud support; ignored",
            unknown,
        )
    return DatasourceCapabilities(MANIFEST_VERSION, methods, fields)
