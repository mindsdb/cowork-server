from __future__ import annotations

from sqlmodel import Session, select

from cowork.channels.plugin import ChannelPlugin
from cowork.channels.registry import PluginRegistry, get_registry
from cowork.common.encryption import decrypt, encrypt
from cowork.models.channel import ChannelInstallation
from cowork.models.setting import Setting
from cowork.schemas.channels import (
    ChannelConfigResponse,
    ChannelInstallationResponse,
    ChannelStatusItem,
    ChannelStatusResponse,
    CredentialFieldSpec,
    CredentialValue,
    PluginResponse,
)

_KEY_PREFIX = "channel."


def _cred_key(channel_type: str, field: str) -> str:
    return f"{_KEY_PREFIX}{channel_type}.{field}"


class UnknownChannelError(Exception):
    """Raised when a channel_type has no registered plugin (→ 404 at the edge)."""


class ChannelConfigService:
    def __init__(self, session: Session, registry: PluginRegistry | None = None) -> None:
        self.session = session
        self.registry = registry if registry is not None else get_registry()


    def list_plugins(self) -> list[PluginResponse]:
        return [self._plugin_dto(p) for p in self.registry.all()]

    def list_installations(self) -> list[ChannelInstallationResponse]:
        rows = self.session.exec(select(ChannelInstallation)).all()
        return [ChannelInstallationResponse.model_validate(r, from_attributes=True) for r in rows]

    def status(self) -> ChannelStatusResponse:
        installs = {r.channel_type: r for r in self.session.exec(select(ChannelInstallation)).all()}
        items: list[ChannelStatusItem] = []
        for plugin in self.registry.all():
            row = installs.get(plugin.channel_type)
            items.append(
                ChannelStatusItem(
                    channel_type=plugin.channel_type,
                    display_name=plugin.display_name,
                    enabled=row.enabled if row else False,
                    status=row.status if row else "disconnected",
                    configured=self._is_configured(plugin),
                )
            )
        return ChannelStatusResponse(
            plugin_count=len(self.registry.all()),
            installation_count=len(installs),
            channels=items,
        )

    def get_config(self, channel_type: str) -> ChannelConfigResponse:
        plugin = self._require_plugin(channel_type)
        return self._config_dto(plugin)

    def load_credentials(self, channel_type: str) -> dict[str, str]:
        """Decrypted credential values for internal runtime use only — building
        the live adapter. NEVER exposed via the API (get_config masks secrets)."""
        plugin = self._require_plugin(channel_type)
        creds: dict[str, str] = {}
        for field in plugin.credentials.fields:
            row = self._fetch_setting(_cred_key(channel_type, field.name))
            if row is None:
                continue
            creds[field.name] = decrypt(row.value) if field.secret else row.value
        return creds

    def set_config(self, channel_type: str, values: dict[str, str]) -> ChannelConfigResponse:
        plugin = self._require_plugin(channel_type)
        known = set(plugin.credentials.field_names())
        unknown = [name for name in values if name not in known]
        if unknown:
            raise ValueError(f"unknown credential field(s) for {channel_type!r}: {', '.join(sorted(unknown))}")

        for name, raw in values.items():
            field = plugin.credentials.get(name)
            assert field is not None  # guarded by the unknown check above
            store_val = encrypt(raw) if field.secret else raw
            self._upsert_setting(_cred_key(channel_type, name), store_val)

        self._ensure_installation(plugin)
        self.session.commit()
        return self._config_dto(plugin)

    def delete_config(self, channel_type: str) -> bool:
        plugin = self._require_plugin(channel_type)
        removed = False
        for name in plugin.credentials.field_names():
            row = self._fetch_setting(_cred_key(channel_type, name))
            if row is not None:
                self.session.delete(row)
                removed = True
        install = self._fetch_installation(channel_type)
        if install is not None:
            self.session.delete(install)
            removed = True
        if removed:
            self.session.commit()
        return removed

    def _require_plugin(self, channel_type: str) -> ChannelPlugin:
        plugin = self.registry.get(channel_type)
        if plugin is None:
            raise UnknownChannelError(channel_type)
        return plugin

    def _plugin_dto(self, plugin: ChannelPlugin) -> PluginResponse:
        return PluginResponse(
            channel_type=plugin.channel_type,
            display_name=plugin.display_name,
            credentials=[
                CredentialFieldSpec(
                    name=f.name, label=f.label, secret=f.secret,
                    required=f.required, description=f.description,
                )
                for f in plugin.credentials.fields
            ],
            has_oauth=plugin.oauth is not None,
            webhook_paths=[w.path for w in plugin.webhooks],
        )

    def _config_dto(self, plugin: ChannelPlugin) -> ChannelConfigResponse:
        fields: dict[str, CredentialValue] = {}
        for field in plugin.credentials.fields:
            row = self._fetch_setting(_cred_key(plugin.channel_type, field.name))
            is_set = row is not None
            # Secret values are never returned; non-secret values may be echoed.
            value = None
            if is_set and not field.secret:
                value = row.value
            fields[field.name] = CredentialValue(is_set=is_set, value=value)
        return ChannelConfigResponse(
            channel_type=plugin.channel_type,
            configured=self._is_configured(plugin),
            fields=fields,
        )

    def _is_configured(self, plugin: ChannelPlugin) -> bool:
        """A channel is configured once all required credential fields are set."""
        required = plugin.credentials.required_field_names()
        if not required:
            return False
        return all(
            self._fetch_setting(_cred_key(plugin.channel_type, name)) is not None
            for name in required
        )

    def _fetch_setting(self, key: str) -> Setting | None:
        return self.session.exec(select(Setting).where(Setting.key == key)).first()

    def _upsert_setting(self, key: str, value: str) -> None:
        row = self._fetch_setting(key)
        if row is None:
            row = Setting(key=key, value=value)
        else:
            row.value = value
        self.session.add(row)

    def _fetch_installation(self, channel_type: str) -> ChannelInstallation | None:
        return self.session.exec(
            select(ChannelInstallation).where(ChannelInstallation.channel_type == channel_type)
        ).first()

    def _ensure_installation(self, plugin: ChannelPlugin) -> None:
        """Create the installation row on first config write. Enable/status are
        runtime concerns (a later slice) — left at their defaults here."""
        if self._fetch_installation(plugin.channel_type) is None:
            self.session.add(
                ChannelInstallation(
                    channel_type=plugin.channel_type,
                    display_name=plugin.display_name,
                )
            )
