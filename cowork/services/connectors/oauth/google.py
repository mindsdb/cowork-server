from __future__ import annotations

import html as html_lib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import HTTPException

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.schemas.connectors import OAuthStartResponse
from cowork.services.connectors.encrypted_vault import build_vault
from cowork.services.connectors.oauth import pkce as pkce_utils
from cowork.services.connectors.oauth.config import GOOGLE_SERVICES
from cowork.services.connectors.oauth.state import OAuthStateStore

import logging as _logging

_GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"
_GOOGLE_REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

_log = _logging.getLogger("cowork.oauth.google")

_SERVICE_CREDENTIAL_ATTRS: dict[str, tuple[str, str]] = {
    "google-drive":     ("google_drive_client_id",     "google_drive_client_secret"),
    "google-calendar":  ("google_calendar_client_id",  "google_calendar_client_secret"),
    "gmail":            ("gmail_client_id",             "gmail_client_secret"),
    "google-ads":       ("google_ads_client_id",        "google_ads_client_secret"),
    "google-analytics": ("google_analytics_client_id",  "google_analytics_client_secret"),
}

# engine name (e.g. "google_drive") → service id (e.g. "google-drive")
_ENGINE_TO_SERVICE: dict[str, str] = {cfg.engine: svc for svc, cfg in GOOGLE_SERVICES.items()}

_VERIFY_URLS: dict[str, str] = {
    "google-drive":     "https://www.googleapis.com/drive/v3/about?fields=user",
    "gmail":            "https://gmail.googleapis.com/gmail/v1/users/me/profile",
    "google-calendar":  "https://www.googleapis.com/calendar/v3/colors",
    "google-ads":       "https://www.googleapis.com/oauth2/v3/userinfo",
    "google-analytics": "https://www.googleapis.com/oauth2/v3/userinfo",
}


class GoogleOAuthService:
    def _resolve_credentials(self, service: str, settings: OAuthSettings) -> tuple[str, str]:
        id_attr, secret_attr = _SERVICE_CREDENTIAL_ATTRS[service]
        client_id = getattr(settings, id_attr)
        client_secret = getattr(settings, secret_attr)
        if not client_id or not client_secret:
            raise HTTPException(status_code=400, detail=f"OAuth credentials not configured for {service}.")
        return client_id, client_secret

    def get_outcome(self, state: str, settings: OAuthSettings) -> dict | None:
        return self._store(settings).get_outcome(state)

    def clear_outcome(self, state: str, settings: OAuthSettings) -> None:
        self._store(settings).clear_outcome(state)

    def _store(self, settings: OAuthSettings) -> OAuthStateStore:
        return OAuthStateStore(settings.state_path)

    def _redirect_uri(self, service: str, settings: OAuthSettings) -> str:
        return f"{settings.server_origin.rstrip('/')}/api/v1/connectors/oauth/{service}/callback"

    def start(self, service: str, settings: OAuthSettings, *, client_id: str = "", client_secret: str = "", extra_fields: dict[str, str] | None = None) -> OAuthStartResponse:
        if client_id and client_secret:
            cid, csecret = client_id, client_secret
        else:
            cid, csecret = self._resolve_credentials(service, settings)
        client_id = cid

        cfg = GOOGLE_SERVICES[service]
        verifier = pkce_utils.generate_verifier()
        challenge = pkce_utils.generate_challenge(verifier)
        state = pkce_utils.generate_state()
        redirect_uri = self._redirect_uri(service, settings)
        started_at = datetime.now(timezone.utc).isoformat()

        self._store(settings).set_pending(
            service,
            state=state,
            verifier=verifier,
            redirect_uri=redirect_uri,
            started_at=started_at,
            client_id=cid,
            client_secret=csecret,
            extra_fields=extra_fields,
        )
        self._store(settings).set_outcome(state, {"status": "pending"})

        auth_url = _GOOGLE_AUTH_ENDPOINT + "?" + urlencode({
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
            "scope": " ".join(cfg.scopes),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })

        return OAuthStartResponse(auth_url=auth_url, redirect_uri=redirect_uri, started_at=started_at, state=state)

    def callback(self, service: str, code: str, state: str, error: str, settings: OAuthSettings) -> str:
        cfg = GOOGLE_SERVICES[service]
        store = self._store(settings)
        service_label = service.replace("-", " ").title()

        if error:
            err_msg = f"Google sign-in returned: {error}"
            store.clear_pending(service, error=err_msg)
            if state:
                store.set_outcome(state, {"status": "error", "error": err_msg})
            return _callback_page(
                f"{service_label} connection was cancelled",
                "You can return to CoWork and try the connection again whenever you are ready.",
                success=False,
            )

        pending = store.get_pending(service)
        if not pending:
            return _callback_page(
                f"{service_label} sign-in expired",
                f"CoWork could not find a pending {service_label} sign-in request. Start the connection again.",
                success=False,
            )

        if not state or state != str(pending.get("state", "")).strip():
            store.clear_pending(service, error="Google sign-in state did not match the pending request.")
            return _callback_page(
                f"{service_label} connection could not be verified",
                "CoWork rejected the callback because the Google sign-in state did not match.",
                success=False,
            )

        if not code:
            store.clear_pending(service, error="Google sign-in did not return an authorization code.")
            return _callback_page(
                f"{service_label} connection could not be completed",
                "Google did not return an authorization code.",
                success=False,
            )

        pending_client_id = str(pending.get("clientId", "")).strip()
        pending_client_secret = str(pending.get("clientSecret", "")).strip()
        if pending_client_id and pending_client_secret:
            client_id, client_secret = pending_client_id, pending_client_secret
        else:
            try:
                client_id, client_secret = self._resolve_credentials(service, settings)
            except HTTPException:
                err_msg = f"OAuth credentials not configured for {service}."
                store.clear_pending(service, error=err_msg)
                store.set_outcome(state, {"status": "error", "error": err_msg})
                return _callback_page(
                    f"{service_label} connection is not configured",
                    "Google OAuth credentials are not configured on this server.",
                    success=False,
                )

        started_at = str(pending.get("startedAt", "")).strip()
        if started_at:
            try:
                started_dt = datetime.fromisoformat(started_at)
                if datetime.now(timezone.utc) - started_dt > timedelta(minutes=20):
                    store.clear_pending(service, error="Google sign-in timed out before it completed.")
                    return _callback_page(
                        f"{service_label} sign-in expired",
                        "That Google sign-in request took too long. Start the connection again.",
                        success=False,
                    )
            except ValueError:
                pass

        try:
            token_data = self._exchange_code(
                code=code,
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=str(pending.get("redirectUri") or self._redirect_uri(service, settings)),
                verifier=str(pending.get("verifier", "")),
            )
            access_token = str(token_data.get("access_token", "")).strip()
            if not access_token:
                raise HTTPException(status_code=502, detail="Token exchange did not return an access token.")

            userinfo = self._fetch_userinfo(access_token)
            account_email = str(userinfo.get("email", "")).strip()
            account_name = str(userinfo.get("name", "")).strip()
            connection_name = account_email or cfg.engine

            self.verify_connection(service, access_token)

            expires_in = int(token_data.get("expires_in", 0) or 0)
            expires_at = (
                (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
                if expires_in else ""
            )

            extra = {k: v for k, v in (pending.get("extraFields") or {}).items() if v}
            build_vault(Path(ConnectorSettings().vault_dir)).save(
                cfg.engine,
                connection_name,
                {
                    "auth_type": "oauth",
                    "access_token": access_token,
                    "refresh_token": str(token_data.get("refresh_token", "")).strip(),
                    "token_type": str(token_data.get("token_type", "Bearer")).strip(),
                    "scope": str(token_data.get("scope", "")).strip(),
                    "expires_at": expires_at,
                    "account_email": account_email,
                    "account_name": account_name,
                    **extra,
                },
            )
        except HTTPException as exc:
            err_msg = str(exc.detail)
            store.clear_pending(service, error=err_msg)
            store.set_outcome(state, {"status": "error", "error": err_msg})
            return _callback_page(
                f"{service_label} connection failed",
                "An error occurred during the sign-in flow. Return to CoWork and try again.",
                success=False,
            )
        except Exception as exc:
            err_msg = str(exc)
            store.clear_pending(service, error=err_msg)
            store.set_outcome(state, {"status": "error", "error": err_msg})
            return _callback_page(
                f"{service_label} connection failed",
                f"CoWork could not finish the Google sign-in flow: {err_msg}",
                success=False,
            )

        store.clear_pending(service)
        store.set_outcome(state, {"status": "success", "name": connection_name})
        return _callback_page(
            f"{service_label} connected",
            f"{account_name or account_email or 'Your Google account'} is now connected. You can close this tab and return to CoWork.",
            success=True,
        )

    def revoke(self, engine: str, name: str, connector_settings: ConnectorSettings) -> None:
        if engine not in _ENGINE_TO_SERVICE:
            return
        _log.info("Revoking Google token for %s/%s", engine, name)
        try:
            fields = build_vault(Path(connector_settings.vault_dir)).load(engine, name) or {}
        except Exception:
            return
        if fields.get("auth_type") != "oauth":
            return
        token = fields.get("refresh_token", "").strip() or fields.get("access_token", "").strip()
        if not token:
            return
        try:
            request = Request(
                f"{_GOOGLE_REVOKE_ENDPOINT}?{urlencode({'token': token})}",
                method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urlopen(request, timeout=10):
                pass
            _log.info("Revoked Google token for %s/%s", engine, name)
        except HTTPError as exc:
            if exc.code == 400:
                try:
                    body = json.loads(exc.read().decode())
                except Exception:
                    body = {}
                if body.get("error") == "invalid_token":
                    _log.debug("Token for %s/%s already expired or revoked — skipping revocation", engine, name)
                else:
                    _log.warning("Could not revoke Google token for %s/%s: %s %s", engine, name, exc.code, body)
            else:
                _log.warning("Could not revoke Google token for %s/%s: %s", engine, name, exc)
        except Exception as exc:
            _log.warning("Could not revoke Google token for %s/%s: %s", engine, name, exc)

    def refresh_all_tokens(self, connector_settings: ConnectorSettings, oauth_settings: OAuthSettings) -> None:
        try:
            vault = build_vault(Path(connector_settings.vault_dir))
            all_connections = vault.list_connections() or []
        except Exception:
            return

        now = datetime.now(timezone.utc)
        threshold = now + timedelta(minutes=10)

        for item in all_connections:
            engine = item.get("engine", "")
            name = item.get("name", "")
            if engine not in _ENGINE_TO_SERVICE or not name:
                continue
            try:
                fields = vault.load(engine, name) or {}
            except Exception:
                continue
            if fields.get("auth_type") != "oauth":
                continue
            refresh_token = fields.get("refresh_token", "").strip()
            if not refresh_token:
                continue

            expires_at_str = fields.get("expires_at", "").strip()
            if expires_at_str:
                try:
                    expires_dt = datetime.fromisoformat(expires_at_str)
                    if expires_dt.tzinfo is None:
                        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                    if expires_dt > threshold:
                        continue
                except ValueError:
                    pass

            service_id = _ENGINE_TO_SERVICE[engine]
            try:
                cid, csecret = self._resolve_credentials(service_id, oauth_settings)
            except HTTPException:
                continue

            try:
                token_data = _json_request(
                    _GOOGLE_TOKEN_ENDPOINT,
                    method="POST",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": cid,
                        "client_secret": csecret,
                    },
                )
                new_access_token = str(token_data.get("access_token", "")).strip()
                if not new_access_token:
                    continue
                expires_in = int(token_data.get("expires_in", 0) or 0)
                new_expires_at = (
                    (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
                    if expires_in else ""
                )
                updated = {**fields, "access_token": new_access_token, "expires_at": new_expires_at}
                new_refresh = str(token_data.get("refresh_token", "")).strip()
                if new_refresh:
                    updated["refresh_token"] = new_refresh
                vault.save(engine, name, updated)
                _log.info("Refreshed token for %s/%s", engine, name)
            except Exception as exc:
                _log.warning("Could not refresh token for %s/%s: %s", engine, name, exc)

    def get_catalogue(self, connector_settings: ConnectorSettings, oauth_settings: OAuthSettings) -> list[dict]:
        try:
            vault = build_vault(Path(connector_settings.vault_dir))
            all_connections = vault.list_connections() or []
        except Exception as exc:
            _log.warning("Could not load vault for catalogue: %s", exc)
            all_connections = []

        state_data = self._store(oauth_settings)._load()
        items = []

        for service_id, cfg in GOOGLE_SERVICES.items():
            engine = cfg.engine
            id_attr, secret_attr = _SERVICE_CREDENTIAL_ATTRS[service_id]
            cid = getattr(oauth_settings, id_attr, "")
            csecret = getattr(oauth_settings, secret_attr, "")
            ready = bool(cid and csecret)
            config_error = "" if ready else f"OAuth credentials not configured for {service_id}."

            connections = [
                {"engine": engine, "name": c.get("name", ""), "label": c.get("name", "")}
                for c in all_connections if c.get("engine") == engine
            ]

            entry = state_data.get(service_id) or {}
            service_label = " ".join(w.capitalize() for w in service_id.replace("-", " ").split())

            items.append({
                "id": engine,
                "title": service_label,
                "engine": engine,
                "status": "connected" if connections else ("available" if ready else "needs_config"),
                "connections": connections,
                "connectionCount": len(connections),
                "oauth": {
                    "ready": ready,
                    "configError": config_error,
                    "pending": bool((entry.get("pending") or {}).get("state")),
                    "lastSuccessAt": entry.get("lastSuccessAt", ""),
                    "lastError": entry.get("lastError", ""),
                    "lastErrorAt": entry.get("lastErrorAt", ""),
                    "launchLabel": f"Connect {service_label}",
                    "redirectUri": self._redirect_uri(service_id, oauth_settings),
                },
            })

        return items

    def _exchange_code(
        self,
        *,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        verifier: str,
    ) -> dict[str, Any]:
        return _json_request(
            _GOOGLE_TOKEN_ENDPOINT,
            method="POST",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": verifier,
            },
        )

    def _fetch_userinfo(self, access_token: str) -> dict[str, Any]:
        return _json_request(
            _GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    def verify_connection(self, connector_id_or_service: str, access_token: str) -> None:
        """Make a lightweight API call to confirm the token works before vault save.
        Accepts either an engine name (e.g. 'google_drive') or a service id
        (e.g. 'google-drive') — maps engine names via _ENGINE_TO_SERVICE.
        Raises HTTPException(502) on failure. No-ops for services not in _VERIFY_URLS."""
        service = _ENGINE_TO_SERVICE.get(connector_id_or_service, connector_id_or_service)
        url = _VERIFY_URLS.get(service)
        if url is None:
            return
        _json_request(url, headers={"Authorization": f"Bearer {access_token}"})


def _json_request(
    url: str,
    *,
    method: str = "GET",
    data: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", **(headers or {})}
    body = None
    if data is not None:
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        body = urlencode(data).encode("utf-8")
    request = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        detail = raw
        try:
            payload = json.loads(raw)
            detail = (
                payload.get("error_description")
                or payload.get("error", {}).get("message")
                or payload.get("error")
                or raw
            )
        except json.JSONDecodeError:
            pass
        raise HTTPException(status_code=502, detail=f"Google OAuth request failed: {detail}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail="Could not reach Google OAuth services") from exc


def _callback_page(title: str, message: str, *, success: bool) -> str:
    accent = "#0f766e" if success else "#b42318"
    safe_title = html_lib.escape(title)
    safe_message = html_lib.escape(message)
    safe_state = "Connected" if success else "Connection failed"
    return dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>{safe_title}</title>
          <style>
            body {{
              margin: 0;
              min-height: 100vh;
              display: grid;
              place-items: center;
              background: #f6f5f1;
              color: #161616;
              font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            main {{
              width: min(92vw, 520px);
              background: #fff;
              border: 1px solid #e7e3da;
              border-radius: 18px;
              padding: 28px 28px 24px;
              box-shadow: 0 20px 60px rgba(20, 17, 12, 0.08);
            }}
            h1 {{
              margin: 0 0 10px;
              font-size: 24px;
              line-height: 1.15;
            }}
            p {{
              margin: 0;
              font-size: 15px;
              line-height: 1.55;
              color: #55514a;
            }}
            .pill {{
              display: inline-flex;
              align-items: center;
              gap: 8px;
              margin-bottom: 14px;
              padding: 6px 10px;
              border-radius: 999px;
              background: #f7f2ee;
              color: {accent};
              font-size: 12px;
              font-weight: 600;
              letter-spacing: 0.02em;
              text-transform: uppercase;
            }}
          </style>
        </head>
        <body>
          <main>
            <div class="pill">{safe_state}</div>
            <h1>{safe_title}</h1>
            <p>{safe_message}</p>
          </main>
        </body>
        </html>
        """
    ).strip()


google_service = GoogleOAuthService()
