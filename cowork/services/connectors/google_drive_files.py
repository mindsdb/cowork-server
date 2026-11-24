"""Search a Google Drive connection's files for the in-app file chooser.

Only useful when the connection's token carries drive.readonly: a
drive.file-only token sees just the files this app created, so the
client keeps the Google Picker for those connections.
"""
from __future__ import annotations

import httpx

DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
PAGE_SIZE = 25
_FIELDS = (
    "nextPageToken,"
    "files(id,name,mimeType,iconLink,webViewLink,modifiedTime,resourceKey,owners(displayName))"
)
_FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveSearchError(Exception):
    """Drive answered with an error the user can't fix by reconnecting."""


class DriveAccessDenied(DriveSearchError):
    """Drive refused the token: expired, revoked, or without drive.readonly."""


def drive_query(text: str) -> str:
    """The files.list `q` for a name search, or for every file when empty.
    Folders and trashed files are left out: the chooser adds files."""
    clauses = ["trashed = false", f"mimeType != '{_FOLDER_MIME}'"]
    text = text.strip()
    if text:
        escaped = text.replace("\\", "\\\\").replace("'", "\\'")
        clauses.append(f"name contains '{escaped}'")
    return " and ".join(clauses)


def _to_file(raw: dict) -> dict:
    owners = raw.get("owners") or []
    return {
        "id": raw.get("id"),
        "name": raw.get("name") or "Untitled",
        "mimeType": raw.get("mimeType"),
        "iconUrl": raw.get("iconLink"),
        "url": raw.get("webViewLink"),
        "resourceKey": raw.get("resourceKey"),
        "modifiedTime": raw.get("modifiedTime"),
        "owner": (owners[0] or {}).get("displayName") if owners else None,
    }


async def search_files(
    access_token: str,
    query: str = "",
    page_token: str = "",
    *,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """One page of the user's Drive files, most recently viewed first.

    Covers My Drive, files shared with the user and Shared Drives."""
    params = {
        "q": drive_query(query),
        "orderBy": "viewedByMeTime desc,modifiedTime desc",
        "pageSize": PAGE_SIZE,
        "fields": _FIELDS,
        "corpora": "allDrives",
        "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true",
    }
    if page_token:
        params["pageToken"] = page_token
    headers = {"Authorization": f"Bearer {access_token}"}

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=15)
    try:
        response = await client.get(DRIVE_FILES_URL, params=params, headers=headers)
    except httpx.HTTPError as exc:
        raise DriveSearchError("Could not reach Google Drive.") from exc
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code in (401, 403):
        raise DriveAccessDenied(f"Google Drive refused the request ({response.status_code}).")
    if response.status_code >= 400:
        raise DriveSearchError(f"Google Drive search failed ({response.status_code}).")
    data = response.json()
    files = [_to_file(f) for f in data.get("files") or [] if f.get("id")]
    return {"files": files, "next_page_token": data.get("nextPageToken") or ""}
