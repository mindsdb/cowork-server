"""People in the caller's MindsHub organization, read as them.

Three reads of MindsHub's ``GET /v1/me/org-members/``, each with the caller's
own bearer against the operator's host (cowork.services.mindshub_client):

- no params: the whole active-org directory (MindsHub caps it at
  MAX_DIRECTORY_ROWS and says ``truncated``), the caller excluded. What the share
  dialog lists and filters locally. Never cached here: the app is its only
  reader and keeps its own copy, so a second one here would only make the
  app's copy older.
- ``?ids=a,b``: which of those user ids are members of the active org, with
  their names and addresses. What sharing a skill with people checks a newly
  added id against, and what the app names stored ids with
  (``GET /hub/people/?ids=``). Never cached: a refusal has to reflect who is
  in the org now.
- ``?emails=a,b``: the members of the active org who hold those addresses.
  What the app asks about an address its capped directory doesn't have, so a
  member past the cap is shared with by id rather than as an outside address
  (``GET /hub/people/?emails=``). Never cached, for the same reason.

MindsHub couldn't be asked (no answer, a 5xx, a body that isn't the list): the
directory is ``reachable`` false and a lookup is None, which the id check
turns into a 503 at the caller. MindsHub answered with a 4xx, such as its 403
for a caller who is not a current member of the active org or its 429: that
is MindsHubRefused (cowork.services.mindshub_client), never "unreachable".
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Literal
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from cowork.services.mindshub_client import MindsHubAnswer, MindsHubRefused, get_mindshub_reply
from cowork.services.sharing_contract import (
    MAX_EMAIL_CHARS,
    MAX_EMAILS_PER_LOOKUP,
    MAX_IDS_PER_LOOKUP,
    ORG_MEMBERS_PATH,
)

logger = logging.getLogger(__name__)


class OrgMember(BaseModel):
    """One member, as MindsHub puts it on the wire."""

    id: str
    name: str = ""
    email: str = ""


class OrgDirectory(BaseModel):
    """The whole active-org directory. ``reachable`` false means MindsHub
    could not be asked; ``truncated`` is MindsHub's: it held more members
    than it returns."""

    people: list[OrgMember] = Field(default_factory=list)
    reachable: bool = False
    truncated: bool = False


# The rule MindsHub's org-member lookup applies to each address it is asked
# about: Django 6.0's validate_email (django.core.validators.EmailValidator).
# Its regular expressions, copied: a dot-atom or quoted local part, then a
# host name, its labels and a top-level label of two or more letters (or a
# punycode label), or a bracketed IP address, or "localhost". An address
# MindsHub would refuse makes it refuse the whole lookup, so the lookup
# checks each one first (is_email_address).
# Letters past ASCII, as Django's range spells them.
_LETTERS = "\u00a1-\uffff"
_EMAIL_LOCAL_PART = re.compile(
    r"(^[-!#$%&'*+/=?^_`{}|~0-9A-Z]+(\.[-!#$%&'*+/=?^_`{}|~0-9A-Z]+)*\Z"
    r'|^"([\001-\010\013\014\016-\037!#-\[\]-\177]|\\[\001-\011\013\014\016-\177])*"\Z)',
    re.IGNORECASE,
)
_EMAIL_DOMAIN = re.compile(
    r"^[a-z" + _LETTERS + r"0-9](?:[a-z" + _LETTERS + r"0-9-]{0,61}[a-z" + _LETTERS + r"0-9])?"
    r"(?:\.(?!-)[a-z" + _LETTERS + r"0-9-]{1,63}(?<!-))*"
    r"\.(?!-)(?:[a-z" + _LETTERS + r"-]{2,63}|xn--[a-z0-9]{1,59})(?<!-)\Z",
    re.IGNORECASE,
)
_EMAIL_IP_LITERAL = re.compile(r"\[([A-F0-9:.]+)\]\Z", re.IGNORECASE)
_EMAIL_DOMAIN_ALLOWLIST = frozenset({"localhost"})
# An IPv6 literal past 39 characters is refused, as Django refuses it; the
# whole address is bound by MAX_EMAIL_CHARS (RFC 3696 section 3, as Django
# bounds it).
_MAX_IPV6_CHARS = 39


def _is_ip_address(value: str) -> bool:
    try:
        IPv4Address(value)
        return True
    except ValueError:
        pass
    if len(value) > _MAX_IPV6_CHARS:
        return False
    try:
        IPv6Address(value)
        return True
    except ValueError:
        return False


def is_email_address(value: str) -> bool:
    """Whether MindsHub's org-member lookup takes ``value`` as an address:
    Django's validate_email, which it applies to each lowercased address."""
    if not value or "@" not in value or len(value) > MAX_EMAIL_CHARS:
        return False
    local_part, domain = value.rsplit("@", 1)
    if not _EMAIL_LOCAL_PART.match(local_part):
        return False
    if domain in _EMAIL_DOMAIN_ALLOWLIST or _EMAIL_DOMAIN.match(domain):
        return True
    literal = _EMAIL_IP_LITERAL.match(domain)
    return bool(literal) and _is_ip_address(literal[1])


def _rows(payload: Any) -> list[OrgMember] | None:
    """``{"results": [...]}`` or a bare list; None when it is neither. A row
    with no id is dropped."""
    if payload is None:
        return None
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None
    members: list[OrgMember] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        try:
            members.append(OrgMember.model_validate({**row, "id": str(row["id"])}))
        except Exception as exc:
            logger.debug("dropping an unreadable org member row: %s", exc)
    return members


async def _answered(path: str, bearer_token: str) -> Any | None:
    """One read of ``path``: its JSON body, or None when MindsHub couldn't be
    asked. Raises MindsHubRefused for a 4xx, a 429 included."""
    reply = await get_mindshub_reply(path, bearer_token)
    if reply.answer in (MindsHubAnswer.REFUSED, MindsHubAnswer.THROTTLED):
        raise MindsHubRefused(mindshub_status=reply.status_code)
    return reply.payload if reply.answer is MindsHubAnswer.OK else None


async def _members_by(
    *, bearer_token: str, field: Literal["id", "email"], values: Iterable[str], per_lookup: int
) -> list[OrgMember] | None:
    """The members of the caller's active org whose ``field`` is one of
    ``values`` (compared lowercased), in the order asked, or None when
    MindsHub couldn't be asked. One call per ``per_lookup`` values; any chunk
    failing makes the whole answer None, and any chunk MindsHub refuses
    raises MindsHubRefused. A value it doesn't return is no member's."""
    wanted = list(dict.fromkeys(v.strip().lower() for v in values if v and v.strip()))
    if not wanted:
        return []
    if not bearer_token:
        return None
    asked = set(wanted)
    found: dict[str, OrgMember] = {}
    for start in range(0, len(wanted), per_lookup):
        chunk = wanted[start : start + per_lookup]
        # MindsHub names each lookup by the field's plural: ids, emails.
        path = f"{ORG_MEMBERS_PATH}?{urlencode({f'{field}s': ','.join(chunk)})}"
        members = _rows(await _answered(path, bearer_token))
        if members is None:
            return None
        for member in members:
            key = getattr(member, field).strip().lower()
            if key in asked:
                found.setdefault(key, member)
    return [found[v] for v in wanted if v in found]


async def org_members_by_ids(
    *, bearer_token: str, ids: Iterable[str]
) -> list[OrgMember] | None:
    """The members of the caller's active org among ``ids`` (user ids), or
    None when MindsHub couldn't be asked; MindsHubRefused when it
    refused. An id it doesn't return is not a member."""
    return await _members_by(
        bearer_token=bearer_token, field="id", values=ids, per_lookup=MAX_IDS_PER_LOOKUP
    )


async def org_members_by_emails(
    *, bearer_token: str, emails: Iterable[str]
) -> list[OrgMember] | None:
    """The members of the caller's active org who hold ``emails``, or None
    when MindsHub couldn't be asked; MindsHubRefused when it refused. An
    address it doesn't return is no member's, whether or not somebody outside
    the org holds it. The caller checks each address first
    (is_email_address): MindsHub refuses a whole lookup for one it doesn't
    take."""
    return await _members_by(
        bearer_token=bearer_token, field="email", values=emails, per_lookup=MAX_EMAILS_PER_LOOKUP
    )


async def list_org_directory(*, bearer_token: str) -> OrgDirectory:
    """Everyone in the caller's active org but the caller (MindsHub's
    parameterless read), read now. MindsHubRefused when MindsHub refused."""
    if not bearer_token:
        return OrgDirectory()
    payload = await _answered(ORG_MEMBERS_PATH, bearer_token)
    members = _rows(payload)
    if members is None:
        return OrgDirectory()
    return OrgDirectory(
        people=members,
        reachable=True,
        truncated=isinstance(payload, dict) and payload.get("truncated") is True,
    )
