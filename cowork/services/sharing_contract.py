"""What sharing a skill or an artifact agrees on with MindsHub and the app.

The names of who a rule is shared with, the MindsHub reads and capabilities
sharing uses, the code of a refused share and what the agent is told after
one, how an artifact's rule was keyed before it had a stable id, and every
bound this server owns or mirrors. A bound MindsHub enforces is mirrored
here so a share it would refuse is refused before anything is published;
each comment says which side owns it. The app mirrors five of them,
MAX_IDS_PER_LOOKUP, MAX_EMAILS_PER_LOOKUP, INVALID_ID_CODE,
INVALID_EMAIL_CODE, and SHARE_ORG_REQUIRED_CODE, in its sharing contract
module (src/renderer/cowork/lib/sharingContract.js).
MindsHub, in its sharing contract module, and the upload service hold
SHARE_ORG_REQUIRED_CODE under the same name, and anton's publisher holds
SHARE_ORG_REQUIRED_CODE and SHARE_ORG_REQUIRED_INSTRUCTION under the same
names. The upload service holds LEGACY_RULE_USER_DIR_CHARS and
REPORT_ID_PATTERN under the same names.
"""

from __future__ import annotations

import re
from enum import StrEnum


class AudienceField(StrEnum):
    """Who a rule is shared with, under the names MindsHub's artifact rule
    gives them. The app sends them, this server stores a skill's audience
    under them (cowork.models.skill_access), and MindsHub stores an
    artifact's."""

    # Everyone in the organization ("Everyone at <org>").
    ORG_ALLOWED = "org_allowed"
    # Teams of the organization, by UUID.
    ALLOWED_TEAM_IDS = "allowed_team_ids"
    # People in the organization, by UUID.
    ALLOWED_USER_IDS = "allowed_user_ids"
    # People outside the organization, by address (artifacts only).
    ALLOWED_EMAILS = "allowed_emails"


# MindsHub capabilities, as its ``permissions[]`` names them.
# Share a skill with everyone in the organization.
SKILLS_SHARE_ORG = "skills.share_org"
# Change a skill someone else shared with everyone.
SKILLS_WRITE_ORG = "skills.write_org"
# Share an artifact with everyone in the organization.
ARTIFACTS_SHARE_ORG = "artifacts.share_org"

# The ``code`` of a refused share with everyone in the organization by
# someone without ARTIFACTS_SHARE_ORG. MindsHub answers it when a publish's
# rule newly sets org_allowed, and the upload service passes it on; this
# server answers it, with HTTP 403 and the reason as ``detail``, for that
# refusal and for its own check of the same before anything publishes.
# MindsHub's sharing contract module, the upload service, the app's
# sharing contract module, and anton's publisher (anton.publisher) hold it
# under the same name.
SHARE_ORG_REQUIRED_CODE = "share_org_required"
# The ``code`` of the people lookup's 400 when an id in ``?ids=`` isn't a
# UUID, and when an address in ``?emails=`` is one MindsHub wouldn't take;
# the body's ``ids`` or ``emails`` lists them. Owned by this server. The
# app's sharing contract module holds both under the same names.
INVALID_ID_CODE = "invalid_id"
INVALID_EMAIL_CODE = "invalid_email"
# What the agent's publish tool tells the model after the reason a share
# with everyone was refused. The model picks the access mode itself, so
# without it a model can get past the refusal by publishing wider. The
# tool takes anton's publish schema, which names people by email and has no
# teams, so the sentence offers only that. anton's publisher tells its own
# agent the same sentence under the same name.
SHARE_ORG_REQUIRED_INSTRUCTION = (
    "Do not publish it publicly or to a wider audience instead. Tell the user that an "
    "organization admin can share it with everyone, or offer to share it with specific "
    "people by email."
)

# A report published restricted before it had a stable ``artifact/<uuid>``
# id has its rule under ``<user_dir>/<report_id>`` until its next publish
# moves the rule. ``user_dir`` is the first this many hex digits of md5 of the
# owner's id in its canonical lowercase form. Owned by the upload service,
# which holds it under the same name for the folder its ``_user_prefix``
# names each owner.
LEGACY_RULE_USER_DIR_CHARS = 9
# The form of ``report_id`` in that old id. Owned by the upload service,
# which holds it under the same name for its report-id check and refuses a
# publish naming a report of another form, so no publish reaches a rule
# under one.
REPORT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")

# MindsHub reads sharing makes with the caller's bearer.
# The teams the caller belongs to.
TEAMS_PATH = "/me/teams/"
# Every team of the caller's organization.
ORG_TEAMS_PATH = "/organizations/current/teams/"
# The people in the caller's organization: the directory, or a lookup by
# ``ids`` or ``emails``.
ORG_MEMBERS_PATH = "/me/org-members/"

# The most people one skill or artifact is shared with by id. Mirrors
# MindsHub's bound on an artifact's rule; this server applies it to skills
# too.
MAX_PEOPLE_PER_RULE = 100
# The most teams one skill or artifact is shared with. Mirrors MindsHub's
# bound on an artifact's rule; this server applies it to skills too.
MAX_TEAMS_PER_RULE = 100
# The longest id an artifact's rule is stored under. Mirrors MindsHub's
# bound of the same name on the ``artifact_id`` its artifact-access routes
# take: they answer 400 for a longer one.
MAX_ARTIFACT_ID_CHARS = 512


def mindshub_artifact_id(key: str) -> str:
    """``key`` as MindsHub's artifact-access routes read an artifact id:
    stripped of surrounding whitespace. "" for one they refuse with a 400:
    blank once stripped, longer than MAX_ARTIFACT_ID_CHARS, or holding a NUL
    or a lone surrogate, which no request body or database column can carry
    either."""
    key = key.strip()
    if len(key) > MAX_ARTIFACT_ID_CHARS or "\x00" in key or any("\ud800" <= ch <= "\udfff" for ch in key):
        return ""
    return key

# The most addresses one save of an artifact's share asks MindsHub to email.
# Mirrors MindsHub's bound on the share email; the ids it emails are bound
# by MAX_PEOPLE_PER_RULE.
MAX_NOTIFY_EMAILS = 100
# The longest artifact title the share email names. Mirrors MindsHub's bound.
MAX_NOTIFY_TITLE_CHARS = 200
# The longest artifact link the share email carries. Mirrors MindsHub's
# bound; MindsHub's own allowlist decides which hosts it links to.
MAX_NOTIFY_URL_CHARS = 2048

# The most ids, and the most addresses, one MindsHub org-member lookup takes.
# Mirrors MindsHub's bound; a longer batch is asked in chunks.
MAX_IDS_PER_LOOKUP = 50
MAX_EMAILS_PER_LOOKUP = 50
# The most ids, and the most addresses, one GET /hub/people/ may ask about:
# four MindsHub lookups' worth, each asked in turn, so one request can't fan
# out into an unbounded run of them. Owned by this server.
MAX_IDS_PER_REQUEST = 4 * MAX_IDS_PER_LOOKUP
MAX_EMAILS_PER_REQUEST = 4 * MAX_EMAILS_PER_LOOKUP
# The longest address MindsHub takes. Mirrors its address check (Django's
# validate_email), which cowork.services.mindshub_people.is_email_address
# ports.
MAX_EMAIL_CHARS = 320
# The longest ``ids`` and ``emails`` query values GET /hub/people/ reads,
# roomy enough for a full request: each UUID, or each longest address, with
# its comma and room for spaces around it. Owned by this server.
MAX_IDS_QUERY_CHARS = MAX_IDS_PER_REQUEST * (36 + 4)
MAX_EMAILS_QUERY_CHARS = MAX_EMAILS_PER_REQUEST * (MAX_EMAIL_CHARS + 4)
# The most members MindsHub's directory read returns; past it, it answers
# ``truncated``. Mirrors MindsHub's bound; this server passes the flag on.
MAX_DIRECTORY_ROWS = 500
