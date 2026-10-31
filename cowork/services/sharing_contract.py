"""What sharing a skill or an artifact agrees on with MindsHub and the app.

The names of who a rule is shared with, the MindsHub reads and capabilities
sharing uses, and every bound this server owns or mirrors. A bound MindsHub
enforces is mirrored here so a share it would refuse is refused before
anything is published; each comment says which side owns it. The app
mirrors two of them, MAX_IDS_PER_LOOKUP and MAX_EMAILS_PER_LOOKUP, in its
sharing contract module (src/renderer/cowork/lib/sharingContract.js).
"""

from __future__ import annotations

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
