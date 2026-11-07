"""An organization's branding, read from and written to auth (ENG-4020).

auth is the source of truth for the whole org look: the images, the
appearance (colors, font, radius, skin, theme, greeting, sidebar title,
whether members may choose their own look, and the admin's guidelines), the
org's name, the resolved sidebar title and whether the caller may edit it
(``GET /v1/organizations/current/branding/``; ``PATCH
/v1/organizations/current/branding/appearance/``, gated by auth's
``organization.branding.manage``). cowork-server stores none of it. Its
routes (cowork.api.v1.endpoints.org_branding) pass the caller's own
credential through, and this module reads auth's answer, caches it per
caller (CallerTtlCache, at most 60 s; the caller's own write replaces their
entry) and keeps the org's Brand guidelines skill in step with it.

The skill (BRAND_GUIDELINES_SLUG) is a folder generated from auth's data,
marked as managed (META_MANAGED_BY) and rewritten only when its text
changes (sync_brand_skill):

- In hosted org mode it sits in the org's store with a skill_access row
  shared with everyone and always on, so it reaches every turn through the
  skills block (cowork.services.skills.build_turn_skills), and each
  member's own off switch (skill_user_setting) still applies.
- On a local server (the desktop) it is an ordinary skill of the local
  store, recalled on demand: the in-process turn has no always-on path.

It exists only while the org has some branding. SkillService refuses every
normal edit, rename, delete or share of it (BRAND_GUIDELINES_REFUSAL). The
skill links the images at their fixed public URLs, auth's URLs without
``?v=``, so a new logo needs no regeneration.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from anton.core.tools.skill_format import SKILL_FILE

from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.skill import META_CREATED_AT, META_DISPLAY_NAME, Skill
from cowork.services.mindshub_client import (
    MINDSHUB_TTL_OK_S,
    CallerCacheKey,
    CallerTtlCache,
    MindsHubAnswer,
    MindsHubReply,
    get_mindshub_reply,
    send_mindshub,
)
from cowork.services.skill_access import access_row, stage_access, stage_delete
from cowork.services.skill_sharing import SkillAccessRule
from cowork.services.skills import (
    BRAND_GUIDELINES_MANAGER,
    BRAND_GUIDELINES_SLUG,
    META_MANAGED_BY,
    SkillService,
    _skill_from_dir,
)

logger = logging.getLogger(__name__)

AUTH_BRANDING_PATH = "/organizations/current/branding/"
AUTH_APPEARANCE_PATH = "/organizations/current/branding/appearance/"

BRAND_SKILL_DISPLAY_NAME = "Brand guidelines"
# The admin's guidelines that don't fit in SKILL.md go here, beside it, for
# the agent to read on demand.
GUIDELINES_FILE = "guidelines.md"
# What SKILL.md's body may use: under the pod's always-on per-skill cap
# (anton's ALWAYS_ON_SKILL_MAX_CHARS, 6000) with room to spare, so the always
# on section never truncates the generated instructions.
BRAND_SKILL_BODY_BUDGET = 5000

# The skill's access in hosted org mode: everyone in the org, always on,
# owned by nobody.
BRAND_SKILL_RULE = SkillAccessRule(org_allowed=True, always_on=True)

# The sidebar title when auth couldn't be asked.
DEFAULT_SIDEBAR_TITLE = "MindsHub"

# CSS stacks for the app's two font keys. Mono is the app's own stack
# (customTheme.ts MONO_STACK); standard is a system sans stack.
_FONT_STACKS = {
    "standard": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    "mono": "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace",
}


# ── auth's answer ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BrandLook:
    """The org's appearance as auth reports it. Every field may be None
    (stock), except the two auth always resolves."""

    accent: str | None = None
    background_light: str | None = None
    background_dark: str | None = None
    font: str | None = None
    radius: int | None = None
    skin: str | None = None
    default_theme: str | None = None
    greeting: str | None = None
    sidebar_title_mode: str = "company"
    sidebar_title: str | None = None
    members_can_choose: bool = True
    guidelines: str | None = None


STOCK_LOOK = BrandLook()


@dataclass(frozen=True)
class AuthBranding:
    """auth's answer for the caller's current org.

    ``answer`` is how auth answered: OK; REFUSED, which here means the caller
    has no current org (a personal account, or signed out of every org) and
    branding is off; or UNREACHABLE (also THROTTLED), when nothing is known.
    Only an OK answer may change the skill."""

    answer: MindsHubAnswer = MindsHubAnswer.UNREACHABLE
    icon_url: str | None = None
    logo_url: str | None = None
    logo_dark_url: str | None = None
    org_name: str | None = None
    sidebar_title: str | None = DEFAULT_SIDEBAR_TITLE
    can_edit: bool = False
    look: BrandLook = STOCK_LOOK

    @property
    def reachable(self) -> bool:
        """auth answered (the cache's TTL choice: a refusal is an answer)."""
        return self.answer in (MindsHubAnswer.OK, MindsHubAnswer.REFUSED)

    @property
    def known(self) -> bool:
        """auth answered with the org's branding: enough to generate the skill from."""
        return self.answer is MindsHubAnswer.OK

    @property
    def enabled(self) -> bool:
        """Branding applies: auth answered for an org, or couldn't be asked
        (the app then shows the stock look)."""
        return self.answer is not MindsHubAnswer.REFUSED


def _absolute_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parts = urlsplit(value)
    if parts.scheme not in ("https", "http") or not parts.netloc:
        return None
    return value


def _text(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _look(payload: Any) -> BrandLook:
    if not isinstance(payload, dict):
        return STOCK_LOOK
    radius = payload.get("radius")
    mode = payload.get("sidebar_title_mode")
    members = payload.get("members_can_choose")
    return BrandLook(
        accent=_text(payload.get("accent")),
        background_light=_text(payload.get("background_light")),
        background_dark=_text(payload.get("background_dark")),
        font=_text(payload.get("font")),
        radius=radius if isinstance(radius, int) and not isinstance(radius, bool) else None,
        skin=_text(payload.get("skin")),
        default_theme=_text(payload.get("default_theme")),
        greeting=_text(payload.get("greeting")),
        sidebar_title_mode=mode if mode in ("company", "custom", "none") else "company",
        sidebar_title=_text(payload.get("sidebar_title")),
        members_can_choose=members if isinstance(members, bool) else True,
        guidelines=_text(payload.get("guidelines")),
    )


def parse_auth_branding(payload: Any) -> AuthBranding:
    """auth's branding body (GET, or the PATCH that answers the same body)."""
    if not isinstance(payload, dict):
        return AuthBranding()
    sidebar_title = payload.get("sidebar_title")
    return AuthBranding(
        answer=MindsHubAnswer.OK,
        icon_url=_absolute_url(payload.get("icon_url")),
        logo_url=_absolute_url(payload.get("logo_url")),
        logo_dark_url=_absolute_url(payload.get("logo_dark_url")),
        org_name=_text(payload.get("org_name")),
        # auth resolves it; null is the 'none' mode (only the logo).
        sidebar_title=sidebar_title if isinstance(sidebar_title, str) and sidebar_title else None,
        can_edit=payload.get("can_edit") is True,
        look=_look(payload.get("appearance")),
    )


_branding_cache: CallerTtlCache[AuthBranding] = CallerTtlCache(ttl_ok_s=MINDSHUB_TTL_OK_S)


def _cache_key(*, bearer_token: str, org_id: str, user_id: str) -> CallerCacheKey:
    return CallerCacheKey.for_caller(org_id=org_id, user_id=user_id, bearer_token=bearer_token)


async def _read_auth_branding(bearer_token: str) -> AuthBranding:
    reply = await get_mindshub_reply(AUTH_BRANDING_PATH, bearer_token)
    if reply.answer is MindsHubAnswer.OK and isinstance(reply.payload, dict):
        return parse_auth_branding(reply.payload)
    if reply.answer is MindsHubAnswer.REFUSED:
        return AuthBranding(answer=MindsHubAnswer.REFUSED)
    return AuthBranding()


async def fetch_auth_branding(*, bearer_token: str, org_id: str, user_id: str) -> AuthBranding:
    """The caller's current org's branding from auth, read with their own
    credential and cached per caller. No credential is no org. Never raises."""
    if not bearer_token:
        return AuthBranding(answer=MindsHubAnswer.REFUSED)
    key = _cache_key(bearer_token=bearer_token, org_id=org_id, user_id=user_id)
    cached = _branding_cache.fresh(key)
    if cached is not None:
        return cached
    value = await _read_auth_branding(bearer_token)
    _branding_cache.put(key, value)
    return value


class AuthAppearanceError(Exception):
    """auth refused or couldn't take an appearance change. ``status_code`` is
    what the route answers; ``detail`` is auth's own message, or ours when
    auth gave none."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


APPEARANCE_UNREACHABLE = "Couldn't reach MindsHub to save the organization's branding. Try again."


def _auth_message(payload: Any) -> str | None:
    """auth's own message from a refusal, picked from the keys auth uses and
    only as text: ``detail`` (or ``message``), else DRF's field errors
    (``{"field": ["message", ...]}``) as "field: message; ...". Nothing else
    of auth's body is passed on."""
    if not isinstance(payload, dict):
        return None
    for key in ("detail", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:500]
    errors = [
        f"{field}: {message}"
        for field, messages in payload.items()
        if isinstance(field, str) and field in _AUTH_FIELD_NAMES and isinstance(messages, list)
        for message in messages
        if isinstance(message, str)
    ]
    return "; ".join(errors)[:500] or None


# The fields whose errors auth's 422 may name.
_AUTH_FIELD_NAMES = frozenset(
    {
        "accent", "background_light", "background_dark", "font", "radius", "skin", "default_theme",
        "greeting", "sidebar_title_mode", "sidebar_title", "members_can_choose", "guidelines",
        "non_field_errors",
    }
)


def _refusal(reply: MindsHubReply) -> AuthAppearanceError:
    message = _auth_message(reply.payload)
    code = reply.status_code or 400
    if code in (400, 422):
        return AuthAppearanceError(422, message or "The organization's branding was refused.")
    if code in (401, 403, 404):
        return AuthAppearanceError(code, message or "You can't change the organization's branding.")
    return AuthAppearanceError(code, message or APPEARANCE_UNREACHABLE)


async def update_auth_appearance(
    *, bearer_token: str, org_id: str, user_id: str, changes: dict[str, Any]
) -> AuthBranding:
    """PATCH ``changes`` to auth's org appearance as the caller, and answer
    the branding after it. The caller's cache entry is replaced, so their own
    next read shows the change. AuthAppearanceError on any other answer:
    auth's 403 and 422 (a 400 too) keep their status and auth's message; no
    answer is a 502."""
    reply = await send_mindshub("PATCH", AUTH_APPEARANCE_PATH, bearer_token, json_body=changes)
    if reply.answer is MindsHubAnswer.REFUSED or reply.answer is MindsHubAnswer.THROTTLED:
        raise _refusal(reply)
    if reply.answer is not MindsHubAnswer.OK:
        raise AuthAppearanceError(502, APPEARANCE_UNREACHABLE)
    if isinstance(reply.payload, dict) and "appearance" in reply.payload:
        value = parse_auth_branding(reply.payload)
    else:
        # A PATCH that answers less than the whole body: read it again.
        value = await _read_auth_branding(bearer_token)
    _branding_cache.put(_cache_key(bearer_token=bearer_token, org_id=org_id, user_id=user_id), value)
    return value


def fixed_image_url(url: str | None) -> str | None:
    """Auth's image URL without its ``?v=<hash>``: the fixed public URL that
    always serves the org's current image."""
    if not url:
        return None
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


# ── the Brand guidelines skill ───────────────────────────────────────────────


def _rgb(hex_color: str) -> tuple[int, int, int]:
    n = int(hex_color[1:], 16)
    return (n >> 16) & 255, (n >> 8) & 255, n & 255


def _relative_luminance(hex_color: str) -> float:
    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in _rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio of two #rrggbb colors."""
    la, lb = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _text_on(color: str) -> str:
    """White or near-black, whichever reads better on ``color``."""
    return max(("#ffffff", "#111111"), key=lambda ink: contrast_ratio(color, ink))


def has_brand_content(branding: AuthBranding) -> bool:
    """Whether the org has anything the skill can teach: an image, a color,
    a font, a radius, a light/dark default or the admin's guidelines. The
    app-only fields (skin, greeting, the sidebar title, members_can_choose)
    alone don't make a skill."""
    if branding.icon_url or branding.logo_url or branding.logo_dark_url:
        return True
    look = branding.look
    return any(
        getattr(look, name) is not None
        for name in ("accent", "background_light", "background_dark", "font", "radius", "default_theme", "guidelines")
    )


def _cut(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` chars at a line boundary."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = head.rfind("\n")
    return (head[:cut] if cut > 0 else head).rstrip()


def render_brand_skill(images: AuthBranding) -> tuple[str, str, str | None]:
    """The skill's description, SKILL.md body and, when the admin's
    guidelines don't fit in the body, the full guidelines for GUIDELINES_FILE."""
    org = images.org_name or "the organization"
    # "Acme's", or "the organization's" with no name; capitalized as a heading.
    owner = f"{images.org_name}'s" if images.org_name else "the organization's"
    owner_title = owner[0].upper() + owner[1:]
    logo = fixed_image_url(images.logo_url)
    logo_dark = fixed_image_url(images.logo_dark_url)
    icon = fixed_image_url(images.icon_url)
    row = images.look
    accent = row.accent
    background_light, background_dark = row.background_light, row.background_dark
    font = row.font
    radius = row.radius
    theme = row.default_theme
    guidelines = row.guidelines

    description = (
        f"{owner_title} brand: logo, colors, type and voice. Apply it to every deck, document, "
        "report, HTML page or app, dashboard and chart you make."
    )
    lines = [
        f"# {images.org_name} brand guidelines" if images.org_name else "# Brand guidelines",
        "",
        f"Use {owner} brand in everything visual or shareable you make: slide decks, documents, "
        "reports, HTML pages and apps, dashboards and charts. Plain chat answers don't need it. "
        "The user's explicit instructions in the conversation win over these.",
    ]
    # The org's name is used throughout whatever the sidebar shows; an
    # admin's custom title is only named as the assistant's.
    title = row.sidebar_title if row.sidebar_title_mode == "custom" else None
    if title and title != images.org_name:
        lines += [
            "",
            f"In {owner} workspace the assistant appears as **{title}**: use that name when "
            "branded output refers to the assistant or the workspace.",
        ]

    if logo or logo_dark or icon:
        lines += ["", "## Logo and icon", ""]
        if logo:
            lines.append(f"- Logo (for light backgrounds): {logo}")
        if logo_dark:
            lines.append(f"- Logo for dark backgrounds: {logo_dark}")
        if icon:
            lines.append(f"- Square icon: {icon}")
        lines += [
            "",
            "These are stable public image URLs: link them as they are, e.g. "
            f'`<img src="{logo or logo_dark or icon}" alt="{org} logo" style="height:32px">`. '
            "Never redraw, recolor, stretch, crop or re-type the logo, and never invent one. "
            "Keep clear space around it at least half its height.",
        ]
        if logo and logo_dark:
            lines.append("On a dark background use the dark-background logo.")
        elif logo:
            lines.append("On a dark background, place the logo on a light panel rather than recoloring it.")

    if accent or background_light or background_dark or theme:
        lines += ["", "## Colors", ""]
        if accent:
            lines.append(
                f"- Accent `{accent}`: primary buttons, links, highlights, key numbers, the first "
                f"chart series. Text on it: `{_text_on(accent)}`."
            )
        if background_light and background_light == background_dark:
            lines.append(
                f"- Background `{background_light}`: page, slide and app background. Text on it: "
                f"`{_text_on(background_light)}`."
            )
        else:
            if background_light:
                lines.append(
                    f"- Light background `{background_light}`: page, slide and app background in a "
                    f"light look. Text on it: `{_text_on(background_light)}`."
                )
            if background_dark:
                lines.append(
                    f"- Dark background `{background_dark}`: page, slide and app background in a "
                    f"dark look. Text on it: `{_text_on(background_dark)}`."
                )
        if theme:
            lines.append(f"- Default to a {theme} look unless the user asks otherwise.")
        low_contrast = [
            name
            for name, color in (("light", background_light), ("dark", background_dark))
            if accent and color and contrast_ratio(accent, color) < 3
        ]
        if accent and low_contrast:
            where = (
                "the background"
                if background_light == background_dark or len(low_contrast) == 2
                else f"the {low_contrast[0]} background"
            )
            lines.append(
                f"- The accent is low-contrast on {where}: use it there for fills and large "
                "shapes, not for text or thin lines."
            )

    if font or radius is not None:
        lines += ["", "## Type and shape", ""]
        if font:
            lines.append(f"- Font: `font-family: {_FONT_STACKS.get(font, _FONT_STACKS['standard'])}`.")
        if radius is not None:
            lines.append(f"- Corner radius {radius}px on cards, buttons, inputs and image frames.")

    lines += [
        "",
        "## How to apply",
        "",
        "- Decks and documents: logo on the cover or title slide and small in the header or "
        "footer; accent for headings' rules, callouts and highlights.",
        "- HTML pages and apps: logo in the header, "
        + (f'`<link rel="icon" href="{icon}">` as the favicon, ' if icon else "")
        + "brand values as CSS variables"
        + (f" (`--brand-accent: {accent};`)" if accent else "")
        + ".",
        "- Charts: the accent for the first or highlighted series, tints and shades of it and "
        "neutral grays for the rest; no rainbow palettes.",
        "- Contrast: body text at least 4.5:1 against its background, large text and chart marks "
        "at least 3:1. If a brand color fails, darken or lighten it for that use.",
    ]

    body = "\n".join(lines)
    extra: str | None = None
    if guidelines:
        heading = f"\n\n## {owner_title} own guidelines\n\n"
        room = BRAND_SKILL_BODY_BUDGET - len(body) - len(heading)
        if len(guidelines) <= room:
            body += heading + guidelines
        else:
            note = f"\n\n(Continued: read {GUIDELINES_FILE} in this skill's folder for the rest.)"
            body += heading + _cut(guidelines, max(room - len(note), 0)) + note
            extra = guidelines
    return description, body + "\n", extra


@dataclass(frozen=True)
class BrandSkillSync:
    """What sync_brand_skill did: ``present`` whether the skill now exists,
    ``changed`` whether anything was written or removed."""

    present: bool
    changed: bool


def _write_beside_skill(service: SkillService, skill_dir: Path, name: str, text: str) -> None:
    """Write ``name`` in the skill's folder atomically, against a pinned
    descriptor, as SkillService writes SKILL.md: a temporary file, then a
    replace that never follows a link."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=skill_dir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        service._replace_direct_child(skill_dir, Path(tmp_name).name, name)
    except Exception:
        service._unlink_direct_child(skill_dir, Path(tmp_name).name, ignore_errors=True)
        raise


def _current_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _is_ours(skill_dir: Path) -> bool:
    """Whether the folder holds the generated skill (META_MANAGED_BY). On a
    local server a person's own skill may already be named
    BRAND_GUIDELINES_SLUG; it is never overwritten or removed."""
    skill = _skill_from_dir(skill_dir) if (skill_dir / SKILL_FILE).exists() else None
    return skill is not None and skill.metadata.get(META_MANAGED_BY) == BRAND_GUIDELINES_MANAGER


def remove_brand_skill(scope: TenantScope | None, session: ScopedSession | None = None) -> bool:
    """Take the generated skill out of ``scope``'s store, and in hosted org
    mode its skill_access row. Returns whether anything was removed. A
    local folder that isn't ours is left alone."""
    service = SkillService(scope)
    skill_dir = service.root / BRAND_GUIDELINES_SLUG
    changed = False
    # In hosted org mode the name is reserved for it, so the folder is ours
    # whether or not an earlier build marked it.
    if skill_dir.exists() and (service.org_sharing or _is_ours(skill_dir)):
        service._rmtree_direct_child(service.root, BRAND_GUIDELINES_SLUG, ignore_errors=True)
        changed = True
    if service.org_sharing and session is not None and access_row(session, BRAND_GUIDELINES_SLUG) is not None:
        try:
            stage_delete(session, BRAND_GUIDELINES_SLUG)
            session.commit()
        except Exception:
            session.rollback()
            raise
        changed = True
    return changed


def sync_brand_skill(
    scope: TenantScope | None, branding: AuthBranding, session: ScopedSession | None = None
) -> BrandSkillSync:
    """Make the Brand guidelines skill of ``scope``'s store match the org's
    branding: written when the org has some, removed when it has none, and
    on a local server when the caller has no org. Writes only what changed, so calling it on
    every read is cheap. An answer from auth that isn't known changes
    nothing: unknown images must not take the logo out of the skill.

    In hosted org mode ``session`` stages and commits the skill's
    skill_access row (everyone, always on). On a local server it is an
    ordinary skill, recalled on demand, and a person's own skill of the same
    name is left as it is.
    """
    service = SkillService(scope)
    if branding.answer is MindsHubAnswer.REFUSED:
        # No org for this caller. On a local server that is the person's own
        # store: signed out, or in a personal account. In hosted org mode one
        # member's refused read must not take the org's skill from everyone.
        if service.org_sharing:
            return BrandSkillSync(present=False, changed=False)
        return BrandSkillSync(present=False, changed=remove_brand_skill(scope, session))
    if not branding.known:
        return BrandSkillSync(present=False, changed=False)
    if not has_brand_content(branding):
        return BrandSkillSync(present=False, changed=remove_brand_skill(scope, session))

    skill_dir = service.root / BRAND_GUIDELINES_SLUG
    if not service.org_sharing and skill_dir.exists() and not _is_ours(skill_dir):
        logger.info(
            "Leaving a local skill named %s alone: it isn't the generated one",
            BRAND_GUIDELINES_SLUG,
            extra={"event": "org_branding_skill_name_taken"},
        )
        return BrandSkillSync(present=False, changed=False)

    description, body, extra = render_brand_skill(branding)
    changed = False
    current = _skill_from_dir(skill_dir) if (skill_dir / SKILL_FILE).exists() else None
    if (
        current is None
        or current.description != description
        or current.instructions.strip() != body.strip()
        or current.display_name != BRAND_SKILL_DISPLAY_NAME
        or current.metadata.get(META_MANAGED_BY) != BRAND_GUIDELINES_MANAGER
    ):
        created_at = (
            current.metadata.get(META_CREATED_AT) if current is not None else None
        ) or datetime.now(UTC).isoformat()
        skill = Skill(
            name=BRAND_GUIDELINES_SLUG,
            description=description,
            instructions=body,
            metadata={
                META_DISPLAY_NAME: BRAND_SKILL_DISPLAY_NAME,
                META_CREATED_AT: created_at,
                META_MANAGED_BY: BRAND_GUIDELINES_MANAGER,
            },
        )
        service._write(skill, reconcile=False)
        changed = True

    extra_path = skill_dir / GUIDELINES_FILE
    if extra is None:
        if extra_path.exists():
            service._unlink_direct_child(skill_dir, GUIDELINES_FILE, ignore_errors=True)
            changed = True
    elif _current_text(extra_path) != extra:
        _write_beside_skill(service, skill_dir, GUIDELINES_FILE, extra)
        changed = True

    if service.org_sharing and session is not None:
        row_now = access_row(session, BRAND_GUIDELINES_SLUG)
        if row_now is None or SkillAccessRule.from_row(row_now) != BRAND_SKILL_RULE.normalized():
            try:
                stage_access(session, BRAND_GUIDELINES_SLUG, BRAND_SKILL_RULE)
                session.commit()
            except Exception:
                session.rollback()
                raise
            changed = True
    return BrandSkillSync(present=True, changed=changed)
