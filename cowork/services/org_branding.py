"""An organization's branding in hosted org mode (ENG-4020).

Three things live here:

- The org's look (cowork.models.org_appearance.OrgAppearance): read by every
  member, written by an org admin.
- The org's images, read from auth's ``GET /v1/organizations/current/branding/``
  with the caller's own bearer, cached per caller (CallerTtlCache, 60 s when
  auth answers). Auth is the source of truth for the images; cowork-server
  stores none of them. The org's name comes from auth's
  ``GET /v1/organizations/current/`` in the same read.
- The org's Brand guidelines skill (BRAND_GUIDELINES_SLUG), a folder in the
  org's skill store generated from both, with a skill_access row shared with
  everyone and always on, so it reaches every turn through the skills block
  (cowork.services.skills.build_turn_skills) like any always-on org skill,
  and each member's own off switch (skill_user_setting) still applies. It
  exists only while the org has some branding, and is rewritten only when
  its generated text changes (sync_brand_skill). SkillService refuses every
  normal edit, rename, delete or share of it (BRAND_GUIDELINES_REFUSAL).

The skill links the images at their fixed public URLs, auth's URLs without
``?v=``, so a new logo needs no regeneration.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from anton.core.tools.skill_format import SKILL_FILE

from cowork.db.scoped import ScopedSession
from cowork.models.org_appearance import OrgAppearance
from cowork.models.skill import META_CREATED_AT, META_DISPLAY_NAME, Skill
from cowork.schemas.org_branding import OrgAppearanceUpdate, OrgAppearanceView
from cowork.services.mindshub_client import (
    MINDSHUB_TTL_OK_S,
    CallerCacheKey,
    CallerTtlCache,
    MindsHubAnswer,
    get_mindshub_reply,
)
from cowork.services.skill_access import access_row, stage_access, stage_delete
from cowork.services.skill_sharing import SkillAccessRule
from cowork.services.skills import BRAND_GUIDELINES_SLUG, SkillService, _skill_from_dir

logger = logging.getLogger(__name__)

AUTH_BRANDING_PATH = "/organizations/current/branding/"
AUTH_CURRENT_ORG_PATH = "/organizations/current/"

BRAND_SKILL_DISPLAY_NAME = "Brand guidelines"
# The admin's guidelines that don't fit in SKILL.md go here, beside it, for
# the agent to read on demand.
GUIDELINES_FILE = "guidelines.md"
# What SKILL.md's body may use: under the pod's always-on per-skill cap
# (anton's ALWAYS_ON_SKILL_MAX_CHARS, 6000) with room to spare, so the always
# on section never truncates the generated instructions.
BRAND_SKILL_BODY_BUDGET = 5000

# The skill's access: everyone in the org, always on, owned by nobody.
BRAND_SKILL_RULE = SkillAccessRule(org_allowed=True, always_on=True)

_APPEARANCE_FIELDS = (
    "accent",
    "background",
    "font",
    "radius",
    "skin",
    "default_theme",
    "greeting",
    "members_can_choose",
    "guidelines",
)

# CSS stacks for the app's two font keys. Mono is the app's own stack
# (customTheme.ts MONO_STACK); standard is a system sans stack.
_FONT_STACKS = {
    "standard": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    "mono": "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace",
}


# ── the org's look ───────────────────────────────────────────────────────────


def appearance_row(session: ScopedSession) -> OrgAppearance | None:
    return session.exec(
        session.select(OrgAppearance).execution_options(populate_existing=True)
    ).first()


def appearance_view(row: OrgAppearance | None) -> OrgAppearanceView:
    if row is None:
        return OrgAppearanceView(members_can_choose=True)
    return OrgAppearanceView(**{name: getattr(row, name) for name in _APPEARANCE_FIELDS})


def save_appearance(session: ScopedSession, update: OrgAppearanceUpdate) -> OrgAppearance:
    """Apply ``update`` to the org's row, creating it, and commit. A field the
    body left out keeps its value; ``members_can_choose`` sent as null is
    the default, true."""
    row = appearance_row(session) or OrgAppearance()
    for name in update.model_fields_set:
        value = getattr(update, name)
        if name == "members_can_choose" and value is None:
            value = True
        setattr(row, name, value)
    try:
        session.add(row)
        session.commit()
    except Exception:
        session.rollback()
        raise
    session.refresh(row)
    return row


# ── the org's images, from auth ──────────────────────────────────────────────


@dataclass(frozen=True)
class AuthBranding:
    """The org's images and name as auth reported them to the caller.
    ``reachable`` false: auth's branding couldn't be read, so no image is
    known. ``org_known`` false: the org's name couldn't be read."""

    icon_url: str | None = None
    logo_url: str | None = None
    logo_dark_url: str | None = None
    org_name: str | None = None
    reachable: bool = False
    org_known: bool = False

    @property
    def complete(self) -> bool:
        """Both reads answered: enough to generate the skill from."""
        return self.reachable and self.org_known


_branding_cache: CallerTtlCache[AuthBranding] = CallerTtlCache(ttl_ok_s=MINDSHUB_TTL_OK_S)


def _absolute_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parts = urlsplit(value)
    if parts.scheme not in ("https", "http") or not parts.netloc:
        return None
    return value


def _org_name(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("display_name", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def fetch_auth_branding(*, bearer_token: str, org_id: str, user_id: str) -> AuthBranding:
    """The caller's org's images and name from auth, read with their own
    bearer and cached per caller. Never raises: anything but two answers
    from auth is an unreachable AuthBranding, cached for the short failure
    TTL, and the route still answers with null URLs."""
    if not bearer_token:
        return AuthBranding()
    key = CallerCacheKey.for_caller(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    cached = _branding_cache.fresh(key)
    if cached is not None:
        return cached
    branding_reply, org_reply = await asyncio.gather(
        get_mindshub_reply(AUTH_BRANDING_PATH, bearer_token),
        get_mindshub_reply(AUTH_CURRENT_ORG_PATH, bearer_token),
    )
    payload = branding_reply.payload
    if branding_reply.answer is MindsHubAnswer.OK and isinstance(payload, dict):
        value = AuthBranding(
            icon_url=_absolute_url(payload.get("icon_url")),
            logo_url=_absolute_url(payload.get("logo_url")),
            logo_dark_url=_absolute_url(payload.get("logo_dark_url")),
            org_name=_org_name(org_reply.payload),
            reachable=True,
            org_known=org_reply.answer is MindsHubAnswer.OK,
        )
    else:
        value = AuthBranding()
    _branding_cache.put(key, value)
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


def has_brand_content(images: AuthBranding, row: OrgAppearance | None) -> bool:
    """Whether the org has anything the skill can teach: an image, a color,
    a font, a radius, a light/dark default or the admin's guidelines. The
    app-only fields (skin, greeting, members_can_choose) alone don't make a
    skill."""
    if images.icon_url or images.logo_url or images.logo_dark_url:
        return True
    if row is None:
        return False
    return any(
        getattr(row, name) is not None
        for name in ("accent", "background", "font", "radius", "default_theme", "guidelines")
    )


def _cut(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` chars at a line boundary."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = head.rfind("\n")
    return (head[:cut] if cut > 0 else head).rstrip()


def render_brand_skill(images: AuthBranding, row: OrgAppearance | None) -> tuple[str, str, str | None]:
    """The skill's description, SKILL.md body and, when the admin's
    guidelines don't fit in the body, the full guidelines for GUIDELINES_FILE."""
    org = images.org_name or "the organization"
    # "Acme's", or "the organization's" with no name; capitalized as a heading.
    owner = f"{images.org_name}'s" if images.org_name else "the organization's"
    owner_title = owner[0].upper() + owner[1:]
    logo = fixed_image_url(images.logo_url)
    logo_dark = fixed_image_url(images.logo_dark_url)
    icon = fixed_image_url(images.icon_url)
    accent = row.accent if row else None
    background = row.background if row else None
    font = row.font if row else None
    radius = row.radius if row else None
    theme = row.default_theme if row else None
    guidelines = row.guidelines if row else None

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

    if accent or background or theme:
        lines += ["", "## Colors", ""]
        if accent:
            lines.append(
                f"- Accent `{accent}`: primary buttons, links, highlights, key numbers, the first "
                f"chart series. Text on it: `{_text_on(accent)}`."
            )
        if background:
            lines.append(
                f"- Background `{background}`: page, slide and app background. Text on it: "
                f"`{_text_on(background)}`."
            )
        if theme:
            lines.append(f"- Default to a {theme} look unless the user asks otherwise.")
        if accent and background and contrast_ratio(accent, background) < 3:
            lines.append(
                "- The accent is low-contrast on the background: use it for fills and large "
                "shapes, not for text or thin lines on that background."
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


def sync_brand_skill(
    session: ScopedSession, images: AuthBranding, row: OrgAppearance | None
) -> BrandSkillSync:
    """Make the org's Brand guidelines skill match its branding: written (and
    its always-on, everyone row staged and committed) when the org has some,
    removed when it has none. Writes only what changed, so calling it on
    every read is cheap. Call it only with ``images`` auth answered in full
    (AuthBranding.complete): unknown images must not take the logo out of
    the skill.
    """
    service = SkillService(session.scope)
    skill_dir = service.root / BRAND_GUIDELINES_SLUG
    if not has_brand_content(images, row):
        changed = False
        if skill_dir.exists():
            service._rmtree_direct_child(service.root, BRAND_GUIDELINES_SLUG, ignore_errors=True)
            changed = True
        if access_row(session, BRAND_GUIDELINES_SLUG) is not None:
            stage_delete(session, BRAND_GUIDELINES_SLUG)
            session.commit()
            changed = True
        return BrandSkillSync(present=False, changed=changed)

    description, body, extra = render_brand_skill(images, row)
    changed = False
    current = _skill_from_dir(skill_dir) if (skill_dir / SKILL_FILE).exists() else None
    if (
        current is None
        or current.description != description
        or current.instructions.strip() != body.strip()
        or current.display_name != BRAND_SKILL_DISPLAY_NAME
    ):
        created_at = (
            current.metadata.get(META_CREATED_AT) if current is not None else None
        ) or datetime.now(UTC).isoformat()
        skill = Skill(
            name=BRAND_GUIDELINES_SLUG,
            description=description,
            instructions=body,
            metadata={META_DISPLAY_NAME: BRAND_SKILL_DISPLAY_NAME, META_CREATED_AT: created_at},
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
