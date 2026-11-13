"""One-time pairing codes: the owner creates one, a sender types it to the bot and is let in.

A code is 8 characters from Crockford's base32 alphabet, lives 10 minutes,
works once, and is stored only as its sha256. It lets in one sender, in the
chat where they typed it.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlmodel import col

from cowork.db.scoped import ScopedSession, unsafe_unscoped_session
from cowork.models.channel import (
    SENDER_ALLOWED,
    SENDER_DISPLAY_NAME_MAX,
    ChannelPairingCode,
    ChannelSender,
)
from cowork.services.channel_senders import ChannelSenderService, never_allowed

CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
CODE_LENGTH = 8
CODE_TTL = timedelta(minutes=10)
UNUSED_CODES_MAX = 5
PAIRED_NOTICE = "You're paired. Send your message again and I'll answer."

_CODE = re.compile(rf"[{CODE_ALPHABET}]{{{CODE_LENGTH}}}")
# Platform mentions of the bot: Telegram/Discord-style @name, Slack <@U1|name>,
# Discord <@id> and <@!id>.
_MENTION = re.compile(r"<@!?[^>\s]+>|@\w+")


class PairingCodeLimitError(Exception):
    """The channel already has UNUSED_CODES_MAX codes that are neither used nor expired (-> 409)."""


@dataclass(frozen=True)
class IssuedCode:
    """A new code, returned to the owner once; only its hash is kept."""

    code: str
    expires_at: datetime


def hash_code(code: str) -> str:
    """sha256 hex of the code, as stored."""
    return hashlib.sha256(code.upper().encode("ascii")).hexdigest()


def code_in(text: str) -> str | None:
    """The pairing code a message carries, or None.

    The whole message, with mentions of the bot and a leading Telegram
    ``/start`` removed, must be the code; case does not matter.
    """
    words = _MENTION.sub(" ", text or "").split()
    if words and words[0].lower().split("@", 1)[0] == "/start":
        words = words[1:]
    if len(words) != 1:
        return None
    candidate = words[0].upper()
    return candidate if _CODE.fullmatch(candidate) else None


class ChannelPairingService:
    """Creates and redeems pairing codes, in one scoped session."""

    def __init__(self, session: ScopedSession) -> None:
        """Bind the service to a scoped session; code lookups are org scoped by it."""
        self.session = session

    def create(self, channel_type: str) -> IssuedCode:
        """Make a new code for the channel and return it once.

        Raises PairingCodeLimitError when UNUSED_CODES_MAX live codes exist.
        Side effects: commits the code's hash.
        """
        now = datetime.now(timezone.utc)
        live = [
            row for row in self.session.exec(
                self.session.select(ChannelPairingCode).where(
                    ChannelPairingCode.channel_type == channel_type,
                    col(ChannelPairingCode.used_at).is_(None),
                )
            ).all()
            if _aware(row.expires_at) > now
        ]
        if len(live) >= UNUSED_CODES_MAX:
            raise PairingCodeLimitError(f"{channel_type} already has {UNUSED_CODES_MAX} unused pairing codes")
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        expires_at = now + CODE_TTL
        self.session.add(ChannelPairingCode(channel_type=channel_type, code_hash=hash_code(code), expires_at=expires_at))
        self.session.commit()
        return IssuedCode(code=code, expires_at=expires_at)

    def redeem(
        self,
        *,
        channel_type: str,
        code: str,
        group_id: str,
        sender_id: str,
        sender_name: str | None,
        is_group: bool,
    ) -> bool:
        """Use the code up and allow the sender in that chat, in one transaction.

        True only for the one caller whose conditional update claimed an
        unused, unexpired code of this channel; a second caller with the same
        code gets False. A platform account that stands for many people never
        pairs, and its attempt leaves the code unused. Side effects: commits
        the code's use and the sender.
        """
        if never_allowed(channel_type, sender_id):
            return False
        for _attempt in range(2):
            row = self.session.exec(
                self.session.select(ChannelPairingCode).where(
                    ChannelPairingCode.channel_type == channel_type,
                    ChannelPairingCode.code_hash == hash_code(code),
                )
            ).first()
            if row is None:
                return False
            if not self._claim(row.id):
                self.session.rollback()
                return False
            try:
                self._allow(channel_type, group_id, sender_id, sender_name, is_group)
                self.session.commit()
                return True
            except IntegrityError:
                # Another thread of the chat wrote the sender's first row; the
                # rollback also returns the code, so the next pass claims it again.
                self.session.rollback()
        return False

    def _claim(self, code_id) -> bool:
        """Mark the code used if it is still unused and unexpired; True when this call did."""
        now = datetime.now(timezone.utc)
        raw = unsafe_unscoped_session(self.session)
        claimed = raw.execute(
            sa.update(ChannelPairingCode)
            .where(ChannelPairingCode.id == code_id)
            .where(col(ChannelPairingCode.used_at).is_(None))
            .where(ChannelPairingCode.expires_at > now)
            .values(used_at=now)
            .execution_options(synchronize_session=False)
        ).rowcount
        return claimed == 1

    def _allow(self, channel_type: str, group_id: str, sender_id: str, sender_name: str | None, is_group: bool) -> None:
        """Set the sender allowed in the chat, adding their row when they have none. Flushes."""
        row = ChannelSenderService(self.session).find(channel_type, group_id, sender_id)
        if row is None:
            row = ChannelSender(
                channel_type=channel_type,
                external_group_id=group_id,
                external_sender_id=sender_id,
                is_group=is_group,
            )
        row.status = SENDER_ALLOWED
        row.last_seen_at = datetime.now(timezone.utc)
        if sender_name:
            row.display_name = sender_name[:SENDER_DISPLAY_NAME_MAX]
        self.session.add(row)
        self.session.flush()


def _aware(value: datetime) -> datetime:
    """The stored time as UTC-aware; SQLite returns it naive."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
