"""The credential the autopublish reconciler publishes with.

Who owns a published artifact on view.mindshub.ai is decided by the token used to
upload it: `html_upload` takes `owner_keycloak_id` from the token and folds
`md5(user_id)[:9]` into the URL. So publishing on a user's behalf requires a turn
key minted for that user.

Three properties, each one load-bearing:

* **Lazy.** Nothing is minted until a publish actually happens, so a turn that
  produced no publishable change costs no auth round-trip.
* **One key per reconciliation.** A key per slug would leave up to `limit`
  un-revoked keys per turn, each alive for `turn_key_ttl_seconds` (20 minutes by
  default).
* **A fresh uuid4 `instance_id`, never the turn's correlation id.** Auth keys
  active turn keys on `(user, organization, instance_id)` and answers an
  idempotent hit with **200 and `key: None`** — which `raise_for_status()` does
  not catch. In org mode a key for the turn's own correlation id already exists
  (minted for inference), so reusing that id would reliably yield `None`. The
  turn's correlation id is not even reachable here: the harness does not receive
  one.

`get()` returns None on any failure — a missing key means "skip publishing this
turn", never a TypeError inside the publisher. A failed mint is not retried within
one reconciliation: the cause (missing internal secret, unreachable auth) will not
clear in the next few hundred milliseconds.
"""
from __future__ import annotations

import logging
import uuid

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.turnqueue.auth_keys import mint_turn_key, revoke_turn_key

logger = logging.getLogger(__name__)

# Ceiling for the requested TTL. auth rejects an `expiry_date` beyond
# `turn_key_max_ttl_seconds` with a 400 (auth/common/app_settings.py, default
# 3600s), and that failure looks like "publishing silently stopped working". The
# cap lives here because cowork-server cannot read auth's setting; keep it at or
# below auth's default, and re-check it against the target deployment.
MAX_PUBLISH_KEY_TTL_S = 3600


class PublishKey:
    def __init__(self, user_id: str, org_id: str, *, min_ttl_s: float) -> None:
        self._user_id = user_id
        self._org_id = org_id
        self._min_ttl_s = min_ttl_s
        self._instance_id = str(uuid.uuid4())
        self._key: str | None = None
        self._attempted = False

    @property
    def instance_id(self) -> str:
        return self._instance_id

    async def get(self) -> str | None:
        """The publish credential, minting it on first use. None on failure."""
        if self._attempted:
            return self._key
        self._attempted = True
        settings = TurnQueueSettings()
        # The key must outlive an abandoned upload thread (asyncio.to_thread is
        # not cancellable), otherwise that thread reaches /upload with an expired
        # credential — and it must stay under auth's cap, which rejects anything
        # longer with a 400.
        ttl = int(min(max(settings.turn_key_ttl_seconds, self._min_ttl_s), MAX_PUBLISH_KEY_TTL_S))
        try:
            self._key = await mint_turn_key(
                user_id=self._user_id,
                org_id=self._org_id,
                correlation_id=self._instance_id,
                ttl_seconds=ttl,
                settings=settings,
            )
        except Exception:
            logger.warning(
                "artifact_autopublish result=no_key reason=mint_failed instance_id=%s",
                self._instance_id,
                exc_info=True,
            )
            self._key = None
            return None
        if not self._key:
            # 200 with no plaintext — an active key already exists for this
            # (user, org, instance_id). Should be unreachable with a fresh uuid4,
            # so it means something re-used our id.
            logger.warning(
                "artifact_autopublish result=no_key reason=idempotent_hit instance_id=%s",
                self._instance_id,
            )
            self._key = None
        return self._key

    async def revoke(self) -> None:
        """Best-effort revoke. Never raises: the turn has already succeeded."""
        if not self._attempted or self._key is None:
            return
        try:
            await revoke_turn_key(instance_id=self._instance_id, settings=TurnQueueSettings())
        except Exception:
            logger.warning(
                "artifact_autopublish result=revoke_failed instance_id=%s",
                self._instance_id,
                exc_info=True,
            )
