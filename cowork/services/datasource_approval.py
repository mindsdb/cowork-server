"""Which connections' credentials a full-stack publish may send.

The agent writes an artifact's metadata.json, so it decides which connections
a full-stack backend reads, and anton's publisher uploads the credentials of
every one it names. publish_artifact sends them only for the connections the
person approved for this artifact, in the chat card or the publish dialog.
The record is a row the server writes (cowork.models.artifact_datasource_approval),
never a file in the artifact's folder, which the agent edits all the time.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from cowork.db.personal_resources import MODEL_RESOURCE
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.units import unit_session
from cowork.models.artifact_datasource_approval import ArtifactDatasourceApproval

# The vault's label fields (layered_vault.LABEL_FIELDS), the person's own first.
_LABEL_ORDER = ("_user_label", "_label")
# The agent can set a connection's label (label_connection), and the label
# shows inside the card's sentence, so it is cut down to a plain short name.
_LABEL_MAX_CHARS = 60
_LABEL_DROPPED = re.compile(r"[^\w .,()/&+:#@-]+")


class DatasourceKey(NamedTuple):
    """How every check here compares a connection: the exact spelling, as
    anton's vault keys its records. `prod` and `Prod` can be two connections
    with different credentials, so approving one never covers the other."""

    engine: str
    name: str


def is_saved(vault: Any, engine: str, name: str) -> bool:
    """Whether ``vault`` holds the connection; a failed read counts as not saved."""
    try:
        return vault.read_record(engine, name) is not None
    except Exception:
        return False


class DatasourceApprovalUnavailable(RuntimeError):
    """The approval record could not be read or written. The publish stops
    before anything is uploaded; asking again may work."""


def declared_datasources(artifact_dir: Path) -> list[DatasourceKey]:
    """The connections the artifact's metadata.json names, once each and in
    its order, spelled as written (the vault looks them up that way). Empty
    without a valid metadata.json, where anton's publisher sends none."""
    from anton.core.artifacts.models import Artifact

    try:
        raw = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        artifact = Artifact.model_validate(raw)
    except Exception:
        return []
    return list(dict.fromkeys(DatasourceKey(ref.engine, ref.name) for ref in artifact.datasources))


def _keys(rows: Iterable[Any]) -> frozenset[DatasourceKey]:
    return frozenset(
        DatasourceKey(str(row["engine"]), str(row["name"]))
        for row in rows
        if isinstance(row, Mapping) and "engine" in row and "name" in row
    )


def _row(
    session: ScopedSession, *, artifact_key: str, artifact_dir: Path, for_update: bool = False
) -> ArtifactDatasourceApproval | None:
    statement = session.select(ArtifactDatasourceApproval).where(
        ArtifactDatasourceApproval.artifact_key == artifact_key,
        ArtifactDatasourceApproval.artifact_dir == str(artifact_dir),
    )
    if not session.scope.partitioned_for(MODEL_RESOURCE[ArtifactDatasourceApproval.__tablename__]):
        # Where the table is not partitioned the session does not filter by
        # org, so keep it to the no-org rows.
        statement = statement.where(ArtifactDatasourceApproval.org_id.is_(None))
    if for_update:
        # Two approvals of one folder (the chat card and the publish dialog)
        # each merge into the row; without the lock the later write drops
        # the other's connections. SQLite renders no lock and serializes
        # writers itself.
        statement = statement.with_for_update()
    return session.exec(statement).first()


def stored_approvals(
    *, artifact_key: str, artifact_dir: Path, scope: TenantScope | None
) -> frozenset[DatasourceKey]:
    """The connections approved for this artifact folder; empty without a row."""
    try:
        with unit_session(scope=scope or LOCAL_SCOPE) as session:
            row = _row(session, artifact_key=artifact_key, artifact_dir=artifact_dir)
            return _keys(row.datasources) if row is not None else frozenset()
    except SQLAlchemyError as exc:
        raise DatasourceApprovalUnavailable("Could not read the connections approved for this artifact") from exc


def _stage(session: ScopedSession, *, artifact_key: str, artifact_dir: Path, added: frozenset[DatasourceKey]) -> None:
    row = _row(session, artifact_key=artifact_key, artifact_dir=artifact_dir, for_update=True)
    if row is None:
        row = ArtifactDatasourceApproval(artifact_key=artifact_key, artifact_dir=str(artifact_dir), datasources=[])
    merged = _keys(row.datasources) | added
    row.datasources = [{"engine": key.engine, "name": key.name} for key in sorted(merged)]
    row.approved_by = session.scope.user_id
    session.add(row)


def record_approval(
    *,
    artifact_key: str,
    artifact_dir: Path,
    datasources: Iterable[DatasourceKey],
    scope: TenantScope | None,
) -> None:
    """Add ``datasources`` to what the person approved for this folder."""
    added = frozenset(datasources)
    if not added:
        return
    try:
        with unit_session(scope=scope or LOCAL_SCOPE) as session:
            _stage(session, artifact_key=artifact_key, artifact_dir=artifact_dir, added=added)
            try:
                session.commit()
            except IntegrityError:
                # Another approval of the same folder inserted its row first.
                session.rollback()
                _stage(session, artifact_key=artifact_key, artifact_dir=artifact_dir, added=added)
                session.commit()
    except SQLAlchemyError as exc:
        raise DatasourceApprovalUnavailable("Could not record the connections approved for this artifact") from exc


def _label(vault: Any, engine: str, name: str) -> str:
    try:
        record = vault.read_record(engine, name)
    except Exception:
        return ""
    fields = (record or {}).get("fields") or {}
    for key in _LABEL_ORDER:
        value = fields.get(key)
        if isinstance(value, str):
            label = " ".join(_LABEL_DROPPED.sub(" ", value).split())[:_LABEL_MAX_CHARS].strip()
            if label:
                return label
    return ""


def datasource_labels(vault: Any, connections: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    """How the card and the dialog name each connection: `label (engine)`, or
    `name (engine)` when the record has no label. The engine always shows,
    since the agent can save a connection under any label; only the record's
    label fields are read."""
    return [
        {"engine": engine, "name": name, "label": f"{_label(vault, engine, name) or name} ({engine})"}
        for engine, name in connections
    ]


class AllowedDataVault:
    """The vault a full-stack publish hands anton's publisher: only the
    approved connections read as saved. Any other reads as missing, so the
    publisher sends none of its credentials. ``refused`` keeps each saved
    connection it hid, as anton asked for it: metadata.json can name a
    connection after the approval check ran."""

    def __init__(self, *, source: Any, allowed: Iterable[DatasourceKey]) -> None:
        self.source = source
        self._allowed = frozenset(allowed)
        self.refused: list[DatasourceKey] = []

    def _allows(self, engine: str, name: str) -> bool:
        key = DatasourceKey(engine, name)
        if key in self._allowed:
            return True
        if key not in self.refused and is_saved(self.source, engine, name):
            self.refused.append(key)
        return False

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        return self.source.env_for(engine, name, flat=flat) if self._allows(engine, name) else None

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        return self.source.read_record(engine, name) if self._allows(engine, name) else None

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        return self.source.load(engine, name) if self._allows(engine, name) else None
