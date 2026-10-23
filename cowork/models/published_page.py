"""Ownership of pages served by a deployment's shared publisher.

A record's owner never changes. Its source path follows only a folder move
the server itself makes (published_page_ownership.rekey_published_page_paths).
"""
import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class PublishedPageOwner(BaseSQLModel, table=True):
    __tablename__ = "published_page_owners"
    __table_args__ = (
        sa.UniqueConstraint("publisher_key", "report_id", name="uq_published_page_owner"),
    )

    # Global within this database, including when projects are personal: a
    # foreign page id must find its owner instead of disappearing behind a
    # per-user query filter. The endpoint hash separates publisher instances.
    publisher_key: str = Field(max_length=64)
    report_id: str = Field(max_length=255)
    artifact_path: str = Field(sa_type=sa.Text)
    owner_user_id: str = Field(max_length=36)
    # No artifact FK or cascade: removing a source cannot release its live URL.
