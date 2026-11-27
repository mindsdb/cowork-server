from datetime import datetime

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


class TeamsContact(SQLModel, table=True):
    """Where to reach a person in Teams without being asked: their Microsoft ID in the bot's tenant
    and the Teams service address of their messages. The self-hosted recording proxy, which pairs
    Teams users with people, sends each pairing here over the Teams connector's signed route; Cowork
    never pairs anyone itself.

    Keyed by user alone, with no org: the route that fills it is mounted only where the Teams bot is
    set up, which is a self-hosted install serving one organization. A Microsoft ID names one person
    at a time (save_contact)."""

    __tablename__ = "teams_contacts"

    user_id: str = Field(primary_key=True, max_length=36, description="The person's sub, as X-User-Id names them")
    tenant_id: str = Field(max_length=64, description="The Teams tenant")
    aad_object_id: str = Field(max_length=64, description="Their Microsoft Entra object id")
    service_url: str = Field(max_length=512, description="The Teams service address their messages came from")
    updated_at: datetime = Field(
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When the proxy last sent it",
    )
