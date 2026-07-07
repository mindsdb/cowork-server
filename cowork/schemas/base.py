from datetime import datetime

from pydantic import BaseModel, model_validator
from pydantic.alias_generators import to_camel

from cowork.common.datetime_utils import ensure_utc

# SHIM:client-compat — remove alias_generator and populate_by_name from both
# classes when the client is updated to use snake_case field names.


class CamelResponse(BaseModel):
    """Response base that serializes snake_case fields as camelCase."""

    model_config = {
        "from_attributes": True,
        "alias_generator": to_camel,
        "populate_by_name": True,
    }

    @model_validator(mode="after")
    def _ensure_utc_datetimes(self) -> 'CamelResponse':
        """
        Ensure that all datetime fields are converted to UTC.
        SQLite does not support timezone-aware datetimes, so we need to convert them to UTC.
        """
        for name, _ in self.__class__.model_fields.items():
            value = getattr(self, name)

            if isinstance(value, datetime):
                value = ensure_utc(value)
                setattr(self, name, value)

        return self

    @classmethod
    def serialize(cls, obj):
        """Validate from ORM/dict and dump with camelCase aliases."""
        return cls.model_validate(obj, from_attributes=True).model_dump(by_alias=True)


class CamelRequest(BaseModel):
    """Request base that accepts both camelCase and snake_case input."""

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }
