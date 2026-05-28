from pydantic import BaseModel
from pydantic.alias_generators import to_camel


# SHIM:client-compat — remove alias_generator and populate_by_name from both
# classes when the client is updated to use snake_case field names.


class CamelResponse(BaseModel):
    """Response base that serializes snake_case fields as camelCase."""

    model_config = {
        "from_attributes": True,
        "alias_generator": to_camel,
        "populate_by_name": True,
    }

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
